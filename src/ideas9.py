"""Sweep 9 (M-series) - Monte-Carlo rollout of the stock-flow model.

The shipped p1b propagates a single mean path through a nonlinear recursion:

    P_t = clip(P_{t-1} + p*m - rf*P_{t-1}, 0, 1)            h(E[X])

but the scored quantity is the conditional mean over random paths:

    E[OSI_t | h71 prefix, weather]  =  E[h(X_path)]         E[h(X)]

and E[h(X)] != h(E[X]) for three verifiable reasons:

  1. absorbing boundary - true damage is zero on 59% of hours and OSI is clipped
     at 0 (44% exact zeros); the deterministic path adds a positive expected
     damage every hour and never reaches the absorbing state. This is the
     documented tail pathology (stock bias +0.00207 on truth 0.00033, REPORT 5p).
  2. degenerate state feature - `hrs_since_damage` resets whenever up_t > 1e-9,
     and the deterministic up_t = p*m is essentially always positive, so the
     clock is pegged at ~0 for the whole 144-hour rollout while training data
     has it growing through every quiet stretch. The model is evaluated on a
     state distribution it never saw; DAgger patches exactly this and R5 showed
     the patch overshoots (cumulative damage 1.92x -> 0.70x).
  3. Jensen through the nonlinear transition - the XGBoost flows are nonlinear
     in the state, so E[f(state)] != f(E[state]).

The MC estimator samples B paths: damage occurrence ~ Bernoulli(p), damage
magnitude = pred * ratio drawn from the empirical residual-ratio pool, and
(optionally) restoration fraction = pred * ratio from its own pool. Pools are
rescaled to mean 1 (after a p99.5 cap, re-normalised).

Mean preservation, corrected after an adversarial audit: the DAMAGE flow's
one-step conditional mean is preserved exactly (E[occ*mag*ratio|X] = p*mag;
verified at B=20,000 within MC noise). The RESTORE flow's is NOT when sampled:
clip(pred_rf*ratio, 0, 1) truncates the pool's heavy tail (32% of draws > 1,
forced by the 56% zeros holding mean 1), a compounding ~7-10%/step
under-restoration. sample_restore=True variants therefore confound the Jensen
mechanism with a systematic restoration slowdown; **M1c (sample_restore=False)
is the only mathematically clean variant** - damage stochastic and mean-exact,
restore deterministic - and it is also the empirical winner. Physically
sensible too: damage is lumpy (ignition events), restoration is continuous
crew work. OSI is reconstructed per path with the organizers' exact definition
and averaged across paths: an unbiased estimator of E[h(X)] given the fitted
transitions.

Prediction made before running (falsifiable): the TEACHER-FORCED model + MC
should not need DAgger, because MC removes the train/inference state mismatch
DAgger was invented to patch. If tf+MC lands near or below d1+det (0.01042),
that mechanism is confirmed.

Nothing here modifies StockFlowModel or the shipped p1b; MCStockFlow only
subclasses it and adds a residual-pool pass plus an MC predictor.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .stockflow import (EPS, SCOLS, WCOLS, StockFlowModel, _state_block,
                        _static_matrix, _weather_matrices, reconstruct_osi)

RATIO_CAP_Q = 99.5      # cap pool ratios at this percentile, then re-normalise


class MCStockFlow(StockFlowModel):
    """StockFlowModel + empirical residual pools + Monte-Carlo rollout."""

    # ------------------------------------------------------------- pools
    def fit(self, train_full, dagger_rounds=1):
        super().fit(train_full, dagger_rounds=dagger_rounds)
        self._build_pools(train_full)
        return self

    def _teacher_rows(self, train_full):
        """Teacher-forced feature rows and true flow targets (same construction
        as StockFlowModel.fit; rebuilt here so fit() stays untouched)."""
        masked = mask_after_freeze(train_full)
        W = _weather_matrices(train_full)
        S, fips, _ = _static_matrix(masked)
        P = (train_full.pivot_table(index="fipsCode", columns="hour", values="P_t")
             .sort_index().reindex(columns=np.arange(216)).to_numpy())
        P = np.nan_to_num(P)
        dP = np.diff(P, axis=1, prepend=P[:, :1])
        up, dn = np.clip(dP, 0, None), np.clip(-dP, 0, None)
        rows, y_up, y_rf, pos = [], [], [], []
        cum = np.zeros(len(P))
        hrs = np.full(len(P), 99.0)
        up_h = np.zeros((len(P), 6))
        dn_h = np.zeros((len(P), 6))
        for t in range(1, 216):
            stock = P[:, t - 1]
            blk = _state_block(stock, up_h, dn_h, cum, hrs, W["gust"][:, t])
            wblk = np.column_stack([W[c][:, t] for c in WCOLS])
            rows.append(np.hstack([blk, wblk, S]))
            y_up.append(up[:, t])
            y_rf.append(np.where(stock > EPS, dn[:, t] / np.maximum(stock, EPS), np.nan))
            pos.append(stock > EPS)
            cum = cum + up[:, t]
            hrs = np.where(up[:, t] > 0, 0.0, hrs + 1)
            up_h = np.roll(up_h, -1, 1); up_h[:, -1] = up[:, t]
            dn_h = np.roll(dn_h, -1, 1); dn_h[:, -1] = dn[:, t]
        return (np.vstack(rows), np.concatenate(y_up), np.concatenate(y_rf),
                np.concatenate(pos))

    @staticmethod
    def _norm_pool(ratios):
        """Cap the extreme tail, then rescale to mean exactly 1 so sampling
        preserves the per-step conditional mean."""
        r = np.asarray(ratios, float)
        r = r[np.isfinite(r)]
        r = np.clip(r, 0, np.percentile(r, RATIO_CAP_Q))
        return r / max(r.mean(), 1e-12)

    def _build_pools(self, train_full):
        X, yu, yr, pos = self._teacher_rows(train_full)
        m = yu > 0
        pred_mag = np.clip(self.reg.predict(X[m]), 0, None) ** 2 * self.smear
        self.pool_dmg = self._norm_pool(yu[m] / np.maximum(pred_mag, 1e-10))
        rm = pos & np.isfinite(yr)
        pred_rf = np.clip(self.res.predict(X[rm]), 0, 1)
        # pool includes the true zeros (52% of hours) - that IS the lumpiness
        self.pool_rf = self._norm_pool(
            np.clip(yr[rm], 0, 1) / np.maximum(pred_rf, 1e-6))

    # --------------------------------------------------------------- MC rollout
    def predict_mc(self, val_masked, n_paths=256, sample_restore=True, seed=42):
        """B stochastic paths per county; returns (fips, mean OSI over paths).

        Occurrence is Bernoulli(p) per path; magnitude and restoration carry
        empirical residual ratios. State features (rolling flow histories,
        cum_damage, hrs_since_damage) evolve per path, so the state
        distribution matches what the transition model saw in training.
        """
        rng = np.random.RandomState(seed)
        W = _weather_matrices(val_masked)
        S, fips, _ = _static_matrix(val_masked)
        Pobs = (val_masked.pivot_table(index="fipsCode", columns="hour", values="P_t")
                .sort_index().reindex(columns=np.arange(216)).to_numpy())
        Pobs = np.nan_to_num(Pobs)
        n, B = len(fips), n_paths
        N = n * B                                       # path-expanded rows

        rep = lambda a: np.repeat(a, B, axis=0)          # county-major expansion
        S_x = rep(S)
        Wx = {c: rep(W[c]) for c in
              set(WCOLS) | {"gust"}}

        pre = Pobs[:, :FREEZE_H + 1]
        d = np.diff(pre, axis=1, prepend=pre[:, :1])
        pre_up, pre_dn = np.clip(d, 0, None), np.clip(-d, 0, None)
        last = np.where(pre_up > 0, np.arange(FREEZE_H + 1), -99).max(1)
        hrs0 = np.where(last < 0, 99.0, FREEZE_H - last).astype(float)

        cur = rep(pre[:, -1:]).ravel().copy()            # (N,) stock at h71
        cum = rep(pre_up.sum(1, keepdims=True)).ravel().copy()
        hrs = rep(hrs0[:, None]).ravel().copy()
        up_h = rep(pre_up[:, -6:]).copy()
        dn_h = rep(pre_dn[:, -6:]).copy()

        H = 216 - FREEZE_H - 1
        P_paths = np.zeros((N, H))
        UP_paths = np.zeros((N, H))
        DN_paths = np.zeros((N, H))

        for i, t in enumerate(range(FREEZE_H + 1, 216)):
            blk = _state_block(cur, up_h, dn_h, cum, hrs, Wx["gust"][:, t])
            X = np.hstack([blk, np.column_stack([Wx[c][:, t] for c in WCOLS]), S_x])
            p = self.clf.predict_proba(X)[:, 1]
            mag = np.clip(self.reg.predict(X), 0, None) ** 2 * self.smear
            occ = rng.random_sample(N) < p
            up_t = np.where(occ, mag * rng.choice(self.pool_dmg, N), 0.0)
            rf = np.clip(self.res.predict(X), 0, 1)
            if sample_restore:
                rf = np.clip(rf * rng.choice(self.pool_rf, N), 0, 1)
            dn_t = np.minimum(cur * rf, cur + up_t)
            cur = np.clip(cur + up_t - dn_t, 0, 1)
            cum = cum + up_t
            hrs = np.where(up_t > 1e-9, 0.0, hrs + 1)
            up_h = np.roll(up_h, -1, 1); up_h[:, -1] = up_t
            dn_h = np.roll(dn_h, -1, 1); dn_h[:, -1] = dn_t
            P_paths[:, i], UP_paths[:, i], DN_paths[:, i] = cur, up_t, dn_t

        # per-path OSI with the exact definition, then average over paths.
        # prepend the observed prefix so D_t (rolling 6h) and the centered flows
        # are seeded with real history at the boundary, exactly as in predict().
        full_P = np.concatenate([rep(pre), P_paths], axis=1)
        full_UP = np.concatenate([rep(pre_up), UP_paths], axis=1)
        full_DN = np.concatenate([rep(pre_dn), DN_paths], axis=1)
        osi = reconstruct_osi(full_P, full_UP, full_DN)   # (N, 216)
        osi = osi.reshape(n, B, 216).mean(axis=1)         # E over paths
        return fips, osi


def _mc_forecaster(train_full, val_masked, dagger_rounds, n_paths=256,
                   sample_restore=True):
    m = MCStockFlow(monotone=False).fit(train_full, dagger_rounds=dagger_rounds)
    fips, osi = m.predict_mc(val_masked, n_paths=n_paths,
                             sample_restore=sample_restore)
    sel = np.isin(np.arange(216), PRED_HOURS)
    return pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips)),
                         "osi_pred": osi[:, sel].ravel()})


def m1_tf_mc(tr, va):       return _mc_forecaster(tr, va, dagger_rounds=0)
def m1b_d1_mc(tr, va):      return _mc_forecaster(tr, va, dagger_rounds=1)
def m1c_tf_mc_dmg(tr, va):  return _mc_forecaster(tr, va, dagger_rounds=0,
                                                  sample_restore=False)
def m1d_tf_mc64(tr, va):    return _mc_forecaster(tr, va, dagger_rounds=0,
                                                  n_paths=64)


# ------------------- M2 cumulative-curve reformulation (integral targets)
# Hourly flows are spiky, zero-inflated, hard targets; their INTEGRALS are
# smooth monotone curves - a fundamentally easier regression target. Predict
#   CD(t) = cumulative damage fraction since freeze      (sqrt-scale XGB)
#   RF(t) = cumulative restored / (P71 + CD(t)) in [0,1] (stock-anchored,
#                                                         per law 11)
# from the same frozen h71 state + weather-at-t rows as R8b (no recursion),
# enforce monotonicity by running max, then P = clip(P71 + CD - CR, 0, 1),
# flows = diffs, OSI via the exact reconstruction. Differs from R8b in ONE
# way: R8b predicted per-hour flows; M2 predicts their integrals.
class CumulativeCurveModel(  # noqa: N801  (matches file's class naming)
        __import__("src.stockflow", fromlist=["StockFlowDirectModel"]).StockFlowDirectModel):

    def fit(self, train_full):
        import xgboost as xgb
        from .stockflow import _weather_matrices, _static_matrix
        masked = mask_after_freeze(train_full)
        W = _weather_matrices(train_full)
        S, fips, self.scols = _static_matrix(masked)
        P = (train_full.pivot_table(index="fipsCode", columns="hour", values="P_t")
             .sort_index().reindex(columns=np.arange(216)).to_numpy())
        P = np.nan_to_num(P)
        dP = np.diff(P, axis=1, prepend=P[:, :1])
        up, dn = np.clip(dP, 0, None), np.clip(-dP, 0, None)

        rows = self._rows(P, W, S)                    # frozen-state rows, t-major
        hours = np.arange(FREEZE_H + 1, 216)
        X = np.vstack(rows)
        p71 = P[:, FREEZE_H]
        CD = np.cumsum(up[:, hours], axis=1)          # (n, 144)
        CR = np.cumsum(dn[:, hours], axis=1)
        y_cd = CD.T.ravel()
        y_rf = np.clip(CR / np.maximum(p71[:, None] + CD, 1e-9), 0, 1).T.ravel()

        common = dict(n_estimators=400, learning_rate=0.06, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                      reg_lambda=1.0, random_state=self.seed, n_jobs=4,
                      tree_method="hist")
        self.reg_cd = xgb.XGBRegressor(**common).fit(X, np.sqrt(y_cd))
        ps = self.reg_cd.predict(X)
        self.smear_cd = float(y_cd.mean() / max((np.clip(ps, 0, None) ** 2).mean(),
                                                1e-12))
        self.reg_rf = xgb.XGBRegressor(**common).fit(X, y_rf)
        return self

    def predict(self, val_masked):
        from .stockflow import _weather_matrices, _static_matrix
        W = _weather_matrices(val_masked)
        S, fips, _ = _static_matrix(val_masked)
        Pobs = (val_masked.pivot_table(index="fipsCode", columns="hour", values="P_t")
                .sort_index().reindex(columns=np.arange(216)).to_numpy())
        Pobs = np.nan_to_num(Pobs)
        n = len(fips)
        pre = Pobs[:, :FREEZE_H + 1]
        d = np.diff(pre, axis=1, prepend=pre[:, :1])
        pre_up, pre_dn = np.clip(d, 0, None), np.clip(-d, 0, None)
        p71 = pre[:, -1]

        P0 = np.zeros((n, 216)); P0[:, :FREEZE_H + 1] = pre
        X = np.vstack(self._rows(P0, W, S))
        H = 216 - FREEZE_H - 1
        cd = (np.clip(self.reg_cd.predict(X), 0, None) ** 2 * self.smear_cd
              ).reshape(H, n).T
        cd = np.maximum.accumulate(cd, axis=1)        # monotone damage integral
        rf = np.clip(self.reg_rf.predict(X), 0, 1).reshape(H, n).T
        cr = rf * (p71[:, None] + cd)
        cr = np.maximum.accumulate(cr, axis=1)        # monotone restore integral
        cr = np.minimum(cr, p71[:, None] + cd)        # cannot restore more than out
        P = np.clip(p71[:, None] + cd - cr, 0, 1)

        UP = np.diff(cd, axis=1, prepend=np.zeros((n, 1)))
        DN = np.diff(cr, axis=1, prepend=np.zeros((n, 1)))
        full_P = np.concatenate([pre, P], axis=1)
        full_UP = np.concatenate([pre_up, UP], axis=1)
        full_DN = np.concatenate([pre_dn, DN], axis=1)
        return fips, full_P, full_UP, full_DN


def m2_cumcurve(train_full, val_masked):
    m = CumulativeCurveModel(restore="fraction").fit(train_full)
    fips, P, UP, DN = m.predict(val_masked)
    osi = reconstruct_osi(P, UP, DN)
    sel = np.isin(np.arange(216), PRED_HOURS)
    return pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips)),
                         "osi_pred": osi[:, sel].ravel()})


# ------------------- M3 mixture-of-trajectories decoder (learned archetypes)
# The project's central identifiability result: county-specific SHAPE is
# predictable from the prefix, county-specific MAGNITUDE is not (archetype
# gating I05 sat 0.0008 from its own oracle; every magnitude attack failed).
# M3 builds that structure into the decoder: K trajectory heads (learned
# archetypes, raw scale) and a soft gate from the encoder summary. The point
# output is the mixture MEAN in RAW space - the RMSE-optimal functional - and
# a multiple-choice (per-sample min over heads) auxiliary keeps the heads from
# collapsing onto one another. Same encoder/decoder trunk, inputs, seeds and
# S62 objective as the rest of the family; one variable changed (the head).
class MixTraj(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, hidden=48, K=4, dropout=0.15):
        super().__init__()
        self.K = K
        self.enc = nn.GRU(n_enc, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(K)])
        self.gate = nn.Linear(hidden, K)

    def forward(self, enc, dec, static):
        _, h = self.enc(enc)
        e = self.proj(torch.cat([h[0], h[1]], -1))
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        out, _ = self.dec(torch.cat([dec, s], -1), e[None].contiguous())
        out = self.drop(out)
        raws = torch.stack([torch.clamp(hd(out).squeeze(-1), min=0) ** 2
                            for hd in self.heads], 0)          # (K, B, H) raw scale
        w = torch.softmax(self.gate(e), -1).T[:, :, None]       # (K, B, 1)
        mix_raw = (w * raws).sum(0)                              # RMSE-optimal mean
        return mix_raw, raws


def _fit_mix(Xe, Xd, Xs, Y, seed, w_hours, K=4, mcl=0.3,
             max_epochs=300, patience=30, lr=2e-3):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = MixTraj(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1], K=K)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(w_hours, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))

    def dual_per_sample(raw_pred, i):
        """S62's dual raw/sqrt objective, per sample (summed over hours)."""
        den = (msk[i] * tw).sum(-1).clamp(min=1e-9)
        raw = ((raw_pred - t[3][i]) ** 2 * tw * msk[i]).sum(-1) / den
        root = ((torch.sqrt(raw_pred + 1e-9) - sq[i]) ** 2 * tw * msk[i]).sum(-1) / den
        return 0.5 * raw + 0.5 * root

    best, state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            mix, raws = model(t[0][b], t[1][b], t[2][b])
            l_mix = dual_per_sample(mix, b).mean()
            l_mcl = torch.stack([dual_per_sample(raws[k], b)
                                 for k in range(raws.shape[0])], 0).min(0).values.mean()
            (l_mix + mcl * l_mcl).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mix, _ = model(t[0][va_i], t[1][va_i], t[2][va_i])
            v = float((((mix - t[3][va_i]) ** 2) * tw * msk[va_i]).sum()
                      / (msk[va_i] * tw).sum())
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def m3_mixture(train_full, val_masked, seeds=(0, 1, 2), K=4, mcl=0.3):
    from .ideas import _prep, _dl_out
    from .ideas4 import scoring_weights
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    w = scoring_weights()
    preds = []
    for s in seeds:
        m = _fit_mix(Xe, Xd, Xs, Y, s, w, K=K, mcl=mcl)
        with torch.no_grad():
            mix, _ = m(*tv)
            preds.append(mix.numpy())
    return _dl_out(vf, np.mean(preds, 0))


def m3_k4(tr, va):   return m3_mixture(tr, va, K=4)
def m3_k8(tr, va):   return m3_mixture(tr, va, K=8)
def m3_nomcl(tr, va): return m3_mixture(tr, va, K=4, mcl=0.0)


IDEAS9 = {
    "M1 stock-flow MC rollout, teacher-forced": m1_tf_mc,
    "M1b stock-flow MC rollout, DAgger-1": m1b_d1_mc,
    "M1c MC damage only, deterministic restore": m1c_tf_mc_dmg,
    "M1d MC teacher-forced, 64 paths": m1d_tf_mc64,
    "M2 cumulative-curve reformulation (integrals)": m2_cumcurve,
    "M3 mixture-of-trajectories K=4 + MCL": m3_k4,
    "M3b mixture K=8": m3_k8,
    "M3c mixture K=4, no MCL (collapse control)": m3_nomcl,
}
