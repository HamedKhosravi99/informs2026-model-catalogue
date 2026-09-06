"""Sweep 7: the five pre-registered experiments.

E1  component-GRU trained with the S62 objective (scoring-aligned + dual raw/sqrt)
E2  Transformer with the dual raw/sqrt objective (architecture untouched)
E3  N_t transition-timing auxiliary supervision
E5  one controlled 2-layer bi-encoder probe (only if E1-E4 fail)

Every model keeps its existing architecture, inputs, seeds and folds; only the
training objective changes, so each result isolates one variable.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .dl import (Y_SCALE, TransformerTraj, ComponentSeq2Seq, DEC_HOURS,
                 _seq_tensor, _pivot, _tensors)
from .ideas import _prep, _dl_out
from .ideas3 import BiEncSeq2Seq
from .ideas4 import scoring_weights
from .models import SEED


# =========================== E1  component-GRU + scoring-aligned dual raw/sqrt loss
def _component_weights():
    """Scoring weights over the component decoder's hours (72..215). Hour 72 is never
    a scored target, so it carries zero weight."""
    w = np.zeros(len(DEC_HOURS))
    w[1:] = scoring_weights()          # DEC_HOURS[1:] == PRED_HOURS
    return w


def component_s62_forecaster(train_full, val_masked, seeds=(0, 1, 2), aux_w=0.3):
    """Existing component architecture; only the objective changes to S62's:
    scoring_weight(hour) * [0.5*MSE(OSI) + 0.5*MSE(sqrt OSI)], component aux losses
    retained at their existing 0.3 weight (they train the P/N/R heads)."""
    masked_tr = mask_after_freeze(train_full)
    _, Xe, Xd, Xs = _tensors(masked_tr, DEC_HOURS)
    obs_hours = np.arange(66, FREEZE_H + 1)
    Pobs = np.nan_to_num(_pivot(masked_tr, "P_t", obs_hours).to_numpy()) * Y_SCALE
    YP = _pivot(train_full, "P_t", DEC_HOURS).to_numpy() * Y_SCALE
    YN = _pivot(train_full, "N_t", DEC_HOURS).to_numpy() * Y_SCALE
    YR = _pivot(train_full, "R_t", DEC_HOURS).to_numpy() * Y_SCALE
    Yosi = _pivot(train_full, "osi", DEC_HOURS).to_numpy() * Y_SCALE

    stats = [(a.reshape(-1, a.shape[-1]).mean(0), a.reshape(-1, a.shape[-1]).std(0) + 1e-6)
             for a in (Xe, Xd, Xs)]
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]
    vf, Ve, Vd, Vs = _tensors(val_masked, DEC_HOURS)
    VPobs = np.nan_to_num(_pivot(val_masked, "P_t", obs_hours).to_numpy()) * Y_SCALE
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs, VPobs)]

    w = torch.tensor(_component_weights(), dtype=torch.float32)
    t = {k: torch.tensor(np.nan_to_num(v), dtype=torch.float32) for k, v in
         dict(Xe=Xe, Xd=Xd, Xs=Xs, Pobs=Pobs, YP=YP, YN=YN, YR=YR, Yosi=Yosi).items()}
    msk = torch.tensor(np.isfinite(Yosi).astype(float), dtype=torch.float32)
    sq_true = torch.sqrt(torch.clamp(t["Yosi"], min=0))

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = ComponentSeq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

        def wmean(err, i):
            return (err * w * msk[i]).sum() / (w * msk[i]).sum()

        best, state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                osi, P, N, R = model(t["Xe"][b], t["Xd"][b], t["Xs"][b], t["Pobs"][b])
                raw = wmean((osi - t["Yosi"][b]) ** 2, b)
                root = wmean((torch.sqrt(torch.clamp(osi, min=0)) - sq_true[b]) ** 2, b)
                aux = (nn.functional.mse_loss(P, t["YP"][b])
                       + nn.functional.mse_loss(N, t["YN"][b])
                       + nn.functional.mse_loss(R, t["YR"][b]))
                (0.5 * raw + 0.5 * root + aux_w * aux).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                osi, *_ = model(t["Xe"][va_i], t["Xd"][va_i], t["Xs"][va_i], t["Pobs"][va_i])
                v = float(wmean((osi - t["Yosi"][va_i]) ** 2, va_i))   # scored quantity
            if v < best - 1e-6:
                best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            osi, *_ = model(*tv)
            preds.append(osi.numpy())
    P = np.clip(np.mean(preds, 0) / Y_SCALE, 0.0, None)[:, 1:]        # hours 73..215
    return pd.DataFrame({"fipsCode": np.repeat(vf, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(vf)),
                         "osi_pred": P.ravel()})


# ================================ E2  Transformer + dual raw/sqrt objective only
def transformer_dual_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    """Identical Transformer architecture, features and seeds; the only change is the
    objective, using S51's formulation: the head predicts on the sqrt scale and the
    loss keeps both scales."""
    _, X = _seq_tensor(mask_after_freeze(train_full))
    Y = (train_full.pivot_table(index="fipsCode", columns="hour", values="osi")
         .sort_index().reindex(columns=PRED_HOURS).to_numpy() * Y_SCALE)
    m = X.reshape(-1, X.shape[-1]).mean(0)
    s = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    X = (X - m) / s
    vf, V = _seq_tensor(val_masked)
    V = torch.tensor((V - m) / s, dtype=torch.float32)
    tX = torch.tensor(X, dtype=torch.float32)
    tY = torch.tensor(np.nan_to_num(Y), dtype=torch.float32)
    tS = torch.sqrt(torch.clamp(tY, min=0))

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = TransformerTraj(X.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, state, bad = np.inf, None, 0
        for _ in range(400):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                o = model(tX[b])
                loss = (0.5 * nn.functional.mse_loss(torch.clamp(o, min=0) ** 2, tY[b])
                        + 0.5 * nn.functional.mse_loss(o, tS[b]))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.clamp(model(tX[va_i]), min=0) ** 2
                v = nn.functional.mse_loss(p, tY[va_i]).item()      # raw scale
            if v < best - 1e-6:
                best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 40:
                    break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds.append((torch.clamp(model(V), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ====================== E3  N_t transition-timing auxiliary supervision on S62
class TransitionAuxNet(BiEncSeq2Seq):
    """S62's architecture with an auxiliary head for outage-onset timing."""

    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15, n_aux=2):
        super().__init__(n_enc, n_dec, n_static, hidden, dropout)
        self.aux = nn.Linear(hidden, n_aux)

    def both(self, enc, dec, static):
        _, h = self.enc(enc)
        h0 = self.proj(torch.cat([h[0], h[1]], -1))[None].contiguous()
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        o, _ = self.dec(torch.cat([dec, s], -1), h0)
        o = self.drop(o)
        return self.head(o).squeeze(-1), self.aux(o)


def transition_aux_forecaster(train_full, val_masked, seeds=(0, 1, 2), aux_w=0.1):
    """Auxiliary target: does N_t exceed a training-quantile threshold within the next
    3h / 6h? This supervises WHEN damage starts, which the project's evidence says is
    more learnable than county-specific magnitude. Auxiliary output discarded at
    inference."""
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    Nt = _pivot(train_full, "N_t", PRED_HOURS).to_numpy()
    thr = np.nanquantile(Nt[Nt > 0], 0.75) if np.isfinite(Nt).any() else 1e-3
    hot = np.nan_to_num(Nt) > thr
    aux = np.stack([
        pd.DataFrame(hot).T.rolling(3, min_periods=1).max().T.shift(-3, axis=1).fillna(0).to_numpy(),
        pd.DataFrame(hot).T.rolling(6, min_periods=1).max().T.shift(-6, axis=1).fillna(0).to_numpy(),
    ], -1).astype(np.float32)
    print(f"    [E3] N_t threshold = {thr:.5f}; positives 3h {aux[:,:,0].mean():.3f}, "
          f"6h {aux[:,:,1].mean():.3f}", flush=True)

    w = torch.tensor(scoring_weights(), dtype=torch.float32)
    t = [torch.tensor(v, dtype=torch.float32)
         for v in (Xe, Xd, Xs, np.nan_to_num(Y), aux)]
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = TransitionAuxNet(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

        def wmean(err, i):
            return (err * w * msk[i]).sum() / (w * msk[i]).sum()

        best, state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                o, a = model.both(t[0][b], t[1][b], t[2][b])
                raw = wmean(((torch.clamp(o, min=0) ** 2) - t[3][b]) ** 2, b)
                root = wmean((o - sq[b]) ** 2, b)
                bce = nn.functional.binary_cross_entropy_with_logits(a, t[4][b])
                (0.5 * raw + 0.5 * root + aux_w * bce).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.clamp(model(t[0][va_i], t[1][va_i], t[2][va_i]), min=0) ** 2
                v = float(wmean((p - t[3][va_i]) ** 2, va_i))
            if v < best - 1e-6:
                best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds.append((torch.clamp(model(*tv), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ============================ E5  one controlled 2-layer bi-encoder probe
class BiEnc2Layer(BiEncSeq2Seq):
    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__(n_enc, n_dec, n_static, hidden, dropout)
        self.enc = nn.GRU(n_enc, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)

    def forward(self, enc, dec, static):
        _, h = self.enc(enc)
        h0 = self.proj(torch.cat([h[-2], h[-1]], -1))[None].contiguous()
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        o, _ = self.dec(torch.cat([dec, s], -1), h0)
        return self.head(self.drop(o)).squeeze(-1)


def bienc2_s62_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    from .ideas6 import _fit_scoring_dual
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    w = scoring_weights()
    preds = []
    for s in seeds:
        m = _fit_scoring_dual(BiEnc2Layer, Xe, Xd, Xs, Y, s, w)
        with torch.no_grad():
            preds.append((torch.clamp(m(*tv), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


IDEAS7 = {
    "E1 component-GRU + S62 objective": component_s62_forecaster,
    "E2 Transformer + dual raw/sqrt": transformer_dual_forecaster,
    "E3 N_t transition auxiliary head": transition_aux_forecaster,
    "E5 2-layer bi-encoder + S62 loss": bienc2_s62_forecaster,
}


# ============ E4  new external county fragility data -> TREES ONLY
E4_MIN = ["eia_all_meters", "elev_range"]                     # the two the screen kept
E4_PLUS = E4_MIN + ["eia_ami_share", "eia_dist_circuits"]     # sanctioned sources only
E4_HIFLD = E4_MIN + ["tl_segments"]                           # rules-uncertain source


def _load_e4():
    from .data import ROOT
    import functools
    e = pd.read_csv(ROOT / "data_static/eia861/county_eia861_2024.csv").set_index("fipsCode")
    t = pd.read_csv(ROOT / "data_static/county_terrain.csv").set_index("fipsCode")
    h = pd.read_csv(ROOT / "data_static/eia861/county_transmission_hifld.csv").set_index("fipsCode")
    want = set(E4_PLUS + E4_HIFLD)
    frames = [df[[c for c in df.columns if c in want]] for df in (e, t, h)]
    d = pd.concat([f for f in frames if len(f.columns)], axis=1)
    return d.loc[:, ~d.columns.duplicated()]


def _e4_rows(df, cols, val=False):
    from .ideas6 import _rows
    out = _rows(df, True, True, val=val)
    rows, y = (out, None) if val else out
    tab = _load_e4()
    rows = rows.join(tab[cols], on="fipsCode")
    return rows if val else (rows, y)


def _e4_model(train_full, val_masked, cols, model="xgb"):
    from .features import feature_cols
    from .ideas import _out
    tr, y = _e4_rows(train_full, cols)
    fc = feature_cols(tr)
    keep = np.isfinite(y)
    if model == "xgb":
        import xgboost as xgb
        mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                               reg_lambda=1.0, random_state=SEED, n_jobs=4,
                               tree_method="hist")
    else:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        mdl = make_pipeline(SimpleImputer(strategy="median"),
                            ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                                max_features=0.5, n_jobs=4,
                                                random_state=SEED))
    mdl.fit(tr.loc[keep, fc], np.sqrt(y[keep]))
    va = _e4_rows(val_masked, cols, val=True)
    return _out(va, np.clip(mdl.predict(va[fc]), 0, None) ** 2)


def e4_min_xgb(tr, va):    return _e4_model(tr, va, E4_MIN)
def e4_plus_xgb(tr, va):   return _e4_model(tr, va, E4_PLUS)
def e4_hifld_xgb(tr, va):  return _e4_model(tr, va, E4_HIFLD)
def e4_min_et(tr, va):     return _e4_model(tr, va, E4_MIN, model="et")


IDEAS7["E4 sqrt-XGB + EIA meters + elev_range"] = e4_min_xgb
IDEAS7["E4b sqrt-XGB + 4 sanctioned cols"] = e4_plus_xgb
IDEAS7["E4c sqrt-XGB + HIFLD transmission"] = e4_hifld_xgb
IDEAS7["E4d ExtraTrees + EIA meters + elev"] = e4_min_et


# ================================ P1  damage/restoration state-space hurdle model
def p1_stockflow(tr, va):
    from .stockflow import stockflow_forecaster
    return stockflow_forecaster(tr, va)


def p1_stockflow_nomono(tr, va):
    from .stockflow import stockflow_nomono_forecaster
    return stockflow_nomono_forecaster(tr, va)


IDEAS7["P1 stock-flow hurdle (monotone)"] = p1_stockflow
IDEAS7["P1b stock-flow hurdle (no monotone)"] = p1_stockflow_nomono


def p1_stockflow_dagger0(tr, va):
    from .stockflow import stockflow_teacher_only
    return stockflow_teacher_only(tr, va)


IDEAS7["P1c stock-flow, teacher-forced only"] = p1_stockflow_dagger0


def p1_stockflow_d3(tr, va):
    from .stockflow import stockflow_forecaster
    return stockflow_forecaster(tr, va, monotone=False, dagger_rounds=3)


IDEAS7["P1d stock-flow, 3 DAgger rounds"] = p1_stockflow_d3


def p3_latent(tr, va):
    from .adapt import latent_adapt_forecaster
    return latent_adapt_forecaster(tr, va)


def p3_latent_z0(tr, va):
    from .adapt import latent_adapt_z0_forecaster
    return latent_adapt_z0_forecaster(tr, va)


def p4_canonical(tr, va):
    from .adapt import canonical_forecaster
    return canonical_forecaster(tr, va)


IDEAS7["P3 test-time latent adaptation (1D)"] = p3_latent
IDEAS7["P3b same model, no adaptation (z=0)"] = p3_latent_z0
IDEAS7["P4 storm-phase canonicalization"] = p4_canonical


# ==================== B1  SARIMAX(2,0,2) per-county classical baseline
def sarimax_forecaster(train_full, val_masked):
    """The classical baseline the 2025 winner's report benchmarked against
    (SARIMAX(2,0,2)), adapted to this task: fitted per held-out county on its OWN
    observed 72 hours of OSI with exogenous weather, forecasting h72-215 with the
    known future exog. Convergence fallbacks mirror the 2025 4th-place recipe:
    SARIMAX+exog -> ARIMA(2,0,2) no-exog -> decayed persistence (tau=16)."""
    import warnings
    import numpy as np, pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from .data import FREEZE_H, PRED_HOURS

    out = []
    for fips, g in val_masked.sort_values("hour").groupby("fipsCode"):
        y = g.loc[g.hour <= FREEZE_H, "osi"].to_numpy()
        ex_obs = g.loc[g.hour <= FREEZE_H, ["gust", "tp"]].to_numpy()
        ex_fut = g.loc[g.hour > FREEZE_H, ["gust", "tp"]].to_numpy()
        n_f = int((g.hour > FREEZE_H).sum())   # numpy int64 makes statsmodels treat steps as a LABEL
        pred = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for use_exog in (True, False):
                try:
                    m = SARIMAX(y, exog=ex_obs if use_exog else None, order=(2, 0, 2),
                                enforce_stationarity=False, enforce_invertibility=False)
                    r = m.fit(disp=0, maxiter=100)
                    pred = np.asarray(r.forecast(steps=n_f, exog=ex_fut if use_exog else None))
                    # explosive AR roots on 72 points produce finite but absurd
                    # forecasts over 144 steps; OSI lives in [0, 0.65]
                    if np.all(np.isfinite(pred)) and np.abs(pred).max() <= 1.0:
                        break
                    pred = None
                except Exception:
                    pred = None
        if pred is None:                                   # decay fallback
            dt = np.arange(1, n_f + 1)
            pred = (y[-1] if len(y) else 0.0) * np.exp(-dt / 16.0)
        hours = g.loc[g.hour > FREEZE_H, "hour"].to_numpy()
        out.append(pd.DataFrame({"fipsCode": fips, "hour": hours,
                                 "osi_pred": np.clip(np.asarray(pred), 0, 1)}))
    d = pd.concat(out, ignore_index=True)
    return d[d.hour.isin(PRED_HOURS)]


IDEAS7["B1 SARIMAX(2,0,2) per-county baseline"] = sarimax_forecaster


# ============ H  the 2025-winner IDEA (hurdle) applied to our best sequence model
class HurdleBiEnc(nn.Module):
    """S62's bi-encoder trunk with a hurdle-style gated emission: an occurrence gate
    and a magnitude head, output = sigmoid(gate) * softplus(mag) on the sqrt scale.
    The 2025 winner's decomposition (P(outage) x E[magnitude|outage]) as an
    end-to-end output structure rather than two separate fitted models. Distinct
    from the failed aux-head experiments (E3, S54): the gate is IN the output path,
    shaping every prediction, not a discarded side task."""

    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden, 1)
        self.mag = nn.Linear(hidden, 1)

    def trunk(self, enc, dec, static):
        _, h = self.enc(enc)
        h0 = self.proj(torch.cat([h[0], h[1]], -1))[None].contiguous()
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        o, _ = self.dec(torch.cat([dec, s], -1), h0)
        return self.drop(o)

    def forward(self, enc, dec, static):
        o = self.trunk(enc, dec, static)
        return torch.sigmoid(self.gate(o)).squeeze(-1) * nn.functional.softplus(
            self.mag(o)).squeeze(-1)

    def with_gate(self, enc, dec, static):
        o = self.trunk(enc, dec, static)
        return (torch.sigmoid(self.gate(o)).squeeze(-1) * nn.functional.softplus(
            self.mag(o)).squeeze(-1), self.gate(o).squeeze(-1))


def _fit_hurdle(Xe, Xd, Xs, Y, seed, w, bce_w=0.0, max_epochs=300, patience=30):
    """S62's exact scoring-aligned dual raw/sqrt objective on the gated output;
    optional occurrence BCE on the gate (the winner's explicit stage-1 supervision)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = HurdleBiEnc(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(w, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))
    occ = (t[3] > 0).float()

    def wmean(err, i):
        return (err * tw * msk[i]).sum() / (tw * msk[i]).sum()

    best, state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            if bce_w > 0:
                o, glogit = model.with_gate(t[0][b], t[1][b], t[2][b])
            else:
                o = model(t[0][b], t[1][b], t[2][b])
            loss = 0.5 * wmean((o ** 2 - t[3][b]) ** 2, b) + 0.5 * wmean((o - sq[b]) ** 2, b)
            if bce_w > 0:
                loss = loss + bce_w * nn.functional.binary_cross_entropy_with_logits(
                    glogit, occ[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            o = model(t[0][va_i], t[1][va_i], t[2][va_i])
            v = float(wmean((o ** 2 - t[3][va_i]) ** 2, va_i))
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def _run_hurdle(train_full, val_masked, bce_w, seeds=(0, 1, 2)):
    from .ideas import _prep, _dl_out
    from .ideas4 import scoring_weights
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        m = _fit_hurdle(Xe, Xd, Xs, Y, s, scoring_weights(), bce_w=bce_w)
        with torch.no_grad():
            preds.append((m(*tv) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def h1_hurdle_bienc(tr, va):
    return _run_hurdle(tr, va, bce_w=0.0)


def h2_hurdle_bienc_bce(tr, va):
    return _run_hurdle(tr, va, bce_w=0.1)


IDEAS7["H1 hurdle-gated bi-encoder (S62 loss)"] = h1_hurdle_bienc
IDEAS7["H2 hurdle-gated + occurrence BCE 0.1"] = h2_hurdle_bienc_bce


# ============ T1  transductive multi-origin: train on the TEST counties' own observed 72h
def transductive_origin_forecaster(train_full, val_masked, seeds=(0, 1, 2),
                                   tr_origins=(47, 59, 71), va_origins=(35, 47, 59),
                                   aux_weight=0.5, va_weight=0.5):
    """The held-out counties' h0-71 is fully observed and legal (every timestamp <= 71),
    yet no model has ever trained on it. Here they join training as extra sequences at
    early origins, loss-masked to their observed hours, so the network learns the
    DEPLOYMENT counties' own damage response before forecasting them. Control: S40
    (identical machinery without the val blocks). Simulated honestly in CV; at
    submission time the same code path consumes the real test counties."""
    from .ideas4 import _origin_tensors
    from .dl import Seq2Seq
    H = len(PRED_HOURS)
    masked_tr = mask_after_freeze(train_full)
    blocks = []
    for o in tr_origins:
        dec_hours = np.arange(o + 2, o + 2 + H); dec_hours = dec_hours[dec_hours <= 215]
        pad = H - len(dec_hours)
        enc, dec, st, Y, _ = _origin_tensors(
            train_full if o != FREEZE_H else masked_tr, train_full, o, dec_hours)
        if pad:
            dec = np.concatenate([dec, np.repeat(dec[:, -1:], pad, 1)], 1)
            Y = np.concatenate([Y, np.full((len(Y), pad), np.nan)], 1)
        w = 1.0 if o == FREEZE_H else aux_weight
        blocks.append((enc, dec, st, Y, w, o == FREEZE_H))
    # the held-out counties: targets exist only for hours <= FREEZE_H
    for o in va_origins:
        dec_hours = np.arange(o + 2, o + 2 + H); dec_hours = dec_hours[dec_hours <= 215]
        pad = H - len(dec_hours)
        enc, dec, st, Y, _ = _origin_tensors(val_masked, val_masked, o, dec_hours)
        if pad:
            dec = np.concatenate([dec, np.repeat(dec[:, -1:], pad, 1)], 1)
            Y = np.concatenate([Y, np.full((len(Y), pad), np.nan)], 1)
        blocks.append((enc, dec, st, Y, va_weight, False))

    L = max(b[0].shape[1] for b in blocks)
    Xe = np.concatenate([np.pad(b[0], ((0, 0), (L - b[0].shape[1], 0), (0, 0)))
                         for b in blocks], 0)
    Xd = np.concatenate([b[1] for b in blocks], 0)
    Xs = np.concatenate([b[2] for b in blocks], 0)
    Y = np.concatenate([b[3] for b in blocks], 0)
    W = np.concatenate([np.full(len(b[0]), b[4]) for b in blocks])
    is_main = np.concatenate([np.full(len(b[0]), b[5]) for b in blocks])

    stats = [(a.reshape(-1, a.shape[-1]).mean(0), a.reshape(-1, a.shape[-1]).std(0) + 1e-6)
             for a in (Xe, Xd, Xs)]
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]

    from .dl import _tensors
    vf, Ve, Vd, Vs = _tensors(val_masked, PRED_HOURS)
    Vs = np.concatenate([Vs, np.full((len(Vs), 1), FREEZE_H / 216.0)], 1)
    Ve = np.pad(Ve, ((0, 0), (L - Ve.shape[1], 0), (0, 0)))
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]

    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(W, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    from .ideas import _dl_out
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        main_idx = np.where(is_main)[0]
        hold = rng.permutation(main_idx)[:max(12, len(main_idx) // 8)]
        hold_set = set(hold.tolist())
        tr_i = np.array([i for i in range(len(Xe)) if i not in hold_set])
        model = Seq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                out = model(t[0][b], t[1][b], t[2][b])
                se = ((out - t[3][b]) ** 2) * msk[b]
                ((se.sum(1) / msk[b].sum(1).clamp(min=1)) * tw[b]).mean().backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                out = model(t[0][hold], t[1][hold], t[2][hold])
                v = float((((out - t[3][hold]) ** 2) * msk[hold]).sum() / msk[hold].sum())
            if v < best - 1e-6:
                best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


IDEAS7["T1 transductive multi-origin (test counties' own 72h)"] = transductive_origin_forecaster


# ============ R8  non-recursive cumulative-flow formulation (P1 ablation)
def r8_stockflow_direct(tr, va):
    from .stockflow import stockflow_direct_forecaster
    return stockflow_direct_forecaster(tr, va)


def r8b_stockflow_direct_fraction(tr, va):
    from .stockflow import stockflow_direct_fraction_forecaster
    return stockflow_direct_fraction_forecaster(tr, va)


IDEAS7["R8 non-recursive stock-flow (gross flows)"] = r8_stockflow_direct
IDEAS7["R8b non-recursive stock-flow (fraction restore)"] = r8b_stockflow_direct_fraction
