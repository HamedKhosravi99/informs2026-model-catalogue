"""Sweep 4: sample-size expansion and retrieval-augmented forecasting.

Motivated by the audit finding that every measured win in this project has been a
variance-reduction or effective-sample-size move, while every change to the
estimand (Tweedie/quantile/sqrt-on-sequence) has traded RMSE for MAE.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .dl import (Y_SCALE, ENC_COLS, DEC_COLS, STATIC_COLS, Seq2Seq, _fit_one,
                 _pivot, _tensors)
from .features import county_static, weather_by_hour
from .ideas import _prep, _dl_out
from .models import SEED


# ============================================== S40 multi-origin training augmentation
def _origin_tensors(masked, full, origin, dec_hours):
    """Encoder over hours 0..origin, decoder over dec_hours, target from `full`.

    Only training counties are used this way (they are fully observed), so no
    outage value after the county's own origin ever enters the inputs.
    """
    st = county_static(masked)
    w = weather_by_hour(masked)
    fips = _pivot(masked, "gust", np.arange(origin + 1)).index.to_numpy()
    enc = np.stack([_pivot(full, c, np.arange(origin + 1)).to_numpy() for c in ENC_COLS], -1)
    dec = np.stack([_pivot(w, c, dec_hours).to_numpy() for c in DEC_COLS], -1)
    dt = (dec_hours - origin).astype(float)
    osi_o = _pivot(full, "osi", np.array([origin])).to_numpy()          # state at origin
    extras = np.stack([np.repeat(dt[None, :] / 100.0, len(fips), 0),
                       osi_o * np.exp(-dt / 16.0) * Y_SCALE,
                       osi_o * np.exp(-dt / 32.0) * Y_SCALE], -1)
    dec = np.concatenate([dec, extras], -1)
    static = np.concatenate([st.loc[fips, STATIC_COLS].to_numpy(),
                             np.full((len(fips), 1), origin / 216.0)], 1)  # origin phase
    Y = _pivot(full, "osi", dec_hours).to_numpy() * Y_SCALE
    return (np.nan_to_num(enc), np.nan_to_num(dec), np.nan_to_num(static),
            np.nan_to_num(Y), fips)


def multi_origin_forecaster(train_full, val_masked, seeds=(0, 1, 2),
                            origins=(47, 59, 71), aux_weight=0.5):
    """Train on several prediction origins per county (3x the sequences), with an
    origin-phase feature; early stopping stays on the scored origin (71)."""
    H = len(PRED_HOURS)
    masked_tr = mask_after_freeze(train_full)
    blocks = []
    for o in origins:
        dec_hours = np.arange(o + 2, o + 2 + H)
        dec_hours = dec_hours[dec_hours <= 215]
        pad = H - len(dec_hours)
        enc, dec, st, Y, _ = _origin_tensors(
            train_full if o != FREEZE_H else masked_tr, train_full, o, dec_hours)
        if pad:                                     # right-pad shorter horizons
            dec = np.concatenate([dec, np.repeat(dec[:, -1:], pad, 1)], 1)
            Y = np.concatenate([Y, np.full((len(Y), pad), np.nan)], 1)
        # encoder length must match across origins: left-pad to the longest
        blocks.append((enc, dec, st, Y, o))
    L = max(b[0].shape[1] for b in blocks)
    Xe = np.concatenate([np.pad(b[0], ((0, 0), (L - b[0].shape[1], 0), (0, 0)))
                         for b in blocks], 0)
    Xd = np.concatenate([b[1] for b in blocks], 0)
    Xs = np.concatenate([b[2] for b in blocks], 0)
    Y = np.concatenate([b[3] for b in blocks], 0)
    W = np.concatenate([np.full(len(b[0]), 1.0 if b[4] == FREEZE_H else aux_weight)
                        for b in blocks])
    is_main = np.concatenate([np.full(len(b[0]), b[4] == FREEZE_H) for b in blocks])

    stats = []
    for a in (Xe, Xd, Xs):
        stats.append((a.reshape(-1, a.shape[-1]).mean(0),
                      a.reshape(-1, a.shape[-1]).std(0) + 1e-6))
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]

    # validation tensors at the real origin, with the same feature layout
    vf, Ve, Vd, Vs = _tensors(val_masked, PRED_HOURS)
    Vs = np.concatenate([Vs, np.full((len(Vs), 1), FREEZE_H / 216.0)], 1)
    Ve = np.pad(Ve, ((0, 0), (L - Ve.shape[1], 0), (0, 0)))
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]

    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(W, dtype=torch.float32)
    mask = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        main_idx = np.where(is_main)[0]
        hold = rng.permutation(main_idx)[:max(12, len(main_idx) // 8)]
        hold_c = set(hold % len(main_idx))
        tr_i = np.array([i for i in range(len(Xe)) if (i % len(main_idx)) not in hold_c])
        model = Seq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, best_state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                out = model(t[0][b], t[1][b], t[2][b])
                se = ((out - t[3][b]) ** 2) * mask[b]
                (se.mean(1) * tw[b]).mean().backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                out = model(t[0][hold], t[1][hold], t[2][hold])
                v = float((((out - t[3][hold]) ** 2) * mask[hold]).mean())
            if v < best - 1e-6:
                best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ================================================ S41 retrieved-analog futures (RAFT)
def _retrieval_keys(masked, full_weather):
    """Key = observed OSI shape (z-scored) + full-window gust profile + exposure."""
    obs = _pivot(masked, "osi", np.arange(FREEZE_H + 1)).fillna(0.0).to_numpy()
    obs_z = (obs - obs.mean(1, keepdims=True)) / (obs.std(1, keepdims=True) + 1e-9)
    gust = _pivot(masked, "gust", np.arange(216)).fillna(0.0).to_numpy()
    gust_z = (gust - gust.mean()) / (gust.std() + 1e-9)
    return np.concatenate([obs_z, gust_z], 1)


def raft_forecaster(train_full, val_masked, seeds=(0, 1, 2), k=5):
    """Append the realized future trajectories of the k most similar TRAINING
    counties as extra decoder channels, so the network learns per-horizon how much
    to trust retrieved analogs. Retrieval is leave-one-out at training time."""
    masked_tr = mask_after_freeze(train_full)
    Ktr = _retrieval_keys(masked_tr, None)
    Kva = _retrieval_keys(val_masked, None)
    Ftr = _pivot(train_full, "osi", PRED_HOURS).fillna(0.0).to_numpy() * Y_SCALE

    def _norm(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    Ntr, Nva = _norm(Ktr), _norm(Kva)
    sim_tr = Ntr @ Ntr.T
    np.fill_diagonal(sim_tr, -np.inf)                    # leave-one-out
    sim_va = Nva @ Ntr.T

    def analog_channels(sim):
        idx = np.argsort(-sim, 1)[:, :k]
        w = np.take_along_axis(sim, idx, 1)
        w = np.clip(w, 0, None)
        w = w / (w.sum(1, keepdims=True) + 1e-9)
        chans = Ftr[idx]                                  # (N, k, H)
        blend = (chans * w[:, :, None]).sum(1)            # weighted analog mean
        return np.concatenate([chans.transpose(0, 2, 1), blend[:, :, None]], -1), w

    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    Atr, Wtr = analog_channels(sim_tr)
    Ava, Wva = analog_channels(sim_va)
    Xd = np.concatenate([Xd, Atr / Y_SCALE], -1)
    Vd = np.concatenate([Vd, Ava / Y_SCALE], -1)
    Xs = np.concatenate([Xs, Wtr], 1)
    Vs = np.concatenate([Vs, Wva], 1)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        model = _fit_one(Xe, Xd, Xs, Y, s)
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ============================================================ S42 TiDE (dense encoder)
class TiDE(nn.Module):
    """All-MLP encoder-decoder with a temporal decoder that re-injects each
    hour's own future weather — designed for known-future covariates."""

    def __init__(self, n_enc, n_dec, n_static, hidden=128, proj=8, dropout=0.15):
        super().__init__()
        H = len(PRED_HOURS)
        self.H = H
        self.feat_proj = nn.Sequential(nn.Linear(n_dec, 32), nn.ELU(), nn.Linear(32, proj))
        self.enc = nn.Sequential(
            nn.Linear(n_enc * 72 + n_static + proj * H, hidden), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ELU())
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, H * proj))
        self.temporal = nn.Sequential(nn.Linear(proj * 2, 32), nn.ELU(), nn.Linear(32, 1))
        self.skip = nn.Linear(n_enc * 72, H)
        self.proj_dim = proj

    def forward(self, enc, dec, static):
        B = enc.shape[0]
        flat = enc.reshape(B, -1)
        p = self.feat_proj(dec)                                  # (B, H, proj)
        h = self.enc(torch.cat([flat, static, p.reshape(B, -1)], -1))
        g = self.dec(h).reshape(B, self.H, self.proj_dim)
        out = self.temporal(torch.cat([g, p], -1)).squeeze(-1)
        return out + self.skip(flat)


def tide_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    from .ideas3 import _fit
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        model = _fit(TiDE, Xe, Xd, Xs, Y, s, lr=1e-3)
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# One member per distinct mechanism, equal weights, nothing fitted.
# s62 (bi-encoder + scoring-aligned weights + dual raw/sqrt loss) supersedes both the
# plain bi-encoder and the scoring-aligned one -- same architecture, strictly better
# objective -- so keeping all three was redundant.
# Quiet-tail bias correction. The ensemble systematically over-predicts after the
# event subsides: over hours 168-215 the training-county truth averages 0.00035 while
# the ensemble predicts 0.00065, whereas the bias in every earlier hour band is
# <= 0.00005 in absolute value. A least-squares scale fitted on the training counties'
# out-of-fold predictions is 0.294; we apply half that shrinkage so a test county with
# a genuine late outage is not crushed. Nested: RMSE 0.00888 -> 0.00886, MAE 0.00280 ->
# 0.00273, paired bootstrap P(better)=0.995 with a 95% CI excluding zero.
# Reproduce with fit_tail_scale.py.
TAIL_SPLIT_HOUR = 168
TAIL_SCALE = 0.667          # refit (same nested LS + half-shrink procedure) for the
                            # 8-member pool; the 7-member value was 0.647

# Metric interpretation committed 2026-08-13 (see PLAN.md / HANDOFF.md): the 2026
# wording ("assessed across all four horizons") mirrors the 2025 instrument verbatim
# ("average rank across the two horizons using RMSE"), and MAE appears in no document
# of either year. Under per-horizon RMSE with rank aggregation, the 8-member set --
# better at ALL four horizons, 3/5 folds -- is the right choice; the stock-flow member
# p1b also contributes the most decorrelated errors in the project (0.752).
# 2026-08-13: tabular member upgraded s45 -> f2 (same family + the physics
# interactions from the independence audit, REPORT 5x/5y). Supersession, not
# selection: standalone 0.00930 vs 0.00942, swap pointwise better at 4/4
# horizons (P 0.682), diversity profile unchanged (corr 0.845 vs others).
ENSEMBLE_FINAL_W = {k: 1/8 for k in
                    ["s62", "s51", "s39", "s40", "i11", "f2", "s35", "p1b"]}


# 2026-08-14: the shipped forecast is a 50/50 blend of the 8-member equal-weight
# ensemble with A3x25 (snapshot-ensembled bi-encoder, 25 seeds x 4 cosine-restart
# snapshots = 100 averaged predictions). Rationale: the 1/9-weight seat test caps
# any single added member's effect in the fifth decimal regardless of quality; two
# comparably good, structurally different predictors combined at the a-priori 50/50
# rule is the untested move. Measured 0.008544 vs 0.008621 (paired -0.000076,
# P 0.896), flat over lam in [0.3, 0.6]. lam=0.5 is the same "no reason to prefer
# either object" rule that justifies the 1/8 member weights - not a fitted value.
BLEND_LAMBDA = 0.5


def ensemble_final_forecaster(train_full, val_masked):
    """Final submission model: one member per distinct technique family.

    Six sequence architectures (bidirectional GRU under two training objectives,
    cross-county attention, component-structured GRU, Transformer, LSTM), three
    augmentation schemes (storm jitter, multi-origin, mixup), and two tabular
    families (sqrt-target boosting, extremely randomized trees).

    Member set follows an a priori rule -- include the best representative of each
    family that measured in the top tier, exclude redundant members of a family
    already represented -- rather than CV rank, because the paired county bootstrap
    cannot separate the candidates (this vs the 9-member set: -0.00008, P=0.96, CI
    marginally includes zero). An earlier 11-member set built from *redundant*
    members scored significantly worse, which is the evidence behind the rule.
    """
    from .ideas import augmented_gru_forecaster
    from .ideas3 import (bienc_forecaster, cross_county_fixed_forecaster,
                         mixup_forecaster)
    from .ideas5 import bienc_dual_forecaster
    from .ideas6 import bienc_scoring_dual_forecaster
    from .ideas10 import f2_member_upgrade
    from .stockflow import stockflow_nomono_forecaster
    fns = {"s62": bienc_scoring_dual_forecaster,
           "s51": bienc_dual_forecaster,
           "s39": cross_county_fixed_forecaster,
           "s40": multi_origin_forecaster,
           "i11": augmented_gru_forecaster,
           "f2": f2_member_upgrade,
           "s35": mixup_forecaster,
           "p1b": stockflow_nomono_forecaster}
    out = None
    for name, fn in fns.items():
        p = fn(train_full, val_masked).rename(columns={"osi_pred": name})
        out = p if out is None else out.merge(p, on=["fipsCode", "hour"])
    from .ideas8 import a3_s25
    out["ens8"] = sum(w * out[n] for n, w in ENSEMBLE_FINAL_W.items())
    a3 = a3_s25(train_full, val_masked).rename(columns={"osi_pred": "a3x25"})
    out = out.merge(a3, on=["fipsCode", "hour"])
    out["osi_pred"] = ((1 - BLEND_LAMBDA) * out["ens8"]
                       + BLEND_LAMBDA * out["a3x25"])
    tail = out["hour"] >= TAIL_SPLIT_HOUR
    out.loc[tail, "osi_pred"] *= TAIL_SCALE
    return out[["fipsCode", "hour", "osi_pred"]]


# --------------------------------------------------------------- freeze spec
# The W17 rebuild target. Differences from ensemble_final_forecaster, each
# measured and recorded in CANONICAL.md / REPORT.md:
#   s51  -> x9    cosine-restart snapshot ensembling
#   i11  -> i11w  metric per-hour weights (dominates x3 on all four horizons)
#   s35  -> x4    25 seeds
#   s40  dropped  nested drop-one picks it on all five folds
#   ts2c added    PatchTST-KF with metric per-hour weights
# County-holdout CV: 0.008468 vs 0.008543 for the shipped file.
FREEZE_MEMBERS = ["s62", "x9", "s39", "i11w", "f2", "x4", "p1b", "ts2c"]
FREEZE_W = {k: 1 / len(FREEZE_MEMBERS) for k in FREEZE_MEMBERS}


def ensemble_freeze_forecaster(train_full, val_masked):
    """Freeze-spec ensemble: 8 equal-weight members, 50/50 with A3x25, tail scaled.

    Mirrors `ensemble_final_forecaster` exactly in structure -- same blend
    lambda, same tail split and scale -- so the only differences are the member
    set and three members' training configuration.
    """
    from .ideas import augmented_gru_forecaster
    from .ideas3 import cross_county_fixed_forecaster, mixup_forecaster
    from .ideas5 import bienc_dual_forecaster
    from .ideas6 import bienc_scoring_dual_forecaster
    from .ideas10 import f2_member_upgrade
    from .ideas16 import t2c_patchtst_weighted
    from .stockflow import stockflow_nomono_forecaster
    w = scoring_weights()
    fns = {
        "s62":  bienc_scoring_dual_forecaster,
        "x9":   lambda tr, va: bienc_dual_forecaster(tr, va, seeds=(0, 1, 2),
                                                     cyclic=(40, 4)),
        "s39":  cross_county_fixed_forecaster,
        "i11w": lambda tr, va: augmented_gru_forecaster(tr, va, w=w),
        "f2":   f2_member_upgrade,
        "x4":   lambda tr, va: mixup_forecaster(tr, va, seeds=tuple(range(25))),
        "p1b":  stockflow_nomono_forecaster,
        "ts2c": t2c_patchtst_weighted,
    }
    out = None
    for name, fn in fns.items():
        p = fn(train_full, val_masked).rename(columns={"osi_pred": name})
        out = p if out is None else out.merge(p, on=["fipsCode", "hour"])
    from .ideas8 import a3_s25
    out["ens"] = sum(w_ * out[n] for n, w_ in FREEZE_W.items())
    a3 = a3_s25(train_full, val_masked).rename(columns={"osi_pred": "a3x25"})
    out = out.merge(a3, on=["fipsCode", "hour"])
    out["osi_pred"] = ((1 - BLEND_LAMBDA) * out["ens"] + BLEND_LAMBDA * out["a3x25"])
    tail = out["hour"] >= TAIL_SPLIT_HOUR
    out.loc[tail, "osi_pred"] *= TAIL_SCALE
    return out[["fipsCode", "hour", "osi_pred"]]


# --------------------------------------------------------- freeze spec v2 (W21)
# W23: f2 trains on sqrt(OSI) and squares back; this least-squares rescale removes part of
# the resulting Jensen bias. Nested: adopted on 5/5 folds, zero optimism gap, better on all
# four horizons. Fully correcting the bias (Duan smearing) makes the blend worse.
F2_RESCALE = 1.090366
FREEZE_V2_MEMBERS = ["s62", "x9", "s39", "i11w", "f2", "x4", "p1b", "ts2c", "p1c", "r6corr", "s35w"]


def ensemble_freeze_v2_forecaster(train_full, val_masked, cache_dir=None):
    """Freeze spec v2: 11 equal-weight members combined by amplitude x shape (src/combine.py),
    50/50 with A3x25, tail scaled. Differences from v1: +p1c (stock-flow, no DAgger), +r6corr
    (stock-flow with two flow-bias scalars), +s35w (mixup + metric hour weights), and the
    combination rule. CV 0.008360 (nested selection over the search: 0.008378).

    `cache_dir`: if given, each member's predictions for this target set are written there
    and reused on the next call, so a member-level change no longer costs a full 11-hour
    rebuild. The cache is keyed by member name and county count.
    """
    from pathlib import Path
    from .ideas import augmented_gru_forecaster
    from .ideas3 import cross_county_fixed_forecaster, mixup_forecaster
    from .ideas5 import bienc_dual_forecaster
    from .ideas6 import bienc_scoring_dual_forecaster
    from .ideas10 import f2_member_upgrade
    from .ideas16 import t2c_patchtst_weighted
    from .ideas8 import a3_s25
    from .stockflow import stockflow_forecaster, stockflow_nomono_forecaster, stockflow_r6_forecaster
    from .combine import ampshape
    w = scoring_weights()
    fns = {
        "s62":    bienc_scoring_dual_forecaster,
        "x9":     lambda tr, va: bienc_dual_forecaster(tr, va, seeds=(0, 1, 2), cyclic=(40, 4)),
        "s39":    cross_county_fixed_forecaster,
        "i11w":   lambda tr, va: augmented_gru_forecaster(tr, va, w=w),
        "f2":     f2_member_upgrade,
        "x4":     lambda tr, va: mixup_forecaster(tr, va, seeds=tuple(range(25))),
        "p1b":    stockflow_nomono_forecaster,
        "ts2c":   t2c_patchtst_weighted,
        "p1c":    lambda tr, va: stockflow_forecaster(tr, va, dagger_rounds=0),
        "r6corr": stockflow_r6_forecaster,
        "s35w":   lambda tr, va: mixup_forecaster(tr, va, w=w),
        "a3x25":  a3_s25,
    }
    n_c = val_masked["fipsCode"].nunique()
    preds, idx = {}, None
    for name, fn in fns.items():
        path = Path(cache_dir) / f"{name}_{n_c}c.csv" if cache_dir else None
        if path is not None and path.exists():
            p = pd.read_csv(path)
        else:
            p = fn(train_full, val_masked)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True); p.to_csv(path, index=False)
        v = p.set_index(["fipsCode", "hour"])["osi_pred"]
        if name == "f2":
            v = v * F2_RESCALE                      # W23: sqrt back-transform rescale
        preds[name] = v; idx = v.index if idx is None else idx.intersection(v.index)
    ens = ampshape([preds[m].reindex(idx) for m in FREEZE_V2_MEMBERS], idx)
    out = ((1 - BLEND_LAMBDA) * ens + BLEND_LAMBDA * preds["a3x25"].reindex(idx)).reset_index()
    out.columns = ["fipsCode", "hour", "osi_pred"]
    out.loc[out["hour"] >= TAIL_SPLIT_HOUR, "osi_pred"] *= TAIL_SCALE
    return out


ENSEMBLE_V9_W = {"s31": 0.18, "s39": 0.16, "i11": 0.14, "gru_comp": 0.11,
                 "transformer": 0.09, "s30": 0.09, "s29d": 0.08, "s29c": 0.08,
                 "s23b": 0.07}


def ensemble_v9_forecaster(train_full, val_masked):
    """Final ensemble: six sequence models spanning four architectures plus three
    tabular models spanning two families.

    Member set chosen for family diversity and protocol correctness, NOT for CV
    rank: the paired county bootstrap cannot distinguish this from the previous
    ensemble (delta +0.00003, P=0.27) or from several alternatives, so selecting on
    the fifth decimal would be fitting the validation set. The cross-county member
    is the self-leak-fixed S39 rather than the nominally-better I07 for the same
    reason -- indistinguishable in CV, correct in protocol.
    """
    from .dl import gru_component_forecaster, transformer_forecaster
    from .ideas import augmented_gru_forecaster
    from .ideas2 import sqrt_xgb_forecaster, sqrt_lgbm_forecaster, extratrees_forecaster
    from .ideas3 import (bienc_forecaster, cross_county_fixed_forecaster,
                         lstm_forecaster)
    fns = {"s31": bienc_forecaster,
           "s39": cross_county_fixed_forecaster,
           "i11": augmented_gru_forecaster,
           "gru_comp": gru_component_forecaster,
           "transformer": transformer_forecaster,
           "s51": bienc_dual_forecaster,
           "s45": subcounty_gbm_forecaster,
           "s29c": sqrt_lgbm_forecaster,
           "s23b": extratrees_forecaster}
    out = None
    for name, fn in fns.items():
        p = fn(train_full, val_masked).rename(columns={"osi_pred": name})
        out = p if out is None else out.merge(p, on=["fipsCode", "hour"])
    out["osi_pred"] = sum(w * out[n] for n, w in ENSEMBLE_V9_W.items())
    return out[["fipsCode", "hour", "osi_pred"]]


# ======================================== S44 scoring-aligned per-hour loss weighting
def scoring_weights():
    """Each submission column averages squared error over its own scored cells, so
    a target hour's weight in the metric is sum_j 1/n_j over the columns j that
    contain it. Late hours appear in all four columns and are weighted ~4.7x the
    earliest ones; uniform training loss therefore under-serves exactly the hours
    the metric cares about most."""
    from .data import HORIZONS
    w = np.zeros(len(PRED_HOURS))
    for h in HORIZONS.values():
        tgt = np.arange(FREEZE_H + 1 + h, 216)
        for t in tgt:
            w[t - PRED_HOURS[0]] += 1.0 / len(tgt)
    return w / w.mean()


def _fit_weighted(cls, Xe, Xd, Xs, Y, seed, w, max_epochs=300, patience=30, lr=2e-3):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = cls(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(w, dtype=torch.float32)
    mask = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)

    def wloss(out, tgt, msk):
        return (((out - tgt) ** 2) * tw * msk).sum() / (msk * tw).sum()

    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            wloss(model(t[0][b], t[1][b], t[2][b]), t[3][b], mask[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(wloss(model(t[0][va_i], t[1][va_i], t[2][va_i]),
                            t[3][va_i], mask[va_i]))
        if v < best - 1e-6:
            best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def _run_weighted(cls, train_full, val_masked, seeds=(0, 1, 2)):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    w = scoring_weights()
    preds = []
    for s in seeds:
        model = _fit_weighted(cls, Xe, Xd, Xs, Y, s, w)
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def scoring_aligned_gru_forecaster(train_full, val_masked):
    return _run_weighted(Seq2Seq, train_full, val_masked)


def scoring_aligned_bienc_forecaster(train_full, val_masked):
    from .ideas3 import BiEncSeq2Seq
    return _run_weighted(BiEncSeq2Seq, train_full, val_masked)


def bagged_sequence_forecaster(train_full, val_masked, n_bags=5, frac=0.8):
    """Bag a sequence model over county subsets (the move that made ExtraTrees beat
    boosting, applied to the best sequence architecture)."""
    from .ideas3 import BiEncSeq2Seq, _fit
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    rng = np.random.RandomState(SEED)
    preds = []
    for b in range(n_bags):
        sub = rng.choice(len(Xe), int(frac * len(Xe)), replace=False)
        model = _fit(BiEncSeq2Seq, Xe[sub], Xd[sub], Xs[sub], Y[sub], seed=b)
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


IDEAS4 = {
    "S44 scoring-aligned GRU": scoring_aligned_gru_forecaster,
    "S44b scoring-aligned bi-encoder": scoring_aligned_bienc_forecaster,
    "S43 bagged bi-encoder (county subsets)": bagged_sequence_forecaster,
    "S40 multi-origin augmentation": multi_origin_forecaster,
    "S41 RAFT retrieved analogs": raft_forecaster,
    "S42 TiDE (known-future covariates)": tide_forecaster,
}


# ============================ S46 sub-county heterogeneity inside the sequence models
def _tensors_sub(masked, dec_hours=PRED_HOURS):
    """_tensors with within-county gust-spread channels appended to the decoder."""
    from .subcounty import load_subcounty
    fips, enc, dec, static = _tensors(masked, dec_hours)
    sc = load_subcounty()
    if sc is False:
        return fips, enc, dec, static
    extra = []
    for c in ("gmax_ratio", "gp90_ratio", "gstd_ratio"):
        piv = sc[c].unstack("hour").reindex(index=fips, columns=dec_hours)
        extra.append(np.nan_to_num(piv.to_numpy(), nan=1.0 if "ratio" in c else 0.0))
    # local peak gust = county mean gust x within-county max ratio
    gi = DEC_COLS.index("gust")
    extra.append(dec[:, :, gi] * extra[0])
    return fips, enc, np.concatenate([dec, np.stack(extra, -1)], -1), static


def _prep_sub(train_full, val_masked):
    from .data import mask_after_freeze
    _, Xe, Xd, Xs = _tensors_sub(mask_after_freeze(train_full))
    Y = (train_full.pivot_table(index="fipsCode", columns="hour", values="osi")
         .sort_index().reindex(columns=PRED_HOURS).to_numpy() * Y_SCALE)
    stats = [(a.reshape(-1, a.shape[-1]).mean(0), a.reshape(-1, a.shape[-1]).std(0) + 1e-6)
             for a in (Xe, Xd, Xs)]
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]
    vf, Ve, Vd, Vs = _tensors_sub(val_masked)
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]
    return (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs)


def subcounty_bienc_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    from .ideas3 import BiEncSeq2Seq, _fit
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep_sub(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        model = _fit(BiEncSeq2Seq, Xe, Xd, Xs, Y, s)
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def subcounty_scoring_bienc_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    """Best single model (scoring-aligned bi-encoder) + sub-county channels."""
    from .ideas3 import BiEncSeq2Seq
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep_sub(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    w = scoring_weights()
    preds = []
    for s in seeds:
        model = _fit_weighted(BiEncSeq2Seq, Xe, Xd, Xs, Y, s, w)
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


IDEAS4["S46 sub-county bi-encoder"] = subcounty_bienc_forecaster
IDEAS4["S46b sub-county scoring-aligned bi-enc"] = subcounty_scoring_bienc_forecaster
