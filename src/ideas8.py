"""Sweep 8 (A-series) - variance reduction and objective alignment on S62.

All of these are built on `S62` (bi-encoder + scoring-aligned hour weights + dual
raw/sqrt objective, CV 0.00897), the best single model in the project, and every
one changes exactly one thing about it. Control for all rows: `oof/s62.csv`.

Motivation, per the standing empirical laws (HANDOFF section 4):

  law 1  raise independent, deployment-relevant effective sample size  -> A7, A8
  law 2  variance reduction wins                                       -> A2, A3
  and a measured gap: 22 of 239 counties carry 66% of the scored
  squared error while the training loss weights all 239 equally        -> A4

A5 closes a documented objective mismatch: `scoring_weights()` implements the
metric's cell-membership term (sum_j 1/n_j) but not the Jacobian of the actual
objective, which is a mean of per-horizon RMSEs rather than a weighted MSE.

None of these adds a parameter to the combination stage and none touches an
existing member, cache, or submission.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS
from .dl import Y_SCALE

IDEAS8 = {}

N_OUTAGE_CH = 4          # ENC_COLS[:4] = osi, P_t, N_t, R_t; the rest is weather


# --------------------------------------------------------------- objectives
def scoring_weights_exact(rmse_per_horizon=(0.01025, 0.00940, 0.00783, 0.00707)):
    """A5. The scored objective is mean_j RMSE_j, not a weighted MSE.

    d/dp mean_j sqrt(MSE_j) carries a 1/(2*RMSE_j) factor per horizon, so a target
    hour's true weight is sum_{j: h in j} 1 / (n_j * RMSE_j). `scoring_weights()`
    keeps the 1/n_j term and drops the 1/RMSE_j term, which silently under-weights
    the long horizons (where RMSE is smallest and each unit of squared error
    therefore moves the metric most).

    The RMSE_j constants are the incumbent ensemble's out-of-fold per-horizon
    values - four fixed numbers, not fitted per fold, so this adds no free
    parameter and cannot overfit a fold.
    """
    from .data import HORIZONS
    w = np.zeros(len(PRED_HOURS))
    for (_, h), r in zip(HORIZONS.items(), rmse_per_horizon):
        tgt = np.arange(FREEZE_H + 1 + h, 216)
        w[tgt - PRED_HOURS[0]] += 1.0 / (len(tgt) * r)
    return w / w.mean()


def county_weights(Y, gamma):
    """A4. Weight each training county by its expected share of the metric.

    The metric is decided by a few severe counties (12 of 239 carry ~65% of the
    scored squared error) while the training loss weights every county equally.
    Weight ~ (county rms target)^gamma, normalised to mean 1; gamma=0 is the
    incumbent. Computed from the TRAINING counties' own targets only - a loss
    weight, never a feature, so no test-side information is involved.

    gamma is deliberately kept small: upweighting severe counties trades effective
    sample size for metric alignment, and law 4 says this project loses whenever
    sample size is fragmented.
    """
    rms = np.sqrt(np.nanmean(np.where(np.isfinite(Y), Y, np.nan) ** 2, axis=1))
    rms = np.nan_to_num(rms, nan=float(np.nanmin(rms)))
    w = (rms / rms.mean()) ** gamma
    return w / w.mean()


# ------------------------------------------------------------ augmentations
def _augment(batch_enc, rng, mode):
    """Encoder-side augmentation, applied per batch, on standardised inputs
    (so 0 is the channel mean and masking to 0 means 'no information')."""
    x = batch_enc.clone()
    if mode == "chan":                       # A7a: drop whole input channels
        for i in range(len(x)):
            if rng.rand() < 0.5:
                c = rng.randint(0, x.shape[-1])
                x[i, :, c] = 0.0
    elif mode == "block":                    # A7b: drop a contiguous hour block
        for i in range(len(x)):
            if rng.rand() < 0.5:
                L = rng.randint(4, 13)
                s = rng.randint(0, max(1, x.shape[1] - L))
                x[i, s:s + L, :] = 0.0
    elif mode == "prefix":                   # A8: truncate the early prefix
        for i in range(len(x)):
            t = int(rng.choice([0, 0, 12, 24, 36]))
            if t:
                x[i, :t, :] = 0.0
    return x


def _tta_views(Ve, n_views, rng, sigma=0.05, max_shift=2):
    """A2. Inference-side views: small noise plus +-1..2h encoder time shifts.
    View 0 is always the unperturbed input, so TTA strictly contains the control."""
    views = [Ve]
    for _ in range(n_views - 1):
        v = Ve.copy()
        s = rng.randint(-max_shift, max_shift + 1)
        if s:
            v = np.roll(v, s, axis=1)
            if s > 0:
                v[:, :s, :] = Ve[:, :1, :]
            else:
                v[:, s:, :] = Ve[:, -1:, :]
        v = v + rng.randn(*v.shape).astype(v.dtype) * sigma
        views.append(v)
    return views


# ------------------------------------------------------------------ trainer
def _fit(cls, Xe, Xd, Xs, Y, seed, w, cw=None, aug=None,
         max_epochs=300, patience=30, lr=2e-3, cyclic=None):
    """S62's trainer (`_fit_scoring_dual`) with three optional hooks.

    cw      per-county loss weights (A4)
    aug     encoder augmentation mode (A7/A8)
    cyclic  (period, n_snapshots) -> cosine warm restarts, returns snapshots (A3)
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = cls(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(w, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))
    cwt = (torch.ones(len(Xe)) if cw is None
           else torch.tensor(cw, dtype=torch.float32))

    def loss_fn(out, i):
        cwi = cwt[i][:, None]
        den = (msk[i] * tw * cwi).sum()
        raw = (((torch.clamp(out, min=0) ** 2) - t[3][i]) ** 2 * tw * msk[i] * cwi).sum() / den
        root = ((out - sq[i]) ** 2 * tw * msk[i] * cwi).sum() / den
        return 0.5 * raw + 0.5 * root

    snaps, best, state, bad = [], np.inf, None, 0
    n_ep = max_epochs if cyclic is None else cyclic[0] * cyclic[1]
    for ep in range(n_ep):
        if cyclic is not None:                       # cosine warm restarts
            pos = (ep % cyclic[0]) / cyclic[0]
            for gp in opt.param_groups:
                gp["lr"] = lr * 0.5 * (1 + np.cos(np.pi * pos))
        model.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            enc = t[0][b] if aug is None else _augment(t[0][b], rng, aug)
            loss_fn(model(enc, t[1][b], t[2][b]), b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.clamp(model(t[0][va_i], t[1][va_i], t[2][va_i]), min=0) ** 2
            v = float((((p - t[3][va_i]) ** 2) * tw * msk[va_i]).sum() / (msk[va_i] * tw).sum())
        if cyclic is not None:
            if (ep + 1) % cyclic[0] == 0:            # end of a cycle -> snapshot
                snaps.append({k: q.clone() for k, q in model.state_dict().items()})
            continue
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if cyclic is not None:
        return model, snaps
    model.load_state_dict(state)
    model.eval()
    return model, None


def _run(train_full, val_masked, seeds=(0, 1, 2), weights="s44", gamma=0.0,
         aug=None, tta=0, cyclic=None):
    from .ideas import _prep, _dl_out
    from .ideas3 import BiEncSeq2Seq
    from .ideas4 import scoring_weights
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    w = scoring_weights() if weights == "s44" else scoring_weights_exact()
    cw = county_weights(Y, gamma) if gamma else None
    preds = []
    for s in seeds:
        m, snaps = _fit(BiEncSeq2Seq, Xe, Xd, Xs, Y, s, w, cw=cw, aug=aug, cyclic=cyclic)
        rng = np.random.RandomState(1000 + s)
        views = _tta_views(Ve, tta, rng) if tta else [Ve]
        states = snaps if snaps else [None]
        for st in states:
            if st is not None:
                m.load_state_dict(st)
                m.eval()
            for V in views:
                tv = [torch.tensor(v, dtype=torch.float32) for v in (V, Vd, Vs)]
                with torch.no_grad():
                    preds.append((torch.clamp(m(*tv), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ----------------------------------------------------------------- registry
def a2_tta4(tr, va):      return _run(tr, va, tta=4)
def a2_tta8(tr, va):      return _run(tr, va, tta=8)
def a3_snapshot(tr, va):  return _run(tr, va, cyclic=(40, 4))
def a4_g025(tr, va):      return _run(tr, va, gamma=0.25)
def a4_g050(tr, va):      return _run(tr, va, gamma=0.50)
def a5_exact(tr, va):     return _run(tr, va, weights="exact")
def a7_chan(tr, va):      return _run(tr, va, aug="chan")
def a7_block(tr, va):     return _run(tr, va, aug="block")
def a8_prefix(tr, va):    return _run(tr, va, aug="prefix")
def a1_seed8(tr, va):     return _run(tr, va, seeds=tuple(range(8)))


# ---------------------- RT1: Ray Tune (ASHA) winning config, nested selection
# Config chosen purely on fold-0's inner training split (ray_tune_bienc.py);
# no outer fold influenced it. hidden 64, dropout 0.062, lr 2.97e-3,
# wd 7.8e-6, batch 16, dual-loss alpha 0.669 (vs defaults 48/0.15/2e-3/1e-4/
# 32/0.5). Evaluated here with the untouched harness.
def _fit_rt(Xe, Xd, Xs, Y, seed, w, cfg, max_epochs=300, patience=30):
    from .ideas3 import BiEncSeq2Seq
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = BiEncSeq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1],
                         hidden=cfg["hidden"], dropout=cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(w, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))
    a = cfg["alpha"]

    def loss_fn(out, i):
        raw = torch.clamp(out, min=0) ** 2
        den = (msk[i] * tw).sum()
        return (a * ((raw - t[3][i]) ** 2 * tw * msk[i]).sum() / den
                + (1 - a) * ((out - sq[i]) ** 2 * tw * msk[i]).sum() / den)

    best, state, bad = np.inf, None, 0
    nb = max(1, len(tr_i) // cfg["batch"])
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr_i), nb):
            opt.zero_grad(); loss_fn(model(t[0][b], t[1][b], t[2][b]), b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.clamp(model(t[0][va_i], t[1][va_i], t[2][va_i]), min=0) ** 2
            v = float((((p - t[3][va_i]) ** 2) * tw * msk[va_i]).sum()
                      / (msk[va_i] * tw).sum())
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model


RT_CFG = {"hidden": 64, "dropout": 0.0619, "lr": 2.966e-3, "wd": 7.82e-6,
          "batch": 16, "alpha": 0.6694}
# RT2: winner of the metric-objective search (ray_tune_metric.py) - trial
# scored by the TRUE competition metric on an inner 3-fold county-holdout of
# fold-0's training counties (seed-42 stratified machinery). Unlike RT1's
# loss-proxy winner, this one KEEPS regularization (dropout 0.26).
# Inner metric 0.01045 vs default's 0.01099 under the identical protocol.
RT2_CFG = {"hidden": 64, "dropout": 0.2642, "lr": 5.344e-3, "wd": 1.40e-5,
           "batch": 16, "alpha": 0.6699}
# RT3: Optuna TPE winner (optuna_tune_metric.py) - same nested true-metric
# protocol as RT2, model-based sampler warm-started AT the defaults. Went the
# opposite way from ASHA: smaller (hidden 32), near-default dropout/alpha.
# Inner metric 0.01068 vs defaults' 0.01099.
RT3_CFG = {"hidden": 32, "dropout": 0.1405, "lr": 1.200e-3, "wd": 9.31e-6,
           "batch": 16, "alpha": 0.5117}


def rt2_tuned(tr_full, va_masked, seeds=(0, 1, 2)):
    from .ideas import _prep, _dl_out
    from .ideas4 import scoring_weights
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(tr_full, va_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    w = scoring_weights()
    preds = []
    for s2 in seeds:
        m = _fit_rt(Xe, Xd, Xs, Y, s2, w, RT2_CFG)
        with torch.no_grad():
            preds.append((torch.clamp(m(*tv), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def rt3_tuned(tr_full, va_masked, seeds=(0, 1, 2)):
    from .ideas import _prep, _dl_out
    from .ideas4 import scoring_weights
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(tr_full, va_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    w = scoring_weights()
    preds = []
    for s3 in seeds:
        m = _fit_rt(Xe, Xd, Xs, Y, s3, w, RT3_CFG)
        with torch.no_grad():
            preds.append((torch.clamp(m(*tv), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def rt1_tuned(tr_full, va_masked, seeds=(0, 1, 2)):
    from .ideas import _prep, _dl_out
    from .ideas4 import scoring_weights
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(tr_full, va_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    w = scoring_weights()
    preds = []
    for s in seeds:
        m = _fit_rt(Xe, Xd, Xs, Y, s, w, RT_CFG)
        with torch.no_grad():
            preds.append((torch.clamp(m(*tv), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ---------------------- A-series top-3 at 25 seeds (user request, 2026-08-13)
# Identical code paths to A3 / A11 / A1; the ONLY change is seeds=range(25).
# A3/A11 average 25 seeds x 4 snapshots = 100 predictions; A1 averages 25.
def a3_s25(tr, va):   return _run(tr, va, seeds=tuple(range(25)), cyclic=(40, 4))
def a11_s25(tr, va):  return _run(tr, va, seeds=tuple(range(25)), weights="exact",
                                  cyclic=(40, 4))
def a1_s25(tr, va):   return _run(tr, va, seeds=tuple(range(25)))


# ------------------------------------- A9 post-freeze training origins (vs S40)
# Every origin ever evaluated in this project sits at or before hour 79: S40
# (47/59/71, 0.00936, adopted), S63 (39..71, 0.01016), S63b (23..79, 0.01008),
# S63c (55..71, 0.01026). The measured failure mode was REDUNDANCY - origins four
# hours apart yield near-identical encoder sequences - not lateness.
#
# An origin in the post-wave-1 lull is the opposite of redundant. It manufactures
# the one training pair the current scheme never produces: observe the wave-1
# decay, forecast wave 2. That is structurally the hardest part of the scored
# window, and from origin 71 it is supervised only as a distant target.
#
# Legality: only TRAINING counties are used at these origins, and they are fully
# observed, so no county's inputs ever include a value after its own origin. The
# deployed model is evaluated at origin 71 only (validation tensors are built at
# FREEZE_H), and early stopping stays on the origin-71 block.
def _mo(tr, va, origins):
    from .ideas4 import multi_origin_forecaster
    return multi_origin_forecaster(tr, va, origins=origins)


def a9_lull(tr, va):      return _mo(tr, va, (47, 71, 103))
def a9b_min(tr, va):      return _mo(tr, va, (71, 103))
def a9c_four(tr, va):     return _mo(tr, va, (47, 59, 71, 103))
def a9d_late(tr, va):     return _mo(tr, va, (47, 71, 127))


# ------------------- A10 trailing-window multi-origin: PREMISE FALSIFIED, NOT RUN
#
# A10 was written to remove a confound I believed A9 suffered from: S40 left-pads
# every origin block to the longest encoder, so introducing a late origin makes
# the deployment (origin-71) block acquire ~32 hours of leading zeros. I assumed
# inference did not see the same padding, making it a train/deploy mismatch.
#
# That assumption is wrong. `multi_origin_forecaster` pads the VALIDATION tensor
# to the same L:
#     Ve = np.pad(Ve, ((0, 0), (L - Ve.shape[1], 0), (0, 0)))
# so the training origin-71 block and the inference tensor carry identical leading
# zeros. There is no mismatch, and A9's four negatives are therefore already clean
# tests of late origins (0.00960 / 0.00973 / 0.00984 / 0.00997 vs S40's 0.00936).
#
# Kept, unrun, because the reimplementation also surfaced two real bugs worth
# recording: targets must come from the unmasked frame while statics come from the
# masked one (collapsing them zeroes the main block's targets), and the original's
# loss normalises per-sample over ALL decoder hours before weighting samples, not
# globally over valid cells - a matched control that got this wrong scored 0.00900
# against S40's 0.00712 on the same fold. Anyone reviving this must fix the loss
# normalisation first.
def _trailing_tensors(masked, full, origin, dec_hours, enc_len=FREEZE_H + 1):
    """Statics/weather from `masked` (so the main origin's statics match
    deployment); encoder, origin state and targets from `full`. Mirrors
    `_origin_tensors` - collapsing the two frames into one silently drew the
    main block's TARGETS from the masked frame, where osi is NaN after h71."""
    from .dl import ENC_COLS, DEC_COLS, STATIC_COLS, _pivot
    from .features import county_static, weather_by_hour
    st = county_static(masked)
    w = weather_by_hour(masked)
    enc_hours = np.arange(origin - enc_len + 1, origin + 1)
    fips = _pivot(masked, "gust", enc_hours).index.to_numpy()
    enc = np.stack([_pivot(full, c, enc_hours).to_numpy() for c in ENC_COLS], -1)
    dec = np.stack([_pivot(w, c, dec_hours).to_numpy() for c in DEC_COLS], -1)
    dt = (dec_hours - origin).astype(float)
    osi_o = _pivot(full, "osi", np.array([origin])).to_numpy()
    extras = np.stack([np.repeat(dt[None, :] / 100.0, len(fips), 0),
                       osi_o * np.exp(-dt / 16.0) * Y_SCALE,
                       osi_o * np.exp(-dt / 32.0) * Y_SCALE], -1)
    dec = np.concatenate([dec, extras], -1)
    static = np.concatenate([st.loc[fips, STATIC_COLS].to_numpy(),
                             np.full((len(fips), 1), origin / 216.0)], 1)
    Y = _pivot(full, "osi", dec_hours).to_numpy() * Y_SCALE
    return (np.nan_to_num(enc), np.nan_to_num(dec), np.nan_to_num(static),
            np.nan_to_num(Y, nan=np.nan), fips)


def trailing_multi_origin(train_full, val_masked, seeds=(0, 1, 2),
                          origins=(47, 71, 103), aux_weight=0.5):
    from .data import mask_after_freeze
    from .dl import Seq2Seq, _tensors
    from .ideas import _dl_out
    H = len(PRED_HOURS)
    masked_tr = mask_after_freeze(train_full)
    blocks = []
    for o in origins:
        dh = np.arange(o + 2, o + 2 + H)
        dh = dh[dh <= 215]
        pad = H - len(dh)
        enc, dec, st, Y, _ = _trailing_tensors(
            masked_tr if o == FREEZE_H else train_full, train_full, o, dh)
        if pad:
            dec = np.concatenate([dec, np.repeat(dec[:, -1:], pad, 1)], 1)
            Y = np.concatenate([Y, np.full((len(Y), pad), np.nan)], 1)
        blocks.append((enc, dec, st, Y, o))
    Xe = np.concatenate([b[0] for b in blocks], 0)
    Xd = np.concatenate([b[1] for b in blocks], 0)
    Xs = np.concatenate([b[2] for b in blocks], 0)
    Y = np.concatenate([b[3] for b in blocks], 0)
    W = np.concatenate([np.full(len(b[0]), 1.0 if b[4] == FREEZE_H else aux_weight)
                        for b in blocks])
    is_main = np.concatenate([np.full(len(b[0]), b[4] == FREEZE_H) for b in blocks])

    stats = [(a.reshape(-1, a.shape[-1]).mean(0), a.reshape(-1, a.shape[-1]).std(0) + 1e-6)
             for a in (Xe, Xd, Xs)]
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]

    vf, Ve, Vd, Vs = _tensors(val_masked, PRED_HOURS)
    Vs = np.concatenate([Vs, np.full((len(Vs), 1), FREEZE_H / 216.0)], 1)
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]

    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(W, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    n_main = int(is_main.sum())
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        main_idx = np.where(is_main)[0]
        hold_c = set(rng.permutation(n_main)[:max(12, n_main // 8)].tolist())
        tr_i = np.array([i for i in range(len(Xe)) if (i % n_main) not in hold_c])
        va_i = np.array([i for i in main_idx if (i % n_main) in hold_c])
        model = Seq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

        def loss(i):
            o = model(t[0][i], t[1][i], t[2][i])
            w2 = tw[i][:, None]
            return (((o - t[3][i]) ** 2) * msk[i] * w2).sum() / (msk[i] * w2).sum()

        best, state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad(); loss(b).backward(); opt.step()
            model.eval()
            with torch.no_grad():                     # early stop on the MAIN origin
                v = float(loss(va_i))
            if v < best - 1e-6:
                best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(state); model.eval()
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ------------------------- A11 composing the two mechanisms that worked alone
# S62 itself was built this way: bi-encoder + scoring weights + dual loss, each of
# which worked on its own. A3 (snapshot ensembling, 0.00869) and A5 (exact-metric
# hour weights, 0.00888) are independent of each other - one changes the optimiser
# trajectory, the other the loss - so they should compose rather than conflict.
# A11b/A11c vary the cycle budget to check that A3's gain is not a lucky (40, 4).
def a11_snap_exact(tr, va):   return _run(tr, va, weights="exact", cyclic=(40, 4))
def a11b_snap6(tr, va):       return _run(tr, va, weights="exact", cyclic=(30, 6))
def a11c_snap_long(tr, va):   return _run(tr, va, weights="exact", cyclic=(60, 3))
def a11d_snap_s44(tr, va):    return _run(tr, va, cyclic=(30, 6))


def a10_ctl(tr, va):      return trailing_multi_origin(tr, va, origins=(47, 59, 71))
def a10_lull(tr, va):     return trailing_multi_origin(tr, va, origins=(47, 71, 103))
def a10b_four(tr, va):    return trailing_multi_origin(tr, va, origins=(47, 59, 71, 103))
def a10c_two(tr, va):     return trailing_multi_origin(tr, va, origins=(71, 103))


IDEAS8 = {
    "A1 S62 8 seeds": a1_seed8,
    "A2 S62 + TTA 4 views": a2_tta4,
    "A2b S62 + TTA 8 views": a2_tta8,
    "A3 S62 + snapshot ensemble (cosine restarts)": a3_snapshot,
    "A4 S62 + county loss weight g=0.25": a4_g025,
    "A4b S62 + county loss weight g=0.50": a4_g050,
    "A5 S62 + exact-metric hour weights": a5_exact,
    "A7 S62 + channel-dropout augmentation": a7_chan,
    "A7b S62 + hour-block dropout augmentation": a7_block,
    "A8 S62 + prefix-truncation augmentation": a8_prefix,
    "A9 multi-origin 47/71/103 (lull origin)": a9_lull,
    "A9b multi-origin 71/103": a9b_min,
    "A9c multi-origin 47/59/71/103": a9c_four,
    "A9d multi-origin 47/71/127 (wave-2 origin)": a9d_late,
    "RT1 ray-tuned bi-encoder (nested ASHA config)": rt1_tuned,
    "RT2 ray-tuned on the true metric (nested)": rt2_tuned,
    "RT3 optuna-TPE on the true metric (nested)": rt3_tuned,
    "A3x25 snapshot ensemble, 25 seeds": a3_s25,
    "A11x25 snapshot + exact-metric, 25 seeds": a11_s25,
    "A1x25 S62, 25 seeds": a1_s25,
    "A11 snapshot + exact-metric weights": a11_snap_exact,
    "A11b snapshot 6 cycles + exact-metric": a11b_snap6,
    "A11c snapshot 3 long cycles + exact-metric": a11c_snap_long,
    "A11d snapshot 6 cycles, S44 weights": a11d_snap_s44,
}
