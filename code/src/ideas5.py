"""Sweep 5: county information for the sequence models, target transforms on
sequences, per-column specialists, state residuals, and a severe-event auxiliary head.

Motivated by two observations:
  (a) the external county attributes that produced the largest tabular gain
      (0.01094 -> 0.01047) are NOT in STATIC_COLS, so no sequence model has ever
      seen them;
  (b) knowing a county's true current state would make t+1h far better (0.00456)
      but t+24h/t+48h far worse than our weather-driven model -- the horizons want
      different things, so they need not share one predictor.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS, HORIZONS, mask_after_freeze
from .dl import Y_SCALE, STATIC_COLS, Seq2Seq, _tensors, _pivot
from .ideas import _dl_out, load_static, add_static
from .ideas3 import BiEncSeq2Seq, _fit
from .ideas4 import scoring_weights, _fit_weighted
from .features import county_static
from .models import SEED

EXT = ["log_pop", "log_density", "log_area", "rucc", "lat", "lon"]


# ------------------------------------------------- tensors with external county data
def _tensors_ext(masked, dec_hours=PRED_HOURS):
    """_tensors, with public county attributes appended to the static vector.

    These are per-county constants, so they cost a handful of input dimensions
    rather than the per-hour channels that hurt the sequence models in S46.
    """
    fips, enc, dec, static = _tensors(masked, dec_hours)
    s = load_static().reindex(fips)
    cust = np.expm1(county_static(masked).loc[fips, "log_customers"].to_numpy())
    extra = np.column_stack([
        s["log_pop"].to_numpy(), s["log_density"].to_numpy(), s["log_area"].to_numpy(),
        s["rucc"].to_numpy(), s["lat"].to_numpy(), s["lon"].to_numpy(),
        np.log1p(cust / np.exp(s["log_area"].to_numpy())),      # customers per sq mile
        np.log1p(cust) - s["log_pop"].to_numpy(),               # customers per capita
    ])
    return fips, enc, dec, np.concatenate([static, np.nan_to_num(extra)], 1)


def _prep_ext(train_full, val_masked, tensor_fn=_tensors_ext):
    _, Xe, Xd, Xs = tensor_fn(mask_after_freeze(train_full))
    Y = (train_full.pivot_table(index="fipsCode", columns="hour", values="osi")
         .sort_index().reindex(columns=PRED_HOURS).to_numpy() * Y_SCALE)
    stats = [(a.reshape(-1, a.shape[-1]).mean(0), a.reshape(-1, a.shape[-1]).std(0) + 1e-6)
             for a in (Xe, Xd, Xs)]
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]
    vf, Ve, Vd, Vs = tensor_fn(val_masked)
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]
    return (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs)


def _run(cls, train_full, val_masked, seeds=(0, 1, 2), weighted=False, **kw):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep_ext(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    w = scoring_weights()
    preds = []
    for s in seeds:
        model = (_fit_weighted(cls, Xe, Xd, Xs, Y, s, w) if weighted
                 else _fit(cls, Xe, Xd, Xs, Y, s, **kw))
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# =============================== S47-S49  external county data in the sequence models
def bienc_ext_forecaster(train_full, val_masked):
    return _run(BiEncSeq2Seq, train_full, val_masked)


def bienc_ext_scoring_forecaster(train_full, val_masked):
    return _run(BiEncSeq2Seq, train_full, val_masked, weighted=True)


def gru_ext_forecaster(train_full, val_masked):
    return _run(Seq2Seq, train_full, val_masked)


# ======================================== S50-S51  target transforms on the sequences
def _wmse(a, b, w):
    """MSE with per-target-hour weights, normalised to mean 1 so the loss scale
    (and hence the learning rate) is unchanged when w is supplied."""
    return (((a - b) ** 2) * w).mean() if w is not None else nn.functional.mse_loss(a, b)


def _fit_transformed(cls, Xe, Xd, Xs, Y, seed, fwd, dual=False,
                     max_epochs=300, patience=30, cyclic=None, w=None):
    """cyclic=(period, n_snapshots) enables cosine warm restarts and returns a
    LIST of models, one per cycle end (snapshot ensembling, law 13). Default
    None reproduces the original single-model path exactly."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = cls(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tf = fwd(t[3])
    tw = None if w is None else torch.tensor(w, dtype=torch.float32)
    best, state, bad = np.inf, None, 0
    snaps = []
    n_ep = max_epochs if cyclic is None else cyclic[0] * cyclic[1]
    for ep in range(n_ep):
        if cyclic is not None:                    # cosine warm restart schedule
            pos = (ep % cyclic[0]) / cyclic[0]
            for gp in opt.param_groups:
                gp["lr"] = 2e-3 * 0.5 * (1 + np.cos(np.pi * pos))
        model.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            out = model(t[0][b], t[1][b], t[2][b])
            loss = _wmse(out, tf[b], tw)
            if dual:      # keep the raw scale in the objective so peaks are not lost
                loss = 0.5 * loss + 0.5 * _wmse(
                    torch.clamp(out, min=0) ** 2, t[3][b], tw)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.clamp(model(t[0][va_i], t[1][va_i], t[2][va_i]), min=0) ** 2
            v = _wmse(p, t[3][va_i], tw).item()                # early stop on RAW scale
        if cyclic is not None:
            if (ep + 1) % cyclic[0] == 0:         # end of a cycle -> take a snapshot
                m2 = cls(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
                m2.load_state_dict({k: q.clone() for k, q in model.state_dict().items()})
                m2.eval()
                snaps.append(m2)
            continue
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if cyclic is not None:
        return snaps
    model.load_state_dict(state)
    model.eval()
    return model


def _run_transformed(cls, train_full, val_masked, dual=False, seeds=(0, 1, 2),
                     cyclic=None, w=None):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep_ext(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        m = _fit_transformed(cls, Xe, Xd, Xs, Y, s, torch.sqrt, dual=dual,
                             cyclic=cyclic, w=w)
        with torch.no_grad():
            for mm in (m if cyclic is not None else [m]):
                preds.append((torch.clamp(mm(*tv), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def bienc_sqrt_forecaster(train_full, val_masked):
    return _run_transformed(BiEncSeq2Seq, train_full, val_masked, dual=False)


def bienc_dual_forecaster(train_full, val_masked, seeds=(0, 1, 2), cyclic=None):
    return _run_transformed(BiEncSeq2Seq, train_full, val_masked, dual=True,
                            seeds=seeds, cyclic=cyclic)


# ============================================== S52  per-column horizon specialists
def per_column_specialist_forecaster(train_full, val_masked, seeds=(0, 1)):
    """Four models, each trained only on the cells ITS submission column scores.

    The columns overlap but are not identical -- t+1h scores hours 73-215 while
    t+48h scores only 120-215 -- so a specialist can spend all its capacity on the
    part of the trajectory it is graded on. Returns an origin-indexed frame.
    """
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep_ext(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    out = []
    for name, h in HORIZONS.items():
        w = np.zeros(len(PRED_HOURS))
        w[PRED_HOURS >= FREEZE_H + 1 + h] = 1.0          # only this column's cells
        w = w / w.mean()
        preds = []
        for s in seeds:
            model = _fit_weighted(BiEncSeq2Seq, Xe, Xd, Xs, Y, s, w)
            with torch.no_grad():
                preds.append(model(*tv).numpy())
        P = np.clip(np.mean(preds, 0) / Y_SCALE, 0, None)
        df = pd.DataFrame(P, index=vf, columns=PRED_HOURS).stack().reset_index()
        df.columns = ["fipsCode", "hour", "osi_pred"]
        df["h"] = float(h)
        df["origin"] = df["hour"] - h
        out.append(df[(df["origin"] >= 72) & (df["origin"] <= 215)])
    return pd.concat(out, ignore_index=True)[["fipsCode", "origin", "h", "hour", "osi_pred"]]


# ================================================ S53  shared encoder + state residuals
class StateResidualNet(nn.Module):
    """One pooled decoder plus a small per-state residual head, shrunk toward zero."""

    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15, n_states=4):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.res = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_states)])
        for r in self.res:                       # start at zero: pure pooled model
            nn.init.zeros_(r.weight); nn.init.zeros_(r.bias)

    def forward(self, enc, dec, static):
        _, h = self.enc(enc)
        h0 = self.proj(torch.cat([h[0], h[1]], -1))[None].contiguous()
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        o, _ = self.dec(torch.cat([dec, s], -1), h0)
        o = self.drop(o)
        base = self.head(o).squeeze(-1)
        # first four static dims are the state one-hots
        w = static[:, :4]
        resid = torch.stack([r(o).squeeze(-1) for r in self.res], -1)
        return base + 0.25 * (resid * w[:, None, :]).sum(-1)


def state_residual_forecaster(train_full, val_masked):
    return _run(StateResidualNet, train_full, val_masked)


# ============================================ S54  severe-event auxiliary head
class SevereAuxNet(BiEncSeq2Seq):
    """BiGRU with an auxiliary head predicting threshold exceedance, to force the
    encoder to retain information about rare severe episodes."""

    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__(n_enc, n_dec, n_static, hidden, dropout)
        self.aux = nn.Linear(hidden, 3)

    def both(self, enc, dec, static):
        _, h = self.enc(enc)
        h0 = self.proj(torch.cat([h[0], h[1]], -1))[None].contiguous()
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        o, _ = self.dec(torch.cat([dec, s], -1), h0)
        o = self.drop(o)
        return self.head(o).squeeze(-1), self.aux(o)


def severe_aux_forecaster(train_full, val_masked, seeds=(0, 1, 2), aux_w=0.15):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep_ext(train_full, val_masked)
    thr = np.array([0.02, 0.05, 0.10]) * Y_SCALE
    A = (np.nan_to_num(Y)[:, :, None] > thr[None, None, :]).astype(np.float32)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y), A)]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = SevereAuxNet(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                o, a = model.both(t[0][b], t[1][b], t[2][b])
                loss = (nn.functional.mse_loss(o, t[3][b])
                        + aux_w * nn.functional.binary_cross_entropy_with_logits(a, t[4][b]))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(
                    model(t[0][va_i], t[1][va_i], t[2][va_i]), t[3][va_i]).item()
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


def x9w_s51_snap_weighted(tr, va):
    """X9 (S51 + cosine-restart snapshots) with the metric's per-hour weights.

    S51 trains on a flat MSE while S62, I11, F2 and A3x25 all train against
    `scoring_weights()`. X9 is the strongest individual member of the freeze
    spec, so it is the most valuable place to close that gap."""
    from .ideas4 import scoring_weights
    return _run_transformed(BiEncSeq2Seq, tr, va, dual=True, cyclic=(40, 4),
                            w=scoring_weights())


IDEAS5 = {
    "S47 BiGRU + external county data": bienc_ext_forecaster,
    "S47b BiGRU + ext + scoring-aligned": bienc_ext_scoring_forecaster,
    "S48 GRU + external county data": gru_ext_forecaster,
    "S50 BiGRU sqrt target": bienc_sqrt_forecaster,
    "S51 BiGRU dual raw+sqrt loss": bienc_dual_forecaster,
    "X9w s51 + snapshots + metric hour weights": x9w_s51_snap_weighted,
    "S52 per-column horizon specialists": per_column_specialist_forecaster,
    "S53 shared encoder + state residuals": state_residual_forecaster,
    "S54 severe-event auxiliary head": severe_aux_forecaster,
}
