"""P3 test-time latent county adaptation, and P4 storm-phase canonicalization.

P3. A global sequence model carries one regularized latent scalar z_i per county,
injected FiLM-style into the decoder's initial state (not concatenated onto inputs).
For an unseen county the network is frozen, z is initialised at 0 and optimised for a
few gradient steps against that county's OBSERVED hours only, then frozen and used to
forecast. Self-consistency matters: the model decodes hours 48-215 from an encoder over
h0-47, so the hours used to fit z (48-71, observed) are part of the very same decode
the forecast comes from. No future outage value is ever touched.

P4. Counties experience the second storm at slightly different clock times. Using
weather only, each county's wave-2 gust peak is mapped to a common canonical hour by a
piecewise-linear warp anchored at the freeze boundary and the end of the window, the
model is trained on warped trajectories, and predictions are inverse-warped back to
clock time before scoring.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .dl import Y_SCALE, DEC_COLS, STATIC_COLS, ENC_COLS, _pivot
from .features import county_static, weather_by_hour
from .ideas import _dl_out, _prep
from .ideas3 import BiEncSeq2Seq, _fit
from .ideas4 import scoring_weights
from .models import SEED

ADAPT_START = 48        # decode from here; hours 48..71 are observed and fit z


# ============================================================ P3 latent adaptation
class LatentBiEnc(nn.Module):
    """Bi-encoder whose decoder initial state is modulated by a scalar latent."""

    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        # NOT zeros: with z initialised at 0 too, dL/dA is proportional to z and
        # dL/dz to A, so both stay at the saddle and the latent channel never
        # activates -- the first version of this experiment was uninformative for
        # exactly that reason.
        self.A = nn.Parameter(torch.randn(hidden) * 0.1)

    def forward(self, enc, dec, static, z):
        _, h = self.enc(enc)
        h0 = self.proj(torch.cat([h[0], h[1]], -1)) + z[:, None] * self.A
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        o, _ = self.dec(torch.cat([dec, s], -1), h0[None].contiguous())
        return self.head(self.drop(o)).squeeze(-1)


def _adapt_tensors(masked, full=None):
    """Encoder over h0..47; decoder over h48..215 with the usual weather channels."""
    dec_hours = np.arange(ADAPT_START, 216)
    enc_hours = np.arange(0, ADAPT_START)
    st = county_static(masked)
    w = weather_by_hour(masked)
    fips = _pivot(masked, "gust", enc_hours).index.to_numpy()
    enc = np.stack([_pivot(masked, c, enc_hours).to_numpy() for c in ENC_COLS], -1)
    dec = np.stack([_pivot(w, c, dec_hours).to_numpy() for c in DEC_COLS], -1)
    dt = (dec_hours - ADAPT_START).astype(float)
    osi0 = _pivot(masked, "osi", np.array([ADAPT_START - 1])).to_numpy()
    extras = np.stack([np.repeat(dt[None, :] / 100.0, len(fips), 0),
                       np.nan_to_num(osi0) * np.exp(-dt / 16.0) * Y_SCALE,
                       np.nan_to_num(osi0) * np.exp(-dt / 32.0) * Y_SCALE], -1)
    dec = np.concatenate([dec, extras], -1)
    static = st.loc[fips, STATIC_COLS].to_numpy()
    src = full if full is not None else masked
    Y = _pivot(src, "osi", dec_hours).to_numpy() * Y_SCALE
    return (fips, np.nan_to_num(enc), np.nan_to_num(dec), np.nan_to_num(static),
            Y, dec_hours)


def latent_adapt_forecaster(train_full, val_masked, seeds=(0, 1, 2),
                            lam=1.0, steps=25, lr=0.05):
    fips_t, Xe, Xd, Xs, Y, dh = _adapt_tensors(mask_after_freeze(train_full), train_full)
    stats = [(a.reshape(-1, a.shape[-1]).mean(0), a.reshape(-1, a.shape[-1]).std(0) + 1e-6)
             for a in (Xe, Xd, Xs)]
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]
    fips_v, Ve, Vd, Vs, Yv_obs, _ = _adapt_tensors(val_masked)
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]

    w_full = np.zeros(len(dh))
    w_full[dh >= PRED_HOURS[0]] = scoring_weights()          # score only h73+
    obs_mask = (dh >= ADAPT_START) & (dh <= FREEZE_H)        # hours available to fit z

    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(w_full, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    obs_t = torch.tensor(np.nan_to_num(Yv_obs), dtype=torch.float32)
    obs_w = torch.tensor(obs_mask.astype(float), dtype=torch.float32)

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        nv = max(12, len(idx) // 8)
        va_i, tr_i = idx[:nv], idx[nv:]
        model = LatentBiEnc(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        Z = nn.Parameter(torch.randn(len(Xe)) * 0.01)
        opt = torch.optim.Adam(list(model.parameters()) + [Z], lr=2e-3, weight_decay=1e-4)

        def wloss(o, i):
            raw = (((torch.clamp(o, min=0) ** 2) - t[3][i]) ** 2 * tw * msk[i]).sum() / (tw * msk[i]).sum()
            root = ((o - sq[i]) ** 2 * tw * msk[i]).sum() / (tw * msk[i]).sum()
            return 0.5 * raw + 0.5 * root

        best, state, bad = np.inf, None, 0
        for _ in range(250):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                o = model(t[0][b], t[1][b], t[2][b], Z[b])
                (wloss(o, b) + lam * (Z[b] ** 2).mean()).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = float(wloss(model(t[0][va_i], t[1][va_i], t[2][va_i], Z[va_i]), va_i))
            if v < best - 1e-6:
                best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 25:
                    break
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # ---- test-time system identification: fit z on OBSERVED hours 48-71 only
        zv = torch.zeros(len(fips_v), requires_grad=True)
        zopt = torch.optim.Adam([zv], lr=lr)
        for _ in range(steps):
            zopt.zero_grad()
            o = torch.clamp(model(tv[0], tv[1], tv[2], zv), min=0) ** 2
            loss = (((o - obs_t) ** 2) * obs_w).sum() / obs_w.sum() + lam * (zv ** 2).mean()
            loss.backward()
            zopt.step()
        with torch.no_grad():
            out = torch.clamp(model(tv[0], tv[1], tv[2], zv.detach()), min=0) ** 2
        preds.append(out.numpy()[:, dh >= PRED_HOURS[0]])
    return _dl_out(fips_v, np.mean(preds, 0))


def latent_adapt_z0_forecaster(train_full, val_masked):
    """Control: identical model, z forced to 0 at inference (no adaptation)."""
    return latent_adapt_forecaster(train_full, val_masked, steps=0)


# ====================================================== P4 storm-phase canonicalization
def _wave2_peak(masked):
    g = _pivot(masked, "gust", np.arange(216)).fillna(0.0)
    seg = g.loc[:, 108:180].to_numpy()
    return g.index.to_numpy(), 108 + seg.argmax(1)


def canonical_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    """Align each county's wave-2 gust peak to the median peak hour with a
    piecewise-linear warp anchored at h72 and h215 (weather-only mapping), train on
    warped trajectories, inverse-warp predictions before scoring."""
    def warp_maps(masked):
        fips, tau = _wave2_peak(masked)
        return fips, tau

    ftr, tau_tr = warp_maps(mask_after_freeze(train_full))
    fva, tau_va = warp_maps(val_masked)
    anchor = int(np.median(np.r_[tau_tr]))

    def to_canonical(A, tau):
        """Resample rows of A (over hours 72..215) onto the canonical axis."""
        H = PRED_HOURS
        out = np.empty_like(A)
        for i in range(len(A)):
            src = np.interp(H, [PRED_HOURS[0], anchor, 215],
                            [PRED_HOURS[0], tau[i], 215])
            out[i] = np.interp(src, H, A[i])
        return out

    def from_canonical(A, tau):
        H = PRED_HOURS
        out = np.empty_like(A)
        for i in range(len(A)):
            src = np.interp(H, [PRED_HOURS[0], tau[i], 215],
                            [PRED_HOURS[0], anchor, 215])
            out[i] = np.interp(src, H, A[i])
        return out

    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    order_tr = {f: i for i, f in enumerate(np.sort(train_full.fipsCode.unique()))}
    tau_tr_o = np.array([tau_tr[list(ftr).index(f)] for f in sorted(order_tr)])
    tau_va_o = np.array([tau_va[list(fva).index(f)] for f in vf])
    # warp decoder channels and targets into canonical time
    Xd_w = np.stack([to_canonical(Xd[:, :, k], tau_tr_o) for k in range(Xd.shape[-1])], -1)
    Y_w = to_canonical(np.nan_to_num(Y), tau_tr_o)
    Vd_w = np.stack([to_canonical(Vd[:, :, k], tau_va_o) for k in range(Vd.shape[-1])], -1)

    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd_w, Vs)]
    preds = []
    for s in seeds:
        m = _fit(BiEncSeq2Seq, Xe, Xd_w, Xs, Y_w, s)
        with torch.no_grad():
            preds.append(m(*tv).numpy())
    P = np.mean(preds, 0)
    P = from_canonical(P, tau_va_o)                       # back to clock time
    return _dl_out(vf, P)
