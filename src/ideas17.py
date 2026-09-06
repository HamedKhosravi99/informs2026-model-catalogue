"""Sweep 19 (SOTA-series) - what the 2024-2026 literature adds that the ledger has not tried.

The ledger already covers Tweedie (I09, I13), hurdle (P1/H1/H2), MC-dropout
(I10), mixture-of-trajectories (M3), GNN (B3), TiDE (S42), N-BEATS (S33), TCN
(S32), TFT-lite (I12), KAN (D2/D5), TabNet/FT-Transformer (D3/D4), and the
DLinear/PatchTST/iTransformer trio (TS1-3). Every architecture outside the
bi-GRU family landed at 0.0095-0.0120 solo against ~0.0088-0.0090 for the GRU
members, so the realistic value of anything new is as an ensemble error basis
(law 12), and that is how each candidate here is judged.

What is genuinely untried, and why it is relevant to THIS problem (72 h context,
143 h horizon, 20 known-future weather channels, 42% exact zeros):

  SX1 XLinear-KF   Ma et al. 2026 (arXiv 2601.09237). Lightweight MLP for
                   forecasting WITH exogenous inputs: RevIN'd endogenous tokens +
                   a global token, a time-wise gate, then a variate-wise gate
                   that routes exogenous information into the global token.
                   Reports lowest MSE in 25/28 configurations vs TimeXer /
                   PatchTST / DLinear. -KF: the exogenous variates are embedded
                   over the FUTURE window, since that is where our signal is.
  SX2 TSMixer-Ext-KF  Chen et al. 2023 (TMLR). All-MLP time/feature mixing with
                   an explicit align stage for future-known covariates and
                   conditional feature mixing for statics -- the one MLP design
                   that treats known-future inputs as first-class.
  SX3 CondFlow-KF  Conditional normalizing flow (affine-coupling, RealNVP-style)
                   over the 143-dim sqrt-target, conditioned on an encoder of
                   history + future weather + statics. Trained by maximum
                   likelihood; the point forecast is the sample mean. Included
                   because it was asked for and because it is the cleanest test
                   of the theory: RMSE is minimised by the conditional mean,
                   which an MSE-trained network estimates directly, whereas a
                   generative model estimates it through sampling. Expect the
                   flow to lose on RMSE unless likelihood training regularises
                   better than MSE does on this small sample.
  SX4 TimeXer-KF   Wang et al., NeurIPS 2024. Defined in this module once the
                   architecture is confirmed against the paper (see below).

All four consume the same tensors as ideas16 (winsorised at +-5 sigma, outputs
bounded by OSI's definitional range) and SX1/SX2/SX4 reuse `_fit_arch`, so the
architecture is again the only variable.
"""
import numpy as np
import torch
import torch.nn as nn

from .data import PRED_HOURS
from .ideas import _dl_out
from .ideas5 import _prep_ext
from .ideas16 import CLAMP, SQRT_MAX, _clip_inputs, _run_arch

IDEAS17 = {}
H = len(PRED_HOURS)          # 143
L = 72                       # context hours


# ------------------------------------------------------------------ SX1 XLinear
class _RevIN(nn.Module):
    """Instance normalisation over time with learnable affine (Kim et al. 2022)."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):                       # x: (B, L)
        mu = x.mean(1, keepdim=True)
        sd = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps)
        return ((x - mu) / sd) * self.w + self.b


class XLinearKF(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, d=32, ff=64, dropout=0.1):
        super().__init__()
        self.revin = _RevIN()
        self.emb = nn.Linear(n_enc, d)                 # per-step endogenous embed
        self.glob = nn.Parameter(torch.zeros(1, 1, d))
        # time-wise gate over the L+1 tokens (endogenous + global)
        self.tg1, self.tg2 = nn.Linear(L + 1, ff), nn.Linear(ff, L + 1)
        # exogenous variate tokens: each future variate embedded over the horizon
        self.exo = nn.Linear(H, d)
        # variate-wise gate over the n_dec + 1 tokens (exogenous + global)
        self.vg1, self.vg2 = nn.Linear(n_dec + 1, ff), nn.Linear(ff, n_dec + 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear((L + 1) * d, H)
        self.stat = nn.Linear(n_static, H)

    def forward(self, xe, xd, xs):
        xe, xd, xs = _clip_inputs(xe, xd, xs)
        xe = torch.cat([self.revin(xe[:, :, 0]).unsqueeze(-1), xe[:, :, 1:]], -1)
        tok = torch.cat([self.emb(xe), self.glob.expand(xe.size(0), -1, -1)], 1)  # (B, L+1, d)
        g = torch.sigmoid(self.tg2(torch.relu(self.tg1(tok.transpose(1, 2))))).transpose(1, 2)
        tok = tok * g                                                             # TGM
        endo, glob = tok[:, :L], tok[:, L:]                                       # (B,L,d),(B,1,d)
        ex = self.exo(xd.transpose(1, 2))                                         # (B, n_dec, d)
        vt = torch.cat([ex, glob], 1)                                             # (B, n_dec+1, d)
        g = torch.sigmoid(self.vg2(torch.relu(self.vg1(vt.transpose(1, 2))))).transpose(1, 2)
        vt = vt * g                                                               # VGM
        glob = vt[:, -1:]
        z = torch.cat([endo, glob], 1).flatten(1)
        return self.head(self.drop(z)) + self.stat(xs)


# -------------------------------------------------------------- SX2 TSMixer-Ext
class _MixerLayer(nn.Module):
    def __init__(self, T, d, ds, ff, dropout):
        super().__init__()
        self.ln_t, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d)
        self.time = nn.Linear(T, T)
        self.f1, self.f2 = nn.Linear(d + ds, ff), nn.Linear(ff, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, s):                    # x: (B,T,d)  s: (B,T,ds)
        y = self.ln_t(x).transpose(1, 2)
        x = x + self.drop(torch.relu(self.time(y))).transpose(1, 2)
        y = torch.cat([self.ln_f(x), s], -1)   # conditional feature mixing
        return x + self.drop(self.f2(self.drop(torch.relu(self.f1(y)))))


class TSMixerExtKF(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, d=48, ds=16, ff=96, layers=3, dropout=0.1):
        super().__init__()
        self.align = nn.Linear(L, H)                    # temporal projection L -> T
        self.proj = nn.Linear(n_enc + n_dec, d)
        self.stat_tok = nn.Linear(n_static, ds)
        self.layers = nn.ModuleList([_MixerLayer(H, d, ds, ff, dropout) for _ in range(layers)])
        self.out = nn.Linear(d, 1)
        self.stat = nn.Linear(n_static, H)

    def forward(self, xe, xd, xs):
        xe, xd, xs = _clip_inputs(xe, xd, xs)
        xp = self.align(xe.transpose(1, 2)).transpose(1, 2)        # (B, H, n_enc)
        x = self.proj(torch.cat([xp, xd], -1))                     # (B, H, d)
        s = self.stat_tok(xs).unsqueeze(1).expand(-1, H, -1)       # (B, H, ds)
        for lyr in self.layers:
            x = lyr(x, s)
        return self.out(x).squeeze(-1) + self.stat(xs)


# ------------------------------------------------------------- SX3 CondFlow-KF
class _Coupling(nn.Module):
    """Affine coupling: transform the unmasked half conditioned on the masked half + context."""

    def __init__(self, dim, cdim, mask, hidden=128):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(dim + cdim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * dim))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, c):                    # forward: data -> latent
        zm = z * self.mask
        s, t = self.net(torch.cat([zm, c], -1)).chunk(2, -1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        out = zm + (1 - self.mask) * (z * torch.exp(s) + t)
        return out, s.sum(-1)

    def inverse(self, z, c):
        zm = z * self.mask
        s, t = self.net(torch.cat([zm, c], -1)).chunk(2, -1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return zm + (1 - self.mask) * ((z - t) * torch.exp(-s))


class CondFlowKF(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, d=64, cdim=64, n_coupling=6):
        super().__init__()
        self.gru = nn.GRU(n_enc, d, batch_first=True)
        self.fut = nn.Sequential(nn.Linear(H * n_dec, 256), nn.ReLU(), nn.Linear(256, d))
        self.sta = nn.Linear(n_static, d)
        self.ctx = nn.Sequential(nn.Linear(3 * d, cdim), nn.ReLU())
        masks = []
        for k in range(n_coupling):
            m = torch.zeros(H); m[k % 2::2] = 1.0
            masks.append(m)
        self.flows = nn.ModuleList([_Coupling(H, cdim, m) for m in masks])

    def context(self, xe, xd, xs):
        xe, xd, xs = _clip_inputs(xe, xd, xs)
        _, h = self.gru(xe)
        return self.ctx(torch.cat([h[-1], self.fut(xd.flatten(1)), self.sta(xs)], -1))

    def log_prob(self, y, c):                   # y on the sqrt scale, (B, H)
        z, ld = y, torch.zeros(y.size(0), device=y.device)
        for f in self.flows:
            z, l = f(z, c); ld = ld + l
        return -0.5 * (z ** 2).sum(-1) - 0.5 * H * np.log(2 * np.pi) + ld

    def sample_mean(self, c, n=64):
        B = c.size(0)
        z = torch.randn(n, B, H, device=c.device)
        c2 = c.unsqueeze(0).expand(n, -1, -1).reshape(n * B, -1)
        z = z.reshape(n * B, H)
        for f in reversed(self.flows):
            z = f.inverse(z, c2)
        y = torch.clamp(z, min=0, max=SQRT_MAX) ** 2
        return y.reshape(n, B, H).mean(0)


def _fit_flow(Xe, Xd, Xs, Y, seed, max_epochs=200, patience=25, noise=0.05):
    """Maximum likelihood on the sqrt target with Gaussian dequantisation noise
    (42% of targets are exactly zero -- a point mass a continuous density cannot
    represent). Early stopping on the RMSE of the sample-mean point forecast, so
    model selection is on the scored quantity, not on likelihood."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = CondFlowKF(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    ys = torch.sqrt(t[3])
    best, state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            c = model.context(t[0][b], t[1][b], t[2][b])
            yb = ys[b] + noise * torch.randn_like(ys[b])
            loss = -model.log_prob(yb, c).mean() / H
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            c = model.context(t[0][va_i], t[1][va_i], t[2][va_i])
            v = nn.functional.mse_loss(model.sample_mean(c, n=16), t[3][va_i]).item()
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    return model


def sx3_condflow(tr, va, seeds=(0, 1, 2)):
    torch.set_num_threads(4)
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep_ext(tr, va)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        m = _fit_flow(Xe, Xd, Xs, Y, s)
        with torch.no_grad():
            torch.manual_seed(1000 + s)
            preds.append(m.sample_mean(m.context(*tv), n=128).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def sx1_xlinear(tr, va):
    return _run_arch(XLinearKF, tr, va)


def sx2_tsmixer(tr, va):
    return _run_arch(TSMixerExtKF, tr, va)


IDEAS17.update({
    "SX1 XLinear-KF (Ma 2026; gated MLP with exogenous routing)": sx1_xlinear,
    "SX2 TSMixer-Ext-KF (Chen 2023; align + conditional feature mixing)": sx2_tsmixer,
    "SX3 CondFlow-KF (conditional affine-coupling flow, sample-mean forecast)": sx3_condflow,
})


# --------------------------------------------------------------- SX4 TimeXer-KF
class _TimeXerBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, endo, exo):               # endo: (B, N+1, d) patches+global; exo: (B, M, d)
        a, _ = self.self_attn(endo, endo, endo)
        endo = self.n1(endo + self.drop(a))
        g = endo[:, -1:]                        # the global token is the only cross-attention query
        c, _ = self.cross_attn(g, exo, exo)
        endo = torch.cat([endo[:, :-1], self.n2(g + self.drop(c))], 1)
        return self.n3(endo + self.drop(self.ffn(endo)))


class TimeXerKF(nn.Module):
    """TimeXer (Wang et al., NeurIPS 2024) + known-future exogenous tokens.

    Faithful core: endogenous history is patched and linearly embedded, a
    learnable global token is appended, each block runs self-attention over
    [patches + global], then cross-attention with the global token as the sole
    query and variate-level exogenous tokens as keys/values, then a
    feed-forward layer; the head is a linear map from the flattened
    [patches, global] representation to the horizon.

    Adaptation (-KF): the paper embeds each exogenous variable over the LOOKBACK
    window only and never touches the forecast horizon. Here the variate
    embedding is taken over the FUTURE window instead, because our exogenous
    variables are known-future weather and that is where the signal is. Nothing
    else is changed. d_model is 64 (the paper searches 128-512 on datasets 100x
    larger; 64 matches PatchTST-KF so the two are comparable).
    """

    def __init__(self, n_enc, n_dec, n_static, d=64, heads=4, layers=2, ff=256,
                 patch=8, dropout=0.15):
        super().__init__()
        self.patch, n_p = patch, L // patch
        self.emb_p = nn.Linear(patch * n_enc, d)
        self.glob = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n_p + 1, d) * 0.02)
        self.emb_x = nn.Linear(H, d)                   # variate token over the future window
        self.blocks = nn.ModuleList([_TimeXerBlock(d, heads, ff, dropout) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear((n_p + 1) * d, H)
        self.stat = nn.Linear(n_static, H)

    def forward(self, xe, xd, xs):
        xe, xd, xs = _clip_inputs(xe, xd, xs)
        B = xe.size(0)
        p = xe.reshape(B, L // self.patch, self.patch * xe.size(-1))      # non-overlapping patches
        endo = torch.cat([self.emb_p(p), self.glob.expand(B, -1, -1)], 1) + self.pos
        exo = self.emb_x(xd.transpose(1, 2))                              # (B, n_dec, d)
        for blk in self.blocks:
            endo = blk(endo, exo)
        return self.head(self.drop(endo.flatten(1))) + self.stat(xs)


def sx4_timexer(tr, va):
    return _run_arch(TimeXerKF, tr, va)


IDEAS17["SX4 TimeXer-KF (Wang NeurIPS 2024; global-token cross-attention to future variates)"] = sx4_timexer
