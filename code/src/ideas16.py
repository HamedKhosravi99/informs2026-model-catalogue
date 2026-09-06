"""Sweep 18 (TS-series) - the modern long-horizon forecasting architectures.

The ledger's architecture sweep (S32 TCN, S33 N-BEATS basis, S42 TiDE, I12
TFT-lite, S37 multi-task) predates the patch/linear era, and REPORT dismissed
foundation models in one line. This tests the family that actually beat the
architectures we did try, on our own protocol.

WHY NOT A PRETRAINED FOUNDATION MODEL (Chronos / TimesFM / Moirai / Time-LLM):

1. Shape mismatch, and it is severe. Our task is 72 h of context -> 143 h of
   horizon with **20 channels of fully known future weather**. Horizon is 2x
   context, and the known-future covariates carry the dominant signal - that is
   why a bi-encoder works here at all. The first-generation models (Chronos,
   Lag-Llama) have no covariate channel; Chronos-2 (Oct 2025) and Moirai-2 do.
   MEASURED (tools/chronos2_zeroshot.py, W19): Chronos-2 zero-shot with our 23
   weather covariates scores 0.0214 on the 239 training counties, worse than
   all-zeros (0.0163), and the covariates make it slightly worse than the
   univariate run (0.0201). It predicts a non-zero level almost everywhere
   (3.2% exact zeros against 42% in the truth) and misses the spikes. The
   argument below is therefore no longer the reason it is excluded; the number
   is.
2. Distribution mismatch: 44% exact zeros and p99/median ~ 1000. These models
   instance-normalise a series and predict a smooth continuation; ours is a
   flat-zero prefix followed by a single storm spike.
3. Compliance: a checkpoint pretrained on large public corpora is external data
   of a far more aggressive kind than a land-cover table, and it breaks
   "self-contained code that runs end-to-end on the provided data files"
   (REPORT 5aq). Also blocked by the pinned torch 1.12 (all require >= 2.1).
4. For Time-LLM specifically there is direct published evidence against it:
   Tan et al., "Are Language Models Actually Useful for Time Series
   Forecasting?" (NeurIPS 2024) ablate the LLM out of Time-LLM, OneFitsAll and
   LLaTA - replacing the backbone with plain attention, or deleting it - and
   performance does not degrade, in most cases improves, at a fraction of the
   cost. The reprogramming layer, not the language model, is doing the work.

So the pretrained route is deprioritised on evidence rather than on effort. What
IS worth testing is the *architecture* half of that literature, which is cheap,
runs on torch 1.12, and carries no compliance risk at all:

  TS1 DLinear-KF  - Zeng et al., AAAI 2023 ("Are Transformers Effective for Time
                   Series Forecasting?"). Trend/seasonal decomposition + linear
                   maps. Beat Informer/Autoformer/FEDformer on every benchmark.
  TS2 PatchTST-KF - Nie et al., ICLR 2023. Patching + channel independence.
  TS3 iTransformer-KF - Liu et al., ICLR 2024. Attention over VARIATES, not time.

All three are adapted to consume the known-future covariates, because testing
them without that channel would be a strawman - the same mistake that made the
first KAN attempt worthless (REPORT 5al.1). The adaptation is named "-KF" and
described per class. Training loop, optimiser, epochs, patience, early-stop
scale and seed set are copied from `_fit_transformed` so the architecture is the
only variable.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import PRED_HOURS
from .dl import Y_SCALE
from .ideas import _dl_out
from .ideas5 import _prep_ext

IDEAS16 = {}

# Winsorising bound for the standardised inputs. `_prep_ext` z-scores against
# the TRAINING pool, so a held-out county can land far outside it: in fold 0,
# county 18111's static vector reaches 140 sigma while every other validation
# county maxes at 6.4. All three architectures here map statics through an
# unbounded Linear (an addition of ours -- none of the three papers takes static
# covariates), so one extreme value multiplies straight through into a 17x
# trajectory blow-up that carried 99.9% of the fold's squared error. The GRU
# members are immune because their recurrent nonlinearity saturates.
# Winsorising is the standard remedy and is applied identically to every input.
CLAMP = 5.0

# OSI is a SHARE, so it is bounded by construction. sqrt(Y) with Y = osi *
# Y_SCALE therefore cannot exceed sqrt(Y_SCALE). Enforcing the definitional
# bound costs nothing and is what `verify_submission.py` already checks.
SQRT_MAX = float(np.sqrt(Y_SCALE))


def _clip_inputs(xe, xd, xs):
    return xe.clamp(-CLAMP, CLAMP), xd.clamp(-CLAMP, CLAMP), xs.clamp(-CLAMP, CLAMP)


# ----------------------------------------------------------------- TS1 DLinear
class DLinearKF(nn.Module):
    """DLinear (Zeng et al. 2023) + a linear known-future covariate path.

    Faithful core: moving-average decomposition into trend + seasonal, one
    linear map per component from context to horizon, summed. No nonlinearity
    in the series path - that is the paper's whole point.

    Adaptation (-KF): the original has no exogenous input, so a single linear
    layer maps the known-future covariates at each horizon step to a per-step
    correction, plus a static term. Both are linear, so the model stays a linear
    predictor end to end.
    """

    def __init__(self, n_enc, n_dec, n_static, L=72, H=len(PRED_HOURS), kernel=25):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(L, H)
        self.seasonal = nn.Linear(L, H)
        self.fut = nn.Linear(n_dec, 1)          # per-horizon-step covariate term
        self.stat = nn.Linear(n_static, H)
        for m in (self.trend, self.seasonal):   # paper's init: uniform averaging
            nn.init.constant_(m.weight, 1.0 / L)
            nn.init.zeros_(m.bias)

    def _decompose(self, x):                    # x: (B, L)
        pad = self.kernel // 2
        xp = torch.cat([x[:, :1].repeat(1, pad), x, x[:, -1:].repeat(1, pad)], 1)
        trend = torch.nn.functional.avg_pool1d(xp.unsqueeze(1), self.kernel, 1).squeeze(1)
        return trend, x - trend

    def forward(self, xe, xd, xs):
        xe, xd, xs = _clip_inputs(xe, xd, xs)
        y = xe[:, :, 0]                         # the outage channel
        t, s = self._decompose(y)
        out = self.trend(t) + self.seasonal(s) + self.stat(xs)
        return out + self.fut(xd).squeeze(-1)


# ---------------------------------------------------------------- TS2 PatchTST
class PatchTSTKF(nn.Module):
    """PatchTST (Nie et al. 2023) + known-future patches.

    Faithful core: (i) instance normalisation, (ii) split each channel's series
    into patches, (iii) CHANNEL INDEPENDENCE - one shared backbone applied to
    every channel separately, which is the paper's main claim, (iv) flatten +
    linear head.

    Adaptation (-KF): the known-future covariates are patched with the same
    patch length and embedded by a second shared projection, then concatenated
    to the token sequence. Channel independence is preserved: the transformer
    sees (B * n_channels, n_patches, d) exactly as in the paper.
    """

    def __init__(self, n_enc, n_dec, n_static, L=72, H=len(PRED_HOURS),
                 patch=12, stride=12, d=64, heads=4, layers=2, dropout=0.15):
        super().__init__()
        self.patch, self.stride, self.n_enc, self.n_dec = patch, stride, n_enc, n_dec
        n_p_e = (L - patch) // stride + 1
        n_p_d = (H - patch) // stride + 1
        self.n_p_e, self.n_p_d = n_p_e, n_p_d
        self.emb_e = nn.Linear(patch, d)
        self.emb_d = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.randn(1, n_p_e + n_p_d, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.drop = nn.Dropout(dropout)
        # channel-independent head, then mix channels once at the end
        self.head = nn.Linear((n_p_e + n_p_d) * d, H)
        self.mix = nn.Linear(n_enc + n_dec, 1)
        self.stat = nn.Linear(n_static, H)

    @staticmethod
    def _patchify(x, patch, stride):
        """x: (B, T, C) -> (B*C, n_patches, patch), channel-independent."""
        B, T, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, T)
        return x.unfold(-1, patch, stride)

    def forward(self, xe, xd, xs):
        xe, xd, xs = _clip_inputs(xe, xd, xs)
        B = xe.shape[0]
        # instance norm on the observed series (RevIN, simplified to mean/std)
        mu = xe.mean(1, keepdim=True)
        sd = xe.std(1, keepdim=True) + 1e-6
        xe = (xe - mu) / sd

        pe = self.emb_e(self._patchify(xe, self.patch, self.stride))   # (B*Ce,Pe,d)
        pd_ = self.emb_d(self._patchify(xd, self.patch, self.stride))  # (B*Cd,Pd,d)
        pe = pe.reshape(B, self.n_enc, self.n_p_e, -1)
        pd_ = pd_.reshape(B, self.n_dec, self.n_p_d, -1)
        # every channel gets the full future context (broadcast), then is
        # processed independently by the shared backbone
        fut = pd_.mean(1, keepdim=True)
        tok_e = torch.cat([pe, fut.expand(-1, self.n_enc, -1, -1)], 2)
        tok_d = torch.cat([pe.mean(1, keepdim=True).expand(-1, self.n_dec, -1, -1),
                           pd_], 2)
        tok = torch.cat([tok_e, tok_d], 1)                    # (B,C,P,d)
        C, P, d = tok.shape[1], tok.shape[2], tok.shape[3]
        z = self.enc(tok.reshape(B * C, P, d) + self.pos)
        z = self.head(self.drop(z).reshape(B * C, -1)).reshape(B, C, -1)
        return self.mix(z.permute(0, 2, 1)).squeeze(-1) + self.stat(xs)


# ------------------------------------------------------------ TS3 iTransformer
class ITransformerKF(nn.Module):
    """iTransformer (Liu et al. 2024): invert the attention axis.

    Each VARIATE becomes one token (its whole series embedded by an MLP), and
    attention runs across variates rather than across time. Adaptation (-KF):
    known-future covariates are embedded as additional variate tokens over the
    horizon window, so past and future channels attend to each other directly.
    """

    def __init__(self, n_enc, n_dec, n_static, L=72, H=len(PRED_HOURS),
                 d=96, heads=4, layers=2, dropout=0.15):
        super().__init__()
        self.emb_e = nn.Linear(L, d)
        self.emb_d = nn.Linear(H, d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d, H)
        self.pool = nn.Linear(n_enc + n_dec, 1)
        self.stat = nn.Linear(n_static, H)

    def forward(self, xe, xd, xs):
        xe, xd, xs = _clip_inputs(xe, xd, xs)
        te = self.emb_e(xe.permute(0, 2, 1))       # (B, n_enc, d)
        td = self.emb_d(xd.permute(0, 2, 1))       # (B, n_dec, d)
        z = self.enc(torch.cat([te, td], 1))
        y = self.proj(self.drop(z))                # (B, C, H)
        return self.pool(y.permute(0, 2, 1)).squeeze(-1) + self.stat(xs)


# --------------------------------------------------------------------- harness
def _wmse(a, b, w):
    """MSE with per-target-hour weights. `w` is normalised to mean 1, so this is
    on the same scale as the unweighted loss and no learning rate change is
    implied."""
    return (((a - b) ** 2) * w).mean() if w is not None else nn.functional.mse_loss(a, b)


def _fit_arch(cls, Xe, Xd, Xs, Y, seed, max_epochs=300, patience=30, w=None):
    """Copied from ideas5._fit_transformed so the architecture is the only
    variable: same optimiser, lr, weight decay, batching, dual raw/sqrt loss,
    and early stopping on the RAW scale.

    `w` optionally supplies the metric's per-hour weights. The organisers score
    each submission column over its own set of target hours, so a late hour is
    scored in all four columns while hours 73-77 appear only in t01h; a flat
    loss therefore under-serves exactly the hours the metric weights most. The
    ideas4 members already train this way -- this parameter closes the same gap
    for the ideas16 architectures. It is applied to both loss branches and to
    the early-stopping criterion, so the whole fit sees one objective."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = cls(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tf = torch.sqrt(t[3])
    tw = None if w is None else torch.tensor(w, dtype=torch.float32)
    best, state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            out = model(t[0][b], t[1][b], t[2][b])
            loss = 0.5 * _wmse(out, tf[b], tw) + 0.5 * _wmse(
                torch.clamp(out, min=0) ** 2, t[3][b], tw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.clamp(model(t[0][va_i], t[1][va_i], t[2][va_i]),
                            min=0, max=SQRT_MAX) ** 2
            v = _wmse(p, t[3][va_i], tw).item()
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def _run_arch(cls, train_full, val_masked, seeds=(0, 1, 2), w=None):
    torch.set_num_threads(4)
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep_ext(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        m = _fit_arch(cls, Xe, Xd, Xs, Y, s, w=w)
        with torch.no_grad():
            preds.append((torch.clamp(m(*tv), min=0, max=SQRT_MAX) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def t1_dlinear(tr, va):
    return _run_arch(DLinearKF, tr, va)


def t2_patchtst(tr, va):
    return _run_arch(PatchTSTKF, tr, va)


def t3_itransformer(tr, va):
    return _run_arch(ITransformerKF, tr, va)


def t2c_patchtst_weighted(tr, va):
    """TS2c. TS2b with the metric's hour weights (sum_j 1/n_j)."""
    from .ideas4 import scoring_weights
    return _run_arch(PatchTSTKF, tr, va, w=scoring_weights())


def t2d_patchtst_exact(tr, va):
    """TS2d. TS2b with the exact objective Jacobian, sum_j 1/(n_j * RMSE_j).

    The scored objective is a mean of per-horizon RMSEs, not a weighted MSE, so
    each horizon carries an extra 1/(2*RMSE_j) factor. See ideas8.A5, where the
    same correction applied to S62 moved it 0.00897 -> 0.00888."""
    from .ideas8 import scoring_weights_exact
    return _run_arch(PatchTSTKF, tr, va, w=scoring_weights_exact())


IDEAS16 = {
    "TS1b DLinear-KF winsorised (Zeng AAAI 2023 + known-future)": t1_dlinear,
    "TS2b PatchTST-KF winsorised (Nie ICLR 2023 + known-future patches)": t2_patchtst,
    "TS3b iTransformer-KF winsorised (Liu ICLR 2024 + known-future variates)": t3_itransformer,
    "TS2c PatchTST-KF + metric hour weights": t2c_patchtst_weighted,
    "TS2d PatchTST-KF + exact metric Jacobian": t2d_patchtst_exact,
}


# ------------------------------------------------- metric-alignment probes (W18)
# Six of the nine freeze members train on a flat MSE while the metric weights a
# target hour by how many submission columns score it (4.7x spread). ts2c closed
# that gap for PatchTST and the ensemble improved. These probe the same change on
# the remaining neural members at 3 seeds -- cheap enough to test before paying
# for the 25-seed production variants they would replace.
def i11w_weighted(tr, va):
    """I11 storm augmentation with the metric's hour weights (vs x3 = i11 @25)."""
    from .ideas import augmented_gru_forecaster
    from .ideas4 import scoring_weights
    return augmented_gru_forecaster(tr, va, w=scoring_weights())


def s35w_weighted(tr, va):
    """S35 mixup with the metric's hour weights (vs x4 = s35 @25)."""
    from .ideas3 import mixup_forecaster
    from .ideas4 import scoring_weights
    return mixup_forecaster(tr, va, w=scoring_weights())


IDEAS16["I11w i11 storm augmentation + metric hour weights"] = i11w_weighted
IDEAS16["S35w s35 mixup + metric hour weights"] = s35w_weighted


def x11_i11w_seed25(tr, va):
    """X11. i11w at 25 seeds. i11w at 3 seeds already beats x3 (i11 @ 25 seeds,
    flat loss) as a freeze member on all four horizons; law 13b says seed
    scaling pays in proportion to un-averaged variance, and a 3-seed member has
    the most of it. The single highest-expected-value shippable run available."""
    from .ideas import augmented_gru_forecaster
    from .ideas4 import scoring_weights
    return augmented_gru_forecaster(tr, va, seeds=tuple(range(25)), w=scoring_weights())


IDEAS16["X11 i11w metric-weighted, 25 seeds"] = x11_i11w_seed25


def s39w_weighted(tr, va):
    """S39 cross-county attention with the metric's hour weights -- the last
    neural freeze member never tested with them (probe results so far: i11w yes,
    x9w no, s35w no)."""
    from .ideas3 import cross_county_fixed_forecaster
    from .ideas4 import scoring_weights
    return cross_county_fixed_forecaster(tr, va, w=scoring_weights())


IDEAS16["S39w s39 cross-county + metric hour weights"] = s39w_weighted
