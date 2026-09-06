"""Sweep 17 (D-series) - MLP and KAN on the CURRENT tabular feature set.

The project's only MLP result is rung 5 (0.01050), measured on the original
~50-column feature set before statics, sub-county weather and the physics
interactions existed. The shipped tabular member f2 now scores 0.00930 on a
much richer design matrix, so "neural nets do nothing here" has never actually
been tested against the modern features.

KAN (Kolmogorov-Arnold Networks, Liu et al. arXiv:2404.19756) is genuinely
absent from the ledger. The pitch is a good fit on paper: KANs put learnable
univariate functions on the EDGES and sum them, so a KAN layer is a learned
generalised additive model with interactions built by composition - the exact
structure §5ad says this problem has (decay bases dominate additively; canopy
matters only in interaction). `pykan` requires torch>=2.0, so this implements
the Chebyshev-basis variant (SS Sidharth, arXiv:2405.07200), which is the
same idea with a cheaper, numerically stable basis and runs on torch 1.12.

Both consume f2's exact rows and the sqrt target, so the only variable versus
the shipped member is the function class.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import mask_after_freeze
from .features import feature_cols
from .ideas import _out
from .models import SEED

IDEAS15 = {}


def _frames(train_full, val_masked):
    from .ideas6 import _rows
    from .ideas10 import add_indep
    DROP = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
            "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
    tr, y = _rows(train_full, statics=True, sub=True)
    va = _rows(val_masked, statics=True, sub=True, val=True)
    tr = add_indep(tr, mask_after_freeze(train_full))
    va = add_indep(va, val_masked)
    tr = tr.drop(columns=[c for c in DROP if c in tr.columns])
    va = va.drop(columns=[c for c in DROP if c in va.columns])
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    return tr, y, va, cols, keep


def _standardise(Xtr, Xva):
    mu = np.nanmean(Xtr, 0)
    sd = np.nanstd(Xtr, 0) + 1e-6
    f = lambda A: np.nan_to_num((A - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return f(Xtr), f(Xva)


class ChebyKANLayer(nn.Module):
    """y_j = sum_i sum_k c[i,j,k] * T_k(tanh(x_i)) - learnable univariate
    functions on edges, the KAN construction with a Chebyshev basis."""

    def __init__(self, n_in, n_out, degree=4):
        super().__init__()
        self.n_in, self.n_out, self.degree = n_in, n_out, degree
        self.coef = nn.Parameter(torch.empty(n_in, n_out, degree + 1))
        nn.init.normal_(self.coef, std=1.0 / (n_in * (degree + 1)) ** 0.5)

    def forward(self, x):
        x = torch.tanh(x)                                  # into [-1, 1]
        T = [torch.ones_like(x), x]
        for k in range(2, self.degree + 1):
            T.append(2 * x * T[-1] - T[-2])                # Chebyshev recurrence
        T = torch.stack(T, -1)                             # (B, n_in, degree+1)
        return torch.einsum("bik,iok->bo", T, self.coef)


class KAN(nn.Module):
    def __init__(self, n_in, hidden=32, degree=4, dropout=0.1):
        super().__init__()
        self.l1 = ChebyKANLayer(n_in, hidden, degree)
        self.n1 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.l2 = ChebyKANLayer(hidden, 1, degree)

    def forward(self, x):
        return self.l2(self.drop(self.n1(self.l1(x)))).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, n_in, hidden=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_net(kind, Xtr, ytr, Xva, seeds=(0, 1, 2), epochs=60, bs=4096, lr=2e-3):
    preds = []
    n_val = max(1, len(Xtr) // 10)
    for s in seeds:
        torch.manual_seed(s)
        rng = np.random.RandomState(s)
        perm = rng.permutation(len(Xtr))
        vi, ti = perm[:n_val], perm[n_val:]
        model = {"kan": KAN, "kan2": KAN2, "mlp": MLP, "tabnet": TabNet,
                 "ftt": FTTransformer}[kind](Xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        Xt = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.float32)
        best, state, bad = np.inf, None, 0
        for _ in range(epochs):
            model.train()
            for b in np.array_split(rng.permutation(ti), max(1, len(ti) // bs)):
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(Xt[b]), yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                v = float(nn.functional.mse_loss(model(Xt[vi]), yt[vi]))
            if v < best - 1e-9:
                best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 10:
                    break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds.append(model(torch.tensor(Xva, dtype=torch.float32)).numpy())
    return np.mean(preds, 0)


def _quantile(Xtr, Xva):
    """Rank -> [-1,1]. Preserves the ORDERING of the heavy tail instead of
    saturating it, and puts every feature exactly on the KAN's spline grid."""
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution="uniform", n_quantiles=1000,
                             subsample=200000, random_state=SEED)
    A = qt.fit_transform(np.nan_to_num(Xtr))
    B = qt.transform(np.nan_to_num(Xva))
    return (2 * A - 1).astype(np.float32), np.clip(2 * B - 1, -1, 1).astype(np.float32)


def _run(train_full, val_masked, kind):
    torch.set_num_threads(4)
    tr, y, va, cols, keep = _frames(train_full, val_masked)
    prep = _quantile if kind == "kan2" else _standardise
    Xtr, Xva = prep(tr.loc[keep, cols].to_numpy(float), va[cols].to_numpy(float))
    p = _fit_net(kind, Xtr, np.sqrt(y[keep]), Xva)
    return _out(va, np.clip(p, 0, None) ** 2)


# ---------------------------------------------------------------- proper KAN
# The first KAN attempt (D2, 0.01092) was a weak implementation, and the
# defects are specific:
#   1. tanh() squashing into [-1,1] SATURATES past ~2 sigma, destroying the
#      heavy tail this metric is decided on (p99/median ~ 1000).
#   2. No base-activation residual. The KAN paper's edge function is
#      phi(x) = w_b*silu(x) + w_s*spline(x); dropping the silu path removes
#      the skip connection that makes the spline a correction rather than the
#      whole function.
#   3. Chebyshev polynomials are GLOBAL basis functions - a coefficient change
#      moves the curve everywhere - whereas B-splines are LOCAL, which is the
#      property that lets KANs fit sharp local structure without ringing.
# D5 fixes all three: quantile transform to [-1,1] (rank-preserving, no
# saturation), Cox-de Boor B-spline basis on a real grid, and the base+spline
# residual form. Reference implementation: efficient-kan.
def _b_splines(x, grid, k):
    """Cox-de Boor recursion. x: (B, n_in); grid: (n_in, G + 2k + 1)."""
    x = x.unsqueeze(-1)
    b = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for p in range(1, k + 1):
        b = ((x - grid[:, :-(p + 1)]) / (grid[:, p:-1] - grid[:, :-(p + 1)]) * b[:, :, :-1]
             + (grid[:, p + 1:] - x) / (grid[:, p + 1:] - grid[:, 1:-p]) * b[:, :, 1:])
    return b


class KANLayer(nn.Module):
    def __init__(self, n_in, n_out, grid_size=8, k=3, rng=(-1.0, 1.0)):
        super().__init__()
        self.n_in, self.n_out, self.k, self.G = n_in, n_out, k, grid_size
        h = (rng[1] - rng[0]) / grid_size
        g = (torch.arange(-k, grid_size + k + 1) * h + rng[0]).expand(n_in, -1)
        self.register_buffer("grid", g.contiguous())
        self.base_w = nn.Parameter(torch.empty(n_out, n_in))
        self.spline_w = nn.Parameter(torch.empty(n_out, n_in, grid_size + k))
        nn.init.kaiming_uniform_(self.base_w, a=5 ** 0.5)
        nn.init.normal_(self.spline_w, std=0.1 / (n_in ** 0.5))

    def forward(self, x):
        base = nn.functional.linear(nn.functional.silu(x), self.base_w)   # residual path
        sp = _b_splines(x, self.grid, self.k).reshape(x.size(0), -1)  # non-contiguous
        return base + nn.functional.linear(sp, self.spline_w.reshape(self.n_out, -1))


class KAN2(nn.Module):
    """Proper KAN: B-spline edges + base-activation residual."""

    def __init__(self, n_in, hidden=24, grid_size=8, k=3, dropout=0.1):
        super().__init__()
        self.l1 = KANLayer(n_in, hidden, grid_size, k)
        self.n1 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.l2 = KANLayer(hidden, 1, grid_size, k)

    def forward(self, x):
        h = torch.tanh(self.n1(self.l1(x)))     # keep layer-2 input inside the grid
        return self.l2(self.drop(h)).squeeze(-1)


class Sparsemax(nn.Module):
    """TabNet's feature-selection activation: projects onto the simplex and
    yields exact zeros, so each decision step attends to a sparse subset."""

    def forward(self, x):
        d = x.shape[-1]
        z, _ = torch.sort(x, dim=-1, descending=True)
        cs = z.cumsum(-1)
        k = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
        support = (1 + k * z) > cs
        k_sup = support.sum(-1, keepdim=True).clamp(min=1)
        tau = (cs.gather(-1, k_sup - 1) - 1) / k_sup.to(x.dtype)
        return torch.clamp(x - tau, min=0)


class TabNet(nn.Module):
    """Sequential-attention tabular net (Arik & Pfister, AAAI 2021).

    n_steps decision steps; each applies a learned sparse mask over the raw
    features (sparsemax of a transform of the previous step's output), passes
    the masked input through a shared+step-specific block, and contributes
    additively to the output. Prior scales enforce that a feature reused across
    steps is down-weighted (gamma), which is the mechanism the paper credits
    for interpretable, instance-wise feature selection.
    """

    def __init__(self, n_in, n_d=32, n_steps=3, gamma=1.3, dropout=0.1):
        super().__init__()
        self.n_steps, self.gamma, self.n_in = n_steps, gamma, n_in
        self.bn = nn.BatchNorm1d(n_in)
        self.shared = nn.Sequential(nn.Linear(n_in, n_d * 2), nn.ReLU(),
                                    nn.Dropout(dropout))
        self.step_fc = nn.ModuleList(nn.Linear(n_d * 2, n_d) for _ in range(n_steps))
        self.att = nn.ModuleList(nn.Linear(n_d, n_in) for _ in range(n_steps))
        self.sparsemax = Sparsemax()
        self.head = nn.Linear(n_d, 1)

    def forward(self, x):
        x = self.bn(x)
        prior = torch.ones_like(x)
        a = torch.zeros(x.shape[0], self.step_fc[0].out_features, device=x.device)
        out = 0.0
        for i in range(self.n_steps):
            mask = self.sparsemax(self.att[i](a) * prior)
            prior = prior * (self.gamma - mask)
            d = torch.relu(self.step_fc[i](self.shared(x * mask)))
            out = out + d
            a = d
        return self.head(out).squeeze(-1)


class FTTransformer(nn.Module):
    """Feature-Tokenizer Transformer (Gorishniy et al., NeurIPS 2021): every
    scalar feature becomes a token via its own linear embedding, a [CLS] token
    is prepended, and standard self-attention runs over the FEATURE axis - so
    attention learns which feature pairs interact rather than which time steps."""

    def __init__(self, n_in, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_in, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=d * 2,
                                         dropout=dropout, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        t = x[:, :, None] * self.w[None] + self.b[None]      # (B, n_in, d)
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        return self.head(self.norm(self.tf(t)[:, 0])).squeeze(-1)


def d1_mlp(tr, va):  return _run(tr, va, "mlp")
def d2_kan(tr, va):     return _run(tr, va, "kan")
def d3_tabnet(tr, va):  return _run(tr, va, "tabnet")
def d4_ftt(tr, va):     return _run(tr, va, "ftt")
def d5_kan2(tr, va):    return _run(tr, va, "kan2")


# ================= X-series: apply the PROVEN levers to every member
# Gap found 2026-08-15: seed scaling to 25 (law 13, worth 0.00897 -> 0.00862 on
# S62) and snapshot ensembling have only ever been applied to ONE of the six
# sequence members. Every other member still runs at its original 3 seeds. If
# the lever generalises, improving all six compounds through the ensemble.
def _seed25(fn, tr, va):
    return fn(tr, va, seeds=tuple(range(25)))


def x_s39(tr, va):
    from .ideas3 import cross_county_fixed_forecaster
    return _seed25(cross_county_fixed_forecaster, tr, va)


def x_s40(tr, va):
    from .ideas4 import multi_origin_forecaster
    return _seed25(multi_origin_forecaster, tr, va)


def x_i11(tr, va):
    from .ideas import augmented_gru_forecaster
    return _seed25(augmented_gru_forecaster, tr, va)


def x_s35(tr, va):
    from .ideas3 import mixup_forecaster
    return _seed25(mixup_forecaster, tr, va)


def x_i11_snap(tr, va):
    """i11 with cosine-restart snapshot ensembling (3 seeds x 4 snapshots)."""
    from .ideas import augmented_gru_forecaster
    return augmented_gru_forecaster(tr, va, seeds=(0, 1, 2), cyclic=(40, 4))


def x_s35_snap(tr, va):
    from .ideas3 import mixup_forecaster
    return mixup_forecaster(tr, va, seeds=(0, 1, 2), cyclic=(40, 4))


def x_i11_snap25(tr, va):
    """The a3x25 recipe applied to i11: 25 seeds x 4 snapshots = 100 models.

    X5 showed snapshots reach 0.00911 at 1/21 the cost of X3's 25 plain seeds
    (0.00898), so the two axes are complementary rather than substitutes --
    this is the cell that makes a3x25 the best single model in the project.
    """
    from .ideas import augmented_gru_forecaster
    return augmented_gru_forecaster(tr, va, seeds=tuple(range(25)), cyclic=(40, 4))


def x_s35_snap25(tr, va):
    from .ideas3 import mixup_forecaster
    return mixup_forecaster(tr, va, seeds=tuple(range(25)), cyclic=(40, 4))


def c1_f2_provided_only(train_full, val_masked):
    """f2's XGBoost pipeline restricted to the competition data package.

    Compliance contingency (PLAN W6). The rules require "self-contained code
    that reproduces the submitted predictions when run end-to-end on the
    provided data files" and say nothing either way about external covariates.
    Under the strict reading, the shipped f2 member is implicated: it uses
    external statics (Census/NLCD/FIA/terrain) AND external sub-county weather
    (Open-Meteo). This is the same model with both dropped, so the cost of full
    compliance can be quoted as a measured number rather than guessed.
    """
    import xgboost as xgb
    from .ideas6 import _rows
    from .features import feature_cols
    from .ideas import _out
    from .models import SEED
    tr, y = _rows(train_full, statics=False, sub=False)
    va = _rows(val_masked, statics=False, sub=False, val=True)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4,
                           tree_method="hist")
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)


def x9_s51_snap(tr, va):
    """s51 with cosine-restart snapshot ensembling (3 seeds x 4 snapshots).

    The last untreated sequence member. s51 is the SECOND-best member (0.00916)
    and is a single-model trainer with no internal averaging, so law 13b
    predicts it should gain -- but it was never seed-scaled or snapshotted
    because its training loop (`_fit_transformed`) is separate from the shared
    `_fit_one` that i11 and s35 use. Snapshots first: X5/X6 showed they buy most
    of the seed-averaging gain at ~1/15 the cost.
    """
    from .ideas5 import bienc_dual_forecaster
    return bienc_dual_forecaster(tr, va, seeds=(0, 1, 2), cyclic=(40, 4))


def x10_s51_seed25(tr, va):
    from .ideas5 import bienc_dual_forecaster
    return bienc_dual_forecaster(tr, va, seeds=tuple(range(25)))


IDEAS15 = {
    "X9 s51 dual-loss + snapshot ensembling": x9_s51_snap,
    "X10 s51 dual-loss, 25 seeds": x10_s51_seed25,
    "C1 f2 restricted to provided data only (compliance contingency)": c1_f2_provided_only,
    "X7 i11 storm-jitter, 25 seeds x 4 snapshots": x_i11_snap25,
    "X8 s35 mixup, 25 seeds x 4 snapshots": x_s35_snap25,
    "X5 i11 storm-jitter + snapshot ensembling": x_i11_snap,
    "X6 s35 mixup + snapshot ensembling": x_s35_snap,
    "X1 s39 cross-county attention, 25 seeds": x_s39,
    "X2 s40 multi-origin, 25 seeds": x_s40,
    "X3 i11 storm-jitter, 25 seeds": x_i11,
    "X4 s35 mixup, 25 seeds": x_s35,
    "D1 MLP on the f2 feature set": d1_mlp,
    "D2 Chebyshev-KAN on the f2 feature set": d2_kan,
    "D3 TabNet on the f2 feature set": d3_tabnet,
    "D4 FT-Transformer on the f2 feature set": d4_ftt,
    "D5 proper B-spline KAN (quantile inputs, base residual)": d5_kan2,
}
