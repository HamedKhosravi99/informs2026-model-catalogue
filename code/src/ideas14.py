"""Sweep 16 (K-series) - closed-form kernel-ridge functional-output member.

Maps a fixed-length summary of (72h observed prefix + full-window weather) to
the ENTIRE 143-hour trajectory in one closed-form RKHS solve:
    A = (K + lam I)^-1 Y,  Y in R^(n x 143)
No gradient descent, no per-row regression, no rollout. At n=239 the exact
Gram matrix is trivial and the solution is deterministic.

Motivation (literature sweep, 2026-08-14): Toner & Darlow (ICML 2024) show the
linear-forecaster family admits a closed form that beats its gradient-trained
version in 72% of settings; Kadri et al. (JMLR 2016) and Bouche et al.
(AISTATS 2021) give the operator-valued/functional-output kernel machinery;
Arora et al. (ICLR 2020) is the n<=640 kernels-beat-nets evidence.

Why it matters here: reported error correlation ~0.75 with the eight shipped
members - below the project's 0.80 decorrelation floor, matching p1b's 0.752,
at no accuracy cost. A fixed smooth stationary kernel plus one joint 143-dim
solve is a structurally different computation from tree/RNN members.

Causality: inputs are OSI hours 0-71 only, weather at any hour (legal), and
customersTracked/statics. Nothing post-freeze. alpha/gamma are selected on an
inner 3-fold of the outer-TRAINING counties, so nesting holds by construction.
"""
import numpy as np
import pandas as pd

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .dl import _pivot
from .features import county_static

IDEAS14 = {}

WBLOCKS = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144),
           (144, 168), (168, 192), (192, 216)]


def _design(masked):
    """Fixed-length county descriptor: prefix OSI shape + blocked weather."""
    from .ideas import load_static
    O = np.nan_to_num(_pivot(masked, "osi", np.arange(FREEZE_H + 1)).to_numpy())
    G = np.nan_to_num(_pivot(masked, "gust", np.arange(216)).to_numpy())
    TP = np.nan_to_num(_pivot(masked, "tp", np.arange(216)).to_numpy())
    SM = np.nan_to_num(_pivot(masked, "soil_moist", np.arange(216)).to_numpy())
    st = county_static(masked)
    fips = st.index.to_numpy()
    ext = load_static().reindex(fips)

    feats = [
        O[:, -1][:, None], O[:, -6:].mean(1)[:, None], O[:, -24:].mean(1)[:, None],
        O.max(1)[:, None], O.sum(1)[:, None],
        (O[:, -6:].mean(1) - O[:, -18:-12].mean(1))[:, None],
        np.sqrt(np.clip(O[:, -1], 0, None))[:, None],
    ]
    # coarse prefix shape: 12 six-hour means
    feats.append(O.reshape(len(O), 12, 6).mean(2))
    for a, b in WBLOCKS:
        feats += [G[:, a:b].max(1)[:, None], G[:, a:b].mean(1)[:, None],
                  np.clip(G[:, a:b] - 35, 0, None).sum(1)[:, None],
                  TP[:, a:b].sum(1)[:, None]]
    feats += [SM.mean(1)[:, None], G.max(1)[:, None],
              np.clip(G - 35, 0, None).sum(1)[:, None],
              st[["log_customers"]].to_numpy(),
              ext[["log_pop", "log_density", "rucc", "lat", "lon"]].to_numpy()]
    X = np.nan_to_num(np.hstack(feats))
    return fips, X


def _fit_predict(Xtr, Ytr, Xva, alpha, gamma):
    from sklearn.kernel_ridge import KernelRidge
    kr = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    kr.fit(Xtr, Ytr)
    return kr.predict(Xva)


def krr_forecaster(train_full, val_masked, target="sqrt"):
    from sklearn.preprocessing import StandardScaler
    from .validate import make_folds

    masked_tr = mask_after_freeze(train_full)
    f_tr, X_tr = _design(masked_tr)
    f_va, X_va = _design(val_masked)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_va = sc.transform(X_tr), sc.transform(X_va)

    Yraw = (train_full.pivot_table(index="fipsCode", columns="hour", values="osi")
            .sort_index().reindex(index=f_tr, columns=PRED_HOURS).to_numpy())
    Yraw = np.nan_to_num(Yraw)
    Y = np.sqrt(np.clip(Yraw, 0, None)) if target == "sqrt" else Yraw

    # inner selection on TRAINING counties only
    inner = list(make_folds(train_full[train_full.fipsCode.isin(f_tr)], n_splits=3))
    idx_of = {f: i for i, f in enumerate(f_tr)}
    best, best_v = (1.0, 1.0 / X_tr.shape[1]), np.inf
    for alpha in (0.03, 0.1, 0.3, 1.0, 3.0):
        for gmul in (0.25, 0.5, 1.0, 2.0):
            g = gmul / X_tr.shape[1]
            errs = []
            for tr_f, va_f in inner:
                ti = np.array([idx_of[f] for f in tr_f if f in idx_of])
                vi = np.array([idx_of[f] for f in va_f if f in idx_of])
                if len(vi) == 0:
                    continue
                P = _fit_predict(X_tr[ti], Y[ti], X_tr[vi], alpha, g)
                P = np.clip(P, 0, None) ** 2 if target == "sqrt" else np.clip(P, 0, None)
                errs.append(np.mean((P - Yraw[vi]) ** 2))
            v = float(np.mean(errs))
            if v < best_v:
                best_v, best = v, (alpha, g)

    P = _fit_predict(X_tr, Y, X_va, *best)
    P = np.clip(P, 0, None) ** 2 if target == "sqrt" else np.clip(P, 0, None)
    return pd.DataFrame({"fipsCode": np.repeat(f_va, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(f_va)),
                         "osi_pred": P.ravel()})


def k1_krr(tr, va):      return krr_forecaster(tr, va, target="sqrt")
def k1b_krr_raw(tr, va): return krr_forecaster(tr, va, target="raw")


IDEAS14 = {
    "K1 kernel-ridge functional output (sqrt)": k1_krr,
    "K1b kernel-ridge functional output (raw)": k1b_krr_raw,
}
