"""Nested search over ensemble weighting schemes.

Every fitted scheme is evaluated honestly: parameters are fit on four folds and
applied to the held-out fifth, then the five held-out blocks are pooled and scored
once. The non-nested (optimistic) number is printed beside it so the overfitting
gap is visible.
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from src.data import load_train, osi_trajectory, FREEZE_H, HORIZONS
from src.validate import make_folds, score_trajectory

PROD = {"s39": .14, "i11": .12, "s31": .10, "s44b": .10, "s40": .10, "s29d": .08,
        "s35": .08, "gru_comp": .08, "transformer": .07, "s30": .07, "s23b": .06}
LIB = ["s31", "s44b", "s39", "i07", "i11", "s35", "s40", "gru_comp", "transformer",
       "s30", "gru", "i20", "s29d", "s29c", "s24b", "s23b", "i16", "perstate_hurdle",
       "mlp", "i05", "s22b"]

train = load_train()
truth = osi_trajectory(train)
m = None
for k in LIB:
    p = pd.read_csv(f"oof/{k}.csv").rename(columns={"osi_pred": k})
    m = p if m is None else m.merge(p, on=["fipsCode", "hour"])
m["truth"] = truth.stack().reindex(
    pd.MultiIndex.from_frame(m[["fipsCode", "hour"]])).to_numpy()
m = m.dropna(subset=["truth"]).reset_index(drop=True)

folds = list(make_folds(train))
m["fold"] = -1
for k, (_, va) in enumerate(folds):
    m.loc[m.fipsCode.isin(va), "fold"] = k

# masks marking which rows are scored by which horizon column
HMASK = {h: (m.hour >= FREEZE_H + 1 + h).to_numpy() for h in HORIZONS.values()}


def sc(pred):
    d = m[["fipsCode", "hour"]].copy()
    d["osi_pred"] = pred
    t = score_trajectory(d, truth)
    return t.loc["mean", "rmse"], t.loc["mean", "mae"]


def objective(w, X, y, masks):
    """The competition objective: mean of the four per-horizon RMSEs."""
    p = X @ w
    return np.mean([np.sqrt(np.mean((p[mk] - y[mk]) ** 2)) for mk in masks])


# ------------------------------------------------------------------ weight fitters
def f_equal(X, y, masks):
    return np.ones(X.shape[1]) / X.shape[1]


def f_nnls(X, y, masks):
    return nnls(X, y)[0]


def f_simplex(X, y, masks):
    n = X.shape[1]
    w0 = np.ones(n) / n
    r = minimize(lambda w: np.mean((X @ w - y) ** 2), w0, method="SLSQP",
                 bounds=[(0, 1)] * n,
                 constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                 options={"maxiter": 200})
    return r.x


def f_objective(X, y, masks):
    """Fit the actual metric (mean of per-horizon RMSEs) rather than pooled MSE."""
    n = X.shape[1]
    w0 = np.ones(n) / n
    r = minimize(objective, w0, args=(X, y, masks), method="SLSQP",
                 bounds=[(0, 1)] * n,
                 constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                 options={"maxiter": 300})
    return r.x


def f_invmse(X, y, masks):
    mse = np.array([np.mean((X[:, j] - y) ** 2) for j in range(X.shape[1])])
    w = 1.0 / mse
    return w / w.sum()


def f_bma(X, y, masks, n_eff=200.0):
    mse = np.array([np.mean((X[:, j] - y) ** 2) for j in range(X.shape[1])])
    ll = -n_eff / 2 * mse / mse.min()
    w = np.exp(ll - ll.max())
    return w / w.sum()


def f_ridge_to_equal(X, y, masks, lam=1.0):
    n = X.shape[1]
    eq = np.ones(n) / n
    A = X.T @ X / len(X) + lam * np.eye(n)
    b = X.T @ y / len(X) + lam * eq
    w = np.linalg.solve(A, b)
    return np.clip(w, 0, None)


def f_greedy(X, y, masks, iters=40):
    """Caruana forward selection with replacement."""
    n = X.shape[1]
    chosen, cur = [], np.zeros(len(X))
    for it in range(iters):
        best, bj = np.inf, None
        for j in range(n):
            cand = (cur * len(chosen) + X[:, j]) / (len(chosen) + 1)
            e = np.mean((cand - y) ** 2)
            if e < best:
                best, bj = e, j
        chosen.append(bj)
        cur = X[:, chosen].mean(1)
    w = np.zeros(n)
    for j in chosen:
        w[j] += 1.0 / len(chosen)
    return w


def f_shrunk_nnls(X, y, masks, alpha=0.5):
    w = nnls(X, y)[0]
    w = w / max(w.sum(), 1e-9)
    return alpha * w + (1 - alpha) * np.ones(len(w)) / len(w)


SCHEMES = {
    "equal (all 21)": f_equal,
    "NNLS": f_nnls,
    "simplex LS": f_simplex,
    "fit-the-metric (simplex)": f_objective,
    "inverse-MSE": f_invmse,
    "BMA": f_bma,
    "ridge->equal (lam=1)": lambda X, y, k: f_ridge_to_equal(X, y, k, 1.0),
    "ridge->equal (lam=10)": lambda X, y, k: f_ridge_to_equal(X, y, k, 10.0),
    "greedy Caruana": f_greedy,
    "NNLS shrunk 50% to equal": f_shrunk_nnls,
}


def run_nested(fitter, members=LIB):
    X_all = m[members].to_numpy()
    y = m["truth"].to_numpy()
    pred = np.zeros(len(m))
    for k in range(len(folds)):
        tr = (m.fold != k).to_numpy()
        te = ~tr
        masks = [HMASK[h][tr] for h in HORIZONS.values()]
        w = fitter(X_all[tr], y[tr], masks)
        if w.sum() > 0:
            w = w / w.sum()
        pred[te] = X_all[te] @ w
    return np.clip(pred, 0, None)


def run_nonnested(fitter, members=LIB):
    X_all = m[members].to_numpy()
    y = m["truth"].to_numpy()
    masks = [HMASK[h] for h in HORIZONS.values()]
    w = fitter(X_all, y, masks)
    if w.sum() > 0:
        w = w / w.sum()
    return np.clip(X_all @ w, 0, None)


print("=" * 78)
prod_pred = sum(v * m[k] for k, v in PROD.items()).to_numpy()
pr, pm = sc(prod_pred)
print(f"PRODUCTION (fixed round weights, nothing fitted): RMSE {pr:.5f}  MAE {pm:.5f}")
print("=" * 78)
print(f"{'scheme':32s} {'nested':>9s} {'non-nested':>11s} {'gap':>9s} {'vs prod':>9s}")
rows = [("production (no fitting)", pr, pr, 0.0, 0.0)]
for name, f in SCHEMES.items():
    nn = sc(run_nested(f))[0]
    on = sc(run_nonnested(f))[0]
    print(f"{name:32s} {nn:9.5f} {on:11.5f} {on-nn:+9.5f} {nn-pr:+9.5f}")
    rows.append((name, nn, on, on - nn, nn - pr))

# per-horizon weights (4x the parameters)
def run_per_horizon(fitter):
    X_all = m[LIB].to_numpy()
    y = m["truth"].to_numpy()
    pred = np.zeros(len(m))
    counts = np.zeros(len(m))
    for k in range(len(folds)):
        tr = (m.fold != k).to_numpy(); te = ~tr
        for h in HORIZONS.values():
            hm = HMASK[h]
            sel_tr = tr & hm
            w = fitter(X_all[sel_tr], y[sel_tr], [hm[sel_tr]])
            if w.sum() > 0: w = w / w.sum()
            sel_te = te & hm
            pred[sel_te] += X_all[sel_te] @ w
            counts[sel_te] += 1
    pred = np.where(counts > 0, pred / np.maximum(counts, 1), 0)
    return np.clip(pred, 0, None)

for nm, f in [("per-horizon NNLS", f_nnls), ("per-horizon simplex", f_simplex)]:
    v = sc(run_per_horizon(f))[0]
    print(f"{nm:32s} {v:9.5f} {'-':>11s} {'-':>9s} {v-pr:+9.5f}")
    rows.append((nm, v, np.nan, np.nan, v - pr))

# robust pooling operators (no fitting at all)
P = m[list(PROD.keys())].to_numpy()
for nm, pred in [("median of production members", np.median(P, 1)),
                 ("trimmed mean (drop hi+lo)", (P.sum(1) - P.max(1) - P.min(1)) / (P.shape[1] - 2)),
                 ("equal mean of production members", P.mean(1))]:
    v, mm = sc(np.clip(pred, 0, None))
    print(f"{nm:32s} {v:9.5f} {'(no fit)':>11s} {'-':>9s} {v-pr:+9.5f}")
    rows.append((nm, v, np.nan, np.nan, v - pr))

print("\nBest nested schemes:")
for r in sorted(rows, key=lambda r: r[1])[:6]:
    print(f"  {r[0]:34s} nested RMSE {r[1]:.5f}  ({r[4]:+.5f} vs production)")
pd.DataFrame(rows, columns=["scheme", "nested_rmse", "nonnested_rmse", "gap", "vs_prod"]
             ).to_csv("results/weight_results.csv", index=False)
