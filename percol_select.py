"""Per-horizon-column model selection and blending from the cached OOF library.

The user's hypothesis: different models may be best for different horizon
columns, and nothing forces one trajectory to fill all four columns. The WLOG
argument (REPORT 5u) says the OPTIMAL predictions coincide across columns, but
finite models are not optimal - per-column selection can still pay on
bias-variance grounds. S52 (training four specialists) failed by fragmenting
data; this instead trains nothing: every candidate is a cached full-trajectory
model, and each column selects/blends per column.

All selection is NESTED: the winner (or blend weights) for a column is chosen
on four folds and applied to the held-out fifth, per fold. Non-nested numbers
are printed alongside to measure the selection optimism directly (5g style).

Read-only over oof/. Writes oof/percol_*.csv summaries only.
"""
import numpy as np
import pandas as pd

from src.data import FREEZE_H, HORIZONS, load_train, osi_trajectory
from src.validate import make_folds
from src.ideas4 import TAIL_SPLIT_HOUR

# strong, distinct candidates only - a huge library maximises selection noise
LIB = ["ens8p1b", "a3x25", "a11x25", "a3", "a1", "e2", "m1c", "s62", "s51",
       "s39", "s40", "s45", "i11", "s35", "p1b", "e_top8"]

train = load_train()
truth = osi_trajectory(train)
folds = list(make_folds(train))
fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}

P = {}
for k in LIB:
    try:
        P[k] = pd.read_csv(f"oof/{k}.csv").pivot_table(
            index="fipsCode", columns="hour", values="osi_pred")
    except FileNotFoundError:
        pass
LIB = [k for k in LIB if k in P]
idx = P[LIB[0]].index
fold_arr = np.array([fold_of[f] for f in idx])

# per-column cell definitions
COLS = {}
for name, h in HORIZONS.items():
    tgt = np.arange(FREEZE_H + 1 + h, 216)
    COLS[name] = tgt

T = {name: truth.loc[idx, tgt].to_numpy() for name, tgt in COLS.items()}
M = {k: {name: P[k].reindex(index=idx, columns=tgt).to_numpy()
         for name, tgt in COLS.items()} for k in LIB}


def col_rmse(pred, name, county_sel):
    e = pred[county_sel] - T[name][county_sel]
    return np.sqrt(np.nanmean(e ** 2))


print(f"library ({len(LIB)}): {LIB}\n")
print("per-column RMSE of each candidate (pooled, all folds):")
hdr = "  ".join(f"{n:>8s}" for n in HORIZONS)
print(f"{'model':12s} {hdr}   {'mean':>8s}")
rows = []
for k in LIB:
    rs = [col_rmse(M[k][n], n, slice(None)) for n in HORIZONS]
    rows.append((np.mean(rs), k, rs))
    print(f"{k:12s} " + "  ".join(f"{r:8.5f}" for r in rs) + f"   {np.mean(rs):8.5f}")

# ---------------- nested per-column WINNER selection
print("\nnested per-column selection (winner chosen on 4 folds, applied to 5th):")
picked = {n: [] for n in HORIZONS}
nested_cols = {}
for name in HORIZONS:
    out = np.full_like(T[name], np.nan)
    for k in range(len(folds)):
        tr_sel = fold_arr != k
        te_sel = fold_arr == k
        best, best_v = None, np.inf
        for m in LIB:
            v = col_rmse(M[m][name], name, tr_sel)
            if v < best_v:
                best_v, best = v, m
        picked[name].append(best)
        out[te_sel] = M[best][name][te_sel]
    nested_cols[name] = out
rs_sel = [np.sqrt(np.nanmean((nested_cols[n] - T[n]) ** 2)) for n in HORIZONS]
print("  picks: " + "; ".join(f"{n}: {picked[n]}" for n in HORIZONS))
print("  nested per-column selection: " + "/".join(f"{r:.5f}" for r in rs_sel)
      + f"   mean {np.mean(rs_sel):.5f}")

# non-nested (hindsight) winner per column, for the optimism gap
rs_hind = []
for name in HORIZONS:
    best = min(LIB, key=lambda m: col_rmse(M[m][name], name, slice(None)))
    rs_hind.append(col_rmse(M[best][name], name, slice(None)))
print(f"  hindsight per-column winners:      "
      + "/".join(f"{r:.5f}" for r in rs_hind) + f"   mean {np.mean(rs_hind):.5f}"
      "   <- non-nested, optimism reference")

# ---------------- nested per-column TOP-2 EQUAL BLEND
print("\nnested per-column top-2 equal blend:")
rs_b2 = []
for name in HORIZONS:
    out = np.full_like(T[name], np.nan)
    for k in range(len(folds)):
        tr_sel, te_sel = fold_arr != k, fold_arr == k
        ranked = sorted(LIB, key=lambda m: col_rmse(M[m][name], name, tr_sel))
        blend = (M[ranked[0]][name] + M[ranked[1]][name]) / 2
        out[te_sel] = blend[te_sel]
    rs_b2.append(np.sqrt(np.nanmean((out - T[name]) ** 2)))
print("  " + "/".join(f"{r:.5f}" for r in rs_b2) + f"   mean {np.mean(rs_b2):.5f}")

# ---------------- nested per-column top-3 equal blend
rs_b3 = []
for name in HORIZONS:
    out = np.full_like(T[name], np.nan)
    for k in range(len(folds)):
        tr_sel, te_sel = fold_arr != k, fold_arr == k
        ranked = sorted(LIB, key=lambda m: col_rmse(M[m][name], name, tr_sel))
        blend = np.mean([M[r][name] for r in ranked[:3]], axis=0)
        out[te_sel] = blend[te_sel]
    rs_b3.append(np.sqrt(np.nanmean((out - T[name]) ** 2)))
print("nested per-column top-3 equal blend:")
print("  " + "/".join(f"{r:.5f}" for r in rs_b3) + f"   mean {np.mean(rs_b3):.5f}")

print(f"\nreference: shipped ensemble same accounting: "
      + "/".join(f"{col_rmse(M['ens8p1b'][n], n, slice(None)):.5f}" for n in HORIZONS)
      + f"   mean {np.mean([col_rmse(M['ens8p1b'][n], n, slice(None)) for n in HORIZONS]):.5f}")
