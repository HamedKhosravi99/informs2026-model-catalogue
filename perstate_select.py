"""Per-STATE model selection from the cached OOF library (user hypothesis:
each state may have a different best model).

The ledger tested per-state TRAINING (rung 13: worked in the pooled-GBM era;
S53 per-state heads: lost against modern members - law 4). This tests
per-state SELECTION: every candidate stays trained on all counties; each
state chooses its serving model. Selection is NESTED (winner for a state
chosen on 4 folds, applied to that state's counties in the 5th), with the
hindsight (non-nested) number printed as the optimism reference, exactly as
percol_select.py did for the horizon axis.

Also: the per-state pooled table itself (which model IS best per state), and
nested per-state top-3 equal blends.
"""
import numpy as np
import pandas as pd

from src.data import FREEZE_H, HORIZONS, load_train, osi_trajectory
from src.validate import make_folds

LIB = ["ens8p1b", "a3x25", "a11x25", "a3", "e2", "m1c", "s62", "s51",
       "s39", "s40", "f2", "i11", "s35", "p1b", "b3", "e_top8"]

train = load_train()
truth = osi_trajectory(train)
folds = list(make_folds(train))
fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}
state_of = train.groupby("fipsCode")["stateAbbr"].first()

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
st_arr = state_of.reindex(idx).to_numpy()
STATES = ["IN", "OH", "PA", "WV"]

COLS = {n: np.arange(FREEZE_H + 1 + h, 216) for n, h in HORIZONS.items()}
T = {n: truth.loc[idx, c].to_numpy() for n, c in COLS.items()}
M = {k: {n: P[k].reindex(index=idx, columns=c).to_numpy()
         for n, c in COLS.items()} for k in LIB}


def metric(pred_by_col, county_sel):
    """Official form: per-horizon RMSE over the selected counties, averaged."""
    rs = []
    for n in HORIZONS:
        e = pred_by_col[n][county_sel] - T[n][county_sel]
        rs.append(np.sqrt(np.nanmean(e ** 2)))
    return float(np.mean(rs))


print(f"library ({len(LIB)}): {LIB}\n")
print("per-state metric of each candidate (pooled over all folds):")
print(f"{'model':10s} " + "  ".join(f"{s:>8s}" for s in STATES) + f"   {'ALL':>8s}")
best_by_state = {}
for k in LIB:
    vals = []
    for s in STATES:
        v = metric(M[k], st_arr == s)
        vals.append(v)
        if s not in best_by_state or v < best_by_state[s][1]:
            best_by_state[s] = (k, v)
    print(f"{k:10s} " + "  ".join(f"{v:8.5f}" for v in vals)
          + f"   {metric(M[k], slice(None)):8.5f}")
print("\nhindsight best per state: "
      + ", ".join(f"{s}: {best_by_state[s][0]} ({best_by_state[s][1]:.5f})"
                  for s in STATES))

# hindsight per-state composite (non-nested optimism reference)
comp = {n: np.full_like(T[n], np.nan) for n in HORIZONS}
for s in STATES:
    k = best_by_state[s][0]
    for n in HORIZONS:
        comp[n][st_arr == s] = M[k][n][st_arr == s]
print(f"hindsight per-state composite: {metric(comp, slice(None)):.5f}"
      "   <- non-nested optimism reference")

# ---------------- nested per-state winner selection
picked = {s: [] for s in STATES}
nest = {n: np.full_like(T[n], np.nan) for n in HORIZONS}
for kf in range(len(folds)):
    tr_sel, te_sel = fold_arr != kf, fold_arr == kf
    for s in STATES:
        s_tr = tr_sel & (st_arr == s)
        s_te = te_sel & (st_arr == s)
        best = min(LIB, key=lambda m: metric(M[m], s_tr))
        picked[s].append(best)
        for n in HORIZONS:
            nest[n][s_te] = M[best][n][s_te]
print("\nnested per-state selection:")
for s in STATES:
    print(f"  {s}: picks across folds = {picked[s]}")
print(f"  nested per-state selection metric: {metric(nest, slice(None)):.5f}")

# ---------------- nested per-state top-3 equal blend
nest3 = {n: np.full_like(T[n], np.nan) for n in HORIZONS}
for kf in range(len(folds)):
    tr_sel, te_sel = fold_arr != kf, fold_arr == kf
    for s in STATES:
        s_tr = tr_sel & (st_arr == s)
        s_te = te_sel & (st_arr == s)
        ranked = sorted(LIB, key=lambda m: metric(M[m], s_tr))[:3]
        for n in HORIZONS:
            nest3[n][s_te] = np.mean([M[r][n][s_te] for r in ranked], 0)
print(f"  nested per-state top-3 blend:      {metric(nest3, slice(None)):.5f}")

print(f"\nreference: shipped ensemble everywhere = "
      f"{metric(M['ens8p1b'], slice(None)):.5f}")
