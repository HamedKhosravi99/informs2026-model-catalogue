"""One model, seven honest 'CV RMSE' numbers - the aggregation-convention table.

Scores the shipped ensemble's cached OOF predictions (oof/ens8p1b.csv, which
already carries the nested tail correction) under seven defensible conventions
for turning per-cell errors into one number. Regenerates the table in
the experiment catalogue. Read-only.
"""
import numpy as np
import pandas as pd

from src.data import FREEZE_H, HORIZONS, load_train, osi_trajectory
from src.validate import make_folds

train = load_train()
truth = osi_trajectory(train)
d = pd.read_csv("oof/ens8p1b.csv")
piv = d.pivot_table(index="fipsCode", columns="hour", values="osi_pred")
folds = list(make_folds(train))
fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}

cells = []
for name, h in HORIZONS.items():
    tgt = np.arange(FREEZE_H + 1 + h, 216)
    t = truth.loc[piv.index, tgt].to_numpy()
    p = piv.reindex(columns=tgt).to_numpy()
    e2 = (p - t) ** 2
    for i, f in enumerate(piv.index):
        cells.append((name, f, np.nansum(e2[i]), int(np.isfinite(e2[i]).sum())))
df = pd.DataFrame(cells, columns=["col", "fips", "sse", "n"])

r_h = df.groupby("col").apply(lambda g: np.sqrt(g.sse.sum() / g.n.sum()))
print(f"1. per-horizon RMSE, averaged (headline)    : {r_h.mean():.5f}")
print(f"2. pooled RMSE over all cells               : "
      f"{np.sqrt(df.sse.sum() / df.n.sum()):.5f}")
pf = []
for k in range(5):
    g = df[df.fips.map(fold_of) == k]
    pf.append(g.groupby("col").apply(
        lambda x: np.sqrt(x.sse.sum() / x.n.sum())).mean())
print(f"3. metric per fold, averaged over folds     : {np.mean(pf):.5f}")
pc = df.groupby("fips").apply(lambda g: np.sqrt(g.sse.sum() / g.n.sum()))
print(f"4. per-county RMSE, averaged over counties  : {pc.mean():.5f}")
pch = df.assign(r=np.sqrt(df.sse / df.n.clip(lower=1))).groupby("col").r.mean()
print(f"5. per-county-per-horizon RMSE, averaged    : {pch.mean():.5f}")
print(f"6. median county RMSE                       : {pc.median():.5f}")
e = (np.sqrt(np.clip(piv.to_numpy(), 0, None))
     - np.sqrt(np.clip(truth.loc[piv.index, piv.columns].to_numpy(), 0, None)))
print(f"7. RMSE on sqrt(OSI) scale, trajectory      : {np.sqrt(np.nanmean(e**2)):.5f}")
