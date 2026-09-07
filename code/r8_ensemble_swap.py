"""R8b ensemble-level test: does swapping p1b -> r8b (or adding r8b) improve the
shipped 8-member ensemble? Identical nested tail-scale procedure as fit_tail_scale.py.

Result (the experiment catalogue): no — the swap is worse at all four horizons (0.00873 vs
0.00864, P(swap better) 0.153) and the 9-member add is also worse (0.00870, P 0.184).
The swap does hold the best ensemble MAE measured (0.00260), kept as an MAE
contingency. Writes nested-corrected ensemble OOFs to oof/ for compare_models.py.
Requires cached member OOFs (regenerate with run_all_ideas.py).
"""
import numpy as np
import pandas as pd

from src.data import load_train, osi_trajectory
from src.validate import make_folds, score_trajectory
from src.ideas4 import TAIL_SPLIT_HOUR

BASE7 = ["s62", "s51", "s39", "s40", "i11", "s45", "s35"]
POOLS = {
    "ens8p1b": BASE7 + ["p1b"],          # shipped -- must reproduce 0.00864/0.00275
    "ens8r8b": BASE7 + ["r8b"],          # the swap
    "ens9both": BASE7 + ["p1b", "r8b"],  # both stock-flow members
}

train = load_train()
truth = osi_trajectory(train)
folds = list(make_folds(train))
fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}

for name, members in POOLS.items():
    m = None
    for k in members:
        p = pd.read_csv(f"oof/{k}.csv").rename(columns={"osi_pred": k})
        m = p if m is None else m.merge(p, on=["fipsCode", "hour"])
    m["truth"] = truth.stack().reindex(
        pd.MultiIndex.from_frame(m[["fipsCode", "hour"]])).to_numpy()
    m = m.dropna(subset=["truth"]).reset_index(drop=True)
    pred = np.mean([m[k].to_numpy() for k in members], axis=0)
    y, hour = m["truth"].to_numpy(), m["hour"].to_numpy()
    fold = m["fipsCode"].map(fold_of).to_numpy()
    tail = hour >= TAIL_SPLIT_HOUR

    nested = pred.copy()
    scales = []
    for k in range(len(folds)):
        a = (fold != k) & tail
        s = 0.5 * (pred[a] @ y[a]) / (pred[a] @ pred[a]) + 0.5
        scales.append(s)
        nested[(fold == k) & tail] = pred[(fold == k) & tail] * s

    d = m[["fipsCode", "hour"]].copy()
    d["osi_pred"] = np.clip(nested, 0, None)
    d.to_csv(f"oof/{name}.csv", index=False)
    t = score_trajectory(d, truth)
    ph = " / ".join(f"{t.loc[h, 'rmse']:.5f}" for h in ["t01h", "t06h", "t24h", "t48h"])
    fw = []
    for k, (_, va) in enumerate(folds):
        fw.append(score_trajectory(d.query("fipsCode in @va"), truth).loc["mean", "rmse"])
    print(f"{name:10s} RMSE {t.loc['mean','rmse']:.5f}  MAE {t.loc['mean','mae']:.5f}"
          f"  per-horizon {ph}")
    print(f"{'':10s} per-fold " + " ".join(f"{v:.5f}" for v in fw)
          + f"   tail scales {np.round(scales, 3)}")
