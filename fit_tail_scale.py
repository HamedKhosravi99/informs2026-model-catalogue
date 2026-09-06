"""Reproduce the quiet-tail bias correction used in src/ideas4.py.

Prints (a) the ensemble's bias by hour band, (b) the least-squares tail scale fitted
on all 239 training counties, and (c) the nested validation of the correction (scale
fit on four folds, applied to the held-out fifth). Requires the cached member
out-of-fold predictions in oof/ (regenerate with run_all_ideas.py).
"""
import numpy as np
import pandas as pd

from src.data import load_train, osi_trajectory
from src.validate import make_folds, score_trajectory
from src.ideas4 import ENSEMBLE_FINAL_W, TAIL_SPLIT_HOUR, TAIL_SCALE

train = load_train()
truth = osi_trajectory(train)
m = None
for k in ENSEMBLE_FINAL_W:
    p = pd.read_csv(f"oof/{k}.csv").rename(columns={"osi_pred": k})
    m = p if m is None else m.merge(p, on=["fipsCode", "hour"])
m["truth"] = truth.stack().reindex(
    pd.MultiIndex.from_frame(m[["fipsCode", "hour"]])).to_numpy()
m = m.dropna(subset=["truth"]).reset_index(drop=True)
pred = sum(w * m[k] for k, w in ENSEMBLE_FINAL_W.items()).to_numpy()
y, hour = m["truth"].to_numpy(), m["hour"].to_numpy()

print("ensemble bias by hour band (mean predicted - mean true):")
for lo, hi in [(73, 96), (96, 120), (120, 168), (168, 216)]:
    s = (hour >= lo) & (hour < hi)
    print(f"  hours {lo:3d}-{hi-1:3d}: {(pred[s]-y[s]).mean():+.6f}"
          f"   true {y[s].mean():.5f}  pred {pred[s].mean():.5f}")

tail = hour >= TAIL_SPLIT_HOUR
s_fit = (pred[tail] @ y[tail]) / (pred[tail] @ pred[tail])
print(f"\nleast-squares tail scale on all 239 counties: {s_fit:.4f}")
print(f"half-shrunk toward 1 (deployed value):        {0.5*s_fit + 0.5:.4f}"
      f"   (src/ideas4.py uses {TAIL_SCALE})")

folds = list(make_folds(train))
fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}
fold = m["fipsCode"].map(fold_of).to_numpy()


def score(p):
    d = m[["fipsCode", "hour"]].copy()
    d["osi_pred"] = np.clip(p, 0, None)
    t = score_trajectory(d, truth)
    return t.loc["mean", "rmse"], t.loc["mean", "mae"]


nested = pred.copy()
for k in range(len(folds)):
    tr, te = fold != k, fold == k
    a = tr & tail
    s = 0.5 * (pred[a] @ y[a]) / (pred[a] @ pred[a]) + 0.5
    nested[te & tail] = pred[te & tail] * s
print(f"\nnested validation: uncorrected {score(pred)[0]:.5f}/{score(pred)[1]:.5f}"
      f"  ->  corrected {score(nested)[0]:.5f}/{score(nested)[1]:.5f}  (RMSE/MAE)")
