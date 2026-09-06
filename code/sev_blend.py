"""Severity-conditional blend weight, nested.

Mechanism: A3x25 is a snapshot ensemble -- smooth, strong on quiet counties;
the 8-member mean carries more diverse spike shapes. If that is true, the
50/50 weight should depend on how severe the county already looks at the
origin, which is a legal feature (max observed OSI over hours 0-71). One extra
parameter: a separate weight for counties above/below the training-fold median
severity, chosen on four folds and applied to the fifth.
"""
import numpy as np, pandas as pd
from canonical import MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP, load, truth, TAIL_SPLIT_HOUR, TAIL_SCALE
from src.data import load_train
from src.validate import make_folds

HZ = {"t01h": 1, "t06h": 6, "t24h": 24, "t48h": 48}
t = truth(); train = load_train()
fz = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
parts = {m: load(m).set_index(["fipsCode", "hour"])["osi_pred"] for m in fz + ["a3x25"]}
idx = parts["a3x25"].index
for m in fz: idx = idx.intersection(parts[m].index)
E = sum(parts[m].reindex(idx) for m in fz) / len(fz); A = parts["a3x25"].reindex(idx)
f = idx.get_level_values(0).to_numpy(); h = idx.get_level_values(1).to_numpy()
tail = np.where(h >= TAIL_SPLIT_HOUR, TAIL_SCALE, 1.0)
Y = np.array([t.loc[a, b] if b in t.columns else np.nan for a, b in idx]); ok = np.isfinite(Y)
sev = train[train.hour <= 71].groupby("fipsCode").osi.max()          # legal: observed window only
S = sev.reindex(f).to_numpy()
fold_of = {c: k for k, (_, va) in enumerate(make_folds(train, 5)) for c in va}; fold = np.array([fold_of[c] for c in f])
W = np.arange(0.0, 1.01, 0.05)

def blend(w_lo, w_hi, thr):
    w = np.where(S >= thr, w_hi, w_lo)
    return (w * E.to_numpy() + (1 - w) * A.to_numpy()) * tail

def mr(pred, mask):
    return float(np.mean([np.sqrt(((pred - Y) ** 2)[mask & ok & (h >= 72 + k)].mean()) for k in HZ.values()]))

base = mr(blend(0.5, 0.5, 0.0), np.ones_like(ok))
print(f"baseline (0.5/0.5): {base:.6f}")
num = {k: [0.0, 0] for k in HZ}; picks = []
for k in range(5):
    tr, te = fold != k, fold == k
    thr = np.median(sev.reindex(sorted(set(f[tr]))).to_numpy())
    best = min(((wl, wh) for wl in W for wh in W), key=lambda p: mr(blend(p[0], p[1], thr), tr))
    picks.append((round(best[0], 2), round(best[1], 2), round(float(thr), 4)))
    pred = blend(best[0], best[1], thr)
    for kk, hh in HZ.items():
        s = te & ok & (h >= 72 + hh); num[kk][0] += ((pred - Y) ** 2)[s].sum(); num[kk][1] += s.sum()
nested = float(np.mean([np.sqrt(v[0] / v[1]) for v in num.values()]))
print("per-fold (w_low_severity, w_high_severity, threshold):", picks)
print(f"nested {nested:.6f}  -> {nested - base:+.6f}   ({'pays' if nested < base else 'does NOT pay'})")
