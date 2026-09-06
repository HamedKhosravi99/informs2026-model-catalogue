"""Two-stage by county group: an out-of-fold random forest splits counties into
SEVERE / NORMAL from legal features; then, per group, the best configuration among
everything already built is chosen on four folds and applied to the fifth.

Candidates per group: the built ensemble, each of its 8 members alone, A3x25 alone,
the 8-member mean alone, the blend at w in {0.25, 0.75}, and the ensemble scaled by
c in {1.1, 1.25, 1.5, 0.9} (the cheapest 'severe specialist': less shrinkage).
Run once with the RF gate and once with a PERFECT gate (true group) for the ceiling.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
from canonical import MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP, load, truth, TAIL_SPLIT_HOUR, TAIL_SCALE
from src.data import load_train
from src.validate import make_folds

ROWS = Path(sys.argv[1]); Q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75   # severe = top (1-Q) by true total OSI
t = truth(); train = load_train()
fz = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
P = {m: load(m).set_index(["fipsCode", "hour"])["osi_pred"] for m in fz + ["a3x25"]}
idx = P["a3x25"].index
for m in fz: idx = idx.intersection(P[m].index)
f = idx.get_level_values(0).to_numpy(); h = idx.get_level_values(1).to_numpy()
Y = np.array([t.loc[a, b] if b in t.columns else np.nan for a, b in idx]); ok = np.isfinite(Y)
tail = np.where(h >= TAIL_SPLIT_HOUR, TAIL_SCALE, 1.0)
M8 = sum(P[m].reindex(idx).to_numpy() for m in fz) / len(fz); A = P["a3x25"].reindex(idx).to_numpy()
ens = (0.5 * M8 + 0.5 * A) * tail
C = {"ensemble": ens, "mean8": M8 * tail, "a3x25": A * tail, "w=0.25": (0.25 * M8 + 0.75 * A) * tail, "w=0.75": (0.75 * M8 + 0.25 * A) * tail}
for m in fz: C[m] = P[m].reindex(idx).to_numpy() * tail
for c in (0.9, 1.1, 1.25, 1.5): C[f"ens x{c}"] = ens * c
HZ = (1, 6, 24, 48)
fold_of = {c: k for k, (_, va) in enumerate(make_folds(train, 5)) for c in va}; fold = np.array([fold_of[c] for c in f])
tot = pd.Series(np.where(ok, Y, 0)).groupby(f).sum()                   # true total OSI per county (diagnosis only / oracle gate)
counties = np.array(sorted(set(f))); cf = np.array([fold_of[c] for c in counties])

# ---- stage 1: OOF random-forest gate on legal per-county features (f2's rows aggregated)
va = pd.concat([pd.read_csv(ROWS / f"fold{k}_val.csv") for k in range(5)])
cols = [c for c in va.columns if c not in ("fipsCode", "hour")]
X = pd.concat([va.groupby("fipsCode")[cols].mean().add_suffix("_mean"), va.groupby("fipsCode")[cols].max().add_suffix("_max"),
               va.groupby("fipsCode")[cols].min().add_suffix("_min")], axis=1).reindex(counties).fillna(0)
X = X.loc[:, X.std() > 0]
rf_prob = np.zeros(len(counties)); true_sev = np.zeros(len(counties), bool)
for k in range(5):
    tr, te = cf != k, cf == k
    thr = np.quantile(tot.reindex(counties[tr]), Q); lab = (tot.reindex(counties) >= thr).to_numpy()
    true_sev[te] = lab[te]
    rf = RandomForestClassifier(500, min_samples_leaf=3, random_state=0).fit(X[tr], lab[tr]); rf_prob[te] = rf.predict_proba(X[te])[:, 1]
rf_sev = rf_prob >= 0.5
prec = (true_sev & rf_sev).sum() / max(rf_sev.sum(), 1); rec = (true_sev & rf_sev).sum() / max(true_sev.sum(), 1)
print(f"STAGE 1 (RF gate, out-of-fold): severe = top {100*(1-Q):.0f}% by true total OSI -> {true_sev.sum()} counties")
print(f"  RF flags {rf_sev.sum()} counties as severe; precision {prec:.2f}, recall {rec:.2f}")
share = ((ens - Y) ** 2)[ok]; share_sev = share[np.isin(f, counties[true_sev])[ok]].sum() / share.sum()
print(f"  the true severe group carries {100*share_sev:.0f}% of the ensemble's squared error\n")

def run(gate_sev, label):
    grp = np.isin(f, counties[gate_sev])                                   # cell-level group membership
    num = {k: [0.0, 0] for k in HZ}; picks = []
    for k in range(5):
        tr, te = fold != k, fold == k; pk = {}
        for gname, gm in (("severe", grp), ("normal", ~grp)):
            m = tr & gm & ok
            best = min(C, key=lambda c: np.mean([np.sqrt(((C[c] - Y) ** 2)[m & (h >= 72 + kk)].mean()) for kk in HZ]))
            pk[gname] = best
            for kk in HZ:
                s = te & gm & ok & (h >= 72 + kk); num[kk][0] += ((C[best] - Y) ** 2)[s].sum(); num[kk][1] += s.sum()
        picks.append((pk["severe"], pk["normal"]))
    r = float(np.mean([np.sqrt(v[0] / v[1]) for v in num.values()]))
    base = float(np.mean([np.sqrt(((ens - Y) ** 2)[ok & (h >= 72 + kk)].mean()) for kk in HZ]))
    print(f"{label}: nested {r:.6f} vs single global ensemble {base:.6f} -> {r - base:+.6f}")
    for k, p in enumerate(picks): print(f"   fold {k}: severe -> {p[0]:10}   normal -> {p[1]}")

run(rf_sev, "STAGE 2 with the RF gate     ")
print()
run(true_sev, "STAGE 2 with a PERFECT gate  (ceiling for this idea)")
