"""Is there ANY legal feature combination that identifies the severe counties ahead of time?

Per-county table from f2's exact rows (statics, origin-state dynamics, known-future weather and
physics interactions), aggregated over the forecast window: mean/max/min of every column, ~240
features. Targets: top-13 by TRUE severity, the ensemble's worst-13 by squared error, log total
OSI, and the residual multiplier log(pred/true). Everything scored OUT-OF-FOLD on the same five
county folds. Then: unsupervised clustering (do the 13 concentrate?), an in-sample tree (what a
separating rule looks like, explicitly NOT evidence), and -- if an out-of-fold severity score has
signal -- a nested multiplier keyed on it, to see whether identifiability converts to RMSE.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score, r2_score
warnings.filterwarnings("ignore")
from canonical import MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP, ensemble, truth
from src.data import load_train
from src.validate import make_folds

ROWS = Path(sys.argv[1])
t = truth(); train = load_train()
fz = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
e = ensemble(fz, t, blend_with="a3x25").set_index(["fipsCode", "hour"])["osi_pred"]
y = np.array([t.loc[f, h] if h in t.columns else np.nan for f, h in e.index])
d = pd.DataFrame({"f": e.index.get_level_values(0), "h": e.index.get_level_values(1), "p": e.to_numpy(), "y": y}).dropna()
g = d.groupby("f").agg(true=("y", "sum"), pred=("p", "sum"), sse=("p", lambda s: 0.0))
g["sse"] = ((d.p - d.y) ** 2).groupby(d.f).sum()
worst13 = set(g.sse.sort_values(ascending=False).index[:13]); sev13 = set(g.true.sort_values(ascending=False).index[:13])
print(f"overlap between the ensemble's worst-13 and the TRUE top-13 by severity: {len(worst13 & sev13)}/13\n")

# ------------------------------------------------------------- per-county legal features
va = pd.concat([pd.read_csv(ROWS / f"fold{k}_val.csv") for k in range(5)])
cols = [c for c in va.columns if c not in ("fipsCode", "hour")]
agg = pd.concat([va.groupby("fipsCode")[cols].mean().add_suffix("_mean"), va.groupby("fipsCode")[cols].max().add_suffix("_max"),
                 va.groupby("fipsCode")[cols].min().add_suffix("_min")], axis=1)
agg = agg.loc[:, agg.std() > 0].fillna(0)
X = agg.reindex(g.index); print(f"per-county feature table: {X.shape[0]} counties x {X.shape[1]} legal features (f2's rows aggregated over the window)")
fold_of = {c: k for k, (_, va_) in enumerate(make_folds(train, 5)) for c in va_}; fold = np.array([fold_of[c] for c in X.index])
T = {"top13_severity": np.array([c in sev13 for c in X.index]), "worst13_error": np.array([c in worst13 for c in X.index]),
     "log_total_osi": np.log(g.true.to_numpy() + 1e-4), "resid_mult log(pred/true)": np.log((g.pred + 1e-4) / (g.true + 1e-4)).to_numpy()}

def oof(model_fn, target, kind):
    out = np.zeros(len(X))
    for k in range(5):
        tr, te = fold != k, fold == k
        m = model_fn().fit(X[tr], target[tr])
        out[te] = m.predict_proba(X[te])[:, 1] if kind == "clf" else m.predict(X[te])
    return out

print("\nOUT-OF-FOLD PREDICTABILITY (5 county folds).  chance: AUC 0.50, precision@13 ~0.7, R2 0")
print(f"  {'target':28} {'learner':22} {'AUC':>6} {'hits@13':>8} {'R2':>7}")
for name, tgt in T.items():
    is_clf = tgt.dtype == bool
    learners = [("HistGB", lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)),
                ("RandomForest", lambda: RandomForestClassifier(500, min_samples_leaf=3, random_state=0)),
                ("Logistic (scaled)", lambda: __import__("sklearn.pipeline", fromlist=["make_pipeline"]).make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)))] if is_clf else \
               [("HistGB", lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05)),
                ("Ridge (scaled)", lambda: __import__("sklearn.pipeline", fromlist=["make_pipeline"]).make_pipeline(StandardScaler(), Ridge(alpha=10.0)))]
    for lname, fn in learners:
        s = oof(fn, tgt.astype(int) if is_clf else tgt, "clf" if is_clf else "reg")
        if is_clf:
            auc = roc_auc_score(tgt, s); hits = int(tgt[np.argsort(-s)[:13]].sum())
            print(f"  {name:28} {lname:22} {auc:6.3f} {hits:8d}")
        else:
            print(f"  {name:28} {lname:22} {'':>6} {'':>8} {r2_score(tgt, s):7.3f}")
    if name == "top13_severity":
        sev_score = oof(lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05), T["log_total_osi"], "reg")

# ------------------------------------------------------------- unsupervised clustering
print("\nUNSUPERVISED: do the worst-13 fall into one cluster?  (k-means on standardised features; enrichment vs chance)")
Z = StandardScaler().fit_transform(X); rng = np.random.default_rng(0)
w = np.array([c in worst13 for c in X.index])
for k in (4, 6, 8, 12):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
    best = max(range(k), key=lambda c: w[lab == c].sum())
    n_c, hit = int((lab == best).sum()), int(w[lab == best].sum())
    perm = [max(np.bincount(lab[rng.permutation(len(w))][w], minlength=k)) for _ in range(2000)]
    print(f"  k={k:2d}: best cluster holds {hit}/13 of the worst counties in {n_c} counties ({100*n_c/len(w):.0f}% of all); "
          f"P(>= {hit} by chance) = {np.mean(np.array(perm) >= hit):.2f}")

# ------------------------------------------------------------- in-sample rule (illustration only)
print("\nIN-SAMPLE separating rule (depth-3 tree on all 239 counties) -- what a 'combination' looks like; NOT evidence:")
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=0).fit(X, w.astype(int))
print("\n".join("   " + l for l in export_text(tree, feature_names=list(X.columns), max_depth=3).splitlines()[:16]))
print(f"   in-sample hits: {int(w[tree.predict(X) == 1].sum())}/13 flagged, {int((tree.predict(X) == 1).sum())} counties flagged")

# ------------------------------------------------------------- does the OOF severity score convert to RMSE?
print("\nNESTED MULTIPLIER keyed on the out-of-fold severity score (top decile of score gets x c, c chosen on four folds)")
d["fold"] = d.f.map(fold_of); d["score"] = d.f.map(pd.Series(sev_score, index=X.index))
HZ = (1, 6, 24, 48); grid = np.arange(0.8, 2.01, 0.05)
def mr(pred, m): return float(np.mean([np.sqrt(((pred - d.y) ** 2)[m & (d.h >= 72 + k)].mean()) for k in HZ]))
base = mr(d.p, pd.Series(True, index=d.index)); num = {k: [0.0, 0] for k in HZ}; picks = []
for k in range(5):
    tr, te = d.fold != k, d.fold == k
    thr = np.quantile(pd.Series(sev_score, index=X.index)[fold != k], 0.9); hi = d.score >= thr
    c = min(grid, key=lambda c: mr(np.where(hi, c * d.p, d.p), tr)); picks.append(round(float(c), 2)); adj = np.where(hi, c * d.p, d.p)
    for kk in HZ:
        s = te & (d.h >= 72 + kk); num[kk][0] += ((adj - d.y) ** 2)[s].sum(); num[kk][1] += s.sum()
nested = float(np.mean([np.sqrt(v[0] / v[1]) for v in num.values()]))
print(f"  picks {picks}   nested {nested:.6f} vs base {base:.6f} -> {nested - base:+.6f}")
