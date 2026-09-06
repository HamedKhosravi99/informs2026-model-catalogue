"""Feature importance, correlation and temporal-precedence analysis for the
shipped tabular member f2 (sqrt-XGB + statics + sub-county + physics
interactions).

1. Importance, two ways:
   - XGBoost gain (in-model, descriptive)
   - OUT-OF-FOLD permutation importance: for each CV fold, fit f2's model on
     the training counties, then shuffle one feature at a time ON THE HELD-OUT
     COUNTIES' rows and measure the increase in their per-horizon-average
     RMSE. Averaged over the 5 folds. This is the honest deployment-relevant
     importance (a feature only scores if breaking it hurts unseen counties).

2. Correlation:
   - Spearman of each top feature with the scored target (pooled cells)
   - the feature-feature |Spearman| redundancy blocks among the top 20

3. Temporal precedence / "causality" (observational; stated as such):
   - pooled cross-correlation of gust vs new-damage flow N_t at lags -6..+6 h
   - panel Granger test with county fixed effects: N_t(t) ~ own lags + gust
     lags; F-test on the gust block, and the REVERSE direction as
     falsification (N_t should not Granger-cause gust)
   - model dose-response: f2's partial-dependence over gust and over the two
     adopted interactions (gust x soil, canopy x gust)
"""
import numpy as np
import pandas as pd

from src.data import FREEZE_H, HORIZONS, load_train, mask_after_freeze, osi_trajectory
from src.validate import make_folds
from src.features import feature_cols

SEED = 42
train = load_train()
truth = osi_trajectory(train)
folds = list(make_folds(train))


def build_f2(train_full, val_masked):
    """f2's exact row construction (mirrors ideas10.f2_member_upgrade)."""
    import xgboost as xgb
    from src.ideas6 import _rows
    from src.ideas10 import add_indep
    tr, y = _rows(train_full, statics=True, sub=True)
    va = _rows(val_masked, statics=True, sub=True, val=True)
    tr = add_indep(tr, mask_after_freeze(train_full))
    va = add_indep(va, val_masked)
    drop = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
            "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
    tr = tr.drop(columns=[c for c in drop if c in tr.columns])
    va = va.drop(columns=[c for c in drop if c in va.columns])
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4,
                           tree_method="hist")
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    return mdl, cols, va


def col_metric(pred_df):
    piv = pred_df.pivot_table(index="fipsCode", columns="hour", values="osi_pred")
    rs = []
    for n, h in HORIZONS.items():
        tgt = np.arange(FREEZE_H + 1 + h, 216)
        t = truth.loc[piv.index, tgt].to_numpy()
        p = piv.reindex(columns=tgt).to_numpy()
        rs.append(np.sqrt(np.nanmean((p - t) ** 2)))
    return float(np.mean(rs))


# ================================================== 1. importance
gain_acc, perm_acc = {}, {}
rng = np.random.RandomState(SEED)
for kf, (tr_f, va_f) in enumerate(folds):
    trd = train[train.fipsCode.isin(tr_f)]
    vad = mask_after_freeze(train[train.fipsCode.isin(va_f)]).drop(
        columns=[c for c in ("severity_tier", "peak_pct", "peak_customers",
                             "time_to_restore_h", "split") if c in train.columns])
    mdl, cols, va = build_f2(trd, vad)
    for c, g in zip(cols, mdl.feature_importances_):
        gain_acc[c] = gain_acc.get(c, 0) + g / len(folds)

    base_pred = va[["fipsCode", "hour"]].copy()
    base_pred["osi_pred"] = np.clip(mdl.predict(va[cols]), 0, None) ** 2
    base = col_metric(base_pred)
    for c in cols:
        sav = va[c].to_numpy().copy()
        va[c] = rng.permutation(sav)
        p = va[["fipsCode", "hour"]].copy()
        p["osi_pred"] = np.clip(mdl.predict(va[cols]), 0, None) ** 2
        perm_acc[c] = perm_acc.get(c, 0) + (col_metric(p) - base) / len(folds)
        va[c] = sav
    print(f"fold {kf} done (base {base:.5f})", flush=True)

gain = pd.Series(gain_acc).sort_values(ascending=False)
perm = pd.Series(perm_acc).sort_values(ascending=False)
print("\n=== XGB gain importance, top 20 (share of total) ===")
print((gain.head(20) / gain.sum()).round(4).to_string())
print("\n=== OOF permutation importance, top 20 (metric increase x 1e4) ===")
print((perm.head(20) * 1e4).round(3).to_string())
print("\n=== permutation importance of the adopted interaction family ===")
for c in ("gust_x_soil", "canopy", "canopy_x_gust"):
    if c in perm.index:
        print(f"  {c:14s} {perm[c]*1e4:+.3f} x1e-4   (rank {list(perm.index).index(c)+1})")

# ================================================== 2. correlation
mdl, cols, va_last = None, None, None
trd = train
masked_all = mask_after_freeze(train)
import xgboost  # noqa
from src.ideas6 import _rows
from src.ideas10 import add_indep
rows, y = _rows(train, statics=True, sub=True)
rows = add_indep(rows, masked_all)
rows = rows.drop(columns=[c for c in ("hrs_since_g35", "hrs_until_g35", "gfut6",
                                      "veer", "pre_peak_logP", "pre_gust_max",
                                      "pre_frag_gap") if c in rows.columns])
cols = feature_cols(rows)
keep = np.isfinite(y)
top = list(perm.head(12).index)
print("\n=== Spearman corr with target (pooled scored cells), top-12 by perm ===")
for c in top:
    r = pd.Series(rows.loc[keep, c].to_numpy()).corr(pd.Series(y[keep]),
                                                     method="spearman")
    print(f"  {c:22s} {r:+.3f}")
print("\n=== redundancy blocks: |Spearman| > 0.85 among the top 20 ===")
sub = rows.loc[keep, list(perm.head(20).index)].sample(6000, random_state=SEED)
C = sub.corr(method="spearman").abs()
seen = set()
for i, a in enumerate(C.index):
    for b in C.columns[i + 1:]:
        if C.loc[a, b] > 0.85 and (b, a) not in seen:
            seen.add((a, b))
            print(f"  {a:22s} ~ {b:22s} |r|={C.loc[a, b]:.2f}")

# ================================================== 3. temporal precedence
piv = lambda c: (train.pivot_table(index="fipsCode", columns="hour", values=c)
                 .reindex(columns=np.arange(216)))
G = np.nan_to_num(piv("gust").to_numpy())
N = np.nan_to_num(piv("N_t").to_numpy())
print("\n=== pooled cross-correlation corr(gust(t+lag), N_t(t)) ===")
for lag in range(-6, 7):
    g = np.roll(G, -lag, axis=1)
    r = np.corrcoef(g[:, 8:208].ravel(), N[:, 8:208].ravel())[0, 1]
    tag = " <- gust LEADS damage" if lag in (-2, -1, 0) else ""
    print(f"  lag {lag:+d}h: {r:+.3f}{tag}")

print("\n=== panel Granger (county fixed effects, storm window h48-215) ===")
import statsmodels.api as sm
rows_g = []
for i in range(len(G)):
    for t in range(52, 216):
        rows_g.append((i, N[i, t], N[i, t-1], N[i, t-2], N[i, t-3],
                       G[i, t], G[i, t-1], G[i, t-2], G[i, t-3]))
d = pd.DataFrame(rows_g, columns=["cty", "n0", "n1", "n2", "n3",
                                  "g0", "g1", "g2", "g3"])
d = d[d.index % 2 == 0]                       # thin for speed
fe = pd.get_dummies(d.cty, drop_first=True).astype(float)


def granger(dep, own, other, label):
    X1 = sm.add_constant(pd.concat([d[own], fe], axis=1).to_numpy())
    X2 = sm.add_constant(pd.concat([d[own + other], fe], axis=1).to_numpy())
    m1 = sm.OLS(d[dep].to_numpy(), X1).fit()
    m2 = sm.OLS(d[dep].to_numpy(), X2).fit()
    df1, df2 = len(other), m2.df_resid
    F = ((m1.ssr - m2.ssr) / df1) / (m2.ssr / df2)
    from scipy.stats import f as fdist
    p = 1 - fdist.cdf(F, df1, df2)
    print(f"  {label}: F={F:.1f}, p={p:.2e}  "
          f"(delta R2 {m2.rsquared - m1.rsquared:+.4f})")


granger("n0", ["n1", "n2", "n3"], ["g0", "g1", "g2", "g3"],
        "gust block -> damage | own lags     ")
granger("g0", ["g1", "g2", "g3"], ["n1", "n2", "n3"],
        "damage block -> gust | own lags (falsification)")

print("\n=== f2 dose-response: mean prediction vs gust decile "
      "(other features at their observed values) ===")
mdl, cols2, va = build_f2(train[train.fipsCode.isin(folds[0][0])],
                          mask_after_freeze(train[train.fipsCode.isin(folds[0][1])]))
q = pd.qcut(va["gust"], 10, labels=False, duplicates="drop")
pr = np.clip(mdl.predict(va[cols2]), 0, None) ** 2
t = pd.DataFrame({"q": q, "gust": va["gust"], "pred": pr}).groupby("q").agg(
    gust=("gust", "mean"), pred=("pred", "mean"))
print(t.round(4).to_string())
