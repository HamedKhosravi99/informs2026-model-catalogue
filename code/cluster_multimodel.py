"""Cluster-local modelling, done properly: several model families per cluster,
ensembled, then blended with the pooled model.

The first pass tested one family (sqrt-XGB) inside K-means clusters of
weather+geography and lost monotonically with K. That is not a fair test of the
idea: the whole point of local models is that different regimes may want
different function classes, and local models are exactly the setting where
ensembling should pay most (each is fit on less data, so each is higher
variance).

This runs, per fold:
  - K-means clusters on weather+geography descriptors (train counties only;
    val counties assigned by predict, which is legal - weather and lat/lon are
    known for the test file)
  - inside each cluster: sqrt-XGB, ExtraTrees, RandomForest, sqrt-LightGBM,
    Ridge  (five families, the ones that reached the top tier in sweep 2)
  - the equal-weight ensemble of those five, per cluster
  - the pooled (non-clustered) counterpart of every row above
  - blends of the cluster-local ensemble with the pooled ensemble
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from src.data import load_train, mask_after_freeze, osi_trajectory
from src.features import feature_cols
from src.ideas import _out
from src.ideas6 import _rows
from src.ideas10 import add_indep
from src.validate import make_folds, score_trajectory

DROP = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
        "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
SEED = 42
K = 3

train = load_train()
truth = osi_trajectory(train)
folds = list(make_folds(train))


def descriptors():
    piv = lambda c: (train.pivot_table(index="fipsCode", columns="hour", values=c)
                     .reindex(columns=np.arange(216)))
    g = np.nan_to_num(piv("gust").to_numpy())
    tp = np.nan_to_num(piv("tp").to_numpy())
    st = pd.read_csv("data_static/county_static.csv", dtype={"fipsCode": str})
    st["fipsCode"] = st.fipsCode.astype(int)
    st = st.set_index("fipsCode")
    fips = piv("gust").index.to_numpy()
    ll = st.reindex(fips)[["lat", "lon"]].to_numpy()
    X = np.column_stack([g.max(1), g.mean(1), np.clip(g - 35, 0, None).sum(1),
                         tp.sum(1), g[:, 48:96].max(1), g[:, 96:144].max(1),
                         g[:, 144:216].max(1), ll])
    return fips, StandardScaler().fit_transform(np.nan_to_num(X))


FIPS, Z = descriptors()
IDX = {f: i for i, f in enumerate(FIPS)}


def make(name):
    import xgboost as xgb
    if name == "xgb":
        return xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                min_child_weight=5, reg_lambda=1.0,
                                random_state=SEED, n_jobs=4, tree_method="hist")
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8,
                                 min_child_samples=20, random_state=SEED,
                                 n_jobs=4, verbose=-1)
    if name == "et":
        return ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                   max_features=0.5, n_jobs=4, random_state=SEED)
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                     max_features=0.5, n_jobs=4, random_state=SEED)
    if name == "ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        # trees consume NaN natively; linear models do not
        return make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(), RidgeCV())
    raise ValueError(name)


FAMS = ["xgb", "lgbm", "et", "rf", "ridge"]


def build(trd, vad):
    tr, y = _rows(trd, statics=True, sub=True)
    va = _rows(vad, statics=True, sub=True, val=True)
    tr = add_indep(tr, mask_after_freeze(trd))
    va = add_indep(va, vad)
    tr = tr.drop(columns=[c for c in DROP if c in tr.columns])
    va = va.drop(columns=[c for c in DROP if c in va.columns])
    return tr, y, va, feature_cols(tr)


def run():
    store = {f"pooled_{n}": [] for n in FAMS}
    store.update({f"clust_{n}": [] for n in FAMS})
    for tr_f, va_f in folds:
        trd = train[train.fipsCode.isin(tr_f)]
        vad = mask_after_freeze(train[train.fipsCode.isin(va_f)]).drop(
            columns=[c for c in ("severity_tier", "peak_pct", "peak_customers",
                                 "time_to_restore_h", "split") if c in train.columns])
        tr, y, va, cols = build(trd, vad)
        keep = np.isfinite(y)
        trf, vaf = tr.fipsCode.to_numpy(), va.fipsCode.to_numpy()

        km = KMeans(K, n_init=10, random_state=SEED).fit(Z[[IDX[f] for f in tr_f]])
        lab_tr = dict(zip(tr_f, km.labels_))
        lab_va = dict(zip(va_f, km.predict(Z[[IDX[f] for f in va_f]])))
        ctr = np.array([lab_tr[f] for f in trf])
        cva = np.array([lab_va[f] for f in vaf])

        for n in FAMS:
            m = make(n).fit(tr.loc[keep, cols], np.sqrt(y[keep]))
            store[f"pooled_{n}"].append(
                _out(va, np.clip(m.predict(va[cols]), 0, None) ** 2))
            parts = []
            for c in range(K):
                s_tr = (ctr == c) & keep
                s_va = cva == c
                if s_va.sum() == 0:
                    continue
                fit_on = s_tr if s_tr.sum() >= 1500 else keep
                mc = make(n).fit(tr.loc[fit_on, cols], np.sqrt(y[fit_on]))
                parts.append(_out(va[s_va],
                                  np.clip(mc.predict(va.loc[s_va, cols]), 0, None) ** 2))
            store[f"clust_{n}"].append(pd.concat(parts))
        print(f"  fold done", flush=True)

    out = {k: pd.concat(v, ignore_index=True) for k, v in store.items()}
    for k, v in out.items():
        v.to_csv(f"oof/cm_{k}.csv", index=False)
    return out


def sc(df):
    return score_trajectory(df, truth).loc["mean", "rmse"]


res = run()
print("\n=== single families: pooled vs cluster-local (K=3) ===")
print(f"{'family':8s} {'pooled':>9s} {'clustered':>10s}  delta")
for n in FAMS:
    p, c = sc(res[f"pooled_{n}"]), sc(res[f"clust_{n}"])
    print(f"{n:8s} {p:9.5f} {c:10.5f}  {c-p:+.5f}")


def ens(keys):
    base = res[keys[0]][["fipsCode", "hour"]].copy()
    base["osi_pred"] = np.mean([res[k].osi_pred.to_numpy() for k in keys], 0)
    return base


pe = ens([f"pooled_{n}" for n in FAMS])
ce = ens([f"clust_{n}" for n in FAMS])
print(f"\n=== 5-family equal-weight ensembles ===")
print(f"pooled ensemble          {sc(pe):.5f}")
print(f"cluster-local ensemble   {sc(ce):.5f}")
print(f"\n=== blends of the two ensembles ===")
for lam in (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
    b = pe[["fipsCode", "hour"]].copy()
    b["osi_pred"] = (1 - lam) * pe.osi_pred.to_numpy() + lam * ce.osi_pred.to_numpy()
    print(f"  {1-lam:.2f}*pooled + {lam:.2f}*cluster-local : {sc(b):.5f}")
print("\nreference: shipped f2 member 0.00930; shipped full blend 0.008542")
