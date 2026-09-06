"""Optuna TPE over the CLUSTER-LOCAL XGBoost hyperparameters.

The cluster-local test used the pooled model's hyperparameters (600 trees,
depth 6, min_child_weight 5), which are very likely wrong for fits that see
only ~1/3 of the counties: less data wants more regularisation, fewer trees,
shallower depth. Tuning them on their own terms is the fair completion of the
experiment before the direction is closed.

Protocol identical to the other tuning runs: trials scored by the TRUE
competition metric on an inner 3-fold county-holdout of fold-0's outer-training
counties (same seed-42 stratified machinery), warm-started at the pooled
defaults. Clusters are rebuilt inside each inner fold from weather+geography.

Run:  python3 optuna_tune_cluster.py [n_trials]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
ROOT = os.path.dirname(os.path.abspath(__file__))
K = 3
DEFAULT = {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6,
           "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
           "reg_lambda": 1.0}


def main():
    import optuna
    import xgboost as xgb
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from src.data import load_train, mask_after_freeze, osi_trajectory
    from src.features import feature_cols
    from src.ideas import _out
    from src.ideas6 import _rows
    from src.ideas10 import add_indep
    from src.validate import make_folds, score_trajectory

    DROP = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
            "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
    train = load_train()
    truth = osi_trajectory(train)
    tr_f0, _ = next(iter(make_folds(train)))
    inner_train = train[train.fipsCode.isin(tr_f0)]
    inner_folds = list(make_folds(inner_train, n_splits=3))

    piv = lambda c: (train.pivot_table(index="fipsCode", columns="hour", values=c)
                     .reindex(columns=np.arange(216)))
    g = np.nan_to_num(piv("gust").to_numpy())
    tp = np.nan_to_num(piv("tp").to_numpy())
    st = pd.read_csv("data_static/county_static.csv", dtype={"fipsCode": str})
    st["fipsCode"] = st.fipsCode.astype(int); st = st.set_index("fipsCode")
    fips_all = piv("gust").index.to_numpy()
    ll = st.reindex(fips_all)[["lat", "lon"]].to_numpy()
    X = np.column_stack([g.max(1), g.mean(1), np.clip(g - 35, 0, None).sum(1),
                         tp.sum(1), g[:, 48:96].max(1), g[:, 96:144].max(1),
                         g[:, 144:216].max(1), ll])
    Z = StandardScaler().fit_transform(np.nan_to_num(X))
    IDX = {f: i for i, f in enumerate(fips_all)}

    # prebuild the inner folds' row frames + cluster labels once
    prepped = []
    for tr_fips, va_fips in inner_folds:
        trd = inner_train[inner_train.fipsCode.isin(tr_fips)]
        vad = mask_after_freeze(inner_train[inner_train.fipsCode.isin(va_fips)]).drop(
            columns=[c for c in ("severity_tier", "peak_pct", "peak_customers",
                                 "time_to_restore_h", "split") if c in train.columns])
        tr, y = _rows(trd, statics=True, sub=True)
        va = _rows(vad, statics=True, sub=True, val=True)
        tr = add_indep(tr, mask_after_freeze(trd)); va = add_indep(va, vad)
        tr = tr.drop(columns=[c for c in DROP if c in tr.columns])
        va = va.drop(columns=[c for c in DROP if c in va.columns])
        cols = feature_cols(tr); keep = np.isfinite(y)
        km = KMeans(K, n_init=10, random_state=42).fit(Z[[IDX[f] for f in tr_fips]])
        lab_tr = dict(zip(tr_fips, km.labels_))
        lab_va = dict(zip(va_fips, km.predict(Z[[IDX[f] for f in va_fips]])))
        ctr = np.array([lab_tr[f] for f in tr.fipsCode.to_numpy()])
        cva = np.array([lab_va[f] for f in va.fipsCode.to_numpy()])
        prepped.append((tr, y, va, cols, keep, ctr, cva))
    print(f"inner protocol ready: {len(prepped)} folds, K={K} clusters", flush=True)

    def score_cfg(cfg, clustered=True):
        out = []
        for tr, y, va, cols, keep, ctr, cva in prepped:
            if clustered:
                parts = []
                for c in range(K):
                    s_tr = (ctr == c) & keep
                    s_va = cva == c
                    if s_va.sum() == 0:
                        continue
                    fit_on = s_tr if s_tr.sum() >= 1500 else keep
                    m = xgb.XGBRegressor(**cfg, random_state=42, n_jobs=4,
                                         tree_method="hist")
                    m.fit(tr.loc[fit_on, cols], np.sqrt(y[fit_on]))
                    parts.append(_out(va[s_va],
                                      np.clip(m.predict(va.loc[s_va, cols]), 0, None) ** 2))
                p = pd.concat(parts)
            else:
                m = xgb.XGBRegressor(**cfg, random_state=42, n_jobs=4,
                                     tree_method="hist")
                m.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
                p = _out(va, np.clip(m.predict(va[cols]), 0, None) ** 2)
            out.append(float(score_trajectory(p, truth).loc["mean", "rmse"]))
        return float(np.mean(out))

    def objective(trial):
        cfg = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 60),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        }
        return score_cfg(cfg, clustered=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42,
                                                                   n_startup_trials=8))
    study.enqueue_trial(DEFAULT)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    rows = [{**t.params, "metric": t.value} for t in study.trials if t.value is not None]
    pd.DataFrame(rows).sort_values("metric").to_csv(
        os.path.join(ROOT, "results", "optuna_tune_cluster.csv"), index=False)
    print("\n=== top cluster-local configs (inner true metric) ===")
    print(pd.DataFrame(rows).sort_values("metric").head(6).to_string(index=False))
    print(f"\nBEST cluster-local: {study.best_params}\n  inner {study.best_value:.5f}")
    print(f"DEFAULT cluster-local (warm start): {study.trials[0].value:.5f}")
    print(f"POOLED with default cfg, same inner protocol: "
          f"{score_cfg(DEFAULT, clustered=False):.5f}")
    print(f"POOLED with the tuned cfg, same inner protocol: "
          f"{score_cfg(study.best_params, clustered=False):.5f}")


if __name__ == "__main__":
    main()
