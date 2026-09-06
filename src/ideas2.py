"""Sweep 2: broader model families, feature/target engineering, and creative variants.

Same contract as src/ideas.py: f(train_full, val_masked) -> (fipsCode, hour, osi_pred)
over hours 73-215, evaluated with the identical county-holdout CV.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .features import build_rows, feature_cols
from .models import make_gbm, SEED
from .ideas import add_static, add_physics, _out, _val_rows, _rows_and_y

# ------------------------------------------------------------------ shared: rich rows
def rich(df, val=False):
    """All engineered features + static enrichment + physics indices."""
    if val:
        return _val_rows(df, static=True, physics=True)
    return _rows_and_y(df, static=True, physics=True)


# =================================================================== S21 XGBoost
def xgb_forecaster(train_full, val_masked, **kw):
    import xgboost as xgb
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    params = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                  random_state=SEED, n_jobs=4, tree_method="hist")
    params.update(kw)
    mdl = xgb.XGBRegressor(**params).fit(tr.loc[keep, cols], y[keep])
    va = rich(val_masked, val=True)
    return _out(va, mdl.predict(va[cols]))


def xgb_tweedie_forecaster(train_full, val_masked):
    return xgb_forecaster(train_full, val_masked,
                          objective="reg:tweedie", tweedie_variance_power=1.5)


# ================================================================== S22 LightGBM
def lgbm_forecaster(train_full, val_masked, **kw):
    import lightgbm as lgb
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    params = dict(n_estimators=800, learning_rate=0.05, num_leaves=31, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, min_child_samples=30,
                  reg_lambda=1.0, random_state=SEED, n_jobs=4, verbose=-1)
    params.update(kw)
    mdl = lgb.LGBMRegressor(**params).fit(tr.loc[keep, cols], y[keep])
    va = rich(val_masked, val=True)
    return _out(va, mdl.predict(va[cols]))


def lgbm_tweedie_forecaster(train_full, val_masked):
    return lgbm_forecaster(train_full, val_masked, objective="tweedie",
                           tweedie_variance_power=1.5)


def lgbm_dart_forecaster(train_full, val_masked):
    return lgbm_forecaster(train_full, val_masked, boosting_type="dart",
                           n_estimators=500, drop_rate=0.1)


# ============================================================ S23 forests / others
def random_forest_forecaster(train_full, val_masked):
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = make_pipeline(SimpleImputer(strategy="median"),
                        RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                              max_features=0.4, n_jobs=4,
                                              random_state=SEED))
    mdl.fit(tr.loc[keep, cols], y[keep])
    va = rich(val_masked, val=True)
    return _out(va, mdl.predict(va[cols]))


def extratrees_forecaster(train_full, val_masked):
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = make_pipeline(SimpleImputer(strategy="median"),
                        ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                            max_features=0.5, n_jobs=4,
                                            random_state=SEED))
    mdl.fit(tr.loc[keep, cols], y[keep])
    va = rich(val_masked, val=True)
    return _out(va, mdl.predict(va[cols]))


def bayesian_ridge_forecaster(train_full, val_masked):
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        BayesianRidge())
    mdl.fit(tr.loc[keep, cols], y[keep])
    va = rich(val_masked, val=True)
    return _out(va, mdl.predict(va[cols]))


def elasticnet_forecaster(train_full, val_masked):
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    sub = np.random.RandomState(SEED).choice(np.where(keep)[0], min(20000, keep.sum()),
                                             replace=False)
    mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=SEED,
                                     max_iter=5000))
    mdl.fit(tr.iloc[sub][cols], y[sub])
    va = rich(val_masked, val=True)
    return _out(va, mdl.predict(va[cols]))


# ===================================================== S24 target transformations
def _transformed_gbm(train_full, val_masked, fwd, inv):
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = make_gbm().fit(tr.loc[keep, cols], fwd(y[keep]))
    va = rich(val_masked, val=True)
    return _out(va, inv(mdl.predict(va[cols])))


def log_target_forecaster(train_full, val_masked):
    return _transformed_gbm(train_full, val_masked,
                            lambda y: np.log1p(y * 100), lambda p: np.expm1(p) / 100)


def sqrt_target_forecaster(train_full, val_masked):
    return _transformed_gbm(train_full, val_masked,
                            np.sqrt, lambda p: np.clip(p, 0, None) ** 2)


# ============================================ S25 distributional regression by binning
def binned_classification_forecaster(train_full, val_masked, n_bins=24):
    """Bin the skewed target, classify, take the expectation over bin centers.
    A standard competition trick for heavy-tailed targets."""
    import lightgbm as lgb
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    yk = y[keep]
    edges = np.unique(np.quantile(yk[yk > 0], np.linspace(0, 1, n_bins)))
    edges = np.concatenate([[-1e-9, 1e-9], edges[1:]])
    lab = np.digitize(yk, edges) - 1
    centers = np.array([yk[lab == c].mean() if (lab == c).sum() else 0.0
                        for c in range(lab.max() + 1)])
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.06, num_leaves=31,
                             min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=SEED, n_jobs=4,
                             verbose=-1).fit(tr.loc[keep, cols], lab)
    va = rich(val_masked, val=True)
    P = mdl.predict_proba(va[cols])
    cls = np.array(mdl.classes_)
    return _out(va, P @ centers[cls])


# ================================================== S26 per-county target normalization
def county_normalized_forecaster(train_full, val_masked):
    """Predict OSI relative to the county's observed-window peak, then rescale."""
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    scale_tr = (train_full[train_full["hour"] <= FREEZE_H]
                .groupby("fipsCode")["osi"].max().clip(lower=1e-4))
    s = tr["fipsCode"].map(scale_tr).to_numpy()
    keep = np.isfinite(y)
    mdl = make_gbm().fit(tr.loc[keep, cols], (y[keep] / s[keep]))
    va = rich(val_masked, val=True)
    scale_va = (val_masked[val_masked["hour"] <= FREEZE_H]
                .groupby("fipsCode")["osi"].max().clip(lower=1e-4))
    return _out(va, mdl.predict(va[cols]) * va["fipsCode"].map(scale_va).to_numpy())


# ======================================================== S27 feature selection (top-k)
def feature_selection_forecaster(train_full, val_masked, k=40):
    import lightgbm as lgb
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    probe = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.08, num_leaves=31,
                              random_state=SEED, n_jobs=4, verbose=-1
                              ).fit(tr.loc[keep, cols], y[keep])
    top = list(pd.Series(probe.feature_importances_, index=cols).nlargest(k).index)
    mdl = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            min_child_samples=30, random_state=SEED, n_jobs=4,
                            verbose=-1).fit(tr.loc[keep, top], y[keep])
    va = rich(val_masked, val=True)
    return _out(va, mdl.predict(va[top]))


# ================================================= S28 quantile-transformed features
def quantile_features_forecaster(train_full, val_masked):
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                             random_state=SEED, subsample=100000)
    Xtr = qt.fit_transform(SimpleImputer(strategy="median").fit_transform(tr.loc[keep, cols]))
    mdl = make_gbm().fit(Xtr, y[keep])
    va = rich(val_masked, val=True)
    Xva = qt.transform(SimpleImputer(strategy="median").fit_transform(va[cols]))
    return _out(va, mdl.predict(Xva))


# ============================== S29 combining the two winners: sqrt target x model class
def _sqrt_fit(train_full, val_masked, make_model):
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = make_model()
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    va = rich(val_masked, val=True)
    return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)


def sqrt_extratrees_forecaster(train_full, val_masked):
    return _sqrt_fit(train_full, val_masked, lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                            n_jobs=4, random_state=SEED)))


def sqrt_rf_forecaster(train_full, val_masked):
    return _sqrt_fit(train_full, val_masked, lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.4,
                              n_jobs=4, random_state=SEED)))


def sqrt_lgbm_forecaster(train_full, val_masked):
    import lightgbm as lgb
    return _sqrt_fit(train_full, val_masked, lambda: lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=31, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, min_child_samples=30,
        reg_lambda=1.0, random_state=SEED, n_jobs=4, verbose=-1))


def sqrt_xgb_forecaster(train_full, val_masked):
    import xgboost as xgb
    return _sqrt_fit(train_full, val_masked, lambda: xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        random_state=SEED, n_jobs=4, tree_method="hist"))


def sqrt_bagged_trees_forecaster(train_full, val_masked, n_bags=6):
    """ExtraTrees on sqrt target, bagged over county subsamples (double variance
    reduction: extreme randomization + county-level bootstrap)."""
    tr, y = rich(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    fips = train_full["fipsCode"].unique()
    rng = np.random.RandomState(SEED)
    va = rich(val_masked, val=True)
    preds = []
    for b in range(n_bags):
        sub = rng.choice(fips, int(0.8 * len(fips)), replace=False)
        m = keep & tr["fipsCode"].isin(sub).to_numpy()
        mdl = make_pipeline(SimpleImputer(strategy="median"),
                            ExtraTreesRegressor(n_estimators=250, min_samples_leaf=5,
                                                max_features=0.5, n_jobs=4,
                                                random_state=SEED + b))
        mdl.fit(tr.loc[m, cols], np.sqrt(y[m]))
        preds.append(np.clip(mdl.predict(va[cols]), 0, None) ** 2)
    return _out(va, np.mean(preds, 0))


IDEAS2 = {
    "S29 sqrt + ExtraTrees": sqrt_extratrees_forecaster,
    "S29b sqrt + RandomForest": sqrt_rf_forecaster,
    "S29c sqrt + LightGBM": sqrt_lgbm_forecaster,
    "S29d sqrt + XGBoost": sqrt_xgb_forecaster,
    "S29e sqrt + bagged ExtraTrees": sqrt_bagged_trees_forecaster,
    "S21 XGBoost": xgb_forecaster,
    "S21b XGBoost Tweedie": xgb_tweedie_forecaster,
    "S22 LightGBM": lgbm_forecaster,
    "S22b LightGBM Tweedie": lgbm_tweedie_forecaster,
    "S22c LightGBM DART": lgbm_dart_forecaster,
    "S23 Random Forest": random_forest_forecaster,
    "S23b ExtraTrees": extratrees_forecaster,
    "S23c Bayesian Ridge": bayesian_ridge_forecaster,
    "S23d ElasticNet": elasticnet_forecaster,
    "S24 log1p target": log_target_forecaster,
    "S24b sqrt target": sqrt_target_forecaster,
    "S25 binned classification": binned_classification_forecaster,
    "S26 county-normalized target": county_normalized_forecaster,
    "S27 feature selection top-40": feature_selection_forecaster,
    "S28 quantile-transformed features": quantile_features_forecaster,
}


# ===================================== S45 sub-county weather heterogeneity features
def _rich_sub(df, val=False):
    from .subcounty import add_subcounty
    if val:
        return add_subcounty(_val_rows(df, static=True, physics=True), df)
    rows, y = _rows_and_y(df, static=True, physics=True)
    return add_subcounty(rows, df), y


def subcounty_gbm_forecaster(train_full, val_masked):
    """Best tabular recipe (sqrt target + XGBoost) plus within-county gust spread."""
    import xgboost as xgb
    tr, y = _rich_sub(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4,
                           tree_method="hist")
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    va = _rich_sub(val_masked, val=True)
    return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)


IDEAS2["S45 sub-county gust heterogeneity"] = subcounty_gbm_forecaster
