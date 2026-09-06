"""Sweep 13 (FS-series) - the feature-selection matrix on the tabular member.

User-specified design: the tabular pipeline at two feature-engineering levels
x five selection techniques, plus the two no-selection controls.

  A  base features, no selection            (control; ~50 cols)
  B  base + 5 selection techniques
  C  full FE (statics + sub + interactions = f2), no selection  (control)
  D  full FE + the same 5 techniques

Selection techniques (each fitted STRICTLY on the training fold - the
selected column set never sees the held-out counties):
  corr    filter: top 40 by |Spearman corr| with sqrt(y)
  gain    embedded: top 40 by XGBoost gain from a 200-tree preliminary fit
  perm    permutation importance on an inner 20% county holdout; keep > 0
  lasso   L1: LassoCV on standardized features; nonzero coefs (top 40 floor)
  shadow  Boruta-lite: append shuffled copies of every feature; keep features
          whose gain beats the best shadow's gain

Sequence members are out of scope by design: they consume a fixed 8-channel
encoder / 20-channel decoder, and law 3 (three confirmations) already
establishes their input set is at its minimum. Prior art for the tabular
model: S27 top-40 selection measured exactly neutral.
"""
import numpy as np
import pandas as pd

from .data import mask_after_freeze
from .features import feature_cols
from .ideas import _out
from .models import SEED

XGB_KW = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
              random_state=SEED, n_jobs=4, tree_method="hist")
PRE_KW = dict(XGB_KW, n_estimators=200)


def _build(train_full, val_masked, fe):
    from .ideas6 import _rows
    if fe == "base":
        tr, y = _rows(train_full, statics=False, sub=False)
        va = _rows(val_masked, statics=False, sub=False, val=True)
    else:                                    # "full" = the shipped f2 recipe
        from .ideas10 import add_indep
        tr, y = _rows(train_full, statics=True, sub=True)
        va = _rows(val_masked, statics=True, sub=True, val=True)
        tr = add_indep(tr, mask_after_freeze(train_full))
        va = add_indep(va, val_masked)
        drop = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
                "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
        tr = tr.drop(columns=[c for c in drop if c in tr.columns])
        va = va.drop(columns=[c for c in drop if c in va.columns])
    return tr, y, va


def _select(tr, y, cols, method, rng):
    """Column subset chosen from the training fold only."""
    import xgboost as xgb
    X = tr[cols].to_numpy()
    ys = np.sqrt(y)
    if method == "corr":
        r = [abs(pd.Series(X[:, j]).corr(pd.Series(ys), method="spearman"))
             for j in range(len(cols))]
        order = np.argsort(np.nan_to_num(r))[::-1]
        return [cols[j] for j in order[:40]]
    if method == "gain":
        m = xgb.XGBRegressor(**PRE_KW).fit(X, ys)
        imp = m.feature_importances_
        return [cols[j] for j in np.argsort(imp)[::-1][:40]]
    if method == "perm":
        from sklearn.inspection import permutation_importance
        fips = tr["fipsCode"].to_numpy()
        uf = np.unique(fips)
        hold = set(rng.permutation(uf)[:max(8, len(uf) // 5)].tolist())
        va_m = np.isin(fips, list(hold))
        m = xgb.XGBRegressor(**PRE_KW).fit(X[~va_m], ys[~va_m])
        pi = permutation_importance(m, X[va_m], ys[va_m], n_repeats=3,
                                    random_state=SEED, n_jobs=4)
        keep = [cols[j] for j in range(len(cols)) if pi.importances_mean[j] > 0]
        return keep if len(keep) >= 10 else [cols[j] for j in
                                             np.argsort(pi.importances_mean)[::-1][:40]]
    if method == "lasso":
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(np.nan_to_num(X))
        m = LassoCV(cv=3, n_alphas=30, random_state=SEED, n_jobs=4,
                    max_iter=3000).fit(Xs, ys)
        keep = [cols[j] for j in range(len(cols)) if abs(m.coef_[j]) > 1e-10]
        if len(keep) < 10:
            keep = [cols[j] for j in np.argsort(np.abs(m.coef_))[::-1][:40]]
        return keep
    if method == "shadow":
        Xsh = np.concatenate([X] + [rng.permutation(X[:, j])[:, None]
                                    for j in range(len(cols))], axis=1)
        m = xgb.XGBRegressor(**PRE_KW).fit(Xsh, ys)
        imp = m.feature_importances_
        thresh = imp[len(cols):].max()
        keep = [cols[j] for j in range(len(cols)) if imp[j] > thresh]
        return keep if len(keep) >= 10 else [cols[j] for j in
                                             np.argsort(imp[:len(cols)])[::-1][:40]]
    raise ValueError(method)


def _fs_forecaster(train_full, val_masked, fe, method):
    import xgboost as xgb
    rng = np.random.RandomState(SEED)
    tr, y, va = _build(train_full, val_masked, fe)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    sel = cols if method is None else _select(tr.loc[keep], y[keep], cols,
                                              method, rng)
    mdl = xgb.XGBRegressor(**XGB_KW)
    mdl.fit(tr.loc[keep, sel], np.sqrt(y[keep]))
    out = _out(va, np.clip(mdl.predict(va[sel]), 0, None) ** 2)
    return out


def _mk(fe, method):
    return lambda tr, va: _fs_forecaster(tr, va, fe, method)


IDEAS11 = {
    "FS0 base features, no selection (control)": _mk("base", None),
    "FS1a base + corr filter": _mk("base", "corr"),
    "FS1b base + gain top-40": _mk("base", "gain"),
    "FS1c base + permutation importance": _mk("base", "perm"),
    "FS1d base + lasso": _mk("base", "lasso"),
    "FS1e base + shadow (Boruta-lite)": _mk("base", "shadow"),
    "FS2a full FE + corr filter": _mk("full", "corr"),
    "FS2b full FE + gain top-40": _mk("full", "gain"),
    "FS2c full FE + permutation importance": _mk("full", "perm"),
    "FS2d full FE + lasso": _mk("full", "lasso"),
    "FS2e full FE + shadow (Boruta-lite)": _mk("full", "shadow"),
}
