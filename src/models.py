"""Forecasters: baseline ladder + direct-trajectory gradient boosting.

Every forecaster has signature f(train_full, val_masked) -> DataFrame with
columns (fipsCode, hour, osi_pred) for hours 73-215. `val_masked` carries no
outage data after FREEZE_H, so causality violations are structurally impossible.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze, osi_trajectory
from .features import build_rows, feature_cols

SEED = 42


def _grid(val_masked: pd.DataFrame) -> pd.DataFrame:
    fips = val_masked["fipsCode"].unique()
    return pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips))})


def _osi_at_freeze(df: pd.DataFrame) -> pd.Series:
    at = df[df["hour"] == FREEZE_H].set_index("fipsCode")["osi"]
    return at


def zero_forecaster(train_full, val_masked):
    g = _grid(val_masked)
    g["osi_pred"] = 0.0
    return g


def persistence_forecaster(train_full, val_masked):
    g = _grid(val_masked)
    g["osi_pred"] = g["fipsCode"].map(_osi_at_freeze(val_masked)).astype(float)
    return g


def decay_forecaster(train_full, val_masked, taus=(6, 9, 12, 16, 24, 36, 48)):
    """Persistence with exponential decay; tau fit on the fold's training counties."""
    truth = osi_trajectory(train_full)
    o71 = truth[FREEZE_H].to_numpy()[:, None]
    tgt = truth.loc[:, PRED_HOURS[0]:].to_numpy()
    dt = (truth.loc[:, PRED_HOURS[0]:].columns.to_numpy() - FREEZE_H)[None, :]
    best_tau = min(taus, key=lambda t: np.sqrt(np.nanmean((o71 * np.exp(-dt / t) - tgt) ** 2)))
    g = _grid(val_masked)
    g["osi_pred"] = (g["fipsCode"].map(_osi_at_freeze(val_masked)).astype(float)
                     * np.exp(-(g["hour"] - FREEZE_H) / best_tau))
    g.attrs["tau"] = best_tau
    return g


def climatology_forecaster(train_full, val_masked):
    clim = train_full.groupby("hour")["osi"].mean()
    g = _grid(val_masked)
    g["osi_pred"] = g["hour"].map(clim).astype(float)
    return g


def scaled_climatology_forecaster(train_full, val_masked):
    clim = train_full.groupby("hour")["osi"].mean()
    obs = val_masked[val_masked["hour"] <= FREEZE_H]
    ref = train_full[train_full["hour"] <= FREEZE_H]["osi"].mean()
    scale = (obs.groupby("fipsCode")["osi"].mean() / ref).clip(0, 20)
    g = _grid(val_masked)
    g["osi_pred"] = g["hour"].map(clim).astype(float) * g["fipsCode"].map(scale).astype(float)
    return g


def make_gbm(**kw):
    params = dict(loss="squared_error", learning_rate=0.06, max_iter=600,
                  max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=40, random_state=SEED)
    params.update(kw)
    return HistGradientBoostingRegressor(**params)


def fit_gbm(train_full: pd.DataFrame):
    """Train the direct model on train counties under simulated test conditions."""
    tr_rows = build_rows(mask_after_freeze(train_full))
    y_lookup = train_full.set_index(["fipsCode", "hour"])["osi"]
    y = y_lookup.reindex(pd.MultiIndex.from_frame(tr_rows[["fipsCode", "hour"]])).to_numpy()
    keep = np.isfinite(y)
    cols = feature_cols(tr_rows)
    model = make_gbm()
    model.fit(tr_rows.loc[keep, cols], y[keep])
    return model, cols


def gbm_forecaster(train_full, val_masked):
    model, cols = fit_gbm(train_full)
    va_rows = build_rows(val_masked)
    g = va_rows[["fipsCode", "hour"]].copy()
    g["osi_pred"] = np.clip(model.predict(va_rows[cols]), 0.0, None)
    return g


HURDLE_THETA = 0.01  # "active outage" severity threshold


def fit_hurdle(train_full: pd.DataFrame):
    """Two-part model: P(OSI > theta) x E[OSI | regime], for the zero-inflated tail."""
    tr_rows = build_rows(mask_after_freeze(train_full))
    y = train_full.set_index(["fipsCode", "hour"])["osi"].reindex(
        pd.MultiIndex.from_frame(tr_rows[["fipsCode", "hour"]])).to_numpy()
    keep = np.isfinite(y)
    cols = feature_cols(tr_rows)
    X, yk = tr_rows.loc[keep, cols], y[keep]
    clf = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=SEED)
    clf.fit(X, yk > HURDLE_THETA)
    reg_hi = make_gbm().fit(X[yk > HURDLE_THETA], yk[yk > HURDLE_THETA])
    reg_lo = make_gbm().fit(X[yk <= HURDLE_THETA], yk[yk <= HURDLE_THETA])
    return clf, reg_hi, reg_lo, cols


def hurdle_forecaster(train_full, val_masked):
    clf, reg_hi, reg_lo, cols = fit_hurdle(train_full)
    va_rows = build_rows(val_masked)
    p = clf.predict_proba(va_rows[cols])[:, 1]
    hi = np.clip(reg_hi.predict(va_rows[cols]), 0.0, None)
    lo = np.clip(reg_lo.predict(va_rows[cols]), 0.0, None)
    out = va_rows[["fipsCode", "hour"]].copy()
    out["osi_pred"] = p * hi + (1 - p) * lo
    return out


def blend_hurdle_forecaster(train_full, val_masked):
    """Primary model: decay while the h71 state is fresh -> hurdle from the lull on."""
    g = hurdle_forecaster(train_full, val_masked).rename(columns={"osi_pred": "g"})
    d = decay_forecaster(train_full, val_masked).rename(columns={"osi_pred": "d"})
    m = g.merge(d, on=["fipsCode", "hour"])
    w = 1.0 / (1.0 + np.exp(-(m["hour"] - 100) / 10.0))
    m["osi_pred"] = (1 - w) * m["d"] + w * m["g"]
    return m[["fipsCode", "hour", "osi_pred"]]


def blend_perstate_hurdle_forecaster(train_full, val_masked):
    """Per-state hurdle models + decay blend. State specialization beats pooling
    for the tabular family (CV 0.01040 vs 0.01045 pooled; per-state plain GBM
    0.01031 vs 0.01094 pooled)."""
    tr_rows = build_rows(mask_after_freeze(train_full))
    y = train_full.set_index(["fipsCode", "hour"])["osi"].reindex(
        pd.MultiIndex.from_frame(tr_rows[["fipsCode", "hour"]])).to_numpy()
    cols = feature_cols(tr_rows)
    st_tr = train_full.groupby("fipsCode")["stateAbbr"].first()
    st_va = val_masked.groupby("fipsCode")["stateAbbr"].first()
    tr_rows["st"] = tr_rows["fipsCode"].map(st_tr)
    va_rows = build_rows(val_masked)
    va_rows["st"] = va_rows["fipsCode"].map(st_va)
    g = va_rows[["fipsCode", "hour"]].copy()
    g["osi_pred"] = np.nan
    keep = np.isfinite(y)
    for s in ["IN", "OH", "PA", "WV"]:
        m = keep & (tr_rows["st"] == s).to_numpy()
        X, yk = tr_rows.loc[m, cols], y[m]
        clf = HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=400, max_leaf_nodes=31, min_samples_leaf=30,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=SEED).fit(X, yk > HURDLE_THETA)
        hi = make_gbm().fit(X[yk > HURDLE_THETA], yk[yk > HURDLE_THETA])
        lo = make_gbm().fit(X[yk <= HURDLE_THETA], yk[yk <= HURDLE_THETA])
        mv = (va_rows["st"] == s).to_numpy()
        p = clf.predict_proba(va_rows.loc[mv, cols])[:, 1]
        g.loc[mv, "osi_pred"] = (p * np.clip(hi.predict(va_rows.loc[mv, cols]), 0, None)
                                 + (1 - p) * np.clip(lo.predict(va_rows.loc[mv, cols]), 0, None))
    d = decay_forecaster(train_full, val_masked).rename(columns={"osi_pred": "d"})
    m2 = g.merge(d, on=["fipsCode", "hour"])
    w = 1.0 / (1.0 + np.exp(-(m2["hour"] - 100) / 10.0))
    m2["osi_pred"] = (1 - w) * m2["d"] + w * m2["osi_pred"]
    return m2[["fipsCode", "hour", "osi_pred"]]


def fit_gbm_origin(train_full: pd.DataFrame):
    """Origin-indexed model (permissive causality): regional pool = train counties."""
    from .features import build_origin_rows
    tr_rows = build_origin_rows(mask_after_freeze(train_full), train_full, loo=True)
    y = train_full.set_index(["fipsCode", "hour"])["osi"].reindex(
        pd.MultiIndex.from_frame(tr_rows[["fipsCode", "hour"]])).to_numpy()
    keep = np.isfinite(y)
    cols = feature_cols(tr_rows)
    model = make_gbm()
    model.fit(tr_rows.loc[keep, cols], y[keep])
    return model, cols


def gbm_origin_forecaster(train_full, val_masked):
    from .features import build_origin_rows
    model, cols = fit_gbm_origin(train_full)
    va_rows = build_origin_rows(val_masked, train_full, loo=False)
    out = va_rows[["fipsCode", "origin", "h", "hour"]].copy()
    out["osi_pred"] = np.clip(model.predict(va_rows[cols]), 0.0, None)
    return out


def blend_origin_forecaster(train_full, val_masked):
    """Same decay->model time blend, applied to the origin-indexed GBM."""
    g = gbm_origin_forecaster(train_full, val_masked)
    d = decay_forecaster(train_full, val_masked).rename(columns={"osi_pred": "d"})
    m = g.merge(d, on=["fipsCode", "hour"])
    w = 1.0 / (1.0 + np.exp(-(m["hour"] - 100) / 10.0))
    m["osi_pred"] = (1 - w) * m["d"] + w * m["osi_pred"]
    return m[["fipsCode", "origin", "h", "hour", "osi_pred"]]


def blend_forecaster(train_full, val_masked):
    """Decay-dominant while the county's own state is fresh, GBM-dominant later.

    Motivated by hour-band diagnostics: plain exponential decay of the h71 state
    beats the GBM during the wave-1 decay (h73-96); the GBM wins from the lull on.
    """
    g = gbm_forecaster(train_full, val_masked).rename(columns={"osi_pred": "g"})
    d = decay_forecaster(train_full, val_masked).rename(columns={"osi_pred": "d"})
    m = g.merge(d, on=["fipsCode", "hour"])
    w = 1.0 / (1.0 + np.exp(-(m["hour"] - 100) / 10.0))
    m["osi_pred"] = (1 - w) * m["d"] + w * m["g"]
    return m[["fipsCode", "hour", "osi_pred"]]


LADDER = {
    "zero": zero_forecaster,
    "persistence(h71)": persistence_forecaster,
    "decayed persistence": decay_forecaster,
    "climatology": climatology_forecaster,
    "scaled climatology": scaled_climatology_forecaster,
    "GBM direct trajectory": gbm_forecaster,
    "blend decay->GBM": blend_forecaster,
    "GBM origin+regional": gbm_origin_forecaster,
    "blend decay->originGBM": blend_origin_forecaster,
    "hurdle 2-part": hurdle_forecaster,
    "blend decay->hurdle": blend_hurdle_forecaster,
}
