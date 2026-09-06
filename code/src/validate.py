"""County-holdout validation harness mirroring the official test protocol.

Folds hold out whole counties (stratified by state x severity_tier). For each
fold, the forecaster sees the held-out counties with outage data masked after
FREEZE_H and must return an OSI trajectory for hours 73-215; scoring maps that
trajectory into the four submission columns exactly as the organizers will.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import FREEZE_H, HORIZONS, mask_after_freeze, osi_trajectory

SEED = 42


def make_folds(train: pd.DataFrame, n_splits: int = 5):
    """Yield (train_fips, val_fips) with county units stratified by state x tier."""
    counties = train.groupby("fipsCode").agg(
        state=("stateAbbr", "first"), tier=("severity_tier", "first"))
    strata = counties["state"] + "_" + counties["tier"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fips = counties.index.to_numpy()
    for tr_idx, va_idx in skf.split(fips, strata):
        yield fips[tr_idx], fips[va_idx]


def score_origin(pred: pd.DataFrame, truth_piv: pd.DataFrame) -> pd.DataFrame:
    """Score origin-indexed predictions: columns (fipsCode, origin, h, osi_pred)."""
    truth = truth_piv.stack()
    p = pred.copy()
    p["truth"] = truth.reindex(
        pd.MultiIndex.from_arrays([p["fipsCode"], (p["origin"] + p["h"]).astype(int)])).to_numpy()
    p["err"] = p["osi_pred"] - p["truth"]
    out = []
    for name, h in HORIZONS.items():
        e = p.loc[p["h"] == h, "err"]
        out.append({"horizon": name, "rmse": float(np.sqrt(np.nanmean(e ** 2))),
                    "mae": float(np.nanmean(np.abs(e))), "n": int(e.notna().sum())})
    df = pd.DataFrame(out).set_index("horizon")
    df.loc["mean"] = [df["rmse"].mean(), df["mae"].mean(), df["n"].sum()]
    return df


def score_trajectory(pred: pd.DataFrame, truth_piv: pd.DataFrame) -> pd.DataFrame:
    """Per-horizon RMSE/MAE. `pred`: columns (fipsCode, hour, osi_pred), hours 73-215."""
    pred_piv = pred.pivot_table(index="fipsCode", columns="hour", values="osi_pred")
    out = []
    for name, h in HORIZONS.items():
        tgt = np.arange(FREEZE_H + 1 + h, 216)  # scored target hours for this column
        t = truth_piv.loc[pred_piv.index, tgt].to_numpy()
        p = pred_piv.reindex(columns=tgt).to_numpy()
        err = p - t
        out.append({"horizon": name, "rmse": float(np.sqrt(np.nanmean(err ** 2))),
                    "mae": float(np.nanmean(np.abs(err))), "n": int(np.isfinite(err).sum())})
    df = pd.DataFrame(out).set_index("horizon")
    df.loc["mean"] = [df["rmse"].mean(), df["mae"].mean(), df["n"].sum()]
    return df


def run_cv(forecaster, train: pd.DataFrame, n_splits: int = 5, per_fold: bool = False):
    """forecaster(train_full, val_masked) -> DataFrame(fipsCode, hour, osi_pred)."""
    truth_piv = osi_trajectory(train)
    all_pred, fold_tables = [], []
    for k, (tr_fips, va_fips) in enumerate(make_folds(train, n_splits)):
        tr_df = train[train["fipsCode"].isin(tr_fips)]
        va_df = mask_after_freeze(train[train["fipsCode"].isin(va_fips)])
        va_df = va_df.drop(columns=[c for c in
                                    ("severity_tier", "peak_pct", "peak_customers",
                                     "time_to_restore_h", "split") if c in va_df.columns])
        pred = forecaster(tr_df, va_df)
        all_pred.append(pred)
        scorer = score_origin if "origin" in pred.columns else score_trajectory
        if per_fold:
            fold_tables.append(scorer(pred, truth_piv))
    pooled = pd.concat(all_pred, ignore_index=True)
    scorer = score_origin if "origin" in pooled.columns else score_trajectory
    table = scorer(pooled, truth_piv)
    return (table, pooled, fold_tables) if per_fold else (table, pooled)
