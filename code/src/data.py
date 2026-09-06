"""Data loading and protocol constants for the INFORMS 2026 DM Data Challenge.

Protocol: test counties expose outage variables only for hours 0-71 (Mar 11-13).
Predictions cover target hours 73-215; submission origins are hours 72-215 with
osi_target_tXXh at origin t = OSI at hour t+XX (NaN when t+XX > 215).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FREEZE_H = 71                     # last hour with outage data in the test file
PRED_HOURS = np.arange(73, 216)   # target hours that ever get scored
ORIGINS = np.arange(72, 216)      # submission rows (Mar 14-19)
HORIZONS = {"t01h": 1, "t06h": 6, "t24h": 24, "t48h": 48}

OUTAGE_COLS = [
    "outageCount", "outage_pct", "P_t", "N_t", "D_t", "R_t", "osi",
    "outage_pct_lag1h", "outage_pct_lag3h", "outage_pct_lag6h",
    "outage_pct_lag24h", "outage_pct_lag48h",
    "osi_lag1h", "osi_lag3h", "osi_lag6h", "osi_lag24h", "osi_lag48h",
]
# train-only columns that must never become features
FORBIDDEN = ["severity_tier", "peak_pct", "peak_customers", "time_to_restore_h", "split"]

WEATHER_COLS = [
    "gust", "wind_speed_10m", "wind_dir_10m", "t2m", "d2m", "sp", "mslma",
    "blh", "tp", "rain", "csnow", "sdwe", "tcc", "lcc", "mcc", "hcc",
    "sdswrf", "direct_rad", "diffuse_rad", "r2", "vpd", "et0", "soil_moist",
]


def _add_hour(df: pd.DataFrame) -> pd.DataFrame:
    t0 = pd.Timestamp("2026-03-11 00:00:00")
    df["hour"] = ((df["timestamp_et"] - t0).dt.total_seconds() // 3600).astype(int)
    return df


def load_train() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "DM_Train.csv", parse_dates=["timestamp_et"])
    return _add_hour(df).sort_values(["fipsCode", "hour"]).reset_index(drop=True)


def load_test() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "DM_Test.csv", parse_dates=["timestamp_et"])
    df = _add_hour(df).sort_values(["fipsCode", "hour"]).reset_index(drop=True)
    # the test file carries no `osi` column; reconstruct it for the observed window
    df["osi"] = reconstruct_osi(df)
    return df


def reconstruct_osi(df: pd.DataFrame) -> pd.Series:
    return (0.40 * df["P_t"] + 0.35 * df["N_t"] + 0.25 * df["D_t"] - 0.10 * df["R_t"]).clip(lower=0)


def mask_after_freeze(df: pd.DataFrame) -> pd.DataFrame:
    """Simulate the test protocol: NaN all outage-derived columns after FREEZE_H."""
    out = df.copy()
    cols = [c for c in dict.fromkeys(OUTAGE_COLS + ["osi"]) if c in out.columns]
    out.loc[out["hour"] > FREEZE_H, cols] = np.nan
    return out


def osi_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """County x hour matrix of true OSI."""
    return df.pivot_table(index="fipsCode", columns="hour", values="osi")
