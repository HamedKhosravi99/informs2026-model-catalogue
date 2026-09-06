"""Feature construction for the direct trajectory model.

One row per (county, target hour H) for H in PRED_HOURS. Outage-derived inputs
come exclusively from hours <= FREEZE_H — enforced structurally: callers pass a
frame already masked with data.mask_after_freeze (or the real test file, which
arrives masked). Weather may be used at any hour (forecast information).
"""
import numpy as np
import pandas as pd

from .data import FREEZE_H, PRED_HOURS

STATES = ["IN", "OH", "PA", "WV"]
GUST_DAMAGE_THRESH = 30.0  # mph; onset of meaningful damage in EDA


def county_static(masked: pd.DataFrame) -> pd.DataFrame:
    """Per-county features summarizing the observed window (hours 0..FREEZE_H)."""
    obs = masked[masked["hour"] <= FREEZE_H]
    g = obs.groupby("fipsCode")

    st = pd.DataFrame(index=g.size().index)
    st["log_customers"] = np.log1p(g["customersTracked"].median())
    st["pre_mean_pct"] = obs[obs["hour"] <= 47].groupby("fipsCode")["outage_pct"].mean()
    st["obs_osi_h71"] = g.apply(lambda d: d.loc[d["hour"] == FREEZE_H, "osi"].iloc[0], include_groups=False)
    st["obs_osi_slope"] = st["obs_osi_h71"] - g.apply(
        lambda d: d.loc[d["hour"] == FREEZE_H - 6, "osi"].iloc[0], include_groups=False)
    st["obs_osi_mean_w1"] = obs[obs["hour"] >= 48].groupby("fipsCode")["osi"].mean()
    st["obs_osi_max"] = g["osi"].max()
    st["obs_pct_max"] = g["outage_pct"].max()
    st["obs_Nt_sum"] = g["N_t"].sum()
    st["obs_Nt_max"] = g["N_t"].max()
    st["obs_Rt_sum"] = g["R_t"].sum()
    st["restored_frac"] = st["obs_Rt_sum"] / (st["obs_Nt_sum"] + 1e-6)
    st["obs_peak_lag"] = FREEZE_H - g.apply(lambda d: d.loc[d["osi"].idxmax(), "hour"], include_groups=False)
    st["obs_gust_max"] = g["gust"].max()
    st["obs_gust_exposure"] = g.apply(
        lambda d: (d["gust"] - GUST_DAMAGE_THRESH).clip(lower=0).sum(), include_groups=False)
    st["fragility"] = st["obs_Nt_sum"] / (st["obs_gust_exposure"] + 1.0)
    for s in STATES:
        st[f"state_{s}"] = (g["stateAbbr"].first() == s).astype(float)
    return st


def weather_by_hour(masked: pd.DataFrame) -> pd.DataFrame:
    """Per (county, hour) weather features over the full 216-hour window."""
    df = masked.sort_values(["fipsCode", "hour"]).copy()
    g = df.groupby("fipsCode")

    w = df[["fipsCode", "hour"]].copy()
    rad = np.deg2rad(df["wind_dir_10m"])
    w["wind_sin"], w["wind_cos"] = np.sin(rad), np.cos(rad)
    for c in ["gust", "wind_speed_10m", "t2m", "tp", "rain", "csnow", "blh",
              "mslma", "r2", "soil_moist", "sdwe"]:
        w[c] = df[c]
    w["dew_spread"] = df["t2m"] - df["d2m"]

    w["gust_lag1"] = g["gust"].shift(1)
    w["gust_lag3"] = g["gust"].shift(3)
    w["gust_max6"] = g["gust"].rolling(6, min_periods=1).max().reset_index(0, drop=True)
    w["gust_mean6"] = g["gust"].rolling(6, min_periods=1).mean().reset_index(0, drop=True)
    w["gust_max24"] = g["gust"].rolling(24, min_periods=1).max().reset_index(0, drop=True)
    w["tp_cum24"] = g["tp"].rolling(24, min_periods=1).sum().reset_index(0, drop=True)
    w["mslma_trend24"] = df["mslma"] - g["mslma"].shift(24)

    # novel-wind exceedance: fresh damage needs wind beyond what already broke things
    cummax_prior = g["gust"].apply(lambda s: s.cummax().shift(1)).reset_index(0, drop=True)
    w["gust_cummax_prior"] = cummax_prior
    w["gust_exceed"] = (df["gust"] - cummax_prior).fillna(0.0)

    # cumulative damaging-wind exposure and its unobserved-period increment
    excess = (df["gust"] - GUST_DAMAGE_THRESH).clip(lower=0)
    cum = excess.groupby(df["fipsCode"]).cumsum()
    w["cum_exposure"] = cum
    cum71 = w.loc[w["hour"] == FREEZE_H, ["fipsCode", "cum_exposure"]].set_index("fipsCode")["cum_exposure"]
    w["new_exposure"] = cum - w["fipsCode"].map(cum71)

    # restoration-phase indicator: hours since gust last exceeded the damage threshold
    hit = df["hour"].where(df["gust"] >= GUST_DAMAGE_THRESH)
    last_hit = hit.groupby(df["fipsCode"]).ffill()
    w["hrs_since_strong_wind"] = (df["hour"] - last_hit).fillna(96.0).clip(upper=96)

    w["hod_sin"] = np.sin(2 * np.pi * (df["hour"] % 24) / 24)
    w["hod_cos"] = np.cos(2 * np.pi * (df["hour"] % 24) / 24)
    return w


def build_rows(masked: pd.DataFrame, hours=PRED_HOURS) -> pd.DataFrame:
    """Assemble the model matrix: one row per (county, target hour H)."""
    st = county_static(masked)
    w = weather_by_hour(masked)
    rows = w[w["hour"].isin(hours)].merge(st, left_on="fipsCode", right_index=True, how="left")
    rows["H"] = rows["hour"].astype(float)
    rows["hrs_since_freeze"] = rows["hour"] - float(FREEZE_H)
    rows["frag_x_gust6"] = rows["fragility"] * rows["gust_max6"]
    rows["frag_x_exceed"] = rows["fragility"] * rows["gust_exceed"].clip(lower=0)
    # decay basis: persistence of the last observed state at several time constants,
    # so the model can compose county-appropriate restoration curves
    for tau in (8, 16, 32, 64):
        rows[f"osi71_decay{tau}"] = rows["obs_osi_h71"] * np.exp(-rows["hrs_since_freeze"] / tau)
    return rows.reset_index(drop=True)


def feature_cols(rows: pd.DataFrame) -> list[str]:
    return [c for c in rows.columns if c not in ("fipsCode", "hour", "origin")]


# ---------------------------------------------------------------------------
# Origin-indexed rows under the permissive causality reading: at origin t, any
# outage value at timestamp <= t is allowed — including the training counties'
# observed values during Mar 14-19. Regional state comes ONLY from `pool_full`
# (training counties) at hours <= t; the target county's own outage data stays
# frozen at FREEZE_H (it is NaN afterwards in the test file anyway).
# ---------------------------------------------------------------------------
from .data import ORIGINS, HORIZONS  # noqa: E402


def _knn_weights(target_gust: pd.DataFrame, pool_gust: pd.DataFrame, k: int = 10) -> np.ndarray:
    """Storm-cell similarity: correlation of full 216h gust series (weather only)."""
    t = (target_gust - target_gust.mean(1).values[:, None])
    p = (pool_gust - pool_gust.mean(1).values[:, None])
    t = t / np.sqrt((t ** 2).sum(1)).values[:, None]
    p = p / np.sqrt((p ** 2).sum(1)).values[:, None]
    corr = t.values @ p.values.T                      # targets x pool
    self_mask = target_gust.index.values[:, None] == pool_gust.index.values[None, :]
    corr[self_mask] = -1.0                            # never your own neighbor
    W = np.clip(corr, 0, None) ** 2
    thresh = -np.sort(-W, axis=1)[:, k - 1:k]         # keep top-k per target
    W[W < thresh] = 0.0
    W /= W.sum(1, keepdims=True) + 1e-12
    return W


def build_origin_rows(masked_target: pd.DataFrame, pool_full: pd.DataFrame,
                      loo: bool = False) -> pd.DataFrame:
    """One row per (target county, origin t, horizon h) with t+h <= 215.

    loo=True when target counties are themselves in the pool (training rows):
    aggregates then exclude the county itself, matching deployment where the
    test county is never part of the regional pool.
    """
    st = county_static(masked_target)
    w = weather_by_hour(masked_target).set_index(["fipsCode", "hour"])

    posi = pool_full.pivot_table(index="fipsCode", columns="hour", values="osi").sort_index(axis=1)
    pN = pool_full.pivot_table(index="fipsCode", columns="hour", values="N_t").sort_index(axis=1)
    pool_states = pool_full.groupby("fipsCode")["stateAbbr"].first()
    tgt_states = masked_target.groupby("fipsCode")["stateAbbr"].first()

    hours = posi.columns.to_numpy()
    all_sum, all_n = posi.sum(0).to_numpy(), len(posi)
    st_sum = posi.groupby(pool_states).sum()
    st_n = pool_states.value_counts()

    tgt_gust = masked_target.pivot_table(index="fipsCode", columns="hour", values="gust")
    pool_gust = pool_full.pivot_table(index="fipsCode", columns="hour", values="gust")
    W = _knn_weights(tgt_gust, pool_gust.loc[posi.index])
    knn_osi = pd.DataFrame(W @ posi.to_numpy(), index=tgt_gust.index, columns=hours)
    knn_N = pd.DataFrame(W @ pN.fillna(0).to_numpy(), index=tgt_gust.index, columns=hours)

    fips = tgt_gust.index.to_numpy()
    grids = []
    for name, h in HORIZONS.items():
        og = ORIGINS[ORIGINS + h <= 215]
        g = pd.DataFrame({"fipsCode": np.repeat(fips, len(og)),
                          "origin": np.tile(og, len(fips))})
        g["h"] = float(h)
        grids.append(g)
    rows = pd.concat(grids, ignore_index=True)
    rows["hour"] = (rows["origin"] + rows["h"]).astype(int)   # target hour H

    # regional state at origin t (strictly <= t by construction)
    t_idx = rows["origin"].to_numpy()
    c_idx = pd.Index(posi.index).get_indexer(rows["fipsCode"]) if loo else None
    own = posi.to_numpy()[c_idx, t_idx - hours[0]] if loo else 0.0
    own6 = posi.to_numpy()[c_idx, t_idx - 6 - hours[0]] if loo else 0.0
    n_eff = all_n - 1 if loo else all_n
    rows["reg_all_osi"] = (all_sum[t_idx - hours[0]] - own) / n_eff
    rows["reg_all_osi_trend6"] = rows["reg_all_osi"] - (all_sum[t_idx - 6 - hours[0]] - own6) / n_eff

    stt = rows["fipsCode"].map(tgt_states)
    s_sum_t = st_sum.to_numpy()[st_sum.index.get_indexer(stt), t_idx - hours[0]]
    s_n = stt.map(st_n).to_numpy().astype(float)
    if loo:
        rows["reg_state_osi"] = (s_sum_t - own) / (s_n - 1)
    else:
        rows["reg_state_osi"] = s_sum_t / s_n

    ti = pd.Index(tgt_gust.index).get_indexer(rows["fipsCode"])
    rows["knn_osi"] = knn_osi.to_numpy()[ti, t_idx - hours[0]]
    rows["knn_osi_trend6"] = rows["knn_osi"] - knn_osi.to_numpy()[ti, t_idx - 6 - hours[0]]
    rows["knn_Nt"] = knn_N.to_numpy()[ti, t_idx - hours[0]]

    rows = rows.join(w, on=["fipsCode", "hour"])
    rows = rows.merge(st, left_on="fipsCode", right_index=True, how="left")
    rows["H"] = rows["hour"].astype(float)
    rows["hrs_since_freeze"] = rows["hour"] - float(FREEZE_H)
    rows["frag_x_gust6"] = rows["fragility"] * rows["gust_max6"]
    rows["frag_x_exceed"] = rows["fragility"] * rows["gust_exceed"].clip(lower=0)
    for tau in (8, 16, 32, 64):
        rows[f"osi71_decay{tau}"] = rows["obs_osi_h71"] * np.exp(-rows["hrs_since_freeze"] / tau)
    return rows.reset_index(drop=True)
