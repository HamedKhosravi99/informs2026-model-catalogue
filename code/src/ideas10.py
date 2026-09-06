"""Sweep 11 (U-series) - true 2.5-km URMA within-county extreme-wind features.

The provided `gust` is the organizers' polygon MEAN of the URMA 2.5-km
analysis. Damage is driven by within-county extremes, which the mean discards.
`fetch_urma_gust.py` rebuilds the true field from the public archive (the
source listed in weather_variables.pdf) and aggregates the actual within-county
distribution: max, p90/p95, and area fractions above 40/50/60/70 mph.

Distinct from the two prior attempts, which this supersedes:
  - S45/S60 sub-county features used FIVE Open-Meteo ERA5 points (~31 km
    native, resampled) - a proxy for spatial spread, not the real field;
  - S61's area fractions were RECONSTRUCTED from a fitted normal
    approximation of that 5-point spread.
Both measured ~nothing. If the real field also measures nothing, the "extreme
wind information" hypothesis is closed at the source-data level, which is the
strongest closure available. Law 3 says this goes to TREES only.

Controls (identical pipeline minus the new columns):
  U1  vs S60b statics-only sqrt-XGB          (0.00960)
  U1b vs S60d statics + old sub-county       (0.00956)
"""
import numpy as np
import pandas as pd

from .data import mask_after_freeze
from .features import feature_cols
from .ideas import _out, _rows_and_y, _val_rows
from .models import SEED

_URMA = None


def load_urma():
    global _URMA
    if _URMA is None:
        import os
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data_static", "urma_subcounty.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(
                "data_static/urma_subcounty.csv missing - run fetch_urma_gust.py")
        _URMA = pd.read_csv(p).set_index(["fipsCode", "hour"])
    return _URMA


def add_urma(rows):
    """Join the true-field columns + rolling trajectory summaries (weather is
    legal at any timestamp)."""
    u = load_urma()
    idx = pd.MultiIndex.from_frame(rows[["fipsCode", "hour"]])
    for c in ("u_max", "u_p90", "u_p95", "u_f40", "u_f50", "u_f60", "u_f70"):
        rows[c] = u[c].reindex(idx).to_numpy()
    if "gust" in rows.columns:
        rows["u_max_minus_mean"] = rows["u_max"] - rows["gust"]
        rows["u_max_ratio"] = rows["u_max"] / np.maximum(rows["gust"], 1e-6)

    d = rows[["fipsCode", "hour"]].copy()
    d["um"], d["f50"] = rows["u_max"].to_numpy(), rows["u_f50"].to_numpy()
    d = d.sort_values(["fipsCode", "hour"])
    g = d.groupby("fipsCode")
    for w in (6, 24):
        d[f"u_max{w}"] = g["um"].transform(lambda s: s.rolling(w, min_periods=1).max())
        d[f"u_f50sum{w}"] = g["f50"].transform(lambda s: s.rolling(w, min_periods=1).sum())
    d["u_cum_ex40"] = g["um"].transform(lambda s: np.clip(s - 40, 0, None).cumsum())
    d = d.set_index(["fipsCode", "hour"])
    for c in ("u_max6", "u_f50sum6", "u_max24", "u_f50sum24", "u_cum_ex40"):
        rows[c] = d[c].reindex(idx).to_numpy()
    return rows


def _xgb_sqrt_u(train_full, val_masked, sub=False, urma=True):
    import xgboost as xgb
    from .ideas6 import _rows
    tr, y = _rows(train_full, statics=True, sub=sub)
    va = _rows(val_masked, statics=True, sub=sub, val=True)
    if urma:
        tr, va = add_urma(tr), add_urma(va)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4,
                           tree_method="hist")
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)


def u1_urma(tr, va):        return _xgb_sqrt_u(tr, va, sub=False, urma=True)
def u1b_urma_sub(tr, va):   return _xgb_sqrt_u(tr, va, sub=True, urma=True)
def u1c_ctl_stat(tr, va):   return _xgb_sqrt_u(tr, va, sub=False, urma=False)


# --------------- F-series: the independent agent's genuinely-new features
# An independent adviser (PDFs + data only, blind to our record) proposed 20
# features; five families are not in our ledger. All tabular (law 3), all
# causality-clean: outage-side inputs use h<=71 only; weather is legal at any
# hour, including the future-looking clocks.
#   f_pre    pre-event mini-experiment: the Mar-11 wind bump (h0-47) as a
#            SECOND independent fragility measurement (peak P vs peak gust)
#   f_clock  anticipation clocks: hours until/since gust>=35 at the target
#            hour, future-window gust max (tau+1..tau+6)
#   f_inter  explicit interactions: gust x soil_moist, canopy x gust
#   f_veer   wind-direction shift at tau vs the county's wave-1 gust-peak hour
def _indep_extras(df_masked):
    """Per-(county,hour) extras from the raw frame. P_t used at h<=47 only."""
    d = df_masked.sort_values(["fipsCode", "hour"])
    g = d.pivot_table(index="fipsCode", columns="hour", values="gust"
                      ).reindex(columns=np.arange(216))
    fips = g.index.to_numpy()
    G = g.to_numpy()
    # anticipation / memory clocks for gust >= 35 mph
    hit = np.nan_to_num(G) >= 35.0
    since = np.full(G.shape, 99.0)
    until = np.full(G.shape, 99.0)
    last = np.full(len(G), -99)
    for t in range(216):
        last = np.where(hit[:, t], t, last)
        since[:, t] = np.where(last < 0, 99.0, t - last)
    nxt = np.full(len(G), 999)
    for t in range(215, -1, -1):
        nxt = np.where(hit[:, t], t, nxt)
        until[:, t] = np.clip(nxt - t, 0, 99)
    gfut6 = np.stack([np.roll(np.nan_to_num(G), -k, 1) for k in range(1, 7)]).max(0)
    gfut6[:, -6:] = np.nan_to_num(G[:, -6:])
    # wave-1 gust-peak direction, and P/gust peaks of the pre-event window
    # (wind_sin/cos are derived features elsewhere; build them from the raw
    # circular wind_dir_10m here)
    wdir = d.pivot_table(index="fipsCode", columns="hour", values="wind_dir_10m"
                         ).reindex(columns=np.arange(216)).to_numpy()
    wsin = np.sin(2 * np.pi * wdir / 360.0)
    wcos = np.cos(2 * np.pi * wdir / 360.0)
    pk = np.nanargmax(np.where(np.isfinite(G[:, 48:96]), G[:, 48:96], -1), 1) + 48
    s0 = wsin[np.arange(len(G)), pk][:, None]
    c0 = wcos[np.arange(len(G)), pk][:, None]
    veer = 1.0 - (wsin * s0 + wcos * c0)          # 0 = same direction, 2 = opposite
    P = d.pivot_table(index="fipsCode", columns="hour", values="P_t"
                      ).reindex(columns=np.arange(216)).to_numpy()
    pre_p = np.nanmax(np.where(np.isfinite(P[:, :48]), P[:, :48], 0), 1)
    pre_g = np.nanmax(np.where(np.isfinite(G[:, :48]), G[:, :48], 0), 1)
    rows = []
    for i, f in enumerate(fips):
        for t in range(216):
            rows.append((f, t, since[i, t], until[i, t], gfut6[i, t], veer[i, t]))
    e = pd.DataFrame(rows, columns=["fipsCode", "hour", "hrs_since_g35",
                                    "hrs_until_g35", "gfut6", "veer"])
    e = e.set_index(["fipsCode", "hour"])
    pre = pd.DataFrame({"fipsCode": fips,
                        "pre_peak_logP": np.log1p(pre_p * 1e3),
                        "pre_gust_max": pre_g}).set_index("fipsCode")
    return e, pre


def add_indep(rows, df_masked):
    e, pre = _indep_extras(df_masked)
    idx = pd.MultiIndex.from_frame(rows[["fipsCode", "hour"]])
    for c in e.columns:
        rows[c] = e[c].reindex(idx).to_numpy()
    for c in pre.columns:
        rows[c] = pre[c].reindex(rows["fipsCode"]).to_numpy()
    rows["pre_frag_gap"] = rows["pre_peak_logP"] - 0.05 * rows["pre_gust_max"]
    if "gust" in rows.columns and "soil_moist" in rows.columns:
        rows["gust_x_soil"] = rows["gust"] * rows["soil_moist"]
    try:
        import os
        lc = pd.read_csv(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data_static", "county_landcover.csv"),
            dtype={"fipsCode": str})
        lc["fipsCode"] = lc["fipsCode"].astype(int)
        canopy = lc.set_index("fipsCode")["pct_woody_total"]
        rows["canopy"] = canopy.reindex(rows["fipsCode"]).to_numpy()
        if "gust" in rows.columns:
            rows["canopy_x_gust"] = rows["canopy"] * rows["gust"]
    except FileNotFoundError:
        pass
    return rows


def _xgb_sqrt_f(train_full, val_masked, indep=True, only=None):
    import xgboost as xgb
    from .ideas6 import _rows
    tr, y = _rows(train_full, statics=True, sub=False)
    va = _rows(val_masked, statics=True, sub=False, val=True)
    if indep:
        tr = add_indep(tr, mask_after_freeze(train_full))
        va = add_indep(va, val_masked)
        if only is not None:                    # ablation: keep one family
            fam = {"clock": ["hrs_since_g35", "hrs_until_g35", "gfut6"],
                   "pre": ["pre_peak_logP", "pre_gust_max", "pre_frag_gap"],
                   "inter": ["gust_x_soil", "canopy", "canopy_x_gust"],
                   "veer": ["veer"]}
            drop = [c for k, cs in fam.items() if k != only for c in cs]
            tr = tr.drop(columns=[c for c in drop if c in tr.columns])
            va = va.drop(columns=[c for c in drop if c in va.columns])
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4,
                           tree_method="hist")
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)


def add_winter(rows, df):
    """F3 - winter-hazard load features (domain-literature sweep, item 1).

    Ice/wet-snow loading is a damage channel separate from gust; the snow/ice
    OPM literature (Cerrai et al. 2020, SEGAN; DeGaetano et al. 2008, WAF)
    finds accretion quantities dominate. All weather-only (legal at any hour):
      freeze_rain   rain falling in the accretion band (t2m in 271.15-273.65 K)
      ice_accum     cumulative CRREL-style proxy: freeze_rain * (1 + wind/10)
      wet_snow      snowfall near 0 C (272.15-274.35 K) - dense, adhering
      dur45_24      hours with gust >= 45 mph in the trailing 24 (Guikema
                    duration school - duration, not just intensity)
    plus their gust interactions, the mechanism class that produced F1d.
    """
    piv = lambda c: (df.pivot_table(index="fipsCode", columns="hour", values=c)
                     .reindex(columns=np.arange(216)))
    g = piv("gust")
    fips = g.index
    G = np.nan_to_num(g.to_numpy())
    T2 = np.nan_to_num(piv("t2m").to_numpy(), nan=280.0)
    RN = np.nan_to_num(piv("rain").to_numpy())
    SN = np.nan_to_num(piv("csnow").to_numpy())
    WS = np.nan_to_num(piv("wind_speed_10m").to_numpy())

    fr = RN * ((T2 > 271.15) & (T2 < 273.65))
    ice = np.cumsum(fr * (1.0 + WS / 10.0), axis=1)
    wsn = SN * ((T2 > 272.15) & (T2 < 274.35))
    wsn_c = np.cumsum(wsn, axis=1)
    hit45 = (G >= 45.0).astype(float)
    k = np.ones(24)
    dur45 = np.stack([np.convolve(r, k)[:216] for r in hit45])
    dur45_c = np.cumsum(hit45, axis=1)

    cols = {"freeze_rain": fr, "ice_accum": ice, "wet_snow_cum": wsn_c,
            "ice_x_gust": ice * G, "wsnow_x_gust": wsn_c * G,
            "dur45_24": dur45, "dur45_cum": dur45_c}
    frames = []
    for name, arr in cols.items():
        d = pd.DataFrame(arr, index=fips, columns=np.arange(216)).stack()
        d.name = name
        frames.append(d)
    W = pd.concat(frames, axis=1)
    W.index.names = ["fipsCode", "hour"]
    idx = pd.MultiIndex.from_frame(rows[["fipsCode", "hour"]])
    for c in W.columns:
        rows[c] = W[c].reindex(idx).to_numpy()
    return rows


def f3_winter(train_full, val_masked):
    """f2's exact pipeline + the winter-hazard block. Control: f2 (0.00930)."""
    import os
    import xgboost as xgb
    from .ideas6 import _rows
    lc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data_static", "county_landcover.csv")
    assert os.path.exists(lc), "county_landcover.csv required"
    tr, y = _rows(train_full, statics=True, sub=True)
    va = _rows(val_masked, statics=True, sub=True, val=True)
    tr = add_indep(tr, mask_after_freeze(train_full))
    va = add_indep(va, val_masked)
    drop = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
            "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
    tr = tr.drop(columns=[c for c in drop if c in tr.columns])
    va = va.drop(columns=[c for c in drop if c in va.columns])
    tr = add_winter(tr, mask_after_freeze(train_full))
    va = add_winter(va, val_masked)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4,
                           tree_method="hist")
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)


def f4_decay_mod(train_full, val_masked):
    """F4 - decay-modifier interactions, motivated by the 5ad importance
    analysis: the decay bases carry 10x the load of any other feature but use
    one global time constant, while the restoration literature (ORNL RePOWRD)
    and our own canopy finding say restoration speed is conditioned by
    statics (canopy slows tree-clearing; density speeds crew work). Four
    surgical columns on f2's exact pipeline:
      osi71_decay16 x canopy, osi71_decay32 x canopy,
      osi71_decay16 x log_density, canopy x gust_max24
    """
    import os
    import xgboost as xgb
    from .ideas6 import _rows
    lc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data_static", "county_landcover.csv")
    assert os.path.exists(lc), "county_landcover.csv required"
    tr, y = _rows(train_full, statics=True, sub=True)
    va = _rows(val_masked, statics=True, sub=True, val=True)
    tr = add_indep(tr, mask_after_freeze(train_full))
    va = add_indep(va, val_masked)
    drop = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
            "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
    tr = tr.drop(columns=[c for c in drop if c in tr.columns])
    va = va.drop(columns=[c for c in drop if c in va.columns])
    for d in (tr, va):
        d["d16_x_canopy"] = d["osi71_decay16"] * d["canopy"]
        d["d32_x_canopy"] = d["osi71_decay32"] * d["canopy"]
        d["d16_x_logdens"] = d["osi71_decay16"] * d["log_density"]
        d["canopy_x_gmax24"] = d["canopy"] * d["gust_max24"]
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4,
                           tree_method="hist")
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)


# RT4: Optuna TPE winner for the f2 XGB hyperparameters (optuna_tune_f2.py;
# nested true-metric inner protocol, warm-started at the defaults). Unlike
# the sequence-model searches, the top-8 configs AGREE: more trees, lower lr,
# depth 5. Inner 0.01016 vs defaults' 0.01042.
RT4_CFG = dict(n_estimators=1800, learning_rate=0.026689, max_depth=5,
               subsample=0.762967, colsample_bytree=0.701628,
               min_child_weight=4, reg_lambda=0.750390, reg_alpha=1.0026e-4)


def rt4_tuned_f2(train_full, val_masked):
    """f2's exact pipeline with the Optuna-tuned XGB config."""
    import os
    import xgboost as xgb
    from .ideas6 import _rows
    lc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data_static", "county_landcover.csv")
    assert os.path.exists(lc), "county_landcover.csv required"
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
    mdl = xgb.XGBRegressor(**RT4_CFG, random_state=SEED, n_jobs=4,
                           tree_method="hist")
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)


def f1_indep(tr, va):   return _xgb_sqrt_f(tr, va, indep=True)


def f2_member_upgrade(train_full, val_masked, seeds=None):
    """The shipped tabular member (2026-08-13): s45's pipeline (statics +
    sub-county gust spread) PLUS the explicit physics-interaction family from
    the independence audit (gust x soil_moist, canopy, canopy x gust).

    Adopted by supersession (REPORT 5x/5y): same member family, strictly more
    mechanism-backed features, standalone 0.00930 vs s45's 0.00942, and the
    ensemble swap is pointwise better at all four horizons. The canopy table is
    a hard requirement here - a silent fallback would ship a different model
    (PLAN W4)."""
    import os
    import xgboost as xgb
    lc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data_static", "county_landcover.csv")
    assert os.path.exists(lc), "county_landcover.csv is required for the f2 member"
    from .ideas6 import _rows
    tr, y = _rows(train_full, statics=True, sub=True)
    va = _rows(val_masked, statics=True, sub=True, val=True)
    tr = add_indep(tr, mask_after_freeze(train_full))
    va = add_indep(va, val_masked)
    fam_drop = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
                "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
    tr = tr.drop(columns=[c for c in fam_drop if c in tr.columns])
    va = va.drop(columns=[c for c in fam_drop if c in va.columns])
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                           reg_lambda=1.0, random_state=SEED, n_jobs=4,
                           tree_method="hist")
    if seeds is None:                                   # unchanged single-seed path
        mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
        return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)
    preds = []                                          # seed-averaged variant (Y-series)
    for sd in seeds:
        mdl.set_params(random_state=int(sd)); mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
        preds.append(np.clip(mdl.predict(va[cols]), 0, None) ** 2)
    return _out(va, np.mean(preds, 0))
def f1b_clock(tr, va):  return _xgb_sqrt_f(tr, va, only="clock")
def f1c_pre(tr, va):    return _xgb_sqrt_f(tr, va, only="pre")
def f1d_inter(tr, va):  return _xgb_sqrt_f(tr, va, only="inter")
def f1e_veer(tr, va):   return _xgb_sqrt_f(tr, va, only="veer")


IDEAS10 = {
    "U1 sqrt-XGB + statics + true URMA field": u1_urma,
    "U1b sqrt-XGB + statics + sub + URMA": u1b_urma_sub,
    "U1c control rerun (statics only)": u1c_ctl_stat,
    "F1 independent-agent features (pre-event frag, clocks, interactions, veer)": f1_indep,
    "F1b clocks only": f1b_clock,
    "F1c pre-event fragility only": f1c_pre,
    "F1d interactions only": f1d_inter,
    "F1e veer only": f1e_veer,
    "F2 s45 pipeline + interactions (member upgrade)": f2_member_upgrade,
    "F3 f2 + winter-hazard load features": f3_winter,
    "F4 f2 + decay-modifier interactions": f4_decay_mod,
    "RT4 optuna-tuned f2 XGB config": rt4_tuned_f2,
}
