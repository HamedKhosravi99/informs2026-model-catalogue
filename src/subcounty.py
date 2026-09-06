"""Sub-county weather heterogeneity features.

The competition's `gust` column is a county polygon MEAN of URMA, which discards the
local extremes that actually break infrastructure. `fetch_subcounty_weather.py` samples
several points per county from the public archive and derives *relative* spread
features; relative because absolute calibration differs between weather models, while
within-county heterogeneity should transfer.

A missing data file is a HARD ERROR: the shipped ensemble member s45 depends on these
features, and a silent no-op here would produce a different submission with no warning
(this nearly happened — see HANDOFF.md gotchas).
"""
import numpy as np
import pandas as pd

from .data import ROOT

_CACHE = None
COLS = ["gmax_ratio", "gp90_ratio", "gstd_ratio", "gmax_abs"]


def load_subcounty():
    global _CACHE
    if _CACHE is None:
        p = ROOT / "data_static" / "subcounty_weather.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} is missing. The shipped ensemble member s45 requires it; a "
                f"silent fallback here would produce a different submission with no "
                f"warning. Regenerate with: python3 fetch_subcounty_weather.py "
                f"(the file is committed to the repo, so this should only happen on "
                f"a partial checkout).")
        _CACHE = pd.read_csv(p).set_index(["fipsCode", "hour"])
    return _CACHE


def add_subcounty(rows, src_df=None):
    """Attach within-county gust heterogeneity at each row's target hour, plus the
    interaction with the county-mean gust (local peak ~ mean x heterogeneity)."""
    sc = load_subcounty()
    if sc is False:
        return rows
    idx = pd.MultiIndex.from_frame(rows[["fipsCode", "hour"]])
    for c in COLS:
        rows[c] = sc[c].reindex(idx).to_numpy()
    if "gust" in rows.columns:
        rows["local_peak_gust"] = rows["gust"] * rows["gmax_ratio"]
        rows["local_peak_excess"] = (rows["local_peak_gust"] - 30).clip(lower=0)
    # rolling within-county peak over the preceding 6h, from the relative series
    g = rows.sort_values(["fipsCode", "hour"]).groupby("fipsCode")["gmax_ratio"]
    rows["gmax_ratio_roll6"] = g.transform(
        lambda s: s.rolling(6, min_periods=1).max()).reindex(rows.index)
    return rows
