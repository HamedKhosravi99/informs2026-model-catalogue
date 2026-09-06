"""Recover sub-county weather variability that the provided polygon means destroy.

The competition's `gust` column is a county polygon MEAN of URMA. Damage is driven by
local extremes, so the mean discards exactly the signal that matters. This script
samples several points per county from the same public archive the organizers use for
their ERA5 fields (Open-Meteo) and derives *relative* heterogeneity features --
max/mean, p90/mean, std/mean -- which describe within-county spread and are therefore
robust to any difference in absolute calibration between weather models.

Weather at any timestamp is explicitly permitted by the competition rules; no outage
information is involved.

Output: data_static/subcounty_weather.csv (fipsCode, hour, gmax_ratio, gp90_ratio,
gstd_ratio, gmax_abs).
"""
import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

from src.data import ROOT

OUT = ROOT / "data_static" / "subcounty_weather.csv"
START, END = "2026-03-11", "2026-03-19"
BATCH = 40          # locations per request
OFFSETS = [(0, 0), (0.16, 0), (-0.16, 0), (0, 0.20), (0, -0.20)]


def main():
    st = pd.read_csv(ROOT / "data_static" / "county_static.csv")
    lats, lons, keys = [], [], []
    for _, r in st.iterrows():
        # scale the sampling ring by county size so points stay inside the county
        s = np.sqrt(r.land_sqmi) / 25.0
        for j, (dla, dlo) in enumerate(OFFSETS):
            lats.append(round(r.lat + dla * s, 4))
            lons.append(round(r.lon + dlo * s, 4))
            keys.append((int(r.fipsCode), j))

    rows = []
    n = len(lats)
    print(f"querying {n} points ({len(st)} counties x {len(OFFSETS)}) in batches of {BATCH}")
    for i in range(0, n, BATCH):
        la = ",".join(str(x) for x in lats[i:i + BATCH])
        lo = ",".join(str(x) for x in lons[i:i + BATCH])
        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}"
               f"&start_date={START}&end_date={END}&hourly=wind_gusts_10m"
               f"&timezone=America%2FNew_York")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=120) as f:
                    data = json.load(f)
                break
            except Exception as exc:                      # transient rate limits
                if attempt == 4:
                    print(f"  batch {i} failed permanently: {exc}")
                    data = None
                    break
                time.sleep(5 * (attempt + 1))
        if data is None:
            continue
        if isinstance(data, dict):
            data = [data]
        for j, loc in enumerate(data):
            fips, pt = keys[i + j]
            g = loc.get("hourly", {}).get("wind_gusts_10m") or []
            for h, v in enumerate(g):
                if v is not None:
                    rows.append((fips, h, pt, v))
        if (i // BATCH) % 10 == 0:
            print(f"  {i+len(data)}/{n} points, {len(rows)} observations", flush=True)
        time.sleep(0.3)

    df = pd.DataFrame(rows, columns=["fipsCode", "hour", "pt", "gust"])
    print(f"collected {len(df)} point-hours for {df.fipsCode.nunique()} counties")
    agg = df.groupby(["fipsCode", "hour"])["gust"].agg(
        gmax="max", gmean="mean", gstd="std",
        gp90=lambda s: float(np.percentile(s, 90))).reset_index()
    agg["gmax_ratio"] = agg.gmax / agg.gmean.clip(lower=0.1)
    agg["gp90_ratio"] = agg.gp90 / agg.gmean.clip(lower=0.1)
    agg["gstd_ratio"] = agg.gstd.fillna(0) / agg.gmean.clip(lower=0.1)
    agg["gmax_abs"] = agg.gmax
    agg[["fipsCode", "hour", "gmax_ratio", "gp90_ratio", "gstd_ratio", "gmax_abs"]].to_csv(
        OUT, index=False)
    print(f"wrote {OUT}: {len(agg)} county-hours")
    print(agg[["gmax_ratio", "gp90_ratio", "gstd_ratio"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
