"""True 2.5-km URMA gust fields -> within-county extreme-wind features.

The provided `gust` is the polygon MEAN of the URMA 2.5-km analysis (the
organizers' own aggregation) - it discards the within-county extremes that
drive damage. This pulls the actual GUST fields from the public NOAA archive
(the source listed in weather_variables.pdf), byte-ranging just the GUST
message out of each hourly analysis via the .idx sidecars (~5.6 MB/hour
instead of ~75 MB), and aggregates the true within-county distribution.

Outputs data_static/urma_subcounty.csv: one row per (fipsCode, hour 0..215):
  u_mean   polygon mean, mph      (validation column - should reproduce `gust`)
  u_max    true within-county max
  u_p90, u_p95
  u_f40, u_f50, u_f60, u_f70     area fraction with gust > 40/50/60/70 mph

Time mapping: competition labels are fixed EST (UTC-5), per
weather_variables.pdf; label hour h maps to the URMA analysis at UTC h+5.
The u_mean vs provided-gust correlation printed at the end verifies both the
aggregation and the alignment (HANDOFF gotcha: check before believing).

Cache: data_static/urma_cache/ (git-ignored). Rerun is incremental.
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "data_static", "urma_cache")
OUT = os.path.join(ROOT, "data_static", "urma_subcounty.csv")
BASE = "https://noaa-urma-pds.s3.amazonaws.com"
TIGER = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
         "State_County/MapServer/1/query")
STATES = ("18", "39", "42", "54")                    # IN OH PA WV
MS_TO_MPH = 2.2369
START_ET = datetime(2026, 3, 11, 0)                   # label hour 0
UTC_OFFSET = 5                                        # fixed EST per the docs
THRESH = (40.0, 50.0, 60.0, 70.0)


def fetch(url, timeout=180, tries=5, rng=None):
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    for a in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return f.read()
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(3 * (a + 1))


def gust_bytes(day, cyc):
    """Byte range of the GUST message in one analysis file, from its .idx."""
    key = f"urma2p5.{day}/urma2p5.t{cyc:02d}z.2dvaranl_ndfd.grb2_wexp"
    idx = fetch(f"{BASE}/{key}.idx").decode().splitlines()
    starts = [int(l.split(":")[1]) for l in idx]
    for i, l in enumerate(idx):
        if ":GUST:" in l:
            end = starts[i + 1] - 1 if i + 1 < len(idx) else ""
            return key, (starts[i], end)
    raise RuntimeError(f"no GUST in {key}")


def download_all():
    os.makedirs(CACHE, exist_ok=True)
    paths = []
    for h in range(216):
        utc = START_ET + timedelta(hours=h + UTC_OFFSET)
        day, cyc = utc.strftime("%Y%m%d"), utc.hour
        p = os.path.join(CACHE, f"gust_{day}_{cyc:02d}.grb2")
        paths.append((h, p))
        if os.path.exists(p) and os.path.getsize(p) > 100000:
            continue
        key, rng = gust_bytes(day, cyc)
        blob = fetch(f"{BASE}/{key}", rng=rng)
        with open(p, "wb") as f:
            f.write(blob)
        print(f"  {h:3d}/215  {day} t{cyc:02d}z  {len(blob)/1e6:.1f} MB", flush=True)
    return paths


def county_rings():
    rings = {}
    for st in STATES:
        url = (f"{TIGER}?where=STATE%3D%27{st}%27&outFields=GEOID"
               f"&returnGeometry=true&outSR=4326&maxAllowableOffset=0.001&f=json")
        d = json.loads(fetch(url).decode())
        for f in d["features"]:
            rings[int(f["attributes"]["GEOID"])] = f["geometry"]["rings"]
    return rings


def inside(ring_list, x, y):
    hit = np.zeros(len(x), dtype=bool)
    for ring in ring_list:
        p = np.asarray(ring, dtype=float)
        x1, y1 = p[:-1, 0], p[:-1, 1]
        x2, y2 = p[1:, 0], p[1:, 1]
        cond = ((y1[None, :] > y[:, None]) != (y2[None, :] > y[:, None]))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = ((x2 - x1)[None, :] * (y[:, None] - y1[None, :])
                    / (y2 - y1)[None, :] + x1[None, :])
        hit ^= (cond & (x[:, None] < xint)).sum(axis=1) % 2 == 1
    return hit


def build_masks(sample_grb):
    """Flat grid indices per county, via bbox prefilter + ray-cast."""
    import pygrib
    g = pygrib.open(sample_grb).message(1)
    lats, lons = g.latlons()
    lats, lons = lats.ravel(), lons.ravel()
    lons = np.where(lons > 180, lons - 360, lons)
    box = (lons > -89.8) & (lons < -74.2) & (lats > 36.0) & (lats < 42.8)
    bidx = np.where(box)[0]
    bx, by = lons[bidx], lats[bidx]
    print(f"grid {len(lats):,} pts; 4-state bbox {len(bidx):,} pts", flush=True)

    masks = {}
    for fips, ring_list in county_rings().items():
        pts = np.vstack([np.asarray(r) for r in ring_list])
        sel = ((bx >= pts[:, 0].min() - 0.02) & (bx <= pts[:, 0].max() + 0.02)
               & (by >= pts[:, 1].min() - 0.02) & (by <= pts[:, 1].max() + 0.02))
        cand = np.where(sel)[0]
        m = inside(ring_list, bx[cand], by[cand])
        masks[fips] = bidx[cand[m]]
    sizes = np.array([len(v) for v in masks.values()])
    print(f"counties: {len(masks)}; grid pts per county min/med/max "
          f"{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}", flush=True)
    return masks


def main():
    import pygrib
    print("downloading 216 GUST fields (byte-ranged)...", flush=True)
    paths = download_all()
    masks = build_masks(paths[0][1])
    fips_list = sorted(masks)
    rows = []
    for h, p in paths:
        vals = pygrib.open(p).message(1).values
        v = np.asarray(vals).ravel() * MS_TO_MPH
        for fips in fips_list:
            g = v[masks[fips]]
            rows.append((fips, h, g.mean(), g.max(),
                         np.percentile(g, 90), np.percentile(g, 95),
                         *[(g > t).mean() for t in THRESH]))
        if h % 24 == 0:
            print(f"  aggregated hour {h}", flush=True)
    df = pd.DataFrame(rows, columns=["fipsCode", "hour", "u_mean", "u_max",
                                     "u_p90", "u_p95", "u_f40", "u_f50",
                                     "u_f60", "u_f70"])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  {df.shape}", flush=True)

    # ---- validation against the provided gust (organizers' polygon mean)
    tr = pd.read_csv(os.path.join(ROOT, "DM_Train.csv"), low_memory=False)
    tr["ts"] = pd.to_datetime(tr["timestamp_et"])
    tr = tr.sort_values(["fipsCode", "ts"])
    tr["hour"] = tr.groupby("fipsCode").cumcount()
    m = tr[["fipsCode", "hour", "gust"]].merge(df, on=["fipsCode", "hour"])
    for lag in (-2, -1, 0, 1, 2):
        s = m.groupby("fipsCode")["u_mean"].shift(lag)
        r = m["gust"].corr(s)
        print(f"  corr(provided gust, u_mean at lag {lag:+d}) = {r:.4f}", flush=True)
    print(f"  ratio u_max/u_mean: median {(m.u_max/m.u_mean).median():.3f} "
          f"p99 {(m.u_max/m.u_mean).quantile(.99):.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
