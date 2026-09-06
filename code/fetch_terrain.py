"""County terrain/exposure features from public point-elevation APIs.

No GIS stack required: county polygons come from the Census TIGERweb REST API as
plain JSON rings, point-in-polygon is a ~15-line numpy ray-cast, and elevations
come from the Open-Meteo elevation endpoint (Copernicus DEM GLO-90, 100 points
per request, no key).

Output: data_static/county_terrain.csv
"""
import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

TIGER = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
         "State_County/MapServer/1/query")
ELEV = "https://api.opentopodata.org/v1/srtm30m"
STATES = {"18": "IN", "39": "OH", "42": "PA", "54": "WV"}
NPTS = 60          # sample points per county
BATCH = 100        # Open-Meteo elevation max coords per request


def get(url, timeout=120, tries=5):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as f:
                return json.load(f)
        except Exception as exc:
            if a == tries - 1:
                raise
            time.sleep(2 * (a + 1))


def county_polygons(state_fips):
    url = (f"{TIGER}?where=STATE%3D%27{state_fips}%27&outFields=GEOID,NAME"
           f"&returnGeometry=true&outSR=4326&maxAllowableOffset=0.001&f=json")
    d = get(url)
    return {f["attributes"]["GEOID"]: f["geometry"]["rings"] for f in d["features"]}


def inside(rings, x, y):
    """Ray-cast point-in-polygon over all rings (even-odd rule). Vectorized over points."""
    hit = np.zeros(len(x), dtype=bool)
    for ring in rings:
        p = np.asarray(ring, dtype=float)
        x1, y1 = p[:-1, 0], p[:-1, 1]
        x2, y2 = p[1:, 0], p[1:, 1]
        # for each edge, does a ray to +x from (x,y) cross it?
        cond = ((y1[None, :] > y[:, None]) != (y2[None, :] > y[:, None]))
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1)[None, :] * (y[:, None] - y1[None, :]) / \
                   (y2 - y1)[None, :] + x1[None, :]
        hit ^= (cond & (x[:, None] < xint)).sum(axis=1) % 2 == 1
    return hit


def sample_county(rings, n, seed):
    pts = np.vstack([np.asarray(r, dtype=float) for r in rings])
    lo_x, hi_x = pts[:, 0].min(), pts[:, 0].max()
    lo_y, hi_y = pts[:, 1].min(), pts[:, 1].max()
    rng = np.random.default_rng(seed)
    keep_x, keep_y = [], []
    for _ in range(60):                      # rejection sampling
        m = max(4 * n, 400)
        x = rng.uniform(lo_x, hi_x, m)
        y = rng.uniform(lo_y, hi_y, m)
        k = inside(rings, x, y)
        keep_x.append(x[k])
        keep_y.append(y[k])
        if sum(len(a) for a in keep_x) >= n:
            break
    x = np.concatenate(keep_x)[:n]
    y = np.concatenate(keep_y)[:n]
    if len(x) < n:                            # degenerate polygon -> pad with centroid
        x = np.r_[x, np.full(n - len(x), pts[:, 0].mean())]
        y = np.r_[y, np.full(n - len(y), pts[:, 1].mean())]
    return np.round(x, 4), np.round(y, 4)


def main():
    lats, lons, keys = [], [], []
    for sf in STATES:
        polys = county_polygons(sf)
        print(f"  {STATES[sf]}: {len(polys)} county polygons", flush=True)
        for geoid, rings in sorted(polys.items()):
            x, y = sample_county(rings, NPTS, seed=int(geoid))
            lons.extend(x.tolist())
            lats.extend(y.tolist())
            keys.extend([geoid] * NPTS)

    n = len(lats)
    print(f"querying {n} points in batches of {BATCH}", flush=True)
    elev = []
    for i in range(0, n, BATCH):
        locs = "|".join(f"{lats[j]},{lons[j]}" for j in range(i, min(i + BATCH, n)))
        d = get(f"{ELEV}?locations={locs}")
        elev.extend(r["elevation"] for r in d["results"])
        if (i // BATCH) % 25 == 0:
            print(f"  {len(elev)}/{n}", flush=True)
        time.sleep(1.05)                     # OpenTopoData: 1 call/sec

    df = pd.DataFrame({"fipsCode": keys, "lat": lats, "lon": lons, "elev": elev})
    df["elev"] = pd.to_numeric(df["elev"], errors="coerce")

    def agg(g):
        e = g["elev"].to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        # ruggedness: mean |elevation difference| between each point and its
        # nearest neighbours -- a sample-based analogue of the Terrain Ruggedness Index
        la = np.radians(g["lat"].to_numpy()); lo = np.radians(g["lon"].to_numpy())
        xy = np.c_[lo * np.cos(la.mean()), la] * 6371.0
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = d2.argmin(1)
        dz = np.abs(g["elev"].to_numpy(dtype=float) - g["elev"].to_numpy(dtype=float)[nn])
        dist = np.sqrt(d2[np.arange(len(nn)), nn])
        return pd.Series({
            "elev_mean": e.mean(),
            "elev_std": e.std(ddof=1),
            "elev_min": e.min(),
            "elev_max": e.max(),
            "elev_range": e.max() - e.min(),
            "elev_p90_p10": np.percentile(e, 90) - np.percentile(e, 10),
            "tri_km": np.nanmean(dz),
            "slope_pct": np.nanmean(dz / np.maximum(dist, 1e-6)) / 10.0,
        })

    out = df.groupby("fipsCode").apply(agg, include_groups=False).reset_index()
    out["fipsCode"] = out["fipsCode"].astype(int)
    dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/county_terrain.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  {out.shape}")
    print(out.head().to_string())


if __name__ == "__main__":
    main()
