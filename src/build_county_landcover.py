"""Build static county-level land-cover / tree-canopy-proxy features.

Source: USDA NASS CropScape ``GetCDLStat`` web service, which returns *pre-computed
zonal statistics* of the Cropland Data Layer for a whole county given its 5-digit
FIPS code.  The non-agricultural classes of the CDL (values 111-195: water,
developed 121-124, barren, deciduous/evergreen/mixed forest 141/142/143,
shrubland, grass-pasture, woody + herbaceous wetlands) are taken directly from
the most recent NLCD, so this is an NLCD land-cover summary delivered as a table
instead of a 1.4 GB national raster.  MRLC / NLCD is one of the three
infrastructure sources the organizers list in weather_variables.pdf.

No GIS stack required: pure requests + pandas.
Runtime measured on this machine: 302 counties in ~145 s, 0 failures.

Usage:
    python3 src/build_county_landcover.py
Writes:
    data_static/county_landcover.csv     (one row per county, fractions of area)
    data_static/county_landcover_raw.csv (long form: one row per county x CDL class)
"""

from __future__ import annotations

import os
import re
import time
import urllib.request

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(REPO, "data_static")

STAT_URL = (
    "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat"
    "?year={year}&fips={fips}&format=json"
)
# The service replies with an XML envelope holding a <returnURL> that points at a
# cached, almost-JSON document (unquoted keys), so it is parsed with a regex.
RETURN_URL = re.compile(r"<returnURL>(.*?)</returnURL>")
ROW = re.compile(
    r'\{value:(\d+), count:(\d+), category:"([^"]*)", color:"[^"]*", acreage:([\d.]+)\}'
)

# CDL class values.  141/142/143 and 190 are the NLCD-derived woody classes.
FOREST = [141, 142, 143]
DECIDUOUS = [141]
EVERGREEN_MIXED = [142, 143]
WOODY_WETLAND = [190]
WOODY_ALL = [141, 142, 143, 190]
SHRUB = [152]
DEVELOPED = [121, 122, 123, 124]
DEV_OPEN = [121]          # lawns / parks / roadside — where street trees live
DEV_MED_HIGH = [123, 124]
GRASS_PASTURE = [176]
WATER = [111]
BARREN = [131]
HERB_WETLAND = [195]
CROP = (
    set(range(1, 62)) | set(range(66, 81)) | set(range(92, 100)) | set(range(204, 256))
)


def _get(url: str, timeout: int = 240) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def fetch_county(fips: str, year: int = 2024, retries: int = 3) -> list[dict]:
    """Return the CDL class histogram for one county, or raise after `retries`."""
    last = None
    for attempt in range(retries):
        try:
            envelope = _get(STAT_URL.format(year=year, fips=fips))
            match = RETURN_URL.search(envelope)
            if not match:
                raise RuntimeError(f"no returnURL: {envelope[:200]}")
            payload = _get(match.group(1))
            rows = ROW.findall(payload)
            if not rows:
                raise RuntimeError(f"no rows: {payload[:200]}")
            return [
                {
                    "fipsCode": fips,
                    "value": int(v),
                    "category": cat,
                    "count": int(c),
                    "acreage": float(ac),
                }
                for v, c, cat, ac in rows
            ]
        except Exception as exc:  # noqa: BLE001 — transient network/service errors
            last = exc
            if attempt < retries - 1:
                time.sleep(1.5)
    raise RuntimeError(f"{fips} failed after {retries} tries: {last!r}")


def harvest(fips_list, year: int = 2024) -> pd.DataFrame:
    records, failures = [], []
    start = time.time()
    for i, fips in enumerate(fips_list, 1):
        try:
            records.extend(fetch_county(fips, year))
        except Exception as exc:  # noqa: BLE001
            failures.append((fips, repr(exc)))
        if i % 50 == 0:
            print(f"  {i}/{len(fips_list)}  {time.time() - start:.0f}s  fails={len(failures)}")
    if failures:
        print(f"WARNING: {len(failures)} counties failed: {failures[:5]}")
    return pd.DataFrame(records)


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    total = raw.groupby("fipsCode")["acreage"].sum().rename("cdl_total_acres")

    def frac(values, name):
        part = raw[raw["value"].isin(values)].groupby("fipsCode")["acreage"].sum()
        return (part / total).reindex(total.index).fillna(0.0).rename(name)

    out = pd.concat(
        [
            total,
            frac(FOREST, "pct_forest"),
            frac(DECIDUOUS, "pct_deciduous"),
            frac(EVERGREEN_MIXED, "pct_evergreen_mixed"),
            frac(WOODY_WETLAND, "pct_woody_wetland"),
            frac(WOODY_ALL, "pct_woody_total"),
            frac(SHRUB, "pct_shrub"),
            frac(DEVELOPED, "pct_developed"),
            frac(DEV_OPEN, "pct_dev_open"),
            frac(DEV_MED_HIGH, "pct_dev_med_high"),
            frac(GRASS_PASTURE, "pct_grass_pasture"),
            frac(WATER, "pct_water"),
            frac(BARREN, "pct_barren"),
            frac(HERB_WETLAND, "pct_herb_wetland"),
            frac(sorted(CROP), "pct_cropland"),
        ],
        axis=1,
    ).reset_index()

    # Trees standing next to conductors are the failure mechanism, so combine
    # closed-canopy forest, forested wetland, and a share of developed-open-space
    # (residential lawns and roadsides, where NLCD canopy is patchy but present).
    out["tree_proxy"] = out["pct_woody_total"] + 0.15 * out["pct_dev_open"]
    # Interface between forest and built-up land: where lines run under trees.
    out["wui_interaction"] = out["pct_woody_total"] * out["pct_developed"]
    return out


def main(year: int = 2024) -> None:
    static = pd.read_csv(
        os.path.join(STATIC, "county_static.csv"), dtype={"fipsCode": str}
    )
    fips_list = [f.zfill(5) for f in static["fipsCode"]]

    # The CropScape service throttles hard under repeated bulk use — it started
    # timing out after one full 302-county pass — so the long-form harvest is
    # cached on disk and only the missing counties are re-requested.
    raw_path = os.path.join(STATIC, "county_landcover_raw.csv")
    cached = pd.DataFrame()
    if os.path.exists(raw_path):
        cached = pd.read_csv(raw_path, dtype={"fipsCode": str})
        cached["fipsCode"] = cached["fipsCode"].str.zfill(5)
        have = set(cached["fipsCode"])
        fips_list = [f for f in fips_list if f not in have]
        print(f"cache holds {len(have)} counties; {len(fips_list)} still to fetch")

    if fips_list:
        print(f"fetching CDL {year} zonal stats for {len(fips_list)} counties")
        fresh = harvest(fips_list, year)
        raw = pd.concat([cached, fresh], ignore_index=True) if len(cached) else fresh
    else:
        raw = cached
    raw.to_csv(raw_path, index=False)

    feats = summarise(raw)
    feats.to_csv(os.path.join(STATIC, "county_landcover.csv"), index=False)
    print(f"wrote {feats.shape[0]} counties x {feats.shape[1]} cols")
    print(feats[["fipsCode", "pct_forest", "pct_woody_total", "pct_developed", "tree_proxy"]].head())


if __name__ == "__main__":
    main()
