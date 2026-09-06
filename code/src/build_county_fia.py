"""County forest acreage from the USFS Forest Inventory and Analysis database.

Uses the public FIADB-API ``fullreport`` endpoint (no key, no registration), which
returns design-based population estimates straight out of FIADB as JSON.  This is
the same agency (USFS FIA) that builds the NLCD Tree Canopy Cover product, so it
serves as an independent cross-check on the CropScape/NLCD land-cover fractions
produced by ``build_county_landcover.py``.

Caveat worth remembering when using these numbers: FIA is a *plot sample*, not a
census.  Counties with few plots have very wide sampling errors (the API returns
``cellSE`` as a percent), and counties with no forested plots are simply absent
from the response — hence 298 of 302 counties here.  The CropScape fractions are
the better per-county measurement; FIA is the validation set.

Usage:
    python3 src/build_county_fia.py
Writes:
    data_static/county_fia_forest.csv
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(REPO, "data_static")

BASE = "https://apps.fs.usda.gov/fiadb-api/fullreport"

# Evaluation group = state FIPS + report year.  Current most-recent groups, from
# https://apps.fs.usda.gov/fiadb-api/fullreport/parameters/wc (MOST_RECENT == 'Y').
EVAL_GROUPS = {
    "18": "182025",  # Indiana
    "39": "392025",  # Ohio
    "42": "422025",  # Pennsylvania
    "54": "542024",  # West Virginia
}

# snum 2 == "Area of forest land, in acres", from /fullreport/parameters/snum.
SNUM_FOREST_ACRES = "2"

# The API rejects an empty/None column grouping, so a real one is supplied and the
# "Total" column is read back out.
CSELECTED = "Forest type group"

ROW_LABEL = re.compile(r"\s*(\d+)\s+(.*)")


def fetch_state(eval_group: str, timeout: int = 300) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "wc": eval_group,
            "snum": SNUM_FOREST_ACRES,
            "rselected": "County code and name",
            "cselected": CSELECTED,
            "outputFormat": "JSON",
        }
    )
    with urllib.request.urlopen(f"{BASE}?{query}", timeout=timeout) as resp:
        payload = json.loads(resp.read().decode())

    rows = payload["EVALIDatorOutput"]["row"]
    out = []
    for row in rows:
        label = row["content"]
        if label.strip().lower() == "total":
            continue
        totals = [c for c in row["column"] if c["content"] == "Total"]
        if not totals:
            continue
        match = ROW_LABEL.match(label)
        if not match:
            print(f"  skipping unparsed row label: {label!r}")
            continue
        # The row label already carries the full 5-digit state+county FIPS.
        out.append(
            {
                "fipsCode": match.group(1).zfill(5),
                "county_name": match.group(2).strip(),
                "fia_forest_acres": totals[0]["cellValueNumerator"],
                "fia_forest_se_pct": totals[0].get("cellSE"),
                "fia_plots": totals[0].get("cellPlotNumerator"),
            }
        )
    return out


def main() -> None:
    records = []
    for state, group in EVAL_GROUPS.items():
        start = time.time()
        rows = fetch_state(group)
        records.extend(rows)
        print(f"state {state} (wc={group}): {len(rows)} counties, {time.time() - start:.1f}s")

    fia = pd.DataFrame(records)

    static = pd.read_csv(
        os.path.join(STATIC, "county_static.csv"), dtype={"fipsCode": str}
    )
    static["fipsCode"] = static["fipsCode"].str.zfill(5)

    merged = static[["fipsCode", "land_sqmi"]].merge(fia, on="fipsCode", how="left")
    # Counties absent from FIA have no forested plots at all — treat as zero forest
    # rather than missing, which is what the sampling design implies.
    merged["fia_forest_acres"] = merged["fia_forest_acres"].fillna(0.0)
    merged["fia_pct_forest"] = merged["fia_forest_acres"] / (merged["land_sqmi"] * 640.0)

    path = os.path.join(STATIC, "county_fia_forest.csv")
    merged.drop(columns=["land_sqmi"]).to_csv(path, index=False)
    print(f"wrote {len(merged)} rows -> {path}")

    landcover = os.path.join(STATIC, "county_landcover.csv")
    if os.path.exists(landcover):
        cdl = pd.read_csv(landcover, dtype={"fipsCode": str})
        cdl["fipsCode"] = cdl["fipsCode"].str.zfill(5)
        check = merged.merge(cdl[["fipsCode", "pct_forest"]], on="fipsCode")
        check = check[check["fia_plots"].notna()]
        r = check["pct_forest"].corr(check["fia_pct_forest"])
        print(f"cross-check vs CropScape/NLCD pct_forest: r = {r:.4f} on {len(check)} counties")


if __name__ == "__main__":
    main()
