"""Standalone format/sanity audit of a submission file.

Independent of `make_submission.py` on purpose: it re-derives every expectation
from `sample_submission.csv` and the competition calendar rather than trusting
the writer that produced the file. Safe to run on any candidate or contingency
submission, including ones built by an earlier revision of the code.

Usage:
  python3 verify_submission.py                       # audits the shipped file
  python3 verify_submission.py <path> [<path> ...]   # audits specific files
"""
import hashlib
import sys

import numpy as np
import pandas as pd

from src.data import HORIZONS, ROOT

DEFAULT = ROOT / "submissions" / "submission_ensemble_final.csv"
T0 = pd.Timestamp("2026-03-11 00:00:00")
LAST_HOUR = 215  # ground truth ends Mar 19; targets beyond this must be NaN
ID_COLS = ["fipsCode", "countyName", "stateAbbr", "timestamp_et"]
# Scoreable-row counts published in `evaluation_procedure.pdf` (the N_k in the
# organisers' RMSE definition). Our NaN pattern is derived independently from
# the calendar; asserting both agree catches an off-by-one in either.
SCOREABLE = {"t01h": 9009, "t06h": 8694, "t24h": 7560, "t48h": 6048}


def audit(path):
    checks, tmpl = [], pd.read_csv(ROOT / "sample_submission.csv", dtype=str)
    sub = pd.read_csv(path, dtype={c: str for c in ID_COLS})

    def ck(ok, label, detail=""):
        checks.append((bool(ok), label, detail))

    ck(len(sub) == len(tmpl) == 9072, "row count is 9072", f"got {len(sub)}")

    pred_cols = [f"osi_target_{n}" for n in HORIZONS]
    ck(list(sub.columns) == ID_COLS + pred_cols, "column names and order match the template",
       f"got {list(sub.columns)}")

    for c in ID_COLS:
        same = c in sub.columns and (sub[c].to_numpy() == tmpl[c].to_numpy()).all()
        ck(same, f"identifier column '{c}' is byte-identical to the template")

    # Expected NaN pattern is a property of the calendar, not of the model: a
    # target is unscorable exactly when origin + horizon runs past the window.
    origin_h = ((pd.to_datetime(sub["timestamp_et"]) - T0).dt.total_seconds() // 3600).astype(int)
    for name, h in HORIZONS.items():
        col = f"osi_target_{name}"
        v = pd.to_numeric(sub[col], errors="coerce")
        expect_nan = (origin_h + h) > LAST_HOUR
        ck((v.isna().to_numpy() == expect_nan.to_numpy()).all(),
           f"{col}: NaN exactly where origin+{h}h > h{LAST_HOUR}",
           f"expected {int(expect_nan.sum())} NaN, got {int(v.isna().sum())}")
        finite = v[v.notna()]
        ck(len(finite) == SCOREABLE[name],
           f"{col}: {SCOREABLE[name]} scoreable rows, matching evaluation_procedure.pdf",
           f"expected {SCOREABLE[name]}, got {len(finite)}")
        ck(len(finite) and np.isfinite(finite).all(), f"{col}: no inf values")
        ck(len(finite) and (finite >= 0).all(), f"{col}: all predictions >= 0",
           f"min {finite.min():.6f}" if len(finite) else "")
        ck(len(finite) and (finite <= 1).all(), f"{col}: all predictions <= 1 (OSI is a share)",
           f"max {finite.max():.6f}" if len(finite) else "")

    ck(not sub.duplicated(subset=["fipsCode", "timestamp_et"]).any(),
       "no duplicated (county, timestamp) rows")
    ck(sub["fipsCode"].nunique() == 63, "exactly 63 counties",
       f"got {sub['fipsCode'].nunique()}")

    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    npred = {c: int(pd.to_numeric(sub[c], errors="coerce").notna().sum()) for c in pred_cols}
    failed = [c for c in checks if not c[0]]
    print(f"\n=== {path}")
    print(f"sha256 {sha}")
    print(f"populated predictions per column: {npred}")
    for ok, label, detail in checks:
        if not ok:
            print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed"
          + ("  -- ALL OK" if not failed else f"  -- {len(failed)} FAILED"))
    return not failed


if __name__ == "__main__":
    paths = sys.argv[1:] or [DEFAULT]
    sys.exit(0 if all([audit(p) for p in paths]) else 1)
