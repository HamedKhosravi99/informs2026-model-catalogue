"""Train on all 239 counties, forecast the 63 test counties, fill the template."""
import numpy as np
import pandas as pd

from src.data import ROOT, FREEZE_H, HORIZONS, load_train, load_test, mask_after_freeze
from src.features import build_rows
# --spec freeze (default) builds the freeze-spec model, CV 0.008468, into
# submission_freeze.csv. --spec shipped rebuilds the original 8-member model
# (CV 0.008543) into submission_ensemble_final.csv, so the existing file stays
# reproducible from this script.
import sys

from src.ideas4 import (ensemble_final_forecaster, ensemble_freeze_forecaster,
                        ensemble_freeze_v2_forecaster)

_SPEC = sys.argv[sys.argv.index("--spec") + 1] if "--spec" in sys.argv else "v2"
_MEMBER_CACHE = ROOT / "submissions" / "members"          # per-member test predictions (v2)
ensemble_forecaster = {"shipped": ensemble_final_forecaster, "freeze": ensemble_freeze_forecaster,
                       "v2": lambda tr, te: ensemble_freeze_v2_forecaster(tr, te, cache_dir=_MEMBER_CACHE)}[_SPEC]
OUT_NAME = {"shipped": "submission_ensemble_final.csv", "freeze": "submission_freeze.csv",
            "v2": "submission_freeze_v2.csv"}[_SPEC]

OUT = ROOT / "submissions"


def main():
    train, test = load_train(), load_test()
    # causality guard: the test file must carry no outage data after the freeze
    from src.data import OUTAGE_COLS
    late = test[test["hour"] > FREEZE_H]
    present = [c for c in OUTAGE_COLS if c in test.columns and late[c].notna().any()]
    assert not present, f"outage data after h{FREEZE_H} in test file: {present}"

    pred = ensemble_forecaster(train, test)
    traj = pred.set_index(["fipsCode", "hour"])["osi_pred"]

    # keep identifier columns byte-identical to the template (the rules forbid
    # altering them); only the four prediction columns are written
    sub = pd.read_csv(ROOT / "sample_submission.csv", dtype=str)
    t0 = pd.Timestamp("2026-03-11 00:00:00")
    origin_h = ((pd.to_datetime(sub["timestamp_et"]) - t0).dt.total_seconds() // 3600).astype(int)
    sub["fipsCode"] = sub["fipsCode"].astype(int)
    for name, h in HORIZONS.items():
        tgt = origin_h + h
        idx = pd.MultiIndex.from_arrays([sub["fipsCode"], tgt])
        vals = traj.reindex(idx).to_numpy()
        vals[tgt.to_numpy() > 215] = np.nan  # beyond Mar 19: no ground truth
        sub[f"osi_target_{name}"] = vals

    OUT.mkdir(exist_ok=True)
    path = OUT / OUT_NAME
    sub.to_csv(path, index=False)

    tmpl = pd.read_csv(ROOT / "sample_submission.csv")
    assert len(sub) == len(tmpl) == 9072
    assert (sub["fipsCode"].to_numpy() == tmpl["fipsCode"].to_numpy()).all()
    filled = {c: int(sub[c].notna().sum()) for c in sub.columns if c.startswith("osi_")}
    print(f"wrote {path}")
    print("non-NaN predictions per column:", filled)
    print(sub.filter(like="osi_").describe().round(4).to_string())


if __name__ == "__main__":
    main()
