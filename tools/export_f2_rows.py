"""Export the f2 member's exact tabular rows, per CV fold, for models that must
run outside the pinned environment (TabPFN). Same builder calls, same feature
columns, same finite-target mask as `ideas10.f2_member_upgrade`, so any model
scored on these files is a drop-in replacement for XGBoost in that member --
the only variable is the learner."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_train, mask_after_freeze
from src.validate import make_folds
from src.ideas6 import _rows
from src.ideas10 import add_indep, feature_cols

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tabpfn_rows")
OUT.mkdir(exist_ok=True, parents=True)
FAM_DROP = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
            "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
train = load_train()
drop_cols = ["severity_tier", "peak_pct", "peak_customers", "time_to_restore_h", "split"]
for k, (tr_f, va_f) in enumerate(make_folds(train, 5)):
    tr_df = train[train.fipsCode.isin(tr_f)]
    va_df = mask_after_freeze(train[train.fipsCode.isin(va_f)]).drop(
        columns=[c for c in drop_cols if c in train.columns])
    tr, y = _rows(tr_df, statics=True, sub=True)
    va = _rows(va_df, statics=True, sub=True, val=True)
    tr, va = add_indep(tr, mask_after_freeze(tr_df)), add_indep(va, va_df)
    tr = tr.drop(columns=[c for c in FAM_DROP if c in tr.columns])
    va = va.drop(columns=[c for c in FAM_DROP if c in va.columns])
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    X = tr.loc[keep, cols].copy(); X["_sqrt_y"] = np.sqrt(y[keep])
    X.to_csv(OUT / f"fold{k}_train.csv", index=False)
    meta = [c for c in va.columns if c not in cols]
    va[cols + meta].to_csv(OUT / f"fold{k}_val.csv", index=False)
    print(f"fold {k}: train {X.shape}  val {va.shape}  meta cols {meta}")
