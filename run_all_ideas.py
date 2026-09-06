"""CV every candidate idea under the identical county-holdout protocol.

Usage:
  python3 run_all_ideas.py                 # all ideas
  python3 run_all_ideas.py --only I01,I13  # a subset (prefix match)
Results append to results/ideas_results.csv; OOF predictions cache to oof/.
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_train
from src.validate import run_cv
from src.ideas import IDEAS
try:
    from src.ideas2 import IDEAS2
    IDEAS = {**IDEAS, **IDEAS2}
except ImportError:
    pass
try:
    from src.ideas3 import IDEAS3
    IDEAS = {**IDEAS, **IDEAS3}
except ImportError:
    pass
try:
    from src.ideas4 import IDEAS4
    IDEAS = {**IDEAS, **IDEAS4}
except ImportError:
    pass
try:
    from src.ideas5 import IDEAS5
    IDEAS = {**IDEAS, **IDEAS5}
except ImportError:
    pass
try:
    from src.ideas6 import IDEAS6
    IDEAS = {**IDEAS, **IDEAS6}
except ImportError:
    pass
try:
    from src.ideas7 import IDEAS7
    IDEAS = {**IDEAS, **IDEAS7}
except ImportError:
    pass
try:
    from src.ideas8 import IDEAS8
    IDEAS = {**IDEAS, **IDEAS8}
except ImportError:
    pass
try:
    from src.ideas9 import IDEAS9
    IDEAS = {**IDEAS, **IDEAS9}
except ImportError:
    pass
try:
    from src.ideas10 import IDEAS10
    IDEAS = {**IDEAS, **IDEAS10}
except ImportError:
    pass
try:
    from src.ideas11 import IDEAS11
    IDEAS = {**IDEAS, **IDEAS11}
except ImportError:
    pass
try:
    from src.ideas12 import IDEAS12
    IDEAS = {**IDEAS, **IDEAS12}
except ImportError:
    pass
try:
    from src.ideas13 import IDEAS13
    IDEAS = {**IDEAS, **IDEAS13}
except ImportError:
    pass
try:
    from src.ideas14 import IDEAS14
    IDEAS = {**IDEAS, **IDEAS14}
except ImportError:
    pass
try:
    from src.ideas15 import IDEAS15
    IDEAS = {**IDEAS, **IDEAS15}
except ImportError:
    pass
try:
    from src.ideas16 import IDEAS16
    IDEAS = {**IDEAS, **IDEAS16}
    from src.ideas17 import IDEAS17
    IDEAS = {**IDEAS, **IDEAS17}
    from src.ideas18 import IDEAS18
    IDEAS = {**IDEAS, **IDEAS18}
except ImportError:
    pass

RESULTS = Path("results/ideas_results.csv")
OOF = Path("oof")


def slug(name):
    return name.split()[0].lower().replace("+", "")


def main():
    only = None
    if "--only" in sys.argv:
        only = [s.strip() for s in sys.argv[sys.argv.index("--only") + 1].split(",")]
    RESULTS.parent.mkdir(exist_ok=True)
    OOF.mkdir(exist_ok=True)
    train = load_train()

    for name, fc in IDEAS.items():
        if only and not any(name.startswith(p) for p in only):
            continue
        t0 = time.time()
        try:
            table, pooled, folds = run_cv(fc, train, per_fold=True)
            pooled.to_csv(OOF / f"{slug(name)}.csv", index=False)
            row = {"idea": name,
                   "rmse_mean": table.loc["mean", "rmse"], "mae_mean": table.loc["mean", "mae"],
                   **{f"rmse_{h}": table.loc[h, "rmse"] for h in ["t01h", "t06h", "t24h", "t48h"]},
                   "fold_spread": float(np.std([f.loc["mean", "rmse"] for f in folds])),
                   "folds": " ".join(f"{f.loc['mean','rmse']:.5f}" for f in folds),
                   "secs": round(time.time() - t0)}
            print(f"[OK]   {name:38s} RMSE {row['rmse_mean']:.5f}  MAE {row['mae_mean']:.5f}"
                  f"  ({row['secs']}s)", flush=True)
        except Exception as exc:  # keep the batch alive; record the failure
            traceback.print_exc()
            row = {"idea": name, "rmse_mean": np.nan, "mae_mean": np.nan,
                   "folds": f"FAILED: {exc}", "secs": round(time.time() - t0)}
            print(f"[FAIL] {name}: {exc}", flush=True)
        pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)


if __name__ == "__main__":
    main()
