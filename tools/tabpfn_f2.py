"""TabPFN-v2 regressor (Hollmann et al., Nature 2025) as a drop-in for XGBoost in
the f2 member: identical rows, features, sqrt target and finite mask, per fold.
Runs in the separate venv (torch 2.x). TabPFN's pretraining context is 10,000
rows; each fold has ~27,000, so the training context is a seeded random
subsample of 10,000 -- stated here because it is the one place this is NOT a
like-for-like comparison with XGBoost, which sees every row.
Writes oof/tabpfn_f2.csv. Measurement for the report, not a submission
candidate (pretrained checkpoint = external data; pinned torch cannot load it)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
torch.set_num_threads(2)
from tabpfn import TabPFNRegressor

ROWS, ROOT = Path(sys.argv[1]), Path(sys.argv[2])
N_CTX = 10_000
outs = []
for k in range(5):
    tr = pd.read_csv(ROWS / f"fold{k}_train.csv"); va = pd.read_csv(ROWS / f"fold{k}_val.csv")
    cols = [c for c in tr.columns if c != "_sqrt_y"]
    sub = tr.sample(n=min(N_CTX, len(tr)), random_state=42)
    t0 = time.time()
    mdl = TabPFNRegressor(device="cpu", random_state=42)
    mdl.fit(sub[cols].to_numpy(np.float32), sub["_sqrt_y"].to_numpy(np.float32))
    p = np.clip(mdl.predict(va[cols].to_numpy(np.float32)), 0, None) ** 2
    hour_col = "hour" if "hour" in va.columns else [c for c in va.columns if "hour" in c][0]
    o = pd.DataFrame({"fipsCode": va["fipsCode"], "hour": va[hour_col], "osi_pred": p})
    outs.append(o)
    print(f"fold {k}: ctx {len(sub)} val {len(va)}  {time.time()-t0:.0f}s", flush=True)
pd.concat(outs).to_csv(ROOT / "oof" / "tabpfn_f2.csv", index=False)
print("wrote oof/tabpfn_f2.csv")
