"""Tabular foundation models as drop-in learners for the f2 member.

Identical rows, features, sqrt target and finite-target mask as
`ideas10.f2_member_upgrade` (exported by tools/export_f2_rows.py), per CV fold.
The only variable is the learner. Runs in the isolated venv (torch 2.x).

  --model tabdpt      TabDPT (Layer 6; Apache-2.0; retrieval-based ICL, so it
                      takes the full ~27k-row training context)
  --model tabicl      TabICLv2 (Inria; BSD-3; regression + classification)
  --model contexttab  ConTextTab (SAP)
  --model tabpfn      TabPFN-v2 (needs TABPFN_TOKEN after license acceptance)
  --model mitra       Mitra (Amazon; Apache-2.0) via AutoGluon, zero-shot, separate venv
  --ctx N             cap the training context at N rows (seeded random subsample)
                      for models with a pretraining-size limit; 0 = all rows

Writes oof/<model>_f2.csv. Measurements for the report, not submission
candidates: pretrained checkpoints are external data of a kind the rules'
"self-contained code on the provided files" clause does not admit, and the
pinned torch 1.12 cannot load them.
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--rows", required=True); ap.add_argument("--root", required=True)
ap.add_argument("--ctx", type=int, default=0); ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--ag_mem_ratio", type=float, default=1.0,
                help="mitra only: AutoGluon ag.max_memory_usage_ratio (it estimates ~7 GB; raise when the machine is quiet)")
a = ap.parse_args()
torch.set_num_threads(a.threads)
ROWS, ROOT = Path(a.rows), Path(a.root)


def make(model):
    if model == "tabdpt":
        from tabdpt import TabDPTRegressor
        return TabDPTRegressor(device="cpu")
    if model == "tabicl":
        from tabicl import TabICLRegressor
        return TabICLRegressor(device="cpu")
    if model == "contexttab":
        from contexttab import ConTextTabRegressor
        return ConTextTabRegressor()
    if model == "mitra":
        return None                      # built inside the loop (AutoGluon API)
    if model == "tabpfn":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(device="cpu", random_state=42)
    raise SystemExit(f"unknown model {model}")


outs = []
part = lambda k: ROOT / "oof" / f"{a.model}_f2.fold{k}.csv"      # per-fold checkpoint
for k in [int(x) for x in a.folds.split(",")]:
    if part(k).exists():                                          # resume after a crash
        outs.append(pd.read_csv(part(k))); print(f"fold {k}: resumed from checkpoint", flush=True); continue
    tr = pd.read_csv(ROWS / f"fold{k}_train.csv"); va = pd.read_csv(ROWS / f"fold{k}_val.csv")
    cols = [c for c in tr.columns if c != "_sqrt_y"]
    ctx = tr.sample(n=a.ctx, random_state=42) if a.ctx and a.ctx < len(tr) else tr
    t0 = time.time(); mdl = make(a.model)
    Xtr, ytr, Xva = ctx[cols].to_numpy(np.float32), ctx["_sqrt_y"].to_numpy(np.float32), va[cols].to_numpy(np.float32)
    if a.model == "contexttab":
        Xtr, Xva = ctx[cols], va[cols]
    if a.model == "mitra":
        # Mitra (Amazon, NeurIPS 2025) ships only through AutoGluon's TabularPredictor.
        # Zero-shot (fine_tune=False) so it is an in-context learner like the others.
        import shutil, tempfile
        from autogluon.tabular import TabularPredictor
        d = tempfile.mkdtemp(prefix=f"ag_mitra_fold{k}_")
        trdf = ctx[cols].copy(); trdf["_sqrt_y"] = ctx["_sqrt_y"].to_numpy()
        pr = TabularPredictor(label="_sqrt_y", problem_type="regression", verbosity=0, path=d).fit(
            trdf, hyperparameters={"MITRA": {"fine_tune": False}}, num_cpus=a.threads,
            ag_args_fit={"ag.max_memory_usage_ratio": a.ag_mem_ratio})
        pred = np.asarray(pr.predict(va[cols]), dtype=float); shutil.rmtree(d, ignore_errors=True)
    else:
        mdl.fit(Xtr, ytr)
        pred = np.asarray(mdl.predict(Xva), dtype=float)
    p = np.clip(pred, 0, None) ** 2
    outs.append(pd.DataFrame({"fipsCode": va["fipsCode"], "hour": va["hour"], "osi_pred": p}))
    outs[-1].to_csv(part(k), index=False)
    print(f"fold {k}: ctx {len(ctx)} val {len(va)}  {time.time()-t0:.0f}s", flush=True)
out = ROOT / "oof" / f"{a.model}_f2.csv"
pd.concat(outs).to_csv(out, index=False); print("wrote", out)
