"""Ensemble-level test for A-series candidates.

Standalone CV rank is not the decision criterion in this project: the ensemble is
saturated (REPORT section E-synthesis - E2 was better AND the most decorrelated
candidate ever measured, and still added nothing). Any A-series variant is a
modified S62, so the meaningful tests are

    swap   s62 -> candidate      (the variant supersedes its own parent)
    add    candidate as a 9th    (the variant is a distinct error basis)

Both are scored with the identical nested quiet-tail procedure used for the
shipped ensemble, so no configuration is advantaged by a differently-tuned tail.
Also prints each candidate's mean error correlation with the seven non-S62
members, which is what an ensemble seat actually pays for.

Usage:  python3 a_ensemble_swap.py [--seat MEMBER] cand1 cand2 ...
        --seat picks which incumbent the swap test replaces (default s62;
        use --seat p1b for stock-flow candidates like the M-series).
Writes oof/swap_<cand>.csv and oof/add_<cand>.csv. Touches nothing existing.
"""
import sys

import numpy as np
import pandas as pd

from src.data import load_train, osi_trajectory
from src.validate import make_folds, score_trajectory
from src.ideas4 import TAIL_SPLIT_HOUR

BASE8 = ["s62", "s51", "s39", "s40", "i11", "s45", "s35", "p1b"]
SEAT = "s62"
argv = sys.argv[1:]
if argv and argv[0] == "--seat":
    SEAT = argv[1]
    argv = argv[2:]
OTHERS = [k for k in BASE8 if k != SEAT]

train = load_train()
truth = osi_trajectory(train)
folds = list(make_folds(train))
fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}


def load(members):
    m = None
    for k in members:
        p = pd.read_csv(f"oof/{k}.csv").rename(columns={"osi_pred": k})
        m = p if m is None else m.merge(p, on=["fipsCode", "hour"])
    m["truth"] = truth.stack().reindex(
        pd.MultiIndex.from_frame(m[["fipsCode", "hour"]])).to_numpy()
    return m.dropna(subset=["truth"]).reset_index(drop=True)


def evaluate(members, tag, quiet=False):
    m = load(members)
    pred = np.mean([m[k].to_numpy() for k in members], axis=0)
    y, hour = m["truth"].to_numpy(), m["hour"].to_numpy()
    fold = m["fipsCode"].map(fold_of).to_numpy()
    tail = hour >= TAIL_SPLIT_HOUR
    nested = pred.copy()
    for k in range(len(folds)):
        a = (fold != k) & tail
        s = 0.5 * (pred[a] @ y[a]) / (pred[a] @ pred[a]) + 0.5
        nested[(fold == k) & tail] = pred[(fold == k) & tail] * s
    d = m[["fipsCode", "hour"]].copy()
    d["osi_pred"] = np.clip(nested, 0, None)
    if tag:
        d.to_csv(f"oof/{tag}.csv", index=False)
    t = score_trajectory(d, truth)
    ph = [t.loc[h, "rmse"] for h in ["t01h", "t06h", "t24h", "t48h"]]
    if not quiet:
        fw = [score_trajectory(d[d.fipsCode.isin(va)], truth).loc["mean", "rmse"]
              for _, va in folds]
        print(f"  {tag or 'base':16s} RMSE {t.loc['mean','rmse']:.5f}  "
              f"MAE {t.loc['mean','mae']:.5f}  per-h "
              + "/".join(f"{v:.5f}" for v in ph)
              + "  folds " + " ".join(f"{v:.5f}" for v in fw))
    return t.loc["mean", "rmse"], t.loc["mean", "mae"], np.array(ph)


def err_corr(cand):
    m = load(OTHERS + [cand, SEAT])
    y = m["truth"].to_numpy()
    e = {k: m[k].to_numpy() - y for k in OTHERS + [cand, SEAT]}
    c = [np.corrcoef(e[cand], e[k])[0, 1] for k in OTHERS]
    return float(np.mean(c)), float(np.corrcoef(e[cand], e[SEAT])[0, 1])


cands = list(argv)
print("shipped 8-member reference:")
r0, a0, ph0 = evaluate(BASE8, "")
print()
for c in cands:
    try:
        rs, as_, phs = evaluate([c] + OTHERS, f"swap_{SEAT}_{c}", quiet=True)
        ra, aa, pha = evaluate(BASE8 + [c], f"add_{c}", quiet=True)
    except FileNotFoundError:
        print(f"  {c}: no oof/{c}.csv — skipped")
        continue
    mc, sc = err_corr(c)
    print(f"{c}:  err-corr vs other 7 = {mc:.3f}   vs {SEAT} = {sc:.3f}")
    for lab, r, a, ph in [(f"swap {SEAT}->{c}", rs, as_, phs), (f"add {c} (9)", ra, aa, pha)]:
        wins = int((ph < ph0).sum())
        print(f"    {lab:22s} RMSE {r:.5f} ({r-r0:+.5f})  MAE {a:.5f} ({a-a0:+.5f})"
              f"  better at {wins}/4 horizons")
    print()
