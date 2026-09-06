"""Where does the built ensemble's error live, and can a zero-gate, a horizon
calibration curve, or conformal intervals do anything about it?  All from cached
out-of-fold predictions; every fitted quantity is nested (four folds -> fifth).

  1. Error anatomy: share of squared error on cells whose truth is exactly zero,
     on counties whose whole scored trajectory is zero, and on the top-13 counties.
  2. Zero gate on the ensemble: P(y>0) from a nested logistic fit on the
     ensemble's own prediction (or, if oof/f2_cls.csv exists, from a GBM
     classifier trained on f2's rows).  Hard gate y*1[P>tau] with tau nested;
     soft gate y*P^gamma with gamma nested.
  3. Per-horizon-hour calibration curve: one multiplier per block of target
     hours, nested least squares with half-shrink toward 1 (the procedure that
     produced the 0.667 tail scalar), applied to the fifth fold.
  4. Split-conformal 90% intervals per horizon column (absolute residual, and
     locally adaptive |r|/(yhat+eps)); coverage and width on held-out folds.
     Point RMSE is unchanged by construction -- this is a report deliverable.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from canonical import MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP, ensemble, load, truth
from src.data import load_train
from src.validate import make_folds

HZ = {"t01h": 1, "t06h": 6, "t24h": 24, "t48h": 48}
ROOT = Path(__file__).resolve().parent


def mrmse(d, yhat_col="yhat", mask=None):
    m = pd.Series(True, index=d.index) if mask is None else mask
    return float(np.mean([np.sqrt(((d[yhat_col] - d.y) ** 2)[m & (d.h >= 72 + k)].mean()) for k in HZ.values()]))


def main():
    t = truth()
    fz = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
    e = ensemble(fz, t, blend_with="a3x25").set_index(["fipsCode", "hour"])["osi_pred"]
    y = np.array([t.loc[f, h] if h in t.columns else np.nan for f, h in e.index])
    d = pd.DataFrame({"f": e.index.get_level_values(0), "h": e.index.get_level_values(1),
                      "yhat": e.to_numpy(), "y": y}).dropna().reset_index(drop=True)
    fold_of = {f: k for k, (_, va) in enumerate(make_folds(load_train(), 5)) for f in va}
    d["fold"] = d.f.map(fold_of)
    base = mrmse(d)
    print(f"built ensemble, nested-free baseline: {base:.6f}\n")

    # ---------------------------------------------------------------- 1. anatomy
    print("1. ERROR ANATOMY (t01h column, hours 73-215; squared-error shares)")
    se = (d.yhat - d.y) ** 2
    print(f"   cells with truth == 0 : {100 * (d.y == 0).mean():.1f}% of cells, {100 * se[d.y == 0].sum() / se.sum():.1f}% of squared error")
    allzero = d.groupby("f").y.transform("max") == 0
    print(f"   all-zero counties     : {allzero.groupby(d.f).first().sum()} of {d.f.nunique()} counties, {100 * se[allzero].sum() / se.sum():.1f}% of squared error")
    top = se.groupby(d.f).sum().sort_values(ascending=False)
    print(f"   top-13 counties       : {100 * top.iloc[:13].sum() / se.sum():.1f}% of squared error")
    print(f"   truth>0 & yhat<truth  : {100 * se[(d.y > 0) & (d.yhat < d.y)].sum() / se.sum():.1f}% (under-prediction of real outages)")
    print(f"   truth>0 & yhat>truth  : {100 * se[(d.y > 0) & (d.yhat > d.y)].sum() / se.sum():.1f}% (over-prediction of real outages)")
    zero_pred_rmse = mrmse(d.assign(yhat=np.where(d.y == 0, 0.0, d.yhat)))
    print(f"   ORACLE: if every true-zero cell were predicted exactly 0 -> {zero_pred_rmse:.6f} (ceiling for any zero gate)\n")

    # ---------------------------------------------------------------- 2. gate
    cls_path = ROOT / "oof" / "f2_cls.csv"
    if cls_path.exists():
        pc = load("f2_cls").set_index(["fipsCode", "hour"])["osi_pred"]
        d["P"] = pc.reindex(pd.MultiIndex.from_arrays([d.f, d.h])).to_numpy()
        src = "GBM classifier on f2 rows (oof/f2_cls.csv)"
    else:
        d["P"] = np.nan
        for k in range(5):
            tr, te = d.fold != k, d.fold == k
            lr = LogisticRegression().fit(np.log1p(1e3 * d.loc[tr, ["yhat"]]), (d.loc[tr, "y"] > 0).astype(int))
            d.loc[te, "P"] = lr.predict_proba(np.log1p(1e3 * d.loc[te, ["yhat"]]))[:, 1]
        src = "nested logistic on the ensemble's own prediction"
    print(f"2. ZERO GATE  (P(y>0) from {src})")
    print(f"   gate quality: AUC-ish check -> mean P on true zeros {d.P[d.y == 0].mean():.3f}, on true positives {d.P[d.y > 0].mean():.3f}")
    taus = np.arange(0.05, 0.96, 0.05); gammas = np.arange(0.0, 2.01, 0.25)
    num_h = {k: [0.0, 0] for k in HZ}; num_s = {k: [0.0, 0] for k in HZ}; picks_t, picks_g = [], []
    for k in range(5):
        tr, te = d.fold != k, d.fold == k
        tau = min(taus, key=lambda tt: mrmse(d.assign(yhat=np.where(d.P > tt, d.yhat, 0.0)), mask=tr))
        gam = min(gammas, key=lambda g: mrmse(d.assign(yhat=d.yhat * d.P ** g), mask=tr))
        picks_t.append(round(float(tau), 2)); picks_g.append(round(float(gam), 2))
        for kk, hh in HZ.items():
            s = te & (d.h >= 72 + hh)
            num_h[kk][0] += ((np.where(d.P > tau, d.yhat, 0.0) - d.y) ** 2)[s].sum(); num_h[kk][1] += s.sum()
            num_s[kk][0] += ((d.yhat * d.P ** gam - d.y) ** 2)[s].sum(); num_s[kk][1] += s.sum()
    hard = float(np.mean([np.sqrt(v[0] / v[1]) for v in num_h.values()]))
    soft = float(np.mean([np.sqrt(v[0] / v[1]) for v in num_s.values()]))
    print(f"   hard gate  y*1[P>tau]   nested {hard:.6f}  (tau per fold {picks_t})  -> {hard - base:+.6f}")
    print(f"   soft gate  y*P^gamma    nested {soft:.6f}  (gamma per fold {picks_g})  -> {soft - base:+.6f}")
    print(f"   (gamma=0 means 'do nothing'; a gate can only beat the oracle ceiling above if it is sharper than the truth itself)\n")

    # ---------------------------------------------------------------- 3. curve
    print("3. PER-HORIZON-HOUR CALIBRATION CURVE (nested LS, half-shrink, blocks of target hours)")
    edges = [73, 79, 97, 121, 145, 169, 216]
    blk = np.digitize(d.h, edges[1:-1])
    num = {k: [0.0, 0] for k in HZ}; curves = []
    for k in range(5):
        tr, te = d.fold != k, d.fold == k
        c = {}
        for b in range(len(edges) - 1):
            m = tr & (blk == b)
            ls = (d.yhat[m] * d.y[m]).sum() / max((d.yhat[m] ** 2).sum(), 1e-12)
            c[b] = 1 + 0.5 * (ls - 1)
        curves.append([round(c[b], 3) for b in range(len(edges) - 1)])
        adj = d.yhat * np.array([c[b] for b in blk])
        for kk, hh in HZ.items():
            s = te & (d.h >= 72 + hh)
            num[kk][0] += ((adj - d.y) ** 2)[s].sum(); num[kk][1] += s.sum()
    curve = float(np.mean([np.sqrt(v[0] / v[1]) for v in num.values()]))
    print(f"   blocks {edges}")
    for k, cv in enumerate(curves):
        print(f"   fold {k} multipliers {cv}")
    print(f"   nested {curve:.6f}  -> {curve - base:+.6f}  (the shipped tail scalar 0.667 on hours>=168 is already inside the baseline)\n")

    # ---------------------------------------------------------------- 4. conformal
    print("4. SPLIT-CONFORMAL 90% INTERVALS, per horizon column, calibrated on four folds, checked on the fifth")
    print(f"   {'col':6} {'abs-resid cov':>14} {'mean width':>11} | {'adaptive cov':>13} {'mean width':>11}")
    eps = 5e-4
    for kk, hh in HZ.items():
        cov_a, w_a, cov_s, w_s, n = 0, 0.0, 0, 0.0, 0
        for k in range(5):
            tr, te = (d.fold != k) & (d.h >= 72 + hh), (d.fold == k) & (d.h >= 72 + hh)
            r = (d.y - d.yhat).abs()
            q = np.quantile(r[tr], min(1.0, 0.9 * (1 + 1 / tr.sum())))
            qs = np.quantile((r / (d.yhat + eps))[tr], min(1.0, 0.9 * (1 + 1 / tr.sum())))
            cov_a += (r[te] <= q).sum(); w_a += (2 * q) * te.sum()
            cov_s += (r[te] <= qs * (d.yhat[te] + eps)).sum(); w_s += (2 * qs * (d.yhat[te] + eps)).sum()
            n += te.sum()
        print(f"   {kk:6} {cov_a / n:14.3f} {w_a / n:11.5f} | {cov_s / n:13.3f} {w_s / n:11.5f}")
    print("   caveat: cells within a county are dependent, so cell-level exchangeability is approximate;")
    print("   the folds are county splits, which is the right unit for the guarantee to be about.")


if __name__ == "__main__":
    main()
