"""Full evaluation protocol for one candidate against its control.

Reports RMSE, MAE, per-horizon RMSE, per-fold RMSE, hour-band RMSE, tier-4 (severe)
RMSE, error correlation with the current ensemble members, and a paired county
bootstrap against the control.

Usage: python3 evaluate.py <candidate_slug> <control_slug> [ensemble_slug]
"""
import sys

import numpy as np
import pandas as pd

from src.data import FREEZE_H, HORIZONS, load_train, osi_trajectory, PRED_HOURS
from src.validate import make_folds, score_trajectory
from compare_models import county_sse

MEMBERS = ["s62", "s51", "s39", "s40", "i11", "f2", "s35", "p1b"]
N_BOOT = 4000


def main():
    cand, ctrl = sys.argv[1], sys.argv[2]
    ens = sys.argv[3] if len(sys.argv) > 3 else "zC"
    train = load_train()
    truth = osi_trajectory(train)
    tiers = train.groupby("fipsCode")["severity_tier"].first()

    def load(slug):
        return pd.read_csv(f"oof/{slug}.csv")

    print(f"{'':22s} {'candidate':>12s} {'control':>12s}")
    print(f"{'':22s} {cand:>12s} {ctrl:>12s}")
    tc, tk = score_trajectory(load(cand), truth), score_trajectory(load(ctrl), truth)
    for h in ["t01h", "t06h", "t24h", "t48h", "mean"]:
        print(f"  RMSE {h:<16s} {tc.loc[h,'rmse']:12.5f} {tk.loc[h,'rmse']:12.5f}")
    print(f"  MAE  {'mean':<16s} {tc.loc['mean','mae']:12.5f} {tk.loc['mean','mae']:12.5f}")

    print("\n  per fold (RMSE):")
    for k, (_, va) in enumerate(make_folds(train), 1):
        a = score_trajectory(load(cand).query("fipsCode in @va"), truth).loc["mean", "rmse"]
        b = score_trajectory(load(ctrl).query("fipsCode in @va"), truth).loc["mean", "rmse"]
        print(f"    fold {k}: {a:.5f} vs {b:.5f}  {'cand' if a < b else 'ctrl'}")

    print("\n  by hour band and severity tier (RMSE):")
    for slug, lab in ((cand, "cand"), (ctrl, "ctrl")):
        p = load(slug).copy()
        p["truth"] = truth.stack().reindex(
            pd.MultiIndex.from_frame(p[["fipsCode", "hour"]])).to_numpy()
        p = p.dropna(subset=["truth"])
        p["se"] = (p.osi_pred - p.truth) ** 2
        bands = pd.cut(p.hour, [72, 95, 119, 167, 215],
                       labels=["73-95 decay", "96-119 lull", "120-167 wave2", "168-215 tail"])
        b = p.groupby(bands, observed=True).se.mean().pow(0.5)
        t4 = np.sqrt(p.loc[p.fipsCode.map(tiers) == 4, "se"].mean())
        print(f"    {lab}: " + "  ".join(f"{k}={v:.5f}" for k, v in b.items())
              + f"  | tier4={t4:.5f}")

    print("\n  OOF error correlation with current ensemble members:")
    c = load(cand).rename(columns={"osi_pred": "c"})
    c["truth"] = truth.stack().reindex(
        pd.MultiIndex.from_frame(c[["fipsCode", "hour"]])).to_numpy()
    c = c.dropna(subset=["truth"])
    c["ec"] = c.c - c.truth
    cors = []
    for m in MEMBERS:
        o = load(m).rename(columns={"osi_pred": "m"})
        j = c.merge(o, on=["fipsCode", "hour"])
        cors.append((m, np.corrcoef(j.ec, j.m - j.truth)[0, 1]))
    print("    " + "  ".join(f"{m}={v:.3f}" for m, v in cors)
          + f"   (mean {np.mean([v for _, v in cors]):.3f})")

    print(f"\n  paired county bootstrap, {cand} vs {ctrl}:")
    b = county_sse(ctrl, truth)
    cc = county_sse(cand, truth).reindex(b.index)
    rng = np.random.RandomState(42)
    idx = rng.randint(0, len(b), size=(N_BOOT, len(b)))

    def rmse(a, sel=None):
        s, n = a["sse"].to_numpy(), a["n"].to_numpy()
        return np.sqrt(s.sum() / n.sum()) if sel is None else np.sqrt(s[sel].sum(1) / n[sel].sum(1))

    d = rmse(cc, idx) - rmse(b, idx)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"    delta {rmse(cc)-rmse(b):+.5f}   P(better) {(d<0).mean():.3f}"
          f"   95% CI [{lo:+.5f}, {hi:+.5f}]{'  SIGNIFICANT' if hi < 0 else ''}")


if __name__ == "__main__":
    main()
