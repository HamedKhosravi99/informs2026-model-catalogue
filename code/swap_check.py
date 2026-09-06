"""Nested member-SWAP test: replace one freeze-spec member with a candidate.

Usage: python3 swap_check.py <incumbent> <candidate>     e.g.  s39 s39w

Reports per-horizon RMSE for both blends, the nested choice (chosen on four
folds, applied to the fifth), the optimism gap, and a county bootstrap per
horizon. This is the procedure that admitted i11w and ts2c and rejected x9w
and s35w; it is the standing test for any future substitution."""
import sys

import numpy as np
import pandas as pd

from canonical import MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP, ensemble, truth
from src.data import load_train
from src.validate import make_folds

HZ = ["t01h", "t06h", "t24h", "t48h"]


def cells(df, t, fold_of):
    p = df.set_index(["fipsCode", "hour"])["osi_pred"]
    y = np.array([t.loc[f, h] if h in t.columns else np.nan for f, h in p.index])
    d = pd.DataFrame({"f": p.index.get_level_values(0), "h": p.index.get_level_values(1),
                      "e2": (p.to_numpy() - y) ** 2}).dropna()
    d["fold"] = d.f.map(fold_of)
    return d


def mrmse(d, m):
    return float(np.mean([np.sqrt(d[m & (d.h >= 72 + int(h[1:3]))].e2.mean()) for h in HZ]))


def per_h(d):
    return [float(np.sqrt(d[d.h >= 72 + int(h[1:3])].e2.mean())) for h in HZ]


def main(inc, cand):
    t = truth()
    fold_of = {f: k for k, (_, va) in enumerate(make_folds(load_train(), 5)) for f in va}
    base = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
    assert inc in base, f"{inc} is not a freeze member: {base}"
    alt = [cand if m == inc else m for m in base]
    A = cells(ensemble(base, t, blend_with="a3x25"), t, fold_of)
    B = cells(ensemble(alt, t, blend_with="a3x25"), t, fold_of)
    ra, rb = mrmse(A, pd.Series(True, index=A.index)), mrmse(B, pd.Series(True, index=B.index))
    print(f"per-horizon x1e3   {'t01h':>8} {'t06h':>8} {'t24h':>8} {'t48h':>8} {'mean':>8}")
    for lbl, D, r in ((f"keep {inc}", A, ra), (f"swap -> {cand}", B, rb)):
        print(f"  {lbl:16} " + " ".join(f"{v*1e3:8.4f}" for v in per_h(D)) + f" {r*1e3:8.4f}")
    num, picks = {h: [0.0, 0] for h in HZ}, []
    for k in range(5):
        sw = mrmse(B, B.fold != k) < mrmse(A, A.fold != k)
        picks.append("swap" if sw else "keep"); D = B if sw else A
        for h in HZ:
            s = D[(D.fold == k) & (D.h >= 72 + int(h[1:3]))]
            num[h][0] += s.e2.sum(); num[h][1] += len(s)
    nested = float(np.mean([np.sqrt(num[h][0] / num[h][1]) for h in HZ]))
    print(f"\n  per-fold picks {picks}")
    print(f"  nested {nested:.6f}  vs keep {ra:.6f}  -> {'SWAP pays' if nested < ra else 'does NOT pay'}"
          f"   optimism gap {nested - min(ra, rb):+.7f}")
    fips = sorted(A.f.unique()); rng = np.random.default_rng(42)
    idx = rng.integers(0, len(fips), size=(4000, len(fips)))
    ps = []
    for h in HZ:
        k = int(h[1:3])
        g = {n: X[X.h >= 72 + k].groupby("f").e2.agg(["sum", "count"]).reindex(fips).fillna(0) for n, X in (("A", A), ("B", B))}
        r = {n: np.sqrt(g[n]["sum"].to_numpy()[idx].sum(1) / g[n]["count"].to_numpy()[idx].sum(1)) for n in g}
        ps.append((r["B"] < r["A"]).mean())
    print(f"  P({cand} better) per horizon: {[round(p, 3) for p in ps]}   horizons won: {sum(p > 0.5 for p in ps)}/4")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
