"""Judge new candidate members the way this project judges everything: solo
per-horizon RMSE, then value as an ADDED member of the freeze spec, with the
add-one choice nested (chosen on four folds, applied to the fifth), a county
bootstrap on the per-horizon deltas, and each candidate's mean error
correlation with the incumbents -- the error-basis diagnostic behind law 12.

Usage: python3 eval_sota.py sx1 sx2 sx3 sx4 chronos2 ...
"""
import sys

import numpy as np
import pandas as pd

from canonical import MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP, ensemble, load, truth
from src.data import load_train
from src.validate import make_folds, score_trajectory

HZ = ["t01h", "t06h", "t24h", "t48h"]


def cells(df, t):
    p = df.set_index(["fipsCode", "hour"])["osi_pred"]
    y = np.array([t.loc[f, h] if h in t.columns else np.nan for f, h in p.index])
    return pd.DataFrame({"f": p.index.get_level_values(0), "h": p.index.get_level_values(1),
                         "e": p.to_numpy() - y}).dropna()


def mrmse(d, m):
    return float(np.mean([np.sqrt((d[m & (d.h >= 72 + int(h[1:3]))].e ** 2).mean()) for h in HZ]))


def main(cands):
    t = truth()
    base = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
    fold_of = {f: k for k, (_, va) in enumerate(make_folds(load_train(), 5)) for f in va}

    print("SOLO per-horizon RMSE (x1e3)")
    rows = {}
    for c in cands + ["a3x25", "x9", "p1b"]:
        try:
            s = score_trajectory(load(c), t)
            rows[c] = [s.loc[h, "rmse"] for h in HZ] + [s.loc["mean", "rmse"]]
        except FileNotFoundError:
            print(f"  (no oof for {c})")
    print((pd.DataFrame(rows, index=HZ + ["mean"]).T * 1e3).round(4).to_string())

    # error-basis diagnostic: mean |corr| of errors with the incumbent members
    inc = {m: cells(load(m), t).set_index(["f", "h"]).e for m in base + ["a3x25"]}
    print("\nERROR CORRELATION with incumbents (lower = more diverse; p1b is the floor)")
    for c in cands + ["p1b"]:
        try:
            e = cells(load(c), t).set_index(["f", "h"]).e
        except FileNotFoundError:
            continue
        cs = []
        for m, em in inc.items():
            if m == c:
                continue
            j = e.index.intersection(em.index)
            cs.append(np.corrcoef(e.reindex(j), em.reindex(j))[0, 1])
        print(f"  {c:12} mean corr {np.mean(cs):.3f}   min {np.min(cs):.3f}")

    print("\nAS AN ADDED MEMBER of the freeze spec (equal weight, 50/50 with a3x25, tail)")
    cur = ensemble(base, t, blend_with="a3x25")
    C0 = cells(cur, t); C0["fold"] = C0.f.map(fold_of)
    r0 = mrmse(C0, pd.Series(True, index=C0.index))
    print(f"  {'current freeze spec':28} {r0:.6f}")
    fips = sorted(C0.f.unique()); rng = np.random.default_rng(42)
    idx = rng.integers(0, len(fips), size=(3000, len(fips)))
    res = {}
    for c in cands:
        try:
            df = ensemble(base + [c], t, blend_with="a3x25")
        except FileNotFoundError:
            continue
        C = cells(df, t); C["fold"] = C.f.map(fold_of)
        r = mrmse(C, pd.Series(True, index=C.index))
        # nested add-or-not
        num = {h: [0.0, 0] for h in HZ}; picks = []
        for k in range(5):
            add = mrmse(C, C.fold != k) < mrmse(C0, C0.fold != k)
            picks.append("add" if add else "keep"); D = C if add else C0
            for h in HZ:
                s = D[(D.fold == k) & (D.h >= 72 + int(h[1:3]))]
                num[h][0] += (s.e ** 2).sum(); num[h][1] += len(s)
        nested = float(np.mean([np.sqrt(num[h][0] / num[h][1]) for h in HZ]))
        # bootstrap per horizon
        wins = []
        for h in HZ:
            k = int(h[1:3])
            g = {n: X[X.h >= 72 + k].assign(e2=lambda d: d.e ** 2).groupby("f").e2.agg(["sum", "count"]).reindex(fips).fillna(0)
                 for n, X in (("new", C), ("cur", C0))}
            rr = {n: np.sqrt(g[n]["sum"].to_numpy()[idx].sum(1) / g[n]["count"].to_numpy()[idx].sum(1)) for n in g}
            wins.append((rr["new"] < rr["cur"]).mean())
        res[c] = (r, nested, picks, wins)
        print(f"  + {c:26} {r:.6f}  nested {nested:.6f}  picks {picks}  "
              f"P(better) per horizon {[round(w, 2) for w in wins]}")
    if res:
        best = min(res, key=lambda k: res[k][1])
        print(f"\nbest by NESTED score: + {best} -> {res[best][1]:.6f} "
              f"({'beats' if res[best][1] < r0 else 'does not beat'} current {r0:.6f})")


if __name__ == "__main__":
    main(sys.argv[1:] or ["sx1", "sx2", "sx3", "sx4", "chronos2", "chronos2_med"])
