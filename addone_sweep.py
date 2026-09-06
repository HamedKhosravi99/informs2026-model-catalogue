"""W21 step 1: nested add-one sweep over EVERY cached out-of-fold object not in the freeze spec.

For each candidate: the freeze spec plus that candidate as one more equal-weight member, with the
add-or-not decision made on four folds and applied to the fifth. Oracles (anything using truth
after h71, e.g. i05b) must be excluded by hand from the winners; the sweep cannot know.
Usage: python3 addone_sweep.py [rule]   (rule = mean|ampshape; default = the spec's rule)
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from canonical import MEMBERS, FREEZE_SUB, FREEZE_DROP, ensemble, load, truth
from src.data import load_train
from src.validate import make_folds, score_trajectory

HZ = (1, 6, 24, 48)


def main(rule=None):
    t = truth()
    base = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + ["ts2c"]
    fold_of = {c: k for k, (_, va) in enumerate(make_folds(load_train(), 5)) for c in va}

    def cells(df):
        p = df.set_index(["fipsCode", "hour"])["osi_pred"]
        y = np.array([t.loc[a, b] if b in t.columns else np.nan for a, b in p.index])
        d = pd.DataFrame({"f": p.index.get_level_values(0), "h": p.index.get_level_values(1),
                          "e2": (p.to_numpy() - y) ** 2}).dropna()
        d["fold"] = d.f.map(fold_of); return d

    def mr(d, m):
        return float(np.mean([np.sqrt(d[m & (d.h >= 72 + k)].e2.mean()) for k in HZ]))

    C0 = cells(ensemble(base, t, blend_with="a3x25", rule=rule)); r0 = mr(C0, pd.Series(True, index=C0.index))
    skip = set(base + ["a3x25", "cfg_final", "zero", "decay", "i05b"])
    rows = []
    for path in sorted(glob.glob("oof/*.csv")):
        c = Path(path).stem
        if c in skip or ".fold" in c:
            continue
        try:
            df = load(c)
            if df["fipsCode"].nunique() < 230 or score_trajectory(df, t).loc["mean", "rmse"] > 0.0125:
                continue
            C = cells(ensemble(base + [c], t, blend_with="a3x25", rule=rule))
            num = {k: [0.0, 0] for k in HZ}; adds = 0
            for k in range(5):
                add = mr(C, C.fold != k) < mr(C0, C0.fold != k); adds += add; D = C if add else C0
                for kk in HZ:
                    s = D[(D.fold == k) & (D.h >= 72 + kk)]; num[kk][0] += s.e2.sum(); num[kk][1] += len(s)
            rows.append((c, score_trajectory(df, t).loc["mean", "rmse"], mr(C, pd.Series(True, index=C.index)),
                         float(np.mean([np.sqrt(v[0] / v[1]) for v in num.values()])), adds))
        except Exception:
            pass
    R = pd.DataFrame(rows, columns=["candidate", "solo", "added_naive", "added_nested", "folds_adding"]).sort_values("added_nested")
    print(f"base {r0:.6f}; {len(R)} candidates scanned; beating base nested: {(R.added_nested < r0 - 1e-9).sum()}")
    print(R.head(12).to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
