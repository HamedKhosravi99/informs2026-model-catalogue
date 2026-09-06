"""W22: freeze-v2 members at 25 seeds. For every member with a 25-seed twin on disk, test the
single swap into v2 and the all-available-twins configuration: nested adopt-vs-v2, county
bootstrap per horizon. Members without a twin yet are kept at their v2 setting.
Usage: python3 y25_eval.py
"""
from pathlib import Path
import numpy as np, pandas as pd
from canonical import MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP, load, truth, TAIL_SPLIT_HOUR, TAIL_SCALE
from src.combine import ampshape
from src.data import load_train
from src.validate import make_folds, score_trajectory

HZ = (1, 6, 24, 48)
TWIN = {"s62": "a1x25", "s39": "x1", "i11w": "x11", "x9": "y1", "ts2c": "y2", "s35w": "y3",
        "f2": "y4", "p1b": "y5", "p1c": "y6", "r6corr": "y7"}          # x4 and a3x25 are 25-seed already


def main():
    t = truth(); train = load_train()
    base = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
    have = {m: v for m, v in TWIN.items() if (Path("oof") / f"{v}.csv").exists()}
    print("25-seed twins available:", ", ".join(f"{m}->{v}" for m, v in have.items()) or "none")
    print("  solo RMSE, 3 seeds -> 25 seeds:")
    for m, v in have.items():
        print(f"    {m:7} {score_trajectory(load(m), t).loc['mean','rmse']:.5f} -> {score_trajectory(load(v), t).loc['mean','rmse']:.5f}")
    names = sorted(set(base + list(have.values()) + ["a3x25"]))
    P = {m: load(m).set_index(["fipsCode", "hour"])["osi_pred"] for m in names}
    idx = P["a3x25"].index
    for m in names: idx = idx.intersection(P[m].index)
    f = idx.get_level_values(0).to_numpy(); h = idx.get_level_values(1).to_numpy()
    Y = np.array([t.loc[a, b] if b in t.columns else np.nan for a, b in idx]); ok = np.isfinite(Y)
    fold_of = {c: k for k, (_, va) in enumerate(make_folds(train, 5)) for c in va}; fold = np.array([fold_of[c] for c in f])
    tail = np.where(h >= TAIL_SPLIT_HOUR, TAIL_SCALE, 1.0); V = {m: P[m].reindex(idx) for m in names}; A = V["a3x25"].to_numpy()
    build = lambda ms: (0.5 * ampshape([V[m] for m in ms], idx).to_numpy() + 0.5 * A) * tail
    mr = lambda pred, m: float(np.mean([np.sqrt(((pred - Y) ** 2)[m & ok & (h >= 72 + k)].mean()) for k in HZ]))
    B = build(base); allm = np.ones(len(Y), bool); r0 = mr(B, allm)
    fips = np.array(sorted(set(f))); rng = np.random.default_rng(42); bidx = rng.integers(0, len(fips), size=(3000, len(fips))); fi = pd.Series(f)

    def boot(pred):
        ps = []
        for k in HZ:
            m = ok & (h >= 72 + k); cnt = pd.Series(m.astype(int)).groupby(fi.to_numpy()).sum().reindex(fips).fillna(0).to_numpy()
            ga = pd.Series(((pred - Y) ** 2)[m]).groupby(fi[m].to_numpy()).sum().reindex(fips).fillna(0).to_numpy()
            gb = pd.Series(((B - Y) ** 2)[m]).groupby(fi[m].to_numpy()).sum().reindex(fips).fillna(0).to_numpy()
            ps.append((np.sqrt(ga[bidx].sum(1) / cnt[bidx].sum(1)) < np.sqrt(gb[bidx].sum(1) / cnt[bidx].sum(1))).mean())
        return ps

    cfgs = {"v2 as built": base}
    for m, v in have.items(): cfgs[f"v2 with {m} -> {v}"] = [v if x == m else x for x in base]
    cfgs[f"v2 with ALL {len(have)} available twins"] = [have.get(x, x) for x in base]
    print(f"\n{'configuration':38} {'naive':>9} {'nested':>9} {'adopt':>6}  P(better) t01/t06/t24/t48")
    for name, ms in cfgs.items():
        pred = build(ms); num = {k: [0.0, 0] for k in HZ}; adopts = 0
        for k in range(5):
            tr, te = fold != k, fold == k; ad = mr(pred, tr) < mr(B, tr); adopts += ad; D = pred if ad else B
            for kk in HZ:
                s = te & ok & (h >= 72 + kk); num[kk][0] += ((D - Y) ** 2)[s].sum(); num[kk][1] += s.sum()
        nested = float(np.mean([np.sqrt(v[0] / v[1]) for v in num.values()])); ps = boot(pred) if ms != base else [0.5] * 4
        print(f"{name:38} {mr(pred, allm):9.6f} {nested:9.6f} {adopts:>3}/5  " + "  ".join(f"{p:.2f}" for p in ps))


if __name__ == "__main__":
    main()
