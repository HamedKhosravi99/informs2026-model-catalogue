"""W21 step 2: parameter-free combination rules and fixed candidate configurations, each with a
nested adopt-vs-base decision and a county bootstrap per horizon, plus a nested SELECTION over all
configurations (the honest post-search number).
Usage: python3 combo_rules.py
"""
import numpy as np, pandas as pd
from canonical import MEMBERS, FREEZE_SUB, FREEZE_DROP, load, truth, TAIL_SPLIT_HOUR, TAIL_SCALE
from src.combine import ampshape
from src.data import load_train
from src.validate import make_folds

HZ = (1, 6, 24, 48)


def main():
    t = truth(); train = load_train()
    base = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + ["ts2c"]
    names = base + ["p1c", "r6corr", "s35w", "a3x25"]
    P = {m: load(m).set_index(["fipsCode", "hour"])["osi_pred"] for m in names}
    idx = P["a3x25"].index
    for m in names: idx = idx.intersection(P[m].index)
    f = idx.get_level_values(0).to_numpy(); h = idx.get_level_values(1).to_numpy()
    Y = np.array([t.loc[a, b] if b in t.columns else np.nan for a, b in idx]); ok = np.isfinite(Y)
    fold_of = {c: k for k, (_, va) in enumerate(make_folds(train, 5)) for c in va}; fold = np.array([fold_of[c] for c in f])
    tail = np.where(h >= TAIL_SPLIT_HOUR, TAIL_SCALE, 1.0); V = {m: P[m].reindex(idx) for m in names}; A = V["a3x25"].to_numpy()
    mean = lambda ms: sum(V[m].to_numpy() for m in ms) / len(ms)
    amp = lambda ms: ampshape([V[m] for m in ms], idx).to_numpy()
    cfg = {"built v1 (8 mean + A)": (0.5 * mean(base) + 0.5 * A) * tail,
           "median8 + A": (0.5 * np.median(np.column_stack([V[m].to_numpy() for m in base]), 1) + 0.5 * A) * tail,
           "ampshape8 + A": (0.5 * amp(base) + 0.5 * A) * tail,
           "+p1c (mean)": (0.5 * mean(base + ["p1c"]) + 0.5 * A) * tail,
           "+p1c+s35w (mean)": (0.5 * mean(base + ["p1c", "s35w"]) + 0.5 * A) * tail,
           "+p1c+r6corr+s35w (mean)": (0.5 * mean(base + ["p1c", "r6corr", "s35w"]) + 0.5 * A) * tail,
           "ampshape10 + A (+p1c+s35w)": (0.5 * amp(base + ["p1c", "s35w"]) + 0.5 * A) * tail,
           "ampshape11 + A (+p1c+r6corr+s35w) = v2": (0.5 * amp(base + ["p1c", "r6corr", "s35w"]) + 0.5 * A) * tail}

    def mr(pred, m): return float(np.mean([np.sqrt(((pred - Y) ** 2)[m & ok & (h >= 72 + k)].mean()) for k in HZ]))
    B = cfg["built v1 (8 mean + A)"]; allm = np.ones(len(Y), bool); r0 = mr(B, allm)
    fips = np.array(sorted(set(f))); rng = np.random.default_rng(42); bidx = rng.integers(0, len(fips), size=(3000, len(fips))); fi = pd.Series(f)
    print(f"{'configuration':42} {'naive':>9} {'nested':>9} {'adopt':>6}  P(better) t01/t06/t24/t48")
    for name, pred in cfg.items():
        num = {k: [0.0, 0] for k in HZ}; adopts = 0
        for k in range(5):
            tr, te = fold != k, fold == k; ad = mr(pred, tr) < mr(B, tr); adopts += ad; D = pred if ad else B
            for kk in HZ:
                s = te & ok & (h >= 72 + kk); num[kk][0] += ((D - Y) ** 2)[s].sum(); num[kk][1] += s.sum()
        ps = []
        for k in HZ:
            m = ok & (h >= 72 + k); cnt = pd.Series(m.astype(int)).groupby(fi.to_numpy()).sum().reindex(fips).fillna(0).to_numpy()
            ga = pd.Series(((pred - Y) ** 2)[m]).groupby(fi[m].to_numpy()).sum().reindex(fips).fillna(0).to_numpy()
            gb = pd.Series(((B - Y) ** 2)[m]).groupby(fi[m].to_numpy()).sum().reindex(fips).fillna(0).to_numpy()
            ps.append((np.sqrt(ga[bidx].sum(1) / cnt[bidx].sum(1)) < np.sqrt(gb[bidx].sum(1) / cnt[bidx].sum(1))).mean())
        print(f"{name:42} {mr(pred, allm):9.6f} {float(np.mean([np.sqrt(v[0] / v[1]) for v in num.values()])):9.6f} {adopts:>3}/5  " + "  ".join(f"{p:.2f}" for p in ps))
    num = {k: [0.0, 0] for k in HZ}; picks = []
    for k in range(5):
        tr, te = fold != k, fold == k; best = min(cfg, key=lambda c: mr(cfg[c], tr)); picks.append(best)
        for kk in HZ:
            s = te & ok & (h >= 72 + kk); num[kk][0] += ((cfg[best] - Y) ** 2)[s].sum(); num[kk][1] += s.sum()
    print(f"\nNESTED SELECTION over all {len(cfg)}: {float(np.mean([np.sqrt(v[0] / v[1]) for v in num.values()])):.6f} vs built {r0:.6f}; picks {picks}")


if __name__ == "__main__":
    main()
