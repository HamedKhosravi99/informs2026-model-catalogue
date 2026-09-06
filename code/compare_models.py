"""Paired significance testing between cached OOF predictions.

Pooled fold means hide county-difficulty variance: 12 of 239 counties carry ~65%
of the scored squared error, so two models can differ by more than the noise bar
purely through which counties landed in which fold. This pairs by county and
bootstraps over counties, which removes that variance component.

Metric: the bootstrap statistic is the **official** one -- the mean of the four
per-horizon RMSEs, matching `src/validate.py` and every headline number in
REPORT.md. Until 2026-08-15 this script pooled the squared error across all four
horizons into a single RMSE, which is a different (t48h-dominated) statistic;
`--pooled` reproduces that old behaviour for back-comparison.

Usage: python3 compare_models.py <baseline_slug> <candidate_slug> [more_slugs...]
       python3 compare_models.py --pooled <baseline_slug> <candidate_slug> ...
"""
import sys

import numpy as np
import pandas as pd

from src.data import FREEZE_H, HORIZONS, load_train, osi_trajectory

N_BOOT = 4000
SEED = 42


def county_sse_by_horizon(slug, truth):
    """Per-county (sse, n) per horizon -> arrays of shape (n_county, 4).

    Kept separate rather than summed, so the bootstrap can form the official
    per-horizon-mean RMSE. Column order follows HORIZONS.
    """
    p = pd.read_csv(f"oof/{slug}.csv")
    piv = p.pivot_table(index="fipsCode", columns="hour", values="osi_pred")
    sse, n = [], []
    for h in HORIZONS.values():
        tgt = np.arange(FREEZE_H + 1 + h, 216)
        t = truth.loc[piv.index, tgt].to_numpy()
        q = piv.reindex(columns=tgt).to_numpy()
        e = (q - t) ** 2
        sse.append(np.nansum(e, 1))
        n.append(np.isfinite(e).sum(1))
    return piv.index, np.array(sse).T, np.array(n).T


def county_sse(slug, truth):
    """Back-compatible pooled view: per-county (sse, n) summed over horizons."""
    idx, sse, n = county_sse_by_horizon(slug, truth)
    return pd.DataFrame({"sse": sse.sum(1), "n": n.sum(1)}, index=idx)


def main():
    argv = sys.argv[1:]
    pooled = "--pooled" in argv
    argv = [a for a in argv if a != "--pooled"]
    if len(argv) < 2:
        print(__doc__)
        return
    truth = osi_trajectory(load_train())
    base = argv[0]
    idx, sb, nb = county_sse_by_horizon(base, truth)
    rng = np.random.RandomState(SEED)
    boot_idx = rng.randint(0, len(idx), size=(N_BOOT, len(idx)))

    def score(sse, n, sel=None):
        """Official metric: mean over horizons of sqrt(sum sse / sum n).

        With --pooled: sqrt of the horizon-summed sse over the horizon-summed n.
        """
        if sel is None:
            if pooled:
                return np.sqrt(sse.sum() / n.sum())
            return np.sqrt(sse.sum(0) / n.sum(0)).mean()
        S, N = sse[sel], n[sel]                      # (n_boot, n_county, 4)
        if pooled:
            return np.sqrt(S.sum((1, 2)) / N.sum((1, 2)))
        return np.sqrt(S.sum(1) / N.sum(1)).mean(1)

    label = "pooled" if pooled else "official per-horizon mean"
    print(f"baseline {base}: RMSE {score(sb, nb):.5f}   [{label}]\n")
    print(f"{'candidate':38s} {'RMSE':>8s} {'delta':>9s} {'P(better)':>10s}  95% CI on delta")
    for cand in argv[1:]:
        i2, sc, nc = county_sse_by_horizon(cand, truth)
        order = pd.Index(i2).get_indexer(idx)
        assert (order >= 0).all(), f"{cand} is missing counties present in {base}"
        sc, nc = sc[order], nc[order]
        d_point = score(sc, nc) - score(sb, nb)
        d = score(sc, nc, boot_idx) - score(sb, nb, boot_idx)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{cand:38s} {score(sc, nc):8.5f} {d_point:+9.5f} {(d < 0).mean():10.3f}"
              f"  [{lo:+.5f}, {hi:+.5f}]{'  *' if hi < 0 else ''}")
    print("\ndelta < 0 favours the candidate; * marks a 95% CI excluding zero.")


if __name__ == "__main__":
    main()
