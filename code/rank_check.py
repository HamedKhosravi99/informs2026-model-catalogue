"""Rank our candidate configurations the way the organisers will.

`evaluation_procedure.pdf` ranks teams WITHIN each of the four horizons and
averages the four ranks, ties broken on t+1h RMSE. Our harness ranks by the
mean of the four RMSEs. The two rules can disagree, so this script applies both
to our own candidate set and reports where they part company.

Scored on the 239 training counties under 5-fold county-holdout CV -- each
county is scored only by the fold that held it out. The 63 test counties are
different counties, so the ranks here are ours-against-ourselves, not a
leaderboard prediction. What transfers is the *shape* of the disagreement.

Usage: python3 rank_check.py
"""
import sys

import numpy as np
import pandas as pd

from canonical import (MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP,
                       TAIL_SPLIT_HOUR, ensemble, load, truth)
from src.validate import score_trajectory

HZ = ["t01h", "t06h", "t24h", "t48h"]


def build_pool(t):
    fz = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
    fz_s40 = [FREEZE_SUB.get(m, m) for m in MEMBERS] + FREEZE_ADD
    fz_nots = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP]
    c7 = ensemble(["s62", "s51", "s39", "s40", "i11", "s45", "s35"], t, tail=False)
    c7.loc[c7["hour"] >= TAIL_SPLIT_HOUR, "osi_pred"] *= 0.647
    return {
        "FREEZE (9 memb -s40, +A3x25)": ensemble(fz, t, blend_with="a3x25"),
        "freeze, no PatchTST":          ensemble(fz_nots, t, blend_with="a3x25"),
        "freeze, keep s40":             ensemble(fz_s40, t, blend_with="a3x25"),
        "freeze, no A3x25 partner":     ensemble(fz, t),
        "SHIPPED (8 memb +A3x25)":      ensemble(MEMBERS, t, blend_with="a3x25"),
        "compliance fallback (C1)":     ensemble([x if x != "f2" else "f2_provided_only"
                                                  for x in MEMBERS], t, blend_with="a3x25"),
        "8 members alone":              ensemble(MEMBERS, t),
        "A3x25 alone":                  load("a3x25"),
        "7-member contingency":         c7,
    }


def main():
    t = truth()
    pool = build_pool(t)
    S = pd.DataFrame({k: [score_trajectory(v, t).loc[h, "rmse"] for h in HZ]
                      for k, v in pool.items()}, index=HZ).T
    S["mean"] = S[HZ].mean(axis=1)
    S = S.sort_values("mean")

    print("Per-horizon RMSE (official formula), x1e3, 239 held-out counties\n")
    rk = S[HZ].rank(axis=0, method="min")
    show = (S * 1e3).round(4)
    show["avg_rank"] = rk.mean(axis=1)
    print(show.to_string())

    print("\nRanking rule agreement, as the candidate pool grows:")
    for n in (5, 7, len(S)):
        T = S.head(n)
        ar = T[HZ].rank(axis=0, method="min").mean(axis=1)
        official = list(ar.sort_values(kind="stable").index)
        ours = list(T.sort_values("mean").index)
        swaps = [(a, b) for i, a in enumerate(ours) for b in ours[i + 1:]
                 if official.index(a) > official.index(b)]
        print(f"  pool={n:2d}  same winner={official[0] == ours[0]}  "
              f"identical order={official == ours}  swaps={swaps if swaps else 'none'}")

    print("\nWhy the disagreement shrinks as the pool grows: a horizon's upside is")
    print("capped at rank 1, but its downside grows with the size of the field. An")
    print("uneven profile (best on two horizons, worst on two) is therefore punished")
    print("harder the more competitors there are, while a uniformly-good profile is")
    print("not. The real field is far larger than this pool, so the uniformly-good")
    print("configuration is the safer pick under average-rank -- which is what the")
    print("freeze spec is (ranks 2/1/2/2 here), and what dropping the A3x25 partner")
    print("is not (ranks 5/5/1/1 at pool=5, 7/7/1/1 at pool=9).")


if __name__ == "__main__":
    main()
