"""Regenerate CANONICAL.md: every headline number, recomputed from the harness.

Three separate headline figures have drifted from what the code actually
computes (REPORT §5ak stale oracle figures, §5an the pooled-vs-per-horizon
bootstrap, §5ap the production MAE that was really the t01h column). Each was a
hand-carried number. The durable fix is to stop carrying them: this script
recomputes each one from cached out-of-fold predictions with the official
scorer, so a citation can be checked by re-running rather than by trusting.

Usage: python3 canonical.py            # writes CANONICAL.md
       python3 canonical.py --check    # exit 1 if CANONICAL.md is out of date
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.data import load_train, osi_trajectory
from src.validate import score_trajectory

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CANONICAL.md"
TAIL_SPLIT_HOUR, TAIL_SCALE = 168, 0.667
# W23 (2026-09-06): f2 fits squared error on sqrt(OSI) and squares back, so by Jensen it
# estimates below the conditional mean -- the functional RMSE rewards. Measured bias
# E[pred]/E[truth] = 0.841. A least-squares rescale, fitted on four folds and applied to the
# fifth, is adopted on ALL FIVE folds with zero optimism gap and improves the blend on all
# four horizons (0.008358 vs 0.008360). Duan smearing, which removes the bias entirely
# (ratio -> 1.01), makes the blend WORSE: it wants f2 shrunk, not unbiased. Production value
# fitted on all 239 counties from f2's own out-of-fold predictions, as the tail scalar is.
F2_RESCALE = 1.090366
MEMBERS = ["s62", "s51", "s39", "s40", "i11", "f2", "s35", "p1b"]
# Freeze-spec substitutions (REPORT §5am/§5at/§5au/§5av): variance-reduced
# variants of three members, plus PatchTST-KF as a 9th. Not yet built into the
# submission -- scheduled for the W17 rebuild.
# i11 -> i11w (metric hour weights, 3 seeds) rather than x3 (flat, 25 seeds):
# all five folds pick it, so nested == naive == 0.008468 with zero optimism gap,
# and the county bootstrap has it ahead on all four horizons (0.948/0.980/
# 0.982/0.855). s35w was tested the same way and did NOT help, so s35 -> x4.
FREEZE_SUB = {"i11": "i11w", "s35": "x4", "s51": "x9"}
# ts2c, not ts2b: PatchTST retrained against the metric's per-hour weights. It
# was the one member training on a flat MSE while every ideas4 member already
# used scoring_weights(). Nested variant selection does not reverse here
# (nested 0.008482 vs 0.008483 for keeping ts2b, optimism gap +1.4e-6), and it
# recovers most of the t01h cost ts2b was carrying -- t01h is the tie-break
# horizon. The margin is ~1e-6 and the two are rank-tied; see rank_check.py.
FREEZE_ADD = ["ts2c", "p1c", "r6corr", "s35w"]
# W21 (2026-09-05): a nested add-one sweep over all 296 cached objects under the current spec found
# three legal additions that every fold adopts -- p1c (stock-flow, no DAgger; a second physics
# basis, corr 0.85 with p1b), r6corr (p1b with two nested flow-bias scalars) and s35w (as an ADDED
# member; it lost as a swap) -- and a parameter-free combination rule, amplitude x shape (average
# each member's county total and its normalised shape separately, so timing disagreement does not
# flatten peaks). Nested SELECTION over all ten searched configurations: 0.008378 vs 0.008468.
# Bootstrap P(better) per horizon 0.77/0.77/0.91/0.90 (11 members) -- see HANDOFF W21.
FREEZE_RULE = "ampshape"
# s40 DROPPED (REPORT §5ay): nested drop-one selection picks it on all five
# folds, so naive and nested estimates are identical (0.008483) -- no curse.
FREEZE_DROP = ["s40"]

# (label, oof slug or member list, note)
SINGLE = [
    ("all-zeros", "zero", "trivial baseline"),
    ("decayed persistence", "decay", "strongest naive baseline"),
    ("A3x25 (best single model)", "a3x25", "snapshot ensembling, 25 seeds x 4"),
    ("s62", "s62", "member"), ("s51", "s51", "member"),
    ("s39", "s39", "member"), ("s40", "s40", "member"),
    ("i11", "i11", "member"), ("f2", "f2", "member"),
    ("s35", "s35", "member"), ("p1b", "p1b", "member"),
    ("C1 (f2, provided data only)", "f2_provided_only", "compliance contingency"),
    ("x9  = s51 + snapshots", "x9", "freeze-spec member (best individual)"),
    ("x3  = i11 @ 25 seeds, flat loss", "x3", "superseded by i11w"),
    ("i11w = i11 + metric hour weights", "i11w", "freeze-spec member"),
    ("x11  = i11w @ 25 seeds", "x11", "REJECTED as swap: worse 4/4 horizons (law 13c)"),
    ("s35w = s35 + metric hour weights", "s35w", "tested, did not help"),
    ("s39w = s39 + metric hour weights", "s39w", "swap test: nested 0.008472 vs 0.008468, rejected"),
    ("x4  = s35 @ 25 seeds", "x4", "freeze-spec member"),
    ("ts2b = PatchTST-KF, flat loss", "ts2b", "superseded by ts2c"),
    ("ts2c = PatchTST-KF + metric hour weights", "ts2c", "freeze-spec 9th member"),
    ("ts2d = PatchTST-KF + exact metric Jacobian", "ts2d", "not better than ts2c"),
    # SOTA sweep (SX-series) and foundation-model measurements. None joins the
    # freeze spec; each is a measured negative that the report cites.
    ("SX1 XLinear-KF (Ma 2026)", "sx1", "added-member test: worse, nested keep"),
    ("SX2 TSMixer-Ext-KF (Chen 2023)", "sx2", "added-member test: worse, nested keep"),
    ("SX3 CondFlow-KF (conditional normalizing flow)", "sx3", "sample-mean point forecast"),
    ("SX4 TimeXer-KF (Wang NeurIPS 2024)", "sx4", "global-token cross-attention"),
    ("Chronos-2 zero-shot, median, WITH covariates", "chronos2_med", "not shippable; tools/"),
    ("Chronos-2 zero-shot, median, no covariates", "chronos2_med_uni", "univariate ablation"),
    ("Chronos-2 zero-shot, quantile-mean, with covariates", "chronos2", "E[y] estimate"),
    # Tabular foundation models as drop-in learners on f2's exact rows (tools/tabfm_f2.py).
    ("TabDPT on f2 rows (Layer 6, Apache-2.0)", "tabdpt_f2", "retrieval ICL, full 27k context"),
    ("TabICLv2 on f2 rows (Inria, BSD-3)", "tabicl_f2", "in-context regression"),
    ("ConTextTab / SAP RPT-1-OSS on f2 rows", "contexttab_f2", "NOT RUN: HF weights gated (login), GPU-only (80 GB rec.), py3.11"),
    ("Mitra on f2 rows (Amazon, Apache-2.0, via AutoGluon)", "mitra_f2", "separate venv; see tools/tabfm_f2.py"),
    ("TabPFN-v2 on f2 rows (Prior Labs)", "tabpfn_f2", "BLOCKED: license token needed"),
    ("f2 itself (XGBoost) -- the incumbent learner", "f2", "0.00930"),
    ("p1c = stock-flow, no DAgger", "p1c", "freeze v2 member (2nd physics basis)"),
    ("r6corr = p1b + two nested flow-bias scalars", "r6corr", "freeze v2 member"),
    # Decoupled two-stage on f2's rows (f2_twostage.py): P(y>0) x E[y|y>0]. Rejected.
    ("f2_ctrl: plain regressor on exported rows", "f2_ctrl", "reproduces f2 exactly -- export validated"),
    ("f2_pos: regressor on positive cells only", "f2_pos", "E[y|y>0]"),
    ("f2_hurdle = P(y>0) x f2_pos", "f2_hurdle", "swap: worse 4/4 horizons, nested keep; added: worse"),
]


def truth():
    return osi_trajectory(load_train())


HZ = ["t01h", "t06h", "t24h", "t48h"]


def sc(df, t):
    tab = score_trajectory(df, t)
    return tab.loc["mean", "rmse"], tab.loc["mean", "mae"]


def per_h(df, t):
    """The four numbers the organisers actually compute (evaluation_procedure.pdf)."""
    tab = score_trajectory(df, t)
    return [tab.loc[h, "rmse"] for h in HZ]


def load(slug):
    return pd.read_csv(ROOT / "oof" / f"{slug}.csv")


from src.combine import ampshape


def ensemble(members, t, blend_with=None, tail=True, rule=None, f2_scale=None):
    """Combine members by `rule` ('mean' or 'ampshape'; default = FREEZE_RULE), optionally
    50/50 with another object, then tail.

    `f2_scale` is the W23 back-transform rescale on the tabular member; it defaults to
    F2_RESCALE. Configurations BUILT before W23 must pass 1.0, or this function would
    report a number that does not describe the file it names."""
    rule = FREEZE_RULE if rule is None else rule
    f2_scale = F2_RESCALE if f2_scale is None else f2_scale
    parts, idx = {}, None
    for m in members:
        v = load(m).set_index(["fipsCode", "hour"])["osi_pred"]
        if m == "f2":
            v = v * f2_scale
        parts[m] = v
        idx = v.index if idx is None else idx.intersection(v.index)
    if rule == "ampshape":
        ens = ampshape([parts[m].reindex(idx) for m in members], idx)
    else:
        ens = sum(parts[m].reindex(idx) for m in members) / len(members)
    if blend_with is not None:
        other = load(blend_with).set_index(["fipsCode", "hour"])["osi_pred"].reindex(idx)
        ens = 0.5 * ens + 0.5 * other
    df = ens.reset_index()
    df.columns = ["fipsCode", "hour", "osi_pred"]
    if tail:
        df.loc[df["hour"] >= TAIL_SPLIT_HOUR, "osi_pred"] *= TAIL_SCALE
    return df


def main():
    t = truth()
    rows_cfg, rows_single = [], []

    def add(label, df):
        r, m = sc(df, t)
        rows_cfg.append((label, r, m, per_h(df, t)))

    shipped = ensemble(MEMBERS, t, blend_with="a3x25", rule="mean", f2_scale=1.0)
    add("**SHIPPED** 0.5*[8 members] + 0.5*A3x25, tail 0.667", shipped)
    add("8-member ensemble alone, tail 0.667", ensemble(MEMBERS, t, rule="mean", f2_scale=1.0))

    comp = [x if x != "f2" else "f2_provided_only" for x in MEMBERS]
    add("compliance fallback (C1 for f2)", ensemble(comp, t, blend_with="a3x25", rule="mean"))

    fz = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
    add("**FREEZE SPEC v2** (11 members: +p1c +r6corr +s35w; amplitude x shape rule; 50/50 A3x25) "
        "-- BUILT 2026-09-05: submissions/submission_freeze_v2.csv", ensemble(fz, t, blend_with="a3x25"))
    fz1 = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + ["ts2c"]
    add("freeze spec v1 (8 members, mean rule) -- BUILT: submissions/submission_freeze.csv",
        ensemble(fz1, t, blend_with="a3x25", rule="mean", f2_scale=1.0))

    c7 = ["s62", "s51", "s39", "s40", "i11", "s45", "s35"]
    df7 = ensemble(c7, t, tail=False, rule="mean", f2_scale=1.0)
    df7.loc[df7["hour"] >= TAIL_SPLIT_HOUR, "osi_pred"] *= 0.647
    add("previous 7-member set (contingency_7member.csv)", df7)

    for label, slug, note in SINGLE:
        try:
            rr, mm = sc(load(slug), t)
            rows_single.append((label, rr, mm, note))
        except FileNotFoundError:
            rows_single.append((label, float("nan"), float("nan"), note + " [oof missing]"))

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()

    L = []
    L.append("# Canonical numbers\n")
    L.append("**Generated by `python3 canonical.py` — do not hand-edit.** Every figure\n"
             "below is recomputed from cached out-of-fold predictions with the official\n"
             "scorer (`src/validate.score_trajectory`), so any citation elsewhere can be\n"
             "checked by re-running this script.\n")
    L.append(f"Commit: `{sha}`\n")
    L.append("**Metric convention.** Each per-horizon RMSE is computed exactly as\n"
             "`evaluation_procedure.pdf` defines it — pooled over every valid\n"
             "county-hour cell of that horizon — and a verbatim re-implementation in\n"
             "submission-file space reproduces all four to machine precision. The\n"
             "`mean` column is *ours*, not theirs: the organisers rank within each\n"
             "horizon and average the four ranks, ties broken on t01h, so they never\n"
             "form a single number. We report the mean because a scalar is needed to\n"
             "compare configurations, and it is safe wherever one configuration wins\n"
             "on all four horizons — which is checked in SUBMISSION.md. MAE is not\n"
             "scored by the organisers at all; it is a diagnostic. Production\n"
             "headlines were briefly quoted from the `t01h` row instead — that is the\n"
             "REPORT §5ap correction, and it is why this file exists.\n")

    L.append("## Shipped configuration and alternatives\n")
    L.append("The four RMSE_k columns are what the organisers actually score. `mean`\n"
             "is our own aggregation and has no official standing -- ranking is by\n"
             "average *rank* across the four, ties broken on t01h.\n")
    L.append("| configuration | t01h | t06h | t24h | t48h | mean | MAE |")
    L.append("|---|---|---|---|---|---|---|")
    for lbl, rr, mm, ph in rows_cfg:
        cells = " | ".join(f"{v:.5f}" for v in ph)
        L.append(f"| {lbl} | {cells} | **{rr:.5f}** | {mm:.5f} |")

    L.append("\n## Individual models and baselines\n")
    L.append("| model | RMSE | MAE | note |")
    L.append("|---|---|---|---|")
    for lbl, rr, mm, note in rows_single:
        L.append(f"| {lbl} | {rr:.5f} | {mm:.5f} | {note} |")

    L.append("\n## Submission files\n")
    L.append("| file | sha256 |")
    L.append("|---|---|")
    import hashlib
    for f in sorted((ROOT / "submissions").glob("*.csv")):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        L.append(f"| `submissions/{f.name}` | `{h[:16]}...` |")

    text = "\n".join(L) + "\n"
    if "--check" in sys.argv:
        cur = OUT.read_text() if OUT.exists() else ""
        drift = [a for a, b in zip(cur.split("\n"), text.split("\n"))
                 if a != b and not a.startswith("Commit:")]
        if drift:
            print("CANONICAL.md is OUT OF DATE; re-run `python3 canonical.py`")
            for d in drift[:10]:
                print("  ", d)
            sys.exit(1)
        print("CANONICAL.md is current")
        return
    OUT.write_text(text)
    print(f"wrote {OUT}")
    print("\n".join(L[L.index("## Shipped configuration and alternatives\n"):][:8]))


if __name__ == "__main__":
    main()
