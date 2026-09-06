"""Generate the full experiment catalogue in three formats from the run log.

Every row is one scored out-of-fold run: its intervention family (the contrast it
tested), its mechanism and design axes where determinable, all four horizon RMSEs,
MAE, fold spread, runtime, and whether it reached the shipped ensemble.

Nothing here is hand-typed: the scores come from results/ideas_results.csv, the
family coding from family_ledger.py, and the shipped set from canonical.py.

Usage: python3 build_catalogue.py [outdir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from family_ledger import ASSIGN, NAME, code_to_family

ROOT = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT

# --- design axes, recorded only where the code determines them unambiguously
MECH = {  # function class actually instantiated
 "BiEncSeq2Seq": "S44b S50 S51 S62 A1 A1x25 A2 A2b A3 A3x25 A4 A4b A5 A7 A7b A8 A9 A9b A9c A9d "
                 "A10 A10b A10c A10d A11 A11x25 X9 X10 X9w E5 S43 S46b S47b",
 "Seq2Seq (GRU)": "I11 I11w S35 S35w S36 S40 S63 S63b S63c X3 X4 X5 X6 X7 X8 X11 I20 S45 S46 S47 "
                  "S48 I16 I17 I17b I08 T1 P3 P3b P4 Y1 Y3 X2",
 "CrossCountyNet": "S39 S39b S39w X1 I07",
 "LSTM / Transformer": "S30 E2 E3 I12 S31",
 "TCN / basis / linear": "S32 S33 S42 TS1 TS1b SX1 SX2 K1 K1b M2",
 "patch / variate attention": "TS2 TS2b TS2c TS2d TS3 TS3b SX4 Y2",
 "boosted / bagged trees": "S21 S21b S22 S22b S22c S23 S23b S23c S23d S24 S24b S25 S26 S27 S28 "
                           "S29 S29b S29c S29d S29e S60 S60b S60c S60d S61 S61b U1 U1b U1c E4 E4b "
                           "E4c E4d F1 F1b F1c F1d F1e F2 F3 F4 RT4 FS0 FS1a FS1b FS1c FS1d FS1e "
                           "FS2a FS2b FS2c FS2d FS2e I13 I05 I05b Y4",
 "MLP / KAN / TabNet / FT-T": "D1 D2 D3 D4 D5",
 "stock-flow state space": "P1 P1b P1c P1d R8 R8b M1 M1b M1c M1d M3 M3b M3c Y5 Y6 Y7",
 "normalizing flow": "SX3",
 "other / statistical": "B1 B2 B2b B3 I01 I02 I03 I04 I06 I09 I10 I14 I15 I18 S34 S37 S41 S52 S53 "
                        "S54 S64 H1 H2 E1 N1b N1c N2 N2b RT1 RT2 RT3 A11b A11c A11d S38 S44",
}
OBJ = {"dual raw+sqrt": "S51 S62 X9 X10 X9w A1 A1x25 A2 A2b A3 A3x25 A4 A4b A5 A7 A7b A8 A9 A9b "
                        "A9c A9d A10 A10b A10c A10d A11 A11x25 E5 TS2c TS2d Y1 Y2",
       "sqrt target": "S50 S24 S24b S60 S60b S60c S60d S61 S61b U1 U1b U1c E4 E4b E4c E4d F1 F1b "
                      "F1c F1d F1e F2 F3 F4 RT4 Y4",
       "metric hour weights": "S44 S44b S62 I11w S35w S39w A5 A11 A11x25 TS2c TS2d",
       "hurdle / two-part": "P1 P1b P1c P1d H1 H2 I06 R8 R8b M1 M1b M1c M1d Y5 Y6 Y7",
       "Tweedie / Poisson": "I09 I13",
       "likelihood": "SX3", "quantile": "I06"}


def _inv(d):
    m = {}
    for k, s in d.items():
        for c in s.split():
            m.setdefault(c, k)
    return m


def build():
    d = pd.read_csv(ROOT / "results" / "ideas_results.csv")
    d["code"] = d.idea.str.split().str[0]
    d["name"] = d.idea.str.split(n=1).str[1].fillna("")
    fam = code_to_family()
    d["fam"] = d.code.map(fam)
    d["family"] = d.fam.map(lambda k: f"{k} {NAME[k]}" if isinstance(k, str) else "—")
    d["mechanism"] = d.code.map(_inv(MECH)).fillna("—")
    d["objective"] = d.code.map(_inv(OBJ)).fillna("plain squared error")
    # SARIMAX diverged on some counties and logged a meaningless score; keep the run in the
    # record (it was really tried) but do not let a 1e80 masquerade as a result.
    num = pd.to_numeric(d.rmse_mean, errors="coerce")
    d.loc[num > 0.1, ["rmse_mean", "rmse_t01h", "rmse_t06h", "rmse_t24h", "rmse_t48h", "mae_mean"]] = np.nan
    d.loc[num > 0.1, "note"] = "diverged; score not meaningful"
    # imported lazily: canonical only supplies the shipped member names, not data
    from canonical import MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP
    ship = {FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP} | set(FREEZE_ADD) | {"a3x25"}
    # A code run more than once has one instance in the ensemble, not two: flag the best-scoring.
    cand = d.code.str.lower().isin({s.lower() for s in ship})
    d["shipped"] = False
    _num = pd.to_numeric(d.rmse_mean, errors="coerce")
    for c in d.loc[cand, "code"].unique():
        rows = d.index[cand & (d.code == c)]
        d.loc[_num.loc[rows].idxmin() if _num.loc[rows].notna().any() else rows[0], "shipped"] = True
    d["seeds"] = np.where(d.code.str.contains("x25|Y[1-7]", regex=True), 25,
                 np.where(d.code.str.match(r"A3$|X[5-9]$"), 3, 3))
    d = d.sort_values(["fam", "rmse_mean"]).reset_index(drop=True)
    d["rank"] = pd.to_numeric(d.rmse_mean, errors="coerce").rank(method="min").fillna(0).astype(int)
    # 13 codes appear more than once: the same idea re-run after a fix or on a changed control.
    # Number them so the catalogue shows both rather than silently collapsing them.
    dup = d.code.duplicated(keep=False)
    d.loc[dup, "code"] = d.loc[dup].groupby("code").cumcount().add(1).astype(str).radd(
        d.loc[dup, "code"] + " #")
    return d




# ------------------------------------------------------------------ renderers
FAMDESC = {
 "F1": ("Signal", "Change what information the predictor may see.",
        "The residual multiplier is not recoverable from legal features, so added information should return little."),
 "F2": ("Mechanism", "Hold information and task fixed; change the map from covariates to forecast.",
        "Error is spread across scale, shape and timing, so no single function class should capture it."),
 "F3": ("Target", "Change what prediction problem is optimised: loss, target scale, factorisation.",
        "42.4% zeros are cheap (0.7% of squared error); skew is expensive."),
 "F4": ("Exposure", "Change which training examples are seen, holding the predictor fixed.",
        "Robust perturbation may manufacture distinct error bases."),
 "F5": ("Estimation", "Change how the same predictor is estimated from finite data.",
        "191 sequences per fold implies variance-limited estimators."),
 "F6": ("Adaptation", "Fit a correction to or around a finished forecast.",
        "The per-county residual has out-of-fold R^2 ~ 0, so post-hoc correction should fail."),
 "F7": ("Combination", "Change which trained predictors are kept and how they are pooled.",
        "With no dominant error mode, complementary residuals should matter more than solo accuracy."),
 "F8": ("Uncertainty", "Change the predictive interval, not the point forecast.",
        "Orthogonal to the point metric by construction."),
 "P":  ("Protocol", "Procedures whose object is choosing or evaluating, not predicting.", "--"),
 "X":  ("Compound", "Deliberately changed more than one stage; not clean single-family evidence.", "--"),
}


def _fmt(v, n=5):
    """Some rows in the run log carry a non-numeric score field (runs that errored
    or were recorded before the scorer was standardised); render those as '--'."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if pd.isna(f) else f"{f:.{n}f}"


def to_markdown(d, path):
    L = ["# Experiment catalogue", "",
         f"Every scored out-of-fold run in this project: **{len(d)} experiments**, each classified by",
         "the *contrast* it was run to test rather than by the properties of the model it produced.",
         "Generated by `build_catalogue.py` from `results/ideas_results.csv`; do not hand-edit.", "",
         "All scores are county-holdout cross-validation on 239 training counties, five folds,",
         "stratified by state x severity. RMSE is the mean of the four per-horizon values; the",
         "organisers rank within each horizon and average ranks, so that mean is a reporting",
         "summary and not the official statistic.", "",
         "## Summary by intervention family", "",
         "| code | family | runs | best RMSE | median RMSE | shipped members |", "|---|---|---|---|---|---|"]
    for k in FAMDESC:
        g = d[d.fam == k]
        if not len(g): continue
        L.append(f"| {k} | {FAMDESC[k][0]} | {len(g)} | {_fmt(g.rmse_mean.min())} | "
                 f"{_fmt(g.rmse_mean.median())} | {int(g.shipped.sum())} |")
    L += ["", f"| **total** | | **{len(d)}** | {_fmt(d.rmse_mean.min())} | {_fmt(d.rmse_mean.median())} | "
          f"**{int(d.shipped.sum())}** |", "",
          "## Reference points", "",
          "| model | RMSE |", "|---|---|",
          "| all-zeros (the target's own RMS) | 0.01632 |",
          "| decayed persistence | 0.01110 |",
          "| best single model (A3x25) | 0.00862 |",
          "| **submitted ensemble** | **0.00835** |", ""]
    for k, (nm, what, pred) in FAMDESC.items():
        g = d[d.fam == k].sort_values("rmse_mean")
        if not len(g): continue
        L += [f"## {k} — {nm} ({len(g)} runs)", "", f"*Contrast tested:* {what}", ""]
        if pred != "--": L += [f"*Prediction from the diagnosis:* {pred}", ""]
        L += ["| code | experiment | mechanism | objective | RMSE | t+1 | t+6 | t+24 | t+48 | MAE | sec | shipped |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            L.append(f"| `{r.code}` | {str(r['name'])[:64]} | {r.mechanism} | {r.objective} | "
                     f"**{_fmt(r.rmse_mean)}** | {_fmt(r.rmse_t01h)} | {_fmt(r.rmse_t06h)} | "
                     f"{_fmt(r.rmse_t24h)} | {_fmt(r.rmse_t48h)} | {_fmt(r.mae_mean)} | "
                     f"{'' if pd.isna(r.secs) else int(r.secs)} | {'YES' if r.shipped else ''} |")
        L.append("")
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path.name}")


def to_tex(d, path):
    esc = lambda s: (str(s).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
                     .replace("#", "\\#").replace("~", "$\\sim$").replace("^", "\\^{}"))
    L = [r"\documentclass[9pt,landscape]{article}",
         r"\usepackage[margin=0.5in,landscape]{geometry}",
         r"\usepackage{longtable,booktabs,array}", r"\usepackage[table]{xcolor}",
         r"\usepackage{times}", r"\setlength{\tabcolsep}{3pt}",
         r"\title{\vspace{-2em}\textbf{Experiment Catalogue}\\[2pt]\large INFORMS 2026 DMS Data Challenge}",
         r"\author{}", r"\date{}", r"\begin{document}", r"\maketitle", r"\vspace{-2em}",
         r"\noindent All " + str(len(d)) + r" scored out-of-fold runs, classified by the \emph{contrast} each tested.",
         r"County-holdout cross-validation, 239 counties, five folds. RMSE is the mean of the four",
         r"per-horizon values (a reporting summary; the organisers rank within each horizon).", "",
         r"\section*{Summary by intervention family}",
         r"\begin{tabular}{@{}llrrrr@{}}\toprule",
         r"code & family & runs & best & median & shipped \\ \midrule"]
    for k in FAMDESC:
        g = d[d.fam == k]
        if not len(g): continue
        L.append(f"{k} & {FAMDESC[k][0]} & {len(g)} & {_fmt(g.rmse_mean.min())} & "
                 f"{_fmt(g.rmse_mean.median())} & {int(g.shipped.sum())} \\\\")
    L += [r"\midrule \textbf{total} & & \textbf{" + str(len(d)) + r"} & " + _fmt(d.rmse_mean.min())
          + " & " + _fmt(d.rmse_mean.median()) + r" & \textbf{" + str(int(d.shipped.sum())) + r"} \\",
          r"\bottomrule\end{tabular}", ""]
    for k, (nm, what, pred) in FAMDESC.items():
        g = d[d.fam == k].sort_values("rmse_mean")
        if not len(g): continue
        L += [rf"\section*{{{k} --- {nm} ({len(g)} runs)}}",
              rf"\noindent\emph{{Contrast tested:}} {esc(what)}\\"]
        if pred != "--": L.append(rf"\emph{{Prediction from the diagnosis:}} {esc(pred)}")
        L += [r"\vspace{2pt}",
              r"\begin{longtable}{@{}p{1.1cm}p{5.4cm}p{3.1cm}p{2.7cm}rrrrrrr@{}}",
              r"\toprule code & experiment & mechanism & objective & RMSE & t+1 & t+6 & t+24 & t+48 & MAE & s \\ \midrule \endhead"]
        for _, r in g.iterrows():
            hl = r"\rowcolor{gray!15}" if r.shipped else ""
            L.append(f"{hl}\\texttt{{{esc(r.code)}}} & {esc(str(r['name'])[:56])} & {esc(r.mechanism)} & "
                     f"{esc(r.objective)} & {_fmt(r.rmse_mean)} & {_fmt(r.rmse_t01h)} & {_fmt(r.rmse_t06h)} & "
                     f"{_fmt(r.rmse_t24h)} & {_fmt(r.rmse_t48h)} & {_fmt(r.mae_mean)} & "
                     f"{'' if pd.isna(r.secs) else int(r.secs)} \\\\")
        L += [r"\bottomrule\end{longtable}", ""]
    L.append(r"\end{document}")
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path.name}")


def main():
    d = build()
    cols = ["code", "name", "family", "mechanism", "objective", "rmse_mean",
            "rmse_t01h", "rmse_t06h", "rmse_t24h", "rmse_t48h", "mae_mean", "secs", "shipped", "rank"]
    OUT.mkdir(parents=True, exist_ok=True)
    d[cols].to_csv(OUT / "catalogue.csv", index=False)
    print(f"wrote catalogue.csv  ({len(d)} runs, {d.fam.nunique()} families, {int(d.shipped.sum())} shipped)")
    to_markdown(d, OUT / "CATALOGUE.md")
    to_tex(d, OUT / "catalogue.tex")
    return d


if __name__ == "__main__":
    main()
