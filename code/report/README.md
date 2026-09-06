# Competition report

`report.tex` → `report.pdf` (5 pages + references; limit is 6 excluding references).

**No template is imposed by the competition** — no style file, font, margin or
column spec exists in the data package (see `SUBMISSION.md`). Two-column 10pt is
our own choice; it buys roughly a page of content over single-column.

## Structure

The report is written as a **derivation**, not a search log. §4 walks the chain
M1→M8: each measurement, what it forces, and the experiment that later tested
it. The architecture appears at the *end* of §4 as the consequence.

1. Problem and approach (+ contributions)
2. Validation protocol — the nesting rule and four reversals
3. Diagnosis: the error budget — what is recoverable
4. **Deriving the forecaster** — M1–M8 chain → Table 1 (measurement / forces /
   confirmed by) → the resulting model → D1–D4, the four non-obvious
   consequences → coverage of the search
5. Predicted versus measured — the ledger
6. Results and critical discussion
7. What we got wrong — four self-audit defects
8. Relation to prior work / 9. Conclusion / 10. Reproducibility

Body ends on page 6; references follow (the limit is 6 *excluding* references).

Figures: `fig1` error concentration + nesting staircase, `fig2` out-of-fold
trajectories, `fig3` the quality-versus-diversity scatter that carries the
abstract's headline result.

```bash
python3 report/make_figs.py          # regenerate figures from oof/cfg_final.csv
cd report && pdflatex report.tex     # twice, for cross-references
```

## Before submitting — must re-verify

The report quotes the **freeze-spec** model (CV 0.00838 (nested over its selection; configuration score 0.008360) / MAE 0.00255), which is
measured from cached out-of-fold predictions and **built as `submissions/submission_freeze_v2.csv`** (2026-09-05; freeze spec v2). After the W17 freeze rebuild:

1. `python3 canonical.py` and reconcile every number in §6 and the abstract.
2. `python3 report/make_figs.py` — figures are drawn from `oof/cfg_final.csv`.
3. Confirm the member table in §4 matches what `make_submission.py` actually
   trains.

Claims verified as of 2026-08-16:
- corruption invariance (§2): tampering 17 outage columns × 6912 post-origin
  rows changes predictions by exactly 0.0.
- clean-clone reproduction (§8): byte-identical SHA-256.
- coverage table (§4.3): derived from `results/RESULTS.md`, sums to 129.
