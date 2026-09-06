# Submission requirements

Sourced from `problem_description.pdf` (structure, deliverables, page limit) and
`evaluation_procedure.pdf` (metric and ranking rule). The latter was released
separately on 2026-09-02, after the harness was built; the §Scoring section
below records the reconciliation.

**Deadline: September 25, 2026 (AOE).**

## Structure

Everything goes in a single folder named **`TeamName_Submission/`**, uploaded to
our own Google Drive, **shared with view access**, with the link submitted via
the Google Form given to registered participants.

| component | requirement | our status |
|---|---|---|
| **Prediction file** | Start from `sample_submission.csv`; fill the four horizon columns. **"Do not alter identifier columns or row order."** | `verify_submission.py` asserts byte-identity of all four identifier columns against the template element-wise, which enforces both. 28/28 checks pass. |
| **`code.zip`** | Reproduces predictions end-to-end on the provided data files **with no manual steps**. Must "include library versions, explicit random seeds, and relative file paths". | `requirements.txt` pins every version; seeds fixed throughout; absolute paths removed 2026-08-16 (REPORT §5an) — this is an *explicit* requirement, not cosmetic. Clean-clone drill reproduces a byte-identical SHA-256. |
| **`report.pdf`** | **No more than 6 pages excluding references.** Methodology, key modelling decisions, critical discussion of results. "A central part of the evaluation." | `report/report.pdf`, 6 pages including references (body within 6). |

## Report formatting

**No template, style file, font, margin or column specification is given
anywhere in the data package.** The only formal constraints are the 6-page limit
(excluding references), PDF format, and the filename `report.pdf`. Our
two-column 10pt layout is a free choice.

## Scoring (from `evaluation_procedure.pdf`)

RMSE, **computed separately for each of the four horizons** and pooled across
all 63 test counties and all valid rows — not pooled across horizons. This is
the estimator `src/validate.score_trajectory` has always used, and the
published scoreable-row counts match our NaN pattern exactly:

| target | horizon | their N_k | ours |
|---|---|---|---|
| `osi_target_t01h` | t+1h | 9,009 | 9,009 |
| `osi_target_t06h` | t+6h | 8,694 | 8,694 |
| `osi_target_t24h` | t+24h | 7,560 | 7,560 |
| `osi_target_t48h` | t+48h | 6,048 | 6,048 |

`verify_submission.py` now asserts these counts directly, so the calendar
derivation and the organisers' published numbers cross-check each other.

**Ranking is by average *rank* across the four horizons, ties broken by t+1h
RMSE** — not by average RMSE, which is what our harness reports. The two agree
whenever one configuration beats another on all four horizons, so what matters
is dominance, not the mean. Checked for every decision in the freeze spec:

| decision | horizons improved | safe under average-rank? |
|---|---|---|
| freeze spec vs shipped | 4/4 | yes — strict dominance |
| drop s40 | 4/4 | yes — strict dominance |
| add PatchTST | 3/4 (t01h +1.2e-5) | mean-positive, marginal on the tie-break |
| 50/50 blend with A3x25 | 2/4 (helps t01h/t06h, hurts t24h/t48h) | kept — see below |

`python3 rank_check.py` applies both rules to our nine candidate
configurations. **The freeze spec wins under both**, so the choice does not
depend on the aggregation rule. The rules do disagree once, on the two
configurations separated by 1.4e-7 in mean RMSE — a distance the mean cannot
resolve anyway.

The instructive part is *how* they disagree. Dropping the A3x25 partner gives a
lopsided profile: best of all nine on t24h and t48h, near-worst on t01h and
t06h. At a five-configuration pool that profile ranks 3.00 and beats keeping
s40; at nine it ranks 4.00 and loses. A horizon's upside is capped at rank 1
while its downside grows with the field, so uneven profiles are punished harder
the more competitors there are. The real field is much larger than nine. That
makes the uniformly-good configuration the safer pick under average-rank — the
freeze spec ranks 2/1/2/2, whereas dropping the partner ranks 7/7/1/1. This is
a stronger reason to keep the blend than its mean-RMSE margin, which is small.

Because each horizon column is scored independently over a different range of
target hours, the four columns are not *required* to hold the same number where
they target the same hour — a per-column blend weight is admissible. We tested
it and rejected it: naive 0.008457, **nested 0.008519** against 0.008483 for the
single tied trajectory. The per-fold weights for t01h were 0.36/0.46/0.85/0.39/
0.39. The optimal weight as a function of target hour is non-monotonic
(0.07, 0.59, 1.00, 0.88, 0.71, 0.00), i.e. noise rather than structure.

## Judging

> "Finalists will be selected based on a combination of predictive accuracy and
> the **rigour, novelty, and clarity** of the modelling approach described in the
> written report."

Three of the four criteria are properties of the report, not the score.

## Hard disqualification rule

> "Any submission found to have used future outage values will be
> **automatically rejected**, regardless of predictive accuracy."

Code review is part of judging. See REPRODUCE.md §4 for the corruption-invariance
proof (tampering 17 outage columns × 6,912 post-origin rows changes predictions
by exactly 0.0).

## Pre-submission checklist

- [x] W17 freeze rebuild run (2026-09-03, 11.2 h) -> `submissions/submission_freeze.csv`,
      sha256 `7f2b6f9bfb4cf3d2...`, 28/28 checks; `python3 canonical.py` regenerated.
- [x] **Freeze spec v2 built 2026-09-05 -> `submissions/submission_freeze_v2.csv`, sha256
      `eb57fac66b8fd262...`, 28/28 checks. THIS is the file to submit.** 11 members,
      amplitude x shape rule + the W23 tabular rescale, CV 0.008358 (nested over the search 0.008376). Recombination of
      the cached members reproduces it exactly. `submission_freeze.csv` (v1, 0.008468) and
      `submission_ensemble_final.csv` (0.008543) remain reproducible via `--spec freeze|shipped`.
- [x] `report.pdf` reconciled to freeze spec v2 (2026-09-05): quotes 0.00838 (nested over the
      search), 11 members, amplitude x shape; 6 pages, 0 overfull, 0 warnings
- [x] `python3 verify_submission.py submissions/submission_freeze_v2.csv` — 28/28
- [x] `code.zip` built 2026-09-05 with `git archive HEAD` (tracked files only, so the git-ignored
      NDA data, oof caches and venvs are excluded by construction; audited: no DM_*.csv, no
      sample_submission*.csv, no competition PDFs). Assembled with `report.pdf` and the v2
      prediction file under `TeamName_Submission/` (git-ignored) — rename the folder and upload.
- [ ] Folder named `TeamName_Submission/` with the real team name
- [ ] Google Drive link shared with **view access**
