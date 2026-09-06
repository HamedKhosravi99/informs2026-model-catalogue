**W1 SUPERSEDED (2026-08-13):** the metric question was resolved by committed
interpretation instead of an email — see HANDOFF.md §2. The 8-member RMSE-oriented
ensemble shipped (0.00864/0.00275); `contingency_7member.csv` is the MAE fallback.
The chairs email remains drafted but optional. Everything below referring to W1 or
the '3 pre-built submissions' contingency should be read against that decision.

PLAN — INFORMS 2026 DM Challenge: report/presentation endgame (grounded in HANDOFF.md, REPORT.md, results/RESULTS.md, NOVELTY.md; load-bearing claims re-verified in-repo today)

Verification notes that shape the ranking (checked this session):
- requirements.txt pins only pandas/numpy/sklearn/matplotlib/torch — **xgboost and lightgbm are missing**, so a judge's `make_submission.py` crashes on import of the shipped s45 member. Confirmed real.
- Doc drift **RESOLVED 2026-08-15** (was: README "11-member ensemble"; RESULTS.md header 7 members / 0.00870; REPORT §8 naming a _v4.csv output; idea counts drifting 56/67/~100). README now states the shipped blend at 0.00854/0.00296 with tail 0.667; RESULTS.md header is current. Re-run this sweep at freeze — the numbers move again with W17.
- src/subcounty.py line 9: "Falls back silently to a no-op if the file has not been fetched" — confirmed; a mislaid CSV silently produces a different submission.
- results/weight_results.csv is the **0.00888/11-member era** (production row = 0.00888095). The overfitting-staircase figure must be regenerated on the shipped 7-member pool or explicitly captioned with its era, or it hands a judge an internal inconsistency.
- EMAIL_TO_CHAIRS.md is drafted, unsent. Strict 2025-form hurdle 0.01087 confirmed (REPORT §4 row 11). customersTracked confirmed in DM_Test.csv header. oof/ caches for all 7 members + p1b/e2/zero/decay/ens_prod confirmed on disk.

---

## 1. DEDUPLICATION: 59 ideas → 17 workstreams

The six lenses converged heavily. Merges (kept name = the strongest variant):

| Workstream | Merges |
|---|---|
| W1 Chairs email (metric + horizon aggregation + page/appendix rule + permitted-data-classes clarification) | "Send the metric-question email", email parts of "Deliverable parity", email parts of s45-compliance idea |
| W2 Reproduction fix + REPRODUCE.md (requirements, py3.9/py3.12 drill, libomp note) | "Clean-machine reproduction drill", README half of "Deliverable parity" |
| W3 Canonical numbers + claims ledger + doc-drift sweep (CANONICAL.md, git tag) | "Freeze canonical numbers", "Claims ledger + pre-submission audit", doc-drift half of "code.zip builder" |
| W4 Submission hardening (tamper test, FORBIDDEN assert, kill silent fallback, SHA256 log, stale/diverged-seed detector) | "Harden make_submission.py", "Final-fit sanity harness" |
| W5 code.zip builder with NDA-exclusion asserts | "code.zip builder", "Judging-grade code.zip" |
| W6 s45 external-data compliance + cached 6-member fallback score | standalone |
| W7 Contingency pack: 3 pre-built submissions + per-config tail rescale + average-rank table + contingency.py manifest | "Pre-build three metric-contingency submissions", "Metric-contingency drill", "Re-derive quiet-tail scale per config", "Average-rank contingency analysis" |
| W8 R8 non-recursive stock-flow (capped 1 day) | standalone (HANDOFF §5's queued do-first) |
| W9 Report skeleton: thesis-first, named method, INFORMS form | "Error-Budget Thesis Architecture", "The Unfitted Ensemble form factor", narrative half of "Calibrate + build on wins-that-reversed" |
| W10 Winner-calibration memo (fetch 2025 PDFs, structural diff) | both "calibrate against 2025 winner" ideas |
| W11 Figure suite F1–F6 (timeline/trap; error budget + identifiability; graveyard strip; overfitting staircase; skill-by-horizon panel; trajectory money figure with gust overlay + one bad county) | "Five-figure plan", "Money figure", "Ablation Forest/Graveyard", both staircase ideas, figure half of "Act Two", "skill-and-concentration panel", "Forecast Trust Card" |
| W12 Predicted-vs-Measured retrodiction table | standalone (NOVELTY §4's strongest asset) |
| W13 Report boxes: 2025-champion baseline row; P1 sidebar; N_t(71) declined-exploit disclosure + verify_artifact.py + soften "structurally impossible"; three-reversals + corrections trail; per-judge takeaways | "Beat-the-champion row", both P1 sidebar ideas, both declined-exploit ideas, "Negative results as validation discipline", "One-sentence takeaways" |
| W14 PowerOutage.com recommendations (4 measured items, sensors-not-patience framing) | "Recommendations for PowerOutage.com", "What data would make this better", "Sensors, not patience", measured core of "Doctrine slide" |
| W15 One ops-quantification panel: hit rate / false alarms / lead time at 3 fixed thresholds + top-K capture rate vs wave-1-damage allocator + deterministic cost-loss value curve (1 day, from oof/ only) | "Duty-Officer Replay", "Top-K Crew Pre-Positioning Scorecard", value-curve core of "Value-of-forecast cost-loss" |
| W16 Deck (12–14 slides, three acts, 0.00874 reveal, 5 backups) + hostile Q&A bank | "20-slide presentation", "Three-act presentation", "Q&A bank", "protocol-was-the-model triptych" (becomes one slide) |
| W17 **DONE 2026-09-03** — built as `submissions/submission_freeze.csv`, CV 0.008468, 28/28 checks, 11.2 h; spec finalised with i11w and ts2c (metric hour weights), see HANDOFF §1. Original plan follows. | W17 Final-fit seed scaling — **now a measured spec, not a guess** (REPORT §5am): a3x25 at 25 seeds x 4 snapshots (already in the shipped path), **plus i11 and s35 at 25 seeds** (measured −0.00057 and −0.00032 standalone; together −0.000037 on the 8-member ensemble, better at 4/4 horizons; −0.00001 on the shipped blend). Do NOT seed-scale s39 or s40 — both measured WORSE at 25 seeds (+0.00019, +0.00023) because they already average internally. **UPDATED (REPORT §5at/§5au):** also apply **snapshot ensembling to s51**
(3 seeds x 4 snapshots, 246 s, solo 0.00916 -> 0.00884, blend −0.00001 at
P 0.980 with a CI excluding zero). Do NOT use the individually-best i11/s35
variants (25 seeds x 4 snapshots): they score better solo but make the blend
significantly WORSE (P 0.011, CI excludes zero) by converging onto a3x25's
error basis. **PLUS a 9th member: PatchTST-KF (REPORT §5av)** — 0.00951 solo, corr 0.762–0.860,
additive only (both swaps lose), blend 0.008521 -> **0.008507**. Not significant
(P 0.812) and selected from 3 architectures x 4 configs, so treat the point
estimate as optimistic. Expected freeze target: **0.008507**. Cost ~3 h CPU. | "Seed-scale the final fit" |

Stretch (do only if time remains): member-spread uncertainty bands with one nested inflation scalar (diagnostic only); learning-curve 191→239 check (verification only — acting on it would be a new experiment); pre-rendered 30–45s storm-replay animation for the deck.

## 2. RANKING (wow-per-effort × credibility)

QUICK WINS (≤1 day each), in order: **W1** (highest information/minute; two submission decisions and two compliance risks hang on the answers) → **W2** (verified near-certain judge-facing crash today; asymmetric downside) → **W3** (a rigor-themed report citing numbers contradicted inside its own code.zip is the cheapest way to lose; verified live traps) → **W6** (only DQ-shaped risk; fallback score = 30 min from oof/) → **W4** (turns the causality claim into an executable proof; closes verified silent-fallback hole) → **W12** (pure assembly; flips ~40 negatives into evidence — nothing like it exists in competition reports) → staircase F4 + money figure F6 from W11 (each hours; the two highest-value figures) → **W13** boxes (champion row costs one paragraph and anchors everything) → **W7** (converts two documented swap decisions into file swaps; banked -3% MAE branch) → **W10** → **W14** → **W15** → **W17**.

BIG BETS (multi-day): **W9 the report itself** (~5 part-time days; the deliverable that decides winners per 2025 precedent) → **W11 remaining figures** (~2 days) → **W16 deck + Q&A** (~2–3 days, after report freeze) → **W8 R8** (1 day cap; the only sanctioned score-side item; settles the P1 sidebar and possibly the RMSE branch either way).

## 3. CENTERPIECE AND WOW DEMO

**Report centerpiece: "The budget that predicted the ladder"** — one spread: thesis sentence (timing free at 2% of MSE; remainder splits scale 45 / shape 37; prefix R² ≤ 0.08 at every length) + the two-panel error-budget figure F2 + the Predicted-vs-Measured retrodiction table (W12).
Defense vs runners-up: the overfitting staircase (runner-up) is the strongest single *figure*, but it is defensive — it certifies the number rather than explaining the problem, and it slots naturally as row 5 of the retrodiction table ("combiner d.o.f. cannot be estimated from 239 counties → fitted weights must reverse → measured 0.00850→0.00959"). The graveyard has breadth without mechanism. The error budget is the only asset that is simultaneously the scientific claim NOVELTY.md says survives prior art, the explanation of the plateau, the justification for equal weights, and the source of the PowerOutage.com recommendations — every other exhibit becomes a corollary of it, which is what a thesis is. Overclaim guards: MSE-fraction axis labels; claim the level of the identifiability curve, never saturation; "retrodicts/is consistent with" wherever the diagnostic postdates the experiment.

**Presentation wow demo: the 0.00874 reveal**, warmed up by a 20-second live tamper test (corrupt every post-origin outage value on stage, run s45, forecast unchanged). Show only the in-sample curve descending to 0.00874 — "the score we could have reported" — pause, drop in the nested line rising to 0.00959 and the zero-parameter point at 0.00888/0.00870.
Defense vs runner-up (Storm-Replay Situation Room): the reveal is built entirely from numbers already in weight_results.csv — near-zero execution risk, zero new build days — and it converts the driest judging criterion (validation hygiene) into the one moment an academic panel audibly reacts to; the 2025 flip was won on report/presentation quality, not demo flash. The interactive dashboard costs 2–3 of the scarcest days, carries live-demo failure risk in a ~15-minute slot, and its NDA handling is delicate (see rejects). Capture 80% of its value for 5% of the cost: a pre-rendered 30–45s replay animation (matplotlib frames from oof/ens_prod.csv) as one deck slide aimed at the PowerOutage.com judge — post-submission stretch only. Backup slide for the reveal: "this is exactly what naive stacking tutorials do" (big-pool NNLS row), for the judge who asks whether anyone actually reports non-nested numbers.

## 4. TWO-WEEK EXECUTION ORDER (one person, ~3–4 h/day, Aug 12–26)

Week 1 — unblock, de-risk, first figures:
- D1: Fill in and send W1 email (user sends; add appendix-rule and data-classes questions to the existing draft). Fix requirements.txt (xgboost==2.1.4, lightgbm==4.6.0, libomp note). Start clean py3.9 venv drill.
- D2: Finish W2 (second drill on py3.12/torch 2.x; write REPRODUCE.md with measured reproduction tolerance). W4 hardening: kill silent subcounty fallback, FORBIDDEN assert, tamper test, SHA256 logging.
- D3: W3 doc-drift sweep (README 7-member, RESULTS.md line 85, REPORT §8 filename, one idea-count convention) + CANONICAL.md skeleton. W6: compliance re-read + 6-member fallback score from oof/.
- D4: W8 R8, capped at one day; pre-stated win condition: MAE no longer degrades AND ~0.75-class decorrelation survives. Either outcome finalizes the P1 sidebar text.
- D5: W7 contingency pack (R8 outcome feeds the RMSE branch): per-config tail rescale (identical nested procedure, zero new choices), three frozen CSVs, manifest, average-rank table from oof/.
- D6 (weekend slot): W10 winner memo. F4 staircase regenerated on the current 7-member pool via weight_search.py (kills the era landmine); F6 money figure incl. one deliberately bad county + gust overlay.

Week 2 — the report:
- D7: F1, F2, F3, F5.
- D8–9: Draft pages 1–3 (thesis + trap + headline numbers; validation protocol as first-class methods; error budget + W12 table; champion baseline row).
- D10: Draft pages 4–6 (laws synthesis; final model + §5d significance table + staircase; P1 sidebar; declined-exploit box + verify_artifact.py; reversals/corrections box; W14 recommendations; per-judge takeaways).
- D11: W15 ops panel, timeboxed one day, commit in advance to reporting whatever comes out (wave-decorrelation r≈0.16 says we beat the wave-1 allocator; measure Spearman before claiming ranking skill).
- D12: Full-draft number audit — wire every printed number to CANONICAL.md and its regenerating script; run paper-revision-audit skill on the draft.
- D13: W5 code.zip builder with NDA-exclusion asserts; rebuild the clean-machine drill FROM the zip.
- D14: Freeze candidate: final retrain with W17 ~24-seed members, seed-noise diff (report CV stays the 3-seed 0.00870; state final fit uses more seeds for variance only), git tag, CANONICAL.md final.

Aug 27–Sep 25 buffer: cold re-read + second revision-audit pass (~Sep 1), react to chairs' answer (swap pre-built file per manifest — a lookup, not a rebuild), submit ~Sep 18, a week early. After report freeze: W16 deck + Q&A + rehearsal (workshop is after the deadline; do not let deck work precede report freeze). If time runs short inside the two weeks, cut in this order: W15 → stretch items → W8; never cut report days.

## 5. REJECT LIST (with reasons)

1. **Crew Workload Forecast from stock-flow flows** — REJECT. P1's flows are deliberately ~0.70× shrunk (optimal shrinkage under squared loss, REPORT §5r); publishing them as jobs/hour either mislabels shrunk conditional means as calibrated demand or requires the multiplicative calibration that measurably made things worse (R6; empirical law 6, closed list). P1 appears only as the sidebar.
2. **County Storm Ledgers standalone viz** — REJECT. Same shrinkage caveat, days of work; one panel inside the P1 sidebar/backup slide suffices.
3. **County restoration ETA table** — REJECT. Threshold-sensitive, crossing-time validation is noisy at tail truth ≈0.00035, day+ effort, and the miss-rate honesty cost buys little; the timing claim is already carried by the phase measurement. One sentence in W14.
4. **Wave-2 Arrival Board** — REJECT as a product. Requires a pre-registered onset rule to dodge cherry-picking; the ±2h/79% number already exists and carries the claim with zero new machinery.
5. **Full conformal exceedance + probabilistic cost-loss chain** (3 ideas) — REJECT the chain. Days of work, no scored benefit, and severity-conditional calibration re-enters fitted-correction territory the project closed. Kept instead: W15's deterministic hit/FA/lead-time + value curve; member-spread bands with one nested scalar as an optional diagnostic.
6. **County Storm Playbook generator** — REJECT for this window; it is a rendering layer over products being trimmed. Revisit post-submission only if the deck needs one exemplar.
7. **bench/ mini-benchmark packaging** — DEFER post-competition. Refactor risk beside the frozen production path, organizer-approval gating, and the entire pre-deadline payoff is one report sentence — keep the sentence, not the refactor.
8. **Shipping the Storm-Replay HTML inside code.zip** — REJECT this specific part on NDA grounds: the rendered HTML embeds truth trajectories derived from DM_Train.csv, i.e., NDA data leaving the machine via the Drive submission, contradicting W5's own exclusion assert. Permitted variant: ship the builder script; keep the HTML local; pre-rendered clip in the deck.
9. **20-slide deck** — REJECT the length for a likely 15-minute slot; 12–14 slides + 5 backups.
10. **"Mid-storm reallocation is chasing noise" doctrine phrasing** — REJECT the phrasing as overclaim: R² ≤ 0.08 measures outage-prefix→future-scale only, not all reallocation information. Keep the exactly-measured version in W14.
11. **"Causality violations are structurally impossible" phrasing** — must be softened everywhere (the N_t(71) artifact means the test file is not information-tight); replaced by "removed before feature construction and verified by corruption-invariance."
12. **"The Unfitted Ensemble: zero fitted parameters" tagline** — keep the name, fix the claim: "zero fitted **combination** parameters; one nested, mechanism-backed bias parameter (the quiet-tail scalar, 0.667)". TAIL_SCALE is fitted and a judge will find it.
13. **Cross-year superiority claims** ("47% vs their 8%") — context sentence only, never a superiority claim: different target, different metric regime, and the tone risk with a panel that may know the 2025 team.
14. Nothing proposed reopens the closed-experiments list or touches post-origin outage data; R8 is explicitly sanctioned by HANDOFF §5, seed-scaling is law-2-consistent at final fit only, and the per-config tail rescale reuses the adopted procedure unchanged. The learning-curve check is admitted only as verification — any epoch/capacity retuning it suggests would be a new experiment requiring the full adoption protocol.

Key file paths: /Users/hkhosravi7/Documents/GitHub/DM_compettion/{HANDOFF.md, REPORT.md, NOVELTY.md, EMAIL_TO_CHAIRS.md, requirements.txt, make_submission.py, weight_search.py, fit_tail_scale.py, src/ideas4.py (ENSEMBLE_FINAL_W + TAIL_* gotcha), src/subcounty.py (silent fallback), results/{RESULTS.md, weight_results.csv, ideas_results.csv}, oof/ (all member + p1b/e2/zero/decay caches), submissions/submission_ensemble_final.csv}.