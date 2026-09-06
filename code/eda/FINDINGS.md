# EDA Findings — INFORMS 2026 DM Data Challenge

> **Note:** this file is the EDA + early-modeling log. The canonical, up-to-date
> experimental record (all models, results, negatives, submissions) is `REPORT.md`
> at the repo root. Current primary: 5-way ensemble, CV RMSE 0.00930 (v5).

## Modeling status (v1, Aug 2026)

Pipeline: `src/` (data/features/validate/models) + `run_experiments.py` (CV ladder) +
`make_submission.py` (end-to-end submission) + `feature_importance.py`. Pure
pandas/sklearn, seed 42. Protocol-faithful CV: 5-fold county holdout stratified by
state×tier, outage data masked after h71, trajectory scored exactly as the four
submission columns.

Pooled 5-fold RMSE (mean over horizons): zero 0.01632 · persist(h71) 0.06595 ·
decayed persistence 0.01110 · climatology 0.01520 · GBM direct trajectory 0.01094 ·
blend decay→GBM 0.01067 · GBM origin+regional 0.01087 · blend decay→originGBM 0.01073 ·
hurdle 2-part 0.01069 · **blend decay→hurdle 0.01045 (primary)** — best at all four
horizons and −15% MAE (0.00285). Fold spread is large (fold means ~0.0066–0.0147), so
differences <~0.0005 are noise — adopt changes only on consistent multi-horizon wins.
Diagnostics that shaped the model: decay beats learned models in the wave-1 decay band
(h73–96), learned models win from the lull on → sigmoid(H) blend at h≈100; error
concentrates in tier-4 counties (RMSE ~0.025) → two-part hurdle (P(OSI>0.01) ×
regime-conditional GBMs). Top features: osi71·exp(−Δ/16) by far, then new_exposure,
knn_osi, cum_exposure, gust_max24, soil_moist, restored_frac.
Submissions: `submissions/submission_gbm_v1.csv`, `submission_hurdleblend_v2.csv` (current).

Causality reading — DECIDED (user call, Aug 11 2026): permissive. Problem_Description's
table governs: at origin t, any outage value at timestamp ≤ t is allowed, including
train counties' live Mar 14–19 values (operationally realistic — PowerOutage.com sees
all counties). Tested it: origin-indexed model with regional features (LOO state/all
means, gust-correlation kNN neighbors). Result: knn_osi ranks 4th in importance but net
CV gain ≈ 0 — with full future weather provided, regional live outage state is largely
redundant. Kept in the ladder as a documented negative result; primary submission stays
origin-independent (trajectory-consistent), which is also defensible under either
reading. Good report material either way.

## Deep learning results (Aug 12 2026) — it helps, for an identifiable reason

`src/dl.py` (torch 1.12, CPU, deterministic seeds; OOF preds cached in `oof/`):
- **MLP on the same engineered features: 0.01050** — ties the GBM hurdle blend
  (0.01045). Same information → same score; "neural" alone buys nothing. Best t01h
  (0.01164) of the per-row models; needs no decay blend (smooth extrapolation).
- **GRU seq2seq on raw sequences: 0.00978** — beats everything, gap grows with
  horizon (t24h 0.00885, t48h 0.00796 ≈ −9% vs GBM). Architecture: encoder GRU over
  the observed 72h (osi,P,N,R + weather), decoder GRU over hours 73–215 conditioned
  on future weather + static county features + decay-basis inputs, hidden 48,
  dropout 0.15, 3-seed average, inner-county-split early stopping. Why it wins:
  the decoder propagates coherent state across the 143h trajectory and reads the
  full future-weather *sequence* — per-row models can't. This inverts the 2025
  "deep models overfit" lesson precisely because 2026 supplies future weather.
- **Ensemble 0.5 GRU + 0.25 hurdle blend + 0.25 MLP: 0.00944 / MAE 0.00295**
  (PRIMARY). OOF correlations 0.91–0.93; weight optimum is a broad plateau
  (anything near 0.4–0.6 GRU scores 0.00944–0.00948), round weights chosen to avoid
  grid-overfitting. Wins 3/5 folds vs GRU alone and every horizon.
  Submission: `submissions/submission_ensemble_v3.csv` (current).

**Component-structured GRU** (physics head; `gru_component_forecaster`): decoder
predicts P_t/N_t/R_t via three softplus heads, D_t = differentiable 6h rolling mean of
predicted P seeded with observed P(h66–71), OSI composed by its exact formula inside
the network; loss = MSE(OSI) + 0.3·aux component losses, early stop on OSI. Result:
**0.00962** — beats plain GRU (0.00978) at all four horizons and −6% MAE (0.00305).
Honest caveat: tier-4 RMSE did NOT improve (0.02323 vs 0.02284) — gains came from
mid-severity counties (folds 2, 4), so the "fix the severe tail" hypothesis is
unconfirmed; the component structure helps as regularization/inductive bias instead.
The severe tail remains the open problem.

**Primary = 4-way ensemble** 0.3 GRU + 0.3 component-GRU + 0.2 hurdle + 0.2 MLP:
**RMSE 0.00937 / MAE 0.00291**, wins 3/5 folds vs the 3-way (0.00944). Point gain is
within fold noise — adopted for member diversity and metric consistency, stated as
such. OOF correlations: gru↔comp 0.979, seq2seq↔tabular 0.91–0.94. Weight plateau
0.00936–0.00937 is broad; round weights chosen. If the official metric turns out to be
MAE, revisit weights (hurdle-heavy mixes score better on MAE).
Submission: `submissions/submission_ensemble_v4.csv` (current).

Model ladder final table (pooled county-holdout RMSE, mean of 4 horizons):
zero 0.01632 · persist 0.06595 · decay 0.01110 · clim 0.01520 · GBM 0.01094 ·
blend d→GBM 0.01067 · origin+regional 0.01073 · hurdle blend 0.01045 · MLP 0.01050 ·
GRU 0.00978 · comp-GRU 0.00962 · 3-way 0.00944 · **4-way ensemble 0.00937**
(−43% vs zero, −16% vs decay, −10% vs best GBM).

**Per-county residual calibration (Aug 12) — NEGATIVE.** Second-stage Ridge predicting
each county's log(true/predicted) trajectory-sum ratio from observed-window features,
fit out-of-fold, scale clipped to [0.5, 2]: RMSE 0.00937 → 0.01099, tier-4 0.0218 →
0.0266. The county-level residuals of the ensemble are not predictable from the
observed window — tail error is dominated by idiosyncratic realization noise, and any
multiplicative correction amplifies it. Global bias is already negligible (|bias| ≤
0.0006 in every hour band). Two independent tail attacks (component structure,
residual calibration) now agree: the remaining tier-4 error is largely aleatoric.

Next ladder rungs: N_t/R_t as separate targets · per-county restoration rates ·
sharper kNN (windowed gust corr, k=5) or neighbor residual-vs-weather correction ·
lightgbm/tweedie · static enrichment (EIA 861 line miles, NLCD canopy, rural-urban) ·
quantile/uncertainty for the report · GRU capacity/architecture probe (2-layer,
attention over weather) once, with the 0.0005 noise bar enforced.

Figures: `figs/01_timeline.png`, `figs/02_wave_asymmetry.png`, `figs/03_county_heatmap.png`
(regenerate with `python3 eda/eda_figs.py`).

## Event structure

- Wave 1 peaks at **h67 (Mar 13 ~19:00), mean OSI 0.037** — i.e., the test counties'
  observed window (h0–71) ends *right at the wave-1 peak*. The prediction window is
  mostly wave-1 decay (h72–96), a modest wave-2 hump peaking **h112, mean OSI 0.009**,
  and a near-zero tail after Mar 18 (mean OSI < 0.001).
- Wave-2 gusts (peak hourly mean 35.7 mph, h111) rival wave 1 (41.4 mph), but the
  damage response is far weaker: at 40–50 mph gusts, mean N_t is **0.0079 in wave 1 vs
  0.0013 in wave 2 (~6×)**. A pure weather→outage model trained on wave 1 will badly
  overpredict wave 2. Damage response must be conditioned on prior-damage/depletion
  state (osi_lag24h/48h, cumulative damage so far, restoration progress).
- Wave-1 and wave-2 county peaks are nearly uncorrelated (r≈0.16) — different counties
  get hit in each wave; weather drives *where*, prior damage drives *how much*.

## Distribution

- Heavily zero-inflated and tail-driven: 44% of osi values are exactly 0, median
  0.0001, p99 0.125, max 0.65. County peak OSI: median 0.044, 23% exceed 0.10.
  RMSE-type scores will be dominated by a small set of severe county-hours.
- OH is the most severe state (mean OSI 0.011); IN mostly mild but contains the single
  worst county (peak 0.65; FIPS 18111 also shows a real 100% pre-event outage).

## Baselines under the true test protocol (train counties, outage info frozen at h71)

| horizon | RMSE zero | RMSE persist(h71) | RMSE hourly clim | RMSE scaled clim |
|---|---|---|---|---|
| t+1h  | 0.0235 | 0.0633 | 0.0216 | 0.0207 |
| t+6h  | 0.0198 | 0.0642 | 0.0183 | 0.0179 |
| t+24h | 0.0121 | 0.0672 | 0.0114 | 0.0121 |
| t+48h | 0.0099 | 0.0692 | 0.0094 | 0.0104 |

- **Persistence from the last observed hour is ~3× worse than predicting all zeros**,
  because the observed window ends at the wave-1 peak and freezing it ignores decay.
  The competition design explicitly punishes persistence.
- All-zeros is embarrassingly strong; hourly climatology beats it only slightly.
  The real signal to model: (1) wave-1 decay/restoration rate per county,
  (2) the wave-2 hump conditioned on depletion, (3) county fragility heterogeneity.

## Task reframing (important)

The four horizon targets at origin t are just OSI(t+h). Since outage data stops at
h71 for all test origins (h72–215), **the whole task collapses to: forecast each test
county's OSI trajectory for hours 73–215**, then fill the submission by indexing that
trajectory (osi_target_tXXh at origin t = predicted OSI at t+XX; leave targets beyond
h215 as NaN). Predictions should be internally consistent across the four columns.

## OSI component structure (matters for component-based models)

- `D_t` is *exactly* the 6h rolling mean of `P_t` (err < 5e-9) — deterministic given P_t.
- `N_t`/`R_t` are **not** `max(0, ±Δ)/customersTracked` as documented (corr with the
  same-hour county deltas is only 0.73/0.65, and 26.8% of rows have N_t>0 AND R_t>0
  simultaneously, impossible under a net-delta definition).
  **CORRECTION (Aug 12, found by independent audit):** they *are* exactly reconstructible
  — each is a **centered 3-hour rolling mean** of the corresponding county-level gross
  flow. Verified: corr 0.99884 (N_t) and 1.00000 (R_t), mean |err| 2.5e-06 and 1.9e-07,
  with near-symmetric correlations against hours t−1/t/t+1 (0.674/0.733/0.674). An
  earlier version of this file claimed they came from unavailable sub-county data; that
  was wrong.
- **Consequence — the organizers' OSI is not strictly causal.** Because OSI(t) uses
  N_t(t) and R_t(t), it contains a small amount of hour-(t+1) information by
  construction. At the freeze boundary this means `N_t(71)`/`R_t(71)` in the test file
  encode information about hour 72, the first hidden hour.
  **We do not exploit this.** Deliberately inverting those columns to reconstruct
  hour-72 outage would be "computing an outage-derived quantity" for a future timestamp
  — against the clear intent of the causality rule, and code review is part of the
  evaluation. We use the provided columns as-is at timestamps ≤ 71, which the rule
  table explicitly permits, and never invert them.
- Consequence: OSI derived from a predicted P_t trajectory alone (N≈max(0,ΔP),
  R≈max(0,−ΔP), D=roll6) is an approximation: corr 0.996 with true OSI, mean |err|
  4.3e-4, 98.1% of rows within 0.005 — but it misses flow spikes (max err 0.15) exactly
  at storm onset where N_t (weight 0.35) drives the peaks. A component model should
  predict N_t and R_t as separate targets, not derive them from P_t.

## 2025 competition lessons (verified via web, Aug 2026)

- 2025: Michigan 83 counties, Apr–Jun 2023 hourly outage counts + ~109 weather vars,
  24h/48h horizons, RMSE, ranking = average rank across horizons, **no future weather**.
- Winners: 1st UCSC two-stage hurdle (TimesNet weather forecast → LightGBM
  P(outage)×intensity, isotonic calibration, recursive multi-step;
  github.com/shourya01/hurdle_model_outage_forecasting, leaderboard RMSE ≈189);
  2nd ARIMA–VAR + county clustering; 3rd geo-temporal deep learning; 4th SARIMAX
  (arXiv:2511.01017, RMSE 177.2 vs all-zero 193.4, i.e. only 8.4% better than zeros).
- **1st place had worse RMSE (189) than 4th place (177)** → the report/presentation
  carries decisive weight. Budget serious effort for the 6-page report.
- All-zero baseline was hard to beat in 2025 too (best ≈8% improvement) — consistent
  with our zero-inflation findings.
- What does NOT transfer to 2026: the weather-forecasting stage (2026 supplies future
  weather), per-county SARIMAX (test counties unseen), county-ID memorization.
- Test counties are fully identified (real FIPS + names) — no anonymization.
  **Never** scrape PowerOutage.com/EAGLE-I actuals for Mar 14–19 for these counties:
  that is "using future outage values" = disqualification.
- 2026 metric TBA via Drive; likely per-horizon numerical score with average rank
  across the four horizons (2025 used average rank across its two horizons).

## Mechanics verified

- OSI formula `0.40*P_t + 0.35*N_t + 0.25*D_t − 0.10*R_t, clipped ≥0` reproduces the
  train `osi` column to 5e-5 (rounding). Test observed-window OSI must be
  reconstructed this way (no `osi` column or OSI lags in the test file).
- Test counties look like train counties: similar wave-2 gust exposure (25.1 vs
  25.5 mph mean), slightly milder observed window (mean P_t 0.0083 vs 0.0102).
- Train-only columns to keep away from features: `severity_tier` (CV stratification
  only — forbidden), `peak_pct`, `peak_customers`, `time_to_restore_h` (event-level
  outcomes = leakage), `split`.
- `wind_dir_10m` is circular → encode sin/cos. `event_duration_h` is constant (drop).

## Modeling implications

1. Recursive 1-step forecasting will compound error over 144 steps; prefer a direct
   formulation: predict OSI at target hour H from weather around H (levels, lags,
   rolling max gust), hours-since-last-observation, time-of-day, county fragility
   summaries learned from the observed 72h window, and depletion features (cumulative
   predicted or weather-implied damage before H). A hybrid (short-horizon recursive,
   long-horizon direct) is worth testing.
2. Zero-inflation suggests two-part models (occurrence × severity) or tweedie/log1p
   objectives; evaluate on RMSE and MAE since the official metric is TBA.
3. Validation must mirror the protocol: hold out counties (stratified by state ×
   severity_tier), freeze outage data at h71, score h73–215 trajectories.
4. Optional enrichment allowed by the docs: EIA Form 861, NLCD tree canopy,
   USDA rural-urban codes — plausible county-fragility priors.
