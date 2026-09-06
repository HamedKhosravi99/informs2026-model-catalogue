# Complete results table — every idea tried

All rows: identical 5-fold county-holdout CV (whole counties held out, outage data
masked after hour 71, scored as the four submission columns). Regenerate with
`python3 run_all_ideas.py`; raw numbers in `ideas_results.csv`.

Reference points: **all-zeros 0.01632**, persistence-from-h71 0.06595,
SARIMAX(2,0,2) per county 0.02775 (worse than zeros — see REPORT §5s),
**decayed persistence 0.01110**, hourly climatology 0.01520.
**Production SHIPPED (2026-08-14): `0.5 x [8 members, equal weights] + 0.5 x A3x25` + quiet-tail 0.667 — 0.008543 RMSE / 0.00256 MAE.**
**FREEZE SPEC (measured 2026-08-16, not yet built): same structure with i11/s35 at 25 seeds, s51 snapshot-ensembled, and PatchTST-KF as a 9th member — 0.008507 RMSE / 0.00253 MAE.** See REPORT.md 5am-5ax. See REPORT.md 5ag for the blend U-curve and why lambda=0.5 is a-priori rather than fitted.
**Best single model: A3x25 (snapshot ensembling, 25 seeds), 0.00862/0.00247 — ties the shipped ensemble (P 0.630) and does not change it. At matched seed count, snapshots beat plain seeds significantly (A3x25 vs A1x25 −0.00033, P 0.023) — see REPORT.md 5t.**

| Rank | Idea | RMSE | MAE | Family |
|---:|---|---|---|---|
| 1 | **A3x25 snapshot ensemble, 25 seeds** | **0.00862** | **0.00247** | variance |
| 2 | A11x25 snapshot + exact-metric, 25 seeds | 0.00863 | 0.00247 | variance |
| 3 | **A3 snapshot ensemble (cosine restarts)** | **0.00869** | 0.00250 | variance |
| 4 | A11 snapshot + exact-metric weights | 0.00869 | 0.00249 | variance |
| 5 | A1 S62 8 seeds | 0.00883 | 0.00255 | variance |
| 6 | A5 exact-metric hour weights | 0.00888 | 0.00253 | objective |
| 7 | A7b hour-block dropout augmentation | 0.00893 | 0.00255 | augmentation |
| 8 | A7 channel-dropout augmentation | 0.00894 | 0.00258 | augmentation |
| 9 | A1x25 S62, 25 seeds (plain-seed limit) | 0.00894 | 0.00255 | variance |
| 10 | S62 BiGRU + scoring + dual loss | 0.00897 | 0.00257 | sequence |
| 11 | A2b test-time augmentation, 8 views | 0.00899 | 0.00257 | variance |
| 12 | A2 test-time augmentation, 4 views | 0.00900 | 0.00257 | variance |
| 13 | A4 county loss weight g=0.25 | 0.00902 | 0.00268 | objective |
| 14 | S44b scoring-aligned bi-encoder | 0.00904 | 0.00300 | other |
| 15 | S31 bidirectional encoder | 0.00910 | 0.00297 | sequence |
| 16 | RT1 ray-tuned bi-encoder (loss-proxy ASHA) | 0.00914 | 0.00267 | tuning |
| 17 | S51 BiGRU dual raw+sqrt loss | 0.00916 | 0.00265 | sequence |
| 18 | RT3 optuna-TPE bi-encoder (true metric, warm-started) | 0.00916 | 0.00263 | tuning |
| 19 | A4b county loss weight g=0.50 | 0.00918 | 0.00282 | objective |
| 20 | N1b within-group NCL K=4 lam=0.3 | 0.00920 | 0.00262 | diversity |
| 21 | RT4 optuna-tuned f2 XGBoost (ties f2) | 0.00929 | 0.00247 | tuning |
| 22 | I07 cross-county attention | 0.00931 | 0.00306 | attention |
| 23 | B3 geo-temporal adjacency GNN (2025 3rd-place analog) | 0.00932 | 0.00265 | sequence |
| 24 | A8 prefix-truncation augmentation | 0.00934 | 0.00265 | augmentation |
| 25 | FS2e full FE + shadow selection | 0.00934 | 0.00250 | selection |
| 26 | **F1d explicit physics interactions (gust×soil, canopy×gust)** | **0.00935** | 0.00252 | tabular |
| 27 | FS2c full FE + permutation selection | 0.00935 | 0.00253 | selection |
| 28 | S40 multi-origin augmentation | 0.00936 | 0.00323 | augmentation |
| 29 | M3c mixture K=4 no MCL (collapse control) | 0.00938 | 0.00266 | sequence |
| 30 | F1 all independent-agent features | 0.00938 | 0.00253 | tabular |
| 31 | S39 cross-county self-leak fixed | 0.00939 | 0.00314 | attention |
| 32 | S39b cross-county + transductive pool | 0.00939 | 0.00314 | attention |
| 33 | RT2 ray-tuned bi-encoder (true-metric ASHA) | 0.00939 | 0.00266 | tuning |
| 34 | F4 decay-modifier interactions | 0.00941 | 0.00252 | tabular |
| 35 | S45 sub-county gust heterogeneity | 0.00942 | 0.00252 | tabular |
| 36 | S35 mixup augmentation | 0.00944 | 0.00304 | augmentation |
| 37 | FS2b full FE + gain top-40 | 0.00944 | 0.00255 | selection |
| 38 | Cluster-local sqrt-XGB, K=3 (weather+geo) | 0.00944 | — | clustering |
| 39 | I05b archetype ORACLE (ceiling) | 0.00945 | 0.00313 | stage-wise |
| 40 | I20 GRU 8 seeds | 0.00948 | 0.00306 | variance |
| 41 | S46b sub-county scoring-aligned bi-enc | 0.00951 | 0.00307 | sequence |
| 42 | F1e wind-veer only | 0.00954 | 0.00253 | tabular |
| 43 | S30 LSTM seq2seq | 0.00955 | 0.00301 | sequence |
| 44 | I11 storm augmentation | 0.00955 | 0.00295 | augmentation |
| 45 | S47 BiGRU + external county data | 0.00956 | 0.00309 | sequence |
| 46 | S43 bagged bi-encoder (county subsets) | 0.00957 | 0.00310 | augmentation |
| 47 | S46 sub-county bi-encoder | 0.00958 | 0.00313 | sequence |
| 48 | Similarity-weighted local training (soft) | 0.00958 | — | clustering |
| 49 | S29d sqrt + XGBoost | 0.00960 | 0.00251 | tabular |
| 50 | S54 severe-event auxiliary head | 0.00960 | 0.00303 | sequence |
| 51 | A9 multi-origin 47/71/103 (lull origin) | 0.00960 | 0.00315 | augmentation |
| 52 | S29c sqrt + LightGBM | 0.00961 | 0.00255 | tabular |
| 53 | U1 sqrt-XGB + statics + true 2.5-km URMA field | 0.00961 | 0.00258 | tabular |
| 54 | F1b anticipation clocks only | 0.00962 | 0.00256 | tabular |
| 55 | S47b BiGRU + ext + scoring-aligned | 0.00963 | 0.00308 | sequence |
| 56 | S53 shared encoder + state residuals | 0.00964 | 0.00325 | sequence |
| 57 | M3b mixture-of-trajectories K=8 | 0.00966 | 0.00273 | sequence |
| 58 | S24b sqrt target | 0.00967 | 0.00257 | tabular |
| 59 | F3 winter-hazard load features | 0.00968 | 0.00257 | tabular |
| 60 | U1b statics + proxy sub-county + URMA | 0.00969 | 0.00254 | tabular |
| 61 | S23b ExtraTrees | 0.00970 | 0.00310 | tabular |
| 62 | S52 per-column horizon specialists | 0.00970 | 0.00320 | sequence |
| 63 | S50 BiGRU sqrt target | 0.00971 | 0.00261 | sequence |
| 64 | Pooled 5-family ensemble (cluster experiment control) | 0.00971 | — | clustering |
| 65 | A9d multi-origin 47/71/127 | 0.00973 | 0.00317 | augmentation |
| 66 | F1c pre-event fragility only | 0.00974 | 0.00257 | tabular |
| 67 | I12 TFT-lite | 0.00982 | 0.00333 | sequence |
| 68 | S37 multi-task (OSI+P+N) | 0.00984 | 0.00322 | sequence |
| 69 | I08 transductive pretraining | 0.00984 | 0.00322 | other |
| 70 | A9b multi-origin 71/103 | 0.00984 | 0.00326 | augmentation |
| 71 | M3 mixture-of-trajectories K=4 + MCL | 0.00984 | 0.00277 | sequence |
| 72 | Cluster-local sqrt-XGB, K=4 | 0.00985 | — | clustering |
| 73 | S29 sqrt + ExtraTrees | 0.00987 | 0.00248 | tabular |
| 74 | S29b sqrt + RandomForest | 0.00988 | 0.00256 | tabular |
| 75 | D3 TabNet on the f2 feature set | 0.00989 | 0.00250 | neural-tabular |
| 76 | S36 stochastic weight averaging | 0.00993 | 0.00324 | variance |
| 77 | FS2a full FE + corr filter | 0.00994 | 0.00262 | selection |
| 78 | S24 log1p target | 0.00996 | 0.00275 | tabular |
| 79 | FS1d base + lasso selection | 0.00996 | 0.00266 | selection |
| 80 | A9c multi-origin 47/59/71/103 | 0.00997 | 0.00338 | augmentation |
| 81 | D1 MLP on the f2 feature set (corr 0.741) | 0.00998 | 0.00254 | neural-tabular |
| 82 | S29e sqrt + bagged ExtraTrees | 0.01000 | 0.00252 | tabular |
| 83 | S44 scoring-aligned GRU | 0.01004 | 0.00321 | other |
| 84 | S23 Random Forest | 0.01005 | 0.00328 | tabular |
| 85 | FS0 base features, no selection (control) | 0.01005 | 0.00269 | tabular |
| 86 | D4 FT-Transformer on the f2 feature set (4.3 h) | 0.01007 | 0.00262 | neural-tabular |
| 87 | FS1e base + shadow selection | 0.01009 | 0.00264 | selection |
| 88 | FS1b base + gain top-40 | 0.01010 | 0.00266 | selection |
| 89 | Cluster-local 5-family ensemble, K=3 | 0.01011 | — | clustering |
| 90 | FS2d full FE + lasso selection | 0.01012 | 0.00269 | selection |
| 91 | FS1a base + corr filter | 0.01013 | 0.00269 | selection |
| 92 | K1 kernel-ridge functional output (sqrt) | 0.01017 | 0.00264 | kernel |
| 93 | Cluster-local sqrt-XGB, K=3 (multi-family run) | 0.01019 | — | clustering |
| 94 | I09 Tweedie emission head | 0.01022 | 0.00305 | loss |
| 95 | I05 archetype gate | 0.01022 | 0.00327 | stage-wise |
| 96 | S22c LightGBM DART | 0.01025 | 0.00317 | tabular |
| 97 | M2 cumulative-curve reformulation (integrals) | 0.01025 | 0.00313 | stock-flow |
| 98 | K1b kernel-ridge functional output (raw) | 0.01025 | 0.00353 | kernel |
| 99 | S21 XGBoost | 0.01028 | 0.00334 | tabular |
| 100 | I10 MC-dropout sampler | 0.01032 | 0.00342 | other |
| 101 | I14 hierarchical pooling | 0.01032 | 0.00324 | stage-wise |
| 102 | M1c stock-flow MC rollout, damage-only (no DAgger) | 0.01039 | 0.00300 | stock-flow |
| 103 | FS1c base + permutation selection | 0.01040 | 0.00271 | selection |
| 104 | S22 LightGBM | 0.01042 | 0.00338 | tabular |
| 105 | S22b LightGBM Tweedie | 0.01043 | 0.00249 | tabular |
| 106 | S41 RAFT retrieved analogs | 0.01045 | 0.00334 | other |
| 107 | I17b static + physics | 0.01046 | 0.00333 | tabular |
| 108 | I16 static enrichment | 0.01047 | 0.00334 | tabular |
| 109 | S27 feature selection top-40 | 0.01047 | 0.00340 | tabular |
| 110 | S28 quantile-transformed features | 0.01052 | 0.00336 | tabular |
| 111 | M1b stock-flow MC rollout, DAgger-1 | 0.01052 | 0.00291 | stock-flow |
| 112 | S32 temporal CNN (TCN) | 0.01054 | 0.00365 | sequence |
| 113 | D5 KAN proper B-spline (corrected impl) | 0.01059 | 0.00301 | neural-tabular |
| 114 | S33 basis expansion (N-BEATS style) | 0.01060 | 0.00375 | sequence |
| 115 | I06 quantile two-stage | 0.01063 | 0.00331 | other |
| 116 | S21b XGBoost Tweedie | 0.01066 | 0.00254 | tabular |
| 117 | M1 stock-flow MC rollout, teacher-forced | 0.01067 | 0.00296 | stock-flow |
| 118 | S34 physics-informed net | 0.01071 | 0.00378 | physics |
| 119 | I17 physics indices | 0.01081 | 0.00336 | tabular |
| 120 | M1d MC teacher-forced, 64 paths | 0.01084 | 0.00300 | stock-flow |
| 121 | I04 wave-conditioned cascade | 0.01086 | 0.00346 | stage-wise |
| 122 | I18 in-dataset pretraining | 0.01086 | 0.00372 | other |
| 123 | D2 Chebyshev-KAN on the f2 feature set | 0.01092 | 0.00383 | neural-tabular |
| 124 | S23c Bayesian Ridge | 0.01096 | 0.00409 | tabular |
| 125 | N1c within-group NCL lam=0.6 | 0.01111 | 0.00339 | diversity |
| 126 | B2b cluster-VARX direct form (2nd/4th family, adapted) | 0.01113 | 0.00388 | linear |
| 127 | S23d ElasticNet | 0.01115 | 0.00406 | tabular |
| 128 | I02 impulse-response (system ID) | 0.01120 | 0.00413 | physics |
| 129 | S42 TiDE (known-future covariates) | 0.01121 | 0.00382 | sequence |
| 130 | I15 survival restoration | 0.01131 | 0.00288 | physics |
| 131 | I13 Poisson/Tweedie GBM | 0.01142 | 0.00280 | loss |
| 132 | N2b anti-corr vs ensemble lam=1.0 (corr 0.802) | 0.01187 | 0.00340 | diversity |
| 133 | N2 anti-corr vs ensemble lam=0.5 (corr 0.801) | 0.01209 | 0.00347 | diversity |
| 134 | I03 FPCA two-stage | 0.01247 | 0.00407 | decomposition |
| 135 | Cluster-local Ridge, K=3 (only family that IMPROVES) | 0.01373 | — | clustering |
| 136 | I01 amplitude x shape | 0.01386 | 0.00408 | decomposition |
| 137 | S26 county-normalized target | 0.02164 | 0.00543 | other |
| 138 | B2 cluster-VAR closed-loop (2025 2nd-place literal) | 0.02223 | 0.00906 | linear |
| 139 | S25 binned classification | 0.03353 | 0.00610 | other |

Ensembling and meta-learning (from cached out-of-fold predictions):

| Approach | RMSE | MAE | Note |
|---|---|---|---|
| **Production ensemble (8 members, equal weights, + tail 0.667)** | **0.00864** | **0.00275** | shipped |
| previous 7-member set (frozen as contingency_7member.csv) | 0.00868 | 0.00262 | superseded — dominated by the shipped blend on BOTH metrics, see REPORT 5ap |
| Equal-weight averaging | 0.00911 | 0.00281 | beats learned stacking |
| Ridge stacking (nested) | 0.00934 | 0.00275 | no better than fixed weights |
| Greedy weight selection (nested) | 0.00946 | — | **significantly worse** |
| NNLS over 63 models (nested) | 0.00959 | — | 0.00883 in-sample: a 0.00076 illusion |
| GBM stacking + context | 0.01098 | 0.00330 | **overfits badly** |

Full run log with timings: `ideas_results.csv` (every evaluated idea). The ranked
table above covers the sweep-era ideas plus the A-series (REPORT.md 5t); the later B/E/H/P/R/T experiments are
logged with full breakdowns in `REPORT.md`.


## Sweep 17–18 additions (2026-08-15/16) — variance-reduction grid and modern architectures

Appended rather than re-ranked. `*` marks the variant adopted into the freeze spec.

| Idea | RMSE | MAE | note |
|---|---|---|---|
| **X9 s51 dual-loss + snapshots** `*` | **0.00884** | 0.00255 | 246 s; best individual member; blend −0.00001 at P 0.980, CI excludes zero |
| X7 i11 storm-jitter, 25 seeds x 4 snapshots | 0.00892 | 0.00284 | best i11 solo, but blend gets WORSE (corr 0.950 vs a3x25) |
| **X3 i11 storm-jitter, 25 seeds** `*` | 0.00898 | 0.00278 | adopted instead of X7 — see REPORT 5at |
| X6 s35 mixup + snapshots | 0.00902 | 0.00299 | 553 s; beats 25 seeds at 1/15 the cost |
| X8 s35 mixup, 25 seeds x 4 snapshots | 0.00905 | 0.00301 | |
| X5 i11 storm-jitter + snapshots | 0.00911 | 0.00289 | 540 s vs 11364 s for 25 seeds |
| **X4 s35 mixup, 25 seeds** `*` | 0.00912 | 0.00295 | adopted instead of X6 — see REPORT 5at |
| X10 s51 dual-loss, 25 seeds | 0.00924 | 0.00262 | WORSE than its 3-seed baseline; snapshots win outright |
| X1 s39 cross-county attention, 25 seeds | 0.00958 | 0.00303 | worse than 3 seeds (law 13b) |
| X2 s40 multi-origin, 25 seeds | 0.00959 | 0.00303 | worse than 3 seeds (law 13b) |
| **TS2b PatchTST-KF winsorised** `*` | **0.00951** | 0.00291 | Nie ICLR 2023; 9th member; corr 0.762 vs i11 |
| TS3b iTransformer-KF winsorised | 0.01023 | 0.00315 | Liu ICLR 2024 |
| TS1b DLinear-KF winsorised | 0.01204 | 0.00337 | Zeng AAAI 2023 |
| C1 f2 restricted to provided data only | 0.01005 | 0.00269 | external-data compliance contingency (REPORT 5aq) |
| TS1/TS2/TS3 (unwinsorised) | 0.08799 / 0.02550 / 0.05797 | — | **INVALID** — unbounded static path, one county at 140 sigma carried 99.9% of fold error (REPORT 5av) |
