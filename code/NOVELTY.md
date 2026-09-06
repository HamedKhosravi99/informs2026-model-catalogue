# Research contribution: what survives an adversarial prior-art check

This document proposes what is genuinely new in this project, after (a) testing the
ideas against the data and (b) an adversarial prior-art search across five literature
angles. Both steps removed more than they left. What remains is stated as a single
rejectable claim, with the prior art named.

**Bottom line up front: TPAMI is not realistic and is not the right venue.** The
surviving contribution is an empirical measurement plus a validated diagnostic, not a
new learning theory or model class. The honest targets are the International Journal of
Forecasting, IEEE Trans. Power Systems, or a NeurIPS datasets-and-benchmarks track.

---

## 1. The setting (not novel — name it correctly)

**Entity-transfer forecasting with known forcing**: entities fully observed in training;
new entities at deployment carrying only a short prefix of the response; exogenous
forcing known over the whole horizon.

This is the *Prediction in Ungauged Basins* programme (Kratzert et al., WRR 2019;
Nearing et al., HESS 2023), and with a short record it is "almost ungauged" catchment
prediction (Seibert & Beven, HESS 2009; Rojas-Serna et al., WRR 2016). In ML it is the
standard evaluation protocol for covariate-aware zero-shot forecasters and the design
premise of TFT and TiDE. **Any claim that this setting is unstudied is false and would
be a desk-reject trigger.**

## 2. What was proposed and what the literature already owns

| proposed part | closest prior art | verdict |
|---|---|---|
| oracle decomposition of error into scale / shape / phase | **Ebert & McBride, J. Hydrol. 2000 (CRA)** — literally `MSE = displacement + volume + pattern` by oracle substitution; **Murphy, MWR 1988** potential skill; **Gupta et al. 2009 (KGE)** = correlation/variability/bias, the *default reporting format* in the nearest applied literature; **Hoiem ECCV 2012 / TIDE ECCV 2020** for the oracle-substitution protocol | **not novel** |
| prefix-length identifiability curve **[the shape half is withdrawn — audit 2026-08-14, REPORT §5ak: the cosine statistic is uninformative against a shuffled-county control; cite I05 archetype routing instead]** | **Seibert & Beven, HESS 2009** (sweep revealed observations, report skill recovery); **Savic & Karlsson, AAPS J 2009** (η-shrinkage vs observations per subject, used as an ex-ante gate); context-length ablations in time-series foundation models | **method anticipated**; the *scale-vs-shape resolved* version was not found |
| reliability-calibrated shrinkage of entity scale | **Kelley 1923**; **James & Stein 1961**; **Efron & Morris 1975**; **Bühlmann–Straub credibility** `Z = n/(n+k)` with intensity from variance components — exactly the claim; **Miller & Williams, IJF 2003/2004** publish it for per-series forecasting components | **subsumed — dropped** |
| "identification is paid in exposure, not hours" | **Bühlmann–Straub exposure units** (weights are exposure, not calendar time); Gamma frailty is **Vaupel et al. 1979** | **an instantiation, not a discovery** |

Part 3 was dropped on two independent grounds: it is a century old, **and it failed in
our own experiment** (0.00888 → 0.01163, §5). A prescription that is both known and
refuted cannot be a contribution.

## 3. The surviving claim, stated so a reviewer can reject it

> In entity-transfer forecasting with known future forcing, **timing error is
> **[WITHDRAWN 2026-08-14 — see the correction block below. Do not use this claim.]**
> essentially free** — the best per-entity ±12 h shift removes only **2% of MSE**, with
> the optimal shift 0 h for the median entity and |shift| ≤ 2 h for **79%** of them —
> while the remaining error splits into comparable **entity-scale (45%)** and
> **profile-shape (37%)** components with no dominant failure mode; and prefix
> information about entity scale is **low at every observation length tested**
> (R² ≤ 0.08, 95% CI [0.024, 0.157] at its best), with **no measurable gain from 36 h to
> 120 h of observation**. All of it is measurable ex ante, retraining-free, from cached
> out-of-fold predictions.

**[ALSO WITHDRAWN — the "R² ≤ 0.08 at every observation length" sentence above is a
SECOND defective claim from the same script, separate from the phase claim below and
not covered by that withdrawal. It came from an in-sample Pearson against a fixed
h121–215 sub-window with no covariates and no holdout. On the deployment window the
same construction gives R² 0.372, an OOF county-holdout model gives 0.55–0.60, and the
shipped model's own predicted-vs-true county mass gives 0.69 (log) / 0.88 (raw).
County magnitude is substantially predictable; only the post-model RESIDUAL is not.
See HANDOFF law 8 (restated) and REPORT §5ak.]**

**[WITHDRAWN — the "2%" was a `np.roll` circular-shift artifact; the honest figure
is 16.8% of MSE (edge-padded shift), verified twice on 2026-08-14. Phase is NOT
free, and this must not appear in the report. The replacement flagship claim is
the spatial-whiteness result recorded at the end of this file.]**

~~The least-precedented element is "phase is free under known forcing."~~ Spatial
weather-verification decompositions (CRA, SAL) exist *because* displacement error
dominates in that setting. Measuring it at ~2% when the forcing is known is a crisp,
surprising, checkable statement about what this setting does to the error budget, and
the prior-art sweep found no previous measurement of it.

## 4. The strongest asset: retrospective validation against 56 experiments

No prior work validates a forecasting diagnostic by showing it would have predicted an
existing experiment ladder **including the negative results**. Ours does:

- **Six** independent attempts to recover the per-entity **residual multiplier**
  — the scale error that remains *after* the model has spoken — all failed:
  per-county residual calibration (0.00937 → 0.01099), severity-gated experts
  (0.01106, vs 0.00966 with oracle routing — so the ~0.0014 gap is entirely gate
  error), per-county restoration rates (0.01131), retrieval of analog futures
  (0.01045), external county data in a sequence model (0.00956 vs its 0.00910
  control), and test-time latent adaptation (0.0092155 vs 0.0092153 with the
  latent forced to zero — the cleanest null of the set, since the control is the
  identical model).
  *Framing corrected 2026-08-14:* this used to read "infer entity magnitude from
  the observed window", which restated law 8 far more broadly than it was ever
  measured. Entity magnitude **is** substantially predictable (OOF R² 0.55–0.60;
  the shipped model's own predicted-vs-true county mass is R² 0.69 log / 0.88
  raw). What resists correction is the *residual*, which is what all six attacks
  actually targeted — so every verdict stands, but the claim is now the one the
  experiments support.
- The largest tabular feature-family gain came from **static** entity attributes
  (0.01094 → 0.01047), which is where the diagnostic says the scale information
  must come from. (Nominally I17b static+physics edges it at 0.010456 vs
  0.010466 — a difference in the fifth decimal, and it is statics plus physics
  on top, so the mechanism is the same. Superseded in absolute terms: the modern
  feature set later took the tabular member to 0.00930.)
- Five architectural substitutions — TCN (0.01054), N-BEATS-style basis
  (0.01060), TiDE (0.01121), TFT-lite (0.00982) and multi-task (0.00984) — all
  landed **at or behind** a plain GRU (0.00978), consistent with "no dominant
  failure mode for an architecture to fix". TFT-lite and multi-task are behind by
  0.00004 and 0.00006, i.e. ties, and are described as such. *Stochastic weight
  averaging (0.00993) was previously counted here as a sixth "architecture"; it
  is an optimizer/weight-averaging trick, not a function class, and is
  reclassified as variance reduction (where it is subsumed by seed averaging).*

This is the part to lead with.

## 5. Corrections forced by testing and by the prior-art check

- **Saturation was over-read.** With bootstrap CIs (4000 resamples, n = 239):
  R²(L=120) − R²(L=36) = +0.027, 95% CI **[−0.064, +0.116]** — *not distinguishable*.
  Only the L=12 → L=120 difference reaches marginal significance (CI [+0.000, +0.147]).
  The defensible statement is about the **level** (R² ≤ 0.08 everywhere), not about a
  saturation point. "Saturates at 36–48 h" has been removed.
- **Identifiability-derived shrinkage failed** (0.00888 → 0.01163) — the third failure
  of multiplicative scale manipulation here, alongside county-normalised targets
  (0.02164) and binned-classification reconstruction (0.03353), both worse than
  predicting zeros. Diagnosing a component as unreliable does not license correcting it
  multiplicatively.
- **A record-driven fragility model was proposed and refuted** before implementation:
  damage `C_c(t) = N_c·F_c(max_{s≤t} u_s)` predicts damage concentrated at forcing
  records; measured, records are 4.2% of hours and carry 19.5% of damage (5.5×
  elevated rate, real but partial), the second-wave exceedance ratio is 1.3× rather
  than ≫1, and out-of-support counties are predicted *better*, not worse (0.00588 vs
  0.00656). Hourly evidence favours sustained exposure (6 h wind-run > 30 mph,
  r = +0.236) over instantaneous gust (+0.210) and far over record advance (+0.057).
- **The depletion narrative was over-read.** The 6× wave-2 reduction is ~half
  compositional; the within-county matched ratio across 144 counties is 3.0×. Between
  counties, prior damage predicts *more* damage (frailty), not less.

## 6. What would be needed to publish, honestly

1. **External validity**: ≥3 domains plus ≥2 within-domain replications. Best second
   dataset is **CAMELS** (671 basins, rainfall–runoff, known future forcing, the exact
   structural analogue); then ASHRAE Great Energy Predictor (hold out meters),
   PEMS-BAY/METR-LA (hold out sensors), C-MAPSS (hold out units).
2. **Statistical rigour on every curve**: bootstrap CIs and a formal change-point test,
   as applied in §5 — the criticism that killed the saturation claim here would kill the
   paper at review.
3. **The right quantity**: the curve should measure *conditional* information given
   static attributes, not marginal correlation, since the practical question is what a
   prefix adds beyond what a county's attributes already reveal.
4. **Prospective validation**: the diagnostic must predict, ex ante, which modelling
   directions pay off on a dataset the authors had not already mined.

## 7. Why this is still worth more than another ensemble variant

Every competitor will build ensembles, tune weights, and try transformers. Few will
quantify the error budget of their setting, and fewer still will report that their own
elegant hypothesis was refuted by their own test. The durable asset here is not the
RMSE — it is 56 controlled experiments plus a diagnostic that explains, in advance, why
they were always going to land in the same place, and a documented trail of four
over-readings caught and corrected.


---

## CORRECTION AND REPLACEMENT FLAGSHIP CLAIM (2026-08-14)

**Withdrawn.** "Phase is essentially free under known forcing (2% of MSE)." The
figure came from `identifiability_curves.py` using `np.roll`, a *circular* shift
that wrapped each county's quiet tail (mean truth 0.00035) into its wave-1 decay
head (mean truth 0.01935); every nonzero shift therefore paid an artificial seam
penalty and the argmin collapsed to zero. With an edge-padded physical shift the
best per-county ±12 h offset removes **16.8% of MSE** and only **19%** of counties
prefer zero shift (not 61%). Independently reproduced twice. Consequence: E3's
rejection sat inside the real noise bar, so phase modelling is reopened.

**Replacement claim, measured and unanticipated by the prior art.** *With fully
known future forcing, the residual field of a storm-outage forecast is spatially
white.* Moran's I (k=6 NN, row-standardised) of the hourly residual field is
**0.026**, against **0.263** for the truth field and **0.333** for the prediction
field — the fitted model is in fact *more* spatially smooth than reality, and
leaves no exploitable spatial structure behind.

Why this is the better flagship: it is crisp, surprising, and checkable in one
number; it retrodicts SIX independently measured nulls in this project from a
single mechanism (rung 3 regional aggregates, I07/S39b attention pooling, S41
retrieval, the neighbour test in §5i, B3 geo-GNN, and the 2025 fourth-place
team's independently abandoned county-GNN); and unlike the phase claim it is
*not* anticipated by CRA/SAL/KGE, which exist precisely because displacement
error dominates when the forcing is NOT known. The prior-art gap is real: the
spatial-verification literature assumes forecast forcing is uncertain.

**Second report-grade result (measured 2026-08-14).** The submitted score is one
draw from a 63-county lottery: resampling 63 of the 239 counties gives mean
0.00839, **sd 0.00160 (19% of the mean)**, p5–p95 [0.00586, 0.01111]. The 2025
finalist field spanned 177.2–191.2, i.e. **7.9%** of its mean — so the county
sampling noise is **2.4x the entire spread of the field that decided the 2025
competition**. This is the number that makes the project's methodological
discipline quantitatively necessary rather than merely admirable, and it is the
honest frame for every refused fifth-decimal selection in the log.
