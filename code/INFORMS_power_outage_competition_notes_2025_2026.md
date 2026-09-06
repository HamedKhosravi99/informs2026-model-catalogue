# INFORMS Power-Outage Forecasting Data Challenge Notes
## 2025 competition, 2026 competition, winning approaches, key differences, and strategy for 2026

**Last updated:** 2026-08-11

---

## 1. Executive summary

The 2025 and 2026 INFORMS Data Mining Society challenges are both **spatiotemporal power-outage forecasting problems driven by extreme weather**, but the evaluation setups are materially different.

The most important change is:

- **2025:** primarily a **future-time forecasting problem on the Michigan county panel**. Future weather during the official test period was *not* available, so competitors had to forecast or otherwise handle future weather uncertainty.
- **2026:** explicitly a **county-level held-out split**. Models train on **239 fully observed counties** and are applied to **63 held-out test counties**. The test counties provide only the first **72 hours of outage observations**, but **weather is provided for all 216 hours**, including the future prediction window.

This means the 2026 problem is still time series, but it is more accurately described as **spatiotemporal transfer forecasting**:

> learn outage-weather dynamics across known counties, then forecast future outage severity for different held-out counties.

The biggest strategic consequence is that a model that relies heavily on memorizing county-specific behavior may validate well under a normal temporal split but fail on the official 2026 test.

For 2026, validation should therefore imitate the official test structure:

1. Hold out entire counties.
2. For those counties, expose only the first 72 hours of outage information.
3. Provide all 216 hours of weather.
4. Predict the next 144 hours at the required 1h, 6h, 24h, and 48h horizons.
5. Never use future outage values.

---

# 2. 2025 INFORMS Data Mining Society Data Challenge

## Problem

The 2025 competition focused on forecasting short-term county-level power outages during extreme weather.

The official competition page describes the data as hourly county-level outage counts plus a large set of weather variables for **all 83 Michigan counties**.

### Main data structure

- Geography: **83 Michigan counties**
- Outage source: **PowerOutage.us / PowerOutage.com**
- Resolution: **hourly**
- Training period: approximately **April 1, 2023 through June 30, 2023**
- Target: **county-level outage count**
- Weather: roughly **109 variables**
- Forecast horizons:
  - 24 hours
  - 48 hours
- Evaluation metric: **RMSE**
- Official test labels: **hidden from competitors**
- Submission: prediction files based on provided 24h and 48h templates

The official page is slightly inconsistent in wording, describing the dataset once as having **108 weather features** and later as **109 weather variables**. The file itself should be treated as the source of truth.

## Critical deployment constraint in 2025

A very important condition was:

> **Future weather during the test period was not available.**

This means a complete 2025 solution could not simply use the actual future weather values to predict future outages.

This is why the first-place team's repository contains both:

1. a **weather forecasting model**, and
2. an **outage forecasting model**.

## How the test worked

Competitors did **not** receive the true test outage counts.

They received test templates containing the test timestamps/counties, produced predictions, and submitted those predictions.

The organizers held the true values and calculated the official leaderboard score.

So the workflow was:

```text
labeled training data
        ↓
local train/validation experiments
        ↓
fit final model
        ↓
official test inputs/template
        ↓
predictions
        ↓
submit CSV
        ↓
organizers calculate hidden-test score
```

## 2025 competition links

- Official 2025 challenge page:  
  https://sites.google.com/view/dmdaworkshop2025/data-challenge

- INFORMS 2025 announcement:  
  https://connect.informs.org/discussion/2025-informs-data-mining-society-data-challenge-1

---

# 3. 2025 winners

The official final results list four winning teams.

## 1st Place — Two-Stage Hurdle Models for Outage Forecasting

**Team**
- Tomas Kaljevic
- Shourya Bose
- Yu Zhang
- University of California, Santa Cruz

### Links

- **Winner's full public GitHub repository:**  
  https://github.com/shourya01/hurdle_model_outage_forecasting

- Yu Zhang announcement:  
  https://www.linkedin.com/posts/yu-zhang-8b276320_informs2025-datamining-datachallenge-activity-7388948951779119104-2Z-V

- INFORMS award page:  
  https://www.informs.org/Recognizing-Excellence/Community-Prizes/Data-Mining/Data-Mining-Society-Data-Challenge

### What their pipeline actually did

It is useful not to oversimplify this as only "a hurdle model."

The repository contains two major modeling stages.

### Stage A — forecast future weather

The public repository includes a **TimesNet** weather forecasting model.

The code uses approximately:

- 5-day / 120-hour weather lookback
- up to 48-hour forecast horizon
- multivariate weather inputs
- StandardScaler preprocessing
- a TimesNet architecture

This was necessary because **2025 did not provide future weather during the official test window**.

Repository file:

https://github.com/shourya01/hurdle_model_outage_forecasting/blob/master/train_weather.py

### Stage B — two-stage outage hurdle model

The outage model treats outage behavior as zero-inflated.

Conceptually:

```text
Stage 1:
P(outage occurs | history, weather, time, county)

Stage 2:
expected outage magnitude | outage occurs

Final prediction:
P(outage) × predicted intensity
```

The public code uses **LightGBM** components and includes probability calibration with isotonic regression.

The prediction code explicitly combines:

```text
expected = probability_of_outage × predicted_intensity
```

The model engineering includes features such as:

- dense outage lag features
- time since the last nonzero outage
- cyclic hour-of-day encoding
- cyclic day-of-week encoding
- lagged weather/hazard variables
- weather deltas
- rolling weather summaries
- outage × weather interaction terms
- recursive multi-step forecasting

Repository file:

https://github.com/shourya01/hurdle_model_outage_forecasting/blob/master/train_outage.py

### Why the idea is strong

Outage data are heavily zero-inflated:

```text
0
0
0
12
0
0
740
0
...
```

A single standard regression model has to simultaneously learn:

1. whether an outage happens at all, and
2. how severe it becomes.

The hurdle formulation separates those two behaviors.

### Important caution about "1st place"

The repository states that its competition leaderboard RMSE was approximately **189**.

This is important because the competition's **final awards were not determined solely by RMSE**.

The official page says:

- numerical results selected the finalists,
- final winners were chosen using the presentation and written methodology judged by a panel.

Therefore:

> **1st place does not necessarily mean this model had the lowest raw leaderboard RMSE.**

This matters when deciding which model to reproduce for 2026.

---

## 2nd Place — ARIMA–VAR with County Clustering

**Title:**  
*Spatiotemporal Forecasting of Power Outages under Extreme Weather Using ARIMA–VAR with County Clustering*

**Team**
- Setareh Kazemi Kheiri
- Sahand Hajifar

### Core model idea

Based on the official method title, the approach combined:

- county clustering
- ARIMA-style temporal modeling
- VAR / multivariate autoregressive modeling across related counties

The key conceptual idea is to exploit both:

- temporal dependence within a county, and
- cross-county relationships among geographically or behaviorally similar counties.

This is especially relevant for regional storms because outages in neighboring or meteorologically similar counties are not independent.

### Official result source

https://sites.google.com/view/dmdaworkshop2025/data-challenge

INFORMS award page for Sahand Hajifar:

https://www.informs.org/Recognizing-Excellence/Award-Recipients/Sahand-Hajifar

---

## 3rd Place — Geo-Temporal Deep Learning

**Title:**  
*Geo-Temporal Deep Learning Framework for Forecasting Weather-Driven Power Outages*

**Team**
- Sarathkumar Devaraj
- Bharani Nammi
- Ankit Rajan

### Core model idea

The approach explicitly modeled both:

- geographic / spatial structure, and
- temporal outage-weather dynamics.

The public material available from the official results identifies it as a **geo-temporal deep-learning framework**.

This is particularly relevant to 2026 because 2026 makes geographical transfer more central than 2025.

### Official result source

https://sites.google.com/view/dmdaworkshop2025/data-challenge

---

## 4th Place — SARIMAX

**Title:**  
*SARIMAX-Based Power Outage Prediction During Extreme Weather Events*

**Team**
- Haoran Ye
- Qiuzhuang Sun
- Yang Yang

### Links

Paper:

https://arxiv.org/abs/2511.01017

Code:

https://github.com/yhr-code/2025-INFORMS-DM-Challenge-Team12

### Model

Their solution used independent county-level **SARIMAX** models with substantial feature engineering.

The published workflow included:

- removing zero-variance features
- removing unknown / unusable variables
- correlation-based feature filtering
- weather feature selection
- temporal embeddings
- multi-scale outage lags
- lagged weather covariates
- standardization
- fallback to simpler ARIMA when SARIMAX fitting failed
- historical-mean fallback when needed

They report:

- RMSE: approximately **177.2**
- all-zero baseline: approximately **193.4**
- improvement: approximately **8.4%**

Their GitHub README explicitly notes that they experimented with more complex models including:

- Transformers
- LSTMs
- GNNs

but found that these **overfit** the limited/noisy data.

Their stated lesson was essentially:

> for a relatively small, noisy outage dataset, carefully engineered classical models can outperform much more complex deep networks.

---

# 4. A key lesson from the 2025 results

Do **not** interpret the final winner ranking as a clean model leaderboard.

The competition evaluated more than numerical RMSE.

This creates an interesting observation:

- First-place hurdle repository reports roughly **RMSE 189**
- Fourth-place SARIMAX paper reports roughly **RMSE 177.2**

Therefore, the competition award ranking reflects methodology/report/presentation quality as well as numerical performance.

For 2026, we should track two objectives separately:

### Objective A — hidden-test predictive performance

This determines how competitive the actual forecasts are.

### Objective B — methodological contribution

This matters because 2026 again explicitly evaluates:

- predictive accuracy
- rigor
- novelty
- clarity of methodology/report

A sophisticated but unstable model is not automatically better than a simple, defensible model.

---

# 5. 2026 INFORMS Data Mining Society Data Challenge

## Official title

**Forecasting Infrastructure Resilience Under Extreme Weather**

Official competition PDF:

https://connect.informs.org/HigherLogic/System/DownloadDocumentFile.ashx?DocumentFileKey=bdfbe387-c9ae-4772-b2fe-019fa45c3845

## Event

The dataset focuses on two successive wind-driven storm systems from **March 13–19, 2026** across:

- Indiana
- Ohio
- Pennsylvania
- West Virginia

A two-day pre-event period, **March 11–12**, is included to construct lagged and antecedent-condition features.

## Data sources

### Outages

**PowerOutage.com**

Hourly county-level outage observations.

### High-resolution weather

**NOAA URMA**

- approximately 2.5 km
- observation-corrected
- polygon-mean aggregated to counties

Variables listed in the public challenge PDF include:

- wind gust
- sustained wind speed
- wind direction
- 2m temperature
- 2m dew point

### Atmospheric context

**ERA5 reanalysis**

Variables include:

- surface pressure
- sea-level pressure
- boundary-layer height
- precipitation
- snowfall
- snow depth
- cloud cover
- shortwave radiation
- relative humidity
- soil moisture
- reference evapotranspiration

---

# 6. Exact 2026 train/test structure

## Total dataset

- **302 counties**
- **216 hourly observations per county**
- March 11–19, 2026

## Training set

- **239 counties**
- **51,624 rows**
- complete observations for all 216 hours
- complete outage variables
- complete weather variables
- complete OSI-related information

Calculation:

```text
239 counties × 216 hours = 51,624 rows
```

## Test set

- **63 held-out counties**
- **13,608 rows**
- weather available for all 216 hours
- outage variables available only for the first 72 hours
- future outage variables withheld
- OSI forecast targets withheld entirely

Calculation:

```text
63 counties × 216 hours = 13,608 rows
```

### Observed test period

```text
March 11–13
72 hours
outages available
weather available
```

### Forecast period

```text
March 14–19
144 hours
future outages hidden
weather available
```

This is the crucial point:

> **The 63 official test counties are held out from the 239 training counties.**

The official PDF explicitly states:

> "The train/test split is county-level (not time-based), stratified by state and severity level."

So these are not simply later timestamps from the same training counties.

---

# 7. 2026 target

2026 does **not** simply ask for raw outage counts.

The target is an **Outage Severity Index (OSI)**.

The dataset contains:

- raw customer outage counts
- outage fraction
- OSI state/flow components
- Pt
- Nt
- Dt
- Rt
- composite OSI
- lagged outage-related features

The exact OSI formula and full documentation are provided in the registered participant data package.

## Required horizons

Participants must predict:

```text
osi_target_t01h   → t + 1 hour
osi_target_t06h   → t + 6 hours
osi_target_t24h   → t + 24 hours
osi_target_t48h   → t + 48 hours
```

Submission rows cover:

```text
63 counties × 144 forecast hours
= 9,072 rows
```

Targets extending beyond the event window do not have ground truth and remain NaN according to the official instructions.

---

# 8. 2025 vs 2026 — direct comparison

| Dimension | 2025 | 2026 |
|---|---|---|
| Core problem | weather-driven outage forecasting | weather-driven outage severity forecasting |
| Temporal resolution | hourly | hourly |
| Geography | 83 Michigan counties | 302 counties across IN/OH/PA/WV |
| Training geography | Michigan panel | 239 counties |
| Test geography | primarily temporal holdout on Michigan panel | **63 held-out counties** |
| Main generalization problem | future time | **future time + unseen counties** |
| Target | outage count | **OSI severity** |
| Forecast horizons | 24h, 48h | **1h, 6h, 24h, 48h** |
| Future weather in official test | **not available** | **available for all 216 hours** |
| Test outage labels | hidden | hidden |
| Test outage history | historical information before horizon | first **72 hours** supplied |
| Weather dimensionality | ~109 weather variables | URMA + ERA5 curated variables |
| Main outage source | PowerOutage.us | PowerOutage.com |
| Classical models competitive? | yes | very likely worth testing |
| Spatial transfer importance | useful | **central** |
| Final judging | numerical finalists + report/presentation | predictive performance + methodological quality |

---

# 9. What is actually similar between 2025 and 2026?

The underlying scientific problem is very similar:

```text
past outages
+
weather
+
temporal structure
+
geographic relationships
        ↓
future outage behavior
```

Both datasets have:

- county-level panel structure
- hourly temporal resolution
- extreme weather
- sparse / irregular outage behavior
- sharp outage spikes
- lag dependence
- weather-outage interactions
- geographic correlation
- hidden official test targets

So many 2025 ideas remain valuable:

- outage lag features
- weather lag features
- hurdle / zero-inflated modeling
- LightGBM / XGBoost
- SARIMAX
- ARIMA / VAR
- county clustering
- spatial modeling
- temporal deep learning
- ensembles

---

# 10. What is fundamentally different in 2026?

## Difference 1 — unseen counties

This is probably the most important change.

2025 allowed much more benefit from learning county-specific history.

2026 deliberately asks the model to transfer learned relationships to **held-out counties**.

That means features such as:

```text
county_id
county-specific target mean
county-specific model
county-specific embedding
```

can become dangerous if the model relies on memorization.

A model should instead learn relationships that transfer:

```text
weather severity
outage state
recent outage trajectory
population/customer exposure
storm timing
neighboring conditions
state/geographic characteristics
```

---

## Difference 2 — future weather is already supplied

This removes a major modeling difficulty from 2025.

The 2025 winner had to forecast weather because future weather was unavailable.

For 2026:

> **Do not automatically reproduce their TimesNet weather-forecasting stage.**

The official dataset already gives weather across the entire 216-hour event window.

That model capacity can instead be spent on learning:

```text
future weather trajectory
        ↓
outage onset / escalation / restoration
```

This is a major advantage.

---

## Difference 3 — target changed from raw outage count to OSI

A hurdle model built directly for raw positive counts may not transfer perfectly.

We need to inspect:

- OSI distribution
- frequency of zero OSI
- meaning/range of Pt, Nt, Dt, Rt
- nonlinearities in the OSI formula
- whether modeling OSI components separately is better than modeling composite OSI directly

A strong experiment for 2026 could be:

```text
Model A:
predict OSI directly

Model B:
predict outage occurrence + OSI magnitude

Model C:
predict Pt, Nt, Dt, Rt separately
then reconstruct OSI using official formula

Model D:
ensemble A + B + C
```

This could be more appropriate than blindly copying the 2025 count hurdle model.

---

## Difference 4 — four horizons

2026 has:

- 1h
- 6h
- 24h
- 48h

The best feature importance and model may differ dramatically by horizon.

Example:

### 1 hour

Recent outage state may dominate.

```text
OSI(t)
outage(t)
outage(t-1)
recent slope
```

### 48 hours

Future weather trajectory and storm-phase information may matter much more.

```text
max future gust
cumulative precipitation
time to peak wind
storm-wave indicator
antecedent outage burden
restoration trajectory
```

Therefore we should test **horizon-specific models**, not assume one universal model is optimal.

---

## Difference 5 — two successive storms

The 2026 challenge intentionally contains a two-wave outage pattern.

This means naive persistence can fail badly.

A county may:

1. experience outages from storm 1,
2. partially recover,
3. get hit by storm 2,
4. experience a second outage escalation.

A model must capture:

- storm onset
- damage accumulation
- recovery/restoration
- second-event re-escalation

This suggests feature engineering around **storm phase** may be unusually valuable.

---

# 11. Validation strategy for 2026

This is one of the most important parts of the project.

## Do NOT use a normal random row split

Random splitting can leak:

- the same county into train and validation
- nearby timestamps
- future outage information
- event-specific temporal information

It would produce a misleadingly optimistic score.

## Recommended main validation

For each fold:

### Step 1 — hold out entire counties

Example:

```text
Training counties: 80%
Validation counties: 20%
```

Prefer stratification roughly by:

- state
- outage severity

because the official competition says its split is stratified by state and severity.

### Step 2 — simulate the official test mask

For each validation county:

```text
Hours 1–72:
outage + weather visible

Hours 73–216:
outage hidden
weather visible
```

### Step 3 — forecast exactly like the competition

Generate:

```text
t+1
t+6
t+24
t+48
```

for the 144-hour validation prediction window.

### Step 4 — calculate official metric(s)

The public PDF says the exact evaluation metrics/scoring details are communicated to registered participants.

Once available, use those metrics locally exactly.

## Secondary stress tests

### Leave-one-state-out

Train on three states and validate on the fourth.

This is harder than the official split but tests geographic transfer.

### High-severity-only validation

Measure performance specifically during the worst outage periods.

A model with excellent average error may still miss the economically important spikes.

### Storm-wave validation

Evaluate separately around:

- first storm
- restoration period
- second storm

This tells us whether the model actually learned the two-wave structure.

---

# 12. Models I would test for 2026

The goal should not be to pick a single favorite model immediately.

Build a **model ladder**.

---

## Model 0 — simple baselines

Required before anything sophisticated.

Examples:

```text
zero OSI
last observed OSI
decayed persistence
state/global historical mean
weather-only linear model
```

Every complex model should beat these robustly across county-held-out folds.

---

## Model 1 — LightGBM / XGBoost direct multi-horizon models

This should probably be one of the first serious baselines.

Build one model per horizon:

```text
LightGBM_1h
LightGBM_6h
LightGBM_24h
LightGBM_48h
```

Potential features:

### Outage / OSI state

- lag 1
- lag 3
- lag 6
- lag 12
- lag 24
- lag 48
- rolling max
- rolling mean
- rolling change
- time since last large outage
- restoration slope
- recent outage acceleration

### Future weather

Because future weather is supplied:

- gust at target time
- maximum gust next 6h
- maximum gust next 24h
- cumulative precipitation
- temperature/dewpoint changes
- pressure trend
- boundary-layer height
- storm intensity
- time until peak gust

### Interaction features

```text
current OSI × future gust
current outage fraction × max next-24h wind
recent restoration slope × next-storm intensity
```

This is a very strong tabular baseline.

---

## Model 2 — adapted hurdle model

Reproduce the 2025 winner concept, but adapt it for OSI.

Possible structure:

```text
Classifier:
P(OSI > threshold)

Regressor:
E[OSI | active/severe outage]

Final:
P(active) × severity
```

Try several definitions of "active":

```text
OSI > 0
OSI > small threshold
outage fraction > threshold
```

Because OSI may behave differently from raw outage counts, this must be tested rather than assumed.

---

## Model 3 — onset / severity / restoration decomposition

A more 2026-specific approach:

### Model A — outage onset/escalation

Probability that outage severity increases significantly.

### Model B — peak severity

Magnitude model.

### Model C — restoration/recovery

Rate at which severity declines after the storm.

This decomposition may align better with the physical process than a generic regression model.

---

## Model 4 — OSI-component model

If the competition's OSI formula permits clean reconstruction:

```text
predict Pt
predict Nt
predict Dt
predict Rt
        ↓
reconstruct OSI
```

Potential advantage:

The components may be statistically easier to predict and may have more interpretable dynamics than the composite index.

Compare against direct OSI modeling.

---

## Model 5 — SARIMAX / dynamic regression baseline

The 2025 fourth-place result shows that SARIMAX deserves respect.

But 2026 has a major challenge:

> test counties are held out.

So independent county-specific SARIMAX models trained only on each test county's short 72-hour history may be fragile.

More useful versions may include:

- pooled/global SARIMAX structure
- cluster-based parameter sharing
- state-level priors
- short test-county calibration using first 72 hours
- future weather as exogenous covariates

---

## Model 6 — ARIMA/VAR + county/weather clustering

Inspired by the 2025 second-place team.

Potential workflow:

```text
cluster counties by:
geography
weather profile
outage behavior
exposure/severity
        ↓
fit shared temporal / VAR structure
        ↓
map held-out test county to nearest cluster
        ↓
forecast
```

This could transfer better to unseen counties than independent county models.

---

## Model 7 — geo-temporal neural model

Possible architectures:

- Temporal Fusion Transformer
- TCN
- LSTM/GRU
- TimesNet
- patch-based time-series transformer
- graph neural network + temporal encoder

But 2025 provides a warning:

> complex models can overfit this type of small/noisy event dataset.

Therefore, neural models should only be trusted if they beat strong tree/statistical baselines across **county-held-out folds**, not merely random/time splits.

---

## Model 8 — ensemble

A final high-performing system could blend:

```text
LightGBM
+
hurdle model
+
SARIMAX/VAR
+
geo-temporal model
```

Different models may dominate different horizons.

Example:

```text
1h:
more weight on persistence + LightGBM

6h:
LightGBM + hurdle

24h:
future-weather model + tree ensemble

48h:
future-weather + spatiotemporal model
```

---

# 13. Biggest things to be critical about compared with 2025

## 1. Do not blindly copy the winning architecture

The 2025 first-place pipeline had to solve a problem that no longer exists in 2026:

> forecasting future weather.

2026 already supplies future weather.

So the 2025 weather TimesNet is not automatically useful.

---

## 2. County memorization is much more dangerous

Any validation that includes the same counties on both sides can dramatically exaggerate performance.

If the model uses a county categorical ID, embeddings, or target encoding, test whether it works when the entire county is unseen.

---

## 3. A county-specific time-series model does not naturally transfer

ARIMA/SARIMAX fit separately to each training county cannot simply be applied unchanged to a new test county.

2026 requires either:

- pooled models
- shared parameters
- clustering
- transfer learning
- meta-learning
- short adaptation using the first 72 test hours

---

## 4. OSI is not the same statistical target as outage count

Before choosing hurdle/Tweedie/Poisson/etc., inspect the actual target distribution.

Questions to answer immediately:

- Is OSI zero-inflated?
- Is it bounded?
- Is it continuous or quasi-discrete?
- How heavy is the tail?
- How persistent is it?
- How does it relate to raw outage count?
- Which OSI component drives the largest variance?
- Does the target behave differently by state?

---

## 5. 48-hour recursive prediction error can compound

If predicted outage values are recursively fed back as future lag features:

```text
prediction error at t+1
        ↓
becomes feature at t+2
        ↓
causes larger error
        ↓
...
```

Direct-horizon models may be safer:

```text
features at time t → OSI at t+1
features at time t → OSI at t+6
features at time t → OSI at t+24
features at time t → OSI at t+48
```

Both recursive and direct methods should be tested.

---

## 6. Optimize for severe periods, not only easy zeros

Extreme-event datasets can produce deceptively good average scores by predicting low values most of the time.

Track:

- overall metric
- high-OSI metric
- top 10% severity error
- peak timing error
- peak magnitude error
- recovery-period error

Even if the official metric only uses one summary score, these diagnostics will show why models fail.

---

# 14. Public / external data that may help

## PowerOutage.com

The competition's outage data are directly sourced from PowerOutage.com.

Main site:

https://poweroutage.us/

Data products:

https://poweroutage.us/use-our-data

PowerOutage.com aggregates live utility outage information and maintains historical data, but its full historical archive is not simply an unrestricted free bulk dataset.

---

## NOAA URMA

Official NOAA/NCEP URMA/RTMA page:

https://www.nco.ncep.noaa.gov/pmb/products/rtma/

Public cloud distribution information:

https://registry.opendata.aws/noaa-rtma/

URMA is particularly important because the 2026 competition explicitly says it uses county polygon-mean aggregated URMA fields.

This means it may be possible to reproduce weather preprocessing independently.

---

## ERA5

Copernicus Climate Data Store:

https://cds.climate.copernicus.eu/

ERA5 can provide additional historical weather for external pretraining and reconstruction.

---

## ORNL EAGLE-I outage data

A valuable free historical alternative is DOE/ORNL **EAGLE-I**.

2025 dataset:

https://impact.ornl.gov/en/datasets/eagle-i-power-outage-data-2025/

General dataset information:

https://impact.ornl.gov/en/datasets/eagle-i-power-outage-data-information/

The 2025 EAGLE-I release contains county-level outage information at **15-minute intervals**, including:

- FIPS
- county
- state
- customers without power
- timestamp

This could potentially be used to build a much larger historical training corpus, **if external data are allowed by the competition rules**.

Potential idea:

```text
EAGLE-I historical outages
+
NOAA historical weather
+
county metadata
        ↓
pretrain general outage-weather model
        ↓
fine-tune on official 2026 competition data
```

---

# 15. Possible test-county identification through weather

This is a research hypothesis, not something to use without checking the competition rules/NDA.

The official scope is:

- Indiana
- Ohio
- Pennsylvania
- West Virginia
- total 302 counties

The total number of counties in those four states also sums to 302, which strongly suggests the challenge scope may cover the complete four-state county set.

If:

1. the training data reveal the identities of the 239 training counties,
2. the test counties are anonymized,
3. all 302 counties truly represent the complete county set,

then the candidate identities for the 63 test counties could potentially be:

```text
all counties in IN/OH/PA/WV
-
239 known training counties
=
63 candidate test counties
```

Furthermore, the test file supplies all 216 hours of weather.

Because the official weather includes high-resolution county-aggregated URMA fields, one could theoretically reproduce:

```text
URMA public grid
        ↓
county polygon mean
        ↓
216-hour weather fingerprint for each candidate county
```

and compare each anonymous test weather series against public county weather.

With variables such as:

- gust
- sustained wind
- wind direction
- temperature
- dew point

across 216 hours, the resulting weather fingerprint could be highly distinctive.

### Validation of the reconstruction

Before trying to identify any test county:

1. reproduce URMA features for known training counties,
2. compare against the official supplied training weather,
3. measure matching error,
4. verify the aggregation method is replicated correctly.

### Competition-rule caution

The public 2026 PDF explicitly says:

- future outage values may not be used,
- violations cause disqualification,
- code review is part of evaluation.

It does not, in the public document, fully specify every possible external-data or deanonymization restriction.

Therefore:

> review the participant NDA, data-package rules, and external-data policy before using test-identity reconstruction in an official submission.

This idea is useful scientifically, but compliance comes first.

---

# 16. Suggested initial experiment roadmap

## Phase 1 — understand the data

Immediately calculate:

- target distributions
- percentage of zero OSI
- state distributions
- county severity distributions
- missing values
- correlations
- lag autocorrelation
- cross-county correlation
- storm timing
- relationship between OSI and raw outage count
- relationship among Pt/Nt/Dt/Rt
- weather lead-lag relationships

---

## Phase 2 — build correct validation before model tuning

Create a reusable validation function:

```text
split counties
        ↓
retain first 72h outage data for held-out counties
        ↓
mask next 144h outages
        ↓
retain all weather
        ↓
generate 1/6/24/48h forecasts
        ↓
score against hidden local ground truth
```

Do this before tuning advanced models.

A bad validation scheme can waste the entire competition.

---

## Phase 3 — baseline ladder

Run in this order:

1. zero / persistence / decay
2. linear / ElasticNet
3. LightGBM direct horizon models
4. XGBoost/CatBoost
5. adapted hurdle LightGBM
6. OSI-component modeling
7. pooled SARIMAX
8. clustered ARIMA/VAR
9. neural temporal models
10. spatial/graph models
11. ensembles

---

## Phase 4 — robust feature engineering

Prioritize features that make physical sense and transfer across counties.

Examples:

```text
current severity
recent outage slope
hours since outage onset
hours since peak outage
recent restoration rate
future peak wind
time until future peak wind
cumulative wind exposure
gust acceleration
pressure drop
storm-wave indicator
antecedent soil/weather conditions
```

These are more transferable than a raw county identifier.

---

## Phase 5 — external historical pretraining

If rules permit:

1. acquire EAGLE-I historical outages
2. join with NOAA/ERA5 historical weather
3. create county-hour panel
4. pretrain outage-response relationships
5. fine-tune/calibrate on 2026 official event

The biggest potential advantage is increasing the number of:

- storms
- counties
- outage peaks
- recovery cycles

seen during training.

---

# 17. My current recommended starting stack for 2026

If starting today, I would build these first:

### A. Strong tabular baseline

**LightGBM direct horizon models**

```text
1h
6h
24h
48h
```

with extensive lag/future-weather features.

### B. 2025 winner adaptation

**Two-stage hurdle LightGBM**

but target **OSI**, not blindly raw outage count.

### C. Component model

Predict OSI components separately and reconstruct OSI.

### D. Spatial-transfer model

Cluster counties based on non-target transferable characteristics, then share information within clusters.

### E. Classical model

Pooled/clustered SARIMAX or VAR as a low-variance benchmark.

### F. Final ensemble

Blend models by validation horizon.

---

# 18. The single most important validation rule

For 2026, the question is not:

> "Can I predict the future for a county I already trained on?"

The real question is:

> **"Can I learn a transferable weather → outage-severity relationship from some counties and apply it to a held-out county using only its first 72 hours of outage history?"**

Every modeling decision should be tested against that question.

---

# 19. Sources and useful links

## Official competition sources

### 2026

INFORMS 2026 Data Mining Society Data Challenge PDF:

https://connect.informs.org/HigherLogic/System/DownloadDocumentFile.ashx?DocumentFileKey=bdfbe387-c9ae-4772-b2fe-019fa45c3845

### 2025

Official challenge page:

https://sites.google.com/view/dmdaworkshop2025/data-challenge

INFORMS announcement:

https://connect.informs.org/discussion/2025-informs-data-mining-society-data-challenge-1

INFORMS Data Mining Society Data Challenge award page:

https://www.informs.org/Recognizing-Excellence/Community-Prizes/Data-Mining/Data-Mining-Society-Data-Challenge

---

## 2025 winning solutions

### 1st — Hurdle model

https://github.com/shourya01/hurdle_model_outage_forecasting

Winner announcement:

https://www.linkedin.com/posts/yu-zhang-8b276320_informs2025-datamining-datachallenge-activity-7388948951779119104-2Z-V

### 4th — SARIMAX

Paper:

https://arxiv.org/abs/2511.01017

Code:

https://github.com/yhr-code/2025-INFORMS-DM-Challenge-Team12

---

## External/public data

PowerOutage.com:

https://poweroutage.us/

PowerOutage.com data products:

https://poweroutage.us/use-our-data

NOAA URMA/RTMA:

https://www.nco.ncep.noaa.gov/pmb/products/rtma/

NOAA public cloud RTMA/URMA:

https://registry.opendata.aws/noaa-rtma/

Copernicus / ERA5:

https://cds.climate.copernicus.eu/

ORNL EAGLE-I 2025:

https://impact.ornl.gov/en/datasets/eagle-i-power-outage-data-2025/

EAGLE-I dataset information:

https://impact.ornl.gov/en/datasets/eagle-i-power-outage-data-information/

---

# 20. Bottom line

The strongest lesson from 2025 is **not** simply "use the hurdle model."

The better lesson is:

> simple, structured models that encode the physics/statistics of outages can outperform overly complex models when the data are sparse and noisy.

For 2026 we should preserve that principle, but adapt to the new evaluation design:

```text
2025:
forecast future outages
for known county panel
without future weather

2026:
forecast future OSI
for held-out counties
with complete future weather
and only 72h observed outage history
```

That makes **cross-county generalization, validation design, future-weather feature engineering, and OSI decomposition** the areas where we need to be much more critical than simply reproducing last year's winner.
