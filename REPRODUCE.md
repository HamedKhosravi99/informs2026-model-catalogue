# Reproducing the submission

Everything needed to rebuild `submissions/submission_freeze_v2.csv` from a
clean checkout. Written for a reviewer with no prior context.

---

## 1. What you need

| item | note |
|---|---|
| Python **3.9** | the pinned environment; 3.12 is untested |
| `requirements.txt` | exact pinned versions, all wheels |
| `DM_Train.csv`, `DM_Test.csv`, `sample_submission.csv` | **NDA-protected, not in the repo** — place the three files in the repo root |
| `data_static/` | external covariates, **committed** — no download step |

```bash
pip install -r requirements.txt
python3 make_submission.py
```

**macOS OpenMP caveat.** `xgboost` and `lightgbm` wheels need `libomp.dylib`,
which is not bundled and is absent from a stock macOS without Homebrew or conda.
If `import xgboost` raises a dlopen error, install `llvm-openmp` (conda-forge) or
`libomp` (Homebrew). On Linux the system `libgomp` suffices. Full instructions
are in the comment block at the bottom of `requirements.txt`.

## 2. No hidden inputs

The production path reads **only** the three competition CSVs and the committed
`data_static/` tree. Verified by static sweep over every module the submission
path imports (`make_submission.py`, `src/data.py`, `src/ideas{,3,4,8,10}.py`,
`src/dl.py`): no read of `oof/`, no read of `data_static/urma_cache/`, both of
which are git-ignored regenerable caches. A reviewer therefore never has to
regenerate a cache to rebuild the submission.

`oof/` holds cached out-of-fold predictions used only by the *analysis* scripts
(`compare_models.py`, `evaluate.py`). Regenerate any of them with
`python3 run_all_ideas.py --only <ID>`.

## 3. External data provenance

Every file in `data_static/` is derived from a public source, built by a
committed script, and contains **no outage information** — only static
geography, land cover, terrain, utility infrastructure and gridded weather.

| file | source | builder |
|---|---|---|
| `county_static.csv` | Census TIGER (area, centroid), Census population, USDA ERS RUCC codes | *no builder committed* — assembled early, before the build scripts were factored out |
| `county_landcover.csv`, `county_landcover_raw.csv` | NLCD land cover | `src/build_county_landcover.py` |
| `county_terrain.csv` | elevation / terrain ruggedness | `fetch_terrain.py` |
| `county_fia_forest.csv` | USDA FIA forest inventory | `src/build_county_fia.py` |
| `eia861/county_eia861_2024.csv` | EIA-861 utility data | `src/build_eia861_county.py` |
| `eia861/county_transmission_hifld.csv` | HIFLD transmission lines | `src/build_hifld_translines.py` |
| `subcounty_weather.csv` | Open-Meteo historical archive API, sampled at sub-county points | `fetch_subcounty_weather.py` |
| `urma_subcounty.csv` | NOAA URMA 2.5 km analysis | `fetch_urma_gust.py` |

The builder scripts are included for audit. They are **not** on the submission
path — their outputs are committed, so `make_submission.py` never re-downloads.
`county_static.csv` is the one file without a committed builder; its six columns
(`land_sqmi, lat, lon, population, rucc, pop_density`) are standard public
reference values and can be checked directly against the Census/ERS sources.

## 4. Causality guarantee (how to check it yourself)

No outage-derived value from a timestamp after the prediction origin (hour 71)
reaches any feature. Two independent checks:

1. **Assertion.** `make_submission.py` asserts the test file contains no
   post-h71 outage data before doing anything else.
2. **Corruption invariance.** Overwrite every post-origin outage value in the
   input with garbage and re-run: predictions are unchanged, because masking
   happens *before* feature construction. This is the strongest form of the
   claim — it does not depend on reading the feature code.

The validation harness applies the identical mask to held-out counties before
features are built, so CV and deployment run the same code path.

## 5. Expected output

9,072 rows; identifier columns byte-identical to `sample_submission.csv`;
NaN counts 63 / 378 / 1512 / 3024 for t01h / t06h / t24h / t48h respectively
(targets extending past the end of the window); all predictions ≥ 0.
`python3 verify_submission.py` checks all of this.

County-holdout CV of the built configuration (`submission_freeze_v2.csv`, freeze
spec v2: 11 members combined by amplitude x shape, 50/50 with A3x25): **RMSE
0.008358** (nested; 0.008354 with the production constants applied to every
fold), per-horizon 0.00994 / 0.00907 / 0.00757 / 0.00683; choosing this
configuration on four folds and applying it to the fifth gives 0.008378, the
honest post-search figure. The previous build (`submission_freeze.csv`, 0.008468)
is reproducible with `--spec freeze`. MAE 0.00251 is a diagnostic only -- `evaluation_procedure.pdf` scores
RMSE alone.

The previous file, `submission_ensemble_final.csv` (RMSE 0.008543), is still
reproducible with `python3 make_submission.py --spec shipped`. Its MAE was long
quoted as 0.00296, which was the `t01h` row rather than the mean; the correct
value is 0.00256 (REPORT 5ap).

## 6. Determinism

All seeds are fixed. Model fitting is CPU-only (`torch==1.12.0`, no CUDA
non-determinism). Re-running `make_submission.py` on the same machine and
environment reproduces the same file.

**Cross-machine byte-identity is not claimed.** BLAS/OpenMP reduction order
differs across builds, so a different machine may produce predictions that
differ in the far decimals. The CV score is stable to that; the SHA256 is not.
Compare scores, not hashes, across machines.

## 7. Runtime

`make_submission.py` trains every member on all 239 counties and caches each
member's test predictions under `submissions/members/`, so a member-level change
reruns only that member. The freeze spec v2 took **10.1 hours** on a laptop CPU
(sharing the CPU with another run); v1 took 11.2 hours alone, dominated by two 25-seed arms: `x4`
(s35 mixup) and `a3x25` (25 seeds x 4 snapshots = 100 network fits). The older
`--spec shipped` path is roughly 1 hour. No GPU is used or required.
