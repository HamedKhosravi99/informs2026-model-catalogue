# tools/ — measurements that need an environment other than the pinned one

`chronos2_zeroshot.py` scores Amazon's Chronos-2 (120M, Oct 2025, covariate-aware)
zero-shot on the 239 training counties with our known-future weather as
covariates. It needs torch >= 2.x and the `chronos-forecasting` package, so it
runs in a separate virtualenv (python 3.12), never in the project's pinned
environment. It writes `oof/chronos2.csv` (quantile-average mean estimate) and
`oof/chronos2_med.csv` (median) in the standard layout; grade them with
`python3 eval_sota.py chronos2 chronos2_med` from the main environment.

It is a **measurement for the report, not a submission candidate**: a
pretrained checkpoint is external data of a kind the "self-contained code on
the provided files" clause does not admit, and the pinned environment cannot
load it.

```bash
python3.12 -m venv fmenv && fmenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu "chronos-forecasting>=2.0" pandas
fmenv/bin/python tools/chronos2_zeroshot.py
```
