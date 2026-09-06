"""Optuna TPE over the shipped tabular member f2's XGBoost hyperparameters.

Same nested true-metric protocol as the sequence-model searches: trials are
scored by the official metric (trajectory -> four columns -> per-horizon RMSE
mean via score_trajectory) on an inner 3-fold county-holdout of fold-0's
outer-training counties, built by the same seed-42 stratified machinery.
Warm-started AT f2's hand-set defaults. The row frames are built once per
inner fold; each trial pays only for the XGB fit + predict.

Run:  python3 optuna_tune_f2.py [n_trials]
Writes results/optuna_tune_f2.csv; winner is evaluated as RT4 on the
untouched outer 5-fold harness (control: f2 = 0.00930).
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 48
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT = {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6,
           "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
           "reg_lambda": 1.0, "reg_alpha": 1e-4}   # 1e-4 = log-space floor (~0)


def build_fold_frames():
    """f2's exact row construction for each inner fold, built once."""
    from src.data import load_train, mask_after_freeze
    from src.validate import make_folds
    from src.features import feature_cols
    from src.ideas6 import _rows
    from src.ideas10 import add_indep

    train = load_train()
    tr_f, _ = next(iter(make_folds(train)))
    inner_train = train[train.fipsCode.isin(tr_f)]
    inner_folds = list(make_folds(inner_train, n_splits=3))

    drop = ["hrs_since_g35", "hrs_until_g35", "gfut6", "veer",
            "pre_peak_logP", "pre_gust_max", "pre_frag_gap"]
    frames = []
    for tr_fips, va_fips in inner_folds:
        trd = inner_train[inner_train.fipsCode.isin(tr_fips)]
        vad = mask_after_freeze(inner_train[inner_train.fipsCode.isin(va_fips)])
        vad = vad.drop(columns=[c for c in ("severity_tier", "peak_pct",
                                            "peak_customers", "time_to_restore_h",
                                            "split") if c in vad.columns])
        tr, y = _rows(trd, statics=True, sub=True)
        va = _rows(vad, statics=True, sub=True, val=True)
        tr = add_indep(tr, mask_after_freeze(trd))
        va = add_indep(va, vad)
        tr = tr.drop(columns=[c for c in drop if c in tr.columns])
        va = va.drop(columns=[c for c in drop if c in va.columns])
        cols = feature_cols(tr)
        keep = np.isfinite(y)
        frames.append((tr.loc[keep, cols], np.sqrt(y[keep]), va, cols))
    return train, frames


def main():
    import optuna
    import pandas as pd
    import xgboost as xgb
    from src.data import osi_trajectory
    from src.ideas import _out
    from src.validate import score_trajectory

    train, frames = build_fold_frames()
    truth = osi_trajectory(train)
    print(f"frames built: {len(frames)} inner folds, "
          f"{frames[0][0].shape[1]} features", flush=True)

    def objective(trial):
        cfg = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 40),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        }
        scores = []
        for i, (Xtr, ysq, va, cols) in enumerate(frames):
            mdl = xgb.XGBRegressor(**cfg, random_state=42, n_jobs=4,
                                   tree_method="hist")
            mdl.fit(Xtr, ysq)
            pred = _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)
            scores.append(float(score_trajectory(pred, truth).loc["mean", "rmse"]))
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                pruner=pruner)
    study.enqueue_trial(DEFAULT)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    rows = []
    for t in study.trials:
        r = dict(t.params)
        r["state"] = str(t.state).split(".")[-1]
        r["metric"] = t.value if t.value is not None else np.nan
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("metric")
    df.to_csv(os.path.join(ROOT, "results", "optuna_tune_f2.csv"), index=False)
    print("\n=== completed trials by true inner metric ===")
    print(df[df.state == "COMPLETE"].head(8).to_string(index=False))
    print(f"\nBEST: {study.best_params}\n  inner metric {study.best_value:.5f}")
    print(f"DEFAULT (warm-start trial 0) inner metric: {study.trials[0].value:.5f}")


if __name__ == "__main__":
    main()
