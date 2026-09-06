"""Optuna (TPE + MedianPruner) over the S62 bi-encoder config - same nested
true-metric protocol as ray_tune_metric.py, different search strategy.

TPE is model-based (samples where the fitted density of good configs says
improvement is likely) vs ASHA's random-sample-and-prune, and the study is
WARM-STARTED at the hand-set default config, so the search begins from the
known-good point. Objective and nesting are identical to RT2: per trial,
train on inner-training counties and score the TRUE competition metric
(trajectory -> four columns -> per-horizon RMSE mean via score_trajectory) on
an inner 3-fold county-holdout of fold-0's outer-training counties, built by
the same seed-42 stratified machinery. Outer folds never influence selection.
Pruning: MedianPruner after each inner fold.

Run:  python3 optuna_tune_metric.py [n_trials]
Writes results/optuna_tune_metric.csv; winner is evaluated as RT3.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 36
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT = {"hidden": 48, "dropout": 0.15, "lr": 2e-3, "wd": 1e-4,
           "batch": 32, "alpha": 0.5}


def main():
    import optuna
    import torch
    from ray_tune_metric import build_inner_folds, prep_fold, config_metric_one_fold
    from src.data import osi_trajectory

    torch.set_num_threads(3)
    train, inner_train, inner_folds = build_inner_folds()
    truth = osi_trajectory(train)
    print(f"inner protocol: 3-fold county-holdout of "
          f"{inner_train.fipsCode.nunique()} outer-training counties", flush=True)

    prepped = [prep_fold(inner_train, tr_f, va_f) for tr_f, va_f in inner_folds]

    def objective(trial):
        cfg = {
            "hidden": trial.suggest_categorical("hidden", [32, 48, 64, 96]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.35),
            "lr": trial.suggest_float("lr", 5e-4, 6e-3, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            "batch": trial.suggest_categorical("batch", [16, 32, 64]),
            "alpha": trial.suggest_float("alpha", 0.3, 0.7),
        }
        scores = []
        for i, (trd, vad) in enumerate(prepped):
            scores.append(config_metric_one_fold(cfg, trd, vad, truth))
            trial.report(float(np.mean(scores)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=8)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                pruner=pruner)
    study.enqueue_trial(DEFAULT)                 # warm start AT the defaults
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=False)

    rows = []
    for t in study.trials:
        r = dict(t.params)
        r["state"] = str(t.state).split(".")[-1]
        r["metric"] = t.value if t.value is not None else np.nan
        r["folds_done"] = len(t.intermediate_values)
        rows.append(r)
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("metric")
    df.to_csv(os.path.join(ROOT, "results", "optuna_tune_metric.csv"), index=False)
    done = df[df.state == "COMPLETE"]
    print("\n=== completed trials by true inner metric ===")
    print(done.head(8).to_string(index=False))
    print(f"\nBEST: {study.best_params}  inner metric {study.best_value:.5f}")
    d0 = df[(df.hidden == 48) & (df.batch == 32)].head(1)
    print(f"DEFAULT (warm-start trial) inner metric: "
          f"{study.trials[0].value if study.trials[0].value else float('nan'):.5f}")


if __name__ == "__main__":
    main()
