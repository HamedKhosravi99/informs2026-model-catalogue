"""Ray Tune ASHA over the S62 bi-encoder config, optimizing the ACTUAL
competition metric under the project's own CV machinery.

Upgrades over ray_tune_bienc.py (whose inner-loss proxy selected a config
that lost the outer evaluation):
  - trial objective = the true metric: predict the full 143-hour trajectory
    for held-out counties, map into the four horizon columns and average the
    per-horizon RMSEs with the SAME `score_trajectory` code the harness uses;
  - the counties held out inside each trial come from the SAME fold machinery
    (`make_folds`: StratifiedKFold on state x severity_tier, seed 42), run as
    an INNER 3-fold county-holdout of fold-0's outer-training counties only.

Nesting: outer fold-0's held-out counties (and folds 1-4 entirely) never
enter the search. The winner is then evaluated as RT2 with the untouched
outer 5-fold harness. ASHA prunes after each inner fold (max_t=3).

Run:  python3 ray_tune_metric.py [n_trials]
Writes results/ray_tune_metric.csv; prints the best config and the DEFAULT
config's score under the identical inner protocol for reference.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 36
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT = {"hidden": 48, "dropout": 0.15, "lr": 2e-3, "wd": 1e-4,
           "batch": 32, "alpha": 0.5}


def build_inner_folds():
    """Inner 3-fold county-holdout of fold-0's outer-training counties,
    using the project's exact fold machinery (seed 42, state x tier)."""
    from src.data import load_train
    from src.validate import make_folds
    train = load_train()
    tr_f, _ = next(iter(make_folds(train)))          # outer fold-0 split
    inner_train = train[train.fipsCode.isin(tr_f)]
    inner_folds = list(make_folds(inner_train, n_splits=3))
    return train, inner_train, inner_folds


def prep_fold(inner_train, tr_fips, va_fips):
    from src.data import mask_after_freeze
    trd = inner_train[inner_train.fipsCode.isin(tr_fips)]
    vad = mask_after_freeze(inner_train[inner_train.fipsCode.isin(va_fips)])
    vad = vad.drop(columns=[c for c in ("severity_tier", "peak_pct",
                                        "peak_customers", "time_to_restore_h",
                                        "split") if c in vad.columns])
    return trd, vad


def config_metric_one_fold(cfg, trd, vad, truth):
    """Train with cfg (single seed 0), predict the trajectory, score the
    official metric on the inner held-out counties."""
    import torch
    from src.ideas import _prep, _dl_out
    from src.ideas4 import scoring_weights
    from src.ideas8 import _fit_rt
    from src.validate import score_trajectory
    torch.set_num_threads(2)
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(trd, vad)
    m = _fit_rt(Xe, Xd, Xs, Y, 0, scoring_weights(), cfg)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    with torch.no_grad():
        pred = _dl_out(vf, (torch.clamp(m(*tv), min=0) ** 2).numpy())
    return float(score_trajectory(pred, truth).loc["mean", "rmse"])


def trial(config, payload):
    from ray import tune as rt
    train, inner_train, inner_folds, truth = payload
    scores = []
    for tr_fips, va_fips in inner_folds:
        trd, vad = prep_fold(inner_train, tr_fips, va_fips)
        scores.append(config_metric_one_fold(config, trd, vad, truth))
        rt.report({"metric": float(np.mean(scores)), "folds_done": len(scores)})


def main():
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from src.data import osi_trajectory

    train, inner_train, inner_folds = build_inner_folds()
    truth = osi_trajectory(train)
    payload = (train, inner_train, inner_folds, truth)
    print(f"inner protocol: 3-fold county-holdout of {inner_train.fipsCode.nunique()} "
          f"outer-training counties (seed-42 stratified machinery)", flush=True)

    ray.init(num_cpus=8, include_dashboard=False, logging_level="ERROR",
             ignore_reinit_error=True)
    space = {
        "hidden": tune.choice([32, 48, 64, 96]),
        "dropout": tune.uniform(0.05, 0.35),
        "lr": tune.loguniform(5e-4, 6e-3),
        "wd": tune.loguniform(1e-6, 1e-3),
        "batch": tune.choice([16, 32, 64]),
        "alpha": tune.uniform(0.3, 0.7),
    }
    sched = ASHAScheduler(metric="metric", mode="min", max_t=3,
                          grace_period=1, reduction_factor=3)
    tuner = tune.Tuner(
        tune.with_parameters(trial, payload=payload),
        param_space=space,
        tune_config=tune.TuneConfig(num_samples=N_TRIALS, scheduler=sched,
                                    max_concurrent_trials=4),
        run_config=tune.RunConfig(verbose=1),
    )
    res = tuner.fit()
    df = res.get_dataframe()
    keep = [c for c in df.columns if c.startswith("config/")
            or c in ("metric", "folds_done")]
    out = df[keep].sort_values("metric")
    out.to_csv(os.path.join(ROOT, "results", "ray_tune_metric.csv"), index=False)
    full = out[out.folds_done == 3]
    print("\n=== configs that survived all 3 inner folds, by TRUE metric ===")
    print(full.head(8).to_string(index=False))
    best = res.get_best_result(metric="metric", mode="min",
                               filter_nan_and_inf=True)
    print("\nBEST CONFIG:", best.config, f"  inner metric {best.metrics['metric']:.5f}")

    # reference: the DEFAULT config under the identical inner protocol
    scores = []
    for tr_fips, va_fips in inner_folds:
        trd, vad = prep_fold(inner_train, tr_fips, va_fips)
        scores.append(config_metric_one_fold(DEFAULT, trd, vad, truth))
    print(f"DEFAULT config, same inner protocol: {np.mean(scores):.5f} "
          f"(folds: {['%.5f' % s for s in scores]})")


if __name__ == "__main__":
    main()
