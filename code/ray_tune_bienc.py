"""Ray Tune (ASHA) hyperparameter optimization of the S62 bi-encoder trunk.

Every sequence member inherits hand-set hyperparameters (hidden 48, dropout
0.15, lr 2e-3, wd 1e-4, batch 32) chosen for the first GRU and never jointly
tuned. This searches capacity + optimization + the dual-loss mix under the
S62 objective.

Selection is NESTED BY CONSTRUCTION: the objective trains on fold-0's
OUTER-TRAINING counties and reports the weighted OSI loss on the standard
inner 1/8 county split of those SAME training counties - the outer held-out
counties of every fold are never seen by the search. The winning config is
then evaluated with the untouched 5-fold county-holdout harness (T-series).

Run:  python3 ray_tune_bienc.py [n_trials]
Writes results/ray_tune_bienc.csv (all trials) and prints the best config.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")

import numpy as np
import torch
import torch.nn as nn

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_fold0():
    from src.data import load_train
    from src.validate import make_folds
    train = load_train()
    tr_f, va_f = next(iter(make_folds(train)))
    return train[train.fipsCode.isin(tr_f)]


def make_tensors(trd):
    """S62's exact tensor prep, on fold-0 outer-training counties only."""
    from src.data import mask_after_freeze
    from src.ideas import _prep
    from src.ideas4 import scoring_weights
    masked = mask_after_freeze(trd)
    (Xe, Xd, Xs, Y), _ = _prep(trd, masked)     # val tensors unused
    return Xe, Xd, Xs, Y, scoring_weights()


def trial(config, data):
    from src.ideas3 import BiEncSeq2Seq
    Xe, Xd, Xs, Y, w = data
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]

    model = BiEncSeq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1],
                         hidden=config["hidden"], dropout=config["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["wd"])
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(w, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))
    a = config["alpha"]

    def loss_fn(out, i):
        raw = torch.clamp(out, min=0) ** 2
        den = (msk[i] * tw).sum()
        return (a * ((raw - t[3][i]) ** 2 * tw * msk[i]).sum() / den
                + (1 - a) * ((out - sq[i]) ** 2 * tw * msk[i]).sum() / den)

    from ray import tune as rt
    best, bad = np.inf, 0
    n_batches = max(1, len(tr_i) // config["batch"])
    for ep in range(300):
        model.train()
        for b in np.array_split(rng.permutation(tr_i), n_batches):
            opt.zero_grad()
            loss_fn(model(t[0][b], t[1][b], t[2][b]), b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.clamp(model(t[0][va_i], t[1][va_i], t[2][va_i]), min=0) ** 2
            v = float((((p - t[3][va_i]) ** 2) * tw * msk[va_i]).sum()
                      / (msk[va_i] * tw).sum())
        best = min(best, v)
        rt.report({"inner_loss": v, "best_inner": best})
        bad = bad + 1 if v > best - 1e-6 else 0
        if bad >= 30:
            break


def main():
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    trd = load_fold0()
    data = make_tensors(trd)
    print(f"fold-0 outer-training counties: {data[0].shape[0]}", flush=True)

    ray.init(num_cpus=8, include_dashboard=False,
             logging_level="ERROR", ignore_reinit_error=True)
    space = {
        "hidden": tune.choice([32, 48, 64, 96]),
        "dropout": tune.uniform(0.05, 0.35),
        "lr": tune.loguniform(5e-4, 6e-3),
        "wd": tune.loguniform(1e-6, 1e-3),
        "batch": tune.choice([16, 32, 64]),
        "alpha": tune.uniform(0.3, 0.7),      # dual-loss raw-term weight
    }
    sched = ASHAScheduler(metric="best_inner", mode="min",
                          max_t=300, grace_period=25, reduction_factor=3)
    tuner = tune.Tuner(
        tune.with_parameters(trial, data=data),
        param_space=space,
        tune_config=tune.TuneConfig(num_samples=N_TRIALS, scheduler=sched,
                                    max_concurrent_trials=4),
        run_config=tune.RunConfig(verbose=1),
    )
    res = tuner.fit()
    df = res.get_dataframe()
    keep = [c for c in df.columns if c.startswith("config/") or
            c in ("best_inner", "inner_loss", "training_iteration")]
    out = df[keep].sort_values("best_inner")
    out.to_csv(os.path.join(ROOT, "results", "ray_tune_bienc.csv"), index=False)
    print("\n=== top 8 configs by inner loss (never saw any outer fold) ===")
    print(out.head(8).to_string(index=False))
    best = res.get_best_result(metric="best_inner", mode="min")
    print("\nBEST CONFIG:", best.config, "  inner", f"{best.metrics['best_inner']:.5f}")
    print("reference: S62 defaults are hidden 48, dropout 0.15, lr 2e-3, "
          "wd 1e-4, batch 32, alpha 0.5")


if __name__ == "__main__":
    main()
