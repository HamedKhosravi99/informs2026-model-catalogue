"""Optuna TPE over the ensemble's combination stage, under the 5g nested
protocol.

Arms:
  A  full: 8 member weights (simplex-normalized) + tail split hour + tail
     scale                                            (10 parameters)
  B  minimal: equal weights fixed; tail split + scale only  (2 parameters)

Protocol (identical to 5g): for each outer fold k, Optuna fits the parameters
on the OOF cells of the OTHER FOUR folds (objective = the true metric: mean of
per-horizon RMSEs over those counties' scored cells), the best parameters are
applied to fold k, and the five held-out blocks pool into one honest score.
The non-nested number (fit on all cells, score on all cells) is printed
alongside as the optimism reference. Incumbent: equal weights + (168, 0.667)
= 0.00862.

Reads oof/ only. Writes results/optuna_tune_ensemble.csv. Touches nothing.
"""
import os

import numpy as np
import pandas as pd

from src.data import FREEZE_H, HORIZONS, load_train, osi_trajectory
from src.validate import make_folds

MEMBERS = ["s62", "s51", "s39", "s40", "i11", "f2", "s35", "p1b"]
N_TRIALS = 250
ROOT = os.path.dirname(os.path.abspath(__file__))

train = load_train()
truth = osi_trajectory(train)
folds = list(make_folds(train))
fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}

m = None
for k in MEMBERS:
    p = pd.read_csv(f"oof/{k}.csv").rename(columns={"osi_pred": k})
    m = p if m is None else m.merge(p, on=["fipsCode", "hour"])
m["truth"] = truth.stack().reindex(
    pd.MultiIndex.from_frame(m[["fipsCode", "hour"]])).to_numpy()
m = m.dropna(subset=["truth"]).reset_index(drop=True)
X = m[MEMBERS].to_numpy()
y = m["truth"].to_numpy()
hour = m["hour"].to_numpy()
fold = m["fipsCode"].map(fold_of).to_numpy()
col_masks = [(hour >= FREEZE_H + 1 + h) for h in HORIZONS.values()]


def metric(pred, sel):
    return float(np.mean([np.sqrt(np.mean((pred[sel & c] - y[sel & c]) ** 2))
                          for c in col_masks]))


def combine(params, arm):
    if arm == "A":
        w = np.array([params[f"w{i}"] for i in range(len(MEMBERS))])
        w = w / w.sum()
    else:
        w = np.full(len(MEMBERS), 1 / len(MEMBERS))
    pred = X @ w
    tail = hour >= params["tail_split"]
    pred = np.where(tail, pred * params["tail_scale"], pred)
    return np.clip(pred, 0, None)


def run_arm(arm):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def make_objective(sel):
        def obj(trial):
            params = {}
            if arm == "A":
                for i in range(len(MEMBERS)):
                    params[f"w{i}"] = trial.suggest_float(f"w{i}", 0.0, 1.0)
                if sum(params[f"w{i}"] for i in range(len(MEMBERS))) < 1e-9:
                    return 1.0
            params["tail_split"] = trial.suggest_int("tail_split", 120, 208, step=8)
            params["tail_scale"] = trial.suggest_float("tail_scale", 0.3, 1.0)
            return metric(combine(params, arm), sel)
        return obj

    nested = np.empty(len(y))
    fold_params = []
    for k in range(len(folds)):
        tr_sel, te_sel = fold != k, fold == k
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        if arm == "A":
            study.enqueue_trial({**{f"w{i}": 1.0 for i in range(len(MEMBERS))},
                                 "tail_split": 168, "tail_scale": 0.667})
        else:
            study.enqueue_trial({"tail_split": 168, "tail_scale": 0.667})
        study.optimize(make_objective(tr_sel), n_trials=N_TRIALS,
                       show_progress_bar=False)
        bp = dict(study.best_params)
        fold_params.append(bp)
        nested[te_sel] = combine(bp, arm)[te_sel]

    # non-nested: fit on ALL cells, score on ALL cells (optimism reference)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    if arm == "A":
        study.enqueue_trial({**{f"w{i}": 1.0 for i in range(len(MEMBERS))},
                             "tail_split": 168, "tail_scale": 0.667})
    else:
        study.enqueue_trial({"tail_split": 168, "tail_scale": 0.667})
    study.optimize(make_objective(np.ones(len(y), bool)), n_trials=N_TRIALS,
                   show_progress_bar=False)
    non_nested = metric(combine(dict(study.best_params), arm),
                        np.ones(len(y), bool))
    all_sel = np.ones(len(y), bool)
    return metric(nested, all_sel), non_nested, fold_params, dict(study.best_params)


inc = combine({"tail_split": 168, "tail_scale": 0.667}, "B")
print(f"incumbent (equal weights, tail 168/0.667): "
      f"{metric(inc, np.ones(len(y), bool)):.5f}\n")

rows = []
for arm, label in (("B", "tail-only (2 params)"), ("A", "weights+tail (10 params)")):
    nested_score, non_nested, fps, bp_all = run_arm(arm)
    gap = nested_score - non_nested
    print(f"arm {arm} {label}:")
    print(f"  nested (honest): {nested_score:.5f}   non-nested: {non_nested:.5f}"
          f"   optimism gap: {gap:+.5f}")
    if arm == "A":
        w = np.array([bp_all[f"w{i}"] for i in range(len(MEMBERS))])
        w = w / w.sum()
        print("  non-nested weights: "
              + ", ".join(f"{k}={v:.2f}" for k, v in zip(MEMBERS, w)))
    print(f"  per-fold tail params: "
          + " ".join(f"({p['tail_split']},{p['tail_scale']:.2f})" for p in fps))
    rows.append({"arm": label, "nested": nested_score, "non_nested": non_nested})
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results",
                                       "optuna_tune_ensemble.csv"), index=False)
