"""One clean answer to "surely another weight beats equal": every intuitive
weight family on the CURRENT 8-member pool, hindsight vs honest.

Families:
  fixed pre-specified round-weight configs (legitimate: nothing fitted)
  performance-proportional weights (inverse-MSE^k, temperature nested)
  the Optuna-fitted optimum (from optuna_tune_ensemble.py's protocol)
plus the fold-to-fold instability of the "optimal" weights themselves.

All rows get the identical nested tail procedure. Reads oof/ only.
"""
import numpy as np
import pandas as pd

from src.data import FREEZE_H, HORIZONS, load_train, osi_trajectory
from src.validate import make_folds

MEMBERS = ["s62", "s51", "s39", "s40", "i11", "f2", "s35", "p1b"]
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
ALL = np.ones(len(y), bool)


def metric(pred, sel=ALL):
    return float(np.mean([np.sqrt(np.mean((pred[sel & c] - y[sel & c]) ** 2))
                          for c in col_masks]))


def with_tail(pred, fit_sel, apply_sel):
    """The adopted nested tail procedure: LS on fit_sel's tail, half-shrunk."""
    out = pred.copy()
    a = fit_sel & (hour >= 168)
    s = 0.5 * (pred[a] @ y[a]) / (pred[a] @ pred[a]) + 0.5
    t = apply_sel & (hour >= 168)
    out[t] = pred[t] * s
    return out


def nested_eval(weight_fn):
    """weight_fn(fit_sel) -> w. Weights + tail fit on 4 folds, applied to 5th."""
    out = np.empty(len(y))
    ws = []
    for k in range(len(folds)):
        tr_sel, te_sel = fold != k, fold == k
        w = weight_fn(tr_sel)
        ws.append(w)
        pred = np.clip(X @ w, 0, None)
        out[te_sel] = with_tail(pred, tr_sel, te_sel)[te_sel]
    return metric(out), ws


def fixed_eval(w):
    w = np.asarray(w, float)
    w = w / w.sum()
    return nested_eval(lambda sel: w)[0]


def member_mse(sel):
    return np.array([np.mean((X[sel, i] - y[sel]) ** 2)
                     for i in range(len(MEMBERS))])


print(f"members: {MEMBERS}")
print(f"{'scheme':46s} {'honest':>8s}  note")
print("-" * 88)

rows = [
    ("EQUAL (incumbent)", [1] * 8, ""),
    ("double the 2 best singles (s62, f2)", [2, 1, 1, 1, 1, 2, 1, 1], ""),
    ("triple s62", [3, 1, 1, 1, 1, 1, 1, 1], ""),
    ("halve the worst standalone (p1b)", [2, 2, 2, 2, 2, 2, 2, 1], ""),
    ("drop p1b entirely", [1, 1, 1, 1, 1, 1, 1, 0], "= 7-member set"),
    ("drop the 2 worst standalone (i11, p1b)", [1, 1, 1, 1, 0, 1, 1, 0], ""),
    ("sequence-heavy (tabular+stockflow halved)", [2, 2, 2, 2, 2, 1, 2, 1], ""),
]
for name, w, note in rows:
    print(f"{name:46s} {fixed_eval(w):8.5f}  {note}")

for k_pow in (0.5, 1.0, 2.0):
    v, ws = nested_eval(lambda sel, kp=k_pow: (1 / member_mse(sel) ** kp)
                        / (1 / member_mse(sel) ** kp).sum())
    print(f"{'inverse-MSE^%.1f (performance weights)' % k_pow:46s} {v:8.5f}  nested")

# the Optuna-style free-weight optimum, nested, + weight instability
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def opt_w(sel):
        def obj(trial):
            w = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(8)])
            if w.sum() < 1e-9:
                return 1.0
            pred = np.clip(X @ (w / w.sum()), 0, None)
            return metric(with_tail(pred, sel, sel), sel)
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
        st.enqueue_trial({f"w{i}": 1.0 for i in range(8)})
        st.optimize(obj, n_trials=200, show_progress_bar=False)
        w = np.array([st.best_params[f"w{i}"] for i in range(8)])
        return w / w.sum()

    v, ws = nested_eval(opt_w)
    print(f"{'FREE weights, Optuna-fitted (nested honest)':46s} {v:8.5f}  "
          "the 0.00853 hindsight version, honestly evaluated")
    print("\nthe 'optimal' weights themselves, per fold (rows = folds):")
    hdr = "  ".join(f"{k:>5s}" for k in MEMBERS)
    print(f"       {hdr}")
    for i, w in enumerate(ws):
        print(f"fold {i}: " + "  ".join(f"{x:5.2f}" for x in w))
    sd = np.std(np.array(ws), 0)
    print("sd:     " + "  ".join(f"{x:5.2f}" for x in sd)
          + "   <- the instability that IS the overfitting")
except ImportError:
    pass
