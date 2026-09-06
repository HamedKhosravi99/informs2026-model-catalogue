"""A6 - non-arithmetic ensemble pooling, evaluated strictly nested.

The project's combination stage is closed to *fitted weights* (REPORT 5g): every
scheme with per-member freedom overfits, monotonically with parameter count. The
two rank-free operators that were tried - per-cell median and trimmed mean - tie
production exactly and fit nothing.

This tests the one family in between: pooling operators with a *single* shared
shape parameter, fitted nested (4 folds -> held-out 5th), which is the same
protocol that certified the quiet-tail scale.

  power mean   p_q = ( mean_i x_i^q )^(1/q)          q > 0
  JS shrink    p    = mu * mu^2 / (mu^2 + lam*sd^2)  lam >= 0

The arithmetic mean is q = 1 and lam = 0, so both families strictly contain the
incumbent: if nothing is there, the fitted parameter returns to the incumbent
value and the score is unchanged. That makes this a safe test.

Every variant gets the identical nested quiet-tail procedure applied afterwards
(least-squares scale on the fold-training counties, half-shrunk toward 1), so no
variant is advantaged by a differently-tuned tail.

Reads oof/ only. Writes nothing. Does not touch the shipped ensemble.
"""
import numpy as np
import pandas as pd

from src.data import load_train, osi_trajectory
from src.validate import make_folds, score_trajectory
from src.ideas4 import ENSEMBLE_FINAL_W, TAIL_SPLIT_HOUR

MEMBERS = list(ENSEMBLE_FINAL_W)
EPS = 1e-9

train = load_train()
truth = osi_trajectory(train)

m = None
for k in MEMBERS:
    p = pd.read_csv(f"oof/{k}.csv").rename(columns={"osi_pred": k})
    m = p if m is None else m.merge(p, on=["fipsCode", "hour"])
m["truth"] = truth.stack().reindex(
    pd.MultiIndex.from_frame(m[["fipsCode", "hour"]])).to_numpy()
m = m.dropna(subset=["truth"]).reset_index(drop=True)

X = np.clip(m[MEMBERS].to_numpy(), 0, None)   # members, all non-negative
y = m["truth"].to_numpy()
hour = m["hour"].to_numpy()
tail = hour >= TAIL_SPLIT_HOUR

folds = list(make_folds(train))
fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}
fold = m["fipsCode"].map(fold_of).to_numpy()
K = len(folds)


def score(p):
    d = m[["fipsCode", "hour"]].copy()
    d["osi_pred"] = np.clip(p, 0, None)
    t = score_trajectory(d, truth)
    return t.loc["mean", "rmse"], t.loc["mean", "mae"], t


def tail_scale(p, sel):
    """The adopted procedure, unchanged: LS scale on `sel`, half-shrunk toward 1."""
    a = sel & tail
    if a.sum() < 10 or (p[a] @ p[a]) <= 0:
        return 1.0
    return 0.5 * (p[a] @ y[a]) / (p[a] @ p[a]) + 0.5


# ---------------------------------------------------------------- operators
def power_mean(Xs, q):
    if abs(q - 1.0) < 1e-12:
        return Xs.mean(1)
    if abs(q) < 1e-6:                                   # geometric limit
        return np.exp(np.log(Xs + EPS).mean(1)) - EPS
    return np.maximum(np.mean(np.power(Xs + EPS, q), axis=1), 0.0) ** (1.0 / q) - EPS


def js_shrink(Xs, lam):
    mu = Xs.mean(1)
    if lam <= 0:
        return mu
    sd = Xs.std(1)
    return mu * (mu ** 2) / (mu ** 2 + lam * sd ** 2 + EPS)


Q_GRID = np.round(np.arange(0.4, 2.01, 0.05), 2)
L_GRID = np.round(np.concatenate([[0.0], np.arange(0.02, 1.01, 0.02)]), 2)


def rmse_on(p, sel):
    return float(np.sqrt(np.mean((p[sel] - y[sel]) ** 2)))


def _tail_fit(p_sub, y_sub, tail_sub):
    """LS tail scale on a subset, half-shrunk toward 1 (the adopted procedure)."""
    a = tail_sub
    if a.sum() < 10 or (p_sub[a] @ p_sub[a]) <= 0:
        return 1.0
    return 0.5 * (p_sub[a] @ y_sub[a]) / (p_sub[a] @ p_sub[a]) + 0.5


def _apply(op, Xs, g, y_sub, tail_sub, s=None):
    p = op(Xs, g)
    if s is None:
        s = _tail_fit(p, y_sub, tail_sub)
    return np.where(tail_sub, p * s, p), s


def nested(op, grid):
    """Fit the shape parameter and the tail scale on 4 folds, apply to the 5th."""
    out = np.empty(len(y))
    chosen = []
    for k in range(K):
        tr, te = fold != k, fold == k
        best, best_v = grid[0], np.inf
        for g in grid:
            p, _ = _apply(op, X[tr], g, y[tr], tail[tr])
            v = float(np.sqrt(np.mean((p - y[tr]) ** 2)))
            if v < best_v:
                best_v, best = v, g
        chosen.append(best)
        _, s = _apply(op, X[tr], best, y[tr], tail[tr])
        out[te], _ = _apply(op, X[te], best, y[te], tail[te], s=s)
    return out, chosen


def fixed(op, g):
    """Same operator at a fixed parameter, with only the tail scale nested."""
    out = np.empty(len(y))
    for k in range(K):
        tr, te = fold != k, fold == k
        _, s = _apply(op, X[tr], g, y[tr], tail[tr])
        out[te], _ = _apply(op, X[te], g, y[te], tail[te], s=s)
    return out


print(f"members: {MEMBERS}")
print(f"cells: {len(y)}   folds: {K}\n")

rows = []
base = fixed(power_mean, 1.0)
r, a, tbl = score(base)
rows.append(("arithmetic mean (incumbent, q=1)", r, a, "-"))
base_tbl = tbl

p, ch = nested(power_mean, Q_GRID)
r, a, _ = score(p)
rows.append(("power mean, q nested", r, a, f"q={ch}"))

p, ch = nested(js_shrink, L_GRID)
r, a, _ = score(p)
rows.append(("JS disagreement shrink, lam nested", r, a, f"lam={ch}"))

# non-nested (in-sample) power mean, to measure the optimism of this family
best_q, best_v = 1.0, np.inf
for g in Q_GRID:
    v = rmse_on(fixed(power_mean, g), np.ones(len(y), bool))
    if v < best_v:
        best_v, best_q = v, g
r, a, _ = score(fixed(power_mean, best_q))
rows.append((f"power mean, q fit IN-SAMPLE (q={best_q})", r, a, "optimism check"))

print(f"{'scheme':40s} {'RMSE':>9s} {'MAE':>9s}   note")
print("-" * 84)
for n, r, a, note in rows:
    d = r - rows[0][1]
    print(f"{n:40s} {r:9.5f} {a:9.5f}   {note}  ({d:+.5f})")

print("\nincumbent per-horizon:")
print(base_tbl.round(5).to_string())
