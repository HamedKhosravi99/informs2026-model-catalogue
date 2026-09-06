"""Decoupled two-stage learner on f2's exact rows: a GBM classifier for P(y>0)
times a GBM regressor trained ONLY on positive cells (E[y | y>0, x]).  Every
current member learns E[y|x] with the zeros inside the fit; this is the
decomposition E[y|x] = P(y>0|x) * E[y|y>0,x] with each factor fitted separately.

Writes, per fold and concatenated:
  oof/f2_ctrl.csv    plain regressor on all rows (should reproduce f2 ~0.00930; export sanity check)
  oof/f2_cls.csv     P(y>0)                          (osi_pred holds the probability)
  oof/f2_pos.csv     E[y|y>0,x] (squared back from the sqrt scale)
  oof/f2_hurdle.csv  P * E[y|y>0,x]
Hyperparameters are f2's own (ideas10) so the learner structure is the only variable.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

ROWS = Path(sys.argv[1]); ROOT = Path(__file__).resolve().parent
HP = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
          min_child_weight=5, reg_lambda=1.0, random_state=42, n_jobs=2, tree_method="hist")
out = {k: [] for k in ("f2_ctrl", "f2_cls", "f2_pos", "f2_hurdle")}
for k in range(5):
    tr = pd.read_csv(ROWS / f"fold{k}_train.csv"); va = pd.read_csv(ROWS / f"fold{k}_val.csv")
    cols = [c for c in tr.columns if c != "_sqrt_y"]
    X, s, Xv = tr[cols], tr["_sqrt_y"].to_numpy(), va[cols]
    pos = s > 0
    ctrl = np.clip(xgb.XGBRegressor(**HP).fit(X, s).predict(Xv), 0, None) ** 2
    P = xgb.XGBClassifier(**HP, eval_metric="logloss").fit(X, pos.astype(int)).predict_proba(Xv)[:, 1]
    yp = np.clip(xgb.XGBRegressor(**HP).fit(X[pos], s[pos]).predict(Xv), 0, None) ** 2
    for name, v in (("f2_ctrl", ctrl), ("f2_cls", P), ("f2_pos", yp), ("f2_hurdle", P * yp)):
        out[name].append(pd.DataFrame({"fipsCode": va.fipsCode, "hour": va.hour, "osi_pred": v}))
    print(f"fold {k}: positives {pos.mean():.2%} of train rows; mean P on val {P.mean():.3f}", flush=True)
for name, parts in out.items():
    pd.concat(parts).to_csv(ROOT / "oof" / f"{name}.csv", index=False)
print("wrote", ", ".join(f"oof/{n}.csv" for n in out))
