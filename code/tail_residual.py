"""S65: nested ADDITIVE tail-residual model.

Start from the ensemble's out-of-fold prediction, restrict to high-risk rows, and fit
a heavily regularised tree to the ADDITIVE residual r = y - yhat. Additive because
every multiplicative correction tried in this project amplified error on this
heavy-tailed target (per-county calibration 0.00937->0.01099, county-normalised target
0.02164, binned-classification reconstruction 0.03353).

Strictly nested: the residual model for fold k is fitted only on counties outside
fold k.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.data import load_train, osi_trajectory, mask_after_freeze, PRED_HOURS
from src.validate import make_folds, score_trajectory
from src.features import feature_cols
from src.ideas import _rows_and_y

BASE = "oof/zC.csv"          # the shipped 7-member ensemble, out-of-fold


def main():
    train = load_train()
    truth = osi_trajectory(train)
    base = pd.read_csv(BASE).rename(columns={"osi_pred": "base"})

    rows, y = _rows_and_y(train, static=True, physics=True)
    rows = rows.merge(base, on=["fipsCode", "hour"], how="left")
    rows["y"] = y
    rows = rows.dropna(subset=["y", "base"]).reset_index(drop=True)
    rows["resid"] = rows["y"] - rows["base"]

    folds = list(make_folds(train))
    fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}
    rows["fold"] = rows["fipsCode"].map(fold_of)

    # high-risk rows only: where the tail lives
    risk = ((rows["base"] > 0.01)
            | (rows["gust_max24"] > 45)
            | (rows["obs_osi_max"] > 0.05))
    print(f"high-risk rows: {risk.sum()} of {len(rows)} ({100*risk.mean():.1f}%), "
          f"carrying {100*(rows.loc[risk,'resid']**2).sum()/(rows['resid']**2).sum():.1f}% "
          f"of the squared residual")

    cols = [c for c in feature_cols(rows) if c not in ("y", "resid", "fold")]
    out = rows[["fipsCode", "hour"]].copy()
    out["osi_pred"] = rows["base"]
    for shrink in (1.0, 0.5, 0.25):
        pred = rows["base"].to_numpy().copy()
        for k in range(len(folds)):
            tr = (rows["fold"] != k).to_numpy() & risk.to_numpy()
            te = (rows["fold"] == k).to_numpy() & risk.to_numpy()
            if tr.sum() < 200 or te.sum() == 0:
                continue
            mdl = make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesRegressor(n_estimators=300, min_samples_leaf=40,
                                    max_features=0.4, n_jobs=4, random_state=42))
            mdl.fit(rows.loc[tr, cols], rows.loc[tr, "resid"])
            pred[te] = pred[te] + shrink * mdl.predict(rows.loc[te, cols])
        d = out[["fipsCode", "hour"]].copy()
        d["osi_pred"] = np.clip(pred, 0, None)
        t = score_trajectory(d, truth)
        print(f"  residual correction, shrink {shrink:4.2f}: "
              f"RMSE {t.loc['mean','rmse']:.5f}  MAE {t.loc['mean','mae']:.5f}")
        d.to_csv(f"oof/tailres_{str(shrink).replace('.','')}.csv", index=False)

    t0 = score_trajectory(out, truth)
    print(f"  baseline (no correction)          : RMSE {t0.loc['mean','rmse']:.5f}"
          f"  MAE {t0.loc['mean','mae']:.5f}")


if __name__ == "__main__":
    main()
