"""I19: learned stacking over cached out-of-fold member predictions.

Nested by fold: the meta-model for fold k is fit only on OOF rows from the
other folds, then applied to fold k. Compared against the simple weighted
average currently in production.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.data import load_train, osi_trajectory
from src.validate import make_folds, score_trajectory
from src.features import county_static
from src.data import mask_after_freeze
from src.models import make_gbm

MEMBERS = ["gru", "gru_comp", "transformer", "i07", "i11", "perstate_hurdle", "mlp", "i16"]


def load_members():
    m = None
    for n in MEMBERS:
        p = pd.read_csv(f"oof/{n}.csv").rename(columns={"osi_pred": n})
        m = p if m is None else m.merge(p, on=["fipsCode", "hour"])
    return m


def main():
    train = load_train()
    truth = osi_trajectory(train)
    m = load_members()
    m["truth"] = truth.stack().reindex(
        pd.MultiIndex.from_frame(m[["fipsCode", "hour"]])).to_numpy()

    # context features for the meta-model (all legal: hour, state, observed window)
    st = county_static(mask_after_freeze(train))
    m["hrs_since_freeze"] = m["hour"] - 71.0
    for c in ["obs_osi_h71", "obs_osi_max", "restored_frac", "state_OH", "state_IN", "state_PA"]:
        m[c] = m["fipsCode"].map(st[c])
    ctx = ["hrs_since_freeze", "obs_osi_h71", "obs_osi_max", "restored_frac",
           "state_OH", "state_IN", "state_PA"]

    fold_of = {}
    for k, (_, va) in enumerate(make_folds(train)):
        for f in va:
            fold_of[f] = k
    m["fold"] = m["fipsCode"].map(fold_of)

    out_ridge, out_gbm = m[["fipsCode", "hour"]].copy(), m[["fipsCode", "hour"]].copy()
    out_ridge["osi_pred"] = np.nan
    out_gbm["osi_pred"] = np.nan
    for k in sorted(m["fold"].unique()):
        tr, te = m[m["fold"] != k], m[m["fold"] == k]
        r = Ridge(alpha=1.0, positive=True).fit(tr[MEMBERS], tr["truth"])
        out_ridge.loc[te.index, "osi_pred"] = np.clip(r.predict(te[MEMBERS]), 0, None)
        g = make_gbm(max_iter=300).fit(tr[MEMBERS + ctx], tr["truth"])
        out_gbm.loc[te.index, "osi_pred"] = np.clip(g.predict(te[MEMBERS + ctx]), 0, None)

    equal = m[["fipsCode", "hour"]].copy()
    equal["osi_pred"] = m[MEMBERS].mean(1)
    cur = m[["fipsCode", "hour"]].copy()
    cur["osi_pred"] = (0.25 * m.gru + 0.25 * m.gru_comp + 0.2 * m.transformer
                       + 0.15 * m.perstate_hurdle + 0.15 * m.mlp)

    for name, d in [("current 5-way (production)", cur),
                    ("equal-weight 8 members", equal),
                    ("I19a stacking: Ridge (nested)", out_ridge),
                    ("I19b stacking: GBM + context (nested)", out_gbm)]:
        t = score_trajectory(d, truth)
        print(f"{name:40s} RMSE {t.loc['mean','rmse']:.5f}  MAE {t.loc['mean','mae']:.5f}  "
              f"per-h {[round(t.loc[h,'rmse'],5) for h in ['t01h','t06h','t24h','t48h']]}",
              flush=True)


if __name__ == "__main__":
    main()
