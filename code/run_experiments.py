"""Run the model ladder under the county-holdout protocol and print scores.

Pass --dl to also run the deep models (MLP, GRU, component GRU, 4-way ensemble).
"""
import sys

import numpy as np
import pandas as pd

from src.data import load_train, osi_trajectory
from src.validate import run_cv
from src.models import LADDER

if "--dl" in sys.argv:
    from src.dl import (mlp_forecaster, gru_forecaster,
                        gru_component_forecaster, ensemble_forecaster)
    LADDER = {**LADDER,
              "MLP (same features)": mlp_forecaster,
              "GRU seq2seq": gru_forecaster,
              "GRU component-structured": gru_component_forecaster,
              "ensemble 4-way (primary)": ensemble_forecaster}

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v:.5f}")


def main():
    train = load_train()
    truth = osi_trajectory(train)
    tiers = train.groupby("fipsCode").agg(state=("stateAbbr", "first"),
                                          tier=("severity_tier", "first"))
    summary, pooled_preds = {}, {}
    for name, forecaster in LADDER.items():
        table, pooled = run_cv(forecaster, train)
        summary[name] = table["rmse"]
        pooled_preds[name] = pooled
        print(f"\n=== {name} ===")
        print(table.drop(columns="n").to_string())

    print("\n" + "=" * 68)
    print("RMSE summary (rows: model, cols: horizon)")
    print(pd.DataFrame(summary).T.to_string())

    # diagnostics: pooled squared error by state and severity tier
    diag = ("ensemble 4-way (primary)" if "ensemble 4-way (primary)" in pooled_preds
            else "GBM direct trajectory")
    pred = pooled_preds[diag].copy()
    t_lookup = truth.stack()
    idx = pd.MultiIndex.from_frame(pred[["fipsCode", "hour"]])
    pred["truth"] = t_lookup.reindex(idx).to_numpy()
    pred["se"] = (pred["osi_pred"] - pred["truth"]) ** 2
    pred = pred.join(tiers, on="fipsCode")
    print(f"\n{diag} trajectory RMSE by state:")
    print(pred.groupby("state")["se"].mean().pow(0.5).to_string())
    print(f"\n{diag} trajectory RMSE by severity tier:")
    print(pred.groupby("tier")["se"].mean().pow(0.5).to_string())


if __name__ == "__main__":
    main()
