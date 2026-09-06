"""Permutation importance for the direct GBM on one held-out county fold."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data import load_train, mask_after_freeze, osi_trajectory
from src.validate import make_folds
from src.features import build_rows, feature_cols
from src.models import fit_gbm


def main():
    train = load_train()
    tr_fips, va_fips = next(make_folds(train))
    model, cols = fit_gbm(train[train["fipsCode"].isin(tr_fips)])

    va_rows = build_rows(mask_after_freeze(train[train["fipsCode"].isin(va_fips)]))
    truth = osi_trajectory(train).stack()
    y = truth.reindex(pd.MultiIndex.from_frame(va_rows[["fipsCode", "hour"]])).to_numpy()

    imp = permutation_importance(model, va_rows[cols], y, n_repeats=3,
                                 random_state=42, scoring="neg_root_mean_squared_error")
    tab = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
    print("Permutation importance (RMSE increase when shuffled), top 18:")
    print((tab.head(18) * 1e4).round(2).to_string(), "\n(x 1e-4 RMSE units)")


if __name__ == "__main__":
    main()
