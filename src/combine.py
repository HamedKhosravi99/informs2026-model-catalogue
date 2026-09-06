"""Combination rules shared by the analysis harness (canonical.py) and the production builder."""
import numpy as np
import pandas as pd


def ampshape(cols, idx):
    """Amplitude x shape: mean of the members' county totals times the mean of their
    normalised within-county shapes. Parameter-free. Equals the plain mean when members
    agree on timing; preserves peak height when they do not (W21)."""
    M = np.column_stack([np.asarray(c) for c in cols])
    f = idx.get_level_values(0).to_numpy()
    df = pd.DataFrame(M); df["f"] = f
    tot = df.groupby("f").transform("sum").to_numpy()
    shape = np.where(tot > 0, M / np.maximum(tot, 1e-12), 0.0)
    return pd.Series(tot.mean(1) * shape.mean(1), index=idx)
