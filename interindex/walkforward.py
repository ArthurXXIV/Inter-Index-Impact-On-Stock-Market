"""Walk-forward evaluation: refit every January on all data before it.

This mimics live use. Each year is forecast by a model that has only seen
earlier years, and the yearly forecasts are stitched into one out-of-sample record.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .stats import diebold_mariano

TEST_YEARS = range(2021, 2027)


def sequences(design: pd.DataFrame, sources, y_col: str, window: int = 5):
    """X[t] = rows t-window+1..t of [source features known before open t, own return t-1]."""
    step = design[list(sources)].copy()
    step["own_prev"] = design["cc"].shift(1)
    frame = step.join(design[y_col].rename("y")).dropna()
    F, y = frame.drop(columns="y").to_numpy(), frame["y"].to_numpy()
    X = np.stack([F[t - window + 1: t + 1] for t in range(window - 1, len(F))])
    return X, y[window - 1:], frame.index[window - 1:]


def walk_forward(X, y, dates, models, years=TEST_YEARS):
    """Return a DataFrame of out-of-sample predictions, one column per model."""
    chunks = []
    for year in years:
        train = dates < pd.Timestamp(f"{year}-01-01")
        test = dates.year == year
        if not test.any():
            continue
        scaler = StandardScaler().fit(X[train].reshape(-1, X.shape[-1]))

        def scale(a):
            return scaler.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape)

        Xtr, Xte = scale(X[train]), scale(X[test])
        out = {"Actual": y[test], "Mean": np.full(test.sum(), y[train].mean())}
        for m in models:
            out[m.name] = m.fit(Xtr, y[train]).predict(Xte)
        chunks.append(pd.DataFrame(out, index=dates[test]))
    return pd.concat(chunks)


def score(pred: pd.DataFrame, model: str, periods_per_year: int = 252) -> dict:
    y, p, bench = pred["Actual"].to_numpy(), pred[model].to_numpy(), pred["Mean"].to_numpy()
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - bench) ** 2)
    dm, dm_p = diebold_mariano(y, p, bench)
    pnl = np.sign(p) * y  # trade the predicted direction, before costs
    return {"R2_OOS": r2, "Direction_Acc": np.mean(np.sign(p) == np.sign(y)),
            "DM_stat": dm, "DM_p": dm_p,
            "Sharpe_gross": pnl.mean() / pnl.std() * np.sqrt(periods_per_year),
            "N_test": len(y)}
