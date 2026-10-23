"""Leak-free datasets, chronological splits and evaluation metrics.

Two questions, two tasks:

* ``same_week`` (co-movement): can the *other* markets' week-t data explain the
  target's week-t return? The target market contributes no columns at all.
* ``next_week`` (forecasting): can data up to week t predict the target's
  week t+1 return? Only past information is used, so the target's own history
  is allowed.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import columns_for
from .stats import diebold_mariano

TEST_START = "2025-01-01"  # train on 2019-2024, test on 2025 onwards


def make_dataset(panel: pd.DataFrame, target: str, inputs, task: str, window: int = 4):
    """Return X (samples, window, features), y, and the date of each y."""
    if task == "same_week" and target in inputs:
        raise ValueError("same_week inputs must exclude the target market")
    feats = panel[columns_for(inputs, panel)].to_numpy()
    returns = panel[f"{target}_Return"].to_numpy()
    shift = 1 if task == "next_week" else 0

    X, y, dates = [], [], []
    for t in range(window - 1, len(panel) - shift):
        X.append(feats[t - window + 1: t + 1])
        y.append(returns[t + shift])
        dates.append(panel.index[t + shift])
    return np.array(X), np.array(y), pd.DatetimeIndex(dates)


def chronological_split(X, y, dates, test_start=TEST_START):
    """Train on weeks before ``test_start``, test on the weeks from then on.

    The scaler is fit on training rows only, so no test statistics leak in.
    """
    cut = int(np.searchsorted(dates, pd.Timestamp(test_start)))
    scaler = StandardScaler().fit(X[:cut].reshape(-1, X.shape[-1]))

    def scale(a):
        return scaler.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape)

    return scale(X[:cut]), y[:cut], scale(X[cut:]), y[cut:], dates[cut:]


def metrics(y_true, y_pred, train_mean):
    """RMSE in % points, out-of-sample R² vs the training mean, hit rate, and the
    one-sided Diebold-Mariano p-value for beating the training mean."""
    err = y_true - y_pred
    bench = np.full(len(y_true), train_mean)
    r2 = 1 - np.sum(err ** 2) / np.sum((y_true - bench) ** 2)
    hit = np.mean(np.sign(y_pred) == np.sign(y_true)) if np.any(y_pred) else np.nan
    dm_p = diebold_mariano(y_true, y_pred, bench)[1] if np.any(y_pred != bench) else np.nan
    return {"RMSE": np.sqrt(np.mean(err ** 2)), "R2_OOS": r2, "Direction_Acc": hit, "DM_p": dm_p}


def run(panel, target, inputs, task, models, window=4):
    X, y, dates = make_dataset(panel, target, inputs, task, window)
    Xtr, ytr, Xte, yte, test_dates = chronological_split(X, y, dates)
    rows, preds = [], {}
    for model in models:
        p = model.fit(Xtr, ytr).predict(Xte)
        preds[model.name] = p
        rows.append({"Task": task, "Target": target, "Inputs": " + ".join(inputs),
                     "Model": model.name, **metrics(yte, p, ytr.mean()),
                     "N_train": len(ytr), "N_test": len(yte)})
    return rows, pd.DataFrame({"Actual": yte, **preds}, index=test_dates)
