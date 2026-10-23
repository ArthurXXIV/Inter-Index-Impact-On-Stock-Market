"""Significance tests: Diebold-Mariano, robust Granger causality, FDR control."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def diebold_mariano(y, pred_model, pred_bench, max_lag=None):
    """One-sided DM test that the model's squared error is lower than the benchmark's.

    Uses a Newey-West variance so autocorrelated loss differentials are handled.
    Returns (statistic, p-value); a positive statistic favours the model.
    """
    d = (np.asarray(y) - pred_bench) ** 2 - (np.asarray(y) - pred_model) ** 2
    n = len(d)
    max_lag = max_lag if max_lag is not None else int(np.floor(n ** (1 / 3)))
    d_c = d - d.mean()
    var = d_c @ d_c / n
    for k in range(1, max_lag + 1):
        var += 2 * (1 - k / (max_lag + 1)) * (d_c[k:] @ d_c[:-k]) / n
    stat = d.mean() / np.sqrt(var / n)
    return stat, 1 - stats.norm.cdf(stat)


def lag_matrix(s: pd.Series, lags, prefix):
    return pd.concat({f"{prefix}_L{k}": s.shift(k) for k in lags}, axis=1)


def granger(target: pd.Series, source: pd.Series, p: int = 5):
    """Does `source` help explain `target` beyond the target's own past?

    `source` must already be information-aligned (its lag 0 is known before the
    target session opens). Wald test with HC3 errors, robust to volatility
    clustering. Returns (F statistic, p-value, R² gain).
    """
    own = lag_matrix(target, range(1, p + 1), "own")
    src = lag_matrix(source, range(0, p), "src")
    df = pd.concat([target.rename("y"), own, src], axis=1).dropna()
    X_r = sm.add_constant(df[own.columns])
    X_f = sm.add_constant(df[list(own.columns) + list(src.columns)])
    full = sm.OLS(df["y"], X_f).fit(cov_type="HC3")
    restricted = sm.OLS(df["y"], X_r).fit()
    R = np.zeros((p, X_f.shape[1]))
    for i, c in enumerate(src.columns):
        R[i, X_f.columns.get_loc(c)] = 1
    test = full.f_test(R)
    return float(test.fvalue), float(test.pvalue), full.rsquared - restricted.rsquared


def benjamini_hochberg(pvals) -> np.ndarray:
    """FDR-adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out
