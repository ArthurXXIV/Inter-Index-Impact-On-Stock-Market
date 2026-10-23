"""Daily prices and information-time alignment across time zones.

Markets keep different holidays, so everything is placed on the union of all
trading dates. For a target market on date d, a source feature is the source's
log return accumulated between the information cutoff of the target's previous
session and the cutoff of this one, so each source move is counted once even
across holidays.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from .data import DATA_DIR
from .markets import MARKETS, info_lag

START, END = "2019-04-01", "2026-10-22"  # END is exclusive; last complete session 2026-10-21
DAILY_DIR = DATA_DIR / "daily"


def load_daily(refresh: bool = False) -> dict[str, pd.DataFrame]:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, m in MARKETS.items():
        path = DAILY_DIR / f"{name.lower()}.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = yf.download(m.ticker, start=START, end=END, auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open", "High", "Low", "Close"]].dropna()
            df.to_csv(path)
        out[name] = df
    return out


def session_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns in %: close-to-close, overnight gap, and open-to-close."""
    prev = df["Close"].shift()
    return pd.DataFrame({
        "cc": np.log(df["Close"] / prev) * 100,
        "gap": np.log(df["Open"] / prev) * 100,
        "intraday": np.log(df["Close"] / df["Open"]) * 100,
    }).iloc[1:]


class Aligner:
    """Builds source features as seen from a target market's trading calendar."""

    def __init__(self, prices: dict[str, pd.DataFrame]):
        self.prices = prices
        self.calendar = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
        # log close on the union calendar, carried over the source's holidays
        self.log_close = pd.DataFrame(
            {m: np.log(df["Close"]) for m, df in prices.items()}).reindex(self.calendar).ffill()

    def source_feature(self, source: str, target: str) -> pd.Series:
        """Source return (in %) newly known before each of the target's opens."""
        lag = info_lag(source, target)
        pos = self.calendar.get_indexer(self.prices[target].index)
        cutoff = self.log_close[source].to_numpy()[np.clip(pos - lag, 0, None)]
        feat = pd.Series(cutoff, index=self.prices[target].index).diff() * 100
        return feat.iloc[1:]

    def design(self, target: str, sources=None) -> pd.DataFrame:
        sources = sources or [m for m in MARKETS if m != target]
        cols = {s: self.source_feature(s, target) for s in sources}
        return pd.DataFrame(cols).join(session_returns(self.prices[target])).dropna()
