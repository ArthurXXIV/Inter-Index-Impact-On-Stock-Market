"""Download and cache weekly index prices from Yahoo Finance."""

from pathlib import Path

import pandas as pd
import yfinance as yf

INDICES = {
    "US": "^GSPC",   # S&P 500
    "UK": "^FTSE",   # FTSE 100
    "Japan": "^N225",  # Nikkei 225
    "India": "^NSEI",  # NIFTY 50
}

START, END = "2019-04-01", "2026-10-19"  # end on the last complete week
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1wk",
                     auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_prices(start: str = START, end: str = END, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Return {market: weekly OHLCV}, reading from data/ when cached.

    Caching keeps results reproducible, since Yahoo occasionally revises history.
    """
    DATA_DIR.mkdir(exist_ok=True)
    prices = {}
    for market, ticker in INDICES.items():
        path = DATA_DIR / f"{market.lower()}_weekly.csv"
        if path.exists() and not refresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = _download(ticker, start, end)
            df.to_csv(path)
        prices[market] = df
    return prices
