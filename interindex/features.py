"""Technical indicators and the aligned multi-market feature table.

Raw price levels trend over time, so a model trained on 2019-2023 levels sees
values it has never seen in 2024. Every feature here is scale-free (a return,
a bounded oscillator, or a ratio to price) so it means the same thing in any year.
"""

import numpy as np
import pandas as pd

FEATURES = ["Return", "RSI", "StochK", "StochD", "MACD", "MACD_Signal",
            "BB_PctB", "EMA_Gap", "Range"]


def technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["Close"], df["High"], df["Low"]
    out = pd.DataFrame(index=df.index)

    out["Return"] = close.pct_change() * 100

    # RSI (14), Wilder smoothing
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["RSI"] = 100 - 100 / (1 + gain / loss)

    # Slow stochastic oscillator (14, 3, 3)
    low14, high14 = low.rolling(14).min(), high.rolling(14).max()
    fast_k = 100 * (close - low14) / (high14 - low14)
    out["StochK"] = fast_k.rolling(3).mean()
    out["StochD"] = out["StochK"].rolling(3).mean()

    # MACD (12, 26, 9), as % of price so it is comparable across years
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["MACD"] = macd / close * 100
    out["MACD_Signal"] = macd.ewm(span=9, adjust=False).mean() / close * 100

    # Bollinger %B (20, 2σ): where close sits inside the bands
    mid, sd = close.rolling(20).mean(), close.rolling(20).std()
    out["BB_PctB"] = (close - (mid - 2 * sd)) / (4 * sd)

    # Distance from 5-week EMA, and weekly high-low range
    out["EMA_Gap"] = (close / close.ewm(span=5, adjust=False).mean() - 1) * 100
    out["Range"] = (high - low) / close * 100

    return out.replace([np.inf, -np.inf], np.nan)


def build_panel(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per week, columns like 'India_RSI'. Only weeks where all markets traded."""
    frames = [technical_indicators(df).add_prefix(f"{m}_") for m, df in prices.items()]
    return pd.concat(frames, axis=1, join="inner").dropna()


def columns_for(markets, panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c.split("_", 1)[0] in markets]
