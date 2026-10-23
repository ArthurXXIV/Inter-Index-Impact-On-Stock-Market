"""The 12-market universe for the daily study, with trading hours in UTC.

Hours are the *conservative* bound across daylight-saving switches: the
earliest an exchange can open and the latest it can close. A source market's
close only counts as known to a target market if it is strictly earlier than
the target's open, so the alignment never uses information from the future.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    ticker: str
    region: str
    open_utc: float    # earliest open, hours after 00:00 UTC on the local trading date
    close_utc: float   # latest close
    reliable_open: bool = True  # Yahoo's Open is a real opening print


MARKETS = {
    "US":        Market("^GSPC",     "Americas", 13.5, 21.0),
    "Canada":    Market("^GSPTSE",   "Americas", 13.5, 21.0),
    "Brazil":    Market("^BVSP",     "Americas", 13.0, 21.0, reliable_open=False),
    "UK":        Market("^FTSE",     "Europe",    7.0, 16.5, reliable_open=False),
    "Germany":   Market("^GDAXI",    "Europe",    7.0, 16.5),
    "France":    Market("^FCHI",     "Europe",    7.0, 16.5),
    "Japan":     Market("^N225",     "Asia",      0.0,  6.5),
    "HongKong":  Market("^HSI",      "Asia",      1.5,  8.0),
    "China":     Market("000001.SS", "Asia",      1.5,  7.0),
    "Korea":     Market("^KS11",     "Asia",      0.0,  6.5),
    "Australia": Market("^AXJO",     "Asia",     -1.0,  6.0, reliable_open=False),  # opens 23:00 UTC the day before in summer
    "India":     Market("^NSEI",     "Asia",      3.75, 10.0),
}


def info_lag(source: str, target: str) -> int:
    """0 if the source's close on date d is known before the target opens on d, else 1."""
    return 0 if MARKETS[source].close_utc < MARKETS[target].open_utc else 1
