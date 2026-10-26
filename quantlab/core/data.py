"""Public market data loaders with a local parquet cache.

Sources (all free, all public):
  * Yahoo Finance via yfinance  -> total-return adjusted ETF prices
  * FRED (St. Louis Fed)        -> rates, macro, VIX, FX spot

Nothing here touches proprietary or employer data.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def _cached(name: str, fetch, refresh: bool = False) -> pd.DataFrame:
    path = CACHE / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch()
    df.to_parquet(path)
    return df


def fred(series_id: str, refresh: bool = False) -> pd.Series:
    """One FRED series as a float Series indexed by date (missing values dropped)."""

    def fetch():
        for attempt in range(4):
            r = requests.get(FRED_URL.format(series_id), timeout=30)
            if r.ok and r.text.startswith("observation_date"):
                break
            time.sleep(2 * (attempt + 1))
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["observation_date"])
        df = df.set_index("observation_date")
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        return df[[series_id]].dropna()

    return _cached(f"fred_{series_id}", fetch, refresh)[series_id].rename(series_id)


def fred_many(ids: list[str], refresh: bool = False) -> pd.DataFrame:
    return pd.concat([fred(i, refresh) for i in ids], axis=1)


def prices(tickers: list[str], start: str = "1990-01-01", refresh: bool = False) -> pd.DataFrame:
    """Adjusted (total return) daily closes for a list of tickers."""
    import yfinance as yf

    key = "px_" + "_".join(sorted(tickers))
    if len(key) > 120:
        key = "px_" + str(abs(hash(tuple(sorted(tickers)))))

    def fetch():
        df = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers[0])
        return df[tickers]

    out = _cached(key, fetch, refresh)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def to_month_end(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return df.resample("ME").last()
