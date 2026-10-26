"""Standard return panels used by every module (excess of 3m T-bill)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import data, universe as u


@lru_cache(maxsize=None)
def daily_prices() -> pd.DataFrame:
    return data.prices(list(u.MULTI_ASSET) + u.SECTORS + ["^GSPC"], start="1990-01-01")


@lru_cache(maxsize=None)
def rf_daily() -> pd.Series:
    tb = data.fred("DTB3") / 100
    idx = daily_prices().index
    return (tb.reindex(idx.union(tb.index)).ffill().reindex(idx) / 252).rename("rf")


@lru_cache(maxsize=None)
def rf_monthly() -> pd.Series:
    tb = data.fred("DTB3").resample("ME").last().shift(1) / 100  # rate known at start of month
    return (tb / 12).rename("rf")


def daily_excess(tickers: list[str]) -> pd.DataFrame:
    r = daily_prices()[tickers].pct_change(fill_method=None)
    return r.sub(rf_daily(), axis=0)


def monthly_prices(tickers: list[str]) -> pd.DataFrame:
    return daily_prices()[tickers].resample("ME").last()


def monthly_excess(tickers: list[str]) -> pd.DataFrame:
    r = monthly_prices(tickers).pct_change(fill_method=None)
    return r.sub(rf_monthly().reindex(r.index), axis=0)
