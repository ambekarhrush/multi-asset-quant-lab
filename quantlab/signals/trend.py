"""Time-series momentum (trend following).

Moskowitz, Ooi & Pedersen (2012); Hurst, Ooi & Pedersen (2017).
Score per asset = mean of sign(trailing return) over several horizons, in [-1, 1].
Position = score x (vol budget / asset vol), so each asset contributes similar risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trend_score(monthly_px: pd.DataFrame, horizons=(1, 3, 6, 12), skip: int = 0) -> pd.DataFrame:
    scores = []
    for h in horizons:
        ret = monthly_px.shift(skip) / monthly_px.shift(skip + h) - 1
        scores.append(np.sign(ret))
    return sum(scores) / len(scores)


def ewma_vol(daily_ret: pd.DataFrame, halflife: int = 60, ppy: int = 252) -> pd.DataFrame:
    return daily_ret.ewm(halflife=halflife, min_periods=halflife).std() * np.sqrt(ppy)


def trend_weights(
    monthly_px: pd.DataFrame,
    daily_ret: pd.DataFrame,
    horizons=(1, 3, 6, 12),
    asset_vol_target: float = 0.10,
    long_only: bool = False,
) -> pd.DataFrame:
    score = trend_score(monthly_px, horizons)
    if long_only:
        score = score.clip(lower=0)
    vol = ewma_vol(daily_ret).resample("ME").last().reindex(score.index)
    n = score.notna().sum(axis=1).replace(0, np.nan)
    # divide the budget across live assets so total risk doesn't scale with universe size
    w = score * (asset_vol_target / vol)
    return w.div(np.sqrt(n), axis=0).fillna(0.0)
