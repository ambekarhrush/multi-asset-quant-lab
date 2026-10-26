"""Performance and risk statistics.

All functions take *periodic simple returns* and a periodicity (252 daily, 12 monthly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ann_return(r: pd.Series, ppy: int) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float((1 + r).prod() ** (ppy / len(r)) - 1)


def ann_vol(r: pd.Series, ppy: int) -> float:
    return float(r.dropna().std() * np.sqrt(ppy))


def sharpe(r: pd.Series, ppy: int, rf: pd.Series | float = 0.0) -> float:
    ex = (r - rf).dropna()
    sd = ex.std()
    return float(ex.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan


def sortino(r: pd.Series, ppy: int) -> float:
    r = r.dropna()
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / dd * np.sqrt(ppy)) if dd > 0 else np.nan


def drawdown(r: pd.Series) -> pd.Series:
    wealth = (1 + r.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def calmar(r: pd.Series, ppy: int) -> float:
    mdd = max_drawdown(r)
    return ann_return(r, ppy) / abs(mdd) if mdd < 0 else np.nan


def cvar(r: pd.Series, alpha: float = 0.05) -> float:
    r = r.dropna()
    q = r.quantile(alpha)
    return float(r[r <= q].mean())


def deflated_sharpe(r: pd.Series, ppy: int, n_trials: int) -> float:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    Probability that the true Sharpe exceeds the max Sharpe expected from
    ``n_trials`` independent tries of pure noise. Corrects for multiple testing,
    skewness and fat tails. Computed on per-period (non-annualised) Sharpe.
    """
    r = r.dropna()
    T = len(r)
    if T < 30 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    # expected max SR under the null across n_trials (variance of SR estimates ~ 1/T)
    emc = 0.5772156649
    n = max(n_trials, 2)
    sr0 = np.sqrt(1.0 / T) * (
        (1 - emc) * stats.norm.ppf(1 - 1.0 / n) + emc * stats.norm.ppf(1 - 1.0 / (n * np.e))
    )
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def summary(r: pd.Series, ppy: int, turnover: pd.Series | None = None, n_trials: int | None = None) -> dict:
    out = {
        "CAGR": ann_return(r, ppy),
        "Vol": ann_vol(r, ppy),
        "Sharpe": sharpe(r, ppy),
        "Sortino": sortino(r, ppy),
        "MaxDD": max_drawdown(r),
        "Calmar": calmar(r, ppy),
        "CVaR5": cvar(r),
        "HitRate": float((r.dropna() > 0).mean()),
    }
    if turnover is not None:
        out["Turnover/yr"] = float(turnover.mean() * ppy)
    if n_trials is not None:
        out["DSR"] = deflated_sharpe(r, ppy, n_trials)
    return out


def table(rets: dict[str, pd.Series], ppy: int, **kw) -> pd.DataFrame:
    return pd.DataFrame({k: summary(v, ppy, **kw) for k, v in rets.items()}).T


def window_return(r: pd.Series, start: str, end: str) -> float:
    w = r.loc[start:end]
    return float((1 + w).prod() - 1)


def window_drawdown(r: pd.Series, start: str, end: str) -> float:
    return max_drawdown(r.loc[start:end])
