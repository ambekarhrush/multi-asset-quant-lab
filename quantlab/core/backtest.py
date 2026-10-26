"""Vectorised portfolio backtester and walk-forward harness.

Conventions (chosen to make look-ahead impossible by construction):
  * ``weights`` at date t are decided using information available at the close of t.
  * They earn the return from t to t+1, i.e. returns are shifted by one period.
  * Costs = turnover x cost_bps, charged in the period the trade happens.
  * Cash not deployed earns the risk-free rate if ``rf`` is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class Result:
    returns: pd.Series
    gross: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    ppy: int
    meta: dict = field(default_factory=dict)

    def stats(self, n_trials: int | None = None) -> dict:
        return metrics.summary(self.returns, self.ppy, self.turnover, n_trials)

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns.fillna(0)).cumprod()


def run(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cost_bps: float | dict[str, float] = 5.0,
    ppy: int = 12,
    rf: pd.Series | None = None,
    financing: bool = True,
) -> Result:
    """Backtest a weight panel against simple asset returns of the same frequency.

    ``cost_bps`` may be a scalar or a per-asset dict (e.g. wider for EM / HY ETFs).
    If ``financing`` is True and ``rf`` given, leverage above 100% gross pays rf and
    unused capital earns rf, so ETF positions behave like funded futures.
    """
    cols = [c for c in weights.columns if c in asset_returns.columns]
    w = weights[cols].reindex(asset_returns.index).ffill().fillna(0.0)
    r = asset_returns[cols].fillna(0.0)

    w_held = w.shift(1).fillna(0.0)  # weights decided at t-1 earn r_t
    gross = (w_held * r).sum(axis=1)

    if rf is not None and financing:
        rf_p = rf.reindex(r.index).ffill().fillna(0.0)
        net_exposure = w_held.sum(axis=1)
        gross = gross + (1 - net_exposure) * rf_p

    # drift-adjusted turnover: compare target weights with weights after drift
    drift = w_held * (1 + r)
    denom = (1 + (w_held * r).sum(axis=1)).replace(0, np.nan)
    drifted = drift.div(denom, axis=0).fillna(0.0)
    trades = (w - drifted).abs()
    if isinstance(cost_bps, dict):
        cvec = pd.Series({c: cost_bps.get(c, 5.0) for c in cols}) / 1e4
        costs = (trades * cvec).sum(axis=1)
    else:
        costs = trades.sum(axis=1) * cost_bps / 1e4
    turnover = trades.sum(axis=1) / 2

    net = gross - costs
    first = w.abs().sum(axis=1).gt(0).idxmax()
    sl = slice(first, None)
    return Result(net.loc[sl], gross.loc[sl], w.loc[sl], turnover.loc[sl], costs.loc[sl], ppy)


def vol_target(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target: float,
    ppy: int,
    lookback: int,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """Scale a weight panel so its ex-ante portfolio vol hits ``target``.

    Uses a trailing covariance (lookback periods) known at time t only.
    """
    r = asset_returns[weights.columns]
    scale = pd.Series(np.nan, index=weights.index)
    for i, t in enumerate(weights.index):
        hist = r.loc[:t].iloc[-lookback:]
        if len(hist) < lookback // 2:
            continue
        cov = hist.cov().values * ppy
        wv = weights.loc[t].fillna(0).values
        pv = np.sqrt(max(wv @ cov @ wv, 1e-12))
        scale.loc[t] = min(target / pv, max_leverage) if pv > 0 else 0.0
    return weights.mul(scale, axis=0)


def walk_forward(
    fit_predict: Callable[[pd.Timestamp, pd.Timestamp, pd.Timestamp], pd.DataFrame],
    index: pd.DatetimeIndex,
    train_years: int = 10,
    test_years: int = 2,
    expanding: bool = True,
) -> pd.DataFrame:
    """Generic walk-forward: refit on a train window, emit weights for the next test window.

    ``fit_predict(train_start, train_end, test_end)`` must return weights indexed over
    (train_end, test_end]. Only data up to ``train_end`` may be used for fitting.
    Concatenates the out-of-sample slices only.
    """
    out = []
    start = index[0]
    t_end = start + pd.DateOffset(years=train_years)
    while t_end < index[-1]:
        test_end = min(t_end + pd.DateOffset(years=test_years), index[-1])
        tr_start = start if expanding else t_end - pd.DateOffset(years=train_years)
        w = fit_predict(tr_start, t_end, test_end)
        out.append(w.loc[(w.index > t_end) & (w.index <= test_end)])
        t_end = test_end
    return pd.concat(out)
