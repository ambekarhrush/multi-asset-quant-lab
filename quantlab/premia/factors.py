"""Alternative risk premia: carry, momentum, value and defensive.

Following the "everywhere" literature (Asness, Moskowitz & Pedersen 2013;
Koijen, Moskowitz, Pedersen & Vrugt 2018; Frazzini & Pedersen 2014), each premium
is a cross-sectional, dollar-neutral long/short portfolio built from ranks, then
scaled to a common ex-ante volatility so premia can be risk-budgeted.

Universes (free data):
  FX      : 9 G10 currencies vs USD; spot from FRED, 3m rates from OECD/FRED
  Sectors : 9 SPDR US sector ETFs (since Dec-1998)
  Rates   : UST duration (IEF) timed by curve slope (term carry)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import data, panel, universe as u


# ------------------------------------------------------------------ FX panel
def fx_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Monthly (USD-per-foreign spot, rate differential, excess return of long-foreign)."""
    spot, diff = {}, {}
    usd = data.fred(u.USD_RATE).resample("ME").last()
    for ccy, (sid, conv, rid) in u.FX.items():
        s = data.fred(sid).resample("ME").last()
        spot[ccy] = s if conv == "USDper" else 1 / s
        # OECD monthly averages: stamp-of-month value known by month end; ffill short gaps only
        rf = data.fred(rid).resample("ME").last()
        diff[ccy] = (rf - usd.reindex(rf.index)) / 100
    spot = pd.DataFrame(spot)
    diff = pd.DataFrame(diff).reindex(spot.index).ffill(limit=6)
    # long foreign forward: spot change + carry accrued over the month (rate known at start)
    ret = spot.pct_change(fill_method=None) + diff.shift(1) / 12
    return spot, diff, ret


# ------------------------------------------------------------------ helpers
def rank_weights(score: pd.DataFrame, min_assets: int = 4) -> pd.DataFrame:
    """Dollar-neutral rank weights: sum(w)=0, sum|w|=2 (100% long, 100% short)."""
    r = score.rank(axis=1)
    live = score.notna().sum(axis=1)
    dm = r.sub(r.mean(axis=1), axis=0)
    w = dm.div(dm.abs().sum(axis=1) / 2, axis=0)
    return w.where(live >= min_assets, 0.0).fillna(0.0)


def momentum_score(px: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    return px.shift(skip) / px.shift(lookback) - 1


def value_score(px: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """Long-horizon reversal: cheap = fell most over ~5y (skip the most recent year)."""
    return -(np.log(px.shift(12)) - np.log(px.shift(12 * years)))


def betas(ret_d: pd.DataFrame, mkt_d: pd.Series, window: int = 252) -> pd.DataFrame:
    """Frazzini-Pedersen style beta: correlation (5y-ish) x vol ratio (1y), shrunk to 1."""
    vol_i = ret_d.rolling(window).std()
    vol_m = mkt_d.rolling(window).std()
    lr_i = ret_d.rolling(3).sum()
    lr_m = mkt_d.rolling(3).sum()
    corr = lr_i.rolling(window * 3, min_periods=window).corr(lr_m)
    b = corr.mul(vol_i.div(vol_m, axis=0))
    return 0.6 * b + 0.4 * 1.0


# ------------------------------------------------------------------ premia
def fx_premia(cost_bps: float = 3.0) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Returns {name: (weights, asset_returns)} for FX carry, momentum and value."""
    spot, diff, ret = fx_panel()
    # total-return index for momentum so carry accrual counts
    tri = (1 + ret.fillna(0)).cumprod().where(ret.notna().cumsum() > 0)
    return {
        "FX carry": (rank_weights(diff.shift(1)), ret),  # 1m publication lag on OECD rates
        "FX momentum": (rank_weights(momentum_score(tri)), ret),
        "FX value": (rank_weights(value_score(spot)), ret),
    }


def sector_premia() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    px = panel.monthly_prices(u.SECTORS)
    ret = panel.monthly_excess(u.SECTORS)
    d = panel.daily_excess(u.SECTORS + ["SPY"])
    b = betas(d[u.SECTORS], d["SPY"]).resample("ME").last()

    # Betting against beta: long low beta / short high beta, each leg levered to beta 1
    rk = b.rank(axis=1)
    dm = rk.sub(rk.mean(axis=1), axis=0)
    wl = (-dm).clip(lower=0)
    ws = dm.clip(lower=0)
    wl = wl.div(wl.sum(axis=1), axis=0)
    ws = ws.div(ws.sum(axis=1), axis=0)
    bl = (wl * b).sum(axis=1)
    bs = (ws * b).sum(axis=1)
    bab = wl.div(bl, axis=0) - ws.div(bs, axis=0)
    bab = bab.where(b.notna().sum(axis=1) >= 6, 0.0).fillna(0.0)

    return {
        "Equity momentum": (rank_weights(momentum_score(px)), ret),
        "Equity value": (rank_weights(value_score(px)), ret),
        "Defensive (BAB)": (bab, ret),
    }


def rates_carry() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Duration timed by term spread: long IEF scaled by z-scored 10y-3m slope.

    A steeper curve = more carry + roll-down for holding duration.
    """
    slope = data.fred("T10Y3M").resample("ME").last()
    z = ((slope - slope.expanding(60).mean()) / slope.expanding(60).std()).clip(-2, 2)
    ret = panel.monthly_excess(["IEF"])
    w = (z / 2).rename("IEF").to_frame().reindex(ret.index).fillna(0.0)
    return w, ret


# ------------------------------------------------------------------ risk budgeting
def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    pv = w @ cov @ w
    return w * (cov @ w) / pv


def risk_budget_weights(cov: np.ndarray, budget: np.ndarray, iters: int = 500, tol: float = 1e-10) -> np.ndarray:
    """Long-only weights whose risk contributions match ``budget`` (sums to 1).

    Cyclical coordinate descent (Griveau-Billion, Richard & Roncalli 2013) on
    the convex formulation min 0.5 w'Sw - sum(b log w).
    """
    n = len(budget)
    w = np.ones(n) / np.sqrt(np.diag(cov))
    w /= w.sum()
    for _ in range(iters):
        w_old = w.copy()
        for i in range(n):
            s_i = cov[i] @ w - cov[i, i] * w[i]
            a = cov[i, i]
            w[i] = (-s_i + np.sqrt(s_i**2 + 4 * a * budget[i])) / (2 * a)
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w / w.sum()


def allocate(
    premia_rets: pd.DataFrame,
    method: str = "erc",
    budget: dict[str, float] | None = None,
    lookback: int = 36,
    min_periods: int = 24,
) -> pd.DataFrame:
    """Rolling allocation across premium return streams (decided at t, applied t+1)."""
    cols = list(premia_rets.columns)
    W = pd.DataFrame(np.nan, index=premia_rets.index, columns=cols)
    for t in premia_rets.index:
        hist = premia_rets.loc[:t].iloc[-lookback:]
        live = [c for c in cols if hist[c].notna().sum() >= min_periods]
        if not live:
            continue
        h = hist[live].dropna()
        if len(h) < min_periods:
            continue
        cov = h.cov().values
        if method == "equal":
            wv = np.ones(len(live)) / len(live)
        elif method == "inv_vol":
            iv = 1 / np.sqrt(np.diag(cov))
            wv = iv / iv.sum()
        else:
            b = np.array([(budget or {}).get(c, 1.0) for c in live])
            wv = risk_budget_weights(cov, b / b.sum())
        W.loc[t, live] = wv
    return W.fillna(0.0)
