"""Index option overlays on an equity portfolio, simulated daily.

Why model-priced: historical SPX option chains are not free. Options are priced
with Black-Scholes (Merton, continuous dividend) on the S&P 500 price index, with
implied vol taken from the VIX through a parametric skew:

    atm      = VIX/100 * ATM_RATIO          (VIX is a variance-swap level, sits above ATM)
    put  k<0 : sigma = atm * (1 - PUT_SLOPE  * k)
    call k>0 : sigma = max(atm * (1 - CALL_SLOPE * k), 0.6 * atm)     k = ln(K/F)

Defaults give ~+3.5 vol points for a 1m 95% put at ATM 15 - in line with typical
SPX skew. Skew is the key assumption, so ``skew_mult`` lets you stress it.
Bid/ask: buyer pays mid*(1+spread), seller receives mid*(1-spread).

The equity leg is SPY total return; option P&L is marked to model every day, so
drawdowns *inside* a month are captured, not just month-end snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

ATM_RATIO = 0.90
PUT_SLOPE = 4.5
CALL_SLOPE = 2.0
DIV_YIELD = 0.018


def bs_price(S, K, T, r, q, sigma, kind: str):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    T = np.maximum(T, 1e-8)
    fwd = S * np.exp((r - q) * T)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(fwd / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    df = np.exp(-r * T)
    if kind == "put":
        return df * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))
    return df * (fwd * norm.cdf(d1) - K * norm.cdf(d2))


def bs_delta(S, K, T, r, q, sigma, kind: str):
    T = max(T, 1e-8)
    fwd = S * np.exp((r - q) * T)
    sd = sigma * np.sqrt(T)
    d1 = (np.log(fwd / K) + 0.5 * sd**2) / sd
    return np.exp(-q * T) * (norm.cdf(d1) - (kind == "put"))


def skew_vol(vix, S, K, T, r, q=DIV_YIELD, skew_mult: float = 1.0):
    fwd = S * np.exp((r - q) * T)
    k = np.log(K / fwd)
    atm = vix / 100 * ATM_RATIO
    put = atm * (1 - PUT_SLOPE * skew_mult * k)
    call = np.maximum(atm * (1 - CALL_SLOPE * skew_mult * k), 0.6 * atm)
    return np.where(k < 0, put, call)


@dataclass
class Leg:
    kind: str          # "put" / "call"
    moneyness: float   # strike / spot at inception
    qty: float         # +1 long, -1 short, per unit of hedged notional


@dataclass
class Overlay:
    name: str
    legs: list[Leg]
    hedge_ratio: float = 1.0
    tenor_months: int = 1
    spread: float = 0.03          # proportional half-spread on option premium
    gate: pd.Series | None = None  # optional bool series: hedge only when True at roll
    zero_cost_call: bool = False   # solve the short call strike so the package costs ~0
    meta: dict = field(default_factory=dict)


def _roll_dates(idx: pd.DatetimeIndex, months: int) -> pd.DatetimeIndex:
    me = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last()
    return pd.DatetimeIndex(me.values[::months])


def simulate(
    ov: Overlay,
    spx: pd.Series,
    equity_tr: pd.Series,
    vix: pd.Series,
    rf_annual: pd.Series,
    skew_mult: float = 1.0,
) -> pd.DataFrame:
    """Daily NAV of (equity + option overlay). Returns DataFrame with nav, ret, option_value."""
    idx = spx.index.intersection(equity_tr.index).intersection(vix.dropna().index)
    spx, eq, vix, rf = spx[idx], equity_tr[idx], vix[idx], rf_annual.reindex(idx).ffill().fillna(0.0)
    rolls = set(_roll_dates(idx, ov.tenor_months))

    nav = np.empty(len(idx))
    optv = np.zeros(len(idx))
    prem_paid = np.zeros(len(idx))
    eq_units = 0.0
    cash = 0.0
    pos: list[tuple[str, float, float]] = []  # (kind, strike, units)
    expiry = None
    V = 1.0

    def value_book(i):
        if not pos:
            return 0.0
        T = max((expiry - idx[i]).days / 365.0, 0.0)
        S = spx.iloc[i]
        tot = 0.0
        for kind, K, units in pos:
            if T <= 1e-6:
                v = max(K - S, 0) if kind == "put" else max(S - K, 0)
            else:
                sig = skew_vol(vix.iloc[i], S, K, T, rf.iloc[i], skew_mult=skew_mult)
                v = float(bs_price(S, K, T, rf.iloc[i], DIV_YIELD, sig, kind))
            tot += units * v
        return tot

    for i, t in enumerate(idx):
        if i > 0:
            dt = (t - idx[i - 1]).days / 365.0
            cash *= np.exp(rf.iloc[i - 1] * dt)
        ov_val = value_book(i)
        V = eq_units * eq.iloc[i] + cash + ov_val if i > 0 else 1.0

        if t in rolls or i == 0:
            # close existing book at bid (longs) / ask (shorts) and reopen
            close_val = 0.0
            if pos:
                T = max((expiry - t).days / 365.0, 0.0)
                for kind, K, units in pos:
                    if T <= 1e-6:
                        v = max(K - spx.iloc[i], 0) if kind == "put" else max(spx.iloc[i] - K, 0)
                        close_val += units * v
                    else:
                        sig = skew_vol(vix.iloc[i], spx.iloc[i], K, T, rf.iloc[i], skew_mult=skew_mult)
                        v = float(bs_price(spx.iloc[i], K, T, rf.iloc[i], DIV_YIELD, sig, kind))
                        close_val += units * v * (1 - ov.spread * np.sign(units))
            V = eq_units * eq.iloc[i] + cash + close_val if i > 0 else 1.0
            pos = []
            expiry = t + pd.DateOffset(months=ov.tenor_months)
            T = (expiry - t).days / 365.0
            active = ov.gate is None or bool(ov.gate.reindex([t]).ffill().fillna(True).iloc[0])
            cost = 0.0
            if active and ov.hedge_ratio > 0:
                S = spx.iloc[i]
                units = ov.hedge_ratio * V / S
                legs = list(ov.legs)
                if ov.zero_cost_call:
                    legs = _zero_cost(legs, S, T, vix.iloc[i], rf.iloc[i], skew_mult)
                for lg in legs:
                    K = lg.moneyness * S
                    sig = skew_vol(vix.iloc[i], S, K, T, rf.iloc[i], skew_mult=skew_mult)
                    mid = float(bs_price(S, K, T, rf.iloc[i], DIV_YIELD, sig, lg.kind))
                    px = mid * (1 + ov.spread * np.sign(lg.qty))
                    cost += lg.qty * units * px
                    pos.append((lg.kind, K, lg.qty * units))
            prem_paid[i] = cost / V
            # fully invest the rest in equity; premium is funded from the portfolio
            eq_units = (V - cost) / eq.iloc[i]
            cash = 0.0
            ov_val = value_book(i)
            V = eq_units * eq.iloc[i] + cash + ov_val
        nav[i] = V
        optv[i] = ov_val / V if V else 0.0

    out = pd.DataFrame({"nav": nav, "option_value": optv, "premium": prem_paid}, index=idx)
    out["ret"] = out["nav"].pct_change().fillna(0.0)
    return out


def _zero_cost(legs: list[Leg], S, T, vix, r, skew_mult) -> list[Leg]:
    """Replace the short call's moneyness so call premium ~= put premium."""
    put_cost = 0.0
    for lg in legs:
        if lg.kind == "put":
            K = lg.moneyness * S
            put_cost += lg.qty * float(bs_price(S, K, T, r, DIV_YIELD, skew_vol(vix, S, K, T, r, skew_mult=skew_mult), "put"))
    lo, hi = 1.0, 1.5
    for _ in range(40):
        m = 0.5 * (lo + hi)
        K = m * S
        c = float(bs_price(S, K, T, r, DIV_YIELD, skew_vol(vix, S, K, T, r, skew_mult=skew_mult), "call"))
        if c > put_cost:
            lo = m
        else:
            hi = m
    out = [lg for lg in legs if lg.kind == "put"]
    out.append(Leg("call", 0.5 * (lo + hi), -1.0))
    return out


# --------------------------------------------------------------- analytics
EPISODES = {
    "GFC 2007-09": ("2007-10-09", "2009-03-09"),
    "COVID 2020": ("2020-02-19", "2020-03-23"),
    "Inflation 2022": ("2022-01-03", "2022-10-12"),
}


def episode_drawdown(ret: pd.Series, start: str, end: str) -> float:
    """Peak-to-trough loss inside the episode window."""
    w = (1 + ret.loc[start:end]).cumprod()
    return float((w / w.cummax() - 1).min())


def hedge_report(base: pd.Series, hedged: dict[str, pd.Series], calm=("2010-01-01", "2019-12-31")) -> pd.DataFrame:
    from ..core import metrics

    rows = {}
    b_cagr = metrics.ann_return(base, 252)
    b_calm = metrics.ann_return(base.loc[calm[0]:calm[1]], 252)
    for k, r in hedged.items():
        row = {
            "CAGR": metrics.ann_return(r, 252),
            "Vol": metrics.ann_vol(r, 252),
            "Sharpe": metrics.sharpe(r, 252),
            "MaxDD": metrics.max_drawdown(r),
            "Cost p.a. (full)": b_cagr - metrics.ann_return(r, 252),
            "Cost p.a. (2010-19 calm)": b_calm - metrics.ann_return(r.loc[calm[0]:calm[1]], 252),
        }
        saved = []
        for ep, (a, z) in EPISODES.items():
            dd_b = episode_drawdown(base, a, z)
            dd_h = episode_drawdown(r, a, z)
            row[f"DD {ep}"] = dd_h
            row[f"Saved {ep}"] = dd_h - dd_b
            saved.append(dd_h - dd_b)
        c = row["Cost p.a. (2010-19 calm)"]
        row["Avg DD saved / calm cost"] = np.mean(saved) / c if c > 1e-6 else np.nan
        rows[k] = row
    return pd.DataFrame(rows).T
