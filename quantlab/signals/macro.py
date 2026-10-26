"""Macro regime signals from public FRED data, with publication lags respected.

Four state variables, each a 3-month *change* (direction of travel matters more
than level for asset returns):
  growth     : YoY industrial production growth            (INDPRO)
  inflation  : YoY CPI inflation                            (CPIAUCSL)
  conditions : Chicago Fed National Financial Conditions    (NFCI, higher = tighter)
  curve      : 10y-3m Treasury slope                        (T10Y3M)

Two ways to turn states into positions:
  1. ``prior_tilts``  - a fixed, economically-motivated sensitivity matrix, decided
                        before looking at any backtest. No fitting, no look-ahead.
  2. ``ridge_walk_forward`` - per-asset ridge regression of next-month return on the
                        z-scored states, refit on an expanding window. Out of sample only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import data

# release lag in months *after* resampling a monthly-stamped series to month end
LAGS = {"INDPRO": 1, "CPIAUCSL": 1, "NFCI": 0, "T10Y3M": 0}


def macro_states(refresh: bool = False) -> pd.DataFrame:
    grid = pd.date_range("1960-01-31", pd.Timestamp.today() + pd.offsets.MonthEnd(0), freq="ME")

    def monthly(sid, how="last"):
        s = data.fred(sid, refresh).resample("ME")
        s = s.last() if how == "last" else s.mean()
        # extend to the current month *before* lagging, so the latest release lands on this month
        return s.reindex(grid).shift(LAGS[sid])

    ip, cpi = monthly("INDPRO"), monthly("CPIAUCSL")
    nfci, curve = monthly("NFCI"), monthly("T10Y3M", "mean")

    growth = ip.pct_change(12) * 100
    infl = cpi.pct_change(12) * 100
    st = pd.DataFrame(
        {
            "growth": growth.diff(3),
            "inflation": infl.diff(3),
            "conditions": nfci.diff(3),
            "curve": curve.diff(3),
        }
    )
    return st


def zscore_expanding(df: pd.DataFrame, min_periods: int = 60, clip: float = 3.0) -> pd.DataFrame:
    mu = df.expanding(min_periods).mean()
    sd = df.expanding(min_periods).std()
    return ((df - mu) / sd).clip(-clip, clip)


# Sign conventions: +1 means the asset tends to do better when the state *rises*.
PRIOR = pd.DataFrame(
    {
        #            growth inflation conditions curve
        "Equity":    [ 1.0,  -0.5,     -1.0,     0.5],
        "Rates":     [-1.0,  -1.0,      0.5,    -0.5],
        "Credit":    [ 0.5,  -0.5,     -1.0,     0.5],
        "Commodity": [ 0.5,   1.0,     -0.5,     0.0],
    },
    index=["growth", "inflation", "conditions", "curve"],
).T


def prior_tilts(z: pd.DataFrame, asset_class: dict[str, str]) -> pd.DataFrame:
    """Score in roughly [-1, 1] per asset from the fixed sensitivity matrix."""
    scores = {}
    for tkr, ac in asset_class.items():
        b = PRIOR.loc[ac]
        scores[tkr] = (z[b.index] * b).sum(axis=1) / b.abs().sum()
    out = pd.DataFrame(scores)
    if "TIP" in out:  # TIPS: real-rate duration, but hedged to inflation surprises
        out["TIP"] = out["TIP"] + 0.5 * z["inflation"] / 2.5
    if "GLD" in out:  # gold: likes tightening conditions + inflation, not growth
        out["GLD"] = (0.5 * z["inflation"] + 0.5 * z["conditions"] - 0.25 * z["growth"]) / 1.25
    return np.tanh(out)


def ridge_walk_forward(
    z: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    first_train_end: str = "2010-12-31",
    alpha: float = 10.0,
    refit_every: int = 12,
) -> pd.DataFrame:
    """Expected next-month return per asset, predicted strictly out of sample.

    fwd_ret[t] must be the return realised over (t, t+1]; we only train on rows
    whose forward return is fully known by the refit date.
    """
    from sklearn.linear_model import Ridge

    idx = z.index
    preds = pd.DataFrame(np.nan, index=idx, columns=fwd_ret.columns)
    refit_dates = idx[(idx >= first_train_end)][::refit_every]
    for i, t in enumerate(refit_dates):
        nxt = refit_dates[i + 1] if i + 1 < len(refit_dates) else idx[-1] + pd.Timedelta(days=1)
        # training rows: forward return ends <= t  -> feature date <= t - 1 month
        train_idx = idx[idx < t]
        for a in fwd_ret.columns:
            X = z.loc[train_idx]
            y = fwd_ret.loc[train_idx, a]
            m = X.notna().all(axis=1) & y.notna()
            if m.sum() < 36:
                continue
            model = Ridge(alpha=alpha).fit(X[m].values, y[m].values)
            test = idx[(idx >= t) & (idx < nxt)]
            Xt = z.loc[test]
            ok = Xt.notna().all(axis=1)
            preds.loc[test[ok.values], a] = model.predict(Xt[ok].values)
    return preds


def macro_weights(score: pd.DataFrame, monthly_vol: pd.DataFrame, asset_vol_target: float = 0.10) -> pd.DataFrame:
    n = score.notna().sum(axis=1).replace(0, np.nan)
    w = score * (asset_vol_target / monthly_vol.reindex(score.index))
    return w.div(np.sqrt(n), axis=0).fillna(0.0)
