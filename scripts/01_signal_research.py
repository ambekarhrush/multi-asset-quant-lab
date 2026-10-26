"""Module 1 - Signal generation research: trend, macro and volatility signals.

Every result is (a) net of per-asset transaction costs, (b) in excess of T-bills,
(c) either parameter-free, fixed ex ante, or fitted walk-forward out of sample.
A parameter grid is run for trend and reported with a deflated Sharpe ratio so
the headline number is not the best of many tries.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab.core import backtest as bt, metrics, panel, plotting, universe as u  # noqa: E402
from quantlab.signals import macro, trend, volatility  # noqa: E402

OUT = ROOT / "results" / "01_signals"
OUT.mkdir(parents=True, exist_ok=True)
START = "2004-01-31"  # first month with >= 10 of 12 ETFs live

tick = list(u.MULTI_ASSET)
mpx = panel.monthly_prices(tick)
mret = panel.monthly_excess(tick)
dret = panel.daily_excess(tick)
mvol = trend.ewma_vol(dret).resample("ME").last()


def scale_to(r: pd.Series, target=0.10, lookback=36, ppy=12) -> pd.Series:
    """Sleeve-level vol targeting using only trailing realised vol."""
    vol = r.rolling(lookback, min_periods=24).std().shift(1) * np.sqrt(ppy)
    lev = (target / vol).clip(upper=3).fillna(1.0)
    return r * lev


# ---------------------------------------------------------------- benchmarks
bench_6040 = bt.run(pd.DataFrame({"SPY": 0.6, "IEF": 0.4}, index=mret.index), mret, u.COST_BPS).returns
iv = (1 / mvol).where(mret.notna().cumsum() > 0)
iv_w = iv.div(iv.sum(axis=1), axis=0)
bench_iv = bt.run(iv_w, mret, u.COST_BPS).returns

# ---------------------------------------------------------------- A. trend
grid = list(itertools.product([(1, 3, 6, 12), (3, 6, 12), (12,), (1, 3), (6, 12)], [0.05, 0.10]))
grid_res = {}
for hz, tgt in grid:
    w = trend.trend_weights(mpx, dret, horizons=hz, asset_vol_target=tgt)
    grid_res[f"h={hz} t={tgt}"] = bt.run(w, mret, u.COST_BPS).returns.loc[START:]
grid_tbl = metrics.table(grid_res, 12)
N_TRIALS_TREND = len(grid)

w_trend = trend.trend_weights(mpx, dret, horizons=(1, 3, 6, 12), asset_vol_target=0.10)
res_trend = bt.run(w_trend, mret, u.COST_BPS)
trend_r = res_trend.returns.loc[START:]

# ---------------------------------------------------------------- B. macro
states = macro.macro_states()
z = macro.zscore_expanding(states).reindex(mret.index)

score_prior = macro.prior_tilts(z, u.ASSET_CLASS).where(mret.notna().cumsum() > 0)
w_macro = macro.macro_weights(score_prior, mvol)
res_macro = bt.run(w_macro, mret, u.COST_BPS)
macro_r = res_macro.returns.loc[START:]

fwd = mret.shift(-1)
pred = macro.ridge_walk_forward(z, fwd, first_train_end="2010-12-31")
score_ridge = (pred.sub(pred.mean(axis=1), axis=0)).div(pred.std(axis=1), axis=0).clip(-2, 2) / 2
w_ridge = macro.macro_weights(score_ridge, mvol)
res_ridge = bt.run(w_ridge, mret, u.COST_BPS)
ridge_r = res_ridge.returns.loc["2011-01-31":]

# ---------------------------------------------------------------- C. volatility (daily, SPY)
vix = panel.data.fred("VIXCLS")
vix3m = panel.data.fred("VXVCLS")
spy = panel.daily_excess(["SPY"])["SPY"]
spx_raw = panel.daily_prices()["^GSPC"].pct_change(fill_method=None)
idx = spy.dropna().index

sig_ts = volatility.term_structure_signal(vix.reindex(idx).ffill(), vix3m.reindex(idx).ffill())
sig_vrp = volatility.vrp_signal(vix.reindex(idx).ffill(), spx_raw.reindex(idx))
sig_vt = volatility.vol_target_signal(spx_raw.reindex(idx))

spy_df = spy.to_frame()
vol_sleeves = {}
for name, s in {"VIX term structure": sig_ts, "Variance risk premium": sig_vrp, "Vol target 15%": sig_vt}.items():
    r = bt.run(s.to_frame("SPY"), spy_df, cost_bps=u.COST_BPS["SPY"], ppy=252).returns
    vol_sleeves[name] = r
combo_sig = pd.concat([sig_ts, sig_vt], axis=1).mean(axis=1, skipna=False)
vol_combo_d = bt.run(combo_sig.to_frame("SPY"), spy_df, cost_bps=1, ppy=252)
spy_bh_d = spy.loc[vol_combo_d.returns.index]
vol_r_m = (1 + vol_combo_d.returns).resample("ME").prod() - 1

# ---------------------------------------------------------------- D. combined
sleeves = pd.DataFrame(
    {"Trend": scale_to(trend_r), "Macro (ridge)": scale_to(ridge_r), "Vol-timed equity": scale_to(vol_r_m)}
).loc[START:]
# Sleeves are combined at the return level with trailing inverse-vol weights. The macro
# sleeve is the walk-forward ridge model (enters from 2011, when its first OOS month exists).
inv = 1 / sleeves.rolling(36, min_periods=24).std().shift(1)
wts = inv.div(inv.sum(axis=1), axis=0)
wts = wts.where(sleeves.notna()).fillna(0)
wts = wts.div(wts.sum(axis=1), axis=0).fillna(0)
combo = scale_to((sleeves.fillna(0) * wts).sum(axis=1))

# ---------------------------------------------------------------- outputs
monthly = {
    "Trend (TSMOM)": trend_r,
    "Macro regime (prior)": macro_r,
    "Macro ridge (walk-fwd)": ridge_r,
    "Combined signals": combo,
    "60/40": bench_6040.loc[START:],
    "Inverse-vol long only": bench_iv.loc[START:],
}
tbl = metrics.table(monthly, 12)
tbl.loc["Trend (TSMOM)", "DSR"] = metrics.deflated_sharpe(trend_r, 12, N_TRIALS_TREND)
tbl.loc["Trend (TSMOM)", "Turnover/yr"] = res_trend.turnover.loc[START:].mean() * 12
tbl.loc["Macro regime (prior)", "Turnover/yr"] = res_macro.turnover.loc[START:].mean() * 12
tbl.loc["Macro ridge (walk-fwd)", "Turnover/yr"] = res_ridge.turnover.loc["2011":].mean() * 12

daily_start = vol_combo_d.returns.dropna().index[0]
daily = {k: v.loc[daily_start:] for k, v in vol_sleeves.items()}
daily["Term structure + vol target"] = vol_combo_d.returns
daily["SPY buy & hold"] = spy_bh_d
vtbl = metrics.table(daily, 252)

costs = pd.DataFrame(
    {
        "Gross Sharpe": [metrics.sharpe(res_trend.gross.loc[START:], 12), metrics.sharpe(res_macro.gross.loc[START:], 12)],
        "Net Sharpe": [metrics.sharpe(trend_r, 12), metrics.sharpe(macro_r, 12)],
        "Cost drag bps/yr": [res_trend.costs.loc[START:].mean() * 12e4, res_macro.costs.loc[START:].mean() * 12e4],
    },
    index=["Trend", "Macro (prior)"],
)

crisis = {}
for name, (a, b) in {"GFC": ("2007-10-31", "2009-03-31"), "COVID": ("2020-01-31", "2020-04-30"), "2022": ("2021-12-31", "2022-10-31")}.items():
    crisis[name] = {k: metrics.window_return(v, a, b) for k, v in monthly.items()}
crisis = pd.DataFrame(crisis)

tbl.round(3).to_csv(OUT / "summary_monthly.csv")
vtbl.round(3).to_csv(OUT / "summary_vol_daily.csv")
grid_tbl.round(3).to_csv(OUT / "trend_parameter_grid.csv")
costs.round(3).to_csv(OUT / "cost_impact.csv")
crisis.round(3).to_csv(OUT / "crisis_returns.csv")
pd.DataFrame(monthly).to_csv(OUT / "monthly_returns.csv")
sleeves.corr().round(2).to_csv(OUT / "sleeve_correlation.csv")
pd.DataFrame(daily).to_csv(OUT / "vol_daily_returns.csv")
latest = {
    "as_of": str(panel.daily_prices().index[-1].date()),
    "trend_score": trend.trend_score(mpx).iloc[-1].round(2).to_dict(),
    "macro_states_z": z.iloc[-1].round(2).to_dict(),
    "macro_prior_score": score_prior.iloc[-1].round(2).to_dict(),
    "vix": float(vix.iloc[-1]),
    "vix_vix3m": float((vix / vix3m).dropna().iloc[-1]),
    "vol_exposure": {"term_structure": float(sig_ts.dropna().iloc[-1]), "vol_target": float(sig_vt.dropna().iloc[-1]), "vrp": float(sig_vrp.dropna().iloc[-1])},
    "trend_weights": w_trend.iloc[-1].round(3).to_dict(),
}
(OUT / "latest_signals.json").write_text(json.dumps(latest, indent=2))

plotting.equity_curves({k: monthly[k] for k in ["Trend (TSMOM)", "Macro regime (prior)", "Combined signals", "60/40"]},
                       "Signal sleeves vs 60/40 (excess of T-bills, net of costs)", OUT / "equity_curves.png")
plotting.drawdowns({k: monthly[k] for k in ["Combined signals", "Trend (TSMOM)", "60/40"]}, "Drawdowns", OUT / "drawdowns.png")
plotting.equity_curves(daily, "SPY volatility timing (daily, net of costs)", OUT / "vol_timing.png")

pd.set_option("display.width", 200)
print(tbl.round(2).to_string())
print()
print(vtbl.round(2).to_string())
print()
print(costs.round(2).to_string())
print()
print(crisis.round(3).to_string())
print()
print(sleeves.corr().round(2).to_string())
print(grid_tbl[["Sharpe", "MaxDD"]].round(2).to_string())
