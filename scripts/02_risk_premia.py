"""Module 2 - Risk premia toolkit: carry, momentum, value, defensive + risk budgeting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab.core import backtest as bt, metrics, plotting, universe as u  # noqa: E402
from quantlab.premia import factors as f  # noqa: E402

OUT = ROOT / "results" / "02_premia"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 0.10


def scale_to(r: pd.Series, target=TARGET, lookback=36) -> pd.Series:
    vol = r.rolling(lookback, min_periods=24).std().shift(1) * np.sqrt(12)
    return r * (target / vol).clip(upper=4)


raw, turn = {}, {}
for name, (w, r) in {**f.fx_premia(), **f.sector_premia()}.items():
    cost = 3.0 if name.startswith("FX") else u.COST_BPS
    res = bt.run(w, r, cost_bps=cost, ppy=12)
    raw[name] = res.returns
    turn[name] = res.turnover
w_rc, r_rc = f.rates_carry()
res_rc = bt.run(w_rc, r_rc, cost_bps=1, ppy=12)
raw["Rates carry"], turn["Rates carry"] = res_rc.returns, res_rc.turnover

premia = pd.DataFrame({k: scale_to(v) for k, v in raw.items()}).loc["2003-01-31":]
premia = premia.where(premia.notna())

# ----------------------------------------------------------- risk budgeting
STYLE = {
    "FX carry": "Carry", "Rates carry": "Carry",
    "FX momentum": "Momentum", "Equity momentum": "Momentum",
    "FX value": "Value", "Equity value": "Value",
    "Defensive (BAB)": "Defensive",
}
# Equal risk to each *style*, split equally within style
style_n = pd.Series(STYLE).value_counts()
style_budget = {k: 0.25 / style_n[v] for k, v in STYLE.items()}

allocs = {
    "Equal weight": f.allocate(premia, "equal"),
    "Inverse vol": f.allocate(premia, "inv_vol"),
    "Equal risk contribution": f.allocate(premia, "erc"),
    "Style risk budget (25% each)": f.allocate(premia, "erc", budget=style_budget),
}
port = {}
for k, W in allocs.items():
    r = (W.shift(1) * premia.fillna(0)).sum(axis=1)
    r = r.loc[W.abs().sum(axis=1).gt(0).idxmax():].iloc[1:]
    port[k] = scale_to(r)
port = {k: v.loc["2006-01-31":] for k, v in port.items()}

# ex-post risk contribution check for the style budget portfolio (last 5y)
W_last = allocs["Style risk budget (25% each)"].iloc[-1]
cov5 = premia.iloc[-60:].dropna(axis=1, how="any").cov()
wv = W_last[cov5.columns].values
rc = pd.Series(f.risk_contributions(wv, cov5.values), index=cov5.columns)
rc_style = rc.groupby(pd.Series(STYLE)).sum()

tbl_p = metrics.table({k: v.dropna() for k, v in premia.items()}, 12)
tbl_p["Turnover/yr"] = [turn[k].loc["2003":].mean() * 12 for k in tbl_p.index]
tbl_a = metrics.table(port, 12)
corr = premia.corr().round(2)

crisis = {}
for name, (a, b) in {"GFC": ("2007-10-31", "2009-03-31"), "COVID": ("2020-01-31", "2020-04-30"), "2022": ("2021-12-31", "2022-10-31")}.items():
    crisis[name] = {k: metrics.window_return(v.fillna(0), a, b) for k, v in {**premia, **port}.items()}
crisis = pd.DataFrame(crisis)

tbl_p.round(3).to_csv(OUT / "premia_summary.csv")
tbl_a.round(3).to_csv(OUT / "allocation_summary.csv")
corr.to_csv(OUT / "premia_correlation.csv")
crisis.round(3).to_csv(OUT / "crisis_returns.csv")
premia.to_csv(OUT / "premia_returns.csv")
pd.DataFrame(port).to_csv(OUT / "portfolio_returns.csv")
(OUT / "latest_allocation.json").write_text(json.dumps({
    "as_of": str(pd.Timestamp(premia.index[-1]).date()) + " (month-to-date)",
    "weights": {k: W.iloc[-1].round(3).to_dict() for k, W in allocs.items()},
    "risk_contribution_style_budget": rc.round(3).to_dict(),
    "risk_contribution_by_style": rc_style.round(3).to_dict(),
}, indent=2))

plotting.equity_curves({k: premia[k].dropna() for k in premia}, "Individual premia (10% vol target, net of costs)", OUT / "premia_curves.png")
plotting.equity_curves(port, "Risk-budgeted premia portfolios", OUT / "allocation_curves.png")
plotting.bar(rc_style, "Ex-post risk contribution by style (last 5y)", OUT / "risk_contribution.png", pct=True)

pd.set_option("display.width", 200)
print(tbl_p.round(2).to_string()); print()
print(tbl_a.round(2).to_string()); print()
print(corr.to_string()); print()
print(rc_style.round(3).to_string()); print()
print(crisis.round(3).to_string())
