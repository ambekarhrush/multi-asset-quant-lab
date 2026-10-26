"""Module 3 - Options-based portfolio protection, sized against drawdown saved.

Compares protective puts, put spreads, collars (fixed and zero-cost), signal-gated
puts and a non-option volatility overlay on SPY, 1993-today, and measures
each one by what it costs in calm years against what it saves in 2008, 2020, 2022.
Then sizes the put hedge (strike x hedge ratio) to hit a max-drawdown target at
minimum cost, and stresses the skew assumption.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab.core import backtest as bt, data, metrics, panel, plotting  # noqa: E402
from quantlab.overlays import options as op  # noqa: E402
from quantlab.signals import volatility  # noqa: E402

OUT = ROOT / "results" / "03_overlays"
OUT.mkdir(parents=True, exist_ok=True)

px = panel.daily_prices()
spx = px["^GSPC"].dropna()
eq = px["SPY"].dropna()
vix = data.fred("VIXCLS")
idx = spx.index.intersection(eq.index)
vix = vix.reindex(idx).ffill()
rf = (data.fred("DTB3") / 100).reindex(idx.union(data.fred("DTB3").index)).ffill().reindex(idx)
args = dict(spx=spx.loc[idx], equity_tr=eq.loc[idx], vix=vix, rf_annual=rf)

L = op.Leg
trend_gate = (spx < spx.rolling(210).mean())  # hedge only when S&P is below its ~10m average
cheap_gate = (vix < vix.rolling(252).median())  # hedge only when protection is cheap

strategies = [
    op.Overlay("Unhedged SPY", [], hedge_ratio=0.0),
    op.Overlay("Put 95% 1m", [L("put", 0.95, 1)]),
    op.Overlay("Put 90% 1m", [L("put", 0.90, 1)]),
    op.Overlay("Put 95% 3m", [L("put", 0.95, 1)], tenor_months=3),
    op.Overlay("Put spread 95/85 1m", [L("put", 0.95, 1), L("put", 0.85, -1)]),
    op.Overlay("Collar 95/105 1m", [L("put", 0.95, 1), L("call", 1.05, -1)]),
    op.Overlay("Zero-cost collar 95/x 1m", [L("put", 0.95, 1)], zero_cost_call=True),
    op.Overlay("Trend-gated put 95% 1m", [L("put", 0.95, 1)], gate=trend_gate),
    op.Overlay("Cheap-vol-gated put 95% 3m", [L("put", 0.95, 1)], tenor_months=3, gate=cheap_gate),
]

t0 = time.time()
sims = {s.name: op.simulate(s, **args) for s in strategies}
print(f"simulated {len(sims)} overlays in {time.time() - t0:.1f}s")
rets = {k: v["ret"] for k, v in sims.items()}
base = rets["Unhedged SPY"]

# non-option volatility overlay: scale equity exposure by 15% / forecast vol
spx_r = spx.pct_change(fill_method=None)
vt = volatility.vol_target_signal(spx_r.loc[idx], target=0.15, cap=1.0)
eq_r = eq.loc[idx].pct_change(fill_method=None)
rf_d = rf / 252
vt_res = bt.run(vt.to_frame("SPY"), (eq_r - rf_d).to_frame("SPY"), cost_bps=1, ppy=252)
rets["Vol-target overlay (no options)"] = (vt_res.returns + rf_d.reindex(vt_res.returns.index)).reindex(base.index).fillna(0)

report = op.hedge_report(base, {k: v for k, v in rets.items()})
report.round(4).to_csv(OUT / "overlay_report.csv")

# ---------------------------------------------------------------- sizing
grid = []
for m in (0.90, 0.95, 1.00):
    for h in (0.25, 0.5, 0.75, 1.0):
        r = op.simulate(op.Overlay("x", [L("put", m, 1)], hedge_ratio=h), **args)["ret"]
        row = op.hedge_report(base, {"x": r}).iloc[0]
        worst = min(row[f"DD {e}"] for e in op.EPISODES)
        grid.append({"strike": m, "hedge_ratio": h, "Worst episode DD": worst, "Full-sample MaxDD": row["MaxDD"],
                     "Calm cost p.a.": row["Cost p.a. (2010-19 calm)"], "CAGR": row["CAGR"]})
grid = pd.DataFrame(grid)
grid.round(4).to_csv(OUT / "put_sizing_grid.csv", index=False)
# Sizing rule: cheapest (calm-year cost) put programme whose worst drawdown across the
# three stress episodes stays inside the target.
sizing = {}
base_worst = min(op.episode_drawdown(base, a, z) for a, z in op.EPISODES.values())
for target in (-0.50, -0.45, -0.40):
    ok = grid[grid["Worst episode DD"] >= target]
    sizing[str(target)] = ok.sort_values("Calm cost p.a.").iloc[0].to_dict() if len(ok) else None
sizing["unhedged_worst_episode_DD"] = base_worst

# ---------------------------------------------------------------- skew stress
stress = {}
for mult in (0.7, 1.0, 1.3):
    r = op.simulate(op.Overlay("x", [L("put", 0.95, 1)]), skew_mult=mult, **args)["ret"]
    row = op.hedge_report(base, {"x": r}).iloc[0]
    stress[f"skew x{mult}"] = row[["CAGR", "MaxDD", "Cost p.a. (2010-19 calm)", "Saved GFC 2007-09", "Saved COVID 2020", "Saved Inflation 2022"]]
stress = pd.DataFrame(stress).T
stress.round(4).to_csv(OUT / "skew_stress.csv")

# ---------------------------------------------------------------- latest quote
S, v, r0 = float(spx.iloc[-1]), float(vix.iloc[-1]), float(rf.iloc[-1])
T = 1 / 12
quotes = {}
for m in (0.90, 0.95, 1.00):
    K = m * S
    sig = float(op.skew_vol(v, S, K, T, r0))
    p = float(op.bs_price(S, K, T, r0, op.DIV_YIELD, sig, "put"))
    quotes[f"{int(m*100)}% 1m put"] = {"strike": round(K, 1), "model_vol": round(sig, 4), "premium_pct_spot": round(p / S, 4),
                                        "annualised_cost_if_rolled": round(12 * p / S, 4)}
(OUT / "latest_hedge_quotes.json").write_text(json.dumps({"as_of": str(idx[-1].date()), "spx": S, "vix": v, "quotes": quotes, "sizing": sizing}, indent=2, default=float))
pd.DataFrame(rets).to_csv(OUT / "overlay_daily_returns.csv")

sel = ["Unhedged SPY", "Put 95% 1m", "Put spread 95/85 1m", "Zero-cost collar 95/x 1m", "Trend-gated put 95% 1m", "Vol-target overlay (no options)"]
plotting.equity_curves({k: rets[k] for k in sel}, "SPY with protection overlays (total return, model-priced options)", OUT / "overlay_curves.png")
plotting.drawdowns({k: rets[k] for k in ["Unhedged SPY", "Put 95% 1m", "Zero-cost collar 95/x 1m", "Vol-target overlay (no options)"]}, "Drawdowns", OUT / "overlay_drawdowns.png")

pd.set_option("display.width", 250)
cols = ["CAGR", "Sharpe", "MaxDD", "Cost p.a. (2010-19 calm)", "Saved GFC 2007-09", "Saved COVID 2020", "Saved Inflation 2022", "Avg DD saved / calm cost"]
print(report[cols].round(3).to_string()); print()
print(grid.round(3).to_string()); print()
print(json.dumps(sizing, indent=1, default=float)); print()
print(stress.round(3).to_string()); print()
print(json.dumps(quotes, indent=1))
