"""Tools the portfolio LLM can call. Each returns plain JSON-serialisable data.

The LLM never sees raw files or invents numbers: every figure it quotes must come
back from one of these functions, which read the research outputs on disk or
reprice options live with the same model used in the backtests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..overlays import options as op

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / rel, index_col=0)


def _json(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text())


def list_strategies() -> dict:
    return {
        "signals": list(_csv("01_signals/summary_monthly.csv").index),
        "vol_timing": list(_csv("01_signals/summary_vol_daily.csv").index),
        "premia": list(_csv("02_premia/premia_summary.csv").index),
        "premia_portfolios": list(_csv("02_premia/allocation_summary.csv").index),
        "overlays": list(_csv("03_overlays/overlay_report.csv").index),
    }


def get_performance(strategy: str) -> dict:
    """Full-sample stats for any strategy name (case-insensitive substring match)."""
    tables = {
        "01_signals/summary_monthly.csv": "monthly signal sleeve",
        "01_signals/summary_vol_daily.csv": "daily vol-timing sleeve",
        "02_premia/premia_summary.csv": "risk premium",
        "02_premia/allocation_summary.csv": "premia portfolio",
        "03_overlays/overlay_report.csv": "option overlay",
    }
    s = strategy.lower()
    for path, kind in tables.items():
        df = _csv(path)
        hit = [i for i in df.index if s in i.lower()]
        if hit:
            row = df.loc[hit[0]].dropna()
            return {"strategy": hit[0], "type": kind, "stats": {k: round(float(v), 4) for k, v in row.items()}}
    return {"error": f"no strategy matching '{strategy}'", "available": list_strategies()}


def get_latest_signals() -> dict:
    return _json("01_signals/latest_signals.json")


def get_premia_allocation() -> dict:
    return _json("02_premia/latest_allocation.json")


def get_hedge_quotes() -> dict:
    return _json("03_overlays/latest_hedge_quotes.json")


def get_crisis_performance(episode: str | None = None) -> dict:
    out = {}
    for name, path in {"signals": "01_signals/crisis_returns.csv", "premia": "02_premia/crisis_returns.csv"}.items():
        df = _csv(path)
        out[name] = (df[[c for c in df.columns if episode is None or episode.lower() in c.lower()]]
                     .round(4).to_dict())
    rep = _csv("03_overlays/overlay_report.csv")
    cols = [c for c in rep.columns if c.startswith(("DD ", "Saved ")) and (episode is None or episode.lower() in c.lower())]
    out["overlays"] = rep[cols].round(4).to_dict()
    return out


def stress_hedge(equity_shock: float, vix_after: float, strike_pct: float = 0.95, days_elapsed: int = 5,
                 hedge_ratio: float = 1.0) -> dict:
    """Instant scenario: P&L of SPY + 1m put at ``strike_pct`` if the index moves by
    ``equity_shock`` (e.g. -0.10) and VIX reprices to ``vix_after`` after ``days_elapsed``."""
    q = get_hedge_quotes()
    S0, v0 = q["spx"], q["vix"]
    r = 0.04
    T0, T1 = 1 / 12, max(1 / 12 - days_elapsed / 365, 1e-6)
    K = strike_pct * S0
    p0 = float(op.bs_price(S0, K, T0, r, op.DIV_YIELD, op.skew_vol(v0, S0, K, T0, r), "put"))
    S1 = S0 * (1 + equity_shock)
    p1 = float(op.bs_price(S1, K, T1, r, op.DIV_YIELD, op.skew_vol(vix_after, S1, K, T1, r), "put"))
    put_pnl = hedge_ratio * (p1 - p0) / S0
    return {
        "inputs": {"spx": S0, "vix_now": v0, "equity_shock": equity_shock, "vix_after": vix_after,
                   "strike": round(K, 1), "days_elapsed": days_elapsed, "hedge_ratio": hedge_ratio},
        "unhedged_pnl": round(equity_shock, 4),
        "put_pnl": round(put_pnl, 4),
        "hedged_pnl": round(equity_shock + put_pnl, 4),
        "premium_paid_pct": round(hedge_ratio * p0 / S0, 4),
        "note": "Black-Scholes with VIX-implied skew; model price, not a dealer quote.",
    }


TOOLS = {
    "list_strategies": (list_strategies, "List every strategy name available in the research results.", {}),
    "get_performance": (get_performance, "Full-sample performance statistics (CAGR, vol, Sharpe, max drawdown, costs...) for one strategy.",
                        {"strategy": {"type": "string", "description": "Strategy name or part of it, e.g. 'trend', 'FX carry', 'collar'"}}),
    "get_latest_signals": (get_latest_signals, "Current trend scores, macro regime z-scores, VIX term structure and vol-timing exposures.", {}),
    "get_premia_allocation": (get_premia_allocation, "Current risk-budget weights across the risk premia and their risk contributions.", {}),
    "get_hedge_quotes": (get_hedge_quotes, "Current model prices of 1-month S&P 500 puts and the cheapest put programme for each drawdown target.", {}),
    "get_crisis_performance": (get_crisis_performance, "Returns and drawdowns of every strategy in the GFC, COVID and 2022 sell-offs.",
                               {"episode": {"type": "string", "description": "Optional filter: 'GFC', 'COVID' or '2022'"}}),
    "stress_hedge": (stress_hedge, "Reprice a protective put under an instantaneous equity and VIX shock.",
                     {"equity_shock": {"type": "number", "description": "Index move as a decimal, e.g. -0.10"},
                      "vix_after": {"type": "number", "description": "VIX level after the shock"},
                      "strike_pct": {"type": "number", "description": "Put strike as fraction of spot, default 0.95"},
                      "days_elapsed": {"type": "integer", "description": "Days between purchase and shock, default 5"},
                      "hedge_ratio": {"type": "number", "description": "Fraction of notional hedged, default 1.0"}}),
}

REQUIRED = {"get_performance": ["strategy"], "stress_hedge": ["equity_shock", "vix_after"]}


def anthropic_tool_specs() -> list[dict]:
    return [
        {"name": n, "description": d, "input_schema": {"type": "object", "properties": p, "required": REQUIRED.get(n, [])}}
        for n, (_, d, p) in TOOLS.items()
    ]


def call(name: str, args: dict) -> dict:
    fn = TOOLS[name][0]
    try:
        return fn(**(args or {}))
    except Exception as e:  # surfaced to the model so it can recover
        return {"error": f"{type(e).__name__}: {e}"}
