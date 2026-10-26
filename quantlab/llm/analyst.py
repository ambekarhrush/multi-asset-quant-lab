"""Portfolio analyst: an LLM grounded in the research outputs through tool calls.

Design choices that matter for a live portfolio setting:
  * Grounding: the system prompt forbids stating any number not returned by a tool,
    and every tool reads the same results the dashboard displays.
  * Auditability: every tool call and result is kept in ``trace`` so the answer can
    be checked line by line.
  * Graceful degradation: with no API key the analyst runs a deterministic
    rule-based path over the same tools, so the dashboard never breaks.

Set ANTHROPIC_API_KEY (and optionally QUANTLAB_MODEL) to enable the LLM path.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from . import tools

DEFAULT_MODEL = os.environ.get("QUANTLAB_MODEL", "claude-sonnet-5-5")

SYSTEM = """You are a quantitative portfolio analyst embedded in a multi-asset research dashboard.
Rules:
- Every number you state must come from a tool result in this conversation. If a tool does not give it, say you do not have it.
- Call tools before answering questions about performance, positions, signals, hedges or scenarios.
- Be concise: lead with the answer, then 2-4 supporting points. Use percentages with one decimal.
- Flag model limitations when relevant (ETF proxies, model-priced options, backtest not live).
- You do not give personalised investment advice; you explain what the research shows."""


@dataclass
class Answer:
    text: str
    trace: list[dict] = field(default_factory=list)
    mode: str = "llm"


class PortfolioAnalyst:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, max_turns: int = 6):
        self.model = model
        self.max_turns = max_turns
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        if key:
            import anthropic

            self.client = anthropic.Anthropic(api_key=key)

    @property
    def online(self) -> bool:
        return self.client is not None

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        if not self.online:
            return offline_answer(question)
        return self._ask_llm(question, history or [])

    def _ask_llm(self, question: str, history: list[dict]) -> Answer:
        messages = history + [{"role": "user", "content": question}]
        trace = []
        for _ in range(self.max_turns):
            resp = self.client.messages.create(
                model=self.model, max_tokens=1200, system=SYSTEM,
                tools=tools.anthropic_tool_specs(), messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return Answer(text, trace, "llm")
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = tools.call(b.name, b.input)
                    trace.append({"tool": b.name, "input": b.input, "output": out})
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=float)})
            messages.append({"role": "user", "content": results})
        return Answer("Stopped after the maximum number of tool calls.", trace, "llm")

    def daily_brief(self) -> Answer:
        q = ("Write a morning brief for the portfolio: current trend and macro signals, VIX term structure, "
             "the risk premia allocation, and what a 95% put costs today. Under 150 words.")
        return self.ask(q)


# ------------------------------------------------------------------ offline path
def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def offline_answer(question: str) -> Answer:
    """Deterministic routing over the same tools. Used when no API key is set and in tests."""
    q = question.lower()
    trace = []

    def use(name, **kw):
        out = tools.call(name, kw)
        trace.append({"tool": name, "input": kw, "output": out})
        return out

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", q)
    if any(w in q for w in ("stress", "scenario", "what if", "crash", "fall", "drop", "sell-off", "selloff")) and m:
        shock = -abs(float(m.group(1))) / 100
        vm = re.search(r"vix\D{0,12}(\d{2,3})", q)
        vix_after = float(vm.group(1)) if vm else max(20.0, 20 + abs(shock) * 250)
        r = use("stress_hedge", equity_shock=shock, vix_after=vix_after)
        text = (f"If the S&P falls {_pct(-shock)} and VIX goes to {vix_after:.0f}, an unhedged book loses {_pct(-r['unhedged_pnl'])}. "
                f"A 1m {round(r['inputs']['strike'] / r['inputs']['spx'] * 100)}% put adds {_pct(r['put_pnl'])}, "
                f"so the hedged loss is {_pct(-r['hedged_pnl'])}. The put cost {_pct(r['premium_paid_pct'])} of notional. ({r['note']})")
        return Answer(text, trace, "offline")

    if any(w in q for w in ("hedge", "put", "collar", "protect")):
        hq = use("get_hedge_quotes")
        p95 = hq["quotes"]["95% 1m put"]
        rep = use("get_performance", strategy="zero-cost collar")["stats"]
        text = (f"As of {hq['as_of']} (VIX {hq['vix']:.1f}), a 1m 95% S&P put costs {_pct(p95['premium_pct_spot'])} of spot, "
                f"about {_pct(p95['annualised_cost_if_rolled'])} a year if rolled. In the backtest the zero-cost 95% collar cut the worst drawdown "
                f"to {_pct(rep['MaxDD'])} with a calm-year cost of {_pct(rep['Cost p.a. (2010-19 calm)'])}.")
        return Answer(text, trace, "offline")

    if any(w in q for w in ("premia", "carry", "value", "defensive", "allocation", "risk budget")):
        a = use("get_premia_allocation")
        w = a["weights"]["Style risk budget (25% each)"]
        top = sorted(w.items(), key=lambda kv: -kv[1])[:3]
        rc = a["risk_contribution_by_style"]
        text = (f"Current style-budget weights (as of {a['as_of']}): " + ", ".join(f"{k} {_pct(v)}" for k, v in top) +
                ". Ex-post risk contribution by style: " + ", ".join(f"{k} {_pct(v)}" for k, v in rc.items()) + ".")
        return Answer(text, trace, "offline")

    if any(w in q for w in ("crisis", "2008", "gfc", "covid", "2020", "2022")):
        ep = "GFC" if ("2008" in q or "gfc" in q) else "COVID" if ("covid" in q or "2020" in q) else "2022" if "2022" in q else None
        c = use("get_crisis_performance", episode=ep)
        sig = c["signals"]
        col = next(iter(sig))
        text = f"{col}: " + ", ".join(f"{k} {_pct(v)}" for k, v in sig[col].items()) + "."
        return Answer(text, trace, "offline")

    s = use("get_latest_signals")
    longs = [k for k, v in s["trend_score"].items() if v and v > 0.5]
    shorts = [k for k, v in s["trend_score"].items() if v and v < -0.5]
    text = (f"Signals as of {s['as_of']}: trend is long {', '.join(longs) or 'nothing strongly'}"
            f"{' and short ' + ', '.join(shorts) if shorts else ''}. VIX {s['vix']:.1f}, VIX/VIX3M {s['vix_vix3m']:.2f} "
            f"({'contango, risk-on' if s['vix_vix3m'] < 1 else 'backwardation, risk-off'}). "
            f"Macro z-scores: " + ", ".join(f"{k} {v:+.1f}" for k, v in s["macro_states_z"].items()) + ".")
    return Answer(text, trace, "offline")
