"""Live portfolio dashboard with an embedded, tool-grounded LLM analyst.

Run:  streamlit run dashboard/app.py
LLM:  export ANTHROPIC_API_KEY=...   (optional; without it the analyst runs offline)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantlab.llm.analyst import PortfolioAnalyst  # noqa: E402

R = ROOT / "results"
st.set_page_config(page_title="Multi-Asset Quant Lab", layout="wide")


@st.cache_data(ttl=3600)
def load(rel, **kw):
    p = R / rel
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p, index_col=0, **kw)


def curve(df: pd.DataFrame, cols, title):
    eq = (1 + df[cols].fillna(0)).cumprod()
    fig = px.line(eq, log_y=True, title=title)
    fig.update_layout(height=380, legend_title_text="", margin=dict(l=10, r=10, t=40, b=10))
    return fig


st.title("Multi-Asset Quant Lab")
sig = load("01_signals/latest_signals.json")
hq = load("03_overlays/latest_hedge_quotes.json")
c1, c2, c3, c4 = st.columns(4)
c1.metric("VIX", f"{sig['vix']:.1f}")
c2.metric("VIX / VIX3M", f"{sig['vix_vix3m']:.2f}", "contango" if sig["vix_vix3m"] < 1 else "backwardation")
c3.metric("1m 95% put", f"{hq['quotes']['95% 1m put']['premium_pct_spot']:.2%}", "of spot")
c4.metric("Data as of", sig["as_of"])

tab1, tab2, tab3, tab4 = st.tabs(["Signals", "Risk premia", "Option overlays", "Ask the analyst"])

with tab1:
    m = load("01_signals/monthly_returns.csv", parse_dates=True)
    st.plotly_chart(curve(m, ["Trend (TSMOM)", "Macro ridge (walk-fwd)", "Combined signals", "60/40"], "Signal sleeves, excess of T-bills, net of costs"), width="stretch")
    st.dataframe(load("01_signals/summary_monthly.csv").style.format("{:.2f}"), width="stretch")
    a, b = st.columns(2)
    a.subheader("Trend score now")
    a.bar_chart(pd.Series(sig["trend_score"]))
    b.subheader("Macro regime (z-score of 3m change)")
    b.bar_chart(pd.Series(sig["macro_states_z"]))

with tab2:
    p = load("02_premia/portfolio_returns.csv", parse_dates=True)
    st.plotly_chart(curve(p, list(p.columns), "Risk-budgeted premia portfolios (10% vol)"), width="stretch")
    st.dataframe(load("02_premia/premia_summary.csv").style.format("{:.2f}"), width="stretch")
    alloc = load("02_premia/latest_allocation.json")
    st.subheader("Current weights")
    st.dataframe(pd.DataFrame(alloc["weights"]).style.format("{:.1%}"), width="stretch")
    st.subheader("Correlation")
    st.dataframe(load("02_premia/premia_correlation.csv").style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), width="stretch")

with tab3:
    o = load("03_overlays/overlay_daily_returns.csv", parse_dates=True)
    pick = st.multiselect("Overlays", list(o.columns), default=["Unhedged SPY", "Put 95% 1m", "Zero-cost collar 95/x 1m", "Vol-target overlay (no options)"])
    if pick:
        st.plotly_chart(curve(o, pick, "SPY with protection overlays"), width="stretch")
    rep = load("03_overlays/overlay_report.csv")
    st.dataframe(rep.style.format("{:.2%}", subset=[c for c in rep.columns if c not in ("Sharpe", "Avg DD saved / calm cost")]).format("{:.2f}", subset=["Sharpe", "Avg DD saved / calm cost"]), width="stretch")
    st.subheader("Scenario")
    s1, s2, s3 = st.columns(3)
    shock = s1.slider("S&P move", -0.40, 0.10, -0.15, 0.01)
    vix_after = s2.slider("VIX after", 10, 90, 40)
    strike = s3.select_slider("Put strike", [0.90, 0.95, 1.00], 0.95)
    from quantlab.llm.tools import stress_hedge
    res = stress_hedge(shock, vix_after, strike)
    st.write(f"Unhedged **{res['unhedged_pnl']:.1%}**, put **{res['put_pnl']:+.1%}**, hedged **{res['hedged_pnl']:.1%}** (premium {res['premium_paid_pct']:.2%})")

with tab4:
    analyst = PortfolioAnalyst()
    st.caption("Claude with tool access to these results" if analyst.online else "Offline mode: set ANTHROPIC_API_KEY to enable the LLM. Answers below come from deterministic routing over the same tools.")
    if st.button("Generate morning brief"):
        ans = analyst.daily_brief()
        st.markdown(ans.text)
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, msg in st.session_state.chat:
        st.chat_message(role).markdown(msg)
    q = st.chat_input("e.g. What if the S&P drops 20% and VIX hits 50?")
    if q:
        st.chat_message("user").markdown(q)
        ans = analyst.ask(q)
        st.chat_message("assistant").markdown(ans.text)
        with st.expander(f"{len(ans.trace)} tool call(s) behind this answer"):
            st.json(ans.trace)
        st.session_state.chat += [("user", q), ("assistant", ans.text)]
