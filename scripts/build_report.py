"""Build a single-page HTML research summary from the results folder."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def csv(rel, **kw):
    return pd.read_csv(R / rel, index_col=0, **kw)


def svg_lines(df: pd.DataFrame, series: list[tuple[str, str, str]], log=True, height=220, pct_axis=False, start=None):
    """series: (column, label, css var). Returns an inline SVG drawn to scale."""
    df = df.copy()
    if start:
        df = df.loc[start:]
    W, H, L, Rm, T, B = 640, height, 46, 12, 12, 26
    vals = {}
    for c, _, _ in series:
        s = df[c]
        if not pct_axis:
            s = (1 + s.fillna(0)).cumprod()
            s = np.log(s) if log else s
        vals[c] = s
    allv = pd.concat(vals.values())
    lo, hi = float(allv.min()), float(allv.max())
    pad = (hi - lo) * 0.05
    lo, hi = lo - pad, hi + pad
    x0, x1 = df.index[0], df.index[-1]

    def X(t):
        return L + (t - x0) / (x1 - x0) * (W - L - Rm)

    def Y(v):
        return T + (hi - v) / (hi - lo) * (H - T - B)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart">']
    # y ticks
    if pct_axis:
        ticks = [v for v in np.arange(-0.6, 0.01, 0.1) if lo <= v <= hi]
        lab = [f"{v:.0%}" for v in ticks]
    else:
        cand = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]
        ticks = [np.log(c) for c in cand if lo <= np.log(c) <= hi]
        lab = [f"{c:g}×" for c in cand if lo <= np.log(c) <= hi]
    for v, t in zip(ticks, lab):
        y = Y(v)
        parts.append(f'<line x1="{L}" x2="{W-Rm}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{L-6}" y="{y+3:.1f}" text-anchor="end" class="tick">{t}</text>')
    for yr in range(x0.year + 1, x1.year + 1):
        if yr % 4:
            continue
        t = pd.Timestamp(f"{yr}-01-01")
        parts.append(f'<text x="{X(t):.1f}" y="{H-8}" text-anchor="middle" class="tick">{yr}</text>')
    for c, label, var in series:
        s = vals[c].dropna()
        step = max(1, len(s) // 400)
        s = s.iloc[::step]
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in s.items())
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var({var})" stroke-width="1.6" stroke-linejoin="round"/>')
    parts.append("</svg>")
    legend = "".join(f'<span><i style="background:var({v})"></i>{html.escape(l)}</span>' for _, l, v in series)
    return "".join(parts) + f'<div class="legend">{legend}</div>'


def table(df: pd.DataFrame, fmt: dict[str, str], highlight: str | None = None):
    cols = list(fmt)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in ["", *[fmt_label(c) for c in cols]])
    rows = []
    for i, r in df.iterrows():
        cls = ' class="hl"' if i == highlight else ""
        cells = "".join(f"<td>{fmt[c].format(r[c]) if pd.notna(r[c]) else '–'}</td>" for c in cols)
        rows.append(f"<tr{cls}><th>{html.escape(i)}</th>{cells}</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def fmt_label(c):
    return {"Cost p.a. (2010-19 calm)": "Calm cost/yr", "Saved GFC 2007-09": "Saved GFC", "Saved COVID 2020": "Saved COVID",
            "Saved Inflation 2022": "Saved 2022", "Avg DD saved / calm cost": "Saved ÷ cost"}.get(c, c)


m = csv("01_signals/monthly_returns.csv", parse_dates=True)
s1 = csv("01_signals/summary_monthly.csv")
cr1 = csv("01_signals/crisis_returns.csv")
s1 = s1.join(cr1)
p = csv("02_premia/premia_summary.csv")
a = csv("02_premia/allocation_summary.csv")
pr = csv("02_premia/portfolio_returns.csv", parse_dates=True)
o = csv("03_overlays/overlay_report.csv")
od = csv("03_overlays/overlay_daily_returns.csv", parse_dates=True)
sig = json.loads((R / "01_signals/latest_signals.json").read_text())
hq = json.loads((R / "03_overlays/latest_hedge_quotes.json").read_text())
alloc = json.loads((R / "02_premia/latest_allocation.json").read_text())

dd = {}
for c in ["Unhedged SPY", "Zero-cost collar 95/x 1m", "Vol-target overlay (no options)"]:
    w = (1 + od[c]).cumprod()
    dd[c] = w / w.cummax() - 1
dd = pd.DataFrame(dd).resample("W").min()

pct = "{:.1%}"
num = "{:.2f}"
chart1 = svg_lines(m, [("Combined signals", "Combined signals", "--c1"), ("Trend (TSMOM)", "Trend", "--c2"), ("60/40", "60/40", "--c3")], start="2004-01-31")
chart2 = svg_lines(pr, [("Style risk budget (25% each)", "Style risk budget", "--c1"), ("Equal risk contribution", "Equal risk contribution", "--c2"), ("Equal weight", "Equal weight", "--c3")])
chart3 = svg_lines(dd, [("Unhedged SPY", "Unhedged SPY", "--c3"), ("Zero-cost collar 95/x 1m", "Zero-cost collar", "--c2"), ("Vol-target overlay (no options)", "Vol-target overlay", "--c1")], pct_axis=True, start="2000-01-01")

trend_chips = "".join(
    f'<span class="chip {"up" if v > 0.25 else "dn" if v < -0.25 else "flat"}">{k} {v:+.2f}</span>'
    for k, v in sig["trend_score"].items() if v == v
)
q95 = hq["quotes"]["95% 1m put"]
w_style = alloc["weights"]["Style risk budget (25% each)"]
w_top = ", ".join(f"{k} {v:.0%}" for k, v in sorted(w_style.items(), key=lambda kv: -kv[1])[:3])

T1 = table(s1.loc[["Combined signals", "Trend (TSMOM)", "Macro ridge (walk-fwd)", "Macro regime (prior)", "60/40"]],
           {"Sharpe": num, "MaxDD": pct, "GFC": pct, "COVID": pct, "2022": pct}, highlight="Combined signals")
T2 = table(p.sort_values("Sharpe", ascending=False), {"Sharpe": num, "MaxDD": pct, "Turnover/yr": num})
T2b = table(a, {"Sharpe": num, "MaxDD": pct, "CAGR": pct}, highlight="Style risk budget (25% each)")
T3 = table(o.loc[["Unhedged SPY", "Put 95% 1m", "Put spread 95/85 1m", "Zero-cost collar 95/x 1m", "Trend-gated put 95% 1m", "Vol-target overlay (no options)"]],
           {"CAGR": pct, "MaxDD": pct, "Cost p.a. (2010-19 calm)": pct, "Saved GFC 2007-09": pct, "Saved COVID 2020": pct, "Saved Inflation 2022": pct, "Avg DD saved / calm cost": "{:.1f}"},
           highlight="Vol-target overlay (no options)")

page = f"""<title>Multi-Asset Quant Lab</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: single research-note column, 760px measure; tables and charts full column width */
:root {{
  --bg: #f5f7f8; --surface: #ffffff; --ink: #17222b; --muted: #5a6872; --rule: #d9e0e4;
  --c1: #0b6e7f; --c2: #b5651d; --c3: #8a959c; --up: #1d7a46; --dn: #b23a3a;
  --display: "Newsreader", Georgia, serif; --body: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f1519; --surface: #161e23; --ink: #e3e9ec; --muted: #93a1a9; --rule: #2a363d;
  --c1: #4fb8c9; --c2: #e0954f; --c3: #6f7c84; --up: #5cc48a; --dn: #e07a7a; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #0f1519; --surface: #161e23; --ink: #e3e9ec; --muted: #93a1a9; --rule: #2a363d;
  --c1: #4fb8c9; --c2: #e0954f; --c3: #6f7c84; --up: #5cc48a; --dn: #e07a7a; color-scheme: dark }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.6 var(--body); }}
main {{ max-width: 760px; margin: 0 auto; padding-block: 32px 64px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 40px; }}
header {{ display: grid; gap: 10px; }}
.eyebrow {{ font: 500 11px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
h1 {{ font: 600 clamp(30px, 6vw, 42px)/1.1 var(--display); margin: 0; text-wrap: balance; }}
h2 {{ font: 600 24px/1.2 var(--display); margin: 0; text-wrap: balance; }}
p {{ margin: 0; max-width: 65ch; }}
.lede {{ color: var(--muted); font-size: 16px; }}
section {{ display: grid; gap: 16px; }}
.now {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 16px; display: grid; gap: 12px; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
.stat b {{ display: block; font: 500 20px/1.2 var(--mono); font-variant-numeric: tabular-nums; }}
.stat span {{ font-size: 12px; color: var(--muted); }}
.chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.chip {{ font: 500 12px/1 var(--mono); padding: 5px 7px; border-radius: 3px; border: 1px solid var(--rule); }}
.chip.up {{ color: var(--up); border-color: var(--up); }} .chip.dn {{ color: var(--dn); border-color: var(--dn); }} .chip.flat {{ color: var(--muted); }}
.chart {{ width: 100%; height: auto; display: block; }}
.chart .grid {{ stroke: var(--rule); stroke-width: 1; }}
.chart .tick {{ fill: var(--muted); font: 10px var(--mono); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: var(--muted); }}
.legend i {{ display: inline-block; width: 14px; height: 3px; margin-right: 6px; vertical-align: middle; }}
.tw {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ padding: 7px 8px; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }}
thead th {{ font: 500 11px/1.2 var(--mono); color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }}
tbody th {{ text-align: left; font-weight: 500; }}
td {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
tr.hl th, tr.hl td {{ color: var(--c1); font-weight: 600; }}
ul {{ margin: 0; padding-left: 18px; display: grid; gap: 6px; max-width: 65ch; }}
.fine {{ font-size: 13px; color: var(--muted); }}
code {{ font: 13px var(--mono); }}
</style>
<main>
<header>
  <div class="eyebrow">Research portfolio · Hrush Ambekar · data to {sig['as_of']}</div>
  <h1>Multi-Asset Quant Lab</h1>
  <p class="lede">Four connected studies on public data: signal research, a risk premia toolkit, option protection sized against drawdown saved, and a dashboard with an LLM analyst grounded in these results. All returns are net of costs and in excess of T-bills unless stated.</p>
</header>

<div class="now">
  <div class="eyebrow">Where the models sit today</div>
  <div class="stats">
    <div class="stat"><b>{sig['vix']:.1f}</b><span>VIX</span></div>
    <div class="stat"><b>{sig['vix_vix3m']:.2f}</b><span>VIX / VIX3M ({'contango' if sig['vix_vix3m'] < 1 else 'backwardation'})</span></div>
    <div class="stat"><b>{q95['premium_pct_spot']:.2%}</b><span>1m 95% S&amp;P put, of spot</span></div>
    <div class="stat"><b>{q95['annualised_cost_if_rolled']:.1%}</b><span>same put, rolled for a year</span></div>
  </div>
  <div class="chips">{trend_chips}</div>
  <p class="fine">Trend score by ETF, −1 to +1. Largest premia weights: {w_top}.</p>
</div>

<section>
  <div class="eyebrow">Module 1 · 2004–2026, monthly</div>
  <h2>Signal research</h2>
  <p>Trend, macro and volatility signals on 12 multi-asset ETFs. Parameters are fixed ex ante or fitted walk-forward, and the trend grid is scored with a deflated Sharpe ratio.</p>
  {chart1}
  {T1}
  <ul>
    <li>The sleeves are nearly uncorrelated, so the blend reaches Sharpe {s1.loc['Combined signals','Sharpe']:.2f} against {s1.loc['60/40','Sharpe']:.2f} for 60/40.</li>
    <li>Trend is the crisis engine: +{s1.loc['Trend (TSMOM)','GFC']:.0%} through the GFC and +{s1.loc['Trend (TSMOM)','2022']:.0%} in 2022. Its deflated Sharpe of {s1.loc['Trend (TSMOM)','DSR']:.2f} says the standalone edge is not statistically strong.</li>
    <li>The hand-set macro prior barely works. The fitted ridge version is better out of sample but lost money in 2022.</li>
  </ul>
</section>

<section>
  <div class="eyebrow">Module 2 · 2003–2026, monthly</div>
  <h2>Risk premia toolkit</h2>
  <p>Dollar-neutral carry, momentum, value and defensive premia across G10 FX, US sectors and Treasury duration, each scaled to 10% volatility, then combined with a risk-budget solver.</p>
  {chart2}
  {T2b}
  {T2}
  <p class="fine">Budgeting risk equally by style beats budgeting by stream, because momentum and value offset each other. Carry has paid. Sector value has been the clear loser since 2003.</p>
</section>

<section>
  <div class="eyebrow">Module 3 · 1993–2026, daily, model-priced</div>
  <h2>Option overlays sized against drawdown saved</h2>
  <p>Puts, spreads and collars on the S&amp;P 500, priced from VIX with a parametric skew and marked every day. Cost is measured in the calm 2010–19 market; savings are peak-to-trough drawdowns in each stress episode.</p>
  {chart3}
  {T3}
  <ul>
    <li>Rolling 1-month puts deepened the worst drawdown. Premium bleed in the 2000–02 bear and expensive re-strikes at VIX 60 outweighed the payoffs.</li>
    <li>Puts pay in gap crashes like COVID, much less in slow grinds like 2008, and not at all in a rates-led sell-off like 2022.</li>
    <li>A volatility-target overlay saved the most drawdown per unit of cost. The practical design is a vol-target core with convex puts funded by calls or gated by trend.</li>
  </ul>
  <p class="fine">Skew is the key assumption and is stressed ±30% in the repo. Model prices, not dealer quotes.</p>
</section>

<section>
  <div class="eyebrow">Module 4</div>
  <h2>Dashboard with an embedded LLM analyst</h2>
  <p>A Streamlit dashboard shows every module, with a live scenario repricer for the hedges. The analyst tab runs Claude in a tool-use loop over seven functions that read these results or reprice options. It may only quote numbers that come back from a tool, and each answer shows the tool calls behind it. Without an API key it falls back to deterministic routing over the same tools.</p>
  <p class="fine">Example: “What if the S&amp;P drops 20% and VIX goes to 50?” returns the unhedged loss, the put’s gain and the net loss, with the repricing inputs attached.</p>
</section>

<section>
  <div class="eyebrow">Limits</div>
  <ul class="fine">
    <li>ETF proxies start 2002–07, giving about 22 years and three real crises. Short ETF positions stand in for futures.</li>
    <li>Options are Black–Scholes with a parametric skew, not traded chains.</li>
    <li>OECD 3-month rates proxy for FX forward points. This is backtested research, not a live record.</li>
  </ul>
</section>
</main>
"""
(R / "report.html").write_text(page)
print("wrote", R / "report.html", len(page) // 1024, "KB")
