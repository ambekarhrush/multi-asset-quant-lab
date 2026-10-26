"""README figures, rendered in a light and a dark variant from ``results/`` only.

No network and no data cache needed: every number comes from the CSV/JSON
outputs the three research scripts write. Run after ``run_all.sh``.

    python scripts/build_readme_figures.py

GitHub picks the variant that matches the reader's theme through <picture>.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Categorical slots validated (adjacent CVD and contrast) against GitHub's page
# backgrounds; neutral ink is used for benchmarks so the strategy carries the colour.
THEMES = {
    "light": dict(bg="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e8e7e1",
                  axis="#c3c2b7", band="#f3f2ee", s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
                  s4="#eda100", neg="#e34948", bench="#9a988f", div_mid="#f0efec"),
    "dark": dict(bg="#0d1117", ink="#f0efec", ink2="#c3c2b7", muted="#8b8a84", grid="#21262d",
                 axis="#3d444d", band="#161b22", s1="#3987e5", s2="#d95926", s3="#199e70",
                 s4="#c98500", neg="#e66767", bench="#7d7c76", div_mid="#2a2e35"),
}
CRISES = {"GFC": ("2007-10-31", "2009-03-31"), "COVID": ("2020-01-31", "2020-04-30"),
          "2022": ("2021-12-31", "2022-10-31")}


def setup(t: dict):
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "lines.solid_capstyle": "round",
    })


def title(ax, head: str, sub: str | None, t: dict):
    ax.set_title(head, loc="left", fontsize=13, fontweight="bold", color=t["ink"], pad=24 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=t["ink2"], va="bottom")


def save(fig, name: str, theme: str):
    fig.savefig(OUT / f"{name}_{theme}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def growth(r: pd.Series) -> pd.Series:
    return (1 + r.fillna(0)).cumprod()


def shade_crises(ax, t: dict):
    for a, b in CRISES.values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=t["band"], zorder=0, lw=0)


def end_label(ax, x, y, text, color, t, dy=0.0):
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", fontsize=9.5,
                color=t["ink"], va="center", fontweight="bold")
    ax.plot([x], [y], "o", ms=6, color=color, mec=t["bg"], mew=1.5, zorder=5)


def log_money(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ------------------------------------------------------------------ data
sig_m = pd.read_csv(R / "01_signals/monthly_returns.csv", index_col=0, parse_dates=True)
sig_sum = pd.read_csv(R / "01_signals/summary_monthly.csv", index_col=0)
sig_crisis = pd.read_csv(R / "01_signals/crisis_returns.csv", index_col=0)
grid = pd.read_csv(R / "01_signals/trend_parameter_grid.csv", index_col=0)
prem_sum = pd.read_csv(R / "02_premia/premia_summary.csv", index_col=0)
alloc_sum = pd.read_csv(R / "02_premia/allocation_summary.csv", index_col=0)
prem_corr = pd.read_csv(R / "02_premia/premia_correlation.csv", index_col=0)
alloc_latest = json.loads((R / "02_premia/latest_allocation.json").read_text())
ov = pd.read_csv(R / "03_overlays/overlay_report.csv", index_col=0)
ov_d = pd.read_csv(R / "03_overlays/overlay_daily_returns.csv", index_col=0, parse_dates=True)

STYLE = {"FX carry": "Carry", "Rates carry": "Carry", "FX momentum": "Momentum", "Equity momentum": "Momentum",
         "FX value": "Value", "Equity value": "Value", "Defensive (BAB)": "Defensive"}
OV_SHORT = {
    "Put 95% 1m": "95% put, 1m", "Put 90% 1m": "90% put, 1m", "Put 95% 3m": "95% put, 3m",
    "Put spread 95/85 1m": "95/85 put spread", "Collar 95/105 1m": "95/105 collar",
    "Zero-cost collar 95/x 1m": "Zero-cost collar", "Trend-gated put 95% 1m": "Trend-gated put",
    "Cheap-vol-gated put 95% 3m": "Cheap-vol-gated put", "Vol-target overlay (no options)": "Vol target (no options)",
}


def style_colors(t):
    return {"Carry": t["s1"], "Momentum": t["s2"], "Value": t["s3"], "Defensive": t["s4"]}


# ------------------------------------------------------------------ 01 signals
def fig_signals_hero(t, theme):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    series = [("Combined signals", "Combined", t["s1"], 2.4),
              ("Trend (TSMOM)", "Trend alone", t["s2"], 1.6),
              ("60/40", "60/40", t["bench"], 1.6)]
    shade_crises(ax, t)
    for col, lab, c, lw in series:
        g = growth(sig_m[col])
        ax.plot(g.index, g.values, color=c, lw=lw, zorder=3 if col.startswith("Comb") else 2)
        s = sig_sum.loc[col]
        end_label(ax, g.index[-1], g.iloc[-1], f"{lab}\nSharpe {s.Sharpe:.2f} · max DD {s.MaxDD:.0%}", c, t)
    log_money(ax)
    ax.set_yticks([1, 2, 4, 8])
    ax.set_xlim(sig_m.index[0], sig_m.index[-1] + pd.DateOffset(months=2))
    hi = ax.get_ylim()[1]
    for name, (a, _) in CRISES.items():
        ax.text(pd.Timestamp(a) + pd.DateOffset(days=25), hi * 0.95, name, fontsize=8.5, color=t["muted"], va="top")
    ax.grid(axis="x", visible=False)
    title(ax, "Growth of 1, excess of T-bills, net of costs (2004–2026)",
          "Combined = trend + walk-forward macro + vol-timed equity, inverse-vol weighted at 10% vol", t)
    save(fig, "01_signals_growth", theme)


def fig_signals_crisis(t, theme):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    rows = [("Combined signals", "Combined", t["s1"]), ("Trend (TSMOM)", "Trend", t["s2"]), ("60/40", "60/40", t["bench"])]
    lim = 0.75
    heads = {"GFC": "GFC  Oct 07 – Mar 09", "COVID": "COVID  Feb – Apr 20", "2022": "2022  Jan – Oct"}
    for ax, ep in zip(axes, heads):
        ys = np.arange(len(rows))[::-1]
        for y, (k, _, c) in zip(ys, rows):
            v = sig_crisis.loc[k, ep]
            ax.barh(y, v, color=c, height=0.62)
            ax.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.1%}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=10, color=t["ink"], fontweight="bold")
        ax.axvline(0, color=t["axis"], lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_yticks(ys, [r[1] for r in rows])
        ax.xaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.set_xticks([-0.5, 0, 0.5])
        ax.grid(axis="y", visible=False)
        ax.set_title(heads[ep], loc="left", fontsize=10.5, color=t["ink"], fontweight="bold")
    fig.suptitle("Return through each crisis window", x=0.01, ha="left", fontsize=13, fontweight="bold", color=t["ink"])
    fig.tight_layout()
    save(fig, "01_signals_crisis", theme)


def fig_trend_grid(t, theme):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    g = grid.copy()
    g["h"] = g.index.str.extract(r"h=\((.*?)\)", expand=False).str.replace(" ", "").str.rstrip(",").values
    g["tgt"] = g.index.str.extract(r"t=([\d.]+)", expand=False).astype(float).values
    hs = list(dict.fromkeys(g["h"]))
    head = ("1,3,6,12", 0.1)
    for i, h in enumerate(hs):
        sub = g[g["h"] == h]
        is_head = h == head[0]
        c = t["s1"] if is_head else t["bench"]
        ax.plot(sub.Sharpe.mean(), i, "o", ms=11 if is_head else 9, color=c, mec=t["bg"], mew=2, zorder=3)
        ax.text(sub.Sharpe.mean() + 0.012, i, f"{sub.Sharpe.mean():.2f}" + ("   reported headline" if is_head else ""),
                va="center", fontsize=9.5, color=t["ink"], fontweight="bold" if is_head else "normal")
    ax.set_yticks(range(len(hs)), [f"{h} months" for h in hs])
    ax.set_ylim(-0.6, len(hs) - 0.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("Sharpe ratio, net of costs (both vol targets give the same Sharpe to 2 dp)")
    ax.grid(axis="y", visible=False)
    title(ax, "Every trend variant tested, not just the best one",
          "The headline horizons were fixed ex ante and rank 4th of 5. Deflated Sharpe across the 10-variant grid: 0.69.", t)
    save(fig, "01_trend_grid", theme)


# ------------------------------------------------------------------ 02 premia
def fig_premia_sharpe(t, theme):
    fig, ax = plt.subplots(figsize=(10, 3.9))
    sc = style_colors(t)
    d = prem_sum["Sharpe"].sort_values()
    ys = np.arange(len(d))
    for y, (k, v) in zip(ys, d.items()):
        ax.barh(y, v, color=sc[STYLE[k]], height=0.62)
        ax.text(v + (0.012 if v >= 0 else -0.012), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=10, fontweight="bold", color=t["ink"])
    ax.set_yticks(ys, list(d.index))
    ax.axvline(0, color=t["axis"], lw=1)
    ax.set_xlim(-0.48, 0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Sharpe ratio, net of costs, each premium scaled to 10% vol (2003–2026)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in sc.values()]
    ax.legend(handles, list(sc), loc="lower right", ncol=4, fontsize=9.5, handlelength=1, columnspacing=1.2)
    title(ax, "Carry pays; sector value has been the loser", None, t)
    save(fig, "02_premia_sharpe", theme)


def fig_premia_alloc(t, theme):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.5))
    names = {"Equal weight": "Equal weight", "Inverse vol": "Inverse vol", "Equal risk contribution": "Equal risk",
             "Style risk budget (25% each)": "Style budget"}
    d = alloc_sum.loc[list(names)]
    ys = np.arange(len(d))[::-1]
    for y, (k, r) in zip(ys, d.iterrows()):
        best = k.startswith("Style")
        a1.barh(y, r.Sharpe, color=t["s1"] if best else t["bench"], height=0.62)
        a1.text(r.Sharpe + 0.008, y, f"{r.Sharpe:.2f}   max DD {r.MaxDD:.0%}", va="center", fontsize=9.5,
                color=t["ink"], fontweight="bold" if best else "normal")
    a1.set_yticks(ys, list(names.values()))
    a1.set_xlim(0, 0.6)
    a1.grid(axis="y", visible=False)
    a1.set_title("Sharpe by allocation rule", loc="left", fontsize=11, fontweight="bold", color=t["ink"])

    rc = alloc_latest["risk_contribution_by_style"]
    sc = style_colors(t)
    order = ["Carry", "Momentum", "Value", "Defensive"]
    ys2 = np.arange(len(order))[::-1]
    for y, k in zip(ys2, order):
        a2.barh(y, rc[k], color=sc[k], height=0.62)
        a2.text(rc[k] + 0.008, y, f"{rc[k]:.0%}", va="center", fontsize=10, fontweight="bold", color=t["ink"])
    a2.axvline(0.25, color=t["ink2"], lw=1.2)
    a2.text(0.255, -0.75, "25% budget", fontsize=9, color=t["ink2"], ha="left", va="center")
    a2.set_ylim(-1.0, len(order) - 0.5)
    a1.set_ylim(-1.0, len(names) - 0.5)
    a2.set_yticks(ys2, order)
    a2.set_xlim(0, 0.42)
    a2.xaxis.set_major_formatter(PercentFormatter(1, 0))
    a2.grid(axis="y", visible=False)
    a2.set_title("Realised risk share, last 5 years", loc="left", fontsize=11, fontweight="bold", color=t["ink"])
    save(fig, "02_premia_allocation", theme)


def fig_premia_corr(t, theme):
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    cmap = LinearSegmentedColormap.from_list("div", [t["neg"], t["div_mid"], t["s1"]])
    c = prem_corr.values
    labels = list(prem_corr.index)
    shown = c.copy()
    np.fill_diagonal(shown, np.nan)
    cmap.set_bad(t["band"])
    ax.imshow(shown, cmap=cmap, vmin=-0.6, vmax=0.6)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=t["bg"], lw=2))
            if i != j:
                ax.text(j, i, f"{c[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                        color=t["ink"], fontweight="bold" if abs(c[i, j]) >= 0.2 else "normal")
    ax.set_xticks(range(len(c)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(c)), labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    title(ax, "The premia are close to uncorrelated",
          "Monthly correlation. Equity momentum vs value (-0.33) is the main diversifier.", t)
    save(fig, "02_premia_correlation", theme)


# ------------------------------------------------------------------ 03 overlays
def fig_overlay_frontier(t, theme):
    fig, ax = plt.subplots(figsize=(10, 5))
    d = ov.drop("Unhedged SPY").copy()
    d["saved"] = d[["Saved GFC 2007-09", "Saved COVID 2020", "Saved Inflation 2022"]].mean(axis=1)
    d["cost"] = d["Cost p.a. (2010-19 calm)"]
    for ratio in (1, 2, 4, 8):
        xs = np.linspace(0, 0.08, 50)
        ax.plot(xs, xs * ratio, color=t["grid"], lw=1.2, zorder=1)
        x_end = min(0.079, 0.185 / ratio)
        ax.text(x_end, x_end * ratio, f"{ratio}×  ", fontsize=8.5, color=t["muted"], ha="right", va="bottom")
    offsets = {"Collar 95/105 1m": (10, -7), "Zero-cost collar 95/x 1m": (10, 6), "Put spread 95/85 1m": (-10, -10),
               "Trend-gated put 95% 1m": (10, -4)}
    for k, r in d.iterrows():
        hero = k.startswith("Vol-target")
        alt = k.startswith("Zero-cost")
        c = t["s1"] if hero else t["s2"] if alt else t["bench"]
        ax.plot(r.cost, r.saved, "o", ms=13 if hero else 9, color=c, mec=t["bg"], mew=2, zorder=4)
        dx, dy = offsets.get(k, (10, 0))
        ax.annotate(OV_SHORT[k], (r.cost, r.saved), xytext=(dx, dy), textcoords="offset points", va="center",
                    ha="left" if dx > 0 else "right", fontsize=9.5, color=t["ink"],
                    fontweight="bold" if (hero or alt) else "normal")
    ax.axhline(0, color=t["axis"], lw=1)
    ax.set_xlim(0, 0.08)
    ax.set_ylim(-0.03, 0.2)
    ax.xaxis.set_major_formatter(PercentFormatter(1, 0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
    ax.set_xlabel("Cost: annual drag in calm years (2010–19)")
    ax.set_ylabel("Benefit: drawdown cut, avg of 3 crises")
    title(ax, "Protection bought per unit of cost",
          "Each point is an overlay on SPY, 1993–2026. Grey rays mark drawdown saved ÷ calm-year cost. Up and left is better.", t)
    save(fig, "03_overlay_frontier", theme)


def fig_overlay_episodes(t, theme):
    eps = {"GFC 2007–09": ("2007-10-01", "2009-12-31", [1, 7]), "COVID 2020": ("2020-02-01", "2020-07-31", [2, 4, 6]),
           "Inflation 2022": ("2022-01-01", "2022-12-31", [1, 4, 7, 10])}
    cols = [("Unhedged SPY", "Unhedged", t["bench"]), ("Put 95% 1m", "95% put, 1m", t["s2"]),
            ("Zero-cost collar 95/x 1m", "Zero-cost collar", t["s3"]),
            ("Vol-target overlay (no options)", "Vol target", t["s1"])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharey=True, gridspec_kw=dict(width_ratios=[2.2, 1, 1.4]))
    for ax, (name, (a, b, months)) in zip(axes, eps.items()):
        for col, _, c in cols:
            g = growth(ov_d[col].loc[a:b])
            dd = g / g.cummax() - 1
            ax.plot(dd.index, dd.values, color=c, lw=2.3 if col.startswith("Vol") else 1.5)
        ax.set_title(name, loc="left", fontsize=10.5, fontweight="bold", color=t["ink"])
        ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=months))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], color=c, lw=2.4) for _, _, c in cols]
    fig.suptitle("Drawdown inside each crisis, marked daily", x=0.01, ha="left", fontsize=13, fontweight="bold", color=t["ink"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, [c[1] for c in cols], loc="lower left", ncol=4, fontsize=9.5, bbox_to_anchor=(0.0, 0.0))
    save(fig, "03_overlay_episodes", theme)


# ------------------------------------------------------------------ hub
def fig_overview(t, theme):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw=dict(width_ratios=[1.3, 1, 1], wspace=0.45))
    a, b, c = axes

    shade_crises(a, t)
    for col, lab, colr, lw in [("Combined signals", "Signals", t["s1"], 2.4), ("60/40", "60/40", t["bench"], 1.6)]:
        g = growth(sig_m[col])
        a.plot(g.index, g.values, color=colr, lw=lw)
        s = sig_sum.loc[col]
        end_label(a, g.index[-1], g.iloc[-1], f"{lab}\nSharpe {s.Sharpe:.2f}", colr, t)
    log_money(a)
    a.set_yticks([1, 2, 4, 8])
    a.grid(axis="x", visible=False)
    # room inside the axes for the end labels, so they never touch the next panel
    a.set_xlim(sig_m.index[0], sig_m.index[-1] + pd.DateOffset(years=6))
    a.spines["bottom"].set_bounds(mdates.date2num(sig_m.index[0]), mdates.date2num(sig_m.index[-1]))
    a.set_xticks(pd.to_datetime(["2008-01-01", "2014-01-01", "2020-01-01", "2026-01-01"]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a.set_title("01  Signals beat 60/40 after costs", loc="left", fontsize=12.5, fontweight="bold", color=t["ink"], pad=22)
    a.text(0, 1.02, "Growth of 1, excess of T-bills, 2004–2026", transform=a.transAxes, fontsize=9.5, color=t["ink2"])

    sc = style_colors(t)
    d = prem_sum["Sharpe"].sort_values()
    ys = np.arange(len(d))
    for y, (k, v) in zip(ys, d.items()):
        b.barh(y, v, color=sc[STYLE[k]], height=0.64)
        b.text(v + (0.015 if v >= 0 else -0.015), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
               fontsize=9, color=t["ink"])
    b.set_yticks(ys, [k.replace(" (BAB)", "") for k in d.index], fontsize=9.5)
    b.axvline(0, color=t["axis"], lw=1)
    b.set_xlim(-0.55, 0.55)
    b.grid(axis="y", visible=False)
    b.set_title("02  Only carry clearly pays", loc="left", fontsize=12.5, fontweight="bold", color=t["ink"], pad=22)
    b.text(0, 1.02, "Sharpe per premium at 10% vol, 2003–2026", transform=b.transAxes, fontsize=9.5, color=t["ink2"])

    keep = ["Vol-target overlay (no options)", "Zero-cost collar 95/x 1m", "Trend-gated put 95% 1m",
            "Put spread 95/85 1m", "Put 95% 1m", "Put 90% 1m"]
    o = ov.loc[keep, "Avg DD saved / calm cost"].sort_values()
    ys = np.arange(len(o))
    for y, (k, v) in zip(ys, o.items()):
        hero = k.startswith("Vol")
        c.barh(y, v, color=t["s1"] if hero else t["bench"], height=0.64)
        c.text(v + 0.15, y, f"{v:.1f}×", va="center", fontsize=9.5, color=t["ink"], fontweight="bold" if hero else "normal")
    c.set_yticks(ys, [OV_SHORT[k] for k in o.index], fontsize=9.5)
    c.set_xlim(0, 10)
    c.grid(axis="y", visible=False)
    c.set_title("03  Vol targeting beats buying puts", loc="left", fontsize=12.5, fontweight="bold", color=t["ink"], pad=22)
    c.text(0, 1.02, "Drawdown saved ÷ calm-year cost, SPY 1993–2026", transform=c.transAxes, fontsize=9.5, color=t["ink2"])
    save(fig, "overview", theme)


if __name__ == "__main__":
    for theme, t in THEMES.items():
        setup(t)
        for f in (fig_overview, fig_signals_hero, fig_signals_crisis, fig_trend_grid, fig_premia_sharpe,
                  fig_premia_alloc, fig_premia_corr, fig_overlay_frontier, fig_overlay_episodes):
            f(t, theme)
    print(f"wrote {len(list(OUT.glob('*.png')))} figures to {OUT.relative_to(ROOT)}/")
