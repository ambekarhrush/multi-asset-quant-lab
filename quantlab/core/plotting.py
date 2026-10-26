"""Consistent, restrained chart style for research notes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#808080", "#bf9000", "#2e75b6"]


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "font.size": 9,
            "legend.frameon": False,
        }
    )


def equity_curves(rets: dict[str, pd.Series], title: str, path: Path, log: bool = True):
    style()
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, r in rets.items():
        (1 + r.fillna(0)).cumprod().plot(ax=ax, label=k, lw=1.3)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def drawdowns(rets: dict[str, pd.Series], title: str, path: Path):
    from .metrics import drawdown

    style()
    fig, ax = plt.subplots(figsize=(8, 3))
    for k, r in rets.items():
        drawdown(r).plot(ax=ax, label=k, lw=1.1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def bar(series: pd.Series, title: str, path: Path, pct: bool = False):
    style()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    series.plot.bar(ax=ax, color=PALETTE[0])
    ax.set_title(title, loc="left", fontweight="bold")
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
