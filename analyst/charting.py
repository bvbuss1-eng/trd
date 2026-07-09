"""Candlestick chart rendering — the 'eyes' for both you and the AI analyst.

Dark surface, redundant up/down encoding (color + hollow/filled body direction is
implicit in OHLC geometry), recessive grid, status colors reserved for stop/target.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

import pandas as pd

SURFACE = "#12141a"
PANEL = "#181b22"
INK = "#e6e8ee"
INK_MUTED = "#8b90a0"
GRID = "#262a35"
UP = "#2fbf9b"       # teal-green (CVD-safer than pure green vs red)
DOWN = "#e0566b"     # rose-red
EMA9_C = "#e8c468"
EMA21_C = "#6aa1f0"
EMA50_C = "#b98ef0"
VWAP_C = "#f0a05a"
ENTRY_C = "#e6e8ee"
STOP_C = "#e0566b"
TARGET_C = "#2fbf9b"


def plot_chart(df: pd.DataFrame, symbol: str, interval: str,
               out_path: str, last_n: int = 120,
               entry: float | None = None, stop: float | None = None,
               target: float | None = None, title_extra: str = "") -> str:
    d = df.tail(last_n).copy()
    x = range(len(d))

    fig = plt.figure(figsize=(14, 9), dpi=110, facecolor=SURFACE)
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 0.8, 0.8, 0.0001], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    axr = fig.add_subplot(gs[2], sharex=ax)

    for a in (ax, axv, axr):
        a.set_facecolor(PANEL)
        a.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
        a.tick_params(colors=INK_MUTED, labelsize=8)
        for spine in a.spines.values():
            spine.set_color(GRID)

    # candles
    width = 0.7
    for i, (_, row) in enumerate(d.iterrows()):
        color = UP if row["close"] >= row["open"] else DOWN
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.9, zorder=2)
        body_lo = min(row["open"], row["close"])
        body_h = abs(row["close"] - row["open"]) or (row["high"] - row["low"]) * 0.001
        ax.bar(i, body_h, width, bottom=body_lo, color=color,
               edgecolor=color, linewidth=0.5, zorder=3)

    # overlays
    for col, color, label in (("ema9", EMA9_C, "EMA 9"), ("ema21", EMA21_C, "EMA 21"),
                              ("ema50", EMA50_C, "EMA 50")):
        if col in d:
            ax.plot(x, d[col].values, color=color, linewidth=1.3, label=label, zorder=4)
    if "vwap" in d:
        ax.plot(x, d["vwap"].values, color=VWAP_C, linewidth=1.3,
                linestyle="--", label="VWAP", zorder=4)

    # trade levels
    def hline(y: float, color: str, label: str):
        ax.axhline(y, color=color, linewidth=1.2, linestyle=":", zorder=5)
        ax.annotate(f" {label} {y:.6g}", xy=(len(d) - 1, y), xytext=(4, 0),
                    textcoords="offset points", color=color, fontsize=8.5,
                    va="center", fontweight="bold")

    if entry is not None:
        hline(entry, ENTRY_C, "ENTRY")
    if stop is not None:
        hline(stop, STOP_C, "STOP")
    if target is not None:
        hline(target, TARGET_C, "TARGET")

    ax.legend(loc="upper left", fontsize=8, framealpha=0.15,
              facecolor=PANEL, edgecolor=GRID, labelcolor=INK)
    ax.set_title(f"{symbol}  {interval}  {title_extra}".strip(),
                 color=INK, fontsize=12, loc="left", pad=10)

    # volume
    vol_colors = [UP if c >= o else DOWN for o, c in zip(d["open"], d["close"])]
    axv.bar(x, d["volume"].values, width, color=vol_colors, alpha=0.85)
    if "vol_ma20" in d:
        axv.plot(x, d["vol_ma20"].values, color=INK_MUTED, linewidth=1.0)
    axv.set_ylabel("Vol", color=INK_MUTED, fontsize=8)

    # RSI
    if "rsi" in d:
        axr.plot(x, d["rsi"].values, color=EMA21_C, linewidth=1.2)
        axr.axhline(70, color=INK_MUTED, linewidth=0.7, linestyle="--", alpha=0.7)
        axr.axhline(30, color=INK_MUTED, linewidth=0.7, linestyle="--", alpha=0.7)
        axr.set_ylim(0, 100)
        axr.set_ylabel("RSI", color=INK_MUTED, fontsize=8)

    # x labels: sparse timestamps
    ticks = list(range(0, len(d), max(1, len(d) // 8)))
    axr.set_xticks(ticks)
    axr.set_xticklabels([d.index[t].strftime("%m-%d %H:%M") for t in ticks],
                        rotation=0, fontsize=7.5, color=INK_MUTED)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(axv.get_xticklabels(), visible=False)

    fig.savefig(out_path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path
