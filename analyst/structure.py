"""Market-structure research: how does this market actually move?

Answers the questions you must answer BEFORE designing entries:
  - WHEN does it move (hour-of-day / day-of-week volatility and drift)?
  - Does it TREND or FADE on this timeframe (breakout follow-through,
    return autocorrelation, regime share)?
  - How often does it run stops and reverse (liquidity sweeps — the
    "manipulation" pattern: wick through a prior swing level, close back
    inside, then move the other way)?
  - Can the timeframe even pay for itself (typical bar range vs fees)?

This module is deliberately NOT a strategy: it uses future data to measure
what follows each event (research needs hindsight; signals must not have it).
Design entries from what this reports, then verify them in the walk-forward
backtester which has no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSIONS = (
    ("Asia", 0, 7),
    ("London", 7, 13),
    ("NY", 13, 21),
    ("Late US", 21, 24),
)


def _session_of(hour: int) -> str:
    for name, lo, hi in SESSIONS:
        if lo <= hour < hi:
            return name
    return "?"


def _bps(x: float) -> float:
    return x * 10_000


def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per UTC hour: average bar range (bps), net drift (bps), efficiency.

    efficiency = |sum of returns| / sum of |returns| — near 0 means the hour
    churns without going anywhere (chop), higher means directional.
    """
    ret = df["close"].pct_change()
    rng = (df["high"] - df["low"]) / df["close"]
    g = pd.DataFrame({"ret": ret, "rng": rng, "hour": df.index.hour}).dropna()
    rows = []
    for hour, sub in g.groupby("hour"):
        rows.append({
            "hour": hour,
            "range_bps": _bps(sub["rng"].mean()),
            "drift_bps": _bps(sub["ret"].mean()),
            "efficiency": abs(sub["ret"].sum()) / sub["ret"].abs().sum()
            if sub["ret"].abs().sum() else 0.0,
        })
    return pd.DataFrame(rows).set_index("hour")


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["close"].pct_change()
    rng = (df["high"] - df["low"]) / df["close"]
    g = pd.DataFrame({"ret": ret, "rng": rng, "dow": df.index.dayofweek}).dropna()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rows = []
    for dow, sub in g.groupby("dow"):
        rows.append({"day": names[dow],
                     "range_bps": _bps(sub["rng"].mean()),
                     "drift_bps": _bps(sub["ret"].mean())})
    return pd.DataFrame(rows).set_index("day")


def sweep_stats(df: pd.DataFrame, lookback: int = 24, horizon: int = 12) -> pd.DataFrame:
    """Liquidity sweeps (stop hunts): wick pierces the prior `lookback`-bar
    extreme but the bar CLOSES back inside. Then measure the return over the
    next `horizon` bars. If reversal follow-through is strong, sweeps are a
    tradeable pattern; if ~50/50, the 'manipulation' story has no edge here."""
    prior_low = df["low"].shift(1).rolling(lookback).min()
    prior_high = df["high"].shift(1).rolling(lookback).max()
    close = df["close"]
    fwd = close.shift(-horizon) / close - 1

    sweep_lo = (df["low"] < prior_low) & (close > prior_low)
    sweep_hi = (df["high"] > prior_high) & (close < prior_high)

    rows = []
    for name, mask, direction in (("low sweep -> long", sweep_lo, 1),
                                  ("high sweep -> short", sweep_hi, -1)):
        ev = pd.DataFrame({"fwd": fwd[mask] * direction,
                           "hour": df.index[mask].hour}).dropna()
        if not len(ev):
            rows.append({"event": name, "count": 0})
            continue
        row = {"event": name, "count": len(ev),
               "per_day": len(ev) / max(1, len(np.unique(df.index[mask].date))),
               "follow_%": (ev["fwd"] > 0).mean() * 100,
               "avg_bps": _bps(ev["fwd"].mean()),
               "med_bps": _bps(ev["fwd"].median())}
        for sess, lo, hi in SESSIONS:
            sub = ev[(ev["hour"] >= lo) & (ev["hour"] < hi)]
            row[f"{sess}_%"] = (sub["fwd"] > 0).mean() * 100 if len(sub) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("event")


def breakout_stats(df: pd.DataFrame, lookback: int = 24, horizon: int = 12) -> pd.DataFrame:
    """Genuine breakouts: bar CLOSES beyond the prior `lookback`-bar extreme.
    Continuation ~>52% => momentum works on this TF; <48% => breakouts fade
    (mean reversion dominates); in between => no edge either way."""
    prior_low = df["low"].shift(1).rolling(lookback).min()
    prior_high = df["high"].shift(1).rolling(lookback).max()
    close = df["close"]
    fwd = close.shift(-horizon) / close - 1

    brk_up = close > prior_high
    brk_dn = close < prior_low

    rows = []
    for name, mask, direction in (("breakout up", brk_up, 1),
                                  ("breakout down", brk_dn, -1)):
        ev = (fwd[mask] * direction).dropna()
        if not len(ev):
            rows.append({"event": name, "count": 0})
            continue
        rows.append({"event": name, "count": len(ev),
                     "continue_%": (ev > 0).mean() * 100,
                     "avg_bps": _bps(ev.mean()),
                     "med_bps": _bps(ev.median())})
    return pd.DataFrame(rows).set_index("event")


def autocorrelation(df: pd.DataFrame, lags: tuple = (1, 2, 3, 6, 12)) -> dict:
    """Serial correlation of bar returns. Negative at short lags = snap-back
    (mean reversion); positive = momentum; ~0 = noise."""
    ret = df["close"].pct_change().dropna()
    return {lag: float(ret.autocorr(lag)) for lag in lags}


def regime_share(htf: pd.DataFrame) -> dict:
    """Share of time the higher timeframe was trending vs ranging (ADX)."""
    adx = htf["adx"].dropna()
    return {
        "trending_%  (ADX>25)": float((adx > 25).mean() * 100),
        "neutral_%   (20-25)": float(((adx >= 20) & (adx <= 25)).mean() * 100),
        "ranging_%   (ADX<20)": float((adx < 20).mean() * 100),
    }


def cost_floor(df: pd.DataFrame, taker_pct: float = 0.05, maker_pct: float = 0.02,
               slippage_bps: float = 2.0) -> dict:
    """Median bar range and ATR vs round-trip costs, in bps. If costs are a
    large fraction of the ATR, targets on this TF mostly pay the exchange."""
    atr_bps = _bps((df["atr"] / df["close"]).median())
    rng_bps = _bps(((df["high"] - df["low"]) / df["close"]).median())
    taker_rt = taker_pct * 100 * 2 + slippage_bps * 2
    maker_rt = maker_pct * 100 * 2 + slippage_bps
    return {
        "median_bar_range_bps": rng_bps,
        "median_atr_bps": atr_bps,
        "taker_roundtrip_bps": taker_rt,
        "maker_roundtrip_bps": maker_rt,
        "taker_cost_vs_atr_%": taker_rt / atr_bps * 100 if atr_bps else float("inf"),
        "maker_cost_vs_atr_%": maker_rt / atr_bps * 100 if atr_bps else float("inf"),
    }


def largest_move_hours(df: pd.DataFrame, top_n: int = 50) -> pd.Series:
    """Which UTC hours produce the biggest single bars (news/opens cluster)."""
    ret = df["close"].pct_change().abs().dropna()
    top = ret.nlargest(top_n)
    return top.index.hour.value_counts().sort_index()


def structure_report(df: pd.DataFrame, htf: pd.DataFrame, symbol: str,
                     interval: str, sweep_lookback: int = 24,
                     horizon: int = 12) -> str:
    """Full research report. df/htf must be indicator-enriched."""
    days = max(1, len(np.unique(df.index.date)))
    lines = [
        "=" * 70,
        f"MARKET STRUCTURE — {symbol} {interval}  ({len(df)} candles, ~{days} days)",
        f"  sweep/breakout lookback: {sweep_lookback} bars; "
        f"outcome horizon: {horizon} bars",
        "=" * 70,
    ]

    lines.append("\n-- WHEN IT MOVES: UTC hour profile "
                 "(range=activity, efficiency=direction vs chop) --")
    hp = hourly_profile(df)
    for hour, row in hp.iterrows():
        bar = "#" * int(round(row["range_bps"] / hp["range_bps"].max() * 30))
        lines.append(f"  {hour:02d}:00  range {row['range_bps']:5.1f} bps  "
                     f"drift {row['drift_bps']:+5.2f}  eff {row['efficiency']:.3f}  {bar}")
    best = hp["range_bps"].nlargest(4).index.tolist()
    lines.append(f"  Most active hours (UTC): {sorted(best)}")

    lines.append("\n-- DAY OF WEEK --")
    for day, row in weekday_profile(df).iterrows():
        lines.append(f"  {day}  range {row['range_bps']:5.1f} bps  "
                     f"drift {row['drift_bps']:+5.2f} bps/bar")

    lines.append("\n-- LIQUIDITY SWEEPS (stop hunts: wick through prior "
                 f"{sweep_lookback}-bar extreme, close back inside) --")
    lines.append(f"   follow_% = how often price then moved the REVERSAL way "
                 f"over the next {horizon} bars")
    sw = sweep_stats(df, sweep_lookback, horizon)
    for name, row in sw.iterrows():
        if row.get("count", 0) == 0:
            lines.append(f"  {name}: no events")
            continue
        lines.append(f"  {name}: {int(row['count'])} events "
                     f"(~{row['per_day']:.1f}/day)  "
                     f"reversal follows {row['follow_%']:.1f}%  "
                     f"avg {row['avg_bps']:+.1f} bps  med {row['med_bps']:+.1f} bps")
        sess_bits = [f"{s}: {row[f'{s}_%']:.0f}%" for s, _, _ in SESSIONS
                     if pd.notna(row.get(f"{s}_%"))]
        lines.append(f"      by session: {'  '.join(sess_bits)}")

    lines.append("\n-- BREAKOUT FOLLOW-THROUGH (close beyond prior extreme) --")
    lines.append("   >52% continue = momentum TF; <48% = breakouts fade (mean-rev TF)")
    for name, row in breakout_stats(df, sweep_lookback, horizon).iterrows():
        if row.get("count", 0) == 0:
            lines.append(f"  {name}: no events")
            continue
        lines.append(f"  {name}: {int(row['count'])} events  "
                     f"continues {row['continue_%']:.1f}%  "
                     f"avg {row['avg_bps']:+.1f} bps  med {row['med_bps']:+.1f} bps")

    lines.append("\n-- RETURN AUTOCORRELATION (negative = snap-back, positive = momentum) --")
    for lag, v in autocorrelation(df).items():
        lines.append(f"  lag {lag:>2} bars: {v:+.4f}")

    lines.append("\n-- REGIME SHARE (higher timeframe ADX) --")
    for k, v in regime_share(htf).items():
        lines.append(f"  {k}: {v:.1f}%")

    lines.append("\n-- CAN THIS TIMEFRAME PAY FOR ITSELF? --")
    cf = cost_floor(df)
    for k, v in cf.items():
        lines.append(f"  {k}: {v:.1f}")
    lines.append("  (cost_vs_atr over ~30% means most of an ATR-sized target "
                 "goes to fees — trade a higher TF or maker fills)")

    lines.append("\n-- WHERE THE BIGGEST BARS CLUSTER (top-50 moves, by UTC hour) --")
    for hour, count in largest_move_hours(df).items():
        lines.append(f"  {hour:02d}:00  {'*' * int(count)} ({count})")

    lines.append("\nNOTE: this report USES HINDSIGHT (it measures what followed each")
    lines.append("event). It is for designing hypotheses. Any strategy built from it")
    lines.append("must then pass the walk-forward backtester, which has none.")
    return "\n".join(lines)
