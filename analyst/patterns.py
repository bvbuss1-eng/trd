"""Pattern Lab: measure a catalog of classic price patterns on real data.

Every pattern a discretionary trader watches is, mechanically, a condition
on a candle plus a claim about what follows. This module tests the catalog:
for each pattern it reports how often it fired, how often the claimed
follow-through happened, the average move in bps, and a t-statistic.

Reading the table:
  follow_%  > 55 with |t| >= 2  -> real candidate, worth building into entries
  |t| < 1                       -> statistically indistinguishable from noise,
                                   no matter how good the story is
  avg vs med                    -> if avg >> med the edge lives in rare tails

Like structure.py this uses hindsight ON PURPOSE (research, not signals).
Anything promoted from here must pass the walk-forward backtester.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _stats(name: str, signed_fwd: pd.Series, claim: str) -> dict:
    ev = signed_fwd.dropna()
    n = len(ev)
    if n < 5:
        return {"pattern": name, "claim": claim, "count": n}
    mean, std = float(ev.mean()), float(ev.std())
    return {
        "pattern": name, "claim": claim, "count": n,
        "follow_%": float((ev > 0).mean() * 100),
        "avg_bps": mean * 10_000, "med_bps": float(ev.median()) * 10_000,
        "t_stat": mean / (std / np.sqrt(n)) if std else 0.0,
    }


def pattern_table(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """df must be indicator-enriched. Returns one row per pattern/direction."""
    close, o, hi, lo = df["close"], df["open"], df["high"], df["low"]
    atr, vwap = df["atr"], df["vwap"]
    fwd = close.shift(-horizon) / close - 1
    bar_dir = np.sign(close - o)
    body = (close - o).abs()
    rows = []

    # ---- prior-day high/low touch-and-hold (the classic key levels)
    days = df.index.floor("D")
    pdh = pd.Series(days).map(hi.resample("1D").max().shift(1)).to_numpy()
    pdl = pd.Series(days).map(lo.resample("1D").min().shift(1)).to_numpy()
    pdh = pd.Series(pdh, index=df.index)
    pdl = pd.Series(pdl, index=df.index)
    m = (lo <= pdl) & (close > pdl) & (o > pdl)
    rows.append(_stats("prior-day LOW touch & hold", fwd[m],
                       "bounce long off yesterday's low"))
    m = (hi >= pdh) & (close < pdh) & (o < pdh)
    rows.append(_stats("prior-day HIGH touch & hold", -fwd[m],
                       "rejection short off yesterday's high"))

    # ---- round-number levels (grid auto-scaled to price magnitude)
    grid = 10 ** (int(np.floor(np.log10(float(close.median())))) - 2)
    lvl_dn = (o / grid).apply(np.floor) * grid
    lvl_up = (o / grid).apply(np.ceil) * grid
    m = (lo <= lvl_dn) & (close > lvl_dn) & (o > lvl_dn)
    rows.append(_stats(f"round {grid:g} reclaim (support)", fwd[m],
                       "round number holds as support -> long"))
    m = (hi >= lvl_up) & (close < lvl_up) & (o < lvl_up)
    rows.append(_stats(f"round {grid:g} rejection (resistance)", -fwd[m],
                       "round number rejects -> short"))

    # ---- momentum exhaustion: 5 consecutive same-color closes
    sign = np.sign(close.diff())
    m = sign.rolling(5).sum() == -5
    rows.append(_stats("5 red candles in a row", fwd[m], "exhaustion -> snap-back long"))
    m = sign.rolling(5).sum() == 5
    rows.append(_stats("5 green candles in a row", -fwd[m], "exhaustion -> snap-back short"))

    # ---- big bar: continue or fade?
    m = (body >= 2.5 * atr) & (bar_dir != 0)
    rows.append(_stats("big bar (>=2.5 ATR), continuation", (fwd * bar_dir)[m],
                       "momentum continues the big bar's direction"))

    # ---- volume climax: fade the crowd
    m = (df["vol_ratio"] >= 3.0) & (bar_dir != 0)
    rows.append(_stats("volume climax (>=3x avg), fade", (-fwd * bar_dir)[m],
                       "climax volume exhausts the move -> reverse"))

    # ---- VWAP stretch reversion
    m = (close - vwap) <= -2 * atr
    rows.append(_stats("stretched 2 ATR under VWAP", fwd[m], "reversion long to VWAP"))
    m = (close - vwap) >= 2 * atr
    rows.append(_stats("stretched 2 ATR over VWAP", -fwd[m], "reversion short to VWAP"))

    # ---- pin bar at a 24-bar extreme (the pivot-reversal candle)
    rng = (hi - lo).replace(0, np.nan)
    low_wick = (pd.concat([o, close], axis=1).min(axis=1) - lo)
    high_wick = (hi - pd.concat([o, close], axis=1).max(axis=1))
    m = (low_wick / rng >= 0.6) & (lo <= lo.rolling(24).min())
    rows.append(_stats("pin bar at 24-bar low", fwd[m], "pivot reversal long"))
    m = (high_wick / rng >= 0.6) & (hi >= hi.rolling(24).max())
    rows.append(_stats("pin bar at 24-bar high", -fwd[m], "pivot reversal short"))

    # ---- engulfing candle follow-through
    prev_o, prev_c = o.shift(1), close.shift(1)
    bull = (close > o) & (prev_c < prev_o) & (close >= prev_o) & (o <= prev_c)
    bear = (close < o) & (prev_c > prev_o) & (close <= prev_o) & (o >= prev_c)
    rows.append(_stats("bullish engulfing", fwd[bull], "reversal long follows"))
    rows.append(_stats("bearish engulfing", -fwd[bear], "reversal short follows"))

    # ---- NY open first bar sets the session tone
    m = (df.index.hour == 13) & (df.index.minute == 0) & (bar_dir != 0)
    rows.append(_stats("NY-open bar direction", (fwd * bar_dir)[m],
                       "13:00 UTC bar direction persists"))

    out = pd.DataFrame(rows)
    if "t_stat" in out:
        out = out.sort_values("t_stat", key=lambda s: s.abs(), ascending=False)
    return out


def daily_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """'3 days up -> what does day 4 do?' — the daily momentum question."""
    daily = df["close"].resample("1D").last().dropna()
    ret = daily.pct_change()
    rows = []
    for k in (2, 3):
        for direction, name in ((1, f"{k} up days"), (-1, f"{k} down days")):
            streak = (np.sign(ret).rolling(k).sum() == k * direction)
            nxt = ret.shift(-1)[streak].dropna()
            if len(nxt) < 3:
                rows.append({"setup": name, "count": len(nxt)})
                continue
            rows.append({
                "setup": name, "count": len(nxt),
                "next_day_up_%": float((nxt > 0).mean() * 100),
                "avg_next_bps": float(nxt.mean()) * 10_000,
            })
    return pd.DataFrame(rows)


def patterns_report(df: pd.DataFrame, symbol: str, interval: str,
                    horizon: int = 12) -> str:
    days = max(1, len(np.unique(df.index.date)))
    lines = [
        "=" * 78,
        f"PATTERN LAB — {symbol} {interval}  ({len(df)} candles, ~{days} days, "
        f"outcome horizon {horizon} bars)",
        "=" * 78,
        "  follow_% = how often the pattern's claim came true over the horizon",
        "  |t| >= 2: real candidate   1 < |t| < 2: weak   |t| < 1: noise, discard",
        "",
        f"  {'pattern':<38} {'n':>5} {'follow%':>8} {'avg':>7} {'med':>7} {'t':>6}",
        "  " + "-" * 74,
    ]
    tab = pattern_table(df, horizon)
    for _, r in tab.iterrows():
        if r.get("count", 0) < 5 or "follow_%" not in r or pd.isna(r.get("follow_%")):
            lines.append(f"  {r['pattern']:<38} {int(r.get('count', 0)):>5}  (too few events)")
            continue
        lines.append(f"  {r['pattern']:<38} {int(r['count']):>5} "
                     f"{r['follow_%']:>7.1f}% {r['avg_bps']:>+6.1f} "
                     f"{r['med_bps']:>+6.1f} {r['t_stat']:>+6.2f}")

    lines.append("\n  Claims tested:")
    for _, r in tab.iterrows():
        lines.append(f"    - {r['pattern']}: {r['claim']}")

    lines.append("\n-- DAILY STREAKS (your '3 days up -> day 4?' question) --")
    for _, r in daily_streaks(df).iterrows():
        if r.get("count", 0) < 3 or pd.isna(r.get("next_day_up_%")):
            lines.append(f"  {r['setup']}: only {int(r.get('count', 0))} events — not enough")
            continue
        lines.append(f"  {r['setup']}: {int(r['count'])} events, next day up "
                     f"{r['next_day_up_%']:.0f}% of the time, avg {r['avg_next_bps']:+.0f} bps")

    lines.append("\nNOTE: hindsight research. A pattern that survives here still has to")
    lines.append("earn its place through the walk-forward backtester before it trades.")
    return "\n".join(lines)
