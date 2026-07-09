"""Trend-pullback confluence strategy for 5m/15m day trading.

The idea (matches a discretionary intraday style):
  1. Higher-timeframe (HTF) trend filter — only trade WITH the 15m/1h trend.
  2. Wait for a pullback on the entry timeframe to the EMA21/VWAP zone.
  3. Enter on a momentum trigger candle (reclaim of prior high/low) with volume.
  4. Stop below structure (swing low / ATR); target = fixed R multiple.

Every condition is scored and explained, so a signal always carries its "why".
All checks at row i read data <= i only (no lookahead) — the same function is
used live and in the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .signal import Factor, Signal


@dataclass
class StrategyParams:
    rr: float = 2.0                 # target = rr * risk
    atr_mult_sl: float = 1.2        # stop distance floor, in ATRs
    swing_lookback: int = 10
    pullback_window: int = 6        # candles in which the pullback must have touched the zone
    min_vol_ratio: float = 1.1      # trigger candle volume vs 20-bar average
    min_adx: float = 18.0           # HTF trend strength floor (chop filter)
    min_bb_width: float = 0.0015    # entry-TF volatility floor (dead market filter)
    rsi_long_min: float = 38.0
    rsi_long_max: float = 62.0
    rsi_short_min: float = 38.0
    rsi_short_max: float = 62.0
    min_score: int = 7              # of MAX_SCORE below
    max_stop_pct: float = 1.5       # reject setups needing a stop wider than this %

    def to_dict(self) -> dict:
        return asdict(self)


MAX_SCORE = 8  # trend(2) + pullback(2) + trigger(2) + volume(1) + regime(1)
# Default min_score=7 forces trend + pullback + trigger, plus regime or volume —
# a signal can never fire while chasing extended price with no pullback.


def _htf_bias(htf: pd.DataFrame, i_time) -> tuple[str | None, str]:
    """Trend bias from the most recent CLOSED higher-timeframe candle at i_time."""
    prior = htf[htf["close_time"] <= i_time.value // 10**6] if hasattr(i_time, "value") else htf
    if len(prior) < 2:
        return None, "insufficient HTF history"
    row = prior.iloc[-1]
    if pd.isna(row.get("ema200")) or pd.isna(row.get("ema50")):
        return None, "HTF EMAs not formed yet"
    if row["ema50"] > row["ema200"] and row["close"] > row["ema50"]:
        return "LONG", (f"HTF uptrend: close {row['close']:.6g} > EMA50 {row['ema50']:.6g} "
                        f"> EMA200 {row['ema200']:.6g}, ADX {row['adx']:.1f}")
    if row["ema50"] < row["ema200"] and row["close"] < row["ema50"]:
        return "SHORT", (f"HTF downtrend: close {row['close']:.6g} < EMA50 {row['ema50']:.6g} "
                         f"< EMA200 {row['ema200']:.6g}, ADX {row['adx']:.1f}")
    return None, "HTF trend mixed (price/EMA50/EMA200 not aligned)"


def evaluate(df: pd.DataFrame, htf: pd.DataFrame, i: int,
             params: StrategyParams, symbol: str, interval: str,
             context: dict | None = None) -> Signal | None:
    """Evaluate the candle at integer position i (must be a CLOSED candle).

    df and htf must be indicator-enriched (indicators.enrich).
    Returns a Signal when confluence >= params.min_score, else None.
    """
    if i < 30 or i >= len(df):
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    t = df.index[i]

    bias, bias_detail = _htf_bias(htf, t)
    factors: list[Factor] = []
    if bias is None:
        return None  # hard filter: never fight or guess the higher timeframe
    factors.append(Factor("HTF trend alignment", True, 2, bias_detail))

    # HTF trend strength (chop filter)
    htf_prior = htf[htf["close_time"] <= t.value // 10**6]
    htf_adx = float(htf_prior.iloc[-1]["adx"])
    regime_ok = htf_adx >= params.min_adx and row["bb_width"] >= params.min_bb_width
    factors.append(Factor(
        "Regime filter", bool(regime_ok), 1,
        f"HTF ADX {htf_adx:.1f} (need >= {params.min_adx}), "
        f"BB width {row['bb_width']:.4f} (need >= {params.min_bb_width})"))

    window = df.iloc[max(0, i - params.pullback_window): i + 1]
    if bias == "LONG":
        zone_lo = window[["ema21", "vwap"]].min(axis=1)
        touched = bool((window["low"] <= zone_lo * 1.001).any())
        pullback_detail = "price pulled back into the EMA21/VWAP demand zone" if touched \
            else "no recent pullback to EMA21/VWAP — chasing extended price"
        rsi_ok = params.rsi_long_min <= row["rsi"] <= params.rsi_long_max and row["rsi"] > prev["rsi"]
        trigger = row["close"] > prev["high"] and row["close"] > row["open"]
        trigger_detail = (f"close {row['close']:.6g} reclaimed prior high {prev['high']:.6g} "
                          f"with bullish body" if trigger
                          else "no bullish reclaim of prior candle high")
    else:
        zone_hi = window[["ema21", "vwap"]].max(axis=1)
        touched = bool((window["high"] >= zone_hi * 0.999).any())
        pullback_detail = "price pulled back into the EMA21/VWAP supply zone" if touched \
            else "no recent pullback to EMA21/VWAP — chasing extended price"
        rsi_ok = params.rsi_short_min <= row["rsi"] <= params.rsi_short_max and row["rsi"] < prev["rsi"]
        trigger = row["close"] < prev["low"] and row["close"] < row["open"]
        trigger_detail = (f"close {row['close']:.6g} broke prior low {prev['low']:.6g} "
                          f"with bearish body" if trigger
                          else "no bearish break of prior candle low")

    factors.append(Factor("Pullback to value zone", touched, 2, pullback_detail))
    factors.append(Factor("Momentum trigger candle", bool(trigger), 2, trigger_detail))
    factors.append(Factor(
        "RSI reset & turning", bool(rsi_ok), 0,
        f"RSI {row['rsi']:.1f} (prev {prev['rsi']:.1f}) — informational, not scored"))

    vol_ok = bool(row["vol_ratio"] >= params.min_vol_ratio) if pd.notna(row["vol_ratio"]) else False
    factors.append(Factor(
        "Volume confirmation", vol_ok, 1,
        f"volume {row['vol_ratio']:.2f}x its 20-bar average (need >= {params.min_vol_ratio}x)"))

    score = sum(f.weight for f in factors if f.passed)
    if score < params.min_score or not trigger:
        return None

    entry = float(row["close"])
    a = float(row["atr"])
    if bias == "LONG":
        structural = float(row["swing_low"])
        stop = min(structural, entry - params.atr_mult_sl * a)
        risk = entry - stop
        target = entry + params.rr * risk
    else:
        structural = float(row["swing_high"])
        stop = max(structural, entry + params.atr_mult_sl * a)
        risk = stop - entry
        target = entry - params.rr * risk

    if risk <= 0 or risk / entry * 100 > params.max_stop_pct:
        return None  # stop too wide for an intraday leveraged trade

    return Signal(
        symbol=symbol, interval=interval, direction=bias,
        time=str(t), entry=entry, stop=round(stop, 8), target=round(target, 8),
        rr=params.rr, score=score, max_score=MAX_SCORE,
        factors=factors, context=dict(context or {}),
    )


def latest_signal(df: pd.DataFrame, htf: pd.DataFrame, params: StrategyParams,
                  symbol: str, interval: str, context: dict | None = None) -> Signal | None:
    """Evaluate only the most recent closed candle (live scanning)."""
    return evaluate(df, htf, len(df) - 1, params, symbol, interval, context)
