"""VWAP-stretch fade — the composite strategy built from the Pattern Lab.

What survived 90 days of measurement on BTCUSDT (15m AND 1h):
  - Price stretched >= 2 ATR above session VWAP reverts (t +5.2 / +2.6).
  - Exhaustion confluences agree: runs of green candles snap back, pierces
    of the prior 24-bar high reverse in the quiet sessions.
  - The classic "buy the dip at support" patterns FAILED the same exam
    (prior-day-low bounces: 31.5% on 1h, t -4.2), so longs are OFF by
    default and only enabled explicitly.

Trade construction (short side; long side is the mirror when enabled):
  - Setup: close >= VWAP + stretch_atr * ATR — price is statistically
    overextended above the session's volume-weighted average.
  - Confluence (need >= confluence_min): wick into the prior 24-bar high
    (the sweep), >= 4 green closes in the last 5 bars (exhaustion), or a
    dominant upper wick (rejection).
  - Trigger: a bearish-body close. Never short a candle still going up.
  - Stop: above the local 3-bar high + pad. Target: tp_frac of the way
    back to VWAP — the measured magnet. RR is computed per trade from that
    geometry and the trade is skipped if it is worse than min_rr.
  - 1h strong-uptrend veto; time-stop after max_hold bars.

Regime honesty: this edge was measured in ONE 90-day window with a short
bias. The veto limits the damage of a regime flip; re-run the Pattern Lab
monthly and retire the strategy the day the stretch stops reverting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .signal import Factor, Signal
from .sweep import parse_hours


@dataclass
class FadeParams:
    stretch_atr: float = 2.0      # setup: |close - VWAP| >= this many ATRs
    tp_frac: float = 0.75         # target this fraction of the way back to VWAP
    min_rr: float = 0.8           # skip trades with worse geometry than this
    stop_pad_atr: float = 0.3     # stop beyond the 3-bar extreme
    confluence_min: int = 1       # of: sweep touch / candle streak / rejection wick
    hours: str = "0-24"           # entry window, UTC
    max_htf_adx: float = 35.0     # veto fading a strong 1h trend
    max_hold: int = 12
    min_stop_bps: float = 6.0
    max_stop_pct: float = 1.0
    allow_longs: bool = False     # dip-buying failed the 90d exam; opt-in only

    def to_dict(self) -> dict:
        return asdict(self)


MAX_SCORE = 6  # stretch(2) + trigger(2) + confluence(1) + HTF regime(1)


def _htf_row(htf: pd.DataFrame, i_time) -> pd.Series | None:
    prior = htf[htf["close_time"] <= i_time.value // 10**6] if hasattr(i_time, "value") else htf
    if len(prior) < 2:
        return None
    return prior.iloc[-1]


def evaluate_fade(df: pd.DataFrame, htf: pd.DataFrame, i: int,
                  params: FadeParams, symbol: str, interval: str,
                  context: dict | None = None) -> Signal | None:
    """Evaluate the CLOSED candle at position i. Frames must be enriched."""
    if i < 30 or i >= len(df):
        return None
    row = df.iloc[i]
    t = df.index[i]
    a = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
    vwap = float(row["vwap"]) if pd.notna(row["vwap"]) else 0.0
    if a <= 0 or vwap <= 0:
        return None
    hrow = _htf_row(htf, t)
    if hrow is None or pd.isna(hrow.get("adx")):
        return None

    entry = float(row["close"])
    dist = entry - vwap
    if dist >= params.stretch_atr * a:
        direction = "SHORT"
    elif params.allow_longs and -dist >= params.stretch_atr * a:
        direction = "LONG"
    else:
        return None

    if t.hour not in parse_hours(params.hours):
        return None

    window = df.iloc[max(0, i - 24): i]
    last5 = df["close"].diff().iloc[max(0, i - 4): i + 1]
    rng = float(row["high"] - row["low"]) or float("nan")
    ema200_ok = pd.notna(hrow.get("ema200"))

    if direction == "SHORT":
        trigger = row["close"] < row["open"]
        swept = len(window) and float(row["high"]) >= float(window["high"].max())
        streak = int((last5 > 0).sum()) >= 4
        wick = (float(row["high"]) - max(entry, float(row["open"]))) / rng >= 0.5 \
            if rng == rng else False
        strong_against = bool(ema200_ok and hrow["ema50"] > hrow["ema200"]
                              and hrow["adx"] >= params.max_htf_adx)
        stretch_detail = (f"close {entry:.6g} is {dist / a:.2f} ATR above "
                          f"VWAP {vwap:.6g} — measured to revert")
        trigger_detail = "bearish-body close" if trigger else "candle still bullish — no trigger"
    else:
        trigger = row["close"] > row["open"]
        swept = len(window) and float(row["low"]) <= float(window["low"].min())
        streak = int((last5 < 0).sum()) >= 4
        wick = (min(entry, float(row["open"])) - float(row["low"])) / rng >= 0.5 \
            if rng == rng else False
        strong_against = bool(ema200_ok and hrow["ema50"] < hrow["ema200"]
                              and hrow["adx"] >= params.max_htf_adx)
        stretch_detail = (f"close {entry:.6g} is {-dist / a:.2f} ATR below "
                          f"VWAP {vwap:.6g}")
        trigger_detail = "bullish-body close" if trigger else "candle still bearish — no trigger"

    confl = int(swept) + int(streak) + int(wick)
    factors = [
        Factor("VWAP stretch", True, 2, stretch_detail),
        Factor("Reversal trigger", bool(trigger), 2, trigger_detail),
        Factor("Exhaustion confluence", confl >= params.confluence_min, 1,
               f"{confl} of 3 present — sweep touch: {swept}, "
               f"candle streak: {streak}, rejection wick: {wick}"),
        Factor("HTF regime permits fading", not strong_against, 1,
               f"1h ADX {hrow['adx']:.1f} (veto at >= {params.max_htf_adx} "
               f"against the trade)"),
    ]
    if not trigger or confl < params.confluence_min or strong_against:
        return None
    score = sum(f.weight for f in factors if f.passed)

    recent = df.iloc[max(0, i - 2): i + 1]
    if direction == "SHORT":
        stop = float(recent["high"].max()) + params.stop_pad_atr * a
        risk = stop - entry
        target = entry - params.tp_frac * dist
        rr = (entry - target) / risk if risk > 0 else 0.0
    else:
        stop = float(recent["low"].min()) - params.stop_pad_atr * a
        risk = entry - stop
        target = entry + params.tp_frac * (-dist)
        rr = (target - entry) / risk if risk > 0 else 0.0

    stop_bps = risk / entry * 10_000
    if risk <= 0 or rr < params.min_rr:
        return None  # geometry too poor: stop too wide vs the road back to VWAP
    if stop_bps < params.min_stop_bps or risk / entry * 100 > params.max_stop_pct:
        return None

    return Signal(
        symbol=symbol, interval=interval, direction=direction,
        time=str(t), entry=entry, stop=round(stop, 8), target=round(target, 8),
        rr=round(rr, 3), score=score, max_score=MAX_SCORE,
        factors=factors, context=dict(context or {}),
    )


def latest_fade_signal(df: pd.DataFrame, htf: pd.DataFrame, params: FadeParams,
                       symbol: str, interval: str,
                       context: dict | None = None) -> Signal | None:
    return evaluate_fade(df, htf, len(df) - 1, params, symbol, interval, context)


def grid_search_fade(df: pd.DataFrame, htf: pd.DataFrame, symbol: str, interval: str,
                     stretch_values=(1.5, 2.0, 2.5),
                     tp_values=(0.6, 0.75, 1.0),
                     confl_values=(1, 2),
                     fee_pct: float = 0.02, slippage_bps: float = 2.0):
    """Sweep the stretch threshold, target depth, and confluence strictness."""
    from .backtest import run_backtest

    results = []
    for st in stretch_values:
        for tp in tp_values:
            for cf in confl_values:
                p = FadeParams(stretch_atr=st, tp_frac=tp, confluence_min=cf)
                res = run_backtest(df, htf, p, symbol, interval,
                                   evaluate_fn=evaluate_fade, max_hold=p.max_hold,
                                   fee_pct=fee_pct, slippage_bps=slippage_bps)
                results.append((p, res))
    results.sort(key=lambda pr: pr[1].expectancy_r * max(pr[1].n, 1), reverse=True)
    return results
