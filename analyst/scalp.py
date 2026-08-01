"""5m scalp — the last structural configuration the data allows on 5m.

Three levers, all measured, none tried together before:
  1. SESSION: only the NY hours (13:00-16:00 UTC), where 5m ATR runs
     17-24 bps instead of the all-day median 12 bps. Plus a hard ATR
     floor — if the tape isn't paying, there is no trade.
  2. EXECUTION: entries are RESTING LIMIT ORDERS at the stretched extreme
     (sig.limit), so the push fills us at a better price with maker fees.
     The backtester models the miss: no touch, no trade.
  3. GEOMETRY: tight scalp structure — limit at the local extreme + pad,
     stop 0.8 ATR behind it, target 2R. At NY-session ATR that is roughly
     15 bps risk / 30 bps target against ~5 bps round-trip maker cost:
     breakeven near 44% win rate. A beatable bar, unlike RR<1 geometries.

Setup logic (short side): price extended above session VWAP, resting a
fade limit above the local high; long side is the mirror below VWAP.
1h strong-trend veto. Same no-lookahead contract as every other strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .signal import Factor, Signal
from .sweep import parse_hours


@dataclass
class ScalpParams:
    hours: str = "13-16"          # the paid hours (NY), UTC
    min_atr_bps: float = 15.0     # no trade when the 5m tape is dead
    stretch_atr: float = 1.5      # extension from VWAP that arms a fade
    limit_pad_atr: float = 0.15   # rest the limit this far beyond the 3-bar extreme
    sl_atr: float = 0.8           # stop distance beyond the limit
    rr: float = 2.0               # target = rr * stop distance
    max_htf_adx: float = 40.0     # veto fading a very strong 1h trend
    max_hold: int = 24            # 2 hours on 5m, then out
    min_stop_bps: float = 8.0
    max_stop_pct: float = 0.6
    allow_longs: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


MAX_SCORE = 6  # stretch(2) + paid session(1) + live ATR(1) + extreme rest(1) + regime(1)


def _htf_row(htf: pd.DataFrame, i_time) -> pd.Series | None:
    prior = htf[htf["close_time"] <= i_time.value // 10**6] if hasattr(i_time, "value") else htf
    if len(prior) < 2:
        return None
    return prior.iloc[-1]


def evaluate_scalp(df: pd.DataFrame, htf: pd.DataFrame, i: int,
                   params: ScalpParams, symbol: str, interval: str,
                   context: dict | None = None) -> Signal | None:
    """Evaluate the CLOSED candle at position i; may return a limit-entry Signal."""
    if i < 30 or i >= len(df):
        return None
    row = df.iloc[i]
    t = df.index[i]
    a = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
    vwap = float(row["vwap"]) if pd.notna(row["vwap"]) else 0.0
    close = float(row["close"])
    if a <= 0 or vwap <= 0 or close <= 0:
        return None

    if t.hour not in parse_hours(params.hours):
        return None
    atr_bps = a / close * 10_000
    if atr_bps < params.min_atr_bps:
        return None

    hrow = _htf_row(htf, t)
    if hrow is None or pd.isna(hrow.get("adx")):
        return None
    ema200_ok = pd.notna(hrow.get("ema200"))

    dist = close - vwap
    if dist >= params.stretch_atr * a:
        direction = "SHORT"
        strong_against = bool(ema200_ok and hrow["ema50"] > hrow["ema200"]
                              and hrow["adx"] >= params.max_htf_adx)
    elif params.allow_longs and -dist >= params.stretch_atr * a:
        direction = "LONG"
        strong_against = bool(ema200_ok and hrow["ema50"] < hrow["ema200"]
                              and hrow["adx"] >= params.max_htf_adx)
    else:
        return None
    if strong_against:
        return None

    recent = df.iloc[max(0, i - 2): i + 1]
    if direction == "SHORT":
        limit = float(recent["high"].max()) + params.limit_pad_atr * a
        stop = limit + params.sl_atr * a
        target = limit - params.rr * params.sl_atr * a
        stretch_detail = f"close {close:.6g} is {dist / a:.2f} ATR above VWAP {vwap:.6g}"
        rest_detail = (f"limit resting at {limit:.6g} — the 3-bar high "
                       f"{recent['high'].max():.6g} + {params.limit_pad_atr} ATR; "
                       f"the next push fills us at maker price or we skip")
    else:
        limit = float(recent["low"].min()) - params.limit_pad_atr * a
        stop = limit - params.sl_atr * a
        target = limit + params.rr * params.sl_atr * a
        stretch_detail = f"close {close:.6g} is {-dist / a:.2f} ATR below VWAP {vwap:.6g}"
        rest_detail = (f"limit resting at {limit:.6g} — the 3-bar low "
                       f"{recent['low'].min():.6g} - {params.limit_pad_atr} ATR; "
                       f"the next flush fills us at maker price or we skip")

    risk = abs(limit - stop)
    stop_bps = risk / limit * 10_000
    if target <= 0 or stop_bps < params.min_stop_bps:
        return None
    if risk / limit * 100 > params.max_stop_pct:
        return None

    factors = [
        Factor("VWAP stretch", True, 2, stretch_detail),
        Factor("Paid session", True, 1,
               f"{t.hour:02d}:00 UTC inside {params.hours} — the hours where "
               f"5m range covers its costs"),
        Factor("Live ATR", True, 1,
               f"ATR {atr_bps:.1f} bps (floor {params.min_atr_bps}) — the tape is moving"),
        Factor("Resting at the extreme", True, 1, rest_detail),
        Factor("HTF regime", True, 1,
               f"1h ADX {hrow['adx']:.1f} — no strong opposing trend "
               f"(veto at >= {params.max_htf_adx})"),
    ]
    return Signal(
        symbol=symbol, interval=interval, direction=direction,
        time=str(t), entry=limit, stop=round(stop, 8), target=round(target, 8),
        rr=params.rr, score=MAX_SCORE, max_score=MAX_SCORE,
        factors=factors, context=dict(context or {}), limit=limit,
    )


def latest_scalp_signal(df: pd.DataFrame, htf: pd.DataFrame, params: ScalpParams,
                        symbol: str, interval: str,
                        context: dict | None = None) -> Signal | None:
    return evaluate_scalp(df, htf, len(df) - 1, params, symbol, interval, context)


def grid_search_scalp(df: pd.DataFrame, htf: pd.DataFrame, symbol: str, interval: str,
                      stretch_values=(1.0, 1.5, 2.0),
                      sl_values=(0.6, 0.8, 1.2),
                      hours_values=("13-16", "13-21", "0-24"),
                      fee_pct: float = 0.02, slippage_bps: float = 1.0):
    """Sweep extension threshold, stop size, and session. Maker fees + low
    slippage by default: limit entries are the whole point of this strategy."""
    from .backtest import run_backtest

    results = []
    for st in stretch_values:
        for sl in sl_values:
            for hrs in hours_values:
                p = ScalpParams(stretch_atr=st, sl_atr=sl, hours=hrs)
                res = run_backtest(df, htf, p, symbol, interval,
                                   evaluate_fn=evaluate_scalp, max_hold=p.max_hold,
                                   fee_pct=fee_pct, slippage_bps=slippage_bps)
                results.append((p, res))
    results.sort(key=lambda pr: pr[1].expectancy_r * max(pr[1].n, 1), reverse=True)
    return results
