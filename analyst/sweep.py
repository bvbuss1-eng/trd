"""Liquidity-sweep reversal strategy — built FROM the structure report.

The measured pattern (BTCUSDT 15m, 90d): price wicks through a prior
24-bar extreme (running the stops resting there), closes back inside, and
reverses — ~57-76% of the time in the quiet sessions (Asia, Late US), but
only ~39% in London where the same event is a genuine trend ignition.

Trade construction, and why the RR can be honest here:
  - Entry: the close of the reclaim candle (or better, a limit near the
    level — sweeps FILL limit orders; use maker fees in the backtest).
  - Stop: just beyond the sweep wick. The stops that lived there were just
    consumed — if price goes back through, the reversal thesis is wrong.
    This stop is TIGHT, so a 1.5-2R target is only ~0.6-1.2 ATR away —
    inside the measured reversal range, not a hope.
  - Session filter: default 21:00-07:00 UTC (Late US + Asia), where the
    pattern actually works. London is excluded by measurement, not vibes.
  - HTF veto: never fade a strong 1h trend.
  - Time stop: the edge was measured over ~12 bars; past that it's decayed.

Same no-lookahead contract as the other strategies: row i reads data <= i.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .signal import Factor, Signal


def parse_hours(spec: str) -> set[int]:
    """'21-7' -> {21,22,23,0..6}; '0-7,21-24' unions ranges; end-exclusive.
    Empty/invalid spec means all hours."""
    hours: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) % 25 for x in part.split("-", 1))
            if a <= b:
                hours.update(range(a, min(b, 24)))
            else:                       # wraps midnight, e.g. 21-7
                hours.update(range(a, 24))
                hours.update(range(0, b))
        else:
            hours.add(int(part) % 24)
    return hours or set(range(24))


@dataclass
class SweepParams:
    lookback: int = 24            # bars defining the swept extreme
    min_pierce_atr: float = 0.10  # wick must pierce >= this far beyond it (real sweep)
    stop_pad_atr: float = 0.30    # stop this far beyond the sweep wick
    rr: float = 1.5               # target = rr * risk (tight stop makes this honest)
    hours: str = "21-7"           # allowed entry hours, UTC (Late US + Asia)
    max_htf_adx: float = 35.0     # veto fading a 1h trend stronger than this
    max_hold: int = 12            # edge measured over ~12 bars; exit after
    min_stop_bps: float = 6.0     # stop can't be tighter than fees+noise
    max_stop_pct: float = 1.0
    allow_shorts: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


MAX_SCORE = 6  # sweep pierce(2) + reclaim close(2) + session window(1) + HTF regime(1)


def _htf_row(htf: pd.DataFrame, i_time) -> pd.Series | None:
    prior = htf[htf["close_time"] <= i_time.value // 10**6] if hasattr(i_time, "value") else htf
    if len(prior) < 2:
        return None
    return prior.iloc[-1]


def evaluate_sweep(df: pd.DataFrame, htf: pd.DataFrame, i: int,
                   params: SweepParams, symbol: str, interval: str,
                   context: dict | None = None) -> Signal | None:
    """Evaluate the CLOSED candle at position i. Frames must be enriched."""
    if i < max(30, params.lookback + 1) or i >= len(df):
        return None
    row = df.iloc[i]
    t = df.index[i]
    a = float(row["atr"]) if pd.notna(row["atr"]) else 0.0
    if a <= 0:
        return None

    hrow = _htf_row(htf, t)
    if hrow is None or pd.isna(hrow.get("adx")):
        return None

    window = df.iloc[i - params.lookback: i]      # prior bars, current excluded
    prior_low = float(window["low"].min())
    prior_high = float(window["high"].max())

    long_sweep = (row["low"] < prior_low - params.min_pierce_atr * a
                  and row["close"] > prior_low)
    short_sweep = (params.allow_shorts
                   and row["high"] > prior_high + params.min_pierce_atr * a
                   and row["close"] < prior_high)
    if long_sweep:
        direction = "LONG"
    elif short_sweep:
        direction = "SHORT"
    else:
        return None

    allowed = parse_hours(params.hours)
    in_session = t.hour in allowed
    ema200_ok = pd.notna(hrow.get("ema200"))
    if direction == "LONG":
        pierce = (prior_low - float(row["low"])) / a
        sweep_detail = (f"wick pierced the prior {params.lookback}-bar low "
                        f"{prior_low:.6g} by {pierce:.2f} ATR — stops were run")
        reclaim = row["close"] > row["open"]
        reclaim_detail = (f"closed back above the level at {row['close']:.6g} "
                          + ("with a bullish body" if reclaim else "but body is bearish"))
        strong_against = bool(ema200_ok and hrow["ema50"] < hrow["ema200"]
                              and hrow["adx"] >= params.max_htf_adx)
    else:
        pierce = (float(row["high"]) - prior_high) / a
        sweep_detail = (f"wick pierced the prior {params.lookback}-bar high "
                        f"{prior_high:.6g} by {pierce:.2f} ATR — stops were run")
        reclaim = row["close"] < row["open"]
        reclaim_detail = (f"closed back below the level at {row['close']:.6g} "
                          + ("with a bearish body" if reclaim else "but body is bullish"))
        strong_against = bool(ema200_ok and hrow["ema50"] > hrow["ema200"]
                              and hrow["adx"] >= params.max_htf_adx)

    factors = [
        Factor("Liquidity sweep", True, 2, sweep_detail),
        Factor("Reclaim close", bool(reclaim), 2, reclaim_detail),
        Factor("Session window", bool(in_session), 1,
               f"{t.hour:02d}:00 UTC — allowed hours {params.hours} "
               f"(pattern measured weakest in London)"),
        Factor("HTF regime permits fading", not strong_against, 1,
               f"1h ADX {hrow['adx']:.1f} (veto at >= {params.max_htf_adx} "
               f"against the trade)"),
    ]
    if not (reclaim and in_session and not strong_against):
        return None
    score = sum(f.weight for f in factors if f.passed)

    entry = float(row["close"])
    if direction == "LONG":
        wick = float(row["low"])
        stop = wick - params.stop_pad_atr * a
        risk = entry - stop
        target = entry + params.rr * risk
    else:
        wick = float(row["high"])
        stop = wick + params.stop_pad_atr * a
        risk = stop - entry
        target = entry - params.rr * risk

    stop_bps = risk / entry * 10_000
    if risk <= 0 or stop_bps < params.min_stop_bps:
        return None  # tighter than fees+noise can honestly resolve
    if risk / entry * 100 > params.max_stop_pct:
        return None

    return Signal(
        symbol=symbol, interval=interval, direction=direction,
        time=str(t), entry=entry, stop=round(stop, 8), target=round(target, 8),
        rr=params.rr, score=score, max_score=MAX_SCORE,
        factors=factors, context=dict(context or {}),
    )


def latest_sweep_signal(df: pd.DataFrame, htf: pd.DataFrame, params: SweepParams,
                        symbol: str, interval: str,
                        context: dict | None = None) -> Signal | None:
    return evaluate_sweep(df, htf, len(df) - 1, params, symbol, interval, context)


def grid_search_sweep(df: pd.DataFrame, htf: pd.DataFrame, symbol: str, interval: str,
                      rr_values=(1.2, 1.5, 2.0),
                      pad_values=(0.2, 0.3, 0.5),
                      hours_values=("21-7", "0-7", "13-21", "0-24"),
                      fee_pct: float = 0.02, slippage_bps: float = 2.0):
    """Sweep RR, stop pad, and session window. The hours column is the
    interesting one: it tests the session thesis out-of-sample of the
    structure report's exact horizon."""
    from .backtest import run_backtest

    results = []
    for rr in rr_values:
        for pad in pad_values:
            for hrs in hours_values:
                p = SweepParams(rr=rr, stop_pad_atr=pad, hours=hrs)
                res = run_backtest(df, htf, p, symbol, interval,
                                   evaluate_fn=evaluate_sweep, max_hold=p.max_hold,
                                   fee_pct=fee_pct, slippage_bps=slippage_bps)
                results.append((p, res))
    results.sort(key=lambda pr: pr[1].expectancy_r * max(pr[1].n, 1), reverse=True)
    return results
