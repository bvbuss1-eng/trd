"""Offline verification of the whole pipeline using synthetic OHLCV data.

Run:  python -m tests.test_engine
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyst.backtest import BacktestResult, Trade, _simulate_exit, run_backtest  # noqa: E402
from analyst.charting import plot_chart  # noqa: E402
from analyst.indicators import enrich  # noqa: E402
from analyst.meanrev import (MeanRevParams, evaluate_meanrev,  # noqa: E402
                             latest_meanrev_signal)
from analyst.risk import position_size  # noqa: E402
from analyst.strategy import StrategyParams, latest_signal  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def synthetic_ohlcv(n: int = 2000, interval_min: int = 5, trend: float = 0.00012,
                    seed: int = 7) -> pd.DataFrame:
    """Trending random walk with pullback cycles — looks like a real crypto chart."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq=f"{interval_min}min", tz="UTC")
    wave = 0.002 * np.sin(np.arange(n) / 40)            # pullback cycles
    noise = rng.normal(0, 0.0018, n)
    logret = trend + noise + np.diff(wave, prepend=0)
    close = 50_000 * np.exp(np.cumsum(logret))
    o = np.roll(close, 1) * (1 + rng.normal(0, 0.0004, n))
    o[0] = close[0]
    spread = np.abs(rng.normal(0, 0.0012, n))
    hi = np.maximum(o, close) * (1 + spread)
    lo = np.minimum(o, close) * (1 - spread)
    vol = np.abs(rng.normal(100, 35, n)) * (1 + 4 * np.abs(logret) / 0.002)
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": close,
                       "volume": vol}, index=idx)
    df["close_time"] = (idx.view("int64") // 10**6) + interval_min * 60_000 - 1
    return df


def synthetic_meanrev_ohlcv(n: int = 4000, interval_min: int = 5,
                            seed: int = 11) -> pd.DataFrame:
    """Ornstein-Uhlenbeck (mean-reverting) log-price — a ranging market."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq=f"{interval_min}min", tz="UTC")
    theta, sigma = 0.06, 0.0028
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = x[k - 1] - theta * x[k - 1] + rng.normal(0, sigma)
    close = 50_000 * np.exp(x)
    o = np.roll(close, 1) * (1 + rng.normal(0, 0.0004, n))
    o[0] = close[0]
    spread = np.abs(rng.normal(0, 0.0012, n))
    hi = np.maximum(o, close) * (1 + spread)
    lo = np.minimum(o, close) * (1 - spread)
    ret = np.abs(np.diff(np.log(close), prepend=0))
    vol = np.abs(rng.normal(100, 35, n)) * (1 + 4 * ret / 0.002)
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": close,
                       "volume": vol}, index=idx)
    df["close_time"] = (idx.view("int64") // 10**6) + interval_min * 60_000 - 1
    return df


def resample_htf(df: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    agg = df.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum",
                                 "close_time": "last"}).dropna()
    return agg


def main() -> int:
    print("== indicators ==")
    df = synthetic_ohlcv()
    e = enrich(df)
    check("enrich adds columns", all(c in e for c in
          ("ema21", "rsi", "atr", "vwap", "adx", "vol_ratio", "swing_low")))
    check("rsi bounded", bool(e["rsi"].dropna().between(0, 100).all()))
    check("atr positive", bool((e["atr"].dropna() > 0).all()))
    check("no lookahead shape", len(e) == len(df))

    print("== exit simulation (hand-crafted candles) ==")
    idx = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
    sim = pd.DataFrame({
        "open":  [100, 101, 102, 103, 99, 98],
        "high":  [101, 103, 106, 104, 100, 99],
        "low":   [99, 100, 101, 102, 95, 97],
        "close": [101, 102, 105, 103, 96, 98],
    }, index=idx)
    # LONG from 100, stop 97, target 106 -> target hit on candle 2 (high 106)
    j, price, outcome = _simulate_exit(sim, 0, "LONG", 100, 97, 106, 10)
    check("long target hit", outcome == "win" and price == 106 and j == 2, f"{j} {price} {outcome}")
    # LONG from 100, stop 96, target 200 -> stop hit on candle 4 (low 95)
    j, price, outcome = _simulate_exit(sim, 0, "LONG", 100, 96, 200, 10)
    check("long stop hit", outcome == "loss" and j == 4, f"{j} {price} {outcome}")
    # ambiguous candle (both in range) counts as loss: stop 101.5 target 105.5, candle 2 spans both
    j, price, outcome = _simulate_exit(sim, 2, "LONG", 103, 101.5, 105.5, 10)
    check("ambiguous candle = loss (conservative)", outcome == "loss", f"{j} {price} {outcome}")
    # SHORT from 103 (idx 3): stop 104.5, target 96 -> candle 4 low 95 hits target first? candle 3 high 104 < 104.5 ok
    j, price, outcome = _simulate_exit(sim, 3, "SHORT", 103, 104.5, 96, 10)
    check("short target hit", outcome == "win" and price == 96, f"{j} {price} {outcome}")
    # time stop
    j, price, outcome = _simulate_exit(sim, 0, "LONG", 100, 1, 10_000, 3)
    check("time stop exits at close", outcome == "time" and j == 3, f"{j} {price} {outcome}")

    print("== strategy + backtest (synthetic uptrend) ==")
    htf = enrich(resample_htf(df))
    params = StrategyParams(min_score=4, min_adx=10, min_bb_width=0.0005)
    result = run_backtest(e, htf, params, "SYNTH", "5m")
    check("backtest produces trades", result.n > 0, f"n={result.n}")
    check("win rate sane", 0 <= result.win_rate <= 100)
    check("R accounting consistent",
          all(abs(t.r_multiple) < 10 for t in result.trades))
    losses = [t for t in result.trades if t.outcome == "loss"]
    if losses:
        check("losses cost ~-1R (plus fees, minus gap noise)",
              all(-2.5 < t.r_multiple <= 0.1 for t in losses),
              str([round(t.r_multiple, 2) for t in losses[:5]]))
    wins = [t for t in result.trades if t.outcome == "win"]
    if wins:
        check("wins pay ~+rr R net of fees",
              all(t.r_multiple > params.rr * 0.5 for t in wins),
              str([round(t.r_multiple, 2) for t in wins[:5]]))
    print(result.report())

    print("== mean-reversion strategy (synthetic ranging market) ==")
    mr_df = enrich(synthetic_meanrev_ohlcv())
    mr_htf = enrich(resample_htf(synthetic_meanrev_ohlcv()))
    mr_params = MeanRevParams()
    mr = run_backtest(mr_df, mr_htf, mr_params, "SYNTH", "5m",
                      evaluate_fn=evaluate_meanrev, max_hold=mr_params.max_hold)
    check("meanrev produces trades", mr.n >= 15, f"n={mr.n}")
    check("meanrev rr < 1 by construction", mr_params.rr < 1.0)
    check("meanrev win rate >= 60% on mean-reverting data",
          mr.win_rate >= 60.0, f"win_rate={mr.win_rate:.1f}")
    check("meanrev expectancy positive on mean-reverting data",
          mr.expectancy_r > 0, f"exp={mr.expectancy_r:+.3f} R")
    if mr.trades:
        t0 = mr.trades[0]
        ratio = abs(t0.target - t0.entry) / abs(t0.entry - t0.stop)
        check("meanrev tp/sl geometry",
              math.isclose(ratio, mr_params.rr, rel_tol=1e-6), f"ratio={ratio:.3f}")
    # no lookahead: a signal at row i must be identical when future rows are removed
    sig_full, i_sig = None, None
    for i in range(len(mr_df) - 2, 100, -1):
        sig_full = evaluate_meanrev(mr_df, mr_htf, i, mr_params, "SYNTH", "5m")
        if sig_full:
            i_sig = i
            break
    check("meanrev finds a signal in history", sig_full is not None)
    if sig_full:
        sig_trunc = evaluate_meanrev(mr_df.iloc[: i_sig + 1], mr_htf, i_sig,
                                     mr_params, "SYNTH", "5m")
        check("meanrev no lookahead",
              sig_trunc is not None and sig_trunc.entry == sig_full.entry
              and sig_trunc.stop == sig_full.stop,
              "signal changed when future rows were removed")
        check("meanrev live path works",
              latest_meanrev_signal(mr_df.iloc[: i_sig + 1], mr_htf, mr_params,
                                    "SYNTH", "5m") is not None)
    print(mr.report())

    print("== live signal path ==")
    sig = None
    # walk backwards to find any candle that produced a signal via the live API
    for i in range(len(e) - 1, 100, -1):
        sig = latest_signal(e.iloc[: i + 1], htf, params, "SYNTH", "5m")
        if sig:
            break
    check("latest_signal returns a Signal somewhere in history", sig is not None)
    if sig:
        check("signal RR geometry", math.isclose(
            abs(sig.target - sig.entry), params.rr * abs(sig.entry - sig.stop), rel_tol=1e-6))
        report = sig.to_report(position_size(500, 1.0, sig.entry, sig.stop))
        check("report renders", "WHY" in report and "Stop-loss" in report)
        print(report)

    print("== market structure research ==")
    from analyst.structure import structure_report
    rep = structure_report(e, enrich(resample_htf(df, "1h")), "SYNTH", "5m")
    check("structure report renders all sections",
          all(s in rep for s in ("WHEN IT MOVES", "LIQUIDITY SWEEPS",
                                 "BREAKOUT FOLLOW-THROUGH", "AUTOCORRELATION",
                                 "REGIME SHARE", "PAY FOR ITSELF")))
    check("structure report has sweep events", "events" in rep)

    print("== risk sizing ==")
    s = position_size(500, 1.0, entry=100.0, stop=99.0, max_leverage=10)
    check("sizing math", "5.00" in s["risk per trade"] and "1.0x" in s["leverage needed"], str(s))
    s2 = position_size(500, 1.0, entry=100.0, stop=99.9, max_leverage=10)
    check("leverage cap engages", "CAPPED" in s2["leverage needed"], str(s2))

    print("== charting ==")
    out = "/tmp/claude-0/-home-user-trd/ba00ed87-3226-5b84-8f75-0e4cb23073e7/scratchpad/synth_chart.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if sig:
        plot_chart(e, "SYNTH", "5m", out, entry=sig.entry, stop=sig.stop, target=sig.target,
                   title_extra=f"{sig.direction} {sig.score}/{sig.max_score}")
    else:
        plot_chart(e, "SYNTH", "5m", out)
    check("chart PNG written", os.path.exists(out) and os.path.getsize(out) > 20_000)
    print(f"  chart at {out}")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
