"""Walk-forward backtester for the confluence strategy.

Honest by construction:
  - Signals are generated candle-by-candle with the exact live code path.
  - Entries fill at the NEXT candle's open (you can't buy a close you just saw).
  - If a candle touches both stop and target, it counts as a LOSS (conservative).
  - Taker fees + slippage are charged on both sides.
  - One position at a time; optional time-stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import StrategyParams, evaluate

# Any params object with an `rr` attribute and a `to_dict()` works here
# (StrategyParams, meanrev.MeanRevParams, ...).


@dataclass
class Trade:
    symbol: str
    direction: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry: float
    stop: float
    target: float
    exit_price: float
    outcome: str          # "win" | "loss" | "time"
    r_multiple: float     # net of costs, in units of initial risk
    score: int


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    params: object | None = None
    symbol: str = ""
    interval: str = ""
    candles: int = 0

    @property
    def n(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "win")

    @property
    def time_exits(self) -> int:
        return sum(1 for t in self.trades if t.outcome == "time")

    @property
    def win_rate(self) -> float:
        """Wins over ALL trades — time-stop exits count against the win rate."""
        return self.wins / self.n * 100 if self.n else 0.0

    @property
    def expectancy_r(self) -> float:
        return float(np.mean([t.r_multiple for t in self.trades])) if self.n else 0.0

    @property
    def profit_factor(self) -> float:
        gains = sum(t.r_multiple for t in self.trades if t.r_multiple > 0)
        losses = -sum(t.r_multiple for t in self.trades if t.r_multiple < 0)
        return gains / losses if losses else float("inf")

    def equity_curve_r(self) -> list[float]:
        eq, out = 0.0, []
        for t in self.trades:
            eq += t.r_multiple
            out.append(eq)
        return out

    @property
    def max_drawdown_r(self) -> float:
        eq = self.equity_curve_r()
        peak, dd = 0.0, 0.0
        for v in eq:
            peak = max(peak, v)
            dd = max(dd, peak - v)
        return dd

    def report(self) -> str:
        lines = [
            "=" * 62,
            f"BACKTEST — {self.symbol} {self.interval}  ({self.candles} candles)",
            "=" * 62,
        ]
        if self.params:
            lines.append(f"  params: {self.params.to_dict()}")
        if not self.n:
            lines.append("  No trades generated. Loosen min_score or extend the window.")
            return "\n".join(lines)
        longs = [t for t in self.trades if t.direction == "LONG"]
        shorts = [t for t in self.trades if t.direction == "SHORT"]
        lines += [
            f"  Trades          : {self.n}  (long {len(longs)} / short {len(shorts)})",
            f"  Win rate        : {self.win_rate:.1f}%  ({self.wins}W / "
            f"{self.n - self.wins - self.time_exits}L / {self.time_exits} time-outs)",
            f"  Expectancy      : {self.expectancy_r:+.3f} R per trade (net of fees)",
            f"  Profit factor   : {self.profit_factor:.2f}",
            f"  Total           : {sum(t.r_multiple for t in self.trades):+.2f} R",
            f"  Max drawdown    : {self.max_drawdown_r:.2f} R",
        ]
        for label, subset in (("LONG", longs), ("SHORT", shorts)):
            if subset:
                wr = sum(1 for t in subset if t.outcome == "win") / len(subset) * 100
                exp = np.mean([t.r_multiple for t in subset])
                lines.append(f"    {label:<5}: {len(subset)} trades, {wr:.1f}% win, {exp:+.3f} R avg")
        lines.append("")
        lines.append("  Note: at 1% risk per trade, total R * 1% ≈ % account growth.")
        return "\n".join(lines)


def _simulate_exit(df: pd.DataFrame, entry_idx: int, direction: str,
                   entry: float, stop: float, target: float,
                   max_hold: int) -> tuple[int, float, str]:
    """Walk candles after entry until stop/target/time-stop. Conservative on ties."""
    last = min(len(df) - 1, entry_idx + max_hold)
    for j in range(entry_idx, last + 1):
        hi, lo, op = df["high"].iloc[j], df["low"].iloc[j], df["open"].iloc[j]
        if direction == "LONG":
            hit_stop = lo <= stop
            hit_tgt = hi >= target
            if hit_stop and hit_tgt:
                return j, min(op, stop), "loss"          # ambiguous candle -> assume worst
            if hit_stop:
                return j, min(op, stop), "loss"          # gap through stop fills at open
            if hit_tgt:
                return j, target, "win"
        else:
            hit_stop = hi >= stop
            hit_tgt = lo <= target
            if hit_stop and hit_tgt:
                return j, max(op, stop), "loss"
            if hit_stop:
                return j, max(op, stop), "loss"
            if hit_tgt:
                return j, target, "win"
    return last, float(df["close"].iloc[last]), "time"


def run_backtest(df: pd.DataFrame, htf: pd.DataFrame, params,
                 symbol: str, interval: str,
                 fee_pct: float = 0.05, slippage_bps: float = 2.0,
                 max_hold: int = 48, evaluate_fn=None) -> BacktestResult:
    """df/htf must already be indicator-enriched. Fees/slippage per side.

    evaluate_fn(df, htf, i, params, symbol, interval) -> Signal | None
    defaults to the trend-pullback strategy; pass meanrev.evaluate_meanrev
    (with MeanRevParams) to test the mean-reversion strategy instead.
    """
    evaluate_fn = evaluate_fn or evaluate
    result = BacktestResult(params=params, symbol=symbol, interval=interval, candles=len(df))
    cost_r_frac = (fee_pct / 100 * 2) + (slippage_bps / 10_000 * 2)  # fraction of notional
    i = 30
    while i < len(df) - 1:
        sig = evaluate_fn(df, htf, i, params, symbol, interval)
        if sig is None:
            i += 1
            continue
        entry_idx = i + 1                        # fill on next candle open
        fill = float(df["open"].iloc[entry_idx])
        risk = abs(sig.entry - sig.stop)
        # re-anchor stop/target to the actual fill, keeping the same distances
        if sig.direction == "LONG":
            stop, target = fill - risk, fill + risk * params.rr
        else:
            stop, target = fill + risk, fill - risk * params.rr
        exit_idx, exit_price, outcome = _simulate_exit(
            df, entry_idx, sig.direction, fill, stop, target, max_hold)
        pnl = (exit_price - fill) if sig.direction == "LONG" else (fill - exit_price)
        r = pnl / risk - (cost_r_frac * fill / risk)   # costs expressed in R
        result.trades.append(Trade(
            symbol=symbol, direction=sig.direction,
            signal_time=sig.time, entry_time=str(df.index[entry_idx]),
            exit_time=str(df.index[exit_idx]),
            entry=fill, stop=stop, target=target, exit_price=exit_price,
            outcome=outcome, r_multiple=float(r), score=sig.score,
        ))
        i = exit_idx + 1                          # one position at a time
    return result


def grid_search(df: pd.DataFrame, htf: pd.DataFrame, symbol: str, interval: str,
                rr_values=(1.5, 2.0, 2.5, 3.0),
                adx_values=(15.0, 18.0, 22.0),
                score_values=(6, 7)) -> list[tuple[StrategyParams, BacktestResult]]:
    """Small, transparent parameter sweep. Beware overfitting: prefer parameter
    regions that are stable, not the single best cell."""
    results = []
    for rr in rr_values:
        for adx_min in adx_values:
            for min_score in score_values:
                p = StrategyParams(rr=rr, min_adx=adx_min, min_score=min_score)
                r = run_backtest(df, htf, p, symbol, interval)
                results.append((p, r))
    results.sort(key=lambda pr: pr[1].expectancy_r * max(pr[1].n, 1), reverse=True)
    return results
