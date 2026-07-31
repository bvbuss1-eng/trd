"""High-win-rate intraday mean-reversion strategy (BTC/USDT and friends).

Answers the question "can we get a >=60% win rate?" the honest way:

  Win rate is mostly GEOMETRY. With a take-profit at 1.2 ATR and a stop at
  2 ATR (rr = 0.6), even random entries win ~62% of the time — the near
  target simply gets hit before the far stop. The hard part is not the win
  rate, it is keeping EXPECTANCY positive after fees. In R units, with c =
  round-trip cost / risk, breakeven win rate = (1 + c) / (1 + rr): at rr 0.6
  and c 0.1 you must win 68.8%, i.e. ~6 points better than noise. That gap
  is the entire job of the entry filters below. Consequences:
    - trade timeframes with a decent ATR (15m+; on 5m the target is so close
      that taker fees eat most of it),
    - prefer maker/limit fills where possible (backtest with --fee-pct 0.02),
    - a 60% win rate with rr 0.6 and taker fees is a LOSING strategy; the
      backtester is there to tell you which side of the line you are on.

Edge sources stacked on top of the geometry:
  1. Stretch: only fade a candle that is statistically overextended —
     RSI at an extreme AND price piercing the outer Bollinger band.
  2. Reversal trigger: wait for the candle that closes back inside the
     band in your direction (never catch the falling knife itself).
  3. HTF regime veto: never fade against a STRONG higher-timeframe trend
     (that is where mean reversion blows up).
  4. Optional climax volume: exhaustion spikes reverse harder.

Same contract as strategy.py: evaluate at row i reads data <= i only, the
same code runs live and in the backtester.

The docstring is the theory — run `python -m analyst backtest BTCUSDT
--strategy meanrev --days 60` to know what it ACTUALLY does before
trusting it. A 60% win rate with negative expectancy loses money.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .signal import Factor, Signal


@dataclass
class MeanRevParams:
    tp_atr: float = 1.2          # take-profit distance in ATRs (near)
    sl_atr: float = 2.0          # stop-loss distance in ATRs (far)
    rsi_long_max: float = 32.0   # long only when RSI <= this (oversold)
    rsi_short_min: float = 68.0  # short only when RSI >= this (overbought)
    stretch_window: int = 3      # band pierce must be within the last N candles
    max_htf_adx: float = 32.0    # veto fading a HTF trend stronger than this
    min_vol_ratio: float = 1.2   # climax volume bonus threshold
    min_score: int = 5           # of MAX_SCORE below (5 = all required factors)
    min_tp_pct: float = 0.15     # target must be at least this % away (fees floor)
    max_stop_pct: float = 1.5    # reject stops wider than this % (leverage safety)
    max_hold: int = 36           # time-stop, in entry-TF candles
    allow_shorts: bool = True

    @property
    def rr(self) -> float:
        """Reward:risk ratio, < 1 by design — this is what buys the win rate."""
        return self.tp_atr / self.sl_atr

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rr"] = round(self.rr, 3)
        return d


MAX_SCORE = 6  # stretch(2) + reversal trigger(2) + HTF regime(1) + climax volume(1)
# Default min_score=5 forces stretch + trigger + regime; volume is a bonus.


def _htf_row(htf: pd.DataFrame, i_time) -> pd.Series | None:
    """Most recent CLOSED higher-timeframe candle at i_time (no lookahead)."""
    prior = htf[htf["close_time"] <= i_time.value // 10**6] if hasattr(i_time, "value") else htf
    if len(prior) < 2:
        return None
    return prior.iloc[-1]


def evaluate_meanrev(df: pd.DataFrame, htf: pd.DataFrame, i: int,
                     params: MeanRevParams, symbol: str, interval: str,
                     context: dict | None = None) -> Signal | None:
    """Evaluate the CLOSED candle at integer position i. Frames must be enriched."""
    if i < 30 or i >= len(df):
        return None
    row = df.iloc[i]
    t = df.index[i]
    if pd.isna(row["bb_lower"]) or pd.isna(row["atr"]) or row["atr"] <= 0:
        return None

    hrow = _htf_row(htf, t)
    if hrow is None or pd.isna(hrow.get("adx")):
        return None

    window = df.iloc[max(0, i - params.stretch_window + 1): i + 1]

    # -------------------------------------------------- pick a side (or none)
    long_pierce = bool((window["low"] <= window["bb_lower"]).any())
    short_pierce = bool((window["high"] >= window["bb_upper"]).any())
    rsi_now = float(row["rsi"])
    rsi_min = float(window["rsi"].min())
    rsi_max = float(window["rsi"].max())

    if long_pierce and rsi_min <= params.rsi_long_max:
        direction = "LONG"
    elif params.allow_shorts and short_pierce and rsi_max >= params.rsi_short_min:
        direction = "SHORT"
    else:
        return None

    factors: list[Factor] = []
    if direction == "LONG":
        stretch = True
        stretch_detail = (f"low pierced lower Bollinger within last {params.stretch_window} "
                          f"candles, RSI dipped to {rsi_min:.1f} (<= {params.rsi_long_max})")
        trigger = row["close"] > row["open"] and row["close"] >= row["bb_lower"]
        trigger_detail = (f"bullish candle closed back inside the band "
                          f"({row['close']:.6g} >= {row['bb_lower']:.6g})" if trigger
                          else "no bullish close back inside the lower band yet")
        # veto: strong HTF downtrend — falling knives are not mean-reverting
        ema200_ok = pd.notna(hrow.get("ema200"))
        strong_against = bool(ema200_ok and hrow["ema50"] < hrow["ema200"]
                              and hrow["adx"] >= params.max_htf_adx)
        regime_detail = (f"HTF ADX {hrow['adx']:.1f} — "
                         + ("strong downtrend, fading vetoed"
                            if strong_against else f"no strong opposing trend (veto at ADX >= {params.max_htf_adx})"))
    else:
        stretch = True
        stretch_detail = (f"high pierced upper Bollinger within last {params.stretch_window} "
                          f"candles, RSI peaked at {rsi_max:.1f} (>= {params.rsi_short_min})")
        trigger = row["close"] < row["open"] and row["close"] <= row["bb_upper"]
        trigger_detail = (f"bearish candle closed back inside the band "
                          f"({row['close']:.6g} <= {row['bb_upper']:.6g})" if trigger
                          else "no bearish close back inside the upper band yet")
        ema200_ok = pd.notna(hrow.get("ema200"))
        strong_against = bool(ema200_ok and hrow["ema50"] > hrow["ema200"]
                              and hrow["adx"] >= params.max_htf_adx)
        regime_detail = (f"HTF ADX {hrow['adx']:.1f} — "
                         + ("strong uptrend, fading vetoed"
                            if strong_against else f"no strong opposing trend (veto at ADX >= {params.max_htf_adx})"))

    factors.append(Factor("Oversold/overbought stretch", stretch, 2, stretch_detail))
    factors.append(Factor("Reversal trigger candle", bool(trigger), 2, trigger_detail))
    factors.append(Factor("HTF regime permits fading", not strong_against, 1, regime_detail))

    vol_ok = bool(pd.notna(window["vol_ratio"]).any()
                  and (window["vol_ratio"].fillna(0) >= params.min_vol_ratio).any())
    factors.append(Factor(
        "Climax volume (bonus)", vol_ok, 1,
        f"volume spiked to {window['vol_ratio'].max():.2f}x its 20-bar average "
        f"during the stretch (bonus at >= {params.min_vol_ratio}x)"))
    factors.append(Factor(
        "RSI now", True, 0,
        f"RSI {rsi_now:.1f} — informational, not scored"))

    score = sum(f.weight for f in factors if f.passed)
    if score < params.min_score or not trigger:
        return None

    # ----------------------------------------------------------- trade levels
    entry = float(row["close"])
    a = float(row["atr"])
    if direction == "LONG":
        stop = entry - params.sl_atr * a
        target = entry + params.tp_atr * a
    else:
        stop = entry + params.sl_atr * a
        target = entry - params.tp_atr * a

    tp_pct = abs(target - entry) / entry * 100
    sl_pct = abs(entry - stop) / entry * 100
    if tp_pct < params.min_tp_pct:
        return None  # target too close — fees would eat the edge
    if sl_pct > params.max_stop_pct:
        return None  # stop too wide for an intraday leveraged trade

    return Signal(
        symbol=symbol, interval=interval, direction=direction,
        time=str(t), entry=entry, stop=round(stop, 8), target=round(target, 8),
        rr=params.rr, score=score, max_score=MAX_SCORE,
        factors=factors, context=dict(context or {}),
    )


def latest_meanrev_signal(df: pd.DataFrame, htf: pd.DataFrame, params: MeanRevParams,
                          symbol: str, interval: str,
                          context: dict | None = None) -> Signal | None:
    """Evaluate only the most recent closed candle (live scanning)."""
    return evaluate_meanrev(df, htf, len(df) - 1, params, symbol, interval, context)


def grid_search_meanrev(df: pd.DataFrame, htf: pd.DataFrame, symbol: str, interval: str,
                        tp_values=(1.0, 1.2, 1.5),
                        sl_values=(1.5, 2.0, 2.5),
                        rsi_values=(28.0, 32.0, 36.0),
                        fee_pct: float = 0.05, slippage_bps: float = 2.0):
    """Sweep the win-rate/expectancy tradeoff. rsi value r is used as the long
    threshold and 100-r as the short threshold. Prefer stable regions, and
    remember: the row with the best win rate is NOT automatically the row
    that makes money — check expectancy."""
    from .backtest import run_backtest  # local import to avoid a cycle

    results = []
    for tp in tp_values:
        for sl in sl_values:
            for r in rsi_values:
                p = MeanRevParams(tp_atr=tp, sl_atr=sl,
                                  rsi_long_max=r, rsi_short_min=100.0 - r)
                res = run_backtest(df, htf, p, symbol, interval,
                                   evaluate_fn=evaluate_meanrev, max_hold=p.max_hold,
                                   fee_pct=fee_pct, slippage_bps=slippage_bps)
                results.append((p, res))
    results.sort(key=lambda pr: pr[1].expectancy_r * max(pr[1].n, 1), reverse=True)
    return results
