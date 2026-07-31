# AI Trading Analyst — Binance USDC Day Trading

A signal-generation and analysis system for intraday trading on Binance
perpetual futures (5m/15m trend-pullback style, 1:2–1:3 risk-reward).

**It never places orders.** It finds setups, explains *why* with a scored
confluence report, sizes the position for your capital and risk, draws the
chart, optionally asks Claude for a skeptical second opinion — and then **you**
decide and execute manually.

> ⚠️ **Disclaimer**: This is analysis software, not financial advice. Leverage
> trading can lose your entire capital. No win rate is guaranteed — that is
> exactly why the backtester exists: to measure what is real before you trust
> any signal with money.

---

## What it does

1. **Live market data** — public Binance endpoints (no API key needed):
   klines, funding rate, open interest, top-trader long/short ratio,
   order-book imbalance, 24h stats.
2. **Technical analysis** — EMA 9/21/50/200, RSI, MACD, ATR, Bollinger,
   session VWAP, ADX, volume profile, swing structure.
3. **Strategies** — two built in, selected with `--strategy`:

   **`pullback`** (default) — trend-pullback confluence, scored 0–8:
   - HTF (15m) trend alignment (2 pts, hard filter)
   - Pullback into the EMA21/VWAP value zone (2 pts)
   - Momentum trigger candle reclaiming prior high/low (2 pts)
   - Volume ≥ 1.1× average (1 pt)
   - Regime filter: HTF ADX + volatility floor, blocks chop (1 pt)
   - Default threshold 7/8 → a signal cannot fire while chasing extended price.

   **`meanrev`** — high-win-rate intraday mean reversion, scored 0–6:
   - Stretch: RSI extreme + outer Bollinger band pierce (2 pts, required)
   - Reversal trigger candle closing back inside the band (2 pts, required)
   - HTF regime veto: never fade a strong higher-timeframe trend (1 pt, required)
   - Climax volume bonus (1 pt)
   - Take-profit **nearer** than the stop (default 1.2 vs 2.0 ATR, RR 0.6) —
     the asymmetry is what buys the high win rate; the filters + fee math
     decide whether it is also *profitable* (see below).
4. **Risk management** — position sized so a full stop-out costs ~1% of
   capital; leverage computed and capped; fees estimated.
5. **Charting** — dark candlestick PNG with indicators and the trade levels
   drawn on it.
6. **Backtesting** — honest walk-forward engine: fills at next-candle open,
   ambiguous candles count as losses, fees + slippage charged both sides.
   Reports win rate, expectancy in R, profit factor, max drawdown. Includes a
   parameter grid search to iterate the strategy.
7. **AI analyst (optional)** — sends the chart image + all computed data to
   Claude, which returns a structured verdict: `ENTER / WAIT / SKIP`,
   confidence, reasoning, strengths, weaknesses, key levels, invalidation,
   and suggested level adjustments.

## Setup

```bash
pip install -r requirements.txt

# Optional, for the --ai second opinion:
export ANTHROPIC_API_KEY=sk-ant-...
```

Run from the repository root. Binance must be reachable from your network.

## Usage

```bash
# Full analysis of one pair right now (chart saved to ./charts/)
python -m analyst analyze BTCUSDC

# Same, plus Claude's structured second opinion on the setup
python -m analyst analyze BTCUSDC --ai

# Scan a watchlist for live setups
python -m analyst scan BTCUSDC ETHUSDC SOLUSDC XRPUSDC

# Measure the strategy before trusting it (30 days, 5m entries, 15m trend)
python -m analyst backtest BTCUSDC --days 30

# The high-win-rate mean-reversion strategy (15m entries recommended —
# on 5m the near target gets eaten by fees)
python -m analyst backtest BTCUSDT --strategy meanrev --days 60 --interval 15m --htf 1h

# Same, assuming maker/limit fills instead of taker
python -m analyst backtest BTCUSDT --strategy meanrev --days 60 --interval 15m --htf 1h --fee-pct 0.02

# Tune the win-rate/expectancy tradeoff (tp/sl/rsi sweep)
python -m analyst backtest BTCUSDT --strategy meanrev --days 90 --interval 15m --htf 1h --optimize

# Show every simulated trade
python -m analyst backtest BTCUSDC --days 30 --trades

# Tune parameters (grid search over RR / ADX / score threshold)
python -m analyst backtest BTCUSDC --days 60 --optimize

# Just render a chart
python -m analyst chart BTCUSDC
```

Common flags: `--interval 5m --htf 15m --capital 500 --risk-pct 1 --rr 2
--max-leverage 10 --min-score 7 --market futures`

If a USDC pair has thin data, the same analysis on the USDT pair
(e.g. `BTCUSDT`) is a valid proxy — the chart is nearly identical.

## Suggested daily workflow

1. `scan` your watchlist at the sessions you trade (London/NY opens tend to
   have the cleanest 5m trends).
2. When a signal appears, read the **WHY** factors — every pass/fail is shown.
3. Run with `--ai` and read Claude's counter-argument. If the AI says SKIP,
   understand its reason before overriding it.
4. If you take the trade, use the printed sizing (risk-based, not
   leverage-based) and place entry, stop-loss, and take-profit together.
5. Log the outcome. Re-run `backtest --optimize` weekly on recent data and
   prefer *stable* parameter regions, not the single best cell — that's how
   the win rate improves without overfitting.

Aim for 1–3 quality trades per day. No signal = no trade. The edge of an
emotionless system is precisely that it is allowed to do nothing.

## The 60% win-rate question

"Which indicator always gives 60% wins?" — none. But a **60%+ win rate is
easy to engineer**, because win rate is mostly geometry: put the take-profit
nearer than the stop-loss and the near level gets hit more often, even on
random entries. With TP at 1.2 ATR and SL at 2 ATR (RR 0.6), noise alone wins
~62% of the time. That is exactly what `--strategy meanrev` does — and why a
win rate on its own means nothing.

The number that decides whether you make money is **expectancy**:

```
expectancy per trade (R) = win_rate × (rr − c) − (1 − win_rate) × (1 + c)
breakeven win rate       = (1 + c) / (1 + rr)      c = round-trip cost / risk
```

At RR 0.6 with taker fees (`c ≈ 0.1`), breakeven is **68.8%** — so a "60%
win rate" strategy is a *losing* strategy at that geometry. The meanrev entry
filters (RSI + Bollinger stretch, reversal trigger, HTF trend veto) exist to
push the realized win rate above that line, and the backtester exists to tell
you whether they did — on your timeframe, your window, your fees. Levers that
move the result, in order of impact: trade 15m+ (bigger ATR → fees smaller in
R), use maker fills (`--fee-pct 0.02`), and run `--optimize` to find *stable*
tp/sl/rsi regions.

If the backtest says 60%+ **and** expectancy is positive across several
windows — believe it. If it says the win rate is bought at the price of
negative expectancy — believe that too. There is no setting that guarantees
either; anyone who says otherwise is selling something.

## Honest expectations

- The backtester is deliberately pessimistic (next-open fills, worst-case
  ambiguous candles, fees, slippage). If a configuration shows positive
  expectancy here, it has a real chance live; if it shows 60–70% win rate
  here, believe it — if not, don't trade it hoping otherwise.
- At 1% risk per trade, `total R × 1% ≈ % account growth` — e.g. +20 R over a
  month on $500 ≈ +$100, without needing dangerous leverage.

## Testing (offline)

```bash
python -m tests.test_engine
```

Runs the full pipeline on synthetic data: indicator math, exit simulation
(stop/target/ambiguous/time), backtest accounting in R, signal geometry,
sizing, and chart rendering. No network needed.

## Project layout

```
analyst/
  data.py        Binance public REST client (futures + spot fallback)
  indicators.py  EMA/RSI/MACD/ATR/BB/VWAP/ADX/volume/swings (no lookahead)
  strategy.py    trend-pullback confluence scoring -> Signal with "why"
  meanrev.py     high-win-rate mean reversion (RR < 1) -> Signal with "why"
  signal.py      Signal dataclass + human-readable report
  risk.py        risk-based position sizing + leverage cap
  backtest.py    walk-forward engine + grid search
  charting.py    dark candlestick PNG with levels
  ai.py          Claude analyst (vision + structured output), optional
  cli.py         analyze / scan / backtest / chart commands
tests/
  test_engine.py offline synthetic-data verification
```
