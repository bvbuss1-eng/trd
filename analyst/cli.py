"""Command-line interface.

  python -m analyst analyze BTCUSDC            full analysis of one symbol now
  python -m analyst scan BTCUSDC ETHUSDC ...   scan symbols for live setups
  python -m analyst backtest BTCUSDC --days 30 measure the strategy honestly
  python -m analyst backtest BTCUSDC --optimize  parameter sweep
  python -m analyst chart BTCUSDC              render a chart PNG

Common flags: --interval 5m --htf 15m --capital 500 --risk-pct 1 --rr 2
              --ai (send analysis to Claude for a second opinion)
              --market futures|spot
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .backtest import grid_search, run_backtest
from .charting import plot_chart
from .data import BinanceData, BinanceError
from .fade import (FadeParams, evaluate_fade, grid_search_fade,
                   latest_fade_signal)
from .indicators import enrich
from .meanrev import (MeanRevParams, evaluate_meanrev, grid_search_meanrev,
                      latest_meanrev_signal)
from .risk import position_size
from .scalp import (ScalpParams, evaluate_scalp, grid_search_scalp,
                    latest_scalp_signal)
from .strategy import StrategyParams, evaluate, latest_signal
from .sweep import (SweepParams, evaluate_sweep, grid_search_sweep,
                    latest_sweep_signal)


def _strategy_from_args(args):
    """Returns (params, evaluate_fn, latest_fn, max_hold) for the chosen strategy."""
    if args.strategy == "scalp":
        params = ScalpParams(hours=args.hours or "13-16",
                             stretch_atr=args.stretch_atr or 1.5,
                             sl_atr=args.sl_atr or 0.8, rr=args.rr,
                             min_atr_bps=args.min_atr_bps,
                             limit_pad_atr=args.limit_pad_atr,
                             max_hold=args.max_hold or 24)
        return params, evaluate_scalp, latest_scalp_signal, params.max_hold
    if args.strategy == "fade":
        params = FadeParams(stretch_atr=args.stretch_atr or 2.0, tp_frac=args.tp_frac,
                            confluence_min=args.confluence_min,
                            hours=args.hours or "0-24",
                            stop_pad_atr=args.stop_pad_atr,
                            allow_longs=args.allow_longs,
                            max_hold=args.max_hold or 12)
        return params, evaluate_fade, latest_fade_signal, params.max_hold
    if args.strategy == "sweep":
        params = SweepParams(rr=args.rr, hours=args.hours or "21-7",
                             stop_pad_atr=args.stop_pad_atr,
                             lookback=args.sweep_lookback,
                             max_hold=args.max_hold or 12)
        return params, evaluate_sweep, latest_sweep_signal, params.max_hold
    if args.strategy == "meanrev":
        params = MeanRevParams(tp_atr=args.tp_atr or 1.2, sl_atr=args.sl_atr or 2.0,
                               rsi_long_max=args.rsi_fade,
                               rsi_short_min=100.0 - args.rsi_fade,
                               max_hold=args.max_hold or 36)
        return params, evaluate_meanrev, latest_meanrev_signal, params.max_hold
    params = StrategyParams(rr=args.rr, min_score=args.min_score, min_adx=args.min_adx)
    return params, evaluate, latest_signal, args.max_hold or 48


def _load(client: BinanceData, symbol: str, interval: str, htf: str, days: float):
    df = enrich(client.klines_range(symbol, interval, days))
    htf_df = enrich(client.klines_range(symbol, htf, max(days, 5)))
    return df, htf_df


def _context(client: BinanceData, symbol: str) -> dict:
    ctx: dict = {}
    fr = client.funding_rate(symbol)
    if fr:
        ctx["funding rate"] = f"{float(fr.get('lastFundingRate', 0)) * 100:.4f}% (mark {fr.get('markPrice')})"
    oi = client.open_interest(symbol)
    if oi:
        ctx["open interest"] = oi.get("openInterest")
    ls = client.long_short_ratio(symbol)
    if ls:
        ctx["top-trader long/short ratio"] = ls.get("longShortRatio")
    imb = client.order_book_imbalance(symbol)
    if imb is not None:
        ctx["order-book imbalance"] = f"{imb:+.2f} ({'bid' if imb > 0 else 'ask'}-heavy)"
    try:
        t = client.ticker_24h(symbol)
        ctx["24h change"] = f"{t.get('priceChangePercent')}%"
        ctx["24h quote volume"] = t.get("quoteVolume")
    except BinanceError:
        pass
    return ctx


def cmd_analyze(args) -> int:
    client = BinanceData(market=args.market)
    params, _, latest_fn, _ = _strategy_from_args(args)
    df, htf_df = _load(client, args.symbol, args.interval, args.htf, args.days)
    ctx = _context(client, args.symbol)
    sig = latest_fn(df, htf_df, params, args.symbol, args.interval, ctx)

    out_png = os.path.join(args.outdir, f"{args.symbol}_{args.interval}.png")
    os.makedirs(args.outdir, exist_ok=True)

    if sig:
        sizing = position_size(args.capital, args.risk_pct, sig.entry, sig.stop,
                               max_leverage=args.max_leverage)
        plot_chart(df, args.symbol, args.interval, out_png,
                   entry=sig.entry, stop=sig.stop, target=sig.target,
                   title_extra=f"{sig.direction} signal {sig.score}/{sig.max_score}")
        print(sig.to_report(sizing))
        print(f"\n  Chart saved: {out_png}")
        if args.ai:
            _run_ai(sig.to_dict(), df, ctx, out_png, args)
    else:
        last = df.iloc[-1]
        plot_chart(df, args.symbol, args.interval, out_png)
        print(f"No qualifying setup on {args.symbol} {args.interval} right now.")
        print(f"  close {last['close']:.6g} | RSI {last['rsi']:.1f} | "
              f"ADX(HTF) {htf_df.iloc[-1]['adx']:.1f} | vol {last['vol_ratio']:.2f}x avg")
        for k, v in ctx.items():
            print(f"  {k}: {v}")
        print(f"  Chart saved: {out_png}")
        if args.ai:
            snapshot = {"signal": None, "note": "no rule-based setup; general read requested",
                        "market_context": ctx, "last_candles": _tail_json(df)}
            _run_ai(snapshot, df, ctx, out_png, args, wrap=False)
    return 0


def _tail_json(df, n: int = 30) -> list[dict]:
    cols = ["open", "high", "low", "close", "volume", "ema9", "ema21", "ema50",
            "rsi", "atr", "vwap", "adx", "vol_ratio"]
    tail = df[cols].tail(n).round(6)
    return [{"time": str(idx), **row.to_dict()} for idx, row in tail.iterrows()]


def _run_ai(payload: dict, df, ctx: dict, chart_png: str, args, wrap: bool = True) -> None:
    from .ai import assess_signal, format_assessment
    snapshot = ({"signal": payload, "market_context": ctx,
                 "last_candles": _tail_json(df)} if wrap else payload)
    print("\nAsking Claude for a second opinion…")
    try:
        assessment = assess_signal(snapshot, chart_png, model=args.model)
    except Exception as exc:  # keep the rule-based report usable without AI
        print(f"  AI layer unavailable: {exc}")
        return
    print(format_assessment(assessment))


def cmd_scan(args) -> int:
    client = BinanceData(market=args.market)
    params, _, latest_fn, _ = _strategy_from_args(args)
    hits = 0
    for symbol in args.symbols:
        try:
            df, htf_df = _load(client, symbol, args.interval, args.htf, args.days)
            sig = latest_fn(df, htf_df, params, symbol, args.interval)
        except BinanceError as exc:
            print(f"  {symbol}: data error — {exc}")
            continue
        if sig:
            hits += 1
            sizing = position_size(args.capital, args.risk_pct, sig.entry, sig.stop,
                                   max_leverage=args.max_leverage)
            print(sig.to_report(sizing))
        else:
            print(f"  {symbol}: no setup (close {df['close'].iloc[-1]:.6g})")
    print(f"\n{hits} setup(s) found across {len(args.symbols)} symbols.")
    return 0


def cmd_backtest(args) -> int:
    client = BinanceData(market=args.market)
    df, htf_df = _load(client, args.symbol, args.interval, args.htf, args.days)
    if args.optimize:
        print(f"Grid search on {args.symbol} {args.interval}, {args.days} days "
              f"({args.strategy})…\n")
        if args.strategy == "scalp":
            rows = grid_search_scalp(df, htf_df, args.symbol, args.interval,
                                     fee_pct=args.fee_pct, slippage_bps=args.slippage_bps)
            for p, r in rows[:12]:
                print(f"  stretch={p.stretch_atr:<4} sl={p.sl_atr:<4} hours={p.hours:<8} "
                      f"-> {r.n:>3} trades, {r.win_rate:5.1f}% win, "
                      f"{r.expectancy_r:+.3f} R/trade, total {sum(t.r_multiple for t in r.trades):+.1f} R")
            print("\nRemember: fills are limit-modeled — fewer trades than signals is normal.")
        elif args.strategy == "fade":
            rows = grid_search_fade(df, htf_df, args.symbol, args.interval,
                                    fee_pct=args.fee_pct, slippage_bps=args.slippage_bps)
            for p, r in rows[:12]:
                print(f"  stretch={p.stretch_atr:<4} tp={p.tp_frac:<5} "
                      f"confl={p.confluence_min} -> {r.n:>3} trades, "
                      f"{r.win_rate:5.1f}% win, {r.expectancy_r:+.3f} R/trade, "
                      f"total {sum(t.r_multiple for t in r.trades):+.1f} R")
        elif args.strategy == "sweep":
            rows = grid_search_sweep(df, htf_df, args.symbol, args.interval,
                                     fee_pct=args.fee_pct, slippage_bps=args.slippage_bps)
            for p, r in rows[:12]:
                print(f"  rr={p.rr:<4} pad={p.stop_pad_atr:<4} hours={p.hours:<8} "
                      f"-> {r.n:>3} trades, {r.win_rate:5.1f}% win, "
                      f"{r.expectancy_r:+.3f} R/trade, total {sum(t.r_multiple for t in r.trades):+.1f} R")
            print("\nWatch the hours column — it tests the session thesis directly.")
        elif args.strategy == "meanrev":
            rows = grid_search_meanrev(df, htf_df, args.symbol, args.interval,
                                       fee_pct=args.fee_pct, slippage_bps=args.slippage_bps)
            for p, r in rows[:10]:
                print(f"  tp={p.tp_atr:<4} sl={p.sl_atr:<4} rsi={p.rsi_long_max:<4} "
                      f"(rr {p.rr:.2f}) -> {r.n:>3} trades, {r.win_rate:5.1f}% win, "
                      f"{r.expectancy_r:+.3f} R/trade, total {sum(t.r_multiple for t in r.trades):+.1f} R")
            print("\nThe best WIN RATE row and the best EXPECTANCY row usually differ —")
            print("a 60%+ win rate only matters if the R/trade is also positive.")
        else:
            for p, r in grid_search(df, htf_df, args.symbol, args.interval)[:8]:
                print(f"  rr={p.rr:<4} min_adx={p.min_adx:<5} min_score={p.min_score} "
                      f"-> {r.n:>3} trades, {r.win_rate:5.1f}% win, "
                      f"{r.expectancy_r:+.3f} R/trade, total {sum(t.r_multiple for t in r.trades):+.1f} R")
        print("\nPrefer STABLE parameter regions over the single best row (overfitting).")
        return 0
    params, evaluate_fn, _, max_hold = _strategy_from_args(args)
    result = run_backtest(df, htf_df, params, args.symbol, args.interval,
                          evaluate_fn=evaluate_fn, max_hold=max_hold,
                          fee_pct=args.fee_pct, slippage_bps=args.slippage_bps)
    print(result.report())
    if args.trades and result.trades:
        print("\n  Individual trades:")
        for t in result.trades:
            print(f"    {t.entry_time}  {t.direction:<5} entry {t.entry:.6g} "
                  f"-> {t.exit_price:.6g}  {t.outcome:<4}  {t.r_multiple:+.2f} R")
    if args.json:
        print(json.dumps([t.__dict__ for t in result.trades], default=str))
    return 0


def cmd_structure(args) -> int:
    from .structure import structure_report
    client = BinanceData(market=args.market)
    df, htf_df = _load(client, args.symbol, args.interval, args.htf, args.days)
    print(structure_report(df, htf_df, args.symbol, args.interval,
                           sweep_lookback=args.sweep_lookback, horizon=args.horizon))
    return 0


def cmd_patterns(args) -> int:
    from .patterns import patterns_report
    client = BinanceData(market=args.market)
    df = enrich(client.klines_range(args.symbol, args.interval, args.days))
    print(patterns_report(df, args.symbol, args.interval, horizon=args.horizon))
    return 0


def cmd_chart(args) -> int:
    client = BinanceData(market=args.market)
    df = enrich(client.klines_range(args.symbol, args.interval, args.days))
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.symbol}_{args.interval}.png")
    plot_chart(df, args.symbol, args.interval, out)
    print(f"Chart saved: {out}")
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", default="pullback",
                   choices=["pullback", "meanrev", "sweep", "fade", "scalp"],
                   help="pullback = trend-following 1:2 RR; "
                        "meanrev = high-win-rate mean reversion (RR < 1); "
                        "sweep = liquidity-sweep reversal (tight stop, session-filtered); "
                        "fade = VWAP-stretch fade (Pattern Lab composite); "
                        "scalp = 5m NY-session limit-entry scalp (1:2 RR)")
    p.add_argument("--stretch-atr", type=float, default=None,
                   help="[fade/scalp] setup when |close-VWAP| >= this many ATRs "
                        "(default: 2.0 fade, 1.5 scalp)")
    p.add_argument("--min-atr-bps", type=float, default=15.0,
                   help="[scalp] no trade when ATR below this, in bps")
    p.add_argument("--limit-pad-atr", type=float, default=0.15,
                   help="[scalp] rest the limit this far beyond the 3-bar extreme")
    p.add_argument("--tp-frac", type=float, default=0.75,
                   help="[fade] target this fraction of the way back to VWAP")
    p.add_argument("--confluence-min", type=int, default=1,
                   help="[fade] required confluences (sweep/streak/wick, 0-3)")
    p.add_argument("--allow-longs", action="store_true",
                   help="[fade] enable the long side (failed the 90d exam; opt-in)")
    p.add_argument("--tp-atr", type=float, default=None,
                   help="[meanrev] take-profit distance in ATRs (default 1.2)")
    p.add_argument("--sl-atr", type=float, default=None,
                   help="[meanrev/scalp] stop-loss distance in ATRs "
                        "(default: 2.0 meanrev, 0.8 scalp)")
    p.add_argument("--rsi-fade", type=float, default=32.0,
                   help="[meanrev] fade when RSI <= this (longs) / >= 100-this (shorts)")
    p.add_argument("--hours", default=None,
                   help="allowed entry hours UTC, e.g. '21-7' or '13-16' "
                        "(default: 21-7 sweep, 0-24 fade, 13-16 scalp)")
    p.add_argument("--stop-pad-atr", type=float, default=0.3,
                   help="[sweep] stop distance beyond the sweep wick, in ATRs")
    p.add_argument("--sweep-lookback", type=int, default=24,
                   help="bars defining the prior swing extreme (sweep/structure)")
    p.add_argument("--max-hold", type=int, default=None,
                   help="time-stop in entry-TF candles (default: 36 meanrev, 48 pullback)")
    p.add_argument("--fee-pct", type=float, default=0.05,
                   help="fee per side, %% of notional (0.05 taker, 0.02 maker)")
    p.add_argument("--slippage-bps", type=float, default=2.0,
                   help="slippage per side, basis points")
    p.add_argument("--interval", default="5m", help="entry timeframe (default 5m)")
    p.add_argument("--htf", default="15m", help="trend timeframe (default 15m)")
    p.add_argument("--days", type=float, default=3.0, help="history to load")
    p.add_argument("--market", default="futures", choices=["futures", "spot"])
    p.add_argument("--capital", type=float, default=500.0)
    p.add_argument("--risk-pct", type=float, default=1.0)
    p.add_argument("--max-leverage", type=float, default=10.0)
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--min-score", type=int, default=7)
    p.add_argument("--min-adx", type=float, default=18.0)
    p.add_argument("--outdir", default="charts")
    p.add_argument("--ai", action="store_true", help="ask Claude for a second opinion")
    p.add_argument("--model", default="claude-opus-4-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analyst", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="full analysis of one symbol")
    p.add_argument("symbol")
    _add_common(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("scan", help="scan symbols for live setups")
    p.add_argument("symbols", nargs="+")
    _add_common(p)
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("backtest", help="walk-forward backtest")
    p.add_argument("symbol")
    p.add_argument("--optimize", action="store_true", help="parameter grid search")
    p.add_argument("--trades", action="store_true", help="print each trade")
    p.add_argument("--json", action="store_true")
    _add_common(p)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("structure", help="market-structure research report "
                       "(sessions, sweeps, breakouts, regimes, fee floor)")
    p.add_argument("symbol")
    p.add_argument("--horizon", type=int, default=12,
                   help="bars to measure the outcome after each event (default 12)")
    _add_common(p)
    p.set_defaults(func=cmd_structure)

    p = sub.add_parser("patterns", help="Pattern Lab: measure the classic "
                       "pattern catalog with significance stats")
    p.add_argument("symbol")
    p.add_argument("--horizon", type=int, default=12,
                   help="bars to measure the outcome after each event (default 12)")
    _add_common(p)
    p.set_defaults(func=cmd_patterns)

    p = sub.add_parser("chart", help="render a chart PNG")
    p.add_argument("symbol")
    _add_common(p)
    p.set_defaults(func=cmd_chart)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BinanceError as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
