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
from .indicators import enrich
from .risk import position_size
from .strategy import StrategyParams, latest_signal


def _params_from_args(args) -> StrategyParams:
    return StrategyParams(rr=args.rr, min_score=args.min_score, min_adx=args.min_adx)


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
    params = _params_from_args(args)
    df, htf_df = _load(client, args.symbol, args.interval, args.htf, args.days)
    ctx = _context(client, args.symbol)
    sig = latest_signal(df, htf_df, params, args.symbol, args.interval, ctx)

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
    params = _params_from_args(args)
    hits = 0
    for symbol in args.symbols:
        try:
            df, htf_df = _load(client, symbol, args.interval, args.htf, args.days)
            sig = latest_signal(df, htf_df, params, symbol, args.interval)
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
        print(f"Grid search on {args.symbol} {args.interval}, {args.days} days…\n")
        for p, r in grid_search(df, htf_df, args.symbol, args.interval)[:8]:
            print(f"  rr={p.rr:<4} min_adx={p.min_adx:<5} min_score={p.min_score} "
                  f"-> {r.n:>3} trades, {r.win_rate:5.1f}% win, "
                  f"{r.expectancy_r:+.3f} R/trade, total {sum(t.r_multiple for t in r.trades):+.1f} R")
        print("\nPrefer STABLE parameter regions over the single best row (overfitting).")
        return 0
    params = _params_from_args(args)
    result = run_backtest(df, htf_df, params, args.symbol, args.interval)
    print(result.report())
    if args.trades and result.trades:
        print("\n  Individual trades:")
        for t in result.trades:
            print(f"    {t.entry_time}  {t.direction:<5} entry {t.entry:.6g} "
                  f"-> {t.exit_price:.6g}  {t.outcome:<4}  {t.r_multiple:+.2f} R")
    if args.json:
        print(json.dumps([t.__dict__ for t in result.trades], default=str))
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
