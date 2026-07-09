"""Binance public market-data client.

Uses only PUBLIC endpoints — no API key, no account access, no order placement.
Futures (USDT/USDC perpetuals) via fapi.binance.com; spot fallback via
api.binance.com and the data-api.binance.vision mirror.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
import requests

FUTURES_BASE = "https://fapi.binance.com"
SPOT_BASES = ["https://api.binance.com", "https://data-api.binance.vision"]

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


class BinanceError(RuntimeError):
    pass


def _klines_to_df(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
    for col in ("open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"):
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("time")
    return df[["open", "high", "low", "close", "volume", "quote_volume",
               "trades", "taker_buy_base", "close_time"]]


@dataclass
class BinanceData:
    """Public market-data access. market: 'futures' (default) or 'spot'."""

    market: str = "futures"
    timeout: float = 15.0
    session: requests.Session = field(default_factory=requests.Session)

    def _get(self, base: str, path: str, params: dict) -> object:
        try:
            resp = self.session.get(base + path, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise BinanceError(f"network error calling {base}{path}: {exc}") from exc
        if resp.status_code != 200:
            raise BinanceError(f"{base}{path} -> HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _market_get(self, spot_path: str, fut_path: str, params: dict) -> object:
        if self.market == "futures":
            return self._get(FUTURES_BASE, fut_path, params)
        last_err: Exception | None = None
        for base in SPOT_BASES:
            try:
                return self._get(base, spot_path, params)
            except BinanceError as exc:
                last_err = exc
        raise BinanceError(f"all spot endpoints failed: {last_err}")

    # ------------------------------------------------------------------ klines

    def klines(self, symbol: str, interval: str = "5m", limit: int = 500,
               start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
        params: dict = {"symbol": symbol.upper(), "interval": interval,
                        "limit": min(limit, 1500 if self.market == "futures" else 1000)}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        raw = self._market_get("/api/v3/klines", "/fapi/v1/klines", params)
        if not raw:
            raise BinanceError(f"no klines returned for {symbol} {interval}")
        return _klines_to_df(raw)

    def klines_range(self, symbol: str, interval: str, days: float,
                     end_ms: int | None = None, pause: float = 0.25) -> pd.DataFrame:
        """Fetch `days` of history by paginating backwards-compatible forward requests."""
        step = INTERVAL_MS[interval]
        end_ms = end_ms or int(time.time() * 1000)
        start_ms = end_ms - int(days * 86_400_000)
        frames: list[pd.DataFrame] = []
        cursor = start_ms
        per_req = 1500 if self.market == "futures" else 1000
        while cursor < end_ms:
            df = self.klines(symbol, interval, limit=per_req,
                             start_ms=cursor, end_ms=end_ms)
            frames.append(df)
            last_open = int(df["close_time"].iloc[-1])
            if last_open <= cursor:
                break
            cursor = last_open + 1
            if len(df) < per_req:
                break
            time.sleep(pause)  # stay well under public rate limits
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="first")].sort_index()
        # drop the still-forming candle (no lookahead into incomplete bars)
        now_ms = int(time.time() * 1000)
        return out[out["close_time"] <= now_ms]

    # --------------------------------------------------------- market context

    def ticker_24h(self, symbol: str) -> dict:
        return self._market_get("/api/v3/ticker/24hr", "/fapi/v1/ticker/24hr",
                                {"symbol": symbol.upper()})

    def order_book(self, symbol: str, limit: int = 100) -> dict:
        return self._market_get("/api/v3/depth", "/fapi/v1/depth",
                                {"symbol": symbol.upper(), "limit": limit})

    def funding_rate(self, symbol: str) -> dict | None:
        """Futures only: current mark price + funding rate."""
        if self.market != "futures":
            return None
        try:
            return self._get(FUTURES_BASE, "/fapi/v1/premiumIndex",
                             {"symbol": symbol.upper()})
        except BinanceError:
            return None

    def open_interest(self, symbol: str) -> dict | None:
        if self.market != "futures":
            return None
        try:
            return self._get(FUTURES_BASE, "/fapi/v1/openInterest",
                             {"symbol": symbol.upper()})
        except BinanceError:
            return None

    def long_short_ratio(self, symbol: str, period: str = "15m") -> dict | None:
        """Top-trader long/short account ratio (futures sentiment)."""
        if self.market != "futures":
            return None
        try:
            rows = self._get(FUTURES_BASE, "/futures/data/topLongShortAccountRatio",
                             {"symbol": symbol.upper(), "period": period, "limit": 1})
            return rows[0] if rows else None
        except BinanceError:
            return None

    def order_book_imbalance(self, symbol: str, limit: int = 100) -> float | None:
        """(bid_qty - ask_qty) / (bid_qty + ask_qty) over top-of-book depth. >0 = bid heavy."""
        try:
            book = self.order_book(symbol, limit)
        except BinanceError:
            return None
        bid = sum(float(q) for _, q in book.get("bids", []))
        ask = sum(float(q) for _, q in book.get("asks", []))
        total = bid + ask
        return (bid - ask) / total if total else None
