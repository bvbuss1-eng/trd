"""Technical indicators, implemented in pure pandas/numpy (no TA-Lib dependency).

Every indicator at row i uses only data from rows <= i, so an enriched frame is
safe for walk-forward backtesting with no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    mid = sma(close, period)
    std = close.rolling(period).std()
    upper, lower = mid + mult * std, mid - mult * std
    width = (upper - lower) / mid
    return upper, mid, lower, width


def vwap_daily(df: pd.DataFrame) -> pd.Series:
    """Session VWAP anchored to each UTC day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date
    pv = (typical * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def swing_low(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    return df["low"].rolling(lookback).min()


def swing_high(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    return df["high"].rolling(lookback).max()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the full indicator set used by the strategy and charts."""
    out = df.copy()
    close = out["close"]
    out["ema9"] = ema(close, 9)
    out["ema21"] = ema(close, 21)
    out["ema50"] = ema(close, 50)
    out["ema200"] = ema(close, 200)
    out["rsi"] = rsi(close)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["atr"] = atr(out)
    out["bb_upper"], out["bb_mid"], out["bb_lower"], out["bb_width"] = bollinger(close)
    out["vwap"] = vwap_daily(out)
    out["adx"] = adx(out)
    out["vol_ma20"] = sma(out["volume"], 20)
    out["vol_ratio"] = out["volume"] / out["vol_ma20"].replace(0, np.nan)
    out["swing_low"] = swing_low(out)
    out["swing_high"] = swing_high(out)
    body = (close - out["open"]).abs()
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["body_pct"] = (body / rng).fillna(0.0)
    return out
