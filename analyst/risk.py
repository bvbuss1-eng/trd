"""Risk-based position sizing for manual execution on Binance futures."""

from __future__ import annotations


def position_size(capital: float, risk_pct: float, entry: float, stop: float,
                  max_leverage: float = 10.0, taker_fee_pct: float = 0.05) -> dict:
    """Size a position so a full stop-out loses ~risk_pct of capital.

    Returns everything needed to place the order manually.
    """
    risk_usd = capital * risk_pct / 100
    stop_dist = abs(entry - stop)
    if stop_dist <= 0 or entry <= 0:
        raise ValueError("entry/stop invalid")
    stop_pct = stop_dist / entry * 100

    qty = risk_usd / stop_dist
    notional = qty * entry
    leverage_needed = notional / capital
    capped = leverage_needed > max_leverage
    if capped:
        leverage_needed = max_leverage
        notional = capital * max_leverage
        qty = notional / entry
        risk_usd = qty * stop_dist

    round_trip_fees = notional * taker_fee_pct / 100 * 2

    return {
        "capital": f"${capital:,.2f}",
        "risk per trade": f"${risk_usd:,.2f} ({risk_usd / capital * 100:.2f}% of capital)",
        "stop distance": f"{stop_pct:.2f}%",
        "position size": f"{qty:.6g} units (~${notional:,.2f} notional)",
        "leverage needed": f"{leverage_needed:.1f}x" + ("  (CAPPED — risk reduced)" if capped else ""),
        "est. round-trip taker fees": f"${round_trip_fees:,.2f}",
    }
