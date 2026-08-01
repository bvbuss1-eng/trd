"""Signal object and the human-readable 'why' report."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Factor:
    name: str
    passed: bool
    weight: int
    detail: str


@dataclass
class Signal:
    symbol: str
    interval: str
    direction: str            # "LONG" | "SHORT"
    time: str                 # candle close time (UTC ISO)
    entry: float
    stop: float
    target: float
    rr: float
    score: int
    max_score: int
    factors: list[Factor] = field(default_factory=list)
    context: dict = field(default_factory=dict)   # funding, OI, book imbalance...
    limit: float | None = None    # resting limit entry; None = market/next-open

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)

    def to_report(self, sizing: dict | None = None) -> str:
        arrow = "▲" if self.direction == "LONG" else "▼"
        lines = [
            f"{'=' * 62}",
            f"{arrow} {self.direction} SIGNAL — {self.symbol} ({self.interval})  @ {self.time}",
            f"{'=' * 62}",
            f"  Entry      : {self.entry:.6g}",
            f"  Stop-loss  : {self.stop:.6g}   ({self._pct(self.stop):+.2f}%)",
            f"  Take-profit: {self.target:.6g}   ({self._pct(self.target):+.2f}%)",
            f"  Risk:Reward: 1:{self.rr:.2f}",
            f"  Confluence : {self.score}/{self.max_score} points",
            "",
            "  WHY (each factor, pass/fail):",
        ]
        for f in self.factors:
            mark = "✔" if f.passed else "✘"
            lines.append(f"    [{mark}] ({f.weight}pt) {f.name}: {f.detail}")
        if self.context:
            lines.append("")
            lines.append("  MARKET CONTEXT:")
            for k, v in self.context.items():
                lines.append(f"    - {k}: {v}")
        if sizing:
            lines.append("")
            lines.append("  POSITION SIZING (risk-based):")
            for k, v in sizing.items():
                lines.append(f"    - {k}: {v}")
        lines.append("")
        lines.append("  Invalidation: full stop-loss hit, or the HTF trend flips before entry fills.")
        lines.append("  Execution is MANUAL — this is analysis, not financial advice.")
        return "\n".join(lines)

    def _pct(self, level: float) -> float:
        return (level - self.entry) / self.entry * 100

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "interval": self.interval,
            "direction": self.direction, "time": self.time,
            "entry": self.entry, "stop": self.stop, "target": self.target,
            "rr": self.rr, "score": self.score, "max_score": self.max_score,
            "factors": [{"name": f.name, "passed": f.passed,
                         "weight": f.weight, "detail": f.detail} for f in self.factors],
            "context": self.context,
        }
