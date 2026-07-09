"""Claude AI analyst layer (optional).

Sends the computed market snapshot + the rendered chart PNG (vision) to Claude
and gets back a structured second-opinion assessment of the setup. Requires the
`anthropic` package and an ANTHROPIC_API_KEY (or `ant auth login` profile).

This layer never executes anything — it argues for or against YOUR trade.
"""

from __future__ import annotations

import base64
import json

DEFAULT_MODEL = "claude-opus-4-8"

ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["ENTER", "SKIP", "WAIT"],
            "description": "ENTER = setup is valid now; WAIT = valid idea, bad timing; SKIP = do not take it",
        },
        "confidence": {"type": "integer", "description": "0-100 conviction in the verdict"},
        "agreement_with_signal": {"type": "boolean"},
        "reasoning": {"type": "string", "description": "The full argument, referencing chart structure and data"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "suggested_adjustments": {
            "type": "object",
            "properties": {
                "entry": {"type": ["number", "null"]},
                "stop": {"type": ["number", "null"]},
                "target": {"type": ["number", "null"]},
                "note": {"type": "string"},
            },
            "required": ["entry", "stop", "target", "note"],
            "additionalProperties": False,
        },
        "invalidation": {"type": "string", "description": "What would prove this trade idea wrong"},
        "key_levels": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "confidence", "agreement_with_signal", "reasoning",
                 "strengths", "weaknesses", "suggested_adjustments",
                 "invalidation", "key_levels"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a senior intraday crypto trading analyst reviewing a \
proposed day trade on Binance perpetual futures. Your job is to be the skeptical \
second pair of eyes: stress-test the setup, not cheerlead it.

The trader's style: 5m/15m trend-pullback entries, 1:2 to 1:3 risk-reward, small \
risk per trade, manual execution. They want to know WHY a trade is good or bad.

You are given (1) a candlestick chart image with EMAs, VWAP, volume, RSI and the \
proposed entry/stop/target levels drawn on it, and (2) the raw computed data: \
indicator values, the rule-based signal's factor scores, and market context \
(funding rate, open interest, order-book imbalance).

Assess: trend quality, location of entry relative to structure, whether the stop \
sits behind real structure or in an obvious liquidity pocket, whether the target \
is realistic before major resistance/support, volume behavior, and anything on \
the chart the numeric rules would miss. If the setup is bad, say SKIP and say why \
plainly. Prefer missing a trade over taking a bad one — the trader only needs 1-3 \
good trades per day."""


def assess_signal(snapshot: dict, chart_png_path: str | None = None,
                  model: str = DEFAULT_MODEL) -> dict:
    """Ask Claude for a structured assessment. Returns the validated dict."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("pip install anthropic to use the AI analyst layer") from exc

    client = anthropic.Anthropic()

    content: list[dict] = []
    if chart_png_path:
        with open(chart_png_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
        })
    content.append({
        "type": "text",
        "text": ("Here is the computed market snapshot and the rule-based signal "
                 "(JSON). Review it together with the chart image and give your "
                 "structured assessment.\n\n" + json.dumps(snapshot, indent=2, default=str)),
    })

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": ASSESSMENT_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request (stop_reason=refusal)")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def format_assessment(a: dict) -> str:
    lines = [
        "=" * 62,
        f"AI ANALYST VERDICT: {a['verdict']}  (confidence {a['confidence']}/100, "
        f"{'agrees' if a['agreement_with_signal'] else 'DISAGREES'} with the rule-based signal)",
        "=" * 62,
        "",
        "  REASONING:",
        "  " + a["reasoning"].replace("\n", "\n  "),
        "",
        "  STRENGTHS:",
        *[f"    + {s}" for s in a["strengths"]],
        "",
        "  WEAKNESSES / RISKS:",
        *[f"    - {w}" for w in a["weaknesses"]],
        "",
        "  KEY LEVELS:",
        *[f"    • {k}" for k in a["key_levels"]],
        "",
        f"  INVALIDATION: {a['invalidation']}",
    ]
    adj = a.get("suggested_adjustments") or {}
    tweaks = [f"{k}: {v}" for k, v in adj.items() if k != "note" and v is not None]
    if tweaks or adj.get("note"):
        lines += ["", "  SUGGESTED ADJUSTMENTS: " + ", ".join(tweaks)]
        if adj.get("note"):
            lines.append(f"    note: {adj['note']}")
    return "\n".join(lines)
