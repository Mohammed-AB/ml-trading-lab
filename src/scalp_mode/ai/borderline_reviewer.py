"""AI Borderline Reviewer — Approve or reject borderline signals.

Per spec 5.1: max 1 call/minute, timeout 500ms, fail-safe=Reject.

Called only when is_borderline=True. Conservative: on any failure → reject.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger("scalp_mode")


@dataclass
class BorderlineDecision:
    approved: bool
    reason: str
    confidence: float = 0.0


class AIBorderlineReviewer:
    """Reviews borderline signals — approve or reject.

    Usage:
        reviewer = AIBorderlineReviewer(config)
        decision = reviewer.evaluate(signal_summary)
    """

    def __init__(self, config: dict):
        self._enabled = config.get("enabled", False)
        self._timeout_sec = config.get("timeout_ms", 500) / 1000.0
        self._max_calls_per_min = config.get("max_calls_per_minute", 1)
        self._model = config.get("model", "claude-opus-4-20250514")
        self._last_call_time: float = 0
        self._calls_this_minute: int = 0
        self._minute_start: float = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def evaluate(self, signal_summary: dict) -> BorderlineDecision:
        """Review a borderline signal.

        Args:
            signal_summary: Context about the signal including pair, direction,
                model, regime, trigger values, borderline flags, spread.

        Returns:
            BorderlineDecision — approved=False on any failure (conservative).
        """
        if not self._enabled:
            return BorderlineDecision(approved=True, reason="reviewer_disabled",
                                       confidence=1.0)

        # Rate limit check
        now = time.monotonic()
        if now - self._minute_start >= 60:
            self._minute_start = now
            self._calls_this_minute = 0

        if self._calls_this_minute >= self._max_calls_per_min:
            _log.info("Borderline reviewer rate limited — rejecting")
            return BorderlineDecision(approved=False, reason="rate_limited")

        self._calls_this_minute += 1
        self._last_call_time = now

        try:
            return self._call_ai(signal_summary)
        except Exception as e:
            _log.warning(f"Borderline reviewer failed: {e} — rejecting")
            return BorderlineDecision(approved=False, reason=f"error:{type(e).__name__}")

    def _call_ai(self, summary: dict) -> BorderlineDecision:
        """Call AI model for borderline review."""
        import anthropic

        flags = summary.get("borderline_flags", [])
        prompt = (
            f"Borderline forex scalp signal. Reply ONLY with one line:\n"
            f"APPROVE <confidence 0-1> <reason>\n"
            f"or REJECT <confidence 0-1> <reason>\n"
            f"Example: APPROVE 0.7 decent momentum despite wide spread\n\n"
            f"Pair={summary.get('pair')} Dir={summary.get('direction')} "
            f"Regime={summary.get('regime')} Spread={summary.get('spread_pips')}pips\n"
            f"Flags={flags} (B1=edge regime B2=weak breakout B3=ambig momentum "
            f"B4=spread near max B5=weak MACD)\n"
            f"Trigger={summary.get('trigger_values')}"
        )

        import httpx
        client = anthropic.Anthropic(
            timeout=httpx.Timeout(self._timeout_sec, connect=2.0))
        response = client.messages.create(
            model=self._model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip().upper()
        parts = text.split(None, 2)

        if parts[0] == "APPROVE":
            confidence = float(parts[1]) if len(parts) > 1 else 0.5
            reason = parts[2] if len(parts) > 2 else "ai_approved"
            return BorderlineDecision(approved=True, reason=reason.lower(),
                                       confidence=confidence)
        else:
            confidence = float(parts[1]) if len(parts) > 1 else 0.5
            reason = parts[2] if len(parts) > 2 else "ai_rejected"
            return BorderlineDecision(approved=False, reason=reason.lower(),
                                       confidence=confidence)
