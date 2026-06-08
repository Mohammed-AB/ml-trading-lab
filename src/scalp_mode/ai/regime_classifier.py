"""AI Regime Classifier — Enhances rule-based regime detection.

Per spec 5.1: every 5 minutes, timeout 800ms, fail-safe=rule-based.

Not a replacement — an enhancement layer. Rule-based always runs as baseline.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from ..engine.regime_engine import RegimeEngine, RegimeResult, Regime
from ..engine.feature_engine import IndicatorSet

_log = logging.getLogger("scalp_mode")


class AIRegimeClassifier:
    """AI-powered regime classification — enhances rule-based.

    Usage:
        classifier = AIRegimeClassifier(config, rule_based_engine)
        result = classifier.evaluate(ind_m5, close_m5, utc_now)
    """

    def __init__(self, config: dict, rule_based: RegimeEngine):
        self._enabled = config.get("enabled", False)
        self._timeout_sec = config.get("timeout_ms", 800) / 1000.0
        self._frequency_sec = config.get("frequency_minutes", 5) * 60
        self._model = config.get("model", "claude-opus-4-20250514")
        self._rule_based = rule_based
        self._last_call: float = 0
        self._cached_result: Optional[RegimeResult] = None
        self._recent_regimes: list[str] = []  # Last 3 for context

    @property
    def enabled(self) -> bool:
        return self._enabled

    def evaluate(self, ind_m5: IndicatorSet, close_m5: float,
                 utc_now: datetime) -> RegimeResult:
        """Get regime classification.

        Always computes rule-based as baseline.
        Calls AI every N minutes for enhanced classification.
        On AI failure/timeout: returns rule-based result.
        """
        rule_result = self._rule_based.evaluate(ind_m5, close_m5)

        # Track recent regimes for context
        self._recent_regimes.append(rule_result.regime.value)
        if len(self._recent_regimes) > 3:
            self._recent_regimes = self._recent_regimes[-3:]

        if not self._enabled:
            return rule_result

        # Check frequency
        now = time.monotonic()
        if now - self._last_call < self._frequency_sec and self._cached_result:
            return self._cached_result

        # Call AI
        try:
            ai_result = self._call_ai(ind_m5, close_m5, rule_result)
            self._last_call = now
            self._cached_result = ai_result
            return ai_result
        except Exception as e:
            _log.warning(f"AI regime classifier failed: {e} — using rule-based")
            self._last_call = now
            self._cached_result = rule_result
            return rule_result

    def _call_ai(self, ind: IndicatorSet, close: float,
                 rule_result: RegimeResult) -> RegimeResult:
        """Call AI model for regime classification."""
        import anthropic

        prompt = (
            f"Classify this forex M5 data into a regime. "
            f"Reply ONLY with: REGIME <name> <confidence>\n"
            f"Valid names: Trend_Up, Trend_Down, Range, NoTrade\n"
            f"Example: REGIME Trend_Up 0.85\n\n"
            f"EMA_slope={ind.ema_slope:.4f} BB_width={ind.bb_width:.5f} "
            f"RSI={ind.rsi14:.1f} Close={close:.5f} EMA20={ind.ema20:.5f} "
            f"EMA50={ind.ema50:.5f} ATR={ind.atr14:.6f}\n"
            f"Rule-based={rule_result.regime.value} Recent={self._recent_regimes}"
        )

        import httpx
        client = anthropic.Anthropic(
            timeout=httpx.Timeout(self._timeout_sec, connect=2.0))
        response = client.messages.create(
            model=self._model,
            max_tokens=30,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        parts = text.split()

        if len(parts) >= 2 and parts[0].upper() == "REGIME":
            regime_name = parts[1]
            try:
                ai_regime = Regime(regime_name)
            except ValueError:
                _log.warning(f"AI returned invalid regime: {regime_name}")
                return rule_result

            confidence = float(parts[2]) if len(parts) > 2 else 0.5

            # Build result with AI regime but rule-based values
            return RegimeResult(
                regime=ai_regime,
                values={**rule_result.values, "ai_confidence": confidence,
                        "rule_based_regime": rule_result.regime.value},
                is_borderline=rule_result.is_borderline,
                borderline_flags=rule_result.borderline_flags,
            )

        _log.warning(f"AI regime response unparseable: {text}")
        return rule_result
