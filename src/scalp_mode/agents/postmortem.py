"""Post-mortem Analyst — Analyzes each closed trade.

Triggered within 60 seconds of a trade closing (not on a 30-min timer
like Learning Agent). Produces tight, specific feedback that lands
in the brain before the next Strategy cycle runs.

Focus areas:
- Was the entry timing good?
- Was SL/TP sized correctly given realized volatility?
- What would have been a better exit?
- Tag the outcome's root cause (spread bite, news surprise, trend fade, etc.)
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from .brain import Brain

_log = logging.getLogger("scalp_mode")


class PostMortemAnalyst:
    """Fast per-trade review, fires seconds after a trade closes."""

    def __init__(self, brain: Brain, model: str = "claude-opus-4-20250514"):
        self._brain = brain
        self._model = model

    def analyze(self, outcome: dict, context: dict) -> Optional[str]:
        """Run one post-mortem on a just-closed trade.

        Args:
            outcome: the outcome dict being logged to brain
            context: {pair_stats_for_pair, recent_lessons, market_snapshot}
        Returns: short post-mortem string, or None on error.
        """
        pair = outcome.get("pair", "?")
        direction = outcome.get("direction", "?")
        pnl = float(outcome.get("pnl_pips", 0) or 0)
        exit_reason = outcome.get("exit_reason", "?")
        hold_time = outcome.get("hold_time_seconds")

        was_win = pnl > 0
        verdict = "WIN" if was_win else "LOSS"

        prompt = (
            f"Post-mortem for {pair} {direction} trade.\n\n"
            f"Outcome: {verdict} {pnl:+.1f} pips "
            f"(exit: {exit_reason})\n"
            f"Hold time: {hold_time}s\n"
            f"Entry price: {outcome.get('entry_price', '?')}\n"
            f"Exit price: {outcome.get('exit_price', '?')}\n"
            f"SL: {outcome.get('sl_price', '?')}, TP: {outcome.get('tp_price', '?')}\n\n"
        )

        if context.get("pair_stats"):
            prompt += f"Live stats for this pair+dir:\n{context['pair_stats']}\n\n"

        if context.get("recent_lessons"):
            prompt += "Relevant lessons at time of entry:\n"
            for l in context["recent_lessons"][:5]:
                prompt += f"- {l}\n"
            prompt += "\n"

        prompt += (
            "Write a 3-4 line post-mortem covering:\n"
            "1. Root cause: spread bite / news / trend fade / "
            "regime mismatch / lucky / unlucky / other\n"
            "2. Was the SL/TP sized right for the volatility?\n"
            "3. Was there a warning sign before entry that was ignored?\n"
            "4. One actionable lesson (if any) the next trade should apply.\n"
            "Be direct and short. No hedging."
        )

        try:
            import anthropic
            import httpx
            client = anthropic.Anthropic(
                timeout=httpx.Timeout(30.0, connect=10.0))
            response = client.messages.create(
                model=self._model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            _log.info(
                f"Post-mortem [{pair} {direction} {pnl:+.1f}p]:\n{text}")
            # Save as a short lesson
            self._brain.write_lesson({
                "id": (f"postmortem_{pair}_{direction}_"
                       f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"),
                "pattern": (
                    f"[{pair} {direction} {verdict} {pnl:+.1f}p] "
                    + text[:400]),
                "action": "record",
                "scope": {"pair": pair, "direction": direction,
                          "outcome": verdict.lower()},
                "confidence": 0.65,
                "source": "postmortem",
            })
            return text
        except Exception as e:
            _log.error(f"Post-mortem failed: {e}")
            return None
