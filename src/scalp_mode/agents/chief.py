"""Chief Agent — Meta-coordinator.

Runs every 15 minutes during trading sessions. Reviews what the other
agents have been doing and produces a short "boardroom memo" that
becomes context for the next Strategy / Portfolio cycles.

Purpose: catch behavioral loops (e.g. Strategy proposing the same
failing setup 5 times, Portfolio holding a thesis that's been
invalidated by Learning, etc.) that no single agent can notice from
its own narrow view.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from .brain import Brain

_log = logging.getLogger("scalp_mode")


class ChiefAgent:
    """Reviews multi-agent behavior for loops, conflicts, and drift."""

    def __init__(self, brain: Brain, model: str = "claude-opus-4-20250514"):
        self._brain = brain
        self._model = model
        self._last_run: Optional[datetime] = None

    def should_run(self, utc_now: datetime,
                   min_interval_sec: int = 900) -> bool:
        if self._last_run is None:
            return True
        return (utc_now - self._last_run).total_seconds() >= min_interval_sec

    def run(self, utc_now: datetime) -> Optional[str]:
        """Generate a meta-review memo and save to brain."""
        self._last_run = utc_now

        proposals = self._brain.read_recent_proposals(30)
        verdicts = self._brain.read_recent_verdicts(30)
        outcomes = self._brain.read_recent_outcomes(30)
        lessons = self._brain.read_lessons(15)
        market_state = self._brain.read_market_state()

        if not proposals and not outcomes:
            return None

        # Build a compact behavioral history
        prop_lines = []
        for p in proposals[-20:]:
            action = p.get("action", "TRADE")
            if action == "SKIP":
                prop_lines.append(f"  SKIP: {p.get('reasoning', '?')[:80]}")
            else:
                prop_lines.append(
                    f"  {p.get('timestamp', '?')[11:16]} "
                    f"{p.get('pair', '?')} {p.get('direction', '?')} "
                    f"conf={p.get('confidence', 0):.2f} "
                    f"model={p.get('model_source', '?')}")

        outcome_lines = []
        for o in outcomes[-15:]:
            outcome_lines.append(
                f"  {o.get('pair', '?')} {o.get('direction', '?')} "
                f"{o.get('pnl_pips', 0):+.1f}p ({o.get('exit_reason', '?')})")

        verdict_lines = []
        rejections = {}
        for v in verdicts[-20:]:
            reason = v.get("reject_reason") or "approved"
            rejections[reason] = rejections.get(reason, 0) + 1

        rej_summary = ", ".join(
            f"{k}={v}" for k, v in sorted(rejections.items()))

        pair_stats = self._brain.format_pair_stats_summary(min_trades=3)

        prompt = (
            f"Chief coordinator review at "
            f"{utc_now.strftime('%Y-%m-%d %H:%M UTC')}.\n\n"
            f"RECENT STRATEGY PROPOSALS (last 20):\n"
            + ("\n".join(prop_lines) if prop_lines else "  (none)") + "\n\n"
            f"RECENT OUTCOMES (last 15):\n"
            + ("\n".join(outcome_lines) if outcome_lines else "  (none)") + "\n\n"
            f"RISK VERDICT DISTRIBUTION (last 20): {rej_summary}\n\n"
            f"LIVE PAIR STATS:\n{pair_stats}\n\n"
            f"YOUR JOB:\n"
            f"Write a SHORT boardroom memo (max 12 lines) identifying:\n"
            f"1. Any behavioral loops (same losing setup proposed repeatedly)\n"
            f"2. Any agent conflicts (e.g., Portfolio ignoring Learning's warnings)\n"
            f"3. Risk creep or overtrading signs\n"
            f"4. Which pair/direction combos are clearly the current edge\n"
            f"5. Which to avoid this session\n"
            f"6. One specific directive for Strategy and one for Portfolio.\n\n"
            f"Be direct. Numbers only — no hedging language. If everything looks "
            f"healthy, say so in one line.\n"
        )

        try:
            import anthropic
            import httpx
            client = anthropic.Anthropic(
                timeout=httpx.Timeout(45.0, connect=10.0))
            response = client.messages.create(
                model=self._model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            memo = response.content[0].text.strip()
            _log.info(f"Chief Agent memo:\n{memo}")
            # Write to the dedicated memo store (data/brain/memos/chief_memos.jsonl),
            # NOT lessons/patterns.jsonl. Chief memos are coordination notes,
            # not trade heuristics — keeping them out of the lessons file
            # prevents them from ever appearing in Strategy's top-N prompt
            # window (which happened 2026-04-20 and killed trading).
            self._brain.write_chief_memo({
                "id": f"chief_memo_{utc_now.strftime('%Y%m%d_%H%M')}",
                "pattern": memo[:1500],
                "action": "review",
                "source": "chief_agent",
            })
            return memo
        except Exception as e:
            _log.error(f"Chief Agent failed: {e}")
            return None

    def read_latest_memo(self) -> str:
        """Return the most recent chief memo (for agents to include in context)."""
        return self._brain.read_latest_chief_memo()
