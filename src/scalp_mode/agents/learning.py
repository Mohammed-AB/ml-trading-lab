"""Learning Agent — Reviews outcomes, discovers patterns, updates brain.

Runs every 30 minutes + end of day. Correlates trade outcomes with
Strategy Agent reasoning, finds patterns, and writes lessons back
to the shared brain for other agents to read.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from .brain import Brain

_log = logging.getLogger("scalp_mode")


class LearningAgent:
    """Reviews trading outcomes and discovers patterns.

    Writes to brain/lessons/ — these are read by Strategy Agent
    and Portfolio Agent on every cycle.
    """

    def __init__(self, brain: Brain, model: str = "claude-opus-4-20250514"):
        self._brain = brain
        self._model = model
        self._last_run: Optional[datetime] = None

    def should_run(self, utc_now: datetime) -> bool:
        if self._last_run is None:
            return True
        return (utc_now - self._last_run).total_seconds() >= 1800  # 30 min

    def run(self, utc_now: datetime) -> None:
        """Review recent trades and update lessons."""
        self._last_run = utc_now

        proposals = self._brain.read_recent_proposals(30)
        verdicts = self._brain.read_recent_verdicts(30)
        outcomes = self._brain.read_recent_outcomes(30)

        if not outcomes and not proposals:
            _log.info("Learning Agent: no data to review")
            return

        # Build review context
        prompt = (
            f"You are the Learning Agent reviewing recent trading activity at "
            f"{utc_now.strftime('%Y-%m-%d %H:%M UTC')}.\n\n"
            f"Recent proposals ({len(proposals)}):\n"
        )
        for p in proposals[-10:]:
            prompt += (
                f"  {p.get('timestamp', '?')[:16]} | "
                f"{p.get('pair', '?')} {p.get('direction', '?')} "
                f"conf={p.get('confidence', 0):.2f} "
                f"model={p.get('model_source', '?')}\n"
                f"    Reasoning: {p.get('reasoning', 'N/A')[:100]}\n"
            )

        if outcomes:
            prompt += f"\nRecent outcomes ({len(outcomes)}):\n"
            for o in outcomes[-10:]:
                prompt += (
                    f"  {o.get('pair', '?')} {o.get('direction', '?')} "
                    f"P/L={o.get('pnl_pips', 0):+.1f} pips "
                    f"({o.get('exit_reason', '?')})\n"
                )

        prompt += (
            "\nAnalyze the results and output 1-3 specific LESSONS in this format:\n"
            "LESSON: <specific actionable pattern>\n\n"
            "CRITICAL BALANCE REQUIREMENT:\n"
            "- At least ONE lesson MUST be about what IS working or what TO DO "
            "(e.g. 'USD_CHF shorts +55.8p in 3 trades — size up when Model A fires')\n"
            "- Do NOT write only cautionary/avoidance lessons. The system needs "
            "actionable POSITIVE edges, not just warnings.\n"
            "- If all recent outcomes are losses, still identify the BEST setup "
            "type and what conditions would make it tradeable.\n"
            "- Never write lessons about 'degenerate geometry' or SL size — the "
            "AI decides SL sizing autonomously.\n\n"
            "Be SPECIFIC with numbers and conditions, not generic advice."
        )

        try:
            import anthropic
            import httpx
            client = anthropic.Anthropic(
                timeout=httpx.Timeout(60.0, connect=10.0))
            response = client.messages.create(
                model=self._model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            _log.info(f"Learning Agent:\n{text}")

            # Parse lessons
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped.upper().startswith("LESSON:"):
                    pattern = stripped.split(":", 1)[1].strip()
                    if pattern:
                        # Default confidence 0.60 so lessons outrank Chief
                        # memos (0.30) in Strategy's top-40 slice. Backtest
                        # priors seeded at >=0.85 still rank higher.
                        self._brain.write_lesson({
                            "pattern": pattern,
                            "confidence": 0.60,
                            "source": "learning_agent",
                        })
                        _log.info(f"Learning Agent: new lesson: {pattern[:80]}")

        except Exception as e:
            _log.error(f"Learning Agent failed: {e}")

    def _rollup_model_pair_direction_7d(self) -> str:
        """Expectancy table by model × pair × direction (last 7 days)."""
        outs = self._brain.read_recent_outcomes(500)
        if not outs:
            return ""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)
        agg: dict[tuple[str, str, str], dict] = defaultdict(
            lambda: {"n": 0, "pips": 0.0})
        for o in outs:
            ts_s = o.get("timestamp") or ""
            try:
                ts = datetime.fromisoformat(
                    ts_s.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                continue
            mid = str(o.get("model_id") or "?")
            pair = str(o.get("pair") or "?")
            direction = str(o.get("direction") or "?")
            key = (mid, pair, direction)
            agg[key]["n"] += 1
            try:
                agg[key]["pips"] += float(o.get("pnl_pips", 0) or 0)
            except (TypeError, ValueError):
                pass
        if not agg:
            return ""
        lines = [
            "## Weekly rollup — model × pair × direction (last 7 days)",
            "",
            "| model | pair | direction | trades | net pips |",
            "|-------|------|-----------|--------|----------|",
        ]
        for (mid, pair, direction), v in sorted(agg.items()):
            lines.append(
                f"| {mid} | {pair} | {direction} | {v['n']} | "
                f"{v['pips']:+.1f} |")
        return "\n".join(lines) + "\n\n"

    def write_daily_review(self, date: str) -> None:
        """Generate and save end-of-day review."""
        outcomes = self._brain.read_recent_outcomes(100)
        day_outcomes = [o for o in outcomes
                        if o.get("timestamp", "").startswith(date)]

        if not day_outcomes:
            return

        wins = [o for o in day_outcomes if o.get("pnl_pips", 0) > 0]
        losses = [o for o in day_outcomes if o.get("pnl_pips", 0) <= 0]
        total_pnl = sum(o.get("pnl_pips", 0) for o in day_outcomes)
        rollup_md = self._rollup_model_pair_direction_7d()

        prompt = (
            f"Write a daily trading review for {date}.\n\n"
            f"Results: {len(day_outcomes)} trades, {len(wins)} wins, "
            f"{len(losses)} losses, P/L: {total_pnl:+.1f} pips\n\n"
            "Trades:\n"
        )
        for o in day_outcomes[:20]:
            prompt += (
                f"  {o.get('pair')} {o.get('direction')} "
                f"{o.get('pnl_pips', 0):+.1f} pips "
                f"({o.get('exit_reason', '?')})\n"
            )

        prompt += (
            "\nWrite a markdown review with:\n"
            "## Summary\n## What Worked\n## What Failed\n"
            "## Lessons for Tomorrow\n"
        )
        if rollup_md:
            prompt += f"\nUse this evidence table in your analysis:\n{rollup_md}\n"

        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self._model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            review = response.content[0].text.strip()
            body = f"# Daily Review — {date}\n\n"
            if rollup_md:
                body += rollup_md + "\n"
            body += review
            self._brain.write_daily_review(date, body)
        except Exception as e:
            _log.error(f"Learning Agent daily review failed: {e}")
            fallback = (
                f"# Daily Review — {date}\n\n{rollup_md}"
                f"Auto-summary: {len(day_outcomes)} trades, "
                f"{len(wins)} wins, {total_pnl:+.1f} pips total.\n")
            self._brain.write_daily_review(date, fallback)
