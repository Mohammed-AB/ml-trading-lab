"""Research Agent — Market analysis and event research.

Runs every 15 minutes. Scrapes free RSS / Google News headlines, reads the
economic calendar, human notes, and historical context. Writes market
intelligence (including pattern / risk analysis) to the shared brain.
Does not trade.
"""

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Optional

from .brain import Brain
from .web_scraper import gather_market_headlines, headlines_to_prompt_block
from ..gates.news_gate import NewsEvent

_log = logging.getLogger("scalp_mode")

_RESEARCH_SYSTEM = """You are the head of macro and FX research for a professional trading desk.

You receive live headlines from multiple web sources (Google News search, ForexLive, \
FXStreet, and wire/business RSS), plus scheduled economic events and current quotes.

Your job is to produce actionable intelligence for automated trading agents — not generic commentary.

Rules:
- Be precise about which currencies and pairs are most affected.
- When multiple independent headlines point the same way, call that out as a **confirmed** theme.
- Flag sudden-risk scenarios (geopolitical, policy surprises, liquidity events).
- If headlines conflict, say so and explain what to watch to resolve the conflict.
- Compare headline themes to the trader's notes when provided: note agreement or tension.
- Do not invent facts; only infer market implications from the text given.
"""


class ResearchAgent:
    """Analyzes market conditions and upcoming events.

    Usage:
        agent = ResearchAgent(brain, model="claude-opus-4-20250514")
        agent.run(events, current_prices)
    """

    def __init__(self, brain: Brain, model: str = "claude-opus-4-20250514"):
        self._brain = brain
        self._model = model
        self._last_run: Optional[datetime] = None

    def should_run(self, utc_now: datetime,
                   min_interval_sec: int = 900) -> bool:
        if self._last_run is None:
            return True
        return (utc_now - self._last_run).total_seconds() >= min_interval_sec

    def run(self, events: list[NewsEvent], current_prices: dict,
            utc_now: datetime) -> dict:
        """Run market analysis and write to brain."""
        self._last_run = utc_now

        # --- Web headlines (RSS + Google News) ---
        try:
            headline_items = gather_market_headlines()
        except Exception as e:
            _log.warning(f"Research Agent: headline scrape failed: {e}")
            headline_items = []

        by_source = Counter(
            (it.get("source") or "unknown") for it in headline_items
        )
        headlines_block = headlines_to_prompt_block(headline_items, max_lines=85)

        # Gather context
        upcoming = [e for e in events
                    if e.timestamp_utc > utc_now
                    and e.timestamp_utc < utc_now + timedelta(hours=12)
                    and e.impact == "high"]

        session = self._identify_session(utc_now)
        human_notes = self._brain.read_human_notes()

        event_text = "\n".join(
            f"  {e.timestamp_utc.strftime('%H:%M UTC')} - {e.currency} - {e.title}"
            for e in upcoming
        ) if upcoming else "No high-impact events in next 12 hours."

        price_text = "\n".join(
            f"  {pair}: {price:.5f}"
            for pair, price in current_prices.items()
        )

        user_parts = [
            f"Current time: {utc_now.strftime('%Y-%m-%d %H:%M UTC')} "
            f"({utc_now.strftime('%A')})",
            f"Trading session (approx): {session}",
            "",
            "=== CURRENT PRICES ===",
            price_text,
            "",
            "=== UPCOMING HIGH-IMPACT EVENTS (12h) ===",
            event_text,
            "",
            "=== LATEST MARKET HEADLINES (web research) ===",
            "Sources are labeled [google_news], [forexlive], [fxstreet], "
            "[reuters], [bbc_business], [investing_forex], etc.",
            headlines_block,
            "",
        ]

        if human_notes != "No human notes.":
            user_parts.extend([
                "=== TRADER / HUMAN NOTES (Obsidian) ===",
                human_notes,
                "",
            ])

        user_parts.append(
            "=== REQUIRED OUTPUT ===\n"
            "Produce the following sections with clear headers:\n\n"
            "1. MARKET_MOVING_STORIES — Pick the 3–5 headlines that matter most "
            "for FX right now; one line each with why it matters.\n\n"
            "2. FOREX_IMPACT — For each major theme, state: likely direction for "
            "USD, EUR, GBP, JPY, commodity currencies (AUD, NZD, CAD), and CHF "
            "where relevant. Rate conviction: low / medium / high.\n\n"
            "3. CROSS_HEADLINE_PATTERNS — Do multiple independent sources confirm "
            "the same story? Call out confirmation vs. single-source noise.\n\n"
            "4. GEOPOLITICAL_AND_TAIL_RISKS — Items that could cause gaps, "
            "volatility spikes, or safe-haven flows.\n\n"
            "5. SESSION_BIAS — For THIS session only: bullish_usd / bearish_usd / "
            "mixed, with one short paragraph.\n\n"
            "6. PER_PAIR_BRIEF — Produce one concise block for each of "
            "EUR_USD, USD_JPY, GBP_USD, AUD_USD, EUR_GBP, USD_CHF, USD_CAD, "
            "NZD_USD in EXACTLY this format:\n"
            "   <PAIR>: bias=<long|short|flat>, conviction=<low|med|high>, "
            "key_driver=<one sentence>, avoid_if=<one condition or 'none'>\n"
            "   Each on its own line. Do NOT skip any pair.\n\n"
            "7. PAIRS_TO_WATCH — Best risk/reward pairs right now.\n\n"
            "8. PAIRS_TO_AVOID — Pairs likely to chop or gap.\n\n"
            "9. EVENT_RISK — Tie upcoming calendar events to the narrative.\n\n"
            "10. VS_HUMAN_NOTES — If human notes were provided: what confirms, "
            "contradicts, or is unclear.\n\n"
            "11. KEY_LEVELS — Only if inferable; otherwise 'not inferable'.\n\n"
            "Keep total under ~1500 words. Be direct."
        )

        user_message = "\n".join(user_parts)

        try:
            import anthropic
            import httpx
            client = anthropic.Anthropic(
                timeout=httpx.Timeout(60.0, connect=15.0))
            response = client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=_RESEARCH_SYSTEM,
                messages=[{"role": "user", "content": user_message}],
            )
            analysis = response.content[0].text.strip()
        except Exception as e:
            _log.warning(f"Research Agent failed: {e}")
            analysis = f"Research unavailable: {e}"

        per_pair_briefs = self._extract_per_pair_briefs(analysis)

        state = {
            "session": session,
            "analysis": analysis,
            "per_pair_briefs": per_pair_briefs,
            "web_headlines_count": len(headline_items),
            "web_headlines_by_source": dict(by_source),
            "web_headlines_sample": [
                {"title": it.get("title"), "source": it.get("source")}
                for it in headline_items[:15]
            ],
            "upcoming_events": [
                {"time": e.timestamp_utc.isoformat(),
                 "currency": e.currency, "title": e.title}
                for e in upcoming
            ],
            "prices_snapshot": current_prices,
        }

        self._brain.write_market_state(state)
        _log.info(
            "Research Agent: market brief updated (%s), %d web headlines",
            session,
            len(headline_items),
        )
        return state

    def _extract_per_pair_briefs(self, analysis: str) -> dict:
        """Parse PER_PAIR_BRIEF lines like:
        EUR_USD: bias=short, conviction=high, key_driver=..., avoid_if=...
        Returns {pair: {bias, conviction, key_driver, avoid_if}}.
        """
        out = {}
        if not analysis:
            return out
        pairs = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD",
                 "EUR_GBP", "USD_CHF", "USD_CAD", "NZD_USD"]
        for line in analysis.split("\n"):
            line = line.strip().lstrip("-").strip()
            for pair in pairs:
                if line.startswith(pair) or line.startswith(f"{pair}:"):
                    info = {}
                    rest = line.split(":", 1)[1] if ":" in line else line
                    # Parse key=value fragments separated by commas
                    for frag in rest.split(","):
                        if "=" in frag:
                            k, v = frag.split("=", 1)
                            info[k.strip()] = v.strip()
                    if info:
                        out[pair] = info
                    break
        return out

    def _identify_session(self, utc_now: datetime) -> str:
        h = utc_now.hour
        if 22 <= h or h < 7:
            return "Asian"
        elif 7 <= h < 12:
            return "London"
        elif 12 <= h < 16:
            return "London/NY Overlap"
        elif 16 <= h < 21:
            return "New York"
        else:
            return "Late NY / Pre-Asian"
