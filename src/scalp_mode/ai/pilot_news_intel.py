"""AI Pilot News Intelligence — Deep research on economic events.

Goes beyond title+impact: researches what each event means, likely
outcomes, affected pairs, and recommended positioning.  Also handles
post-event analysis and weekly preparation.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ..gates.news_gate import NewsEvent

_log = logging.getLogger("scalp_mode")


@dataclass
class EventBriefing:
    event_title: str
    event_time: str
    currency: str
    what_it_measures: str = ""
    market_expectation: str = ""
    previous_result: str = ""
    impact_scenarios: str = ""
    affected_pairs: str = ""
    positioning_advice: str = ""
    generated_at: str = ""


@dataclass
class PostEventNote:
    event_title: str
    event_time: str
    what_happened: str = ""
    market_reaction: str = ""
    continuation_or_reversal: str = ""
    updated_bias: str = ""
    generated_at: str = ""


@dataclass
class WeekAheadBriefing:
    week_start: str
    key_events: str = ""
    daily_plan: str = ""
    pairs_focus: str = ""
    generated_at: str = ""


class NewsIntelligence:
    """Deep news research for the AI Pilot.

    Runs a background thread that prepares briefings for upcoming events.
    Provides formatted context for injection into the AI prompt.
    """

    def __init__(self, model: str = "claude-opus-4-20250514",
                 cache_dir: str = "data/news_briefings"):
        self._model = model
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._briefings: dict[str, EventBriefing] = {}
        self._post_notes: dict[str, PostEventNote] = {}
        self._week_ahead: Optional[WeekAheadBriefing] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._events: list[NewsEvent] = []
        self._lock = threading.Lock()

    def set_events(self, events: list[NewsEvent]) -> None:
        with self._lock:
            self._events = list(events)

    def start_background(self) -> None:
        """Start hourly background briefing preparation."""
        if self._running:
            return
        self._running = True
        self._load_cached_briefings()
        self._thread = threading.Thread(
            target=self._background_loop, daemon=True,
            name="news-intel")
        self._thread.start()
        _log.info("News Intelligence background thread started")

    def stop_background(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _background_loop(self) -> None:
        while self._running:
            try:
                self._prepare_upcoming_briefings()
            except Exception as e:
                _log.error(f"News intel background error: {e}")
            sleep_end = time.monotonic() + 3600  # hourly
            while self._running and time.monotonic() < sleep_end:
                time.sleep(10)

    # --- Briefing preparation ---

    def _prepare_upcoming_briefings(self) -> None:
        """Prepare briefings for events in the next 12 hours."""
        now = datetime.now(timezone.utc)
        window = now + timedelta(hours=12)

        with self._lock:
            events = list(self._events)

        for event in events:
            if event.impact != "high":
                continue
            if not (now <= event.timestamp_utc <= window):
                continue
            key = f"{event.title}_{event.timestamp_utc.isoformat()}"
            if key in self._briefings:
                continue  # already prepared

            briefing = self._research_event(event)
            if briefing:
                self._briefings[key] = briefing
                self._save_briefing(key, briefing)
                _log.info(f"News intel: prepared briefing for {event.title}")

    def _research_event(self, event: NewsEvent) -> Optional[EventBriefing]:
        """Ask Claude to research an upcoming economic event."""
        try:
            import anthropic
            client = anthropic.Anthropic()

            prompt = (
                f"You are a professional forex analyst preparing a briefing for "
                f"an upcoming economic event.\n\n"
                f"Event: {event.title}\n"
                f"Currency: {event.currency}\n"
                f"Time: {event.timestamp_utc.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"Impact: {event.impact}\n\n"
                f"Provide a concise briefing in this exact format:\n"
                f"WHAT_IT_MEASURES: (1 sentence explaining what this indicator tracks)\n"
                f"MARKET_EXPECTATION: (what the market typically expects, consensus if known)\n"
                f"PREVIOUS_RESULT: (what happened last time this was released, typical market reaction)\n"
                f"IMPACT_SCENARIOS: (2-3 scenarios: beats/meets/misses expectations and likely price moves)\n"
                f"AFFECTED_PAIRS: (which currency pairs will move most and in which direction)\n"
                f"POSITIONING_ADVICE: (what a scalper should do before, during, and after this event)\n"
            )

            response = client.messages.create(
                model=self._model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            briefing = EventBriefing(
                event_title=event.title,
                event_time=event.timestamp_utc.isoformat(),
                currency=event.currency,
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("WHAT_IT_MEASURES:"):
                    briefing.what_it_measures = line.split(":", 1)[1].strip()
                elif line.startswith("MARKET_EXPECTATION:"):
                    briefing.market_expectation = line.split(":", 1)[1].strip()
                elif line.startswith("PREVIOUS_RESULT:"):
                    briefing.previous_result = line.split(":", 1)[1].strip()
                elif line.startswith("IMPACT_SCENARIOS:"):
                    briefing.impact_scenarios = line.split(":", 1)[1].strip()
                elif line.startswith("AFFECTED_PAIRS:"):
                    briefing.affected_pairs = line.split(":", 1)[1].strip()
                elif line.startswith("POSITIONING_ADVICE:"):
                    briefing.positioning_advice = line.split(":", 1)[1].strip()

            return briefing
        except Exception as e:
            _log.warning(f"Event research failed for {event.title}: {e}")
            return None

    # --- Post-event analysis ---

    def analyze_post_event(self, event: NewsEvent,
                           price_changes: dict[str, float]) -> Optional[PostEventNote]:
        """Analyze what happened after a news event."""
        try:
            import anthropic
            client = anthropic.Anthropic()

            price_info = "\n".join(
                f"  {pair}: {change:+.1f} pips" for pair, change in price_changes.items()
            )

            key = f"{event.title}_{event.timestamp_utc.isoformat()}"
            briefing = self._briefings.get(key)
            pre_context = ""
            if briefing:
                pre_context = (
                    f"Pre-event expectation: {briefing.market_expectation}\n"
                    f"Pre-event scenarios: {briefing.impact_scenarios}\n"
                )

            prompt = (
                f"Analyze the market reaction after this economic event:\n\n"
                f"Event: {event.title} ({event.currency})\n"
                f"Time: {event.timestamp_utc.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"{pre_context}"
                f"Price changes in 15 minutes after event:\n{price_info}\n\n"
                f"Respond in this format:\n"
                f"WHAT_HAPPENED: (1 sentence on the actual result vs expectations)\n"
                f"MARKET_REACTION: (describe the price action)\n"
                f"CONTINUATION_OR_REVERSAL: (is the initial move likely to continue or fade?)\n"
                f"UPDATED_BIAS: (directional bias for affected pairs for the next 2-4 hours)\n"
            )

            response = client.messages.create(
                model=self._model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            note = PostEventNote(
                event_title=event.title,
                event_time=event.timestamp_utc.isoformat(),
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            for line in text.split("\n"):
                line = line.strip()
                if line.startswith("WHAT_HAPPENED:"):
                    note.what_happened = line.split(":", 1)[1].strip()
                elif line.startswith("MARKET_REACTION:"):
                    note.market_reaction = line.split(":", 1)[1].strip()
                elif line.startswith("CONTINUATION_OR_REVERSAL:"):
                    note.continuation_or_reversal = line.split(":", 1)[1].strip()
                elif line.startswith("UPDATED_BIAS:"):
                    note.updated_bias = line.split(":", 1)[1].strip()

            self._post_notes[key] = note
            _log.info(f"News intel: post-event analysis for {event.title}")
            return note
        except Exception as e:
            _log.warning(f"Post-event analysis failed for {event.title}: {e}")
            return None

    # --- Weekly preparation ---

    def prepare_week_ahead(self, events: list[NewsEvent]) -> Optional[WeekAheadBriefing]:
        """Generate a week-ahead trading plan from the event calendar."""
        high_impact = [e for e in events if e.impact == "high"]
        if not high_impact:
            return None

        try:
            import anthropic
            client = anthropic.Anthropic()

            event_list = "\n".join(
                f"  {e.timestamp_utc.strftime('%A %H:%M UTC')} - {e.currency} - {e.title}"
                for e in sorted(high_impact, key=lambda x: x.timestamp_utc)
            )

            prompt = (
                f"You are planning the trading week ahead as a forex scalper.\n\n"
                f"High-impact events this week:\n{event_list}\n\n"
                f"Provide a week-ahead plan:\n"
                f"KEY_EVENTS: (rank the top 3-5 events by expected market impact)\n"
                f"DAILY_PLAN: (for each day Mon-Fri, 1-2 sentences on strategy)\n"
                f"PAIRS_FOCUS: (which pairs to focus on each day based on events)\n"
            )

            response = client.messages.create(
                model=self._model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            briefing = WeekAheadBriefing(
                week_start=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                generated_at=datetime.now(timezone.utc).isoformat(),
            )
            current_section = None
            sections = {"KEY_EVENTS": [], "DAILY_PLAN": [], "PAIRS_FOCUS": []}
            for line in text.split("\n"):
                stripped = line.strip()
                for key in sections:
                    if stripped.startswith(key + ":"):
                        current_section = key
                        rest = stripped.split(":", 1)[1].strip()
                        if rest:
                            sections[key].append(rest)
                        break
                else:
                    if current_section and stripped:
                        sections[current_section].append(stripped)

            briefing.key_events = "\n".join(sections["KEY_EVENTS"])
            briefing.daily_plan = "\n".join(sections["DAILY_PLAN"])
            briefing.pairs_focus = "\n".join(sections["PAIRS_FOCUS"])

            self._week_ahead = briefing
            self._save_week_ahead(briefing)
            _log.info("News intel: week-ahead briefing prepared")
            return briefing
        except Exception as e:
            _log.warning(f"Week-ahead preparation failed: {e}")
            return None

    # --- Context for AI prompt ---

    def get_relevant_briefings(self, utc_now: datetime) -> str:
        """Return all relevant briefings as formatted text for the AI prompt."""
        parts = []

        # Upcoming event briefings (within 4 hours)
        window = utc_now + timedelta(hours=4)
        for key, briefing in self._briefings.items():
            try:
                event_time = datetime.fromisoformat(briefing.event_time)
                if utc_now - timedelta(hours=1) <= event_time <= window:
                    minutes_away = (event_time - utc_now).total_seconds() / 60
                    parts.append(
                        f"\n--- UPCOMING: {briefing.event_title} ({briefing.currency}) "
                        f"in {int(minutes_away)} min ---\n"
                        f"What: {briefing.what_it_measures}\n"
                        f"Expectation: {briefing.market_expectation}\n"
                        f"Previous: {briefing.previous_result}\n"
                        f"Scenarios: {briefing.impact_scenarios}\n"
                        f"Pairs affected: {briefing.affected_pairs}\n"
                        f"Advice: {briefing.positioning_advice}"
                    )
            except (ValueError, TypeError):
                continue

        # Recent post-event notes (within last 4 hours)
        for key, note in self._post_notes.items():
            try:
                event_time = datetime.fromisoformat(note.event_time)
                if utc_now - timedelta(hours=4) <= event_time <= utc_now:
                    parts.append(
                        f"\n--- POST-EVENT: {note.event_title} ---\n"
                        f"Result: {note.what_happened}\n"
                        f"Reaction: {note.market_reaction}\n"
                        f"Outlook: {note.continuation_or_reversal}\n"
                        f"Bias: {note.updated_bias}"
                    )
            except (ValueError, TypeError):
                continue

        # Week-ahead context
        if self._week_ahead:
            parts.append(
                f"\n--- WEEK PLAN ---\n"
                f"Key events: {self._week_ahead.key_events}\n"
                f"Today's focus: {self._week_ahead.pairs_focus}"
            )

        if not parts:
            return "No news intelligence available for this window."

        return "=== NEWS INTELLIGENCE ===\n" + "\n".join(parts)

    # --- Persistence ---

    def _save_briefing(self, key: str, briefing: EventBriefing) -> None:
        try:
            safe_key = key.replace("/", "_").replace(":", "_")[:80]
            path = self._cache_dir / f"{safe_key}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(briefing), f, indent=2, default=str)
        except IOError as e:
            _log.error(f"Failed to cache briefing: {e}")

    def _save_week_ahead(self, briefing: WeekAheadBriefing) -> None:
        try:
            path = self._cache_dir / "week_ahead.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(briefing), f, indent=2, default=str)
        except IOError as e:
            _log.error(f"Failed to save week-ahead: {e}")

    def _load_cached_briefings(self) -> None:
        """Load previously cached briefings from disk on startup."""
        try:
            # Load week-ahead
            wa_path = self._cache_dir / "week_ahead.json"
            if wa_path.exists():
                data = json.loads(wa_path.read_text(encoding="utf-8"))
                self._week_ahead = WeekAheadBriefing(**data)

            # Load recent event briefings
            now = datetime.now(timezone.utc)
            for path in self._cache_dir.glob("*.json"):
                if path.name == "week_ahead.json":
                    continue
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    event_time = datetime.fromisoformat(data.get("event_time", ""))
                    if event_time > now - timedelta(hours=24):
                        key = f"{data['event_title']}_{data['event_time']}"
                        self._briefings[key] = EventBriefing(**data)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        except Exception as e:
            _log.warning(f"Failed to load cached briefings: {e}")
