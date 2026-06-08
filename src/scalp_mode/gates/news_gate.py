"""News Gate — Step 2 in the decision pipeline.

Filters out trading during high-impact economic news events.
Per spec: 10 minutes before + 5 minutes after High-impact events,
with automatic extension if spread widens.

V1 implementation: loads events from a local JSON file updated periodically.
The actual fetching mechanism (from ForexFactory, Investing.com, etc.)
is a separate concern — this module only evaluates the gate.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


# Default freeze windows (minutes)
PRE_NEWS_MINUTES = 10
POST_NEWS_MINUTES = 5


@dataclass
class NewsEvent:
    timestamp_utc: datetime
    currency: str
    impact: str  # "high", "medium", "low"
    title: str


@dataclass
class NewsGateResult:
    is_safe: bool
    next_event_minutes: Optional[int] = None
    blocking_event: Optional[str] = None


class NewsGate:
    """Evaluates whether trading is safe given upcoming/recent news events.

    Usage:
        gate = NewsGate()
        gate.load_events("path/to/events.json")
        result = gate.check("EUR_USD", utc_now)
    """

    def __init__(self, pre_minutes: int = PRE_NEWS_MINUTES,
                 post_minutes: int = POST_NEWS_MINUTES):
        self._events: list[NewsEvent] = []
        self._pre_minutes = pre_minutes
        self._post_minutes = post_minutes

    def load_events(self, filepath: str | Path) -> None:
        """Load news events from a JSON file.

        Expected format:
        [
            {
                "timestamp_utc": "2026-03-27T13:30:00Z",
                "currency": "USD",
                "impact": "high",
                "title": "Non-Farm Payrolls"
            },
            ...
        ]
        """
        path = Path(filepath)
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._events = []
        for item in raw:
            self._events.append(NewsEvent(
                timestamp_utc=datetime.fromisoformat(
                    item["timestamp_utc"].replace("Z", "+00:00")
                ),
                currency=item["currency"].upper(),
                impact=item["impact"].lower(),
                title=item["title"],
            ))

        # Sort by time for efficient searching
        self._events.sort(key=lambda e: e.timestamp_utc)

    def set_events(self, events: list[NewsEvent]) -> None:
        """Set events directly (useful for testing)."""
        self._events = sorted(events, key=lambda e: e.timestamp_utc)

    def _pair_currencies(self, pair: str) -> tuple[str, str]:
        """Extract the two currencies from a pair like EUR_USD."""
        clean = pair.upper().replace("/", "_")
        parts = clean.split("_")
        return parts[0], parts[1]

    def check(self, pair: str, utc_now: datetime,
              current_spread_pips: float = 0,
              max_spread_pips: float = 0) -> NewsGateResult:
        """Check if trading is safe for the given pair at the given time.

        Only high-impact events trigger the gate.
        Freeze window: [event_time - pre_minutes, event_time + post_minutes].
        Per spec: auto-extend freeze if spread > max_spread during post-news window.

        Args:
            pair: Instrument
            utc_now: Current UTC time
            current_spread_pips: Current spread (for auto-extension check)
            max_spread_pips: Max allowed spread (for auto-extension check)
        """
        base_ccy, quote_ccy = self._pair_currencies(pair)
        pair_currencies = {base_ccy, quote_ccy}

        closest_minutes: Optional[int] = None
        blocking_event: Optional[str] = None

        for event in self._events:
            if event.impact != "high":
                continue
            if event.currency not in pair_currencies:
                continue

            # Time boundaries
            freeze_start = event.timestamp_utc - timedelta(minutes=self._pre_minutes)
            freeze_end = event.timestamp_utc + timedelta(minutes=self._post_minutes)

            # Currently in freeze window?
            if freeze_start <= utc_now <= freeze_end:
                minutes_to_event = (event.timestamp_utc - utc_now).total_seconds() / 60
                return NewsGateResult(
                    is_safe=False,
                    next_event_minutes=round(minutes_to_event),
                    blocking_event=event.title,
                )

            # Track next upcoming event
            if event.timestamp_utc > utc_now:
                minutes_away = (event.timestamp_utc - utc_now).total_seconds() / 60
                if closest_minutes is None or minutes_away < closest_minutes:
                    closest_minutes = round(minutes_away)
                    blocking_event = event.title

        # Spread auto-extension: if we just exited a post-news window but
        # spread is still elevated, extend the freeze (spec 2.3).
        if (current_spread_pips > 0 and max_spread_pips > 0
                and current_spread_pips > max_spread_pips):
            # Check if any high-impact event ended within the last 15 minutes
            for event in self._events:
                if event.impact != "high" or event.currency not in pair_currencies:
                    continue
                freeze_end = event.timestamp_utc + timedelta(minutes=self._post_minutes)
                extended_end = freeze_end + timedelta(minutes=15)
                if freeze_end <= utc_now <= extended_end:
                    return NewsGateResult(
                        is_safe=False,
                        next_event_minutes=0,
                        blocking_event=f"{event.title} (spread_extended)",
                    )

        return NewsGateResult(
            is_safe=True,
            next_event_minutes=closest_minutes,
            blocking_event=blocking_event,
        )

    def check_elevated_risk(self, pair: str, utc_now: datetime,
                            pre_window_min: int = 30,
                            post_window_min: int = 20
                            ) -> Optional[str]:
        """Return an event title if we're in an elevated-risk (not blocked)
        window around a high-impact event — Strategy should reduce size.

        Returns None when no elevated risk; otherwise returns the event title.
        """
        base_ccy, quote_ccy = self._pair_currencies(pair)
        pair_currencies = {base_ccy, quote_ccy}
        for event in self._events:
            if event.impact != "high" or event.currency not in pair_currencies:
                continue
            pre_start = event.timestamp_utc - timedelta(minutes=pre_window_min)
            post_end = event.timestamp_utc + timedelta(minutes=post_window_min)
            if pre_start <= utc_now <= post_end:
                return event.title
        return None

    @property
    def _events_list(self) -> list:
        return self._events
