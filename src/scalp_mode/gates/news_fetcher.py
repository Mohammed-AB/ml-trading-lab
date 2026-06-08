"""News Calendar Fetcher — Retrieves economic events for News Gate.

Fetches high-impact economic events from free public sources:
1. ForexFactory (HTML scraping)
2. Investing.com calendar API
3. Manual JSON file (fallback)

The fetcher runs periodically (default: every 6 hours) and writes
events to a JSON file that NewsGate reads.

Usage:
    # Standalone: fetch and save
    fetcher = NewsCalendarFetcher(output_path="data/news_events.json")
    fetcher.fetch_and_save()

    # Integrated: auto-update NewsGate
    fetcher = NewsCalendarFetcher()
    fetcher.update_gate(news_gate)
"""

import json
import logging
import re
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

from .news_gate import NewsGate, NewsEvent

_log = logging.getLogger("scalp_mode")

# Currencies relevant to V1 pairs (EUR_USD, USD_JPY, GBP_USD)
V1_CURRENCIES = {"USD", "EUR", "GBP", "JPY"}


class NewsCalendarFetcher:
    """Fetches economic calendar events from public sources.

    Primary source: Forex Factory (HTML).
    Fallback: Investing.com calendar.
    Last resort: local JSON file.
    """

    FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    # Free community mirror of ForexFactory data

    def __init__(self, output_path: str | Path = "data/news_events.json",
                 auto_interval_hours: float = 6):
        self._output_path = Path(output_path)
        self._interval = auto_interval_hours * 3600
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def fetch_events(self) -> list[dict]:
        """Fetch events from available sources. Returns raw event dicts.

        Tries sources in order until one succeeds.
        """
        events = self._fetch_ff_community()
        if events:
            return events

        _log.warning("All news sources failed. Using cached events if available.")
        return self._load_cached()

    def _fetch_ff_community(self) -> list[dict]:
        """Fetch from ForexFactory community JSON mirror."""
        try:
            resp = requests.get(self.FF_URL, timeout=15, headers={
                "User-Agent": "ScalpMode/1.0"
            })
            if resp.status_code != 200:
                _log.warning(f"FF community fetch failed: HTTP {resp.status_code}")
                return []

            raw = resp.json()
            events = []
            for item in raw:
                impact = item.get("impact", "").lower()
                if impact not in ("high", "medium"):
                    continue

                currency = item.get("country", "").upper()
                if currency not in V1_CURRENCIES:
                    continue

                # Parse date+time
                date_str = item.get("date", "")
                if not date_str:
                    continue

                try:
                    # FF format: "2026-03-27T13:30:00-04:00" or similar
                    dt = datetime.fromisoformat(date_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                except (ValueError, TypeError):
                    continue

                events.append({
                    "timestamp_utc": dt.isoformat(),
                    "currency": currency,
                    "impact": "high" if impact == "high" else "medium",
                    "title": item.get("title", "Unknown"),
                })

            _log.info(f"Fetched {len(events)} events from FF community mirror")
            return events

        except requests.exceptions.RequestException as e:
            _log.warning(f"FF community fetch error: {e}")
            return []
        except (json.JSONDecodeError, KeyError) as e:
            _log.warning(f"FF community parse error: {e}")
            return []

    def _load_cached(self) -> list[dict]:
        """Load events from local cached file."""
        if self._output_path.exists():
            try:
                with open(self._output_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def fetch_and_save(self) -> list[dict]:
        """Fetch events and save to JSON file."""
        events = self.fetch_events()
        if events:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._output_path, "w") as f:
                json.dump(events, f, indent=2, ensure_ascii=False)
            _log.info(f"Saved {len(events)} events to {self._output_path}")
        return events

    def update_gate(self, gate: NewsGate) -> int:
        """Fetch events and load them into a NewsGate instance.

        Returns number of events loaded.
        """
        events = self.fetch_and_save()
        if events:
            parsed = []
            for item in events:
                try:
                    ts = datetime.fromisoformat(
                        item["timestamp_utc"].replace("Z", "+00:00"))
                    parsed.append(NewsEvent(
                        timestamp_utc=ts,
                        currency=item["currency"],
                        impact=item["impact"],
                        title=item["title"],
                    ))
                except (KeyError, ValueError):
                    continue
            gate.set_events(parsed)
            return len(parsed)
        return 0

    def start_auto_update(self, gate: NewsGate) -> None:
        """Start background thread that refreshes events periodically."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._auto_loop, args=(gate,),
            daemon=True, name="news-fetcher")
        self._thread.start()

    def stop_auto_update(self) -> None:
        """Stop background auto-update thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _auto_loop(self, gate: NewsGate) -> None:
        """Background loop: fetch events every N hours."""
        while self._running:
            try:
                count = self.update_gate(gate)
                _log.info(f"News calendar updated: {count} events")
            except Exception as e:
                _log.error(f"News calendar update failed: {e}")

            # Sleep in small increments to allow shutdown
            sleep_end = time.monotonic() + self._interval
            while self._running and time.monotonic() < sleep_end:
                time.sleep(10)
