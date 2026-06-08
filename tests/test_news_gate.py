"""Tests for News Gate."""

import pytest
from datetime import datetime, timedelta, timezone
from src.scalp_mode.gates.news_gate import NewsGate, NewsEvent


def _utc(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


class TestNewsGate:
    def setup_method(self):
        self.gate = NewsGate(pre_minutes=10, post_minutes=5)

    def test_no_events_is_safe(self):
        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 13, 30))
        assert result.is_safe is True
        assert result.next_event_minutes is None

    def test_high_impact_blocks_before(self):
        event = NewsEvent(
            timestamp_utc=_utc(2026, 3, 27, 13, 30),
            currency="USD",
            impact="high",
            title="NFP",
        )
        self.gate.set_events([event])

        # 5 minutes before — within 10min freeze
        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 13, 25))
        assert result.is_safe is False
        assert result.blocking_event == "NFP"
        assert result.next_event_minutes == 5

    def test_high_impact_blocks_after(self):
        event = NewsEvent(
            timestamp_utc=_utc(2026, 3, 27, 13, 30),
            currency="USD",
            impact="high",
            title="NFP",
        )
        self.gate.set_events([event])

        # 3 minutes after — within 5min post-freeze
        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 13, 33))
        assert result.is_safe is False

    def test_safe_after_post_window(self):
        event = NewsEvent(
            timestamp_utc=_utc(2026, 3, 27, 13, 30),
            currency="USD",
            impact="high",
            title="NFP",
        )
        self.gate.set_events([event])

        # 6 minutes after — outside post-freeze
        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 13, 36))
        assert result.is_safe is True

    def test_medium_impact_ignored(self):
        event = NewsEvent(
            timestamp_utc=_utc(2026, 3, 27, 13, 30),
            currency="USD",
            impact="medium",
            title="Trade Balance",
        )
        self.gate.set_events([event])

        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 13, 25))
        assert result.is_safe is True

    def test_unrelated_currency_ignored(self):
        event = NewsEvent(
            timestamp_utc=_utc(2026, 3, 27, 13, 30),
            currency="GBP",
            impact="high",
            title="BOE Rate",
        )
        self.gate.set_events([event])

        # GBP event shouldn't affect EUR_USD
        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 13, 25))
        assert result.is_safe is True

    def test_related_currency_blocks(self):
        event = NewsEvent(
            timestamp_utc=_utc(2026, 3, 27, 13, 30),
            currency="GBP",
            impact="high",
            title="BOE Rate",
        )
        self.gate.set_events([event])

        # GBP event should block GBP_USD
        result = self.gate.check("GBP_USD", _utc(2026, 3, 27, 13, 25))
        assert result.is_safe is False

    def test_next_event_tracking(self):
        event = NewsEvent(
            timestamp_utc=_utc(2026, 3, 27, 15, 0),
            currency="USD",
            impact="high",
            title="FOMC",
        )
        self.gate.set_events([event])

        # 30 minutes before the 10-min freeze window
        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 14, 20))
        assert result.is_safe is True
        assert result.next_event_minutes == 40
        assert result.blocking_event == "FOMC"

    def test_multiple_events(self):
        events = [
            NewsEvent(_utc(2026, 3, 27, 13, 30), "USD", "high", "NFP"),
            NewsEvent(_utc(2026, 3, 27, 15, 0), "EUR", "high", "ECB"),
        ]
        self.gate.set_events(events)

        # Between events, after NFP post-window, before ECB pre-window
        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 14, 0))
        assert result.is_safe is True
        assert result.next_event_minutes == 60

    def test_load_events_from_json(self, tmp_path):
        import json
        events = [
            {
                "timestamp_utc": "2026-03-27T13:30:00Z",
                "currency": "USD",
                "impact": "high",
                "title": "NFP",
            }
        ]
        filepath = tmp_path / "events.json"
        filepath.write_text(json.dumps(events))

        self.gate.load_events(filepath)
        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 13, 25))
        assert result.is_safe is False

    def test_load_nonexistent_file(self):
        self.gate.load_events("/nonexistent/file.json")
        # Should not raise, just have no events
        result = self.gate.check("EUR_USD", _utc(2026, 3, 27, 13, 25))
        assert result.is_safe is True
