"""Tests for News Calendar Fetcher."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from src.scalp_mode.gates.news_fetcher import NewsCalendarFetcher
from src.scalp_mode.gates.news_gate import NewsGate


SAMPLE_FF_DATA = [
    {
        "title": "Non-Farm Payrolls",
        "country": "USD",
        "date": "2026-03-27T13:30:00-04:00",
        "impact": "High",
    },
    {
        "title": "ECB Rate Decision",
        "country": "EUR",
        "date": "2026-03-27T12:45:00+00:00",
        "impact": "High",
    },
    {
        "title": "Trade Balance",
        "country": "USD",
        "date": "2026-03-27T13:30:00-04:00",
        "impact": "Medium",
    },
    {
        "title": "BOJ Minutes",
        "country": "JPY",
        "date": "2026-03-27T00:00:00+09:00",
        "impact": "Low",
    },
    {
        "title": "AUD CPI",
        "country": "AUD",
        "date": "2026-03-27T01:30:00+10:00",
        "impact": "High",
    },
]


class TestFetchParsing:
    def test_parses_ff_community_data(self):
        fetcher = NewsCalendarFetcher(output_path="/tmp/test_events.json")

        with patch("src.scalp_mode.gates.news_fetcher.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = SAMPLE_FF_DATA
            mock_get.return_value = mock_resp

            events = fetcher.fetch_events()

        # Should include: NFP (USD/High), ECB (EUR/High), Trade Balance (USD/Medium)
        # Should exclude: BOJ (Low impact), AUD CPI (AUD not in V1)
        assert len(events) == 3
        titles = {e["title"] for e in events}
        assert "Non-Farm Payrolls" in titles
        assert "ECB Rate Decision" in titles
        assert "Trade Balance" in titles
        assert "AUD CPI" not in titles

    def test_utc_conversion(self):
        fetcher = NewsCalendarFetcher(output_path="/tmp/test_events.json")

        with patch("src.scalp_mode.gates.news_fetcher.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = SAMPLE_FF_DATA
            mock_get.return_value = mock_resp

            events = fetcher.fetch_events()

        # NFP: 13:30 EDT (UTC-4) = 17:30 UTC
        nfp = next(e for e in events if e["title"] == "Non-Farm Payrolls")
        ts = datetime.fromisoformat(nfp["timestamp_utc"])
        assert ts.hour == 17
        assert ts.minute == 30

    def test_http_failure_returns_empty(self):
        fetcher = NewsCalendarFetcher(output_path="/tmp/nonexistent.json")

        with patch("src.scalp_mode.gates.news_fetcher.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 503
            mock_get.return_value = mock_resp

            events = fetcher.fetch_events()

        assert events == []

    def test_network_error_returns_empty(self):
        import requests as req
        fetcher = NewsCalendarFetcher(output_path="/tmp/nonexistent.json")

        with patch("src.scalp_mode.gates.news_fetcher.requests.get",
                   side_effect=req.exceptions.ConnectionError("Connection refused")):
            events = fetcher.fetch_events()

        assert events == []


class TestSaveAndLoad:
    def test_save_and_load(self, tmp_path):
        output = tmp_path / "events.json"
        fetcher = NewsCalendarFetcher(output_path=output)

        with patch("src.scalp_mode.gates.news_fetcher.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = SAMPLE_FF_DATA
            mock_get.return_value = mock_resp

            saved = fetcher.fetch_and_save()

        assert output.exists()
        assert len(saved) == 3

        # Load cached
        fetcher2 = NewsCalendarFetcher(output_path=output)
        cached = fetcher2._load_cached()
        assert len(cached) == 3


class TestGateIntegration:
    def test_update_gate_loads_events(self, tmp_path):
        output = tmp_path / "events.json"
        fetcher = NewsCalendarFetcher(output_path=output)
        gate = NewsGate()

        with patch("src.scalp_mode.gates.news_fetcher.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = SAMPLE_FF_DATA
            mock_get.return_value = mock_resp

            count = fetcher.update_gate(gate)

        assert count == 3

        # Verify gate blocks during NFP
        result = gate.check("EUR_USD",
                            datetime(2026, 3, 27, 17, 25, tzinfo=timezone.utc))
        assert result.is_safe is False
        assert "Non-Farm" in result.blocking_event

    def test_update_gate_empty_source(self, tmp_path):
        import requests as req
        output = tmp_path / "events.json"
        fetcher = NewsCalendarFetcher(output_path=output)
        gate = NewsGate()

        with patch("src.scalp_mode.gates.news_fetcher.requests.get",
                   side_effect=req.exceptions.ConnectionError("fail")):
            count = fetcher.update_gate(gate)

        assert count == 0
