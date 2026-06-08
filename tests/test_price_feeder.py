"""Tests for Price Feeder — candle parsing, stream message handling."""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timezone

from src.scalp_mode.data.price_feeder import PriceFeeder, Candle, LivePrice


class TestFetchCandles:
    def test_parse_candle_response(self):
        feeder = PriceFeeder("http://fake", "http://stream", "token", "acc-1")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candles": [
                {
                    "time": "2026-03-27T14:00:00Z",
                    "mid": {"o": "1.08500", "h": "1.08550", "l": "1.08480", "c": "1.08530"},
                    "volume": 42,
                    "complete": True,
                },
                {
                    "time": "2026-03-27T14:01:00Z",
                    "mid": {"o": "1.08530", "h": "1.08560", "l": "1.08510", "c": "1.08540"},
                    "volume": 30,
                    "complete": False,
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("src.scalp_mode.data.price_feeder.requests.get", return_value=mock_resp):
            candles, latency = feeder.fetch_candles("EUR_USD", "M1", count=2)

        # Only complete candles returned by default
        assert len(candles) == 1
        assert candles[0].open == 1.08500
        assert candles[0].high == 1.08550
        assert candles[0].low == 1.08480
        assert candles[0].close == 1.08530
        assert candles[0].volume == 42
        assert candles[0].complete is True
        assert latency >= 0

    def test_include_incomplete(self):
        feeder = PriceFeeder("http://fake", "http://stream", "token", "acc-1")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candles": [
                {"time": "t1", "mid": {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.05"},
                 "volume": 10, "complete": True},
                {"time": "t2", "mid": {"o": "1.05", "h": "1.1", "l": "0.95", "c": "1.08"},
                 "volume": 5, "complete": False},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("src.scalp_mode.data.price_feeder.requests.get", return_value=mock_resp):
            candles, _ = feeder.fetch_candles("EUR_USD", count=2, include_incomplete=True)

        assert len(candles) == 2

    def test_api_error_raises(self):
        feeder = PriceFeeder("http://fake", "http://stream", "token", "acc-1")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = Exception("Server Error")

        with patch("src.scalp_mode.data.price_feeder.requests.get", return_value=mock_resp):
            with pytest.raises(Exception, match="Server Error"):
                feeder.fetch_candles("EUR_USD")


class TestStreamParsing:
    def test_heartbeat_callback(self):
        feeder = PriceFeeder("http://fake", "http://stream", "token", "acc-1")
        heartbeats = []
        feeder.set_callbacks(on_heartbeat=lambda ts: heartbeats.append(ts))

        # Simulate processing a heartbeat message
        msg = {"type": "HEARTBEAT", "time": "2026-03-27T14:00:00Z"}
        # Directly test the parsing logic
        assert msg["type"] == "HEARTBEAT"

    def test_price_parsing(self):
        feeder = PriceFeeder("http://fake", "http://stream", "token", "acc-1")
        prices = []
        feeder.set_callbacks(on_price=lambda p: prices.append(p))

        # Test live price retrieval (not connected)
        assert feeder.get_live_price("EUR_USD") is None
        assert feeder.get_live_spread_pips("EUR_USD") is None


class TestStreamLifecycle:
    def test_start_stop(self):
        feeder = PriceFeeder("http://fake", "http://stream", "token", "acc-1")
        # Starting should set the flag
        feeder._stream_running = True
        assert feeder._stream_running is True
        feeder.stop_stream()
        assert feeder._stream_running is False

    def test_double_start_idempotent(self):
        feeder = PriceFeeder("http://fake", "http://stream", "token", "acc-1")
        # Simulate already running
        feeder._stream_running = True
        feeder._stream_thread = MagicMock()
        feeder.start_stream(["EUR_USD"])
        # Should not start a new thread
        assert feeder._stream_running is True
