"""Tests for Data Quality Gate."""

import pytest
from datetime import datetime, timedelta, timezone
from src.scalp_mode.gates.data_quality_gate import DataQualityGate, DataQualityResult


def _utc_now():
    return datetime.now(timezone.utc)


class TestDataQualityGate:
    def setup_method(self):
        self.config = {
            "heartbeat_timeout_sec": 10,
            "candle_late_sec": 5,
            "stale_price_sec": 15,
            "api_timeout_ms": 2000,
        }
        self.gate = DataQualityGate(self.config)

    def test_initial_state_ok(self):
        # No data received yet — heartbeat and price are None, should pass
        result = self.gate.check(_utc_now())
        assert result.is_ok is True

    def test_heartbeat_ok(self):
        now = _utc_now()
        self.gate.update_heartbeat(now - timedelta(seconds=5))
        result = self.gate.check(now)
        assert result.is_ok is True

    def test_heartbeat_timeout(self):
        now = _utc_now()
        self.gate.update_heartbeat(now - timedelta(seconds=15))
        result = self.gate.check(now)
        assert result.is_ok is False
        assert result.issue == "heartbeat_timeout"

    def test_stale_price(self):
        now = _utc_now()
        self.gate.update_heartbeat(now)  # heartbeat OK
        self.gate.update_price(now - timedelta(seconds=20))
        result = self.gate.check(now)
        assert result.is_ok is False
        assert result.issue == "stale_price"

    def test_price_fresh(self):
        now = _utc_now()
        self.gate.update_heartbeat(now)
        self.gate.update_price(now - timedelta(seconds=5))
        result = self.gate.check(now)
        assert result.is_ok is True

    def test_api_rate_limit_429(self):
        now = _utc_now()
        self.gate.update_heartbeat(now)
        self.gate.update_price(now)
        self.gate.update_api_response(latency_ms=500, status_code=429)
        result = self.gate.check(now)
        assert result.is_ok is False
        assert result.issue == "api_rate_limit"

    def test_api_slow_response(self):
        now = _utc_now()
        self.gate.update_heartbeat(now)
        self.gate.update_price(now)
        self.gate.update_api_response(latency_ms=3000, status_code=200)
        result = self.gate.check(now)
        assert result.is_ok is False
        assert result.issue == "api_rate_limit"
        assert result.details["latency_ms"] == 3000

    def test_api_normal(self):
        now = _utc_now()
        self.gate.update_heartbeat(now)
        self.gate.update_price(now)
        self.gate.update_api_response(latency_ms=500, status_code=200)
        result = self.gate.check(now)
        assert result.is_ok is True

    def test_indicator_nan(self):
        now = _utc_now()
        self.gate.update_heartbeat(now)
        self.gate.update_price(now)
        self.gate.update_api_response(latency_ms=500, status_code=200)
        self.gate.update_indicators(valid=False, invalid_name="RSI_M1")
        result = self.gate.check(now)
        assert result.is_ok is False
        assert result.issue == "indicator_nan"
        assert result.details["indicator"] == "RSI_M1"

    def test_indicator_recovery(self):
        now = _utc_now()
        self.gate.update_heartbeat(now)
        self.gate.update_price(now)
        self.gate.update_api_response(latency_ms=500, status_code=200)
        self.gate.update_indicators(valid=False, invalid_name="EMA20")
        self.gate.update_indicators(valid=True)
        result = self.gate.check(now)
        assert result.is_ok is True
