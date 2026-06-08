"""Tests for Execution Layer."""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.scalp_mode.execution.executor import (
    Executor,
    ExecutionResult,
    parse_account_nav,
    parse_oanda_decimal,
)
from src.scalp_mode.execution.order_builder import OrderSpec, OrderType
from src.scalp_mode.logger import ScalpLogger


@pytest.fixture
def logger(tmp_path):
    log_config = {
        "log_dir": str(tmp_path / "logs"),
        "decision_log_file": "d.jsonl", "trade_log_file": "t.jsonl",
        "cycle_log_file": "c.jsonl", "system_log_file": "s.log",
        "max_file_size_mb": 1, "backup_count": 1, "level": "DEBUG",
    }
    lg = ScalpLogger(log_config)
    yield lg
    lg.close()


def _make_order(signal_id="sig-001"):
    return OrderSpec(
        order_type=OrderType.MARKET, pair="EUR_USD", direction="long",
        units=10000, price=None, price_bound=1.08557,
        sl_price=1.08500, tp_price=1.08600,
        ttl_seconds=0, expire_time=None, signal_id=signal_id,
    )


class TestExecutorIdempotency:
    def test_duplicate_signal_rejected(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        order = _make_order("dup-signal")

        with patch("src.scalp_mode.execution.executor.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_resp.json.return_value = {
                "orderFillTransaction": {
                    "orderID": "1", "price": "1.08555",
                    "time": "2026-03-27T14:00:00Z",
                    "tradeOpened": {"tradeID": "100"},
                }}
            mock_post.return_value = mock_resp

            # First call succeeds
            result1 = executor.submit(order, {"order": {}})
            assert result1.success is True

            # Second call with same signal_id is rejected
            result2 = executor.submit(order, {"order": {}})
            assert result2.success is False
            assert result2.reject_reason == "duplicate_signal_id"


class TestExecutorResponses:
    def test_market_fill_success(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)

        with patch("src.scalp_mode.execution.executor.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_resp.json.return_value = {
                "orderFillTransaction": {
                    "orderID": "42",
                    "price": "1.08556",
                    "time": "2026-03-27T14:00:00.123Z",
                    "tradeOpened": {"tradeID": "101"},
                }}
            mock_post.return_value = mock_resp

            result = executor.submit(_make_order("fill-1"), {"order": {}})
            assert result.success is True
            assert result.broker_status == "filled"
            assert result.fill_price == 1.08556
            assert result.trade_id == "101"
            assert result.e2e_latency_ms >= 0

    def test_limit_pending(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)

        with patch("src.scalp_mode.execution.executor.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_resp.json.return_value = {
                "orderCreateTransaction": {"id": "55"}}
            mock_post.return_value = mock_resp

            result = executor.submit(_make_order("pend-1"), {"order": {}})
            assert result.success is True
            assert result.broker_status == "pending"
            assert result.order_id == "55"

    def test_rejection_400(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)

        with patch("src.scalp_mode.execution.executor.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_resp.json.return_value = {
                "orderRejectTransaction": {
                    "rejectReason": "BOUNDS_VIOLATION"}}
            mock_post.return_value = mock_resp

            result = executor.submit(_make_order("rej-1"), {"order": {}})
            assert result.success is False
            assert result.broker_status == "rejected"
            assert result.reject_reason == "BOUNDS_VIOLATION"

    def test_timeout_handling(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)

        import requests as req
        with patch("src.scalp_mode.execution.executor.requests.post",
                   side_effect=req.exceptions.Timeout):
            result = executor.submit(_make_order("timeout-1"), {"order": {}})
            assert result.success is False
            assert result.reject_reason == "timeout"

    def test_network_error(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)

        import requests as req
        with patch("src.scalp_mode.execution.executor.requests.post",
                   side_effect=req.exceptions.ConnectionError):
            result = executor.submit(_make_order("net-1"), {"order": {}})
            assert result.success is False
            assert "network_error" in result.reject_reason


class TestOandaAccountParsing:
    def test_parse_oanda_decimal(self):
        assert parse_oanda_decimal("556.1234") == pytest.approx(556.1234)
        assert parse_oanda_decimal(100) == 100.0
        assert parse_oanda_decimal(None) is None
        assert parse_oanda_decimal("") is None
        assert parse_oanda_decimal(True) is None

    def test_parse_account_nav_prefers_nav(self):
        assert parse_account_nav({"NAV": "556.0", "balance": "999"}) == 556.0

    def test_parse_account_nav_falls_back_to_balance(self):
        assert parse_account_nav({"balance": "556.50"}) == 556.5

    def test_parse_account_nav_none_when_empty(self):
        assert parse_account_nav({}) is None
        assert parse_account_nav(None) is None
