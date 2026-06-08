"""Tests for PendingOrderManager and BrokerReconciler."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from src.scalp_mode.execution.pending_manager import (
    PendingOrderManager, PendingOrder, BrokerReconciler,
)
from src.scalp_mode.execution.executor import Executor, ExecutionResult
from src.scalp_mode.execution.order_builder import OrderBuilder, OrderSpec, OrderType
from src.scalp_mode.execution.trade_manager import TradeManager, ManagedTrade
from src.scalp_mode.logger import ScalpLogger


ORDER_CONFIG = {
    "limit_ttl_seconds": 180, "fallback_market": True,
    "fallback_max_atr_distance": 0.3, "fallback_cooldown_min": 2,
    "price_bound_slippage": 0.2,
}

MODEL_CONFIG = {
    "time_stop_min": 6, "sl_move_threshold_R": 0.8,
    "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4],
}


def _utc(minute=0):
    return datetime(2026, 1, 7, 14, minute, tzinfo=timezone.utc)


@pytest.fixture
def components(tmp_path):
    log_config = {
        "log_dir": str(tmp_path / "logs"),
        "decision_log_file": "d.jsonl", "trade_log_file": "t.jsonl",
        "cycle_log_file": "c.jsonl", "system_log_file": "s.log",
        "max_file_size_mb": 1, "backup_count": 1, "level": "DEBUG",
    }
    logger = ScalpLogger(log_config)
    executor = Executor("http://fake", "token", "acc-1", logger)
    builder = OrderBuilder(ORDER_CONFIG)
    trade_mgr = TradeManager(MODEL_CONFIG, "http://fake", "token", "acc-1", logger)

    yield {
        "logger": logger, "executor": executor,
        "builder": builder, "trade_mgr": trade_mgr,
    }
    logger.close()


def _make_pending(order_id="ord-1", submitted_at=None):
    return PendingOrder(
        order_id=order_id, signal_id="sig-1",
        pair="EUR_USD", direction="long", units=10000,
        entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
        submitted_at=submitted_at or _utc(0),
        ttl_seconds=180, atr=0.0005,
        spread_at_signal=0.5, max_spread=0.8,
    )


class TestPendingOrderManager:
    def test_track_pending_order(self, components):
        mgr = PendingOrderManager(
            components["executor"], components["builder"],
            components["trade_mgr"], components["logger"],
            poll_interval_sec=0)
        po = _make_pending()
        mgr.track(po)
        assert mgr.pending_count == 1

    def test_filled_order_registers_trade(self, components):
        mgr = PendingOrderManager(
            components["executor"], components["builder"],
            components["trade_mgr"], components["logger"],
            poll_interval_sec=0)
        po = _make_pending()
        mgr.track(po)

        with patch.object(components["executor"], "check_order_status") as mock:
            mock.return_value = ExecutionResult(
                success=True, order_id="ord-1", trade_id="trade-99",
                fill_price=1.08555, broker_status="filled")

            resolved = mgr.poll_all(_utc(1), {})

        assert "ord-1" in resolved
        assert mgr.pending_count == 0
        assert len(components["trade_mgr"].open_trades) == 1
        assert components["trade_mgr"].open_trades[0].trade_id == "trade-99"

    def test_expired_order_triggers_fallback(self, components):
        mgr = PendingOrderManager(
            components["executor"], components["builder"],
            components["trade_mgr"], components["logger"],
            poll_interval_sec=0)
        po = _make_pending(submitted_at=_utc(0))
        mgr.track(po)

        with patch.object(components["executor"], "check_order_status") as mock_status:
            mock_status.return_value = ExecutionResult(
                success=False, order_id="ord-1",
                broker_status="expired", reject_reason="expired")

            with patch.object(components["executor"], "submit") as mock_submit:
                mock_submit.return_value = ExecutionResult(
                    success=True, order_id="ord-fb", trade_id="trade-fb",
                    fill_price=1.08552, broker_status="filled")

                # Provide live prices for fallback
                resolved = mgr.poll_all(
                    _utc(1), {"EUR_USD": (1.08548, 1.08553)})

        assert "ord-1" in resolved
        assert mgr.pending_count == 0
        # Fallback should have registered a trade
        assert len(components["trade_mgr"].open_trades) == 1

    def test_ttl_expiry_triggers_fallback(self, components):
        mgr = PendingOrderManager(
            components["executor"], components["builder"],
            components["trade_mgr"], components["logger"],
            poll_interval_sec=0)
        # Submitted 4 minutes ago, TTL was 180s (3 min)
        po = _make_pending(submitted_at=_utc(0))
        po.ttl_seconds = 180
        mgr.track(po)

        with patch.object(components["executor"], "submit") as mock_submit:
            mock_submit.return_value = ExecutionResult(
                success=True, trade_id="trade-ttl",
                fill_price=1.08551, broker_status="filled")

            # 4 minutes later (> TTL + 5s grace)
            resolved = mgr.poll_all(
                _utc(4), {"EUR_USD": (1.08548, 1.08553)})

        assert "ord-1" in resolved

    def test_still_pending_not_resolved(self, components):
        mgr = PendingOrderManager(
            components["executor"], components["builder"],
            components["trade_mgr"], components["logger"],
            poll_interval_sec=0)
        po = _make_pending()
        mgr.track(po)

        with patch.object(components["executor"], "check_order_status") as mock:
            mock.return_value = ExecutionResult(
                success=True, order_id="ord-1",
                broker_status="pending")

            resolved = mgr.poll_all(_utc(1), {})

        assert len(resolved) == 0
        assert mgr.pending_count == 1


class TestBrokerReconciler:
    def test_detects_broker_closed_trade(self, components):
        trade_mgr = components["trade_mgr"]
        reconciler = BrokerReconciler(
            components["executor"], trade_mgr,
            components["logger"], reconcile_interval_sec=0)

        # Add a local trade
        trade_mgr.add_trade(ManagedTrade(
            trade_id="trade-50", pair="EUR_USD", direction="long",
            entry_price=1.085, sl_price=1.084, tp_price=1.086,
            units=10000, open_time=_utc(0), risk_amount=0.001))
        assert len(trade_mgr.open_trades) == 1

        # Broker says no open trades (TP/SL was hit)
        with patch.object(components["executor"], "get_open_trades",
                          return_value=[]):
            changes = reconciler.reconcile(_utc(5))

        assert "trade-50" in changes["closed_by_broker"]
        assert len(trade_mgr.open_trades) == 0

    def test_no_changes_when_in_sync(self, components):
        trade_mgr = components["trade_mgr"]
        reconciler = BrokerReconciler(
            components["executor"], trade_mgr,
            components["logger"], reconcile_interval_sec=0)

        trade_mgr.add_trade(ManagedTrade(
            trade_id="trade-60", pair="EUR_USD", direction="long",
            entry_price=1.085, sl_price=1.084, tp_price=1.086,
            units=10000, open_time=_utc(0), risk_amount=0.001))

        with patch.object(components["executor"], "get_open_trades",
                          return_value=[{"id": "trade-60"}]):
            changes = reconciler.reconcile(_utc(5))

        assert len(changes["closed_by_broker"]) == 0
        assert len(trade_mgr.open_trades) == 1

    def test_respects_interval(self, components):
        reconciler = BrokerReconciler(
            components["executor"], components["trade_mgr"],
            components["logger"], reconcile_interval_sec=30)

        with patch.object(components["executor"], "get_open_trades") as mock:
            mock.return_value = []
            reconciler.reconcile(_utc(0))
            reconciler.reconcile(_utc(0))  # Should skip (too soon)

        # Only called once due to interval
        assert mock.call_count == 1


class TestExecutorCheckOrderStatus:
    def test_filled_returns_price(self, components):
        with patch("src.scalp_mode.execution.executor.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "order": {
                    "state": "FILLED",
                    "price": "1.08555",
                    "filledTime": "2026-01-07T14:01:00Z",
                    "fillingTransactionID": "123",
                    "tradeOpenedID": "456",
                }}
            mock_get.return_value = mock_resp

            result = components["executor"].check_order_status("ord-1")

        assert result.broker_status == "filled"
        assert result.fill_price == 1.08555
        assert result.trade_id == "456"

    def test_pending_returns_pending(self, components):
        with patch("src.scalp_mode.execution.executor.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "order": {"state": "PENDING"}}
            mock_get.return_value = mock_resp

            result = components["executor"].check_order_status("ord-2")

        assert result.broker_status == "pending"

    def test_cancelled_returns_expired(self, components):
        with patch("src.scalp_mode.execution.executor.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "order": {"state": "CANCELLED", "cancelReason": "TIME_IN_FORCE_EXPIRED"}}
            mock_get.return_value = mock_resp

            result = components["executor"].check_order_status("ord-3")

        assert result.broker_status == "expired"
        assert result.reject_reason == "TIME_IN_FORCE_EXPIRED"

    def test_404_returns_expired(self, components):
        with patch("src.scalp_mode.execution.executor.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock_get.return_value = mock_resp

            result = components["executor"].check_order_status("ord-4")

        assert result.broker_status == "expired"
