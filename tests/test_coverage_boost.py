"""Additional coverage tests — targeting uncovered paths in critical modules."""

import json
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from src.scalp_mode.execution.executor import Executor, ExecutionResult
from src.scalp_mode.execution.trade_manager import (
    TradeManager, ManagedTrade, TradeActionResult, ExitReason,
)
from src.scalp_mode.execution.pending_manager import (
    PendingOrderManager, PendingOrder, BrokerReconciler,
)
from src.scalp_mode.data.price_feeder import PriceFeeder, Candle, LivePrice
from src.scalp_mode.logger import ScalpLogger


def _utc(minute=0):
    return datetime(2026, 1, 7, 14, minute, tzinfo=timezone.utc)


@pytest.fixture
def logger(tmp_path):
    cfg = {
        "log_dir": str(tmp_path / "logs"),
        "decision_log_file": "d.jsonl", "trade_log_file": "t.jsonl",
        "cycle_log_file": "c.jsonl", "system_log_file": "s.log",
        "max_file_size_mb": 1, "backup_count": 1, "level": "DEBUG",
    }
    lg = ScalpLogger(cfg)
    yield lg
    lg.close()


# ─── Executor additional coverage ───────────────────────────────────────

class TestExecutorAdditional:
    def test_submit_404_returns_account_not_found(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        from src.scalp_mode.execution.order_builder import OrderSpec, OrderType
        order = OrderSpec(
            order_type=OrderType.MARKET, pair="EUR_USD", direction="long",
            units=1000, price=None, price_bound=1.08557,
            sl_price=1.084, tp_price=1.086, ttl_seconds=0,
            expire_time=None, signal_id="test-404")

        with patch("src.scalp_mode.execution.executor.requests.post") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock.return_value = mock_resp

            result = executor.submit(order, {"order": {}})

        assert result.broker_status == "rejected"
        assert result.reject_reason == "account_not_found"

    def test_submit_500_returns_error(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        from src.scalp_mode.execution.order_builder import OrderSpec, OrderType
        order = OrderSpec(
            order_type=OrderType.MARKET, pair="EUR_USD", direction="long",
            units=1000, price=None, price_bound=1.08557,
            sl_price=1.084, tp_price=1.086, ttl_seconds=0,
            expire_time=None, signal_id="test-500")

        with patch("src.scalp_mode.execution.executor.requests.post") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock.return_value = mock_resp

            result = executor.submit(order, {"order": {}})

        assert result.broker_status == "error"
        assert "http_500" in result.reject_reason

    def test_handle_cancel_in_201(self, logger):
        """Test 201 response with orderCancelTransaction (rare but possible)."""
        executor = Executor("http://fake", "token", "acc-1", logger)
        from src.scalp_mode.execution.order_builder import OrderSpec, OrderType
        order = OrderSpec(
            order_type=OrderType.LIMIT, pair="EUR_USD", direction="long",
            units=1000, price=1.085, price_bound=None,
            sl_price=1.084, tp_price=1.086, ttl_seconds=180,
            expire_time="2026-01-07T14:03:00Z", signal_id="test-cancel-201")

        with patch("src.scalp_mode.execution.executor.requests.post") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 201
            mock_resp.json.return_value = {
                "orderCancelTransaction": {"reason": "TIME_IN_FORCE_EXPIRED"}}
            mock.return_value = mock_resp

            result = executor.submit(order, {"order": {}})

        assert result.success is False
        assert result.broker_status == "rejected"

    def test_check_order_status_network_error(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        import requests as req
        with patch("src.scalp_mode.execution.executor.requests.get",
                   side_effect=req.exceptions.Timeout):
            result = executor.check_order_status("ord-err")
        assert result.broker_status == "error"
        assert "network_error" in result.reject_reason

    def test_check_order_status_http_error(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        with patch("src.scalp_mode.execution.executor.requests.get") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 503
            mock.return_value = mock_resp
            result = executor.check_order_status("ord-503")
        assert result.broker_status == "error"

    def test_cancel_order_success(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        with patch("src.scalp_mode.execution.executor.requests.put") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock.return_value = mock_resp
            result = executor.cancel_order("ord-x")
        assert result == {"cancelled": True, "reason": "success"}

    def test_cancel_order_404(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        with patch("src.scalp_mode.execution.executor.requests.put") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 404
            mock.return_value = mock_resp
            result = executor.cancel_order("ord-gone")
        assert result == {"cancelled": False, "reason": "not_found"}

    def test_cancel_order_rejected(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        with patch("src.scalp_mode.execution.executor.requests.put") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_resp.headers = {"content-type": "application/json"}
            mock_resp.json.return_value = {
                "orderCancelRejectTransaction": {
                    "rejectReason": "ORDER_CANCEL_REJECTED"}}
            mock.return_value = mock_resp
            result = executor.cancel_order("ord-filled")
        assert result["cancelled"] is False
        assert result["reason"] == "ORDER_CANCEL_REJECTED"

    def test_cancel_order_network_error(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        import requests as req
        with patch("src.scalp_mode.execution.executor.requests.put",
                   side_effect=req.exceptions.Timeout):
            result = executor.cancel_order("ord-timeout")
        assert result["cancelled"] is False
        assert "network_error" in result["reason"]

    def test_cancel_pending_orders(self, logger):
        executor = Executor("http://fake", "token", "acc-1", logger)
        with patch("src.scalp_mode.execution.executor.requests.get") as mock_get, \
             patch("src.scalp_mode.execution.executor.requests.put") as mock_put:
            mock_get_resp = MagicMock()
            mock_get_resp.status_code = 200
            mock_get_resp.json.return_value = {
                "orders": [{"id": "100"}, {"id": "101"}]}
            mock_get.return_value = mock_get_resp

            mock_put_resp = MagicMock()
            mock_put_resp.status_code = 200
            mock_put.return_value = mock_put_resp

            cancelled = executor.cancel_pending_orders()
        assert cancelled == ["100", "101"]


# ─── TradeManager additional coverage ───────────────────────────────────

class TestTradeManagerAdditional:
    def test_execute_sl_move_success(self, logger):
        mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)
        mgr.add_trade(ManagedTrade(
            trade_id="t1", pair="EUR_USD", direction="long",
            entry_price=1.085, sl_price=1.084, tp_price=1.086,
            units=10000, open_time=_utc(0), risk_amount=0.001))

        with patch("src.scalp_mode.execution.trade_manager.requests.put") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock.return_value = mock_resp
            ok = mgr.execute_sl_move("t1", 1.0845)

        assert ok is True
        assert mgr.open_trades[0].sl_moved_to_be is True

    def test_execute_sl_move_failure(self, logger):
        mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)
        mgr.add_trade(ManagedTrade(
            trade_id="t1", pair="EUR_USD", direction="long",
            entry_price=1.085, sl_price=1.084, tp_price=1.086,
            units=10000, open_time=_utc(0), risk_amount=0.001))

        with patch("src.scalp_mode.execution.trade_manager.requests.put") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock.return_value = mock_resp
            ok = mgr.execute_sl_move("t1", 1.0845)

        assert ok is False

    def test_execute_close_retry_succeeds(self, logger):
        mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)
        mgr.add_trade(ManagedTrade(
            trade_id="t1", pair="EUR_USD", direction="long",
            entry_price=1.085, sl_price=1.084, tp_price=1.086,
            units=10000, open_time=_utc(0), risk_amount=0.001))

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.status_code = 200 if call_count[0] >= 2 else 500
            return resp

        with patch("src.scalp_mode.execution.trade_manager.requests.put",
                   side_effect=side_effect):
            ok = mgr.execute_close("t1", ExitReason.TIME_STOP, max_retries=3)

        assert ok is True
        assert call_count[0] == 2

    def test_execute_close_nonexistent_trade(self, logger):
        mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)
        ok = mgr.execute_close("nonexistent", ExitReason.MANUAL)
        assert ok is False

    def test_remove_trade(self, logger):
        mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)
        mgr.add_trade(ManagedTrade(
            trade_id="t1", pair="EUR_USD", direction="long",
            entry_price=1.085, sl_price=1.084, tp_price=1.086,
            units=10000, open_time=_utc(0), risk_amount=0.001))
        removed = mgr.remove_trade("t1")
        assert removed is not None
        assert removed.trade_id == "t1"
        assert len(mgr.open_trades) == 0


# ─── PriceFeeder additional coverage ────────────────────────────────────

class TestPriceFeederAdditional:
    def test_fetch_candles_history(self):
        feeder = PriceFeeder("http://fake", "http://stream", "token", "acc-1")
        with patch("src.scalp_mode.data.price_feeder.requests.get") as mock:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "candles": [
                    {"time": "t1", "mid": {"o": "1.0", "h": "1.1", "l": "0.9", "c": "1.05"},
                     "volume": 10, "complete": True}
                ]}
            mock_resp.raise_for_status = MagicMock()
            mock.return_value = mock_resp

            candles, lat = feeder.fetch_candles_history("EUR_USD", "M1", count=100)
        assert len(candles) == 1

    def test_stop_stream_when_not_running(self):
        feeder = PriceFeeder("http://fake", "http://stream", "token", "acc-1")
        feeder.stop_stream()  # Should not crash
        assert feeder._stream_running is False


# ─── BrokerReconciler additional coverage ───────────────────────────────

class TestReconcilerAdditional:
    def test_adds_unknown_broker_trade(self, logger):
        trade_mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)
        executor = Executor("http://fake", "token", "acc-1", logger)
        reconciler = BrokerReconciler(executor, trade_mgr, logger,
                                       reconcile_interval_sec=0)

        with patch.object(executor, "get_open_trades", return_value=[
            {"id": "trade-new", "instrument": "EUR_USD",
             "currentUnits": "10000", "price": "1.08550",
             "openTime": "2026-01-07T14:00:00.000000000Z",
             "stopLossOrder": {"price": "1.08500"},
             "takeProfitOrder": {"price": "1.08600"}}
        ]):
            changes = reconciler.reconcile(_utc(5))

        assert "trade-new" in changes["added_from_broker"]
        assert len(trade_mgr.open_trades) == 1
        t = trade_mgr.open_trades[0]
        assert t.pair == "EUR_USD"
        assert t.direction == "long"

    def test_skips_trade_without_sl(self, logger):
        trade_mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)
        executor = Executor("http://fake", "token", "acc-1", logger)
        reconciler = BrokerReconciler(executor, trade_mgr, logger,
                                       reconcile_interval_sec=0)

        with patch.object(executor, "get_open_trades", return_value=[
            {"id": "trade-nosl", "instrument": "EUR_USD",
             "currentUnits": "10000", "price": "1.08550",
             "stopLossOrder": {}, "takeProfitOrder": {}}
        ]):
            changes = reconciler.reconcile(_utc(5))

        # Should not be added (no SL price)
        assert len(trade_mgr.open_trades) == 0


# ─── PendingOrderManager cancel and check_trade_by_order ────────────────

class TestPendingManagerAdditional:
    def test_cancel_called_before_fallback(self, logger):
        """executor.cancel_order() is called before Market fallback."""
        executor = Executor("http://fake", "token", "acc-1", logger)
        from src.scalp_mode.execution.order_builder import OrderBuilder
        builder = OrderBuilder({
            "limit_ttl_seconds": 180, "fallback_market": True,
            "fallback_max_atr_distance": 0.3, "fallback_cooldown_min": 2,
            "price_bound_slippage": 0.2})
        trade_mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)

        mgr = PendingOrderManager(executor, builder, trade_mgr, logger,
                                   poll_interval_sec=0)
        po = PendingOrder(
            order_id="ord-cancel", signal_id="sig-c",
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            submitted_at=_utc(0), ttl_seconds=180,
            atr=0.0005, spread_at_signal=0.5, max_spread=0.8)
        mgr.track(po)

        with patch.object(executor, "check_order_status") as mock_status, \
             patch.object(executor, "cancel_order") as mock_cancel, \
             patch.object(executor, "check_trade_by_order", return_value=None), \
             patch.object(executor, "submit") as mock_submit:
            mock_status.return_value = ExecutionResult(
                success=False, order_id="ord-cancel",
                broker_status="expired", reject_reason="expired")
            mock_cancel.return_value = {"cancelled": True, "reason": "success"}
            mock_submit.return_value = ExecutionResult(
                success=False, broker_status="rejected",
                reject_reason="no_liquidity")
            mgr.poll_all(_utc(1), {"EUR_USD": (1.085, 1.08505)})

        # Order was already terminal (expired) at broker — cancel is skipped
        mock_cancel.assert_not_called()

    def test_cancel_rejected_filled_registers_trade(self, logger):
        """When cancel returns ORDER_CANCEL_REJECTED, check if filled and register."""
        executor = Executor("http://fake", "token", "acc-1", logger)
        from src.scalp_mode.execution.order_builder import OrderBuilder
        builder = OrderBuilder({
            "limit_ttl_seconds": 180, "fallback_market": True,
            "fallback_max_atr_distance": 0.3, "price_bound_slippage": 0.2})
        trade_mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)

        mgr = PendingOrderManager(executor, builder, trade_mgr, logger,
                                   poll_interval_sec=0)
        po = PendingOrder(
            order_id="ord-reject", signal_id="sig-r",
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            submitted_at=_utc(0), ttl_seconds=180,
            atr=0.0005, spread_at_signal=0.5, max_spread=0.8)
        mgr.track(po)

        with patch.object(executor, "check_order_status") as mock_status, \
             patch.object(executor, "cancel_order") as mock_cancel, \
             patch.object(executor, "check_trade_by_order") as mock_check:
            mock_status.return_value = ExecutionResult(
                success=False, order_id="ord-reject",
                broker_status="expired", reject_reason="expired")
            mock_cancel.return_value = {
                "cancelled": False, "reason": "ORDER_CANCEL_REJECTED"}
            mock_check.return_value = {
                "id": "trade-racewin", "price": "1.08555",
                "openTime": "2026-01-07T14:01:00Z"}
            mgr.poll_all(_utc(1), {"EUR_USD": (1.085, 1.08505)})

        # Trade should be registered (not fallback)
        assert len(trade_mgr.open_trades) == 1
        assert trade_mgr.open_trades[0].trade_id == "trade-racewin"

    def test_cancel_404_proceeds_to_fallback(self, logger):
        """When cancel returns not_found (404), proceed with Market fallback."""
        executor = Executor("http://fake", "token", "acc-1", logger)
        from src.scalp_mode.execution.order_builder import OrderBuilder
        builder = OrderBuilder({
            "limit_ttl_seconds": 180, "fallback_market": True,
            "fallback_max_atr_distance": 0.3, "price_bound_slippage": 0.2})
        trade_mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)

        mgr = PendingOrderManager(executor, builder, trade_mgr, logger,
                                   poll_interval_sec=0)
        po = PendingOrder(
            order_id="ord-404", signal_id="sig-404",
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            submitted_at=_utc(0), ttl_seconds=180,
            atr=0.0005, spread_at_signal=0.5, max_spread=0.8)
        mgr.track(po)

        with patch.object(executor, "check_order_status") as mock_status, \
             patch.object(executor, "cancel_order") as mock_cancel, \
             patch.object(executor, "check_trade_by_order", return_value=None), \
             patch.object(executor, "submit") as mock_submit:
            mock_status.return_value = ExecutionResult(
                success=False, order_id="ord-404",
                broker_status="expired", reject_reason="expired")
            mock_cancel.return_value = {"cancelled": False, "reason": "not_found"}
            mock_submit.return_value = ExecutionResult(
                success=True, trade_id="trade-fb-404",
                fill_price=1.08552, broker_status="filled",
                fill_time="2026-01-07T14:01:30Z")
            # Prices close to entry (within 0.3*ATR) so fallback is accepted
            mgr.poll_all(_utc(1), {"EUR_USD": (1.08548, 1.08553)})

        # Fallback should have been attempted and filled
        mock_submit.assert_called_once()
        assert len(trade_mgr.open_trades) == 1

    def test_cancel_network_error_proceeds_to_fallback(self, logger):
        """Network error during cancel → still proceed with fallback."""
        executor = Executor("http://fake", "token", "acc-1", logger)
        from src.scalp_mode.execution.order_builder import OrderBuilder
        builder = OrderBuilder({
            "limit_ttl_seconds": 180, "fallback_market": True,
            "fallback_max_atr_distance": 0.3, "price_bound_slippage": 0.2})
        trade_mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)

        mgr = PendingOrderManager(executor, builder, trade_mgr, logger,
                                   poll_interval_sec=0)
        po = PendingOrder(
            order_id="ord-net", signal_id="sig-net",
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            submitted_at=_utc(0), ttl_seconds=180,
            atr=0.0005, spread_at_signal=0.5, max_spread=0.8)
        mgr.track(po)

        with patch.object(executor, "check_order_status") as mock_status, \
             patch.object(executor, "cancel_order") as mock_cancel, \
             patch.object(executor, "check_trade_by_order", return_value=None), \
             patch.object(executor, "submit") as mock_submit:
            mock_status.return_value = ExecutionResult(
                success=False, order_id="ord-net",
                broker_status="expired", reject_reason="expired")
            mock_cancel.return_value = {
                "cancelled": False, "reason": "network_error:Timeout"}
            mock_submit.return_value = ExecutionResult(
                success=True, trade_id="trade-fb-net",
                fill_price=1.08553, broker_status="filled",
                fill_time="2026-01-07T14:01:30Z")
            # Prices close to entry (within 0.3*ATR) so fallback is accepted
            mgr.poll_all(_utc(1), {"EUR_USD": (1.08548, 1.08553)})

        # Should still attempt fallback despite cancel failure
        mock_submit.assert_called_once()

    def test_expired_but_actually_filled(self, logger):
        """check_trade_by_order finds the trade after order disappeared."""
        executor = Executor("http://fake", "token", "acc-1", logger)
        from src.scalp_mode.execution.order_builder import OrderBuilder
        builder = OrderBuilder({
            "limit_ttl_seconds": 180, "fallback_market": True,
            "fallback_max_atr_distance": 0.3, "price_bound_slippage": 0.2})
        trade_mgr = TradeManager(
            {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
             "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
            "http://fake", "token", "acc-1", logger)

        mgr = PendingOrderManager(executor, builder, trade_mgr, logger,
                                   poll_interval_sec=0)
        po = PendingOrder(
            order_id="ord-ghost", signal_id="sig-g",
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            submitted_at=_utc(0), ttl_seconds=180,
            atr=0.0005, spread_at_signal=0.5, max_spread=0.8)
        mgr.track(po)

        with patch.object(executor, "check_order_status") as mock_status:
            mock_status.return_value = ExecutionResult(
                success=False, order_id="ord-ghost",
                broker_status="expired", reject_reason="order_not_found")
            with patch.object(executor, "check_trade_by_order") as mock_check:
                mock_check.return_value = {
                    "id": "trade-ghost", "price": "1.08555",
                    "instrument": "EUR_USD"}
                mgr.poll_all(_utc(1), {})

        assert len(trade_mgr.open_trades) == 1
        assert trade_mgr.open_trades[0].trade_id == "trade-ghost"
