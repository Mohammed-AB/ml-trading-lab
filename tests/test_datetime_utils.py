"""Tests for OANDA timestamp parsing utility."""

import pytest
from datetime import datetime, timezone
from src.scalp_mode.utils.datetime_utils import parse_oanda_timestamp


class TestParseOandaTimestamp:
    def test_nanoseconds_with_z(self):
        result = parse_oanda_timestamp("2026-01-07T14:00:01.123456789Z")
        assert result.hour == 14
        assert result.minute == 0
        assert result.second == 1
        assert result.tzinfo == timezone.utc

    def test_no_fractional_with_z(self):
        result = parse_oanda_timestamp("2026-01-07T14:00:01Z")
        assert result.hour == 14
        assert result.minute == 0
        assert result.second == 1
        assert result.tzinfo == timezone.utc

    def test_explicit_tz_no_fractional(self):
        result = parse_oanda_timestamp("2026-01-07T14:00:01+00:00")
        assert result.hour == 14
        assert result.second == 1
        assert result.tzinfo == timezone.utc

    def test_nanoseconds_with_explicit_tz(self):
        result = parse_oanda_timestamp("2026-01-07T14:00:01.123456789+00:00")
        assert result.hour == 14
        assert result.second == 1
        assert result.tzinfo == timezone.utc

    def test_all_results_utc(self):
        inputs = [
            "2026-01-07T14:00:01.123456789Z",
            "2026-01-07T14:00:01Z",
            "2026-01-07T14:00:01+00:00",
            "2026-01-07T14:00:01.123456789+00:00",
        ]
        for s in inputs:
            result = parse_oanda_timestamp(s)
            assert result.tzinfo == timezone.utc, f"Failed for: {s}"

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_oanda_timestamp(None)

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_oanda_timestamp("")

    def test_microseconds_with_z(self):
        result = parse_oanda_timestamp("2026-03-27T13:30:00.456789Z")
        assert result.hour == 13
        assert result.minute == 30

    def test_non_utc_timezone_converts(self):
        # -04:00 (EDT): 14:00 EDT = 18:00 UTC
        result = parse_oanda_timestamp("2026-01-07T14:00:01-04:00")
        assert result.tzinfo == timezone.utc
        assert result.hour == 18


class TestPendingFillUsesParseTimestamp:
    """Integration: pending fill carries correct open_time from fill_time."""

    def test_register_trade_uses_fill_time(self):
        from unittest.mock import MagicMock
        from src.scalp_mode.execution.pending_manager import (
            PendingOrderManager, PendingOrder,
        )
        from src.scalp_mode.execution.executor import Executor, ExecutionResult
        from src.scalp_mode.execution.order_builder import OrderBuilder
        from src.scalp_mode.execution.trade_manager import TradeManager
        from src.scalp_mode.logger import ScalpLogger
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            log_cfg = {
                "log_dir": tmp, "decision_log_file": "d.jsonl",
                "trade_log_file": "t.jsonl", "cycle_log_file": "c.jsonl",
                "system_log_file": "s.log", "max_file_size_mb": 1,
                "backup_count": 1, "level": "WARNING",
            }
            logger = ScalpLogger(log_cfg)
            executor = Executor("http://fake", "tok", "acc", logger)
            builder = OrderBuilder({
                "limit_ttl_seconds": 180, "fallback_market": True,
                "fallback_max_atr_distance": 0.3, "price_bound_slippage": 0.2,
            })
            trade_mgr = TradeManager(
                {"time_stop_min": 6, "sl_move_threshold_R": 0.8,
                 "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
                "http://fake", "tok", "acc", logger)

            mgr = PendingOrderManager(executor, builder, trade_mgr, logger,
                                       poll_interval_sec=0)

            # Simulate a fill result with a specific fill_time
            fill_result = ExecutionResult(
                success=True, trade_id="t-fill",
                fill_price=1.08555, broker_status="filled",
                fill_time="2026-01-07T14:02:30.500000000Z")

            po = PendingOrder(
                order_id="ord-ft", signal_id="sig-ft",
                pair="EUR_USD", direction="long", units=10000,
                entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
                submitted_at=datetime(2026, 1, 7, 14, 0, tzinfo=timezone.utc),
                ttl_seconds=180, atr=0.0005,
                spread_at_signal=0.5, max_spread=0.8)

            mgr._register_trade(po, fill_result)

            assert len(trade_mgr.open_trades) == 1
            t = trade_mgr.open_trades[0]
            # open_time should be from fill_time, not datetime.now()
            assert t.open_time.hour == 14
            assert t.open_time.minute == 2
            assert t.open_time.second == 30
            assert t.open_time.tzinfo == timezone.utc

            logger.close()
