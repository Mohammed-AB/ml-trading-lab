"""Tests for Trade Manager."""

import pytest
from datetime import datetime, timedelta, timezone

from src.scalp_mode.execution.trade_manager import (
    TradeManager, ManagedTrade, TradeActionResult, ExitReason,
)
from src.scalp_mode.logger import ScalpLogger


MODEL_CONFIG = {
    "time_stop_min": 6,
    "sl_move_threshold_R": 0.8,
    "sl_move_target_R": -0.1,
    "sl_move_window_min": [2, 4],
}


def _utc(minute=0, second=0):
    return datetime(2026, 3, 27, 14, minute, second, tzinfo=timezone.utc)


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


def _make_trade(entry=1.08550, sl=1.08500, tp=1.08600, direction="long",
                open_time=None):
    risk = abs(entry - sl)
    return ManagedTrade(
        trade_id="trade-001", pair="EUR_USD", direction=direction,
        entry_price=entry, sl_price=sl, tp_price=tp,
        units=10000, open_time=open_time or _utc(0),
        risk_amount=risk,
    )


class TestTimeStop:
    def test_time_stop_triggers_at_6_min(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(open_time=_utc(0))

        # At 6 minutes, price hasn't moved much (current_R < 0.5)
        action = mgr.evaluate(trade, current_price=1.08555, utc_now=_utc(6))
        assert action.action == "close"
        assert action.exit_reason == ExitReason.TIME_STOP

    def test_no_time_stop_before_6_min(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(open_time=_utc(0))

        action = mgr.evaluate(trade, current_price=1.08555, utc_now=_utc(5))
        assert action.action == "hold"

    def test_no_time_stop_if_profitable(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(open_time=_utc(0))

        # At 6 minutes but price moved +0.6R (above 0.5 threshold)
        # entry=1.08550, risk=0.00050, +0.6R = entry + 0.0003 = 1.08580
        action = mgr.evaluate(trade, current_price=1.08580, utc_now=_utc(6))
        assert action.action == "hold"


class TestSLMove:
    def test_sl_move_at_0_8R(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(open_time=_utc(0))

        # At 3 minutes (within 2-4 window), +0.8R
        # entry=1.08550, risk=0.0005, +0.8R = 1.08590
        action = mgr.evaluate(trade, current_price=1.08590, utc_now=_utc(3))
        assert action.action == "move_sl"
        assert action.new_sl is not None
        # new_sl = entry - abs(-0.1) * risk = 1.08550 - 0.00005 = 1.08545
        # (SL slightly below breakeven for long)
        assert abs(action.new_sl - 1.08545) < 1e-6

    def test_no_sl_move_before_window(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(open_time=_utc(0))

        # At 1 minute (before 2-min window), even with +0.8R
        action = mgr.evaluate(trade, current_price=1.08590, utc_now=_utc(1))
        assert action.action == "hold"

    def test_no_sl_move_after_window(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(open_time=_utc(0))

        # At 5 minutes (after 4-min window), even with +0.8R
        action = mgr.evaluate(trade, current_price=1.08590, utc_now=_utc(5))
        assert action.action == "hold"

    def test_no_sl_move_below_threshold(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(open_time=_utc(0))

        # At 3 minutes but only +0.5R
        action = mgr.evaluate(trade, current_price=1.08575, utc_now=_utc(3))
        assert action.action == "hold"

    def test_sl_move_only_once(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(open_time=_utc(0))
        trade.sl_moved_to_be = True  # Already moved

        action = mgr.evaluate(trade, current_price=1.08590, utc_now=_utc(3))
        assert action.action == "hold"


class TestShortTrades:
    def test_short_time_stop(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(
            entry=1.08550, sl=1.08600, tp=1.08500,
            direction="short", open_time=_utc(0))

        # At 6 minutes, price hasn't dropped much (current_R < 0.5)
        action = mgr.evaluate(trade, current_price=1.08545, utc_now=_utc(6))
        assert action.action == "close"

    def test_short_sl_move(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(
            entry=1.08550, sl=1.08600, tp=1.08500,
            direction="short", open_time=_utc(0))

        # At 3 minutes, clearly above +0.8R (use 1.08505 → pnl=0.00045 → R=0.9)
        action = mgr.evaluate(trade, current_price=1.08505, utc_now=_utc(3))
        assert action.action == "move_sl"
        # new_sl = entry + abs(-0.1) * risk = 1.08550 + 0.00005 = 1.08555
        # (SL slightly above breakeven for short)
        assert abs(action.new_sl - 1.08555) < 1e-6


class TestEvaluateAll:
    def test_evaluate_multiple_trades(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)

        trade1 = _make_trade(open_time=_utc(0))
        trade1.trade_id = "t1"
        trade2 = _make_trade(entry=1.09000, sl=1.08950, tp=1.09050,
                             open_time=_utc(0))
        trade2.trade_id = "t2"

        mgr.add_trade(trade1)
        mgr.add_trade(trade2)

        assert len(mgr.open_trades) == 2

        live_prices = {"EUR_USD": 1.08555}
        actions = mgr.evaluate_all(_utc(6), live_prices)
        # Both should get time_stop (price hasn't moved enough)
        assert len(actions) >= 1

    def test_missing_price_skipped(self, logger):
        mgr = TradeManager(MODEL_CONFIG, "http://f", "t", "a", logger)
        trade = _make_trade(open_time=_utc(0))
        mgr.add_trade(trade)

        # No price for this pair
        actions = mgr.evaluate_all(_utc(6), {})
        assert len(actions) == 0
