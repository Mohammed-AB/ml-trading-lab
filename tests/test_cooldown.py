"""Tests for Cooldown Manager."""

import pytest
from datetime import datetime, timedelta, timezone
from src.scalp_mode.engine.cooldown import CooldownManager, TradeRecord, CooldownResult


RISK_CONFIG = {
    "cooldown_same_pair_dir_min": 10,
    "consec_loss_circuit": 3,
    "cooldown_minutes": 60,
    "trades_per_hour_pair": 3,
    "trades_per_hour_total": 6,
    "daily_loss": 0.01,
}


def _utc(hour, minute=0):
    return datetime(2026, 3, 27, hour, minute, tzinfo=timezone.utc)


class TestCooldownBasic:
    def test_no_trades_is_ok(self):
        mgr = CooldownManager(RISK_CONFIG)
        result = mgr.check("EUR_USD", "long", _utc(14))
        assert result.is_ok is True

    def test_same_pair_dir_cooldown(self):
        mgr = CooldownManager(RISK_CONFIG)
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(14, 0), pnl_pct=0.001))
        # 5 minutes later — should be blocked (< 10 min)
        result = mgr.check("EUR_USD", "long", _utc(14, 5))
        assert result.is_ok is False
        assert result.reason == "cooldown_active"

    def test_same_pair_different_dir_ok(self):
        mgr = CooldownManager(RISK_CONFIG)
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(14, 0), pnl_pct=0.001))
        result = mgr.check("EUR_USD", "short", _utc(14, 5))
        assert result.is_ok is True

    def test_different_pair_ok(self):
        mgr = CooldownManager(RISK_CONFIG)
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(14, 0), pnl_pct=0.001))
        result = mgr.check("GBP_USD", "long", _utc(14, 5))
        assert result.is_ok is True

    def test_cooldown_expires(self):
        mgr = CooldownManager(RISK_CONFIG)
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(14, 0), pnl_pct=0.001))
        # 11 minutes later — should be ok (> 10 min)
        result = mgr.check("EUR_USD", "long", _utc(14, 11))
        assert result.is_ok is True


class TestConsecutiveLosses:
    def test_3_consecutive_losses_circuit(self):
        mgr = CooldownManager(RISK_CONFIG)
        for i in range(3):
            mgr.record_trade(TradeRecord(
                "EUR_USD", "long", _utc(14, i * 2), pnl_pct=-0.002))

        # Immediately after — circuit breaker active
        result = mgr.check("EUR_USD", "long", _utc(14, 10))
        assert result.is_ok is False
        assert result.reason == "consec_loss_circuit"

    def test_circuit_breaker_expires(self):
        mgr = CooldownManager(RISK_CONFIG)
        for i in range(3):
            mgr.record_trade(TradeRecord(
                "EUR_USD", "long", _utc(13, i * 2), pnl_pct=-0.002))

        # 61 minutes after last loss — circuit breaker expired
        result = mgr.check("GBP_USD", "long", _utc(14, 5))
        assert result.is_ok is True

    def test_2_losses_no_circuit(self):
        mgr = CooldownManager(RISK_CONFIG)
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(14, 0), pnl_pct=-0.002))
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(14, 2), pnl_pct=-0.002))
        result = mgr.check("GBP_USD", "long", _utc(14, 10))
        assert result.is_ok is True

    def test_win_breaks_streak(self):
        mgr = CooldownManager(RISK_CONFIG)
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(14, 0), pnl_pct=-0.002))
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(14, 2), pnl_pct=-0.002))
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(14, 4), pnl_pct=0.001))  # Win
        result = mgr.check("GBP_USD", "long", _utc(14, 10))
        assert result.is_ok is True


class TestTradesPerHour:
    def test_max_trades_per_hour_pair(self):
        mgr = CooldownManager(RISK_CONFIG)
        for i in range(3):
            mgr.record_trade(TradeRecord(
                "EUR_USD", "long", _utc(14, i * 12), pnl_pct=0.001))

        result = mgr.check("EUR_USD", "long", _utc(14, 50))
        assert result.is_ok is False
        assert result.reason == "max_trades_per_hour_pair"

    def test_max_trades_per_hour_total(self):
        mgr = CooldownManager(RISK_CONFIG)
        pairs = ["EUR_USD", "GBP_USD", "USD_JPY"]
        for i in range(6):
            mgr.record_trade(TradeRecord(
                pairs[i % 3], "long", _utc(14, i * 8), pnl_pct=0.001))

        result = mgr.check("EUR_USD", "long", _utc(14, 55))
        assert result.is_ok is False
        assert result.reason == "max_trades_per_hour_total"

    def test_old_trades_dont_count(self):
        mgr = CooldownManager(RISK_CONFIG)
        # 3 trades 2 hours ago
        for i in range(3):
            mgr.record_trade(TradeRecord(
                "EUR_USD", "long", _utc(12, i * 12), pnl_pct=0.001))
        result = mgr.check("EUR_USD", "long", _utc(14, 0))
        assert result.is_ok is True


class TestDailyLoss:
    def test_daily_loss_limit(self):
        mgr = CooldownManager(RISK_CONFIG)
        # Accumulate -1.0% loss using alternating wins/losses to avoid
        # triggering the consecutive loss circuit breaker (3 in a row)
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(12, 0), pnl_pct=-0.004))
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(12, 10), pnl_pct=-0.004))
        mgr.record_trade(TradeRecord("GBP_USD", "long", _utc(12, 20), pnl_pct=0.001))
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(12, 30), pnl_pct=-0.004))
        # Total: -0.004 -0.004 +0.001 -0.004 = -0.011 → over -1.0% limit

        result = mgr.check("GBP_USD", "short", _utc(14))
        assert result.is_ok is False
        assert result.reason == "daily_loss_limit"

    def test_within_daily_limit(self):
        mgr = CooldownManager(RISK_CONFIG)
        mgr.record_trade(TradeRecord("EUR_USD", "long", _utc(13), pnl_pct=-0.005))
        result = mgr.check("EUR_USD", "long", _utc(14))
        assert result.is_ok is True  # -0.5% < -1.0% limit
