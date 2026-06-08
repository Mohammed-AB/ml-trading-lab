"""Tests for Risk Manager."""

import pytest
from src.scalp_mode.execution.risk_manager import RiskManager, RiskResult, OpenPosition


RISK_CONFIG = {
    "risk_pct": 0.0025,
    "max_concurrent": 2,
    "max_margin_pct": 0.08,
}


class TestPositionSizing:
    def test_basic_sizing(self):
        rm = RiskManager(RISK_CONFIG)
        result = rm.evaluate(
            pair="EUR_USD", direction="long", stop_pips=4.0,
            nav=10000, margin_available=9500, open_positions=[])
        assert result.approved is True
        # risk_amount = 10000 * 0.0025 = 25
        # units = floor(25 / (4.0 * 0.0001)) = floor(62500) = 62500
        assert result.units == 62500

    def test_jpy_pair_sizing(self):
        rm = RiskManager(RISK_CONFIG)
        result = rm.evaluate(
            pair="USD_JPY", direction="long", stop_pips=4.0,
            nav=10000, margin_available=9500, open_positions=[])
        assert result.approved is True
        # units = floor(25 / (4.0 * 0.01)) = floor(625) = 625
        assert result.units == 625

    def test_zero_stop_rejected(self):
        rm = RiskManager(RISK_CONFIG)
        result = rm.evaluate(
            pair="EUR_USD", direction="long", stop_pips=0,
            nav=10000, margin_available=9500, open_positions=[])
        assert result.approved is False
        assert result.reject_reason == "invalid_stop"

    def test_negative_stop_rejected(self):
        rm = RiskManager(RISK_CONFIG)
        result = rm.evaluate(
            pair="EUR_USD", direction="long", stop_pips=-1.0,
            nav=10000, margin_available=9500, open_positions=[])
        assert result.approved is False


class TestMaxConcurrent:
    def test_under_limit(self):
        rm = RiskManager(RISK_CONFIG)
        positions = [
            OpenPosition("EUR_USD", "long", 1000, 50.0),
        ]
        # USD_JPY is not correlated with EUR_USD
        result = rm.evaluate(
            pair="USD_JPY", direction="long", stop_pips=5.0,
            nav=10000, margin_available=9000, open_positions=positions)
        assert result.approved is True

    def test_at_limit(self):
        rm = RiskManager(RISK_CONFIG)
        positions = [
            OpenPosition("EUR_USD", "long", 1000, 50.0),
            OpenPosition("USD_JPY", "short", 500, 30.0),
        ]
        result = rm.evaluate(
            pair="GBP_USD", direction="long", stop_pips=5.0,
            nav=10000, margin_available=9000, open_positions=positions)
        assert result.approved is False
        assert result.reject_reason == "max_concurrent_reached"


class TestCorrelationGuard:
    def test_eur_gbp_same_direction_blocked(self):
        rm = RiskManager(RISK_CONFIG)
        positions = [
            OpenPosition("EUR_USD", "long", 1000, 50.0),
        ]
        result = rm.evaluate(
            pair="GBP_USD", direction="long", stop_pips=5.0,
            nav=10000, margin_available=9000, open_positions=positions)
        assert result.approved is False
        assert result.reject_reason == "correlation_guard"

    def test_eur_gbp_opposite_direction_ok(self):
        rm = RiskManager(RISK_CONFIG)
        positions = [
            OpenPosition("EUR_USD", "long", 1000, 50.0),
        ]
        result = rm.evaluate(
            pair="GBP_USD", direction="short", stop_pips=5.0,
            nav=10000, margin_available=9000, open_positions=positions)
        assert result.approved is True

    def test_unrelated_pairs_ok(self):
        rm = RiskManager(RISK_CONFIG)
        positions = [
            OpenPosition("EUR_USD", "long", 1000, 50.0),
        ]
        result = rm.evaluate(
            pair="USD_JPY", direction="long", stop_pips=5.0,
            nav=10000, margin_available=9000, open_positions=positions)
        assert result.approved is True


class TestDuplicatePosition:
    def test_duplicate_blocked(self):
        rm = RiskManager(RISK_CONFIG)
        positions = [
            OpenPosition("EUR_USD", "long", 1000, 50.0),
        ]
        result = rm.evaluate(
            pair="EUR_USD", direction="long", stop_pips=5.0,
            nav=10000, margin_available=9000, open_positions=positions)
        assert result.approved is False
        assert result.reject_reason == "duplicate_position"

    def test_same_pair_opposite_direction_ok(self):
        rm = RiskManager(RISK_CONFIG)
        positions = [
            OpenPosition("EUR_USD", "long", 1000, 50.0),
        ]
        result = rm.evaluate(
            pair="EUR_USD", direction="short", stop_pips=5.0,
            nav=10000, margin_available=9000, open_positions=positions)
        assert result.approved is True


class TestNotionalMarginCap:
    """Broker margin ≈ notional/lev; old pip*100 cap caused ~55k units on ~$556."""

    def test_small_account_eur_caps_below_old_55k(self):
        rm = RiskManager({
            "risk_pct": 0.07,
            "max_concurrent": 2,
            "max_margin_pct": 0.08,
            "leverage": 33,
            "margin_cap_safety": 0.35,
            "account_currency": "USD",
        })
        result = rm.evaluate(
            pair="EUR_USD", direction="long", stop_pips=4.0,
            nav=556.0, margin_available=556.0, open_positions=[],
            mid_price=1.08,
        )
        assert result.approved is True
        assert result.units < 8_000
        assert result.units > 3_000
        nu = 1.08
        margin_at_33 = result.units * nu / 33.0
        assert margin_at_33 <= 556.0 * 0.36

    def test_usd_jpy_uses_unit_notional(self):
        rm = RiskManager({
            "risk_pct": 0.07,
            "max_concurrent": 2,
            "max_margin_pct": 0.08,
            "leverage": 33,
            "margin_cap_safety": 0.35,
            "account_currency": "USD",
        })
        result = rm.evaluate(
            pair="USD_JPY", direction="long", stop_pips=4.0,
            nav=556.0, margin_available=556.0, open_positions=[],
            live_rates={"USD_JPY": 150.0},
            mid_price=150.0,
        )
        assert result.approved is True
        margin_at_33 = result.units * 1.0 / 33.0
        assert margin_at_33 <= 556.0 * 0.36
