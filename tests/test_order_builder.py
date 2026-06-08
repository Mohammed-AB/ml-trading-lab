"""Tests for Order Builder."""

import pytest
from datetime import datetime, timezone
from src.scalp_mode.execution.order_builder import OrderBuilder, OrderType


ORDER_CONFIG = {
    "limit_ttl_seconds": 180,
    "fallback_market": True,
    "fallback_max_atr_distance": 0.3,
    "fallback_cooldown_min": 2,
    "price_bound_slippage": 0.2,
}


def _utc(hour, minute=0):
    return datetime(2026, 3, 27, hour, minute, tzinfo=timezone.utc)


class TestLimitOrder:
    def test_long_limit_order(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_limit(
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            signal_id="test-uuid-1", utc_now=_utc(14, 0))

        assert order.order_type == OrderType.LIMIT
        assert order.units == 10000  # Positive for long
        assert order.price == 1.08550
        assert order.sl_price == 1.08500
        assert order.tp_price == 1.08600
        assert order.price_bound is None
        assert order.ttl_seconds == 180
        assert order.expire_time is not None

    def test_short_limit_order(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_limit(
            pair="EUR_USD", direction="short", units=10000,
            entry_price=1.08550, sl_price=1.08600, tp_price=1.08500,
            signal_id="test-uuid-2", utc_now=_utc(14, 0))

        assert order.units == -10000  # Negative for short
        assert order.direction == "short"

    def test_jpy_price_rounding(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_limit(
            pair="USD_JPY", direction="long", units=1000,
            entry_price=149.5067, sl_price=149.4567, tp_price=149.5567,
            signal_id="test-uuid-3", utc_now=_utc(14, 0))

        # JPY pairs should be rounded to 3 decimals
        assert order.price == 149.507
        assert order.sl_price == 149.457
        assert order.tp_price == 149.557

    def test_expire_time_format(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_limit(
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.085, sl_price=1.084, tp_price=1.086,
            signal_id="test-uuid-4", utc_now=_utc(14, 0))

        assert "2026-03-27T14:03:00" in order.expire_time  # 3 minutes later


class TestMarketFallback:
    def test_valid_fallback(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_market_fallback(
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            current_price=1.08555,  # Within 0.3 * ATR
            atr=0.0005,
            spread_pips=0.5, max_spread_pips=0.8,
            signal_id="test-uuid-5")

        assert order is not None
        assert order.order_type == OrderType.MARKET
        assert order.price is None
        assert order.price_bound is not None
        # priceBound = 1.08555 + 0.2 pips = 1.08555 + 0.00002 = 1.08557
        assert order.price_bound == 1.08557

    def test_spread_too_wide_no_fallback(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_market_fallback(
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            current_price=1.08555, atr=0.0005,
            spread_pips=1.0, max_spread_pips=0.8,
            signal_id="test-uuid-6")

        assert order is None

    def test_price_too_far_no_fallback(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_market_fallback(
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            current_price=1.08600,  # 0.5 pip > 0.3 * 0.0005 ATR distance
            atr=0.0005,
            spread_pips=0.5, max_spread_pips=0.8,
            signal_id="test-uuid-7")

        assert order is None

    def test_fallback_disabled(self):
        config = {**ORDER_CONFIG, "fallback_market": False}
        builder = OrderBuilder(config)
        order = builder.build_market_fallback(
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.085, sl_price=1.084, tp_price=1.086,
            current_price=1.08505, atr=0.0005,
            spread_pips=0.5, max_spread_pips=0.8,
            signal_id="test-uuid-8")

        assert order is None

    def test_short_price_bound(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_market_fallback(
            pair="EUR_USD", direction="short", units=10000,
            entry_price=1.08550, sl_price=1.08600, tp_price=1.08500,
            current_price=1.08548, atr=0.0005,
            spread_pips=0.5, max_spread_pips=0.8,
            signal_id="test-uuid-9")

        assert order is not None
        # priceBound = 1.08548 - 0.00002 = 1.08546
        assert order.price_bound == 1.08546


class TestOandaFormat:
    def test_limit_order_body(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_limit(
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.08550, sl_price=1.08500, tp_price=1.08600,
            signal_id="test-uuid-10", utc_now=_utc(14, 0))

        body = builder.to_oanda_order(order)
        assert body["order"]["type"] == "LIMIT"
        assert body["order"]["instrument"] == "EUR_USD"
        assert body["order"]["units"] == "10000"
        assert body["order"]["timeInForce"] == "GTD"
        assert "stopLossOnFill" in body["order"]
        assert "takeProfitOnFill" in body["order"]

    def test_market_order_body(self):
        builder = OrderBuilder(ORDER_CONFIG)
        order = builder.build_market_fallback(
            pair="EUR_USD", direction="long", units=10000,
            entry_price=1.085, sl_price=1.084, tp_price=1.086,
            current_price=1.08505, atr=0.0005,
            spread_pips=0.5, max_spread_pips=0.8,
            signal_id="test-uuid-11")

        body = builder.to_oanda_order(order)
        assert body["order"]["type"] == "MARKET"
        assert "timeInForce" not in body["order"]
        assert "priceBound" not in body["order"]
