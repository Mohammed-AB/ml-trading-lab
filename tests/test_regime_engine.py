"""Tests for Regime Engine."""

import pytest
from src.scalp_mode.engine.feature_engine import IndicatorSet
from src.scalp_mode.engine.regime_engine import RegimeEngine, Regime


REGIME_CONFIG = {
    "trend": {"ema_slope_thr": 0.20, "rsi_min": 52, "rsi_max": 78},
    "range": {"bb_width_thr": 0.004},
}

BORDERLINE_CONFIG = {
    "ema_slope_low": 0.15,
    "ema_slope_high": 0.25,
}


def _make_engine():
    return RegimeEngine(REGIME_CONFIG, BORDERLINE_CONFIG)


def _trend_up_indicators(slope=0.30, rsi=60.0, close=1.090) -> IndicatorSet:
    """Indicators that satisfy Trend_Up conditions."""
    return IndicatorSet(
        ema20=1.089,       # EMA20 > EMA50
        ema50=1.087,
        atr14=0.0005,
        rsi14=rsi,
        bb_upper=1.092,
        bb_mid=1.088,
        bb_lower=1.084,
        bb_width=0.007,
        ema_slope=slope,
    )


def _trend_down_indicators(slope=-0.30, rsi=35.0, close=1.080) -> IndicatorSet:
    """Indicators that satisfy Trend_Down conditions."""
    return IndicatorSet(
        ema20=1.083,       # EMA20 < EMA50
        ema50=1.085,
        atr14=0.0005,
        rsi14=rsi,
        bb_upper=1.088,
        bb_mid=1.084,
        bb_lower=1.080,
        bb_width=0.007,
        ema_slope=slope,
    )


def _range_indicators(close=1.086) -> IndicatorSet:
    """Indicators that satisfy Range conditions."""
    return IndicatorSet(
        ema20=1.086,
        ema50=1.085,
        atr14=0.0005,
        rsi14=50.0,
        bb_upper=1.088,
        bb_mid=1.086,
        bb_lower=1.084,
        bb_width=0.003,     # < 0.004 threshold
        ema_slope=0.10,     # |slope| < 0.20
    )


class TestTrendUp:
    def test_basic_trend_up(self):
        engine = _make_engine()
        ind = _trend_up_indicators()
        result = engine.evaluate(ind, close=1.090)
        assert result.regime == Regime.TREND_UP

    def test_rsi_at_lower_bound(self):
        engine = _make_engine()
        ind = _trend_up_indicators(rsi=52.0)
        result = engine.evaluate(ind, close=1.090)
        assert result.regime == Regime.TREND_UP

    def test_rsi_at_upper_bound(self):
        engine = _make_engine()
        ind = _trend_up_indicators(rsi=78.0)
        result = engine.evaluate(ind, close=1.090)
        assert result.regime == Regime.TREND_UP

    def test_rsi_too_low_not_trend(self):
        engine = _make_engine()
        ind = _trend_up_indicators(rsi=50.0)
        result = engine.evaluate(ind, close=1.090)
        assert result.regime != Regime.TREND_UP

    def test_rsi_too_high_not_trend(self):
        engine = _make_engine()
        ind = _trend_up_indicators(rsi=80.0)
        result = engine.evaluate(ind, close=1.090)
        assert result.regime != Regime.TREND_UP

    def test_slope_at_threshold(self):
        """Slope exactly at 0.20 — should NOT pass (> required, not >=)."""
        engine = _make_engine()
        ind = _trend_up_indicators(slope=0.20)
        result = engine.evaluate(ind, close=1.090)
        assert result.regime != Regime.TREND_UP

    def test_close_below_ema50_not_trend_up(self):
        engine = _make_engine()
        ind = _trend_up_indicators()
        # close below EMA50 (1.087)
        result = engine.evaluate(ind, close=1.086)
        assert result.regime != Regime.TREND_UP


class TestTrendDown:
    def test_basic_trend_down(self):
        engine = _make_engine()
        ind = _trend_down_indicators()
        result = engine.evaluate(ind, close=1.080)
        assert result.regime == Regime.TREND_DOWN

    def test_rsi_at_bounds(self):
        engine = _make_engine()
        # RSI bounds for down: [22, 48]
        ind = _trend_down_indicators(rsi=22.0)
        result = engine.evaluate(ind, close=1.080)
        assert result.regime == Regime.TREND_DOWN

        ind = _trend_down_indicators(rsi=48.0)
        result = engine.evaluate(ind, close=1.080)
        assert result.regime == Regime.TREND_DOWN

    def test_close_above_ema50_not_trend_down(self):
        engine = _make_engine()
        ind = _trend_down_indicators()
        result = engine.evaluate(ind, close=1.086)  # Above EMA50=1.085
        assert result.regime != Regime.TREND_DOWN


class TestRange:
    def test_basic_range(self):
        engine = _make_engine()
        ind = _range_indicators()
        result = engine.evaluate(ind, close=1.086)
        assert result.regime == Regime.RANGE

    def test_close_inside_bb_required(self):
        engine = _make_engine()
        ind = _range_indicators()
        # Close outside BB upper (1.088)
        result = engine.evaluate(ind, close=1.090)
        assert result.regime != Regime.RANGE

    def test_bb_width_at_threshold_not_range(self):
        """bb_width must be < threshold, not <=."""
        engine = _make_engine()
        ind = _range_indicators()
        ind.bb_width = 0.004  # Exactly at threshold
        result = engine.evaluate(ind, close=1.086)
        assert result.regime != Regime.RANGE

    def test_slope_at_threshold_is_range(self):
        """slope |0.20| should be range (<=)."""
        engine = _make_engine()
        ind = _range_indicators()
        ind.ema_slope = 0.20
        result = engine.evaluate(ind, close=1.086)
        assert result.regime == Regime.RANGE


class TestNoTrade:
    def test_no_conditions_met(self):
        """When nothing matches, regime is NoTrade."""
        engine = _make_engine()
        ind = IndicatorSet(
            ema20=1.086, ema50=1.085, atr14=0.0005,
            rsi14=50.0,  # Outside trend RSI range
            bb_upper=1.088, bb_mid=1.086, bb_lower=1.084,
            bb_width=0.005,  # Too wide for range
            ema_slope=0.10,  # Too flat for trend
        )
        result = engine.evaluate(ind, close=1.086)
        assert result.regime == Regime.NO_TRADE

    def test_none_indicators_notrade(self):
        engine = _make_engine()
        ind = IndicatorSet()  # All None
        result = engine.evaluate(ind, close=1.086)
        assert result.regime == Regime.NO_TRADE

    def test_partial_none_notrade(self):
        engine = _make_engine()
        ind = IndicatorSet(ema20=1.086, ema50=1.085)
        result = engine.evaluate(ind, close=1.086)
        assert result.regime == Regime.NO_TRADE


class TestBorderline:
    def test_borderline_b1_slope_on_edge(self):
        """EMA_slope between 0.15 and 0.25 flags B1."""
        engine = _make_engine()
        ind = _trend_up_indicators(slope=0.22)
        result = engine.evaluate(ind, close=1.090)
        assert result.regime == Regime.TREND_UP
        assert result.is_borderline is True
        assert "B1" in result.borderline_flags

    def test_no_borderline_when_slope_clear(self):
        engine = _make_engine()
        ind = _trend_up_indicators(slope=0.40)
        result = engine.evaluate(ind, close=1.090)
        assert result.is_borderline is False

    def test_borderline_in_range_regime(self):
        engine = _make_engine()
        ind = _range_indicators()
        ind.ema_slope = 0.18  # Between 0.15 and 0.20 → B1
        result = engine.evaluate(ind, close=1.086)
        assert result.regime == Regime.RANGE
        assert result.is_borderline is True


class TestRegimeValues:
    def test_values_dict_populated(self):
        engine = _make_engine()
        ind = _trend_up_indicators()
        result = engine.evaluate(ind, close=1.090)
        assert "ema_slope" in result.values
        assert "bb_width" in result.values
        assert "rsi_m5" in result.values
        assert "close" in result.values
        assert result.values["close"] == 1.090
