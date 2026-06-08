"""Tests for Feature Engine."""

import numpy as np
import pandas as pd
import pytest

from src.scalp_mode.engine.feature_engine import FeatureEngine, IndicatorSet


def _make_candles(n: int = 100, base_price: float = 1.0850,
                  volatility: float = 0.0005, trend: float = 0.0) -> pd.DataFrame:
    """Generate synthetic candle data for testing."""
    np.random.seed(42)
    closes = [base_price]
    for i in range(1, n):
        change = np.random.normal(trend, volatility)
        closes.append(closes[-1] + change)
    closes = np.array(closes)

    highs = closes + np.abs(np.random.normal(0, volatility * 0.5, n))
    lows = closes - np.abs(np.random.normal(0, volatility * 0.5, n))
    opens = closes + np.random.normal(0, volatility * 0.3, n)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.random.randint(10, 100, n),
    })


class TestFeatureEngine:
    def setup_method(self):
        self.engine = FeatureEngine()

    def test_compute_returns_indicator_set(self):
        df = _make_candles(100)
        result = self.engine.compute(df, "M5")
        assert isinstance(result, IndicatorSet)

    def test_all_indicators_present_m5(self):
        df = _make_candles(100)
        result = self.engine.compute(df, "M5")
        assert result.ema20 is not None
        assert result.ema50 is not None
        assert result.atr14 is not None
        assert result.rsi14 is not None
        assert result.bb_upper is not None
        assert result.bb_mid is not None
        assert result.bb_lower is not None
        assert result.bb_width is not None
        assert result.ema_slope is not None
        # M5 should NOT have MACD
        assert result.macd_hist is None

    def test_all_indicators_present_m1(self):
        df = _make_candles(100)
        result = self.engine.compute(df, "M1")
        assert result.macd_hist is not None
        assert result.macd_hist_prev is not None

    def test_ema20_less_than_ema50_not_required(self):
        """EMAs can be in any order depending on trend."""
        df = _make_candles(100)
        result = self.engine.compute(df, "M5")
        # Just verify they're different numbers (not equal by coincidence)
        assert result.ema20 != result.ema50

    def test_rsi_bounded(self):
        df = _make_candles(100)
        result = self.engine.compute(df, "M5")
        assert 0 <= result.rsi14 <= 100

    def test_atr_positive(self):
        df = _make_candles(100)
        result = self.engine.compute(df, "M5")
        assert result.atr14 > 0

    def test_bb_ordering(self):
        """BB upper > mid > lower always."""
        df = _make_candles(100)
        result = self.engine.compute(df, "M5")
        assert result.bb_upper > result.bb_mid > result.bb_lower

    def test_bb_width_positive(self):
        df = _make_candles(100)
        result = self.engine.compute(df, "M5")
        assert result.bb_width > 0

    def test_uptrend_positive_slope(self):
        """Strong uptrend should produce positive EMA slope."""
        df = _make_candles(100, trend=0.001, volatility=0.0003)
        result = self.engine.compute(df, "M5")
        assert result.ema_slope > 0

    def test_downtrend_negative_slope(self):
        """Strong downtrend should produce negative EMA slope."""
        df = _make_candles(100, trend=-0.001, volatility=0.0003)
        result = self.engine.compute(df, "M5")
        assert result.ema_slope < 0

    def test_macd_hist_has_history(self):
        df = _make_candles(100)
        result = self.engine.compute(df, "M1")
        assert result.macd_hist is not None
        assert result.macd_hist_prev is not None
        assert result.macd_hist_prev2 is not None

    def test_too_few_candles_returns_empty(self):
        df = _make_candles(1)
        result = self.engine.compute(df, "M1")
        # With only 1 candle, most indicators will be None
        assert result.ema50 is None or result.atr14 is None

    def test_compute_series_returns_dict(self):
        df = _make_candles(100)
        result = self.engine.compute_series(df, "M1")
        assert isinstance(result, dict)
        assert "ema20" in result
        assert "ema50" in result
        assert "atr14" in result
        assert "rsi14" in result
        assert "macd_hist" in result
        assert len(result["ema20"]) == 100

    def test_compute_series_m5_no_macd(self):
        df = _make_candles(100)
        result = self.engine.compute_series(df, "M5")
        assert "macd_hist" not in result


class TestIndicatorSetNaN:
    def test_no_nan_when_valid(self):
        ind = IndicatorSet(
            ema20=1.085, ema50=1.084, atr14=0.0005,
            rsi14=55.0, bb_upper=1.087, bb_mid=1.085, bb_lower=1.083,
        )
        has_nan, field = ind.has_nan()
        assert has_nan is False
        assert field is None

    def test_detects_none(self):
        ind = IndicatorSet(ema20=1.085, ema50=None)
        has_nan, field = ind.has_nan()
        assert has_nan is True
        assert field == "ema50"

    def test_detects_nan_float(self):
        ind = IndicatorSet(
            ema20=1.085, ema50=1.084, atr14=float("nan"),
            rsi14=55.0, bb_upper=1.087, bb_mid=1.085, bb_lower=1.083,
        )
        has_nan, field = ind.has_nan()
        assert has_nan is True
        assert field == "atr14"

    def test_default_all_none(self):
        ind = IndicatorSet()
        has_nan, field = ind.has_nan()
        assert has_nan is True
        assert field == "ema20"
