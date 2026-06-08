"""Feature Engine — Computes all technical indicators for Scalp Mode V1.

Indicators (from spec 3.1):
  M5 context:  EMA20, EMA50, ATR14, RSI14, BB(20,2), EMA_slope, BB_width
  M1 trigger:  EMA20, EMA50, ATR14, RSI14, BB(20,2), MACD(12,26,9) histogram

All indicators are computed on official candles (REST, per spec A.5).
NaN detection is exposed for the Data Quality Gate.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class IndicatorSet:
    """Complete set of indicators for one timeframe at the latest bar."""
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    atr14: Optional[float] = None
    rsi14: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_mid: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    ema_slope: Optional[float] = None
    # M1-only
    macd_hist: Optional[float] = None
    macd_hist_prev: Optional[float] = None
    macd_hist_prev2: Optional[float] = None

    def has_nan(self) -> tuple[bool, Optional[str]]:
        """Check if any core indicator is NaN/None.

        Returns (has_nan, first_nan_field_name).
        """
        core_fields = [
            ("ema20", self.ema20),
            ("ema50", self.ema50),
            ("atr14", self.atr14),
            ("rsi14", self.rsi14),
            ("bb_upper", self.bb_upper),
            ("bb_mid", self.bb_mid),
            ("bb_lower", self.bb_lower),
        ]
        for name, val in core_fields:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return True, name
        return False, None


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger_bands(close: pd.Series, period: int = 20,
                     std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (mid, upper, lower)."""
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return mid, upper, lower


def _macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26,
                    signal: int = 9) -> pd.Series:
    """MACD Histogram = MACD line - Signal line."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def _ema_slope(ema_series: pd.Series, atr_series: pd.Series) -> pd.Series:
    """EMA slope normalized by ATR: (EMA[0] - EMA[1]) / ATR[0].

    Positive = upward trend, negative = downward.
    Per spec: threshold is 0.20 for trend detection.
    """
    ema_diff = ema_series.diff()
    # Avoid division by zero
    safe_atr = atr_series.replace(0, np.nan)
    return ema_diff / safe_atr


class FeatureEngine:
    """Computes technical indicators from candle DataFrames.

    Usage:
        engine = FeatureEngine()
        indicators_m5 = engine.compute(df_m5, timeframe="M5")
        indicators_m1 = engine.compute(df_m1, timeframe="M1")
    """

    def compute(self, df: pd.DataFrame, timeframe: str = "M1") -> IndicatorSet:
        """Compute all indicators on a candle DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume.
                Must have enough rows for warmup (at least 50 for EMA50).
            timeframe: "M1" or "M5" — M1 includes MACD histogram.

        Returns:
            IndicatorSet with latest values for all indicators.
        """
        if len(df) < 2:
            return IndicatorSet()

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # Core indicators
        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        atr14 = _atr(high, low, close, 14)
        rsi14 = _rsi(close, 14)
        bb_mid, bb_upper, bb_lower = _bollinger_bands(close, 20, 2.0)

        # Derived
        bb_w = (bb_upper - bb_lower) / bb_mid
        slope = _ema_slope(ema20, atr14)

        result = IndicatorSet(
            ema20=_last_val(ema20),
            ema50=_last_val(ema50),
            atr14=_last_val(atr14),
            rsi14=_last_val(rsi14),
            bb_upper=_last_val(bb_upper),
            bb_mid=_last_val(bb_mid),
            bb_lower=_last_val(bb_lower),
            bb_width=_last_val(bb_w),
            ema_slope=_last_val(slope),
        )

        # MACD only for M1 (trigger timeframe)
        if timeframe == "M1":
            macd_hist = _macd_histogram(close)
            result.macd_hist = _last_val(macd_hist)
            result.macd_hist_prev = _last_val(macd_hist, offset=1)
            result.macd_hist_prev2 = _last_val(macd_hist, offset=2)

        return result

    def compute_series(self, df: pd.DataFrame, timeframe: str = "M1") -> dict[str, pd.Series]:
        """Compute all indicators and return full series (for backtesting).

        Returns dict of indicator name → pd.Series.
        """
        if len(df) < 2:
            return {}

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        atr14 = _atr(high, low, close, 14)
        rsi14 = _rsi(close, 14)
        bb_mid, bb_upper, bb_lower = _bollinger_bands(close, 20, 2.0)

        result = {
            "ema20": ema20,
            "ema50": ema50,
            "atr14": atr14,
            "rsi14": rsi14,
            "bb_upper": bb_upper,
            "bb_mid": bb_mid,
            "bb_lower": bb_lower,
            "bb_width": (bb_upper - bb_lower) / bb_mid,
            "ema_slope": _ema_slope(ema20, atr14),
        }

        if timeframe == "M1":
            result["macd_hist"] = _macd_histogram(close)

        return result


def _last_val(series: pd.Series, offset: int = 0) -> Optional[float]:
    """Extract the last value (or last-N) from a series, None if NaN."""
    idx = -(1 + offset)
    if len(series) < abs(idx):
        return None
    val = series.iloc[idx]
    if pd.isna(val):
        return None
    return float(val)
