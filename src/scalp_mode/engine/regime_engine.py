"""Regime Engine — Determines market regime on M5.

Per spec 3.2, regimes are:
  Trend_Up:   close > EMA50 AND EMA20 > EMA50 AND EMA_slope > threshold AND RSI in [52, 78]
  Trend_Down: close < EMA50 AND EMA20 < EMA50 AND EMA_slope < -threshold AND RSI in [22, 48]
  Range:      |EMA_slope| <= threshold AND BB_width < bb_width_thr AND close inside BB
  NoTrade:    none of the above → pause 5 minutes

Per spec 0.5, borderline detection is logged but does not block in V1.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .feature_engine import IndicatorSet


class Regime(str, Enum):
    TREND_UP = "Trend_Up"
    TREND_DOWN = "Trend_Down"
    RANGE = "Range"
    NO_TRADE = "NoTrade"


@dataclass
class RegimeResult:
    regime: Regime
    values: dict
    is_borderline: bool = False
    borderline_flags: list[str] | None = None


class RegimeEngine:
    """Evaluates M5 regime from indicators.

    Usage:
        engine = RegimeEngine(config.regime, config.borderline)
        result = engine.evaluate(indicators_m5, close_m5)
    """

    def __init__(self, regime_config: dict, borderline_config: dict | None = None):
        # Trend thresholds
        self._ema_slope_thr = regime_config["trend"]["ema_slope_thr"]
        self._rsi_min = regime_config["trend"]["rsi_min"]
        self._rsi_max = regime_config["trend"]["rsi_max"]

        # Range thresholds
        self._bb_width_thr = regime_config["range"]["bb_width_thr"]

        # Borderline thresholds (for logging in V1)
        if borderline_config:
            self._bl_slope_low = borderline_config.get("ema_slope_low", 0.15)
            self._bl_slope_high = borderline_config.get("ema_slope_high", 0.25)
            self._bl_spread_warn = borderline_config.get("spread_warn_ratio", 0.70)
        else:
            self._bl_slope_low = 0.15
            self._bl_slope_high = 0.25

    def evaluate(self, ind: IndicatorSet, close: float) -> RegimeResult:
        """Determine the current M5 regime.

        Args:
            ind: IndicatorSet from FeatureEngine.compute(df_m5, "M5")
            close: Latest M5 close price

        Returns:
            RegimeResult with regime classification and diagnostic values.
        """
        values = {
            "ema_slope": ind.ema_slope,
            "bb_width": ind.bb_width,
            "rsi_m5": ind.rsi14,
            "close": close,
            "ema20": ind.ema20,
            "ema50": ind.ema50,
            "bb_upper": ind.bb_upper,
            "bb_lower": ind.bb_lower,
        }

        # Guard: if any critical indicator is None, regime is NoTrade
        if any(v is None for v in [ind.ema_slope, ind.bb_width, ind.rsi14,
                                    ind.ema20, ind.ema50, ind.bb_upper, ind.bb_lower]):
            return RegimeResult(
                regime=Regime.NO_TRADE,
                values=values,
            )

        # Borderline detection (B1: regime on the edge)
        borderline_flags = []
        abs_slope = abs(ind.ema_slope)
        if self._bl_slope_low <= abs_slope <= self._bl_slope_high:
            borderline_flags.append("B1")

        # --- Trend Up ---
        if (close > ind.ema50
                and ind.ema20 > ind.ema50
                and ind.ema_slope > self._ema_slope_thr
                and self._rsi_min <= ind.rsi14 <= self._rsi_max):
            return RegimeResult(
                regime=Regime.TREND_UP,
                values=values,
                is_borderline=len(borderline_flags) > 0,
                borderline_flags=borderline_flags or None,
            )

        # --- Trend Down ---
        rsi_min_down = 100 - self._rsi_max  # 22
        rsi_max_down = 100 - self._rsi_min  # 48
        if (close < ind.ema50
                and ind.ema20 < ind.ema50
                and ind.ema_slope < -self._ema_slope_thr
                and rsi_min_down <= ind.rsi14 <= rsi_max_down):
            return RegimeResult(
                regime=Regime.TREND_DOWN,
                values=values,
                is_borderline=len(borderline_flags) > 0,
                borderline_flags=borderline_flags or None,
            )

        # --- Range ---
        if (abs_slope <= self._ema_slope_thr
                and ind.bb_width < self._bb_width_thr
                and ind.bb_lower <= close <= ind.bb_upper):
            return RegimeResult(
                regime=Regime.RANGE,
                values=values,
                is_borderline=len(borderline_flags) > 0,
                borderline_flags=borderline_flags or None,
            )

        # --- NoTrade (fallback) ---
        return RegimeResult(
            regime=Regime.NO_TRADE,
            values=values,
            is_borderline=len(borderline_flags) > 0,
            borderline_flags=borderline_flags or None,
        )
