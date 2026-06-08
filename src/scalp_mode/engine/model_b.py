"""Model B: Failed-breakout Reversal — Range regime only.

Per spec 3.4:
  Condition: Regime = Range only
  Range:     rolling 12 M5 candles, range_high = max(high), range_low = min(low)
  Sell:      wick_excess <= 0.25*ATR, close < range_high, wick_ratio >= 0.60,
             RSI >= 65 then < 60 within 2 bars
  Buy:       wick_excess <= 0.25*ATR, close > range_low, wick_ratio >= 0.60,
             RSI <= 35 then > 40 within 2 bars
  Entry:     Close of rejection candle
  SL:        Behind wick tip + buffer = max(2*spread, 0.15*ATR_M1)
  TP:        mid_range or bb_mid, whichever is closer to entry

Direction: short at failed top breakout, long at failed bottom breakout.
Uses same TriggerSignal/TriggerPhase as Model A for pipeline compatibility.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .model_a import TriggerSignal, TriggerPhase, Direction
from .feature_engine import IndicatorSet
from .regime_engine import Regime
from ..utils.pip_utils import price_to_pips, pips_to_price


class ModelBTrigger:
    """Failed-breakout Reversal — Range regime only.

    Usage:
        trigger_b = ModelBTrigger(config.model_b, config.borderline)
        signal = trigger_b.evaluate(df_m1, df_m5, ind_m1, ind_m5,
                                     Regime.RANGE, "EUR_USD", spread_pips=0.5)
    """

    def __init__(self, model_config: dict, borderline_config: dict | None = None):
        self._range_window = model_config.get("range_window_M5", 12)
        self._wick_ratio_min = model_config.get("wick_ratio_min", 0.60)
        self._wick_excess_atr = model_config.get("wick_excess_atr", 0.25)
        self._stop_spread_mult = model_config.get("stop_spread_buffer_mult", 2.0)
        self._stop_atr_buffer = model_config.get("stop_atr_buffer", 0.15)
        self._rsi_overbought = model_config.get("rsi_overbought", 65)
        self._rsi_reversal_down = model_config.get("rsi_reversal", 60)
        self._rsi_oversold = model_config.get("rsi_oversold", 35)
        self._rsi_reversal_up = model_config.get("rsi_reversal_up", 40)
        self._bl = borderline_config or {}

    def evaluate(self, df_m1: pd.DataFrame, df_m5: pd.DataFrame,
                 ind_m1: IndicatorSet, ind_m5: IndicatorSet,
                 regime: Regime, pair: str,
                 spread_pips: float = 0) -> TriggerSignal:
        """Evaluate Model B trigger on latest M1 candles with M5 range context.

        Args:
            df_m1: M1 candles (needs at least 3 recent bars for RSI reversal)
            df_m5: M5 candles (needs >= range_window rows)
            ind_m1: Latest M1 IndicatorSet
            ind_m5: Latest M5 IndicatorSet
            regime: Must be RANGE
            pair: For pip normalization
            spread_pips: Current spread for SL buffer calculation

        Returns:
            TriggerSignal with phase and entry/SL/TP if valid.
        """
        # --- Only Range regime ---
        if regime != Regime.RANGE:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "regime_not_range"})

        if ind_m1.atr14 is None or ind_m1.atr14 <= 0:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "atr_invalid"})

        # --- Build range from M5 ---
        if len(df_m5) < self._range_window:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "insufficient_m5",
                                         "rows": len(df_m5),
                                         "needed": self._range_window})

        range_slice = df_m5.iloc[-(self._range_window + 1):-1]  # Exclude current M5 candle
        range_high = float(range_slice["high"].max())
        range_low = float(range_slice["low"].min())
        range_size = range_high - range_low

        # No clear range if too narrow
        if range_size < ind_m1.atr14 * 0.5:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "range_too_narrow",
                                         "range_size": range_size,
                                         "atr": ind_m1.atr14})

        # Need at least 3 M1 bars for RSI reversal check
        if len(df_m1) < 3:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "insufficient_m1"})

        atr = ind_m1.atr14
        bar = df_m1.iloc[-1]  # Latest completed M1 candle
        bar_prev = df_m1.iloc[-2]

        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        candle_range = h - l

        if candle_range <= 0:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "zero_range_candle"})

        # Compute RSI from the last 3 bars (current + 2 previous)
        rsi_current = ind_m1.rsi14
        # We need RSI of previous bars. Since IndicatorSet only has current RSI,
        # compute from the close series for the reversal check.
        closes = df_m1["close"].astype(float)
        rsi_prev = self._rsi_at(closes, -2)
        rsi_prev2 = self._rsi_at(closes, -3) if len(df_m1) >= 4 else None

        if rsi_current is None or rsi_prev is None:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "rsi_unavailable"})

        borderline_flags = []
        mid_range = (range_high + range_low) / 2

        # --- Check for failed TOP breakout (sell signal) ---
        upper_wick = h - max(o, c)  # Upper wick = high - max(open, close)
        wick_excess_top = h - range_high  # How far above range_high
        wick_ratio_top = upper_wick / candle_range

        top_signal = self._check_top_breakout_failure(
            h, c, range_high, upper_wick, wick_excess_top, wick_ratio_top,
            candle_range, atr, rsi_current, rsi_prev, rsi_prev2)

        if top_signal:
            entry_price = c  # Enter at close of rejection candle
            direction = Direction.SHORT
            sl_price, buffer = self._compute_sl(h, atr, spread_pips, direction, pair)
            tp_price = self._compute_tp(entry_price, mid_range, ind_m1.bb_mid,
                                         direction)
            risk = abs(sl_price - entry_price)
            risk_pips = price_to_pips(risk, pair)

            if risk <= 0:
                return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                     values={"reason": "zero_risk"})

            # Borderline: wick_ratio near minimum
            if self._wick_ratio_min <= wick_ratio_top <= self._wick_ratio_min + 0.05:
                borderline_flags.append("B2")
            # Borderline: RSI just barely reversed
            if abs(rsi_current - self._rsi_reversal_down) < 3:
                borderline_flags.append("B3")

            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                risk_pips=risk_pips,
                breakout_level=range_high,
                borderline_flags=borderline_flags or None,
                values={
                    "model": "B",
                    "wick_ratio": round(wick_ratio_top, 3),
                    "wick_excess": round(wick_excess_top, 6),
                    "rsi_current": round(rsi_current, 1),
                    "rsi_prev": round(rsi_prev, 1),
                    "range_high": range_high,
                    "range_low": range_low,
                    "mid_range": mid_range,
                    "buffer": round(buffer, 6),
                },
            )

        # --- Check for failed BOTTOM breakout (buy signal) ---
        lower_wick = min(o, c) - l  # Lower wick = min(open, close) - low
        wick_excess_bottom = range_low - l  # How far below range_low
        wick_ratio_bottom = lower_wick / candle_range

        bottom_signal = self._check_bottom_breakout_failure(
            l, c, range_low, lower_wick, wick_excess_bottom, wick_ratio_bottom,
            candle_range, atr, rsi_current, rsi_prev, rsi_prev2)

        if bottom_signal:
            entry_price = c
            direction = Direction.LONG
            sl_price, buffer = self._compute_sl(l, atr, spread_pips, direction, pair)
            tp_price = self._compute_tp(entry_price, mid_range, ind_m1.bb_mid,
                                         direction)
            risk = abs(entry_price - sl_price)
            risk_pips = price_to_pips(risk, pair)

            if risk <= 0:
                return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                     values={"reason": "zero_risk"})

            if self._wick_ratio_min <= wick_ratio_bottom <= self._wick_ratio_min + 0.05:
                borderline_flags.append("B2")
            if abs(rsi_current - self._rsi_reversal_up) < 3:
                borderline_flags.append("B3")

            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                risk_pips=risk_pips,
                breakout_level=range_low,
                borderline_flags=borderline_flags or None,
                values={
                    "model": "B",
                    "wick_ratio": round(wick_ratio_bottom, 3),
                    "wick_excess": round(wick_excess_bottom, 6),
                    "rsi_current": round(rsi_current, 1),
                    "rsi_prev": round(rsi_prev, 1),
                    "range_high": range_high,
                    "range_low": range_low,
                    "mid_range": mid_range,
                    "buffer": round(buffer, 6),
                },
            )

        # No failed breakout pattern found
        return TriggerSignal(
            phase=TriggerPhase.NO_BREAKOUT,
            values={
                "reason": "no_failed_breakout",
                "range_high": range_high,
                "range_low": range_low,
                "wick_ratio_top": round(wick_ratio_top, 3),
                "wick_ratio_bottom": round(wick_ratio_bottom, 3),
                "rsi": rsi_current,
            })

    def _check_top_breakout_failure(
            self, h, c, range_high, upper_wick, wick_excess,
            wick_ratio, candle_range, atr,
            rsi_now, rsi_prev, rsi_prev2) -> bool:
        """Check if the candle shows a failed breakout at the top of the range."""
        # Must have poked above range_high
        if wick_excess <= 0:
            return False
        # Wick excess must be small (failed, not real breakout)
        if wick_excess > self._wick_excess_atr * atr:
            return False
        # Close must be back inside the range
        if c >= range_high:
            return False
        # Wick ratio must be large (strong rejection)
        if wick_ratio < self._wick_ratio_min:
            return False
        # RSI reversal: was overbought (>= 65) then dropped (< 60)
        # Check: previous bar had RSI >= overbought, current < reversal threshold
        if rsi_prev < self._rsi_overbought:
            # Also accept: 2 bars ago was overbought
            if rsi_prev2 is None or rsi_prev2 < self._rsi_overbought:
                return False
        if rsi_now >= self._rsi_reversal_down:
            return False
        return True

    def _check_bottom_breakout_failure(
            self, l, c, range_low, lower_wick, wick_excess,
            wick_ratio, candle_range, atr,
            rsi_now, rsi_prev, rsi_prev2) -> bool:
        """Check if the candle shows a failed breakout at the bottom of the range."""
        if wick_excess <= 0:
            return False
        if wick_excess > self._wick_excess_atr * atr:
            return False
        if c <= range_low:
            return False
        if wick_ratio < self._wick_ratio_min:
            return False
        # RSI reversal: was oversold (<= 35) then rose (> 40)
        if rsi_prev > self._rsi_oversold:
            if rsi_prev2 is None or rsi_prev2 > self._rsi_oversold:
                return False
        if rsi_now <= self._rsi_reversal_up:
            return False
        return True

    def _compute_sl(self, wick_tip: float, atr: float,
                    spread_pips: float, direction: Direction,
                    pair: str = "EUR_USD") -> tuple[float, float]:
        """Compute SL behind wick tip with buffer.

        buffer = max(2 * spread_in_price, 0.15 * ATR)
        """
        spread_buffer = pips_to_price(spread_pips * self._stop_spread_mult, pair)
        atr_buffer = self._stop_atr_buffer * atr
        buffer = max(spread_buffer, atr_buffer)

        if direction == Direction.SHORT:
            sl = wick_tip + buffer  # SL above the high
        else:
            sl = wick_tip - buffer  # SL below the low
        return sl, buffer

    def _compute_tp(self, entry: float, mid_range: float,
                    bb_mid: Optional[float], direction: Direction) -> float:
        """TP = mid_range or bb_mid, whichever is closer to entry."""
        if bb_mid is None:
            return mid_range

        dist_mid = abs(entry - mid_range)
        dist_bb = abs(entry - bb_mid)

        # Pick the closer target (smaller move = more conservative)
        tp = mid_range if dist_mid <= dist_bb else bb_mid

        # Sanity: TP must be in the right direction
        if direction == Direction.SHORT and tp >= entry:
            tp = mid_range  # Fallback
        if direction == Direction.LONG and tp <= entry:
            tp = mid_range

        return tp

    @staticmethod
    def _rsi_at(closes: pd.Series, offset: int) -> Optional[float]:
        """Approximate RSI at a given offset from end using Wilder's smoothing.

        offset=-2 means 2 bars from end (the previous bar).
        """
        period = 14
        idx = len(closes) + offset
        if idx < period + 1:
            return None

        subset = closes.iloc[:idx + 1]
        delta = subset.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period,
                            adjust=False).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period,
                            adjust=False).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))
