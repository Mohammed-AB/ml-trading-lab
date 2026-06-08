"""Model A: Micro-breakout + Retest Trigger — V1 only model.

Per spec 3.3 and table 12:
  1. Compression: N=8 candles M1, range_N <= 1.2 * ATR_M1
  2. Breakout:    close > highest_high_N + 0.10 * ATR_M1, body_ratio >= 0.55
  3. Momentum:    RSI_M1 >= 55 (long) / <= 45 (short), MACD_hist increasing 2 bars
  4. Retest:      within 3 M1 candles, price returns to breakout_level +/- 0.15 * ATR_M1
  5. SL:          min(low_retest - buffer, breakout_level - 0.8 * ATR_M1)
  6. TP:          1.0R (TP1), 1.5R optional (TP2)

Direction follows regime: Trend_Up → long only, Trend_Down → short only.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .feature_engine import IndicatorSet
from .regime_engine import Regime


class TriggerPhase(str, Enum):
    NO_COMPRESSION = "no_compression"
    NO_BREAKOUT = "no_breakout"
    NO_MOMENTUM = "no_momentum"
    WAITING_RETEST = "waiting_retest"
    RETEST_TIMEOUT = "retest_timeout"
    VALID = "valid"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TriggerSignal:
    """Output of Model A trigger evaluation."""
    phase: TriggerPhase
    direction: Optional[Direction] = None
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    risk_pips: Optional[float] = None
    breakout_level: Optional[float] = None
    # Diagnostic values for logging
    values: Optional[dict] = None
    # Borderline flags (B2-B5)
    borderline_flags: Optional[list[str]] = None


class ModelATrigger:
    """Evaluates Model A: Micro-breakout + Retest on M1 candles.

    Usage:
        trigger = ModelATrigger(config.model_a, config.borderline)
        signal = trigger.evaluate(df_m1, indicators_m1, regime)
    """

    def __init__(self, model_config: dict, borderline_config: dict | None = None):
        self._N = model_config["compression_N"]
        self._compression_atr_mult = model_config.get("compression_atr_mult", 1.2)
        self._breakout_buffer_atr = model_config["breakout_buffer_atr"]
        self._retest_timeout = model_config["retest_timeout"]
        self._retest_tolerance_atr = model_config["retest_tolerance_atr"]
        self._body_ratio_min = model_config["body_ratio_min"]
        self._rsi_min_long = model_config["rsi_min_long"]
        self._sl_atr = model_config["sl_atr"]
        self._tp_R = model_config["tp_R"]

        # Borderline thresholds
        self._bl = borderline_config or {}

    def evaluate(self, df_m1: pd.DataFrame, ind: IndicatorSet,
                 regime: Regime, pair: str = "EUR_USD") -> TriggerSignal:
        """Evaluate Model A trigger on the latest M1 candles.

        Args:
            df_m1: M1 candle DataFrame (needs at least N + retest_timeout + 1 rows)
            ind: Latest M1 IndicatorSet
            regime: Current M5 regime (determines direction)

        Returns:
            TriggerSignal with phase and entry/SL/TP if valid.
        """
        if regime == Regime.RANGE or regime == Regime.NO_TRADE:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "regime_not_trend"})

        direction = Direction.LONG if regime == Regime.TREND_UP else Direction.SHORT

        if ind.atr14 is None or ind.atr14 <= 0:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "atr_invalid"})

        n = self._N
        min_rows = n + self._retest_timeout + 1
        if len(df_m1) < min_rows:
            return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                                 values={"reason": "insufficient_data",
                                         "rows": len(df_m1), "needed": min_rows})

        atr = ind.atr14
        borderline_flags = []

        # --- Step 1: Compression ---
        # Look at N candles before the potential breakout candle
        # The breakout candle is at index -(1 + retest_timeout) to allow retest window
        # We check multiple positions for the breakout within the retest window
        for retest_bars in range(self._retest_timeout + 1):
            breakout_idx = -(1 + retest_bars)
            compression_end = len(df_m1) + breakout_idx
            compression_start = compression_end - n

            if compression_start < 0:
                continue

            compression_slice = df_m1.iloc[compression_start:compression_end]
            breakout_candle = df_m1.iloc[compression_end]

            range_n = compression_slice["high"].max() - compression_slice["low"].min()
            if range_n > self._compression_atr_mult * atr:
                continue  # No compression at this position

            # --- Step 2: Breakout ---
            highest_high = compression_slice["high"].max()
            lowest_low = compression_slice["low"].min()
            bo_open = float(breakout_candle["open"])
            bo_close = float(breakout_candle["close"])
            bo_high = float(breakout_candle["high"])
            bo_low = float(breakout_candle["low"])

            body = abs(bo_close - bo_open)
            candle_range = bo_high - bo_low
            body_ratio = body / candle_range if candle_range > 0 else 0

            if direction == Direction.LONG:
                breakout_level = highest_high + self._breakout_buffer_atr * atr
                if bo_close <= breakout_level:
                    continue  # No breakout
                if body_ratio < self._body_ratio_min:
                    continue
            else:  # SHORT
                breakout_level = lowest_low - self._breakout_buffer_atr * atr
                if bo_close >= breakout_level:
                    continue
                if body_ratio < self._body_ratio_min:
                    continue

            # --- Step 3: Momentum ---
            if direction == Direction.LONG:
                if ind.rsi14 is None or ind.rsi14 < self._rsi_min_long:
                    return TriggerSignal(
                        phase=TriggerPhase.NO_MOMENTUM, direction=direction,
                        values={"rsi": ind.rsi14, "threshold": self._rsi_min_long,
                                "body_ratio": round(body_ratio, 3), "range_n": round(range_n, 6)})
            else:
                rsi_max_short = 100 - self._rsi_min_long  # 45
                if ind.rsi14 is None or ind.rsi14 > rsi_max_short:
                    return TriggerSignal(
                        phase=TriggerPhase.NO_MOMENTUM, direction=direction,
                        values={"rsi": ind.rsi14, "threshold": rsi_max_short,
                                "body_ratio": round(body_ratio, 3), "range_n": round(range_n, 6)})

            # MACD histogram increasing for 2 bars
            if not self._macd_increasing(ind, direction):
                return TriggerSignal(
                    phase=TriggerPhase.NO_MOMENTUM, direction=direction,
                    values={"reason": "macd_not_increasing",
                            "macd_hist": ind.macd_hist,
                            "macd_hist_prev": ind.macd_hist_prev,
                            "body_ratio": round(body_ratio, 3)})

            # --- Borderline checks (B2-B5) ---
            bl_body_low = self._bl.get("body_ratio_low", 0.55)
            bl_body_high = self._bl.get("body_ratio_high", 0.65)
            if bl_body_low <= body_ratio <= bl_body_high:
                borderline_flags.append("B2")

            bl_rsi_long_low = self._bl.get("rsi_long_low", 55)
            bl_rsi_long_high = self._bl.get("rsi_long_high", 62)
            bl_rsi_short_low = self._bl.get("rsi_short_low", 38)
            bl_rsi_short_high = self._bl.get("rsi_short_high", 45)
            if direction == Direction.LONG and ind.rsi14 is not None:
                if bl_rsi_long_low <= ind.rsi14 <= bl_rsi_long_high:
                    borderline_flags.append("B3")
            elif direction == Direction.SHORT and ind.rsi14 is not None:
                if bl_rsi_short_low <= ind.rsi14 <= bl_rsi_short_high:
                    borderline_flags.append("B3")

            bl_macd_ratio = self._bl.get("macd_weak_atr_ratio", 0.30)
            if ind.macd_hist is not None and abs(ind.macd_hist) < bl_macd_ratio * atr:
                borderline_flags.append("B5")

            # --- Step 4: Retest ---
            if retest_bars == 0:
                # Breakout just happened on the latest candle — wait for retest
                return TriggerSignal(
                    phase=TriggerPhase.WAITING_RETEST, direction=direction,
                    breakout_level=breakout_level,
                    borderline_flags=borderline_flags or None,
                    values={"body_ratio": round(body_ratio, 3),
                            "range_n": round(range_n, 6),
                            "breakout_level": breakout_level})

            # Check retest candles (between breakout and now)
            retest_candles = df_m1.iloc[compression_end + 1:]
            tolerance = self._retest_tolerance_atr * atr
            retest_found = False
            retest_low = None
            retest_high = None

            for _, rc in retest_candles.iterrows():
                if direction == Direction.LONG:
                    if rc["low"] <= breakout_level + tolerance:
                        retest_found = True
                        if retest_low is None or rc["low"] < retest_low:
                            retest_low = rc["low"]
                else:
                    if rc["high"] >= breakout_level - tolerance:
                        retest_found = True
                        if retest_high is None or rc["high"] > retest_high:
                            retest_high = rc["high"]

            if not retest_found:
                if retest_bars >= self._retest_timeout:
                    return TriggerSignal(
                        phase=TriggerPhase.RETEST_TIMEOUT, direction=direction,
                        values={"breakout_level": breakout_level,
                                "retest_bars": retest_bars})
                continue

            # --- Step 5: SL & TP ---
            entry_price = breakout_level
            sl_price, tp_price, risk_pips = self._compute_sl_tp(
                direction, entry_price, breakout_level, atr,
                retest_low, retest_high, ind, pair)

            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                risk_pips=risk_pips,
                breakout_level=breakout_level,
                borderline_flags=borderline_flags or None,
                values={
                    "body_ratio": round(body_ratio, 3),
                    "range_n": round(range_n, 6),
                    "rsi": ind.rsi14,
                    "macd_hist": ind.macd_hist,
                    "retest_bars": retest_bars,
                },
            )

        # No valid compression/breakout found in any position
        return TriggerSignal(phase=TriggerPhase.NO_COMPRESSION,
                             values={"reason": "no_compression_found"})

    def _macd_increasing(self, ind: IndicatorSet, direction: Direction) -> bool:
        """Check MACD histogram increasing for 2 bars in the trade direction."""
        if any(v is None for v in [ind.macd_hist, ind.macd_hist_prev, ind.macd_hist_prev2]):
            return False
        if direction == Direction.LONG:
            return (ind.macd_hist > ind.macd_hist_prev
                    and ind.macd_hist_prev > ind.macd_hist_prev2)
        else:
            return (ind.macd_hist < ind.macd_hist_prev
                    and ind.macd_hist_prev < ind.macd_hist_prev2)

    def _compute_sl_tp(self, direction: Direction, entry: float,
                       breakout_level: float, atr: float,
                       retest_low: Optional[float], retest_high: Optional[float],
                       ind: IndicatorSet, pair: str = "EUR_USD"
                       ) -> tuple[float, float, float]:
        """Compute SL, TP, and risk in pips.

        Per spec table 12:
        SL = min(low_retest - buffer, breakout_level - 0.8*ATR)
        buffer = max(1.5 * spread, 0.10 * ATR)
        TP1 = 1.0R
        """
        buffer = 0.10 * atr

        if direction == Direction.LONG:
            sl_from_retest = (retest_low - buffer) if retest_low is not None else float('inf')
            sl_from_breakout = breakout_level - self._sl_atr * atr
            sl = min(sl_from_retest, sl_from_breakout)
            risk = entry - sl
            tp = entry + self._tp_R * risk
        else:
            sl_from_retest = (retest_high + buffer) if retest_high is not None else float('-inf')
            sl_from_breakout = breakout_level + self._sl_atr * atr
            sl = max(sl_from_retest, sl_from_breakout)
            risk = sl - entry
            tp = entry - self._tp_R * risk

        from ..utils.pip_utils import price_to_pips
        risk_pips = abs(price_to_pips(risk, pair))

        return sl, tp, risk_pips
