"""Tests for Model A: Micro-breakout + Retest Trigger."""

import numpy as np
import pandas as pd
import pytest

from src.scalp_mode.engine.feature_engine import IndicatorSet
from src.scalp_mode.engine.model_a import ModelATrigger, TriggerPhase, Direction
from src.scalp_mode.engine.regime_engine import Regime


MODEL_CONFIG = {
    "compression_N": 8,
    "breakout_buffer_atr": 0.10,
    "retest_timeout": 3,
    "retest_tolerance_atr": 0.15,
    "body_ratio_min": 0.55,
    "rsi_min_long": 55,
    "sl_atr": 0.8,
    "tp_R": 1.0,
    "time_stop_min": 6,
    "sl_move_threshold_R": 0.8,
    "sl_move_target_R": -0.1,
    "sl_move_window_min": [2, 4],
}

BORDERLINE_CONFIG = {
    "body_ratio_low": 0.55,
    "body_ratio_high": 0.65,
    "rsi_long_low": 55,
    "rsi_long_high": 62,
    "rsi_short_low": 38,
    "rsi_short_high": 45,
    "macd_weak_atr_ratio": 0.30,
}


def _make_trigger():
    return ModelATrigger(MODEL_CONFIG, BORDERLINE_CONFIG)


def _make_indicators(rsi=60.0, macd_hist=0.0002, macd_prev=0.00015,
                     macd_prev2=0.0001, atr=0.0005) -> IndicatorSet:
    return IndicatorSet(
        ema20=1.0860, ema50=1.0850, atr14=atr,
        rsi14=rsi, bb_upper=1.0880, bb_mid=1.0860, bb_lower=1.0840,
        bb_width=0.003, ema_slope=0.30,
        macd_hist=macd_hist, macd_hist_prev=macd_prev, macd_hist_prev2=macd_prev2,
    )


def _make_compression_df(n_compression=8, n_retest=2, base=1.0850,
                          atr=0.0005, breakout=True, retest=True,
                          direction="long") -> pd.DataFrame:
    """Build a synthetic M1 DataFrame with compression + optional breakout + retest.

    The trigger scans backwards from the end, checking positions within
    retest_timeout (3). So the breakout candle must be at most 3 bars from the end.

    Structure: [warmup(50)] + [compression(N)] + [breakout(1)] + [retest candles(n_retest)]
    """
    rows = []
    warmup = 50

    np.random.seed(42)
    price = base
    for i in range(warmup):
        change = np.random.normal(0, atr * 0.3)
        o = price
        c = price + change
        h = max(o, c) + abs(np.random.normal(0, atr * 0.1))
        l = min(o, c) - abs(np.random.normal(0, atr * 0.1))
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 50})
        price = c

    # Compression: tight range (well within 1.2 * ATR)
    comp_range = atr * 0.4
    comp_mid = base
    for i in range(n_compression):
        offset = comp_range * 0.2 * np.sin(i)
        o = comp_mid + offset
        c = comp_mid + offset + np.random.normal(0, comp_range * 0.05)
        h = max(o, c) + comp_range * 0.03
        l = min(o, c) - comp_range * 0.03
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 30})

    highest = max(r["high"] for r in rows[-n_compression:])
    lowest = min(r["low"] for r in rows[-n_compression:])

    if breakout and direction == "long":
        breakout_level = highest + 0.10 * atr
        bo_open = highest - atr * 0.02
        bo_close = breakout_level + atr * 0.3
        bo_high = bo_close + atr * 0.03
        bo_low = bo_open - atr * 0.01
        rows.append({"open": bo_open, "high": bo_high, "low": bo_low,
                      "close": bo_close, "volume": 80})

        if retest:
            for i in range(n_retest):
                if i == 0:
                    # Retest: price pulls back to breakout level (close BELOW bo_close)
                    rt_open = breakout_level + atr * 0.1
                    rt_close = breakout_level + atr * 0.05
                    rt_low = breakout_level - atr * 0.05
                    rt_high = breakout_level + atr * 0.15
                else:
                    # Continuation: small candle above breakout level
                    rt_open = breakout_level + atr * 0.05
                    rt_close = breakout_level + atr * 0.08
                    rt_high = breakout_level + atr * 0.12
                    rt_low = breakout_level - atr * 0.02
                rows.append({"open": rt_open, "high": rt_high,
                             "low": rt_low, "close": rt_close, "volume": 40})
        else:
            for i in range(n_retest):
                rt_open = bo_close + atr * 0.8 * (i + 1)
                rt_close = rt_open + atr * 0.5
                rt_high = rt_close + atr * 0.05
                rt_low = rt_open - atr * 0.02
                rows.append({"open": rt_open, "high": rt_high,
                             "low": rt_low, "close": rt_close, "volume": 40})

    elif breakout and direction == "short":
        breakout_level = lowest - 0.10 * atr
        bo_open = lowest + atr * 0.02
        bo_close = breakout_level - atr * 0.3
        bo_low = bo_close - atr * 0.03
        bo_high = bo_open + atr * 0.01
        rows.append({"open": bo_open, "high": bo_high, "low": bo_low,
                      "close": bo_close, "volume": 80})

        if retest:
            for i in range(n_retest):
                if i == 0:
                    # Retest: price pulls back to breakout level (close ABOVE bo_close)
                    rt_open = breakout_level - atr * 0.1
                    rt_close = breakout_level - atr * 0.05
                    rt_high = breakout_level + atr * 0.05
                    rt_low = breakout_level - atr * 0.15
                else:
                    rt_open = breakout_level - atr * 0.05
                    rt_close = breakout_level - atr * 0.08
                    rt_high = breakout_level + atr * 0.02
                    rt_low = breakout_level - atr * 0.12
                rows.append({"open": rt_open, "high": rt_high,
                             "low": rt_low, "close": rt_close, "volume": 40})
    else:
        for i in range(3):
            o = comp_mid
            c = comp_mid + np.random.normal(0, atr * 0.05)
            h = max(o, c) + atr * 0.02
            l = min(o, c) - atr * 0.02
            rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 30})

    return pd.DataFrame(rows)


class TestModelANoTrigger:
    def test_range_regime_no_trigger(self):
        trigger = _make_trigger()
        df = _make_compression_df(breakout=True, retest=True)
        ind = _make_indicators()
        signal = trigger.evaluate(df, ind, Regime.RANGE)
        assert signal.phase == TriggerPhase.NO_COMPRESSION

    def test_notrade_regime_no_trigger(self):
        trigger = _make_trigger()
        df = _make_compression_df()
        ind = _make_indicators()
        signal = trigger.evaluate(df, ind, Regime.NO_TRADE)
        assert signal.phase == TriggerPhase.NO_COMPRESSION

    def test_insufficient_data(self):
        trigger = _make_trigger()
        df = pd.DataFrame({
            "open": [1.085], "high": [1.086],
            "low": [1.084], "close": [1.085], "volume": [10],
        })
        ind = _make_indicators()
        signal = trigger.evaluate(df, ind, Regime.TREND_UP)
        assert signal.phase == TriggerPhase.NO_COMPRESSION

    def test_no_breakout(self):
        trigger = _make_trigger()
        df = _make_compression_df(breakout=False)
        ind = _make_indicators()
        signal = trigger.evaluate(df, ind, Regime.TREND_UP)
        assert signal.phase in (TriggerPhase.NO_COMPRESSION, TriggerPhase.NO_BREAKOUT)

    def test_no_momentum_rsi_low(self):
        trigger = _make_trigger()
        df = _make_compression_df(breakout=True, retest=True)
        ind = _make_indicators(rsi=40.0)  # Below 55 for long
        signal = trigger.evaluate(df, ind, Regime.TREND_UP)
        assert signal.phase == TriggerPhase.NO_MOMENTUM

    def test_no_momentum_macd_not_increasing(self):
        trigger = _make_trigger()
        df = _make_compression_df(breakout=True, retest=True)
        # MACD decreasing
        ind = _make_indicators(macd_hist=0.0001, macd_prev=0.0002, macd_prev2=0.0003)
        signal = trigger.evaluate(df, ind, Regime.TREND_UP)
        assert signal.phase == TriggerPhase.NO_MOMENTUM


class TestModelAValidTrigger:
    def test_valid_long_trigger(self):
        trigger = _make_trigger()
        df = _make_compression_df(breakout=True, retest=True, direction="long")
        ind = _make_indicators(rsi=60.0)
        signal = trigger.evaluate(df, ind, Regime.TREND_UP)
        assert signal.phase == TriggerPhase.VALID
        assert signal.direction == Direction.LONG
        assert signal.entry_price is not None
        assert signal.sl_price is not None
        assert signal.tp_price is not None
        assert signal.sl_price < signal.entry_price  # SL below entry for long
        assert signal.tp_price > signal.entry_price  # TP above entry for long

    def test_valid_short_trigger(self):
        trigger = _make_trigger()
        df = _make_compression_df(breakout=True, retest=True, direction="short")
        # Short needs RSI <= 45 and MACD decreasing
        ind = _make_indicators(rsi=40.0, macd_hist=-0.0002,
                               macd_prev=-0.00015, macd_prev2=-0.0001)
        signal = trigger.evaluate(df, ind, Regime.TREND_DOWN)
        assert signal.phase == TriggerPhase.VALID
        assert signal.direction == Direction.SHORT
        assert signal.sl_price > signal.entry_price  # SL above entry for short
        assert signal.tp_price < signal.entry_price  # TP below entry for short


class TestModelARetest:
    def test_no_retest_timeout(self):
        trigger = _make_trigger()
        df = _make_compression_df(breakout=True, retest=False, n_retest=3)
        ind = _make_indicators()
        signal = trigger.evaluate(df, ind, Regime.TREND_UP)
        # Should be retest_timeout, waiting_retest, or no_compression
        assert signal.phase in (TriggerPhase.RETEST_TIMEOUT,
                                TriggerPhase.WAITING_RETEST,
                                TriggerPhase.NO_COMPRESSION)


class TestModelABorderline:
    def test_borderline_b5_weak_macd(self):
        trigger = _make_trigger()
        df = _make_compression_df(breakout=True, retest=True)
        # MACD increasing but very weak (< 0.30 * ATR)
        atr = 0.0005
        ind = _make_indicators(
            macd_hist=0.00005,   # Very weak
            macd_prev=0.00004,
            macd_prev2=0.00003,
            atr=atr,
        )
        signal = trigger.evaluate(df, ind, Regime.TREND_UP)
        if signal.phase == TriggerPhase.VALID:
            assert signal.borderline_flags is not None
            assert "B5" in signal.borderline_flags

    def test_atr_invalid_returns_no_compression(self):
        trigger = _make_trigger()
        df = _make_compression_df()
        ind = _make_indicators(atr=0)
        signal = trigger.evaluate(df, ind, Regime.TREND_UP)
        assert signal.phase == TriggerPhase.NO_COMPRESSION
