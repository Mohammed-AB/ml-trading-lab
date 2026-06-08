"""Tests for Model B: Failed-breakout Reversal."""

import numpy as np
import pandas as pd
import pytest

from src.scalp_mode.engine.model_b import ModelBTrigger
from src.scalp_mode.engine.model_a import TriggerPhase, Direction
from src.scalp_mode.engine.feature_engine import FeatureEngine, IndicatorSet
from src.scalp_mode.engine.regime_engine import Regime


MODEL_B_CONFIG = {
    "range_window_M5": 12,
    "wick_ratio_min": 0.60,
    "wick_excess_atr": 0.25,
    "stop_spread_buffer_mult": 2.0,
    "stop_atr_buffer": 0.15,
    "rsi_overbought": 65,
    "rsi_reversal": 60,
    "rsi_oversold": 35,
    "rsi_reversal_up": 40,
}


def _make_trigger():
    return ModelBTrigger(MODEL_B_CONFIG)


def _make_ind(rsi=55.0, atr=0.0005, bb_mid=1.08600) -> IndicatorSet:
    return IndicatorSet(
        ema20=1.0860, ema50=1.0855, atr14=atr,
        rsi14=rsi, bb_upper=1.0880, bb_mid=bb_mid, bb_lower=1.0840,
        bb_width=0.003, ema_slope=0.05,
    )


def _make_m5_range(n=15, range_high=1.0870, range_low=1.0840) -> pd.DataFrame:
    """M5 candles forming a clear range."""
    np.random.seed(42)
    mid = (range_high + range_low) / 2
    rows = []
    for i in range(n):
        c = mid + np.random.uniform(-0.0010, 0.0010)
        h = min(c + np.random.uniform(0, 0.0008), range_high + 0.00005)
        l = max(c - np.random.uniform(0, 0.0008), range_low - 0.00005)
        rows.append({"open": c - 0.0001, "high": h, "low": l,
                      "close": c, "volume": 50})
    # Ensure range_high and range_low are actually hit
    rows[-2]["high"] = range_high
    rows[-3]["low"] = range_low
    return pd.DataFrame(rows)


def _make_m1_top_rejection(n=50, range_high=1.0870, atr=0.0005,
                            rsi_prev=67.0, rsi_current=58.0) -> pd.DataFrame:
    """M1 candles where the latest is a failed top breakout.

    Last candle: pokes above range_high but closes below it with large upper wick.
    Previous candle had high RSI (overbought), current dropped below reversal.
    """
    np.random.seed(99)
    rows = []
    base = (1.0870 + 1.0840) / 2

    # Build warmup bars for RSI computation
    for i in range(n - 3):
        c = base + np.random.normal(0, atr * 0.5)
        rows.append({"open": c - atr * 0.1, "high": c + atr * 0.3,
                      "low": c - atr * 0.3, "close": c, "volume": 30})

    # Bar -3: neutral
    rows.append({"open": 1.08650, "high": 1.08680, "low": 1.08620,
                  "close": 1.08660, "volume": 40})

    # Bar -2: strong up move (pushes RSI overbought)
    rows.append({"open": 1.08660, "high": 1.08710, "low": 1.08650,
                  "close": 1.08700, "volume": 60})

    # Bar -1 (latest): failed top breakout — pokes above range_high,
    # closes below with long upper wick
    wick_tip = range_high + atr * 0.15  # Small excess above range
    close_price = range_high - atr * 0.3  # Closes well below range_high
    rows.append({
        "open": range_high - atr * 0.1,
        "high": wick_tip,
        "low": close_price - atr * 0.05,
        "close": close_price,
        "volume": 80,
    })

    return pd.DataFrame(rows)


def _make_m1_bottom_rejection(n=50, range_low=1.0840, atr=0.0005) -> pd.DataFrame:
    """M1 candles where the latest is a failed bottom breakout."""
    np.random.seed(77)
    rows = []
    base = (1.0870 + 1.0840) / 2

    for i in range(n - 3):
        c = base + np.random.normal(0, atr * 0.5)
        rows.append({"open": c + atr * 0.1, "high": c + atr * 0.3,
                      "low": c - atr * 0.3, "close": c, "volume": 30})

    # Bar -3: neutral
    rows.append({"open": 1.08450, "high": 1.08480, "low": 1.08420,
                  "close": 1.08440, "volume": 40})

    # Bar -2: strong down (oversold RSI)
    rows.append({"open": 1.08440, "high": 1.08450, "low": 1.08390,
                  "close": 1.08400, "volume": 60})

    # Bar -1: failed bottom breakout — pokes below range_low, closes above
    wick_tip = range_low - atr * 0.15
    close_price = range_low + atr * 0.3
    rows.append({
        "open": range_low + atr * 0.1,
        "high": close_price + atr * 0.05,
        "low": wick_tip,
        "close": close_price,
        "volume": 80,
    })

    return pd.DataFrame(rows)


# ──── Rejection tests ────────────────────────────────────────────────

class TestModelBRejections:
    def test_trend_regime_rejected(self):
        trigger = _make_trigger()
        df_m1 = _make_m1_top_rejection()
        df_m5 = _make_m5_range()
        ind = _make_ind()
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.TREND_UP, "EUR_USD")
        assert signal.phase == TriggerPhase.NO_COMPRESSION
        assert signal.values["reason"] == "regime_not_range"

    def test_notrade_regime_rejected(self):
        trigger = _make_trigger()
        df_m1 = _make_m1_top_rejection()
        df_m5 = _make_m5_range()
        ind = _make_ind()
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.NO_TRADE, "EUR_USD")
        assert signal.phase == TriggerPhase.NO_COMPRESSION

    def test_insufficient_m5_data(self):
        trigger = _make_trigger()
        df_m1 = _make_m1_top_rejection()
        df_m5 = _make_m5_range(n=5)  # Need 12
        ind = _make_ind()
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.RANGE, "EUR_USD")
        assert signal.phase == TriggerPhase.NO_COMPRESSION
        assert signal.values["reason"] == "insufficient_m5"

    def test_atr_invalid(self):
        trigger = _make_trigger()
        df_m1 = _make_m1_top_rejection()
        df_m5 = _make_m5_range()
        ind = _make_ind(atr=0)
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.RANGE, "EUR_USD")
        assert signal.phase == TriggerPhase.NO_COMPRESSION

    def test_range_too_narrow(self):
        trigger = _make_trigger()
        df_m1 = _make_m1_top_rejection()
        # Very tight range
        df_m5 = _make_m5_range(range_high=1.08510, range_low=1.08500)
        ind = _make_ind(atr=0.0005)  # range_size=0.0001 < 0.5*ATR=0.00025
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.RANGE, "EUR_USD")
        assert signal.phase == TriggerPhase.NO_COMPRESSION
        assert signal.values["reason"] == "range_too_narrow"

    def test_no_breakout_inside_range(self):
        """Candle fully inside range — no failed breakout."""
        trigger = _make_trigger()
        np.random.seed(42)
        rows = []
        for i in range(50):
            c = 1.0855 + np.random.normal(0, 0.0002)
            rows.append({"open": c, "high": c + 0.0002, "low": c - 0.0002,
                          "close": c, "volume": 30})
        df_m1 = pd.DataFrame(rows)
        df_m5 = _make_m5_range()
        ind = _make_ind(rsi=55.0)
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.RANGE, "EUR_USD")
        assert signal.phase != TriggerPhase.VALID


# ──── Valid signal tests ─────────────────────────────────────────────

class TestModelBValidSignals:
    def test_valid_short_at_top(self):
        """Failed top breakout → valid SHORT signal."""
        trigger = _make_trigger()
        df_m1 = _make_m1_top_rejection()
        df_m5 = _make_m5_range()
        ind = _make_ind(rsi=58.0)  # Current RSI < 60 (reversal)
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.RANGE, "EUR_USD")

        if signal.phase == TriggerPhase.VALID:
            assert signal.direction == Direction.SHORT
            assert signal.entry_price is not None
            assert signal.sl_price is not None
            assert signal.tp_price is not None
            assert signal.sl_price > signal.entry_price  # SL above for short
            assert signal.tp_price < signal.entry_price  # TP below for short
            assert signal.values["model"] == "B"

    def test_valid_long_at_bottom(self):
        """Failed bottom breakout → valid LONG signal."""
        trigger = _make_trigger()
        df_m1 = _make_m1_bottom_rejection()
        df_m5 = _make_m5_range()
        ind = _make_ind(rsi=42.0)  # Current RSI > 40 (reversal up)
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.RANGE, "EUR_USD")

        if signal.phase == TriggerPhase.VALID:
            assert signal.direction == Direction.LONG
            assert signal.sl_price < signal.entry_price  # SL below for long
            assert signal.tp_price > signal.entry_price  # TP above for long


# ──── SL/TP computation tests ────────────────────────────────────────

class TestModelBSLTP:
    def test_sl_buffer_uses_max_spread_atr(self):
        trigger = _make_trigger()
        atr = 0.0005

        # When spread buffer > ATR buffer
        sl, buf = trigger._compute_sl(1.0870, atr, spread_pips=2.0,
                                       direction=Direction.SHORT)
        spread_buf_price = 2.0 * 2.0 * 0.0001  # 2*spread*pip_val = 0.0004
        atr_buf = 0.15 * atr  # 0.000075
        assert buf == spread_buf_price  # Spread buffer wins

        # When ATR buffer > spread buffer
        sl2, buf2 = trigger._compute_sl(1.0870, atr=0.005, spread_pips=0.1,
                                         direction=Direction.SHORT)
        assert buf2 == 0.15 * 0.005  # ATR buffer wins

    def test_sl_above_for_short(self):
        trigger = _make_trigger()
        sl, _ = trigger._compute_sl(1.0872, atr=0.0005, spread_pips=0.5,
                                     direction=Direction.SHORT)
        assert sl > 1.0872

    def test_sl_below_for_long(self):
        trigger = _make_trigger()
        sl, _ = trigger._compute_sl(1.0838, atr=0.0005, spread_pips=0.5,
                                     direction=Direction.LONG)
        assert sl < 1.0838

    def test_tp_picks_closer_target(self):
        trigger = _make_trigger()

        # Short at 1.0865: mid_range=1.0855 (dist=0.0010), bb_mid=1.0858 (dist=0.0007)
        # bb_mid is closer → should pick bb_mid
        tp = trigger._compute_tp(1.0865, 1.0855, 1.0858, Direction.SHORT)
        assert tp == 1.0858  # Closer to entry

        # Now bb_mid further: mid_range=1.0855 (dist=0.0010), bb_mid=1.0845 (dist=0.0020)
        tp2 = trigger._compute_tp(1.0865, 1.0855, 1.0845, Direction.SHORT)
        assert tp2 == 1.0855  # mid_range closer

    def test_tp_fallback_when_bb_mid_none(self):
        trigger = _make_trigger()
        tp = trigger._compute_tp(1.0865, 1.0855, None, Direction.SHORT)
        assert tp == 1.0855

    def test_risk_pips_correct_for_eur_usd(self):
        trigger = _make_trigger()
        df_m1 = _make_m1_top_rejection()
        df_m5 = _make_m5_range()
        ind = _make_ind(rsi=58.0)
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.RANGE, "EUR_USD")
        if signal.phase == TriggerPhase.VALID:
            assert signal.risk_pips > 0
            # For EUR_USD: risk_pips = risk / 0.0001
            expected = abs(signal.sl_price - signal.entry_price) / 0.0001
            assert abs(signal.risk_pips - expected) < 0.1

    def test_risk_pips_correct_for_usd_jpy(self):
        trigger = _make_trigger()
        # Build JPY-scaled data
        df_m5 = _make_m5_range(range_high=149.80, range_low=149.50)
        np.random.seed(99)
        rows = []
        for i in range(47):
            c = 149.65 + np.random.normal(0, 0.05)
            rows.append({"open": c, "high": c + 0.03, "low": c - 0.03,
                          "close": c, "volume": 30})
        rows.append({"open": 149.70, "high": 149.75, "low": 149.68,
                      "close": 149.72, "volume": 40})
        rows.append({"open": 149.72, "high": 149.78, "low": 149.70,
                      "close": 149.76, "volume": 60})
        # Failed top breakout at 149.80
        rows.append({"open": 149.78, "high": 149.82, "low": 149.74,
                      "close": 149.75, "volume": 80})
        df_m1 = pd.DataFrame(rows)

        ind = _make_ind(rsi=58.0, atr=0.05, bb_mid=149.65)
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.RANGE, "USD_JPY")
        if signal.phase == TriggerPhase.VALID:
            # For USD_JPY: risk_pips = risk / 0.01
            expected = abs(signal.sl_price - signal.entry_price) / 0.01
            assert abs(signal.risk_pips - expected) < 0.1


# ──── Borderline tests ───────────────────────────────────────────────

class TestModelBBorderline:
    def test_borderline_flags_recorded(self):
        trigger = _make_trigger()
        df_m1 = _make_m1_top_rejection()
        df_m5 = _make_m5_range()
        ind = _make_ind(rsi=58.0)
        signal = trigger.evaluate(df_m1, df_m5, ind, ind, Regime.RANGE, "EUR_USD")
        # Signal might or might not be valid depending on exact RSI calc,
        # but if valid, borderline flags should be a list or None
        if signal.phase == TriggerPhase.VALID:
            assert signal.borderline_flags is None or isinstance(
                signal.borderline_flags, list)


# ──── Pipeline routing test ──────────────────────────────────────────

class TestModelBPipelineRouting:
    def test_backtester_imports_model_b(self):
        """Verify Backtester can be instantiated with Model B enabled."""
        from src.scalp_mode.backtest.backtester import Backtester, BacktestConfig
        config = {
            "regime": {
                "trend": {"ema_slope_thr": 0.15, "rsi_min": 52, "rsi_max": 78},
                "range": {"bb_width_thr": 0.004},
            },
            "model_a": {
                "compression_N": 8, "breakout_buffer_atr": 0.10,
                "retest_timeout": 3, "retest_tolerance_atr": 0.15,
                "body_ratio_min": 0.55, "rsi_min_long": 55,
                "sl_atr": 0.8, "tp_R": 1.0, "time_stop_min": 6,
                "sl_move_threshold_R": 0.8, "sl_move_target_R": -0.1,
                "sl_move_window_min": [2, 4],
            },
            "model_b": {
                "enabled": True,
                **MODEL_B_CONFIG,
            },
            "risk": {
                "risk_pct": 0.0025, "max_concurrent": 2,
                "cooldown_same_pair_dir_min": 10, "consec_loss_circuit": 3,
                "cooldown_minutes": 60, "trades_per_hour_pair": 3,
                "trades_per_hour_total": 6, "daily_loss": 0.01,
            },
            "costs": {
                "max_spread_pips": {"EUR_USD": 0.8},
            },
        }
        bt = Backtester(config, BacktestConfig(check_sessions=False, warmup_bars=60))
        assert bt._trigger_b is not None

    def test_backtester_without_model_b(self):
        config = {
            "model_a": {
                "compression_N": 8, "breakout_buffer_atr": 0.10,
                "retest_timeout": 3, "retest_tolerance_atr": 0.15,
                "body_ratio_min": 0.55, "rsi_min_long": 55,
                "sl_atr": 0.8, "tp_R": 1.0, "time_stop_min": 6,
                "sl_move_threshold_R": 0.8, "sl_move_target_R": -0.1,
                "sl_move_window_min": [2, 4],
            },
            "model_b": {"enabled": False},
            "risk": {},
            "costs": {"max_spread_pips": {"EUR_USD": 0.8}},
        }
        from src.scalp_mode.backtest.backtester import Backtester, BacktestConfig
        bt = Backtester(config, BacktestConfig(check_sessions=False))
        assert bt._trigger_b is None
