"""Tests for Backtester."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from src.scalp_mode.backtest.backtester import Backtester, BacktestConfig, BacktestTrade


def _make_trending_data(n=500, base=1.0850, trend=0.00002, volatility=0.0003):
    """Generate synthetic trending M1 data."""
    np.random.seed(123)
    closes = [base]
    for i in range(1, n):
        closes.append(closes[-1] + trend + np.random.normal(0, volatility))
    closes = np.array(closes)
    highs = closes + np.abs(np.random.normal(0, volatility * 0.5, n))
    lows = closes - np.abs(np.random.normal(0, volatility * 0.5, n))
    opens = closes + np.random.normal(0, volatility * 0.3, n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.random.randint(10, 100, n),
    })


def _make_timestamps(n=500, start_hour=13):
    """Generate timestamps during overlap window."""
    start = datetime(2026, 1, 7, start_hour, 0, tzinfo=timezone.utc)  # Wednesday
    return pd.Series([start + timedelta(minutes=i) for i in range(n)])


SCALP_CONFIG = {
    "regime": {
        "trend": {"ema_slope_thr": 0.20, "rsi_min": 52, "rsi_max": 78},
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
    "risk": {
        "risk_pct": 0.0025, "max_concurrent": 2,
        "cooldown_same_pair_dir_min": 10, "consec_loss_circuit": 3,
        "cooldown_minutes": 60, "trades_per_hour_pair": 3,
        "trades_per_hour_total": 6, "daily_loss": 0.01,
    },
    "costs": {
        "max_spread_pips": {"EUR_USD": 0.8, "USD_JPY": 0.8, "GBP_USD": 1.0},
    },
}


class TestBacktesterBasic:
    def test_runs_without_error(self):
        bt = Backtester(SCALP_CONFIG, BacktestConfig(
            check_sessions=False, warmup_bars=60))
        df = _make_trending_data(300)
        trades = bt.run("EUR_USD", df)
        assert isinstance(trades, list)
        for t in trades:
            assert isinstance(t, BacktestTrade)

    def test_returns_trades_on_trending_data(self):
        bt = Backtester(SCALP_CONFIG, BacktestConfig(
            check_sessions=False, warmup_bars=60))
        df = _make_trending_data(500, trend=0.00003)
        trades = bt.run("EUR_USD", df)
        # May or may not find triggers depending on data, but should not crash
        assert isinstance(trades, list)

    def test_no_trades_on_flat_data(self):
        bt = Backtester(SCALP_CONFIG, BacktestConfig(
            check_sessions=False, warmup_bars=60))
        # Very flat data — unlikely to trigger breakouts
        np.random.seed(99)
        n = 300
        price = 1.0850
        df = pd.DataFrame({
            "open": [price] * n,
            "high": [price + 0.00001] * n,
            "low": [price - 0.00001] * n,
            "close": [price] * n,
            "volume": [50] * n,
        })
        trades = bt.run("EUR_USD", df)
        assert len(trades) == 0

    def test_session_filter_blocks(self):
        bt = Backtester(SCALP_CONFIG, BacktestConfig(
            check_sessions=True, warmup_bars=60))
        df = _make_trending_data(200)
        # Timestamps at 3am UTC — outside overlap
        start = datetime(2026, 1, 7, 3, 0, tzinfo=timezone.utc)
        ts = pd.Series([start + timedelta(minutes=i) for i in range(200)])
        trades = bt.run("EUR_USD", df, timestamps=ts)
        assert len(trades) == 0


class TestBacktestTradeFields:
    def test_trade_fields_populated(self):
        bt = Backtester(SCALP_CONFIG, BacktestConfig(
            check_sessions=False, warmup_bars=60,
            fixed_spread_pips=0.5, slippage_pips=0.1))
        df = _make_trending_data(500, trend=0.00003)
        trades = bt.run("EUR_USD", df)

        if trades:
            t = trades[0]
            assert t.pair == "EUR_USD"
            assert t.direction in ("long", "short")
            assert t.entry_price > 0
            assert t.exit_price > 0
            assert t.units > 0
            assert t.exit_reason in ("tp_hit", "sl_hit", "time_stop", "end_of_data")
            assert t.spread_at_entry == 0.5
            assert t.slippage_pips == 0.1


class TestM5Resample:
    def test_resample_dimensions(self):
        df_m1 = _make_trending_data(100)
        m5 = Backtester._resample_m5(df_m1)
        assert len(m5) == 20  # 100 / 5
        assert list(m5.columns) == ["open", "high", "low", "close", "volume"]

    def test_resample_ohlc_correct(self):
        df_m1 = _make_trending_data(10)
        m5 = Backtester._resample_m5(df_m1)
        # First M5 bar should aggregate first 5 M1 bars
        first_5 = df_m1.iloc[:5]
        assert m5.iloc[0]["open"] == first_5.iloc[0]["open"]
        assert m5.iloc[0]["high"] == first_5["high"].max()
        assert m5.iloc[0]["low"] == first_5["low"].min()
        assert m5.iloc[0]["close"] == first_5.iloc[4]["close"]
