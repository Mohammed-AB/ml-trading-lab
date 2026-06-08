"""Tests for Performance Analyzer."""

import pytest
from datetime import datetime, timezone
from src.scalp_mode.backtest.performance import PerformanceAnalyzer, PerformanceMetrics
from src.scalp_mode.backtest.backtester import BacktestTrade


def _utc(day, hour=14, minute=0):
    return datetime(2026, 1, day, hour, minute, tzinfo=timezone.utc)


def _make_trade(pnl_pips, exit_reason="tp_hit", spread=0.5, slippage=0.1,
                direction="long", is_borderline=False, day=1):
    entry = 1.08500
    if direction == "long":
        exit_p = entry + pnl_pips * 0.0001
    else:
        exit_p = entry - pnl_pips * 0.0001
    return BacktestTrade(
        pair="EUR_USD", direction=direction,
        entry_time=_utc(day, 14, 0), exit_time=_utc(day, 14, 3),
        entry_price=entry, exit_price=exit_p,
        sl_price=entry - 0.0004, tp_price=entry + 0.0004,
        units=10000, pnl_pips=pnl_pips, pnl_pct=pnl_pips * 0.0001,
        exit_reason=exit_reason, hold_time_seconds=180,
        spread_at_entry=spread, slippage_pips=slippage,
        is_borderline=is_borderline,
    )


class TestEmptyTrades:
    def test_empty_returns_zeros(self):
        analyzer = PerformanceAnalyzer()
        m = analyzer.compute([])
        assert m.total_trades == 0
        assert m.sharpe_ratio == 0.0
        assert m.win_rate == 0.0


class TestWinRate:
    def test_all_winners(self):
        trades = [_make_trade(3.0) for _ in range(10)]
        m = PerformanceAnalyzer().compute(trades)
        assert m.win_rate == 1.0

    def test_all_losers(self):
        trades = [_make_trade(-3.0, "sl_hit") for _ in range(10)]
        m = PerformanceAnalyzer().compute(trades)
        assert m.win_rate == 0.0

    def test_mixed(self):
        trades = [_make_trade(3.0)] * 6 + [_make_trade(-3.0, "sl_hit")] * 4
        m = PerformanceAnalyzer().compute(trades)
        assert m.win_rate == 0.6


class TestProfitFactor:
    def test_profit_factor(self):
        trades = ([_make_trade(4.0)] * 6 +
                  [_make_trade(-3.0, "sl_hit")] * 4)
        m = PerformanceAnalyzer().compute(trades)
        # PF = (6*4) / (4*3) = 24/12 = 2.0
        assert m.profit_factor == 2.0

    def test_no_losses_infinite(self):
        trades = [_make_trade(3.0) for _ in range(5)]
        m = PerformanceAnalyzer().compute(trades)
        assert m.profit_factor == float('inf')


class TestMaxDrawdown:
    def test_drawdown_from_losses(self):
        # 5 losses of 0.1% each → expect some drawdown
        trades = [_make_trade(-1.0, "sl_hit") for _ in range(5)]
        m = PerformanceAnalyzer().compute(trades)
        assert m.max_drawdown_pct > 0

    def test_no_drawdown_all_wins(self):
        trades = [_make_trade(3.0) for _ in range(5)]
        m = PerformanceAnalyzer().compute(trades)
        assert m.max_drawdown_pct == 0.0


class TestConsecutiveLosses:
    def test_max_consecutive(self):
        trades = ([_make_trade(3.0)] +
                  [_make_trade(-2.0, "sl_hit")] * 4 +
                  [_make_trade(3.0)])
        m = PerformanceAnalyzer().compute(trades)
        assert m.max_consecutive_losses == 4

    def test_no_losses(self):
        trades = [_make_trade(3.0) for _ in range(5)]
        m = PerformanceAnalyzer().compute(trades)
        assert m.max_consecutive_losses == 0


class TestExitReasons:
    def test_exit_breakdown(self):
        trades = ([_make_trade(3.0, "tp_hit")] * 3 +
                  [_make_trade(-2.0, "sl_hit")] * 2 +
                  [_make_trade(0.5, "time_stop")] * 1)
        m = PerformanceAnalyzer().compute(trades)
        assert m.tp_hit_count == 3
        assert m.sl_hit_count == 2
        assert m.time_stop_count == 1


class TestBorderline:
    def test_borderline_stats(self):
        trades = ([_make_trade(3.0, is_borderline=True)] * 2 +
                  [_make_trade(-2.0, "sl_hit", is_borderline=True)] * 1 +
                  [_make_trade(3.0)] * 3)
        m = PerformanceAnalyzer().compute(trades)
        assert m.borderline_count == 3
        assert abs(m.borderline_win_rate - 0.6667) < 0.01


class TestSlippageImpact:
    def test_slippage_impact(self):
        # Each trade: spread=0.5, slippage=0.1 → cost=0.6 pips/trade
        # 5 winning trades at +3.0 pips each → gross_profit = 15.0 pips
        # Total slippage = 5 * 0.6 = 3.0
        # Impact = 3.0 / 15.0 * 100 = 20%
        trades = [_make_trade(3.0, spread=0.5, slippage=0.1) for _ in range(5)]
        m = PerformanceAnalyzer().compute(trades)
        assert m.slippage_impact_pct == 20.0
