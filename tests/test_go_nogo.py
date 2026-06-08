"""Tests for Go/No-Go Evaluator."""

import pytest
from src.scalp_mode.backtest.go_nogo import GoNoGoEvaluator, Verdict
from src.scalp_mode.backtest.performance import PerformanceMetrics


class TestBacktestToPaper:
    def test_all_pass_go(self):
        evaluator = GoNoGoEvaluator()
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5, win_rate=0.55, max_drawdown_pct=3.0,
            profit_factor=1.8, slippage_impact_pct=15.0, total_trades=250)
        result = evaluator.backtest_to_paper(metrics)
        assert result.verdict == Verdict.GO
        assert len(result.failed_criteria) == 0

    def test_sharpe_too_low_nogo(self):
        evaluator = GoNoGoEvaluator()
        metrics = PerformanceMetrics(
            sharpe_ratio=1.0, win_rate=0.55, max_drawdown_pct=3.0,
            profit_factor=1.8, slippage_impact_pct=15.0, total_trades=250)
        result = evaluator.backtest_to_paper(metrics)
        assert result.verdict == Verdict.NO_GO
        assert "sharpe_ratio" in result.failed_criteria

    def test_sharpe_critical_stop(self):
        evaluator = GoNoGoEvaluator()
        metrics = PerformanceMetrics(
            sharpe_ratio=0.5, win_rate=0.45, max_drawdown_pct=8.0,
            profit_factor=0.9, slippage_impact_pct=50.0, total_trades=250)
        result = evaluator.backtest_to_paper(metrics)
        assert result.verdict == Verdict.STOP
        assert len(result.stop_criteria) > 0

    def test_insufficient_trades(self):
        evaluator = GoNoGoEvaluator()
        metrics = PerformanceMetrics(
            sharpe_ratio=2.0, win_rate=0.60, max_drawdown_pct=2.0,
            profit_factor=2.5, slippage_impact_pct=10.0, total_trades=100)
        result = evaluator.backtest_to_paper(metrics)
        assert result.verdict == Verdict.NO_GO
        assert "trade_count" in result.failed_criteria

    def test_drawdown_too_high(self):
        evaluator = GoNoGoEvaluator()
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5, win_rate=0.55, max_drawdown_pct=7.0,
            profit_factor=1.8, slippage_impact_pct=15.0, total_trades=250)
        result = evaluator.backtest_to_paper(metrics)
        assert result.verdict == Verdict.NO_GO
        assert "max_drawdown" in result.failed_criteria

    def test_slippage_critical_stop(self):
        evaluator = GoNoGoEvaluator()
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5, win_rate=0.55, max_drawdown_pct=3.0,
            profit_factor=1.8, slippage_impact_pct=45.0, total_trades=250)
        result = evaluator.backtest_to_paper(metrics)
        assert result.verdict == Verdict.STOP
        assert any("slippage" in s for s in result.stop_criteria)

    def test_walk_forward_optional(self):
        evaluator = GoNoGoEvaluator()
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5, win_rate=0.55, max_drawdown_pct=3.0,
            profit_factor=1.8, slippage_impact_pct=15.0, total_trades=250)
        result = evaluator.backtest_to_paper(metrics, walk_forward_win_pct=0.70)
        assert result.verdict == Verdict.GO

    def test_walk_forward_fails(self):
        evaluator = GoNoGoEvaluator()
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5, win_rate=0.55, max_drawdown_pct=3.0,
            profit_factor=1.8, slippage_impact_pct=15.0, total_trades=250)
        result = evaluator.backtest_to_paper(metrics, walk_forward_win_pct=0.50)
        assert result.verdict == Verdict.NO_GO
        assert "walk_forward" in result.failed_criteria

    def test_criteria_count(self):
        evaluator = GoNoGoEvaluator()
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5, win_rate=0.55, max_drawdown_pct=3.0,
            profit_factor=1.8, slippage_impact_pct=15.0, total_trades=250)
        result = evaluator.backtest_to_paper(metrics)
        # 6 criteria without walk_forward
        assert len(result.criteria) == 6


class TestPaperToLive:
    def test_all_pass_go(self):
        evaluator = GoNoGoEvaluator()
        paper = PerformanceMetrics(win_rate=0.54)
        bt = PerformanceMetrics(win_rate=0.55)
        stats = {
            "trading_days": 12, "avg_slippage_pips": 0.2,
            "avg_latency_ms": 1500, "bounds_reject_pct": 10,
            "kill_switch_count": 0,
        }
        result = evaluator.paper_to_live(paper, bt, stats)
        assert result.verdict == Verdict.GO

    def test_insufficient_days(self):
        evaluator = GoNoGoEvaluator()
        paper = PerformanceMetrics(win_rate=0.55)
        bt = PerformanceMetrics(win_rate=0.55)
        stats = {"trading_days": 5, "avg_slippage_pips": 0.2,
                 "avg_latency_ms": 1500, "bounds_reject_pct": 10,
                 "kill_switch_count": 0}
        result = evaluator.paper_to_live(paper, bt, stats)
        assert result.verdict == Verdict.NO_GO
        assert "min_paper_days" in result.failed_criteria

    def test_win_rate_gap_stop(self):
        evaluator = GoNoGoEvaluator()
        paper = PerformanceMetrics(win_rate=0.42)
        bt = PerformanceMetrics(win_rate=0.55)
        stats = {"trading_days": 12, "avg_slippage_pips": 0.2,
                 "avg_latency_ms": 1500, "bounds_reject_pct": 10,
                 "kill_switch_count": 0}
        result = evaluator.paper_to_live(paper, bt, stats)
        assert result.verdict == Verdict.STOP

    def test_high_latency_nogo(self):
        evaluator = GoNoGoEvaluator()
        paper = PerformanceMetrics(win_rate=0.55)
        bt = PerformanceMetrics(win_rate=0.55)
        stats = {"trading_days": 12, "avg_slippage_pips": 0.2,
                 "avg_latency_ms": 3000, "bounds_reject_pct": 10,
                 "kill_switch_count": 0}
        result = evaluator.paper_to_live(paper, bt, stats)
        assert result.verdict == Verdict.NO_GO
        assert "avg_latency" in result.failed_criteria
