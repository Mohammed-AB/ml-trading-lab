"""Performance Analyzer — Computes metrics for backtest results.

Metrics from spec section 0.7:
- Sharpe Ratio (annualized)
- Win Rate
- Max Drawdown
- Profit Factor
- Slippage Impact
- Trade count
- Walk-forward stability
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PerformanceMetrics:
    """Complete performance report."""
    # Core metrics (spec 0.7)
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    slippage_impact_pct: float = 0.0
    total_trades: int = 0

    # Additional
    total_pnl_pips: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pips: float = 0.0
    avg_hold_seconds: float = 0.0
    avg_winner_pips: float = 0.0
    avg_loser_pips: float = 0.0
    best_trade_pips: float = 0.0
    worst_trade_pips: float = 0.0
    max_consecutive_losses: int = 0

    # Exit reason breakdown
    tp_hit_count: int = 0
    sl_hit_count: int = 0
    time_stop_count: int = 0
    other_exit_count: int = 0

    # Borderline stats
    borderline_count: int = 0
    borderline_win_rate: float = 0.0


class PerformanceAnalyzer:
    """Computes performance metrics from a list of BacktestTrades.

    Usage:
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.compute(trades)
        go_nogo = analyzer.check_go_nogo(metrics)
    """

    # Annualization factor: ~252 trading days, ~3.5h overlap/day, ~210 M1 bars/day
    # Trades per year ≈ trades_per_day * 252
    TRADING_DAYS_PER_YEAR = 252

    def compute(self, trades: list, trading_days: Optional[int] = None) -> PerformanceMetrics:
        """Compute all performance metrics.

        Args:
            trades: List of BacktestTrade objects
            trading_days: Number of trading days in the backtest period.
                         If None, estimated from trade timestamps.
        """
        m = PerformanceMetrics()

        if not trades:
            return m

        m.total_trades = len(trades)

        # PnL arrays
        pnl_pips = np.array([t.pnl_pips for t in trades])
        pnl_pcts = np.array([t.pnl_pct for t in trades])

        m.total_pnl_pips = round(float(np.sum(pnl_pips)), 2)
        m.total_pnl_pct = round(float(np.sum(pnl_pcts)), 6)
        m.avg_pnl_pips = round(float(np.mean(pnl_pips)), 2)

        # Win rate
        winners = pnl_pips > 0
        m.win_rate = round(float(np.sum(winners)) / len(trades), 4) if trades else 0

        # Hold time
        hold_times = [t.hold_time_seconds for t in trades]
        m.avg_hold_seconds = round(float(np.mean(hold_times)), 1) if hold_times else 0

        # Best/worst
        m.best_trade_pips = round(float(np.max(pnl_pips)), 2) if len(pnl_pips) > 0 else 0
        m.worst_trade_pips = round(float(np.min(pnl_pips)), 2) if len(pnl_pips) > 0 else 0

        # Avg winner/loser
        winner_pips = pnl_pips[winners]
        loser_pips = pnl_pips[~winners]
        m.avg_winner_pips = round(float(np.mean(winner_pips)), 2) if len(winner_pips) > 0 else 0
        m.avg_loser_pips = round(float(np.mean(loser_pips)), 2) if len(loser_pips) > 0 else 0

        # Profit Factor
        gross_profit = float(np.sum(winner_pips)) if len(winner_pips) > 0 else 0
        gross_loss = abs(float(np.sum(loser_pips))) if len(loser_pips) > 0 else 0
        m.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

        # Sharpe Ratio (annualized)
        m.sharpe_ratio = self._sharpe(pnl_pcts, trading_days)

        # Max Drawdown
        m.max_drawdown_pct = self._max_drawdown(pnl_pcts)

        # Slippage Impact
        total_slippage = sum(t.slippage_pips + t.spread_at_entry for t in trades)
        m.slippage_impact_pct = round(
            total_slippage / gross_profit * 100, 1) if gross_profit > 0 else 0

        # Max consecutive losses
        m.max_consecutive_losses = self._max_consecutive_losses(pnl_pips)

        # Exit reasons
        for t in trades:
            if t.exit_reason == "tp_hit":
                m.tp_hit_count += 1
            elif t.exit_reason == "sl_hit":
                m.sl_hit_count += 1
            elif t.exit_reason == "time_stop":
                m.time_stop_count += 1
            else:
                m.other_exit_count += 1

        # Borderline stats
        borderline_trades = [t for t in trades if t.is_borderline]
        m.borderline_count = len(borderline_trades)
        if borderline_trades:
            bl_wins = sum(1 for t in borderline_trades if t.pnl_pips > 0)
            m.borderline_win_rate = round(bl_wins / len(borderline_trades), 4)

        return m

    def _sharpe(self, pnl_pcts: np.ndarray,
                trading_days: Optional[int] = None) -> float:
        """Annualized Sharpe Ratio.

        Uses daily returns aggregated from per-trade returns.
        """
        if len(pnl_pcts) < 2:
            return 0.0

        mean_ret = float(np.mean(pnl_pcts))
        std_ret = float(np.std(pnl_pcts, ddof=1))
        if std_ret == 0:
            return 0.0

        # Per-trade Sharpe → annualize by sqrt(trades_per_year)
        trades_per_day = len(pnl_pcts) / max(trading_days or 1, 1)
        trades_per_year = trades_per_day * self.TRADING_DAYS_PER_YEAR
        annualization = math.sqrt(trades_per_year)

        return round((mean_ret / std_ret) * annualization, 2)

    def _max_drawdown(self, pnl_pcts: np.ndarray) -> float:
        """Maximum drawdown as percentage."""
        if len(pnl_pcts) == 0:
            return 0.0

        equity = np.cumsum(pnl_pcts) + 1.0  # Start at 1.0 (100%)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_dd = float(np.min(drawdowns))
        return round(abs(max_dd) * 100, 2)  # As positive percentage

    def _max_consecutive_losses(self, pnl_pips: np.ndarray) -> int:
        """Maximum consecutive losing trades."""
        max_streak = 0
        current = 0
        for pnl in pnl_pips:
            if pnl < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak
