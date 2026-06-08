"""Go/No-Go Evaluator — Spec section 0.7 acceptance criteria.

Evaluates whether to proceed from one phase to the next:
  Backtest → Paper:  Sharpe ≥ 1.2, WR ≥ 52%, MaxDD ≤ 5%, PF ≥ 1.3,
                     Slippage ≤ 25%, Trades ≥ 200
  Paper → Live:      10 days, avg slippage ≤ 0.3 pip, latency ≤ 2s,
                     bounds reject ≤ 15%, WR within ±5% of backtest
  Stop criteria:     Sharpe < 0.8, slippage > 40%, WR gap > 10%, loss > 3%/week
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .performance import PerformanceMetrics


class Verdict(str, Enum):
    GO = "GO"
    NO_GO = "NO_GO"
    STOP = "STOP"


@dataclass
class CriterionResult:
    name: str
    passed: bool
    actual: float
    threshold: float
    description: str


@dataclass
class GoNoGoResult:
    verdict: Verdict
    criteria: list[CriterionResult] = field(default_factory=list)
    failed_criteria: list[str] = field(default_factory=list)
    stop_criteria: list[str] = field(default_factory=list)


class GoNoGoEvaluator:
    """Evaluates Go/No-Go criteria per spec 0.7.

    Usage:
        evaluator = GoNoGoEvaluator()
        result = evaluator.backtest_to_paper(metrics)
        result = evaluator.paper_to_live(paper_metrics, backtest_metrics, paper_stats)
    """

    # --- Backtest → Paper thresholds (spec table 7) ---
    BT_SHARPE_MIN = 1.2
    BT_WIN_RATE_MIN = 0.52
    BT_MAX_DD_MAX = 5.0
    BT_PROFIT_FACTOR_MIN = 1.3
    BT_SLIPPAGE_IMPACT_MAX = 25.0
    BT_MIN_TRADES = 200

    # --- Stop criteria (spec table 9) ---
    STOP_SHARPE = 0.8
    STOP_SLIPPAGE = 40.0
    STOP_WR_GAP = 0.10
    STOP_WEEKLY_LOSS = 0.03

    def backtest_to_paper(self, metrics: PerformanceMetrics,
                          walk_forward_win_pct: Optional[float] = None
                          ) -> GoNoGoResult:
        """Evaluate Backtest → Paper Go/No-Go.

        Args:
            metrics: PerformanceMetrics from backtest
            walk_forward_win_pct: % of walk-forward windows that are profitable
                                  (0.0 to 1.0). If None, criterion is skipped.
        """
        criteria = []
        failed = []
        stop = []

        # Sharpe Ratio
        c = CriterionResult(
            "sharpe_ratio", metrics.sharpe_ratio >= self.BT_SHARPE_MIN,
            metrics.sharpe_ratio, self.BT_SHARPE_MIN,
            "Annualized Sharpe Ratio after costs")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Check stop condition
        if metrics.sharpe_ratio < self.STOP_SHARPE:
            stop.append(f"sharpe_{metrics.sharpe_ratio:.2f}_below_{self.STOP_SHARPE}")

        # Win Rate
        c = CriterionResult(
            "win_rate", metrics.win_rate >= self.BT_WIN_RATE_MIN,
            metrics.win_rate, self.BT_WIN_RATE_MIN,
            "Overall win rate")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Max Drawdown
        c = CriterionResult(
            "max_drawdown", metrics.max_drawdown_pct <= self.BT_MAX_DD_MAX,
            metrics.max_drawdown_pct, self.BT_MAX_DD_MAX,
            "Maximum drawdown %")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Profit Factor
        c = CriterionResult(
            "profit_factor", metrics.profit_factor >= self.BT_PROFIT_FACTOR_MIN,
            metrics.profit_factor, self.BT_PROFIT_FACTOR_MIN,
            "Gross profit / gross loss")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Slippage Impact
        c = CriterionResult(
            "slippage_impact", metrics.slippage_impact_pct <= self.BT_SLIPPAGE_IMPACT_MAX,
            metrics.slippage_impact_pct, self.BT_SLIPPAGE_IMPACT_MAX,
            "Slippage + spread as % of gross profit")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Check stop condition
        if metrics.slippage_impact_pct > self.STOP_SLIPPAGE:
            stop.append(f"slippage_{metrics.slippage_impact_pct:.1f}%_above_{self.STOP_SLIPPAGE}%")

        # Trade Count
        c = CriterionResult(
            "trade_count", metrics.total_trades >= self.BT_MIN_TRADES,
            metrics.total_trades, self.BT_MIN_TRADES,
            "Minimum trades for statistical significance")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Walk-forward stability (optional)
        if walk_forward_win_pct is not None:
            c = CriterionResult(
                "walk_forward", walk_forward_win_pct >= 0.60,
                walk_forward_win_pct, 0.60,
                "% of walk-forward windows profitable")
            criteria.append(c)
            if not c.passed:
                failed.append(c.name)

        # Determine verdict
        if stop:
            verdict = Verdict.STOP
        elif failed:
            verdict = Verdict.NO_GO
        else:
            verdict = Verdict.GO

        return GoNoGoResult(
            verdict=verdict, criteria=criteria,
            failed_criteria=failed, stop_criteria=stop)

    def paper_to_live(self, paper_metrics: PerformanceMetrics,
                      backtest_metrics: PerformanceMetrics,
                      paper_stats: Optional[dict] = None) -> GoNoGoResult:
        """Evaluate Paper → Live Go/No-Go.

        Args:
            paper_metrics: PerformanceMetrics from paper trading
            backtest_metrics: PerformanceMetrics from backtest (for comparison)
            paper_stats: Optional dict with:
                - trading_days: int
                - avg_slippage_pips: float
                - avg_latency_ms: float
                - bounds_reject_pct: float
                - kill_switch_count: int
        """
        criteria = []
        failed = []
        stop = []
        stats = paper_stats or {}

        # Min trading days (spec table 8)
        days = stats.get("trading_days", 0)
        c = CriterionResult("min_paper_days", days >= 10, days, 10,
                            "Minimum paper trading days")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Avg slippage
        avg_slip = stats.get("avg_slippage_pips", 0)
        c = CriterionResult("avg_slippage", avg_slip <= 0.3, avg_slip, 0.3,
                            "Average actual slippage in pips")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Avg latency
        avg_lat = stats.get("avg_latency_ms", 0)
        c = CriterionResult("avg_latency", avg_lat <= 2000, avg_lat, 2000,
                            "Average end-to-end latency ms")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Bounds reject rate
        reject_pct = stats.get("bounds_reject_pct", 0)
        c = CriterionResult("bounds_reject", reject_pct <= 15, reject_pct, 15,
                            "Bounds violation rejection %")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Win rate comparison (within ±5%)
        wr_gap = abs(paper_metrics.win_rate - backtest_metrics.win_rate)
        c = CriterionResult("win_rate_match", wr_gap <= 0.05, wr_gap, 0.05,
                            "Paper vs backtest win rate gap")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        # Stop condition: WR gap > 10%
        if wr_gap > self.STOP_WR_GAP:
            stop.append(f"wr_gap_{wr_gap:.2%}_above_{self.STOP_WR_GAP:.0%}")

        # Kill switch (spec table 8)
        ks_count = stats.get("kill_switch_count", 0)
        c = CriterionResult("kill_switch", ks_count <= 2, ks_count, 2,
                            "Kill switch activations (max 2)")
        criteria.append(c)
        if not c.passed:
            failed.append(c.name)

        if stop:
            verdict = Verdict.STOP
        elif failed:
            verdict = Verdict.NO_GO
        else:
            verdict = Verdict.GO

        return GoNoGoResult(
            verdict=verdict, criteria=criteria,
            failed_criteria=failed, stop_criteria=stop)
