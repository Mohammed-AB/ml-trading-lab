"""AI Post-trade Analyst — Daily analysis after session close.

Per spec 5.1: frequency=daily, non-critical, fail-safe=continue.

Reads decision_log and trade_log, analyzes patterns, generates
daily report. Does NOT affect live trading — purely analytical.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_log = logging.getLogger("scalp_mode")


@dataclass
class DailyReport:
    date: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_pips: float = 0.0
    model_a_trades: int = 0
    model_a_win_rate: float = 0.0
    model_b_trades: int = 0
    model_b_win_rate: float = 0.0
    borderline_trades: int = 0
    borderline_win_rate: float = 0.0
    avg_slippage_pips: float = 0.0
    avg_hold_seconds: float = 0.0
    hourly_pnl: dict = field(default_factory=dict)
    exit_reasons: dict = field(default_factory=dict)
    ai_summary: Optional[str] = None
    suggestions: list = field(default_factory=list)


class PostTradeAnalyst:
    """Daily post-trade analysis — runs after session close.

    Usage:
        analyst = PostTradeAnalyst(config)
        report = analyst.analyze_day("logs/", "2026-03-28")
    """

    def __init__(self, config: dict):
        self._enabled = config.get("enabled", False)
        self._model = config.get("model", "claude-opus-4-20250514")
        self._output_dir = Path(config.get("output_dir", "logs/daily_analysis"))

    @property
    def enabled(self) -> bool:
        return self._enabled

    def analyze_day(self, log_dir: str, date: str) -> DailyReport:
        """Analyze one day's trading from log files.

        Args:
            log_dir: Directory containing decision_log.jsonl and trade_log.jsonl
            date: Date string "YYYY-MM-DD"

        Returns:
            DailyReport with statistics and (optionally) AI insights.
        """
        report = DailyReport(date=date)
        log_path = Path(log_dir)

        # Read trade log
        trades = self._read_log(log_path / "trade_log.jsonl", date)
        decisions = self._read_log(log_path / "decision_log.jsonl", date)

        if not trades:
            report.ai_summary = "No trades executed today."
            self._save_report(report)
            return report

        # Basic stats
        report.total_trades = len(trades)
        report.wins = sum(1 for t in trades if t.get("pnl_pips", 0) > 0)
        report.losses = report.total_trades - report.wins
        report.win_rate = report.wins / report.total_trades if report.total_trades > 0 else 0
        report.total_pnl_pips = sum(t.get("pnl_pips", 0) for t in trades)

        # Model A vs B
        model_a = [t for t in trades if t.get("model") != "B"]
        model_b = [t for t in trades if t.get("model") == "B"]
        report.model_a_trades = len(model_a)
        report.model_b_trades = len(model_b)
        if model_a:
            report.model_a_win_rate = sum(
                1 for t in model_a if t.get("pnl_pips", 0) > 0) / len(model_a)
        if model_b:
            report.model_b_win_rate = sum(
                1 for t in model_b if t.get("pnl_pips", 0) > 0) / len(model_b)

        # Borderline
        bl = [t for t in trades if t.get("is_borderline")]
        report.borderline_trades = len(bl)
        if bl:
            report.borderline_win_rate = sum(
                1 for t in bl if t.get("pnl_pips", 0) > 0) / len(bl)

        # Slippage
        slippages = [abs(t.get("actual_slippage_pips", 0)) for t in trades
                     if t.get("actual_slippage_pips") is not None]
        report.avg_slippage_pips = sum(slippages) / len(slippages) if slippages else 0

        # Hold time
        holds = [t.get("hold_time_seconds", 0) for t in trades]
        report.avg_hold_seconds = sum(holds) / len(holds) if holds else 0

        # Hourly P&L
        for t in trades:
            ts = t.get("order_sent_ts", "")
            if ts and len(ts) >= 13:
                try:
                    hour = ts[11:13]
                    report.hourly_pnl[hour] = (
                        report.hourly_pnl.get(hour, 0) + t.get("pnl_pips", 0))
                except (IndexError, ValueError):
                    pass

        # Exit reasons
        for t in trades:
            reason = t.get("exit_reason", "unknown")
            report.exit_reasons[reason] = report.exit_reasons.get(reason, 0) + 1

        # Generate suggestions (rule-based, no AI needed for basic insights)
        report.suggestions = self._generate_suggestions(report, trades)

        # AI summary (if enabled and API available)
        if self._enabled:
            report.ai_summary = self._get_ai_summary(report, trades, decisions)

        self._save_report(report)
        return report

    def _read_log(self, path: Path, date: str) -> list[dict]:
        """Read JSONL log file, filter to given date."""
        records = []
        if not path.exists():
            return records
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        ts = record.get("timestamp_utc", record.get("order_sent_ts", ""))
                        if ts and ts.startswith(date):
                            records.append(record)
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass
        return records

    def _generate_suggestions(self, report: DailyReport,
                               trades: list[dict]) -> list[str]:
        """Rule-based suggestions from today's patterns."""
        suggestions = []

        if report.avg_slippage_pips > 0.3:
            suggestions.append(
                f"High avg slippage ({report.avg_slippage_pips:.2f} pips) — "
                f"consider tightening priceBound")

        if report.borderline_trades > 0 and report.borderline_win_rate < 0.40:
            suggestions.append(
                f"Borderline trades underperforming ({report.borderline_win_rate:.0%} WR) "
                f"— consider enabling AI Borderline Reviewer")

        if report.model_b_trades > 0 and report.model_b_win_rate < 0.45:
            suggestions.append(
                f"Model B win rate low ({report.model_b_win_rate:.0%}) — "
                f"review Range parameters")

        # Check hourly concentration
        if report.hourly_pnl:
            worst_hour = min(report.hourly_pnl, key=report.hourly_pnl.get)
            if report.hourly_pnl[worst_hour] < -2.0:
                suggestions.append(
                    f"Hour {worst_hour}:00 UTC had significant losses "
                    f"({report.hourly_pnl[worst_hour]:.1f} pips)")

        return suggestions

    def _get_ai_summary(self, report: DailyReport, trades: list[dict],
                         decisions: list[dict]) -> str:
        """Get AI-generated summary. Returns plain text on failure."""
        try:
            import anthropic
            client = anthropic.Anthropic()

            prompt = (
                f"Analyze this day's scalping results:\n"
                f"Date: {report.date}\n"
                f"Trades: {report.total_trades} (W:{report.wins} L:{report.losses})\n"
                f"Win Rate: {report.win_rate:.0%}\n"
                f"PnL: {report.total_pnl_pips:.1f} pips\n"
                f"Model A: {report.model_a_trades} trades, {report.model_a_win_rate:.0%} WR\n"
                f"Model B: {report.model_b_trades} trades, {report.model_b_win_rate:.0%} WR\n"
                f"Borderline: {report.borderline_trades} trades, "
                f"{report.borderline_win_rate:.0%} WR\n"
                f"Avg Slippage: {report.avg_slippage_pips:.2f} pips\n"
                f"Hourly PnL: {json.dumps(report.hourly_pnl)}\n"
                f"Exit Reasons: {json.dumps(report.exit_reasons)}\n\n"
                f"Provide a brief 2-3 sentence summary and any actionable insights."
            )

            response = client.messages.create(
                model=self._model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except Exception as e:
            _log.warning(f"AI summary generation failed: {e}")
            return f"AI summary unavailable. Rule-based: {'; '.join(report.suggestions) or 'No issues detected.'}"

    def _save_report(self, report: DailyReport) -> None:
        """Save report to JSON file."""
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            path = self._output_dir / f"{report.date}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report.__dict__, f, indent=2, default=str,
                          ensure_ascii=False)
        except IOError as e:
            _log.error(f"Failed to save daily report: {e}")
