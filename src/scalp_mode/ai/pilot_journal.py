"""AI Pilot Journal — Learning and memory system.

Records every AI trade decision with reasoning, maintains daily
self-assessment summaries, and provides session stats.  The journal
feeds back into the AI's prompt so it learns from its own results.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

_log = logging.getLogger("scalp_mode")

TRADES_FILE = "pilot_trades.jsonl"
JOURNAL_FILE = "pilot_journal.jsonl"


@dataclass
class PilotTradeRecord:
    timestamp_utc: str
    pair: str
    direction: str
    action: str  # TRADE, SKIP, CLOSE, MODIFY_SL
    risk_pct: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    units: int = 0
    reasoning: str = ""
    confidence: float = 0.0
    # filled after trade closes
    pnl_pips: Optional[float] = None
    pnl_usd: Optional[float] = None
    exit_reason: Optional[str] = None
    hold_seconds: Optional[int] = None
    trade_id: Optional[str] = None


class PilotJournal:
    """Persistent memory for the AI Pilot.

    Stores trade records and daily summaries, loads recent history
    for injection into the AI's context.
    """

    def __init__(self, log_dir: str = "data/logs"):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._trades_path = self._dir / TRADES_FILE
        self._journal_path = self._dir / JOURNAL_FILE
        self._session_trades: list[PilotTradeRecord] = []

    # --- Recording ---

    def record_trade(self, record: PilotTradeRecord) -> None:
        """Append a trade decision to the persistent log."""
        self._session_trades.append(record)
        try:
            with open(self._trades_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), default=str) + "\n")
        except IOError as e:
            _log.error(f"Failed to write pilot trade: {e}")

    def update_trade_result(self, trade_id: str, pnl_pips: float,
                            pnl_usd: float, exit_reason: str,
                            hold_seconds: int) -> None:
        """Update a previously recorded trade with its outcome."""
        for rec in reversed(self._session_trades):
            if rec.trade_id == trade_id:
                rec.pnl_pips = pnl_pips
                rec.pnl_usd = pnl_usd
                rec.exit_reason = exit_reason
                rec.hold_seconds = hold_seconds
                break
        try:
            with open(self._trades_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "trade_result",
                    "trade_id": trade_id,
                    "pnl_pips": pnl_pips,
                    "pnl_usd": pnl_usd,
                    "exit_reason": exit_reason,
                    "hold_seconds": hold_seconds,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
        except IOError as e:
            _log.error(f"Failed to write trade result: {e}")

    # --- Session stats ---

    def get_session_stats(self) -> dict:
        """Get today's running stats for prompt context."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_trades = [
            t for t in self._session_trades
            if t.timestamp_utc.startswith(today) and t.action == "TRADE"
        ]
        completed = [t for t in today_trades if t.pnl_pips is not None]
        wins = [t for t in completed if (t.pnl_pips or 0) > 0]
        losses = [t for t in completed if (t.pnl_pips or 0) <= 0]
        total_pnl = sum(t.pnl_pips or 0 for t in completed)

        return {
            "date": today,
            "trades_opened": len(today_trades),
            "trades_closed": len(completed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(completed) if completed else 0.0,
            "total_pnl_pips": round(total_pnl, 2),
            "open_trades": len(today_trades) - len(completed),
        }

    # --- Recent history for prompt ---

    def load_recent_trades(self, count: int = 20) -> str:
        """Load last N trade records as formatted text for the AI prompt."""
        records = self._load_trades_from_disk(count)
        if not records:
            return "No previous trades recorded yet."

        lines = ["=== RECENT TRADE HISTORY ==="]
        for r in records:
            result_str = ""
            if r.get("pnl_pips") is not None:
                result_str = f" -> P/L={r['pnl_pips']:+.1f} pips ({r.get('exit_reason', '?')})"
            elif r.get("type") == "trade_result":
                continue  # skip standalone result entries
            lines.append(
                f"  {r.get('timestamp_utc', '?')[:16]} | {r.get('pair', '?')} "
                f"{r.get('direction', '?')} {r.get('action', '?')} "
                f"risk={r.get('risk_pct', 0):.1%} conf={r.get('confidence', 0):.2f}"
                f"{result_str}"
                f"\n    Reasoning: {r.get('reasoning', 'N/A')}"
            )
        return "\n".join(lines)

    def _load_trades_from_disk(self, count: int) -> list[dict]:
        """Read last N records from the trades JSONL file."""
        if not self._trades_path.exists():
            return []
        try:
            lines = self._trades_path.read_text(encoding="utf-8").strip().split("\n")
            records = []
            for line in lines[-count * 2:]:  # read extra to account for result entries
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            trade_records = [r for r in records if r.get("action") in
                            ("TRADE", "SKIP", "CLOSE", "MODIFY_SL", None)]
            return trade_records[-count:]
        except IOError:
            return []

    # --- Daily journal ---

    def load_recent_journal(self, days: int = 7) -> str:
        """Load last N days of daily summaries for the system prompt."""
        if not self._journal_path.exists():
            return "No daily journal entries yet."
        try:
            lines = self._journal_path.read_text(encoding="utf-8").strip().split("\n")
            entries = []
            for line in lines:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            recent = entries[-days:]
            if not recent:
                return "No daily journal entries yet."

            parts = ["=== DAILY JOURNAL (last {} days) ===".format(len(recent))]
            for e in recent:
                parts.append(
                    f"\n--- {e.get('date', '?')} ---\n"
                    f"Trades: {e.get('total_trades', 0)} | "
                    f"Win Rate: {e.get('win_rate', 0):.0%} | "
                    f"P/L: {e.get('total_pnl_pips', 0):+.1f} pips\n"
                    f"Self-assessment: {e.get('self_assessment', 'N/A')}"
                )
            return "\n".join(parts)
        except IOError:
            return "Journal read error."

    def write_daily_summary(self, date: str, model: str = "claude-opus-4-20250514") -> None:
        """Generate and save end-of-day self-assessment via Claude."""
        trades = self._load_trades_for_date(date)
        if not trades:
            return

        completed = [t for t in trades if t.get("pnl_pips") is not None]
        wins = sum(1 for t in completed if (t.get("pnl_pips") or 0) > 0)
        total_pnl = sum(t.get("pnl_pips", 0) for t in completed)
        win_rate = wins / len(completed) if completed else 0

        self_assessment = self._generate_self_assessment(
            date, trades, completed, wins, total_pnl, win_rate, model)

        entry = {
            "date": date,
            "total_trades": len([t for t in trades if t.get("action") == "TRADE"]),
            "total_closed": len(completed),
            "wins": wins,
            "win_rate": win_rate,
            "total_pnl_pips": round(total_pnl, 2),
            "self_assessment": self_assessment,
        }

        try:
            with open(self._journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            _log.info(f"Pilot journal: wrote daily summary for {date}")
        except IOError as e:
            _log.error(f"Failed to write journal entry: {e}")

    def _load_trades_for_date(self, date: str) -> list[dict]:
        if not self._trades_path.exists():
            return []
        records = []
        try:
            for line in self._trades_path.read_text(encoding="utf-8").strip().split("\n"):
                try:
                    r = json.loads(line)
                    if r.get("timestamp_utc", "").startswith(date):
                        records.append(r)
                except json.JSONDecodeError:
                    continue
        except IOError:
            pass
        return records

    def _generate_self_assessment(self, date: str, trades: list, completed: list,
                                   wins: int, total_pnl: float, win_rate: float,
                                   model: str) -> str:
        """Ask Claude to write a self-assessment of the day's trading."""
        try:
            import anthropic
            client = anthropic.Anthropic()

            trade_summary = []
            for t in trades[:30]:  # cap to avoid token overflow
                entry = (
                    f"{t.get('pair','?')} {t.get('direction','?')} "
                    f"{t.get('action','?')} conf={t.get('confidence',0):.2f} "
                    f"risk={t.get('risk_pct',0):.1%}"
                )
                if t.get("pnl_pips") is not None:
                    entry += f" -> {t['pnl_pips']:+.1f} pips ({t.get('exit_reason','?')})"
                entry += f"\n  Reasoning: {t.get('reasoning', 'N/A')}"
                trade_summary.append(entry)

            prompt = (
                f"You are reviewing your own trading performance for {date}.\n"
                f"Total trades: {len(trades)}, Closed: {len(completed)}, "
                f"Wins: {wins}, Win rate: {win_rate:.0%}, "
                f"P/L: {total_pnl:+.1f} pips\n\n"
                f"Your trades today:\n" + "\n".join(trade_summary) + "\n\n"
                f"Write a brief self-assessment (3-5 sentences):\n"
                f"1. What strategies/pairs worked well today?\n"
                f"2. What mistakes did you make?\n"
                f"3. What should you do differently tomorrow?\n"
                f"Be specific and honest."
            )

            response = client.messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            _log.warning(f"Self-assessment generation failed: {e}")
            summary = f"{len(completed)} trades, {wins} wins, {total_pnl:+.1f} pips."
            return f"Auto-assessment unavailable. Stats: {summary}"
