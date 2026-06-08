"""Shared Brain — Central knowledge system for all agents.

All agents read from and write to the brain. It persists on disk
and accumulates intelligence over time. Also serves as the
Obsidian vault structure.

Directory layout:
    data/brain/
        historical/       <- processed 25yr stats (read-only after generation)
        market_state/     <- Research Agent writes current analysis
        trade_log/        <- proposals, verdicts, outcomes
        lessons/          <- Learning Agent's accumulated patterns
            daily/        <- daily review markdown files
        human_notes/      <- user's Obsidian notes (agents read)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_log = logging.getLogger("scalp_mode")

BRAIN_DIR = Path("data/brain")


def _text_similar(a: str, b: str, threshold: float = 0.80) -> bool:
    """Cheap similarity check using Jaccard-on-words. No stdlib import beyond set."""
    if not a or not b:
        return False
    if a == b:
        return True
    wa = set(w for w in a.split() if len(w) > 2)
    wb = set(w for w in b.split() if len(w) > 2)
    if not wa or not wb:
        return False
    inter = len(wa & wb)
    union = len(wa | wb)
    if union == 0:
        return False
    return (inter / union) >= threshold


class Brain:
    """Shared knowledge system for multi-agent coordination.

    Usage:
        brain = Brain()
        historical = brain.read_historical("EUR_USD")
        brain.write_market_state({"bias": "bullish", ...})
        brain.log_proposal(proposal_dict)
    """

    def __init__(self, base_dir: str = "data/brain"):
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        for sub in ["historical", "market_state", "trade_log",
                     "lessons", "lessons/daily", "human_notes", "memos",
                     "mutes"]:
            (self._dir / sub).mkdir(parents=True, exist_ok=True)

    # --- Historical Intelligence (read-only) ---

    def read_historical(self, pair: str) -> str:
        """Read processed historical stats for a pair."""
        path = self._dir / "historical" / f"{pair}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"No historical data available for {pair}."

    def read_all_historical(self) -> str:
        """Read all historical intelligence files concatenated."""
        parts = []
        for path in sorted((self._dir / "historical").glob("*.md")):
            parts.append(path.read_text(encoding="utf-8"))
        return "\n\n---\n\n".join(parts) if parts else "No historical data."

    def read_strategies(self) -> str:
        """Read strategy backtest results."""
        path = self._dir / "historical" / "strategies.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "No strategy data."

    # --- Market State (Research Agent writes, others read) ---

    def read_market_state(self) -> dict:
        """Read current market analysis from Research Agent."""
        path = self._dir / "market_state" / "current.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def write_market_state(self, state: dict) -> None:
        """Write current market analysis (Research Agent only)."""
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = self._dir / "market_state" / "current.json"
        try:
            path.write_text(json.dumps(state, indent=2, default=str),
                            encoding="utf-8")
        except IOError as e:
            _log.error(f"Brain: failed to write market state: {e}")

    # --- Trade Log (proposals, verdicts, outcomes) ---

    def log_proposal(self, proposal: dict) -> None:
        """Log a trade proposal from Strategy Agent."""
        proposal["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._append_jsonl("trade_log/proposals.jsonl", proposal)

    def log_verdict(self, verdict: dict) -> None:
        """Log a risk verdict from Risk Agent."""
        verdict["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._append_jsonl("trade_log/verdicts.jsonl", verdict)

    def log_outcome(self, outcome: dict) -> None:
        """Log a trade outcome and update derived knowledge.

        Automatically:
        - Appends to outcomes.jsonl
        - Updates per-pair-per-direction rolling stats
        - Reinforces matching lessons (confidence +/- based on whether
          the outcome confirmed or contradicted them)
        """
        outcome["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._append_jsonl("trade_log/outcomes.jsonl", outcome)

        # Update derived knowledge (Tier 3.8 + Tier 4.10)
        pair = (outcome.get("pair") or "").upper()
        direction = (outcome.get("direction") or "").lower()
        pnl_pips = float(outcome.get("pnl_pips", 0) or 0)
        if pair and direction:
            try:
                now = datetime.now(timezone.utc)
                self.update_pair_stats(pair, direction, pnl_pips, now.hour)
            except Exception as e:
                _log.warning(f"Brain: pair stats update failed: {e}")
            try:
                self.reinforce_lesson_by_outcome(pair, direction, pnl_pips)
            except Exception as e:
                _log.warning(f"Brain: lesson reinforcement failed: {e}")

    def read_recent_proposals(self, count: int = 20) -> list[dict]:
        """Read last N proposals."""
        return self._read_jsonl_tail("trade_log/proposals.jsonl", count)

    def read_recent_verdicts(self, count: int = 20) -> list[dict]:
        """Read last N verdicts."""
        return self._read_jsonl_tail("trade_log/verdicts.jsonl", count)

    def read_recent_outcomes(self, count: int = 20) -> list[dict]:
        """Read last N outcomes."""
        return self._read_jsonl_tail("trade_log/outcomes.jsonl", count)

    # --- Lessons (Learning Agent writes, others read) ---

    def read_lessons(self, count: int = 50) -> list[dict]:
        """Read accumulated trading patterns/lessons.

        Returns lessons sorted by confidence desc (when present), then by
        recency. Backtest-seeded priors (confidence >= 0.85) are always
        kept in the returned list even if older than the tail window.
        """
        path = self._dir / "lessons" / "patterns.jsonl"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").strip().split("\n")
        except IOError:
            return []

        all_lessons = []
        for line in lines:
            try:
                all_lessons.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        # Always include high-confidence priors (backtest seeds)
        priors = [l for l in all_lessons
                  if float(l.get("confidence", 0) or 0) >= 0.85]
        # Recent tail for the rest
        recent = all_lessons[-count:] if count > 0 else all_lessons
        # Merge: priors + recent, dedupe by id or pattern text
        merged = []
        seen_ids = set()
        seen_patterns = set()
        for lesson in priors + recent:
            lid = lesson.get("id")
            patt = (lesson.get("pattern") or "").strip().lower()[:120]
            key = lid or patt
            if key in seen_ids or patt in seen_patterns:
                continue
            if lid:
                seen_ids.add(lid)
            if patt:
                seen_patterns.add(patt)
            merged.append(lesson)

        merged.sort(
            key=lambda l: float(l.get("confidence", 0) or 0), reverse=True)
        return merged

    _MAX_LESSONS = 15

    def write_lesson(self, lesson: dict) -> None:
        """Write a discovered pattern or lesson, deduplicating similar text.

        If the new lesson's pattern text is >60% similar to a lesson written
        in the last 20 entries, we skip the write. File is capped at
        _MAX_LESSONS entries — oldest non-pinned entries are evicted when
        the cap is reached.
        """
        lesson["discovered_at"] = datetime.now(timezone.utc).isoformat()
        new_patt = (lesson.get("pattern") or "").strip().lower()
        if new_patt:
            tail = self._read_jsonl_tail("lessons/patterns.jsonl", 20)
            for existing in tail:
                ep = (existing.get("pattern") or "").strip().lower()
                if not ep:
                    continue
                if _text_similar(new_patt, ep, threshold=0.60):
                    return
        self._append_jsonl("lessons/patterns.jsonl", lesson)
        self._enforce_lesson_cap()

    def _enforce_lesson_cap(self) -> None:
        """Keep patterns.jsonl at most _MAX_LESSONS entries.

        Pinned entries (source=user_wish, postmortem, or confidence>=0.85)
        are never evicted. Oldest non-pinned entries are removed first.
        """
        path = self._dir / "lessons" / "patterns.jsonl"
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8").strip().split("\n")
        except IOError:
            return
        entries = []
        for line in lines:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if len(entries) <= self._MAX_LESSONS:
            return
        pinned = []
        evictable = []
        for e in entries:
            src = (e.get("source") or "").lower()
            conf = float(e.get("confidence", 0) or 0)
            if src in ("user_wish", "postmortem") or conf >= 0.85:
                pinned.append(e)
            else:
                evictable.append(e)
        slots = max(0, self._MAX_LESSONS - len(pinned))
        kept = pinned + evictable[-slots:] if slots > 0 else pinned
        with open(path, "w", encoding="utf-8") as f:
            for e in kept:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def write_chief_memo(self, memo: dict) -> None:
        """Write a Chief Agent boardroom memo to a dedicated file.

        Kept out of lessons/patterns.jsonl so memos (coordination notes,
        not trade heuristics) never pollute Strategy's top-N lesson window.
        Strategy can read the latest memo separately via
        `read_latest_chief_memo()`.
        """
        memo["discovered_at"] = datetime.now(timezone.utc).isoformat()
        self._append_jsonl("memos/chief_memos.jsonl", memo)

    def read_latest_chief_memo(self) -> str:
        """Return the single most-recent Chief memo's text, or empty string."""
        tail = self._read_jsonl_tail("memos/chief_memos.jsonl", 1)
        if not tail:
            return ""
        return tail[-1].get("pattern", "") or tail[-1].get("memo", "")

    def read_mutes(self) -> dict[str, dict]:
        """Read hard-mute blacklist of pair+direction combos Risk must reject.

        File format at data/brain/mutes/blacklist.json:
            {
              "USD_CAD:long": {
                "reason": "...",
                "added_at": "ISO-ts",
                "expires_at": "ISO-ts or null"
              },
              ...
            }

        Returns a dict keyed by "PAIR:direction" (direction lowercased) with
        entries past their expires_at filtered out. Missing file = {}.
        """
        path = self._dir / "mutes" / "blacklist.json"
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return {}
        if not isinstance(raw, dict):
            return {}

        now = datetime.now(timezone.utc)
        active: dict[str, dict] = {}
        for key, val in raw.items():
            if not isinstance(val, dict):
                continue
            exp = val.get("expires_at")
            if exp:
                try:
                    exp_dt = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
                    if exp_dt.tzinfo is None:
                        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                    if exp_dt <= now:
                        continue
                except (ValueError, TypeError):
                    pass
            active[str(key).strip().upper().replace(":LONG", ":long")
                   .replace(":SHORT", ":short")] = val
        return active

    def _bump_lesson_confidence(self, existing: dict, delta: float) -> None:
        """Rewrite the whole patterns file with one lesson's confidence bumped."""
        path = self._dir / "lessons" / "patterns.jsonl"
        if not path.exists():
            return
        target_id = existing.get("id")
        target_patt = (existing.get("pattern") or "").strip().lower()
        try:
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            updated = []
            for line in lines:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    updated.append(line)
                    continue
                matches = False
                if target_id and d.get("id") == target_id:
                    matches = True
                elif (d.get("pattern") or "").strip().lower() == target_patt:
                    matches = True
                if matches:
                    cur = float(d.get("confidence", 0.5) or 0.5)
                    d["confidence"] = max(0.0, min(1.0, cur + delta))
                    d["last_reinforced"] = datetime.now(
                        timezone.utc).isoformat()
                updated.append(json.dumps(d, default=str))
            path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        except IOError as e:
            _log.warning(f"Brain: confidence bump failed: {e}")

    def reinforce_lesson_by_outcome(
        self, pair: str, direction: str, pnl_pips: float) -> None:
        """Walk lessons; if outcome matches / contradicts, bump confidence."""
        path = self._dir / "lessons" / "patterns.jsonl"
        if not path.exists():
            return
        p_low = (pair or "").upper()
        d_low = (direction or "").lower()
        was_win = pnl_pips > 0
        try:
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            updated = []
            for line in lines:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    updated.append(line)
                    continue
                patt = (d.get("pattern") or "").lower()
                scope = d.get("scope") or {}
                scope_pair = (scope.get("pair") or "").upper() if isinstance(
                    scope, dict) else ""
                scope_dir = (scope.get("direction") or "").lower() if isinstance(
                    scope, dict) else ""
                pair_match = (p_low in patt) or (scope_pair == p_low)
                dir_match = (d_low in patt) or (scope_dir == d_low)
                if pair_match and (dir_match or not scope_dir):
                    cur = float(d.get("confidence", 0.5) or 0.5)
                    # Win validates "avoid" lessons -> lower confidence;
                    # win validates "enable" lessons -> raise confidence.
                    action = (d.get("action") or "").lower()
                    if action in ("block", "avoid"):
                        delta = -0.05 if was_win else +0.05
                    else:
                        delta = +0.05 if was_win else -0.05
                    d["confidence"] = max(0.0, min(1.0, cur + delta))
                    d["last_reinforced"] = datetime.now(
                        timezone.utc).isoformat()
                updated.append(json.dumps(d, default=str))
            path.write_text("\n".join(updated) + "\n", encoding="utf-8")
        except IOError as e:
            _log.warning(f"Brain: reinforcement failed: {e}")

    # --- Per-pair-per-direction rolling stats (Tier 4.10) ---

    def update_pair_stats(self, pair: str, direction: str,
                          pnl_pips: float, hour_utc: int) -> None:
        """Update rolling win/loss stats for a pair+direction+hour slice."""
        path = self._dir / "lessons" / "pair_stats.json"
        try:
            if path.exists():
                stats = json.loads(path.read_text(encoding="utf-8") or "{}")
            else:
                stats = {}
        except (json.JSONDecodeError, IOError):
            stats = {}
        key = f"{pair}_{direction}"
        slot = stats.get(key, {
            "trades": 0, "wins": 0, "losses": 0,
            "total_pips": 0.0, "avg_pips": 0.0,
            "win_pip_sum": 0.0,
            "loss_pip_sum": 0.0,  # sum of negative P/L (negative numbers)
            "by_hour": {},
        })
        slot["trades"] += 1
        if pnl_pips > 0:
            slot["wins"] += 1
            slot["win_pip_sum"] = float(slot.get("win_pip_sum", 0.0) or 0.0) + pnl_pips
        else:
            slot["losses"] += 1
            slot["loss_pip_sum"] = float(slot.get("loss_pip_sum", 0.0) or 0.0) + pnl_pips
        slot["total_pips"] = float(slot.get("total_pips", 0.0) or 0.0) + pnl_pips
        slot["avg_pips"] = slot["total_pips"] / max(slot["trades"], 1)
        slot["win_rate"] = slot["wins"] / max(slot["trades"], 1)
        wn = int(slot["wins"])
        ln = int(slot["losses"])
        slot["avg_win"] = (
            float(slot["win_pip_sum"]) / max(wn, 1) if wn else 0.0)
        slot["avg_loss"] = (
            float(slot["loss_pip_sum"]) / max(ln, 1) if ln else 0.0)
        # Bad R:R: average loss magnitude >> average win (sample >= 3 trades)
        aw = float(slot.get("avg_win", 0) or 0)
        al = float(slot.get("avg_loss", 0) or 0)
        slot["bad_rr"] = bool(
            slot["trades"] >= 3 and wn > 0 and ln > 0 and aw > 0
            and abs(al) > 1.5 * aw)
        slot["last_pnl"] = pnl_pips
        slot["last_updated"] = datetime.now(timezone.utc).isoformat()
        hr_key = f"{hour_utc:02d}"
        hr = slot["by_hour"].get(hr_key, {
            "trades": 0, "wins": 0, "total_pips": 0.0})
        hr["trades"] += 1
        if pnl_pips > 0:
            hr["wins"] += 1
        hr["total_pips"] = float(hr.get("total_pips", 0.0) or 0.0) + pnl_pips
        slot["by_hour"][hr_key] = hr
        stats[key] = slot
        try:
            path.write_text(json.dumps(stats, indent=2, default=str),
                            encoding="utf-8")
        except IOError as e:
            _log.warning(f"Brain: pair stats write failed: {e}")

    def read_pair_stats(self) -> dict:
        """Read the rolling per-pair-per-direction stats table."""
        path = self._dir / "lessons" / "pair_stats.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8") or "{}")
        except (json.JSONDecodeError, IOError):
            return {}

    def format_pair_stats_summary(self, min_trades: int = 3) -> str:
        """One-line-per-bucket summary of which pair/direction combos work."""
        stats = self.read_pair_stats()
        if not stats:
            return "No pair stats yet."
        rows = []
        for key, s in stats.items():
            if int(s.get("trades", 0) or 0) < min_trades:
                continue
            rows.append(
                (float(s.get("total_pips", 0) or 0), key, s))
        if not rows:
            return "Pair stats: insufficient samples yet."
        rows.sort(reverse=True)
        lines = []
        for total, key, s in rows:
            lines.append(
                f"  {key}: {s['trades']} trades, "
                f"WR {s.get('win_rate', 0):.0%}, "
                f"{s.get('total_pips', 0):+.1f} pips, "
                f"avg {s.get('avg_pips', 0):+.2f}")
        return "\n".join(lines)

    def format_pair_stats_for_strategy_prompt(self, min_trades: int = 1) -> str:
        """PAIR_STATS block for Strategy Agent (avg win/loss, bad_rr flag)."""
        stats = self.read_pair_stats()
        if not stats:
            return "PAIR_STATS: (no data yet — use models + lessons only.)"
        lines = [
            "PAIR_STATS (rolling per pair_direction from live outcomes):",
        ]
        rows = []
        for key, s in stats.items():
            if int(s.get("trades", 0) or 0) < min_trades:
                continue
            rows.append((float(s.get("total_pips", 0) or 0), key, s))
        rows.sort(key=lambda x: x[1])
        for _tot, key, s in rows:
            aw = float(s.get("avg_win", 0) or 0)
            al = float(s.get("avg_loss", 0) or 0)
            brr = s.get("bad_rr", False)
            lines.append(
                f"  {key}: {s['trades']}t {s['wins']}W {s['losses']}L "
                f"net {s.get('total_pips', 0):+.1f} "
                f"avgW {aw:+.1f} avgL {al:+.1f} "
                f"bad_rr={brr}")
        if len(lines) <= 1:
            return "PAIR_STATS: (insufficient samples — use models + lessons.)"
        lines.append(
            "If bad_rr=True for your pair_direction, skip or tighten SL to "
            "<= ~avg recent winner distance (pips)."
        )
        return "\n".join(lines)

    def read_daily_reviews(self, days: int = 7) -> str:
        """Read last N days of daily reviews as formatted text."""
        daily_dir = self._dir / "lessons" / "daily"
        files = sorted(daily_dir.glob("*.md"), reverse=True)[:days]
        parts = []
        for f in reversed(files):
            parts.append(f.read_text(encoding="utf-8"))
        return "\n\n---\n\n".join(parts) if parts else "No daily reviews yet."

    def write_daily_review(self, date: str, content: str) -> None:
        """Write a daily review markdown file."""
        path = self._dir / "lessons" / "daily" / f"{date}.md"
        try:
            path.write_text(content, encoding="utf-8")
            _log.info(f"Brain: wrote daily review for {date}")
        except IOError as e:
            _log.error(f"Brain: failed to write daily review: {e}")

    # --- Human Notes (user writes via Obsidian, agents read) ---

    def read_human_notes(self) -> str:
        """Read all human notes from the Obsidian vault."""
        notes_dir = self._dir / "human_notes"
        parts = []
        for path in sorted(notes_dir.glob("*.md")):
            parts.append(f"## {path.stem}\n\n{path.read_text(encoding='utf-8')}")
        return "\n\n".join(parts) if parts else "No human notes."

    # --- Helpers ---

    def _append_jsonl(self, rel_path: str, data: dict) -> None:
        path = self._dir / rel_path
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except IOError as e:
            _log.error(f"Brain: write failed {rel_path}: {e}")

    def _read_jsonl_tail(self, rel_path: str, count: int) -> list[dict]:
        path = self._dir / rel_path
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            records = []
            for line in lines[-count:]:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return records
        except IOError:
            return []

    def get_context_summary(self, pair: Optional[str] = None) -> str:
        """Build a compact context string for agents.

        Includes: market state, recent lessons, human notes.
        Pair-specific historical data if pair is provided.
        """
        parts = []

        # Market state
        ms = self.read_market_state()
        if ms:
            parts.append(f"=== MARKET STATE (updated {ms.get('updated_at', '?')}) ===")
            parts.append(json.dumps(ms, indent=2, default=str))

        # Pair-specific historical
        if pair:
            hist = self.read_historical(pair)
            if hist:
                parts.append(f"\n=== {pair} HISTORICAL ===\n{hist}")

        # Recent lessons
        lessons = self.read_lessons(10)
        if lessons:
            parts.append("\n=== RECENT LESSONS ===")
            for l in lessons:
                parts.append(f"- [{l.get('discovered_at', '?')[:10]}] {l.get('pattern', '')}")

        # Daily reviews
        reviews = self.read_daily_reviews(3)
        if reviews != "No daily reviews yet.":
            parts.append(f"\n=== DAILY REVIEWS ===\n{reviews}")

        # Human notes
        notes = self.read_human_notes()
        if notes != "No human notes.":
            parts.append(f"\n=== HUMAN NOTES ===\n{notes}")

        return "\n".join(parts) if parts else "No brain context available."
