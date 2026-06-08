"""Portfolio Agent — Manages open positions (SL moves, exits).

Runs every 1 minute when positions are open. Completely separate
from entry decisions. Focused on capital preservation and profit
maximization of existing trades.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from .brain import Brain

_log = logging.getLogger("scalp_mode")

MIN_SL_MOVE_MINUTES = 10  # Cannot move SL within 10 min of open
MIN_SL_MOVE_REPEAT_MINUTES = 10  # Min minutes between SL moves
MAX_SL_MOVE_FRAC_OF_ORIGINAL_RISK = 0.5  # Max one-way SL step per decision


@dataclass
class PortfolioAction:
    """An action on an open position."""
    action: str  # "hold", "close", "modify_sl", "set_sl"
    trade_id: str
    pair: str = ""
    new_sl_price: float = 0.0
    reasoning: str = ""


class PortfolioAgent:
    """Manages open positions with discipline.

    Hard rules (code-enforced):
    - Cannot move SL within 10 minutes of trade open
    - Cannot move SL further from entry (only closer)
    """

    def __init__(self, brain: Brain, model: str = "claude-opus-4-20250514"):
        self._brain = brain
        self._model = model

    def run(self, open_trades: list[dict], live_prices: dict,
            utc_now: datetime) -> list[PortfolioAction]:
        """Evaluate all open positions and return actions."""
        if not open_trades:
            return []

        now_utc = utc_now if utc_now.tzinfo else utc_now.replace(
            tzinfo=timezone.utc)

        # Build context
        lessons = self._brain.read_lessons(10)
        market_state = self._brain.read_market_state()

        system = (
            "You are the Portfolio Agent managing open forex positions. "
            "Your job is to protect capital and maximize profits on existing trades.\n\n"
            "RULES:\n"
            "1. You CANNOT move stop loss within 10 minutes of trade open.\n"
            "2. Only move SL closer to entry (tighter), never further.\n"
            "3. Only move SL to breakeven after price moved 50%+ toward TP.\n"
            "4. Each MODIFY_SL may move SL by at most 50% of the original "
            "entry-to-SL distance (one step); wait 10+ minutes between moves.\n"
            "5. Close a trade if the original thesis is clearly invalidated.\n"
            "6. Do NOT close a trade just because it's slightly underwater — noise is normal.\n"
            "7. Let winners run to TP. Don't take early profits unless something changed.\n"
            "8. When a trade has a pre-committed exit_plan, FOLLOW IT unless the "
            "thesis is clearly invalidated. Don't second-guess your own plan.\n"
        )

        if lessons:
            system += "\nLESSONS:\n"
            for l in lessons:
                system += f"- {l.get('pattern', '')}\n"

        # Build position info
        parts = [f"Time: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}"]

        if market_state.get("analysis"):
            parts.append(f"\nMarket: {market_state['analysis'][:200]}")

        parts.append("\nOPEN POSITIONS:")
        for t in open_trades:
            pair = t.get("pair", "?")
            mid = live_prices.get(pair, (0, 0))
            mid_price = (mid[0] + mid[1]) / 2 if isinstance(mid, tuple) and mid[0] > 0 else 0

            if t.get("direction") == "long":
                unrealized = (mid_price - t.get("entry_price", 0)) * \
                    (100 if "JPY" in pair else 10000)
            else:
                unrealized = (t.get("entry_price", 0) - mid_price) * \
                    (100 if "JPY" in pair else 10000)

            open_time = t.get("open_time")
            if isinstance(open_time, datetime):
                ot = open_time if open_time.tzinfo else open_time.replace(
                    tzinfo=timezone.utc)
                minutes_open = (now_utc - ot).total_seconds() / 60
            else:
                minutes_open = 999

            sl_distance = abs(t.get("entry_price", 0) - t.get("sl_price", 0)) * \
                (100 if "JPY" in pair else 10000)
            tp_distance = abs(t.get("tp_price", 0) - t.get("entry_price", 0)) * \
                (100 if "JPY" in pair else 10000)
            progress = (unrealized / tp_distance * 100) if tp_distance > 0 else 0

            can_move_sl = "YES" if minutes_open >= MIN_SL_MOVE_MINUTES else \
                f"NO ({MIN_SL_MOVE_MINUTES - minutes_open:.0f} min remaining)"

            exit_plan = t.get("exit_plan") or ""
            plan_line = (
                f"\n    Pre-committed exit plan: {exit_plan}"
                if exit_plan else "")
            parts.append(
                f"\n  Trade {t.get('trade_id')}: {pair} {t.get('direction')}\n"
                f"    Entry: {t.get('entry_price', 0):.5f} | "
                f"SL: {t.get('sl_price', 0):.5f} | TP: {t.get('tp_price', 0):.5f}\n"
                f"    Current: {mid_price:.5f} | Unrealized: {unrealized:+.1f} pips\n"
                f"    Progress toward TP: {progress:.0f}% | Open: {minutes_open:.0f} min\n"
                f"    Can modify SL: {can_move_sl}"
                + plan_line)

        parts.append(
            "\n\nFor each position, respond with one of:\n"
            "HOLD <trade_id>: <brief reason>\n"
            "CLOSE <trade_id>: <brief reason>\n"
            "MODIFY_SL <trade_id> <new_price>: <brief reason>\n"
            "SET_SL <trade_id> <price>: <brief reason>"
            "   (only if the trade currently has SL=0 — installs a fresh SL)\n"
        )

        try:
            import anthropic
            import httpx
            client = anthropic.Anthropic(
                timeout=httpx.Timeout(30.0, connect=10.0))
            response = client.messages.create(
                model=self._model,
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": "\n".join(parts)}],
            )
            text = response.content[0].text.strip()
            _log.info(f"Portfolio Agent:\n{text}")
        except Exception as e:
            _log.error(f"Portfolio Agent failed: {e}")
            return []

        # Parse actions
        actions = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("HOLD"):
                continue  # no action needed

            if stripped.startswith("CLOSE"):
                parts_line = stripped.split(":", 1)
                trade_id = parts_line[0].replace("CLOSE", "").strip()
                reason = parts_line[1].strip() if len(parts_line) > 1 else ""
                actions.append(PortfolioAction(
                    action="close", trade_id=trade_id, reasoning=reason))

            elif stripped.startswith("SET_SL"):
                try:
                    rest = stripped.replace("SET_SL", "").strip()
                    import re as _re
                    rest = _re.sub(r'^(\d+)\s*:\s*', r'\1 ', rest)
                    parts_line = rest.split(":", 1)
                    id_and_price = parts_line[0].strip().split()
                    trade_id = id_and_price[0]
                    new_sl = float(id_and_price[1])
                    reason = parts_line[1].strip() if len(parts_line) > 1 else ""
                    # Only allow SET_SL when trade currently has no SL
                    trade = next(
                        (t for t in open_trades
                         if str(t.get("trade_id")) == trade_id), None)
                    if trade is None:
                        continue
                    current_sl = float(trade.get("sl_price", 0) or 0)
                    if current_sl > 0:
                        _log.info(
                            f"Portfolio: SET_SL rejected on {trade_id} — "
                            f"already has SL={current_sl:.5f}; use MODIFY_SL")
                        continue
                    actions.append(PortfolioAction(
                        action="set_sl", trade_id=trade_id,
                        new_sl_price=new_sl, reasoning=reason))
                except (ValueError, IndexError):
                    _log.warning(f"Portfolio: failed to parse SET_SL: {stripped}")

            elif stripped.startswith("MODIFY_SL"):
                try:
                    rest = stripped.replace("MODIFY_SL", "").strip()
                    # Handle both "ID PRICE: reason" and "ID: PRICE: reason"
                    # by normalising colon-after-ID into space
                    import re as _re
                    rest = _re.sub(r'^(\d+)\s*:\s*', r'\1 ', rest)
                    parts_line = rest.split(":", 1)
                    id_and_price = parts_line[0].strip().split()
                    trade_id = id_and_price[0]
                    new_sl = float(id_and_price[1])
                    reason = parts_line[1].strip() if len(parts_line) > 1 else ""

                    # Hard guardrail: check 10-minute minimum from open
                    trade = next((t for t in open_trades
                                  if str(t.get("trade_id")) == trade_id), None)
                    if trade:
                        open_time = trade.get("open_time")
                        if isinstance(open_time, datetime):
                            ot = open_time if open_time.tzinfo else open_time.replace(
                                tzinfo=timezone.utc)
                            elapsed = (now_utc - ot).total_seconds() / 60
                            if elapsed < MIN_SL_MOVE_MINUTES:
                                _log.info(
                                    f"Portfolio: blocked SL move on {trade_id} "
                                    f"({elapsed:.0f}m < {MIN_SL_MOVE_MINUTES}m)")
                                continue
                        # 10 min since previous SL move
                        last_mv = trade.get("last_sl_move_time")
                        if isinstance(last_mv, datetime):
                            lmv = last_mv if last_mv.tzinfo else last_mv.replace(
                                tzinfo=timezone.utc)
                            if (now_utc - lmv) < timedelta(
                                    minutes=MIN_SL_MOVE_REPEAT_MINUTES):
                                _log.info(
                                    f"Portfolio: blocked SL move on {trade_id} "
                                    f"(<{MIN_SL_MOVE_REPEAT_MINUTES}m since last move)")
                                continue
                        # Max 50% of original risk (|entry-orig_sl|) per step
                        old_sl = float(trade.get("sl_price", 0) or 0)
                        risk_orig = float(trade.get("risk_amount", 0) or 0)
                        direction = trade.get("direction", "long")
                        max_step = (
                            MAX_SL_MOVE_FRAC_OF_ORIGINAL_RISK * risk_orig
                            if risk_orig > 0 else 0.0)
                        if direction == "long":
                            if new_sl < old_sl - 1e-9:
                                _log.info(
                                    f"Portfolio: blocked SL widen on {trade_id}")
                                continue
                            if new_sl - old_sl > max_step + 1e-9:
                                _log.info(
                                    f"Portfolio: blocked SL step too large "
                                    f"on {trade_id} ({new_sl - old_sl:.6f} > {max_step:.6f})")
                                continue
                        else:
                            if new_sl > old_sl + 1e-9:
                                _log.info(
                                    f"Portfolio: blocked SL widen on {trade_id}")
                                continue
                            if old_sl - new_sl > max_step + 1e-9:
                                _log.info(
                                    f"Portfolio: blocked SL step too large "
                                    f"on {trade_id} ({old_sl - new_sl:.6f} > {max_step:.6f})")
                                continue

                    actions.append(PortfolioAction(
                        action="modify_sl", trade_id=trade_id,
                        new_sl_price=new_sl, reasoning=reason))
                except (ValueError, IndexError) as e:
                    _log.warning(f"Portfolio: failed to parse MODIFY_SL: {stripped}")

        return actions
