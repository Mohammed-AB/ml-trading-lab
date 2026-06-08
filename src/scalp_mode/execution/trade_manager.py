"""Trade Manager — Step 11 in the decision pipeline.

Manages open trades per spec 3.3 and table 12:
- Time stop: close after 6 minutes if no significant move
- SL move to breakeven: if +0.8R within 2-4 minutes, move SL to -0.1R
- Partial exit at TP2 (optional, V1 uses TP1 only)
- Tracks hold time and exit reason for logging
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import requests

from ..utils.pip_utils import price_to_pips, pips_to_price
from ..logger import ScalpLogger


class ExitReason(str, Enum):
    TP_HIT = "tp_hit"
    SL_HIT = "sl_hit"
    TIME_STOP = "time_stop"
    PARTIAL_EXIT = "partial_exit"
    MANUAL = "manual"
    KILL_SWITCH = "kill_switch"
    SL_MOVED = "sl_moved"  # Not an exit, but a management action
    AI_PILOT_CLOSE = "ai_pilot_close"


@dataclass
class ManagedTrade:
    """A trade being actively managed."""
    trade_id: str
    pair: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    units: int
    open_time: datetime
    risk_amount: float  # |entry - original_sl| in price (fixed at open)
    sl_moved_to_be: bool = False  # Has SL been moved to breakeven?
    model: str = ""  # Short model id: A/B/C/D/E/M (multi-agent)
    cluster_id: str = ""  # Forensics: same pair+direction opens within 15m share id
    last_sl_move_time: Optional[datetime] = None  # For min interval between SL moves
    exit_plan: str = ""  # Pre-committed exit rules from Strategy Agent


@dataclass
class TradeActionResult:
    """Result of evaluating a trade for management actions."""
    action: str  # "hold", "move_sl", "close"
    new_sl: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    details: Optional[dict] = None


class TradeManager:
    """Manages open trades with time stops and SL management.

    Usage:
        manager = TradeManager(config.model_a, base_url, api_token, account_id, logger)
        manager.add_trade(managed_trade)
        actions = manager.evaluate_all(utc_now, live_prices)
    """

    def __init__(self, model_config: dict, base_url: str, api_token: str,
                 account_id: str, logger: ScalpLogger):
        self._time_stop_min = model_config.get("time_stop_min", 6)
        self._sl_move_threshold_R = model_config.get("sl_move_threshold_R", 0.8)
        self._sl_move_target_R = model_config.get("sl_move_target_R", -0.1)
        self._sl_move_window = model_config.get("sl_move_window_min", [2, 4])

        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._logger = logger
        self._trades: dict[str, ManagedTrade] = {}

    def add_trade(self, trade: ManagedTrade) -> None:
        self._trades[trade.trade_id] = trade

    def remove_trade(self, trade_id: str) -> Optional[ManagedTrade]:
        return self._trades.pop(trade_id, None)

    @property
    def open_trades(self) -> list[ManagedTrade]:
        return list(self._trades.values())

    def evaluate(self, trade: ManagedTrade, current_price: float,
                 utc_now: datetime) -> TradeActionResult:
        """Evaluate a single trade for management actions.

        Args:
            trade: The managed trade
            current_price: Current mid price
            utc_now: Current UTC time

        Returns:
            TradeActionResult with recommended action.
        """
        elapsed_min = (utc_now - trade.open_time).total_seconds() / 60
        risk = trade.risk_amount

        # AI Pilot trades manage their own exits — skip rule-based time stops and SL moves
        if trade.model == "AI_PILOT":
            if trade.direction == "long":
                pnl_price = current_price - trade.entry_price
            else:
                pnl_price = trade.entry_price - current_price
            current_R = pnl_price / risk if risk > 0 else 0
            return TradeActionResult(action="hold",
                                    details={"current_R": round(current_R, 2),
                                             "elapsed_min": round(elapsed_min, 1),
                                             "managed_by": "ai_pilot"})

        # Models F/G/H (H1 strategies) use longer time horizon — SL/TP handle exits
        if trade.model in ("F", "G", "H"):
            if trade.direction == "long":
                pnl_price = current_price - trade.entry_price
            else:
                pnl_price = trade.entry_price - current_price
            current_R = pnl_price / risk if risk > 0 else 0
            return TradeActionResult(action="hold",
                                    details={"current_R": round(current_R, 2),
                                             "elapsed_min": round(elapsed_min, 1),
                                             "managed_by": "h1_strategy_fgh"})

        # Current P&L in R multiples
        if trade.direction == "long":
            pnl_price = current_price - trade.entry_price
        else:
            pnl_price = trade.entry_price - current_price
        current_R = pnl_price / risk if risk > 0 else 0

        # --- Time Stop ---
        if elapsed_min >= self._time_stop_min:
            # Close if no significant move
            if current_R < 0.5:
                return TradeActionResult(
                    action="close",
                    exit_reason=ExitReason.TIME_STOP,
                    details={"elapsed_min": round(elapsed_min, 1),
                             "current_R": round(current_R, 2)})

        # --- SL Move to Breakeven ---
        if (not trade.sl_moved_to_be
                and self._sl_move_window[0] <= elapsed_min <= self._sl_move_window[1]
                and current_R >= self._sl_move_threshold_R):
            # Move SL to entry - 0.1R for long (slightly below breakeven)
            # sl_move_target_R = -0.1 → abs = 0.1 → SL at entry - 0.1R
            sl_offset = abs(self._sl_move_target_R) * risk
            if trade.direction == "long":
                new_sl = trade.entry_price - sl_offset
            else:
                new_sl = trade.entry_price + sl_offset

            return TradeActionResult(
                action="move_sl",
                new_sl=new_sl,
                exit_reason=ExitReason.SL_MOVED,
                details={"current_R": round(current_R, 2),
                         "new_sl": new_sl,
                         "elapsed_min": round(elapsed_min, 1)})

        return TradeActionResult(action="hold",
                                details={"current_R": round(current_R, 2),
                                         "elapsed_min": round(elapsed_min, 1)})

    def evaluate_all(self, utc_now: datetime,
                     live_prices: dict[str, float]) -> list[tuple[str, TradeActionResult]]:
        """Evaluate all open trades.

        Args:
            utc_now: Current UTC time
            live_prices: {pair: mid_price}

        Returns:
            List of (trade_id, TradeAction) for trades needing action.
        """
        actions = []
        for trade_id, trade in self._trades.items():
            price = live_prices.get(trade.pair)
            if price is None:
                continue
            action = self.evaluate(trade, price, utc_now)
            if action.action != "hold":
                actions.append((trade_id, action))
        return actions

    def evaluate_auto_management(
        self, utc_now: datetime,
        live_prices: dict[str, tuple[float, float]],
    ) -> list[tuple[str, TradeActionResult]]:
        """Mechanical SL management for AI_PILOT trades.

        Runs every cycle regardless of model. Triggers:
        - Breakeven + 1 pip once price crosses 60% of TP distance
        - Trail SL to lock 1R once price crosses 2R in profit
        - Never widens SL, only tightens
        """
        actions: list[tuple[str, TradeActionResult]] = []
        for trade_id, trade in self._trades.items():
            if trade.sl_moved_to_be:
                # Still allow further trailing even after BE is hit
                pass
            lp = live_prices.get(trade.pair)
            if not lp:
                continue
            if isinstance(lp, tuple):
                bid, ask = lp
                mid = (bid + ask) / 2
            else:
                mid = float(lp)

            entry = trade.entry_price
            sl = trade.sl_price
            tp = trade.tp_price
            if entry <= 0 or sl <= 0 or tp <= 0:
                continue

            # Enforce minimum 10 min since last SL move / open
            if isinstance(trade.open_time, datetime):
                open_mins = (utc_now - trade.open_time).total_seconds() / 60.0
            else:
                open_mins = 999
            if open_mins < 10:
                continue
            last_mv = trade.last_sl_move_time
            if isinstance(last_mv, datetime):
                since = (utc_now - last_mv).total_seconds() / 60.0
                if since < 10:
                    continue

            pip_size = 0.01 if "JPY" in trade.pair else 0.0001

            # Compute progress toward TP and R multiple
            if trade.direction == "long":
                tp_dist = max(tp - entry, 1e-9)
                progress = (mid - entry) / tp_dist
                risk = max(entry - sl, 1e-9)
                current_R = (mid - entry) / risk
            else:
                tp_dist = max(entry - tp, 1e-9)
                progress = (entry - mid) / tp_dist
                risk = max(sl - entry, 1e-9)
                current_R = (entry - mid) / risk

            new_sl = None
            reason = ""

            # Trigger 1: 60% to TP -> move SL to entry + 1 pip buffer
            if progress >= 0.60 and not trade.sl_moved_to_be:
                buffer = pip_size * 1.0
                if trade.direction == "long":
                    candidate = entry + buffer
                    if candidate > sl:
                        new_sl = candidate
                else:
                    candidate = entry - buffer
                    if candidate < sl:
                        new_sl = candidate
                reason = f"auto_breakeven at {progress*100:.0f}% TP progress"

            # Trigger 2: 2R profit -> trail SL to lock 1R
            if current_R >= 2.0:
                if trade.direction == "long":
                    candidate = entry + risk  # lock 1R
                    if candidate > sl and (new_sl is None or candidate > new_sl):
                        new_sl = candidate
                        reason = (f"auto_trail_1R at "
                                  f"{current_R:.2f}R profit")
                else:
                    candidate = entry - risk
                    if candidate < sl and (new_sl is None or candidate < new_sl):
                        new_sl = candidate
                        reason = (f"auto_trail_1R at "
                                  f"{current_R:.2f}R profit")

            if new_sl is not None:
                actions.append((trade_id, TradeActionResult(
                    action="move_sl",
                    new_sl=new_sl,
                    exit_reason=ExitReason.SL_MOVED,
                    details={"reason": reason,
                             "progress": round(progress, 2),
                             "current_R": round(current_R, 2)})))
        return actions

    def execute_sl_move(self, trade_id: str, new_sl: float) -> bool:
        """Move the SL for a trade via OANDA API."""
        trade = self._trades.get(trade_id)
        if not trade:
            return False

        url = (f"{self._base_url}/v3/accounts/{self._account_id}"
               f"/trades/{trade_id}/orders")

        body = {
            "stopLoss": {
                "price": str(new_sl),
                "timeInForce": "GTC",
            }
        }

        try:
            resp = requests.put(url, headers=self._headers, json=body, timeout=5)
            if resp.status_code == 200:
                trade.sl_price = new_sl
                trade.sl_moved_to_be = True
                trade.last_sl_move_time = datetime.now(timezone.utc)
                self._logger.info(
                    f"SL moved for {trade_id}: {trade.pair} → {new_sl}")
                return True
            else:
                self._logger.error(
                    f"SL move failed for {trade_id}: {resp.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self._logger.error(f"SL move error for {trade_id}: {e}")
            return False

    def execute_close(self, trade_id: str, reason: ExitReason,
                      max_retries: int = 3) -> bool:
        """Close a trade via OANDA API with retry for time_stop/kill_switch.

        Retries up to max_retries times with 1s delay between attempts.
        """
        trade = self._trades.get(trade_id)
        if not trade:
            return False

        url = (f"{self._base_url}/v3/accounts/{self._account_id}"
               f"/trades/{trade_id}/close")

        for attempt in range(max_retries):
            try:
                resp = requests.put(url, headers=self._headers,
                                    json={"units": "ALL"}, timeout=5)
                if resp.status_code == 200:
                    self._logger.info(
                        f"Trade closed {trade_id}: {trade.pair} reason={reason.value}")
                    self.remove_trade(trade_id)
                    return True
                else:
                    self._logger.error(
                        f"Close failed for {trade_id}: HTTP {resp.status_code} "
                        f"(attempt {attempt + 1}/{max_retries})")
            except requests.exceptions.RequestException as e:
                self._logger.error(
                    f"Close error for {trade_id}: {e} "
                    f"(attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1)

        self._logger.error(f"All close attempts exhausted for {trade_id}")
        return False
