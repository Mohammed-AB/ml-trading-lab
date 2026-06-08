"""Pending Order Manager — Tracks and manages limit order lifecycle.

Handles the gap between Limit order submission and fill/expiry:
- Polls pending orders periodically
- Triggers Market fallback when Limit expires (spec A.3)
- Registers filled trades with TradeManager
- Reconciles local state with broker periodically
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from .executor import Executor, ExecutionResult
from .order_builder import OrderBuilder, OrderSpec, OrderType
from .trade_manager import TradeManager, ManagedTrade
from ..utils.datetime_utils import parse_oanda_timestamp
from .risk_manager import OpenPosition
from ..logger import ScalpLogger
from ..utils.pip_utils import price_to_pips


@dataclass
class PendingOrder:
    """A limit order awaiting fill."""
    order_id: str
    signal_id: str
    pair: str
    direction: str
    units: int
    entry_price: float
    sl_price: float
    tp_price: float
    submitted_at: datetime
    ttl_seconds: int
    atr: float
    spread_at_signal: float
    max_spread: float


class PendingOrderManager:
    """Manages the lifecycle of pending limit orders.

    Usage:
        mgr = PendingOrderManager(executor, order_builder, trade_mgr, logger)
        mgr.track(pending_order)
        # Call periodically:
        mgr.poll_all(utc_now, live_prices)
    """

    def __init__(self, executor: Executor, order_builder: OrderBuilder,
                 trade_mgr: TradeManager, logger: ScalpLogger,
                 poll_interval_sec: int = 10):
        self._executor = executor
        self._builder = order_builder
        self._trade_mgr = trade_mgr
        self._logger = logger
        self._poll_interval = poll_interval_sec
        self._pending: dict[str, PendingOrder] = {}
        self._last_poll: float = 0

    def track(self, order: PendingOrder) -> None:
        """Start tracking a pending limit order."""
        self._pending[order.order_id] = order
        self._logger.info(
            f"Tracking pending order {order.order_id}: "
            f"{order.pair} {order.direction} @ {order.entry_price}")

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def snapshot_for_risk(self) -> list[dict]:
        """Pair+direction rows for Risk Agent dedup (pending limit orders)."""
        return [
            {"pair": po.pair, "direction": po.direction}
            for po in self._pending.values()
        ]

    def poll_all(self, utc_now: datetime,
                 live_prices: dict[str, tuple[float, float]]) -> list[str]:
        """Poll all pending orders and handle state transitions.

        Args:
            utc_now: Current UTC time
            live_prices: {pair: (bid, ask)} for fallback decisions

        Returns:
            List of order_ids that were resolved (filled, expired, fallback).
        """
        now_mono = time.monotonic()
        if now_mono - self._last_poll < self._poll_interval:
            return []
        self._last_poll = now_mono

        resolved = []
        for order_id, po in list(self._pending.items()):
            result = self._check_one(po, utc_now, live_prices)
            if result:
                resolved.append(order_id)
                del self._pending[order_id]

        return resolved

    def _check_one(self, po: PendingOrder, utc_now: datetime,
                   live_prices: dict[str, tuple[float, float]]) -> bool:
        """Check a single pending order. Returns True if resolved."""

        # Check if TTL expired locally (don't wait for broker confirmation)
        elapsed = (utc_now - po.submitted_at).total_seconds()
        if elapsed > po.ttl_seconds + 5:  # +5s grace for network delay
            self._logger.info(
                f"Pending order {po.order_id} TTL expired ({elapsed:.0f}s). "
                f"Attempting market fallback...")
            return self._try_fallback(po, utc_now, live_prices)

        # Poll broker for status
        status = self._executor.check_order_status(po.order_id)

        if status.broker_status == "filled":
            self._logger.info(
                f"Pending order {po.order_id} FILLED at {status.fill_price}")
            self._register_trade(po, status)
            return True

        if status.broker_status == "expired":
            # Double-check: order might have been filled and then disappeared
            # Use check_trade_by_order as safety net before fallback
            trade_data = self._executor.check_trade_by_order(po.order_id, po.pair)
            if trade_data and trade_data.get("id"):
                self._logger.info(
                    f"Pending order {po.order_id} was actually filled "
                    f"(found trade {trade_data['id']}). Registering.")
                fill_price = float(trade_data.get("price", po.entry_price))
                result = ExecutionResult(
                    success=True, trade_id=trade_data["id"],
                    fill_price=fill_price, fill_time=trade_data.get("openTime"),
                    broker_status="filled")
                self._register_trade(po, result)
                return True

            self._logger.info(
                f"Pending order {po.order_id} expired: {status.reject_reason}. "
                f"Attempting market fallback...")
            return self._try_fallback(po, utc_now, live_prices)

        # Still pending
        return False

    def _try_fallback(self, po: PendingOrder, utc_now: datetime,
                      live_prices: dict[str, tuple[float, float]]) -> bool:
        """Attempt market fallback for an expired limit order (spec A.3 step 2).

        Steps:
        1. Re-read broker state — if already FILLED, register and stop.
        2. If order already terminal (GTD expired / cancelled / 404), **do not** PUT cancel
           (avoids noisy OANDA "Order Cancel Reject" when nothing is left to cancel).
        3. If still PENDING, cancel explicitly, then handle ORDER_CANCEL_REJECTED → fill check.
        4. Market fallback when appropriate.
        """
        status = self._executor.check_order_status(po.order_id)
        if status.broker_status == "filled":
            self._logger.info(
                f"Pending order {po.order_id} FILLED before fallback cancel "
                f"(trade {status.trade_id})")
            self._register_trade(po, status)
            return True

        skip_cancel = (
            status.broker_status == "expired"
            or status.reject_reason == "order_not_found"
        )
        if skip_cancel:
            self._logger.info(
                f"Order {po.order_id} already terminal at broker "
                f"({status.reject_reason or 'expired'}); skipping cancel, "
                f"attempting market fallback")

        elif status.broker_status == "error":
            self._logger.warning(
                f"Order status check failed for {po.order_id} "
                f"({status.reject_reason}); attempting cancel anyway")

        if not skip_cancel:
            cancel_result = self._executor.cancel_order(po.order_id)

            if cancel_result["reason"] == "ORDER_CANCEL_REJECTED":
                # Order may have been filled in the interim — check before fallback
                trade_data = self._executor.check_trade_by_order(
                    po.order_id, po.pair)
                if trade_data and trade_data.get("id"):
                    self._logger.info(
                        f"Order {po.order_id} was filled during cancel attempt "
                        f"(trade {trade_data['id']}). Registering instead of fallback.")
                    fill_price = float(trade_data.get("price", po.entry_price))
                    fill_time = trade_data.get("openTime")
                    result = ExecutionResult(
                        success=True, trade_id=trade_data["id"],
                        fill_price=fill_price, fill_time=fill_time,
                        broker_status="filled")
                    self._register_trade(po, result)
                    return True

        return self._execute_market_fallback(po, live_prices)

    def _execute_market_fallback(self, po: PendingOrder,
                                 live_prices: dict[str, tuple[float, float]],
                                 ) -> bool:
        """Submit market fallback after limit path is cleared (or skipped)."""
        prices = live_prices.get(po.pair)
        if not prices:
            self._logger.warning(
                f"No live price for {po.pair} — cannot attempt fallback")
            return True  # Resolved (no fallback possible)

        bid, ask = prices
        current_spread = price_to_pips(ask - bid, po.pair)
        current_mid = (bid + ask) / 2

        fallback = self._builder.build_market_fallback(
            pair=po.pair, direction=po.direction, units=po.units,
            entry_price=po.entry_price, sl_price=po.sl_price,
            tp_price=po.tp_price, current_price=current_mid,
            atr=po.atr, spread_pips=current_spread,
            max_spread_pips=po.max_spread,
            signal_id=po.signal_id + "-fallback",
        )

        if fallback is None:
            self._logger.info(
                f"Market fallback conditions not met for {po.pair} "
                f"(spread={current_spread:.1f}, distance too far)")
            return True  # Resolved (no trade)

        body = self._builder.to_oanda_order(fallback)
        result = self._executor.submit(fallback, body)

        if result.success and result.broker_status == "filled":
            self._logger.info(
                f"Market fallback FILLED for {po.pair} at {result.fill_price}")
            self._register_trade(po, result)
        else:
            self._logger.warning(
                f"Market fallback FAILED for {po.pair}: {result.reject_reason}")

        return True  # Resolved regardless

    def _register_trade(self, po: PendingOrder, result: ExecutionResult) -> None:
        """Register a filled order with TradeManager and log fill."""
        fill_price = result.fill_price or po.entry_price
        try:
            open_time = parse_oanda_timestamp(result.fill_time)
        except (ValueError, TypeError, AttributeError):
            open_time = datetime.now(timezone.utc)
        managed = ManagedTrade(
            trade_id=result.trade_id or po.order_id,
            pair=po.pair,
            direction=po.direction,
            entry_price=fill_price,
            sl_price=po.sl_price,
            tp_price=po.tp_price,
            units=po.units,
            open_time=open_time,
            risk_amount=abs(fill_price - po.sl_price),
        )
        self._trade_mgr.add_trade(managed)

        slippage = price_to_pips(fill_price - po.entry_price, po.pair)
        self._logger.log_trade({
            "trade_id": result.trade_id or po.order_id,
            "decision_log_ref": po.signal_id,
            "pair": po.pair,
            "direction": po.direction,
            "order_type": "FILL",
            "expected_entry_price": po.entry_price,
            "fill_price": fill_price,
            "actual_slippage_pips": round(slippage, 2),
            "spread_at_signal": po.spread_at_signal,
            "order_sent_ts": po.submitted_at.isoformat(),
            "fill_received_ts": result.fill_time,
            "broker_status": "filled",
            "sl_price": po.sl_price,
            "tp_price": po.tp_price,
            "units": po.units,
            "signal_id": po.signal_id,
        })


class BrokerReconciler:
    """Periodically reconciles local trade state with broker.

    Handles:
    - Trades closed by broker (TP/SL hit) that we haven't detected locally
    - Trades opened externally that we need to track
    - Stale local state cleanup
    """

    def __init__(self, executor: Executor, trade_mgr: TradeManager,
                 logger: ScalpLogger, reconcile_interval_sec: int = 30):
        self._executor = executor
        self._trade_mgr = trade_mgr
        self._logger = logger
        self._interval = reconcile_interval_sec
        self._last_reconcile: float = 0

    def reconcile(self, utc_now: datetime) -> dict:
        """Reconcile local state with broker. Returns summary of changes.

        Call periodically (e.g., every 30 seconds).
        """
        now_mono = time.monotonic()
        if now_mono - self._last_reconcile < self._interval:
            return {}
        self._last_reconcile = now_mono

        changes = {"closed_by_broker": [], "added_from_broker": []}

        broker_trades = self._executor.get_open_trades()
        if broker_trades is None:
            return changes

        broker_by_id = {t.get("id"): t for t in broker_trades}
        broker_ids = set(broker_by_id.keys())
        local_ids = {t.trade_id for t in self._trade_mgr.open_trades}

        # Trades in local state but NOT on broker → closed by broker (TP/SL hit)
        for local_id in local_ids - broker_ids:
            trade = self._trade_mgr.remove_trade(local_id)
            if trade:
                self._logger.info(
                    f"Reconciliation: {local_id} ({trade.pair}) closed by broker "
                    f"(TP/SL hit). Removed from local state.")
                changes["closed_by_broker"].append(local_id)
                changes.setdefault("closed_trades", {})[local_id] = trade

        # Trades on broker but NOT in local state → add them for management
        for broker_id in broker_ids - local_ids:
            t = broker_by_id[broker_id]
            try:
                try:
                    open_time = parse_oanda_timestamp(t.get("openTime"))
                except (ValueError, TypeError):
                    open_time = utc_now

                entry = float(t.get("price", 0))
                sl = float(t.get("stopLossOrder", {}).get("price", 0))
                tp = float(t.get("takeProfitOrder", {}).get("price", 0))

                if entry == 0 or sl == 0:
                    continue  # Can't manage without entry/SL

                managed = ManagedTrade(
                    trade_id=broker_id,
                    pair=t.get("instrument", ""),
                    direction="long" if int(t.get("currentUnits", 0)) > 0 else "short",
                    entry_price=entry,
                    sl_price=sl,
                    tp_price=tp,
                    units=abs(int(t.get("currentUnits", 0))),
                    open_time=open_time,
                    risk_amount=abs(entry - sl),
                )
                self._trade_mgr.add_trade(managed)
                self._logger.info(
                    f"Reconciliation: added {broker_id} ({managed.pair} "
                    f"{managed.direction}) from broker")
                changes["added_from_broker"].append(broker_id)
            except (KeyError, ValueError) as e:
                self._logger.error(
                    f"Reconciliation: failed to add {broker_id}: {e}")

        summary_parts = []
        if changes["closed_by_broker"]:
            summary_parts.append(
                f"{len(changes['closed_by_broker'])} closed by broker")
        if changes["added_from_broker"]:
            summary_parts.append(
                f"{len(changes['added_from_broker'])} added from broker")
        if summary_parts:
            self._logger.info(
                f"Reconciliation: {', '.join(summary_parts)}, "
                f"{len(self._trade_mgr.open_trades)} total open")

        return changes
