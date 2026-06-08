"""Execution Layer — Step 10 in the decision pipeline.

Submits orders to OANDA v20 REST API and handles responses.
Per spec A.6:
- Each signal has a unique signal_id (UUID) for idempotency
- Checks for duplicate signal_id before sending
- Handles fill, reject, and timeout responses
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from .order_builder import OrderSpec, OrderType
from ..utils.pip_utils import price_to_pips
from ..logger import ScalpLogger


def parse_oanda_decimal(value: object) -> Optional[float]:
    """Parse OANDA numeric fields that may be str, int, float, or missing.

    Returns None for empty/invalid values. Never invents a default.
    """
    if value is None or type(value) is bool:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def parse_account_nav(account: Optional[dict]) -> Optional[float]:
    """Read usable equity for sizing: prefer NAV, then balance.

    Returns None if neither field parses to a positive number — never
    default to 10k (that oversizes small accounts when margin is real).
    """
    if not account:
        return None
    for key in ("NAV", "balance"):
        v = parse_oanda_decimal(account.get(key))
        if v is not None and v > 0:
            return v
    return None


@dataclass
class ExecutionResult:
    success: bool
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    fill_price: Optional[float] = None
    fill_time: Optional[str] = None
    actual_slippage_pips: Optional[float] = None
    broker_status: str = "unknown"     # filled, rejected, expired, error
    reject_reason: Optional[str] = None
    e2e_latency_ms: int = 0


class Executor:
    """Submits orders to OANDA and processes responses.

    Usage:
        executor = Executor(base_url, api_token, account_id, logger)
        result = executor.submit(order_spec)
    """

    def __init__(self, base_url: str, api_token: str, account_id: str,
                 logger: ScalpLogger):
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._logger = logger
        self._recent_signal_ids: OrderedDict[str, float] = OrderedDict()
        self._max_signal_ids = 200

    def submit(self, order: OrderSpec, from_builder_body: dict) -> ExecutionResult:
        """Submit an order to OANDA.

        Args:
            order: The OrderSpec with all details
            from_builder_body: The OANDA-formatted order body from OrderBuilder.to_oanda_order()

        Returns:
            ExecutionResult with fill details or rejection info.
        """
        # Idempotency check (spec A.6)
        if order.signal_id in self._recent_signal_ids:
            self._logger.warning(f"Duplicate signal_id detected: {order.signal_id}")
            return ExecutionResult(
                success=False, broker_status="rejected",
                reject_reason="duplicate_signal_id")

        self._recent_signal_ids[order.signal_id] = time.monotonic()
        # Trim oldest entries to prevent memory growth (ordered by insertion)
        while len(self._recent_signal_ids) > self._max_signal_ids:
            self._recent_signal_ids.popitem(last=False)

        url = f"{self._base_url}/v3/accounts/{self._account_id}/orders"
        start_time = time.monotonic()

        try:
            response = requests.post(
                url, headers=self._headers, json=from_builder_body, timeout=5)
            latency_ms = int((time.monotonic() - start_time) * 1000)

            if response.status_code == 201:
                return self._handle_success(response.json(), order, latency_ms)
            elif response.status_code == 400:
                return self._handle_rejection(response.json(), latency_ms)
            elif response.status_code == 404:
                return ExecutionResult(
                    success=False, broker_status="rejected",
                    reject_reason="account_not_found", e2e_latency_ms=latency_ms)
            else:
                return ExecutionResult(
                    success=False, broker_status="error",
                    reject_reason=f"http_{response.status_code}",
                    e2e_latency_ms=latency_ms)

        except requests.exceptions.Timeout:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            return ExecutionResult(
                success=False, broker_status="error",
                reject_reason="timeout", e2e_latency_ms=latency_ms)
        except requests.exceptions.RequestException as e:
            latency_ms = int((time.monotonic() - start_time) * 1000)
            return ExecutionResult(
                success=False, broker_status="error",
                reject_reason=f"network_error:{type(e).__name__}",
                e2e_latency_ms=latency_ms)

    def _handle_success(self, data: dict, order: OrderSpec,
                        latency_ms: int) -> ExecutionResult:
        """Parse a successful order response."""
        # Market order → immediate fill in orderFillTransaction
        fill_tx = data.get("orderFillTransaction")
        if fill_tx:
            fill_price = float(fill_tx.get("price", 0))
            expected = order.price if order.price else fill_price
            slippage = price_to_pips(fill_price - expected, order.pair) if expected else 0

            trade_ids = fill_tx.get("tradeOpened", {}).get("tradeID")
            return ExecutionResult(
                success=True,
                order_id=fill_tx.get("orderID"),
                trade_id=trade_ids,
                fill_price=fill_price,
                fill_time=fill_tx.get("time"),
                actual_slippage_pips=round(slippage, 2),
                broker_status="filled",
                e2e_latency_ms=latency_ms,
            )

        # Limit order → pending (orderCreateTransaction)
        create_tx = data.get("orderCreateTransaction")
        if create_tx:
            return ExecutionResult(
                success=True,
                order_id=create_tx.get("id"),
                broker_status="pending",
                e2e_latency_ms=latency_ms,
            )

        # Rejection within 201 (rare but possible)
        cancel_tx = data.get("orderCancelTransaction")
        if cancel_tx:
            return ExecutionResult(
                success=False,
                broker_status="rejected",
                reject_reason=cancel_tx.get("reason", "unknown"),
                e2e_latency_ms=latency_ms,
            )

        return ExecutionResult(
            success=True, broker_status="unknown", e2e_latency_ms=latency_ms)

    def _handle_rejection(self, data: dict, latency_ms: int) -> ExecutionResult:
        """Parse a 400 rejection response."""
        reject_tx = data.get("orderRejectTransaction", {})
        reason = reject_tx.get("rejectReason", "unknown")
        return ExecutionResult(
            success=False, broker_status="rejected",
            reject_reason=reason, e2e_latency_ms=latency_ms)

    def check_order_status(self, order_id: str) -> ExecutionResult:
        """Poll the status of a pending order (for Limit order lifecycle).

        Returns:
            ExecutionResult with updated status (filled/expired/pending/cancelled).
        """
        url = (f"{self._base_url}/v3/accounts/{self._account_id}"
               f"/orders/{order_id}")
        start_time = time.monotonic()
        try:
            resp = requests.get(url, headers=self._headers, timeout=5)
            latency = int((time.monotonic() - start_time) * 1000)

            if resp.status_code == 200:
                order_data = resp.json().get("order", {})
                state = order_data.get("state", "PENDING")
                if state == "FILLED":
                    # OANDA: fillingTransactionID links to the fill transaction
                    fill_tx_id = order_data.get("fillingTransactionID")
                    fill_price = float(order_data.get("price", 0))
                    trade_id = order_data.get("tradeOpenedID")
                    return ExecutionResult(
                        success=True, order_id=order_id,
                        trade_id=trade_id,
                        fill_price=fill_price,
                        fill_time=order_data.get("filledTime"),
                        broker_status="filled", e2e_latency_ms=latency)
                elif state in ("CANCELLED", "TRIGGERED"):
                    return ExecutionResult(
                        success=False, order_id=order_id,
                        broker_status="expired",
                        reject_reason=order_data.get("cancelReason", "expired"),
                        e2e_latency_ms=latency)
                else:
                    return ExecutionResult(
                        success=True, order_id=order_id,
                        broker_status="pending", e2e_latency_ms=latency)
            elif resp.status_code == 404:
                # Order not found — may have been filled and converted to trade
                return ExecutionResult(
                    success=False, order_id=order_id,
                    broker_status="expired",
                    reject_reason="order_not_found", e2e_latency_ms=latency)
            else:
                return ExecutionResult(
                    success=False, order_id=order_id,
                    broker_status="error",
                    reject_reason=f"http_{resp.status_code}",
                    e2e_latency_ms=latency)
        except requests.exceptions.RequestException as e:
            latency = int((time.monotonic() - start_time) * 1000)
            self._logger.error(f"Order status check failed for {order_id}: {e}")
            return ExecutionResult(
                success=False, order_id=order_id,
                broker_status="error",
                reject_reason=f"network_error:{type(e).__name__}",
                e2e_latency_ms=latency)

    def check_trade_by_order(self, order_id: str, pair: str) -> Optional[dict]:
        """Check if a pending order resulted in a trade fill.

        Uses the order endpoint first, then falls back to matching by pair
        in open trades. Returns trade details dict or None if not found.
        """
        try:
            # First try: check order state directly
            url = (f"{self._base_url}/v3/accounts/{self._account_id}"
                   f"/orders/{order_id}")
            resp = requests.get(url, headers=self._headers, timeout=5)

            if resp.status_code == 200:
                order_data = resp.json().get("order", {})
                if order_data.get("state") == "FILLED":
                    trade_id = order_data.get("tradeOpenedID")
                    if trade_id:
                        # Fetch the specific trade
                        trade_url = (f"{self._base_url}/v3/accounts/"
                                     f"{self._account_id}/trades/{trade_id}")
                        trade_resp = requests.get(
                            trade_url, headers=self._headers, timeout=5)
                        if trade_resp.status_code == 200:
                            return trade_resp.json().get("trade")
                return None

            if resp.status_code == 404:
                # Order gone — search open trades matching this pair
                trades_url = (f"{self._base_url}/v3/accounts/"
                              f"{self._account_id}/openTrades")
                trades_resp = requests.get(
                    trades_url, headers=self._headers, timeout=5)
                if trades_resp.status_code == 200:
                    for trade in trades_resp.json().get("trades", []):
                        if trade.get("instrument") == pair:
                            return trade
            return None
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Trade check failed for order {order_id}: {e}")
            return None

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a single pending order. Returns {"cancelled": bool, "reason": str}."""
        url = (f"{self._base_url}/v3/accounts/{self._account_id}"
               f"/orders/{order_id}/cancel")
        try:
            resp = requests.put(url, headers=self._headers, timeout=5)
            if resp.status_code == 200:
                return {"cancelled": True, "reason": "success"}
            elif resp.status_code == 404:
                return {"cancelled": False, "reason": "not_found"}
            else:
                body = {}
                if resp.headers.get("content-type", "").startswith("application/json"):
                    body = resp.json()
                reject_reason = (
                    body.get("orderCancelRejectTransaction", {})
                    .get("rejectReason", f"http_{resp.status_code}")
                )
                return {"cancelled": False, "reason": reject_reason}
        except requests.exceptions.RequestException as e:
            self._logger.warning(f"Cancel order {order_id} failed: {e}")
            return {"cancelled": False, "reason": f"network_error:{type(e).__name__}"}

    def cancel_pending_orders(self) -> list[str]:
        """Cancel all pending orders (for graceful shutdown per spec A.6).

        Returns list of cancelled order IDs.
        """
        cancelled = []
        try:
            url = f"{self._base_url}/v3/accounts/{self._account_id}/pendingOrders"
            resp = requests.get(url, headers=self._headers, timeout=5)
            if resp.status_code != 200:
                return cancelled

            orders = resp.json().get("orders", [])
            # Skip SL/TP orders — they protect open trades and must survive restarts
            _protected = {"STOP_LOSS", "TAKE_PROFIT", "TRAILING_STOP_LOSS"}
            for order in orders:
                order_id = order.get("id")
                if order_id and order.get("type") not in _protected:
                    cancel_url = (
                        f"{self._base_url}/v3/accounts/{self._account_id}"
                        f"/orders/{order_id}/cancel"
                    )
                    cancel_resp = requests.put(
                        cancel_url, headers=self._headers, timeout=5)
                    if cancel_resp.status_code == 200:
                        cancelled.append(order_id)
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Failed to cancel pending orders: {e}")

        return cancelled

    def get_open_trades(self) -> list[dict]:
        """Fetch currently open trades (for state recovery per spec A.6)."""
        try:
            url = f"{self._base_url}/v3/accounts/{self._account_id}/openTrades"
            resp = requests.get(url, headers=self._headers, timeout=5)
            if resp.status_code == 200:
                return resp.json().get("trades", [])
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Failed to fetch open trades: {e}")
        return []

    def get_account_details(self) -> Optional[dict]:
        """Fetch account NAV, margin, and balance."""
        try:
            url = f"{self._base_url}/v3/accounts/{self._account_id}/summary"
            resp = requests.get(url, headers=self._headers, timeout=5)
            if resp.status_code == 200:
                return resp.json().get("account", {})
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Failed to fetch account details: {e}")
        return None
