"""Order Builder — Step 9 in the decision pipeline.

Per spec A.3:
  1. Primary: Limit order at breakout_level, GTD = 3 minutes
  2. Fallback: Market + priceBound if Limit expires and price still valid
  3. If fallback fails: NO_TRADE + 2 min cooldown

Builds the OANDA-compatible order object with SL/TP/priceBound/TTL.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from ..utils.pip_utils import pips_to_price, round_price


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class OrderSpec:
    """Complete order specification ready for broker submission."""
    order_type: OrderType
    pair: str
    direction: str          # "long" or "short"
    units: int              # Positive for long, negative for short
    price: Optional[float]  # Entry price (Limit) or None (Market)
    price_bound: Optional[float]  # Max slippage bound
    sl_price: float
    tp_price: float
    ttl_seconds: int        # Time-to-live for the order
    expire_time: Optional[str]  # ISO format GTD time
    signal_id: str          # UUID for idempotency (spec A.6)


class OrderBuilder:
    """Builds order specifications from trigger signals.

    Usage:
        builder = OrderBuilder(config.orders)
        order = builder.build_limit(pair, direction, units, entry, sl, tp, signal_id, utc_now)
        fallback = builder.build_market_fallback(pair, direction, units, entry, sl, tp,
                                                  spread_pips, max_spread, signal_id)
    """

    def __init__(self, order_config: dict):
        self._limit_ttl = order_config.get("limit_ttl_seconds", 180)
        self._fallback_market = order_config.get("fallback_market", True)
        self._fallback_max_atr_distance = order_config.get("fallback_max_atr_distance", 0.3)
        self._fallback_cooldown_min = order_config.get("fallback_cooldown_min", 2)
        self._slippage_pips = order_config.get("price_bound_slippage", 0.2)
        self._primary_order_type = order_config.get("primary_order_type", "LIMIT")

    def build_limit(self, pair: str, direction: str, units: int,
                    entry_price: float, sl_price: float, tp_price: float,
                    signal_id: str, utc_now: datetime) -> OrderSpec:
        """Build a Limit order (primary path per spec A.3 step 1).

        Args:
            pair: Instrument
            direction: "long" or "short"
            units: Position size (always positive, sign applied here)
            entry_price: Limit price at breakout_level
            sl_price: Stop loss price
            tp_price: Take profit price
            signal_id: Unique signal ID for idempotency
            utc_now: Current time for GTD calculation
        """
        signed_units = units if direction == "long" else -units
        expire_time = (utc_now + timedelta(seconds=self._limit_ttl)).strftime(
            "%Y-%m-%dT%H:%M:%S.000000000Z"
        )

        return OrderSpec(
            order_type=OrderType.LIMIT,
            pair=pair,
            direction=direction,
            units=signed_units,
            price=round_price(entry_price, pair),
            price_bound=None,  # Limit orders don't need priceBound
            sl_price=round_price(sl_price, pair),
            tp_price=round_price(tp_price, pair),
            ttl_seconds=self._limit_ttl,
            expire_time=expire_time,
            signal_id=signal_id,
        )

    def build_market(self, pair: str, direction: str, units: int,
                    current_price: float, sl_price: float, tp_price: float,
                    signal_id: str) -> OrderSpec:
        """Build a Market order as primary entry (immediate fill)."""
        signed_units = units if direction == "long" else -units
        slippage_price = pips_to_price(self._slippage_pips, pair)
        if direction == "long":
            price_bound = round_price(current_price + slippage_price, pair)
        else:
            price_bound = round_price(current_price - slippage_price, pair)

        return OrderSpec(
            order_type=OrderType.MARKET,
            pair=pair,
            direction=direction,
            units=signed_units,
            price=None,
            price_bound=price_bound,
            sl_price=round_price(sl_price, pair),
            tp_price=round_price(tp_price, pair),
            ttl_seconds=0,
            expire_time=None,
            signal_id=signal_id,
        )

    @property
    def use_market_primary(self) -> bool:
        return self._primary_order_type.upper() == "MARKET"

    def build_market_fallback(self, pair: str, direction: str, units: int,
                              entry_price: float, sl_price: float, tp_price: float,
                              current_price: float, atr: float,
                              spread_pips: float, max_spread_pips: float,
                              signal_id: str) -> Optional[OrderSpec]:
        """Build a Market fallback order (spec A.3 step 2).

        Only if:
        - spread <= max
        - current price within 0.3 * ATR of breakout_level
        - priceBound = entry ± max_slippage

        Returns None if conditions not met.
        """
        if not self._fallback_market:
            return None

        # Check spread
        if spread_pips > max_spread_pips:
            return None

        # Check price distance
        distance = abs(current_price - entry_price)
        if distance > self._fallback_max_atr_distance * atr:
            return None

        # Compute priceBound
        slippage_price = pips_to_price(self._slippage_pips, pair)
        if direction == "long":
            price_bound = round_price(current_price + slippage_price, pair)
        else:
            price_bound = round_price(current_price - slippage_price, pair)

        signed_units = units if direction == "long" else -units

        return OrderSpec(
            order_type=OrderType.MARKET,
            pair=pair,
            direction=direction,
            units=signed_units,
            price=None,
            price_bound=price_bound,
            sl_price=round_price(sl_price, pair),
            tp_price=round_price(tp_price, pair),
            ttl_seconds=0,
            expire_time=None,
            signal_id=signal_id,
        )

    def to_oanda_order(self, order: OrderSpec) -> dict:
        """Convert OrderSpec to OANDA v20 API order body."""
        if order.order_type == OrderType.LIMIT:
            body = {
                "type": "LIMIT",
                "instrument": order.pair,
                "units": str(order.units),
                "price": str(order.price),
                "timeInForce": "GTD",
                "gtdTime": order.expire_time,
                "stopLossOnFill": {
                    "price": str(order.sl_price),
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": str(order.tp_price),
                },
            }
        else:
            body = {
                "type": "MARKET",
                "instrument": order.pair,
                "units": str(order.units),
                "stopLossOnFill": {
                    "price": str(order.sl_price),
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": str(order.tp_price),
                },
            }

        return {"order": body}
