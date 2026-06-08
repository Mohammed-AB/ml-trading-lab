"""Spread Filter — Step 3 in the decision pipeline.

Checks if the current bid/ask spread is within acceptable limits.
Per spec: spread_pips <= max_spread_config[pair].
"""

from dataclasses import dataclass

from ..utils.pip_utils import price_to_pips


@dataclass
class SpreadResult:
    is_ok: bool
    spread_pips: float
    max_allowed: float


def check_spread(bid: float, ask: float, pair: str,
                 max_spread_pips: float) -> SpreadResult:
    """Check if current spread is within limits.

    Args:
        bid: Current bid price
        ask: Current ask price
        pair: Instrument pair (e.g., "EUR_USD")
        max_spread_pips: Maximum allowed spread in pips for this pair

    Returns:
        SpreadResult with is_ok flag and actual spread in pips
    """
    spread_pips = price_to_pips(ask - bid, pair)
    return SpreadResult(
        is_ok=spread_pips <= max_spread_pips,
        spread_pips=round(spread_pips, 2),
        max_allowed=max_spread_pips,
    )
