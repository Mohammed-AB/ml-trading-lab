"""Risk Manager — Step 8 in the decision pipeline.

Per spec 4.2:
- Position sizing: floor((NAV * risk_pct) / (stop_pips * pip_value))
- Max margin usage: 8% of NAV
- Max concurrent trades: 2
- Correlation guard: no EUR/USD + GBP/USD same direction simultaneously (r > 0.85)
"""

from dataclasses import dataclass
from typing import Optional

from ..utils.pip_utils import pip_value, pip_value_in_account_ccy, price_to_pips


def notional_usd_per_base_unit(pair: str, mid: float, rates: Optional[dict]) -> float:
    """USD notional per 1 base-currency unit (OANDA: units are base currency).

    Used for margin ≈ notional / leverage. The old ``pip * 100`` estimate was
    not comparable to broker margin and caused ~50k+ unit orders on small accounts.
    """
    rates = rates or {}
    parts = pair.upper().split("_")
    if len(parts) != 2 or mid <= 0:
        return max(abs(mid), 1e-9)
    base, quote = parts[0], parts[1]
    if base == "USD":
        return 1.0
    if quote == "USD":
        return float(mid)
    if base == "EUR" and quote == "GBP":
        gbp_usd = float(rates.get("GBP_USD") or 1.25)
        return float(mid) * gbp_usd
    if quote == "GBP":
        gbp_usd = float(rates.get("GBP_USD") or 1.25)
        return float(mid) * gbp_usd
    return float(mid)


# High-correlation pairs that cannot be open in the same direction simultaneously
CORRELATED_PAIRS = {
    frozenset({"EUR_USD", "GBP_USD"}),
    frozenset({"EUR_USD", "EUR_GBP"}),
}


@dataclass
class RiskResult:
    approved: bool
    units: int = 0
    reject_reason: Optional[str] = None
    details: Optional[dict] = None


@dataclass
class OpenPosition:
    pair: str
    direction: str  # "long" or "short"
    units: int
    margin_used: float


class RiskManager:
    """Evaluates risk constraints and computes position size.

    Usage:
        rm = RiskManager(config.risk)
        result = rm.evaluate(
            pair="EUR_USD", direction="long", stop_pips=4.0,
            nav=10000, margin_available=9500, open_positions=[...]
        )
    """

    def __init__(self, risk_config: dict):
        self._risk_pct = risk_config["risk_pct"]
        self._max_concurrent = risk_config["max_concurrent"]
        self._max_margin_pct = risk_config.get("max_margin_pct", 0.08)
        self._account_ccy = risk_config.get("account_currency", "USD")
        self._leverage = float(risk_config.get("leverage", 50))
        self._margin_cap_safety = float(risk_config.get("margin_cap_safety", 0.90))
        self._live_rates: dict = {}

    def update_rates(self, rates: dict) -> None:
        """Update live exchange rates for account currency conversion.

        Call this periodically from the Paper/Live loop with current rates.
        e.g. {"GBP_USD": 1.25, "USD_JPY": 150.0}
        """
        self._live_rates = rates

    def evaluate(self, pair: str, direction: str, stop_pips: float,
                 nav: float, margin_available: float,
                 open_positions: list[OpenPosition],
                 live_rates: dict = None,
                 mid_price: Optional[float] = None) -> RiskResult:
        """Evaluate risk and compute position size.

        Args:
            pair: Instrument
            direction: "long" or "short"
            stop_pips: Stop loss distance in pips
            nav: Current Net Asset Value (in account currency)
            margin_available: Available margin from broker
            open_positions: Currently open positions
            live_rates: Dict of live rates for currency conversion
                        e.g. {"GBP_USD": 1.25, "USD_JPY": 150.0}
            mid_price: Bid/ask mid — required for realistic margin when
                       ``margin_available > 0``. If omitted, a rough default
                       is used (prefer passing from the pipeline).

        Returns:
            RiskResult with approval status and computed units.
        """
        # Max concurrent trades
        if len(open_positions) >= self._max_concurrent:
            return RiskResult(False, reject_reason="max_concurrent_reached",
                              details={"open": len(open_positions),
                                       "limit": self._max_concurrent})

        # Correlation guard
        for pos in open_positions:
            if pos.direction == direction:
                pair_set = frozenset({pair, pos.pair})
                if pair_set in CORRELATED_PAIRS:
                    return RiskResult(
                        False, reject_reason="correlation_guard",
                        details={"existing": pos.pair, "new": pair,
                                 "direction": direction})

        # Duplicate pair/direction check (per spec A.6: no duplicate orders)
        for pos in open_positions:
            if pos.pair == pair and pos.direction == direction:
                return RiskResult(False, reject_reason="duplicate_position",
                                  details={"pair": pair, "direction": direction})

        # Position sizing: floor((NAV * risk_pct) / (stop_pips * pip_value_in_acct_ccy))
        if stop_pips <= 0:
            return RiskResult(False, reject_reason="invalid_stop",
                              details={"stop_pips": stop_pips})

        pip_val = pip_value(pair)
        rates = live_rates or self._live_rates
        pip_val_acct = pip_value_in_account_ccy(
            pair, self._account_ccy, rates)
        risk_amount = nav * self._risk_pct
        units = int(risk_amount / (stop_pips * pip_val_acct))

        if units <= 0:
            return RiskResult(False, reject_reason="units_too_small",
                              details={"risk_amount": risk_amount,
                                       "stop_pips": stop_pips})

        total_margin_used = sum(p.margin_used for p in open_positions)

        # Margin check: use broker-reported margin_available when > 0 (live/paper),
        # fall back to estimate based on NAV * max_margin_pct (backtest).
        if margin_available > 0:
            mid = mid_price
            if mid is None or mid <= 0:
                pu = pair.upper()
                if pu.startswith("USD_") and "JPY" in pu:
                    mid = float(rates.get("USD_JPY", 150.0))
                else:
                    mid = 1.10
            nu = notional_usd_per_base_unit(pair, mid, rates)
            lev = max(self._leverage, 1.0)
            cap = margin_available * self._margin_cap_safety
            # OANDA-style: margin ≈ notional (USD) / leverage for USD accounts
            est_margin = abs(float(units)) * nu / lev
            if est_margin > cap and nu > 0:
                max_units = int(cap * lev / nu)
                if max_units <= 0:
                    return RiskResult(False, reject_reason="margin_exceeded",
                                      details={"estimated_margin": round(est_margin, 2),
                                               "available": round(margin_available, 2),
                                               "notional_per_unit": round(nu, 6)})
                units = max_units
        else:
            # Backtest: use NAV-based estimate (legacy pip heuristic)
            estimated_margin = units * pip_val * 100
            margin_limit = self._max_margin_pct * nav
            if (total_margin_used + estimated_margin) > margin_limit:
                available = margin_limit - total_margin_used
                if available <= 0:
                    return RiskResult(False, reject_reason="margin_exceeded",
                                      details={"total_margin": total_margin_used,
                                               "limit": margin_limit})
                units = int(available / (pip_val * 100))
                if units <= 0:
                    return RiskResult(False, reject_reason="margin_exceeded",
                                      details={"available_margin": available})

        return RiskResult(
            approved=True, units=units,
            details={"risk_amount": round(risk_amount, 2),
                     "stop_pips": stop_pips, "units": units,
                     "account_ccy": self._account_ccy,
                     "pip_val_raw": pip_val,
                     "pip_val_acct": round(pip_val_acct, 8)})
