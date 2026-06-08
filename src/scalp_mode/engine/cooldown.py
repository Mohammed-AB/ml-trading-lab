"""Cooldown Manager — Step 5 in the decision pipeline.

Per spec:
- Same pair + same direction: 10 minutes cooldown after last trade
- 3 consecutive losses: 60 minutes circuit breaker (all pairs)
- Max trades per hour: 3/pair + 6 total
- Daily loss limit: -1.0% NAV → stop for the day
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional


@dataclass
class TradeRecord:
    """Minimal trade record for cooldown tracking."""
    pair: str
    direction: str  # "long" or "short"
    timestamp_utc: datetime
    pnl_pct: Optional[float] = None  # None = still open


@dataclass
class CooldownResult:
    is_ok: bool
    reason: Optional[str] = None
    details: Optional[dict] = None


class CooldownManager:
    """Tracks trade history and enforces cooldown/circuit rules.

    Usage:
        mgr = CooldownManager(config.risk)
        mgr.record_trade(TradeRecord(...))
        result = mgr.check("EUR_USD", "long", utc_now)
    """

    def __init__(self, risk_config: dict, alert_manager=None):
        self._cooldown_min = risk_config.get("cooldown_same_pair_dir_min", 10)
        self._consec_loss_circuit = risk_config.get("consec_loss_circuit", 3)
        self._circuit_cooldown_min = risk_config.get("cooldown_minutes", 60)
        self._trades_per_hour_pair = risk_config.get("trades_per_hour_pair", 3)
        self._trades_per_hour_total = risk_config.get("trades_per_hour_total", 6)
        self._daily_loss_limit = risk_config.get("daily_loss", 0.01)
        self._alert = alert_manager

        self._trades: list[TradeRecord] = []
        self._circuit_breaker_until: Optional[datetime] = None

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed or opened trade."""
        self._trades.append(trade)
        self._check_consecutive_losses()

    def _check_consecutive_losses(self) -> None:
        """Activate circuit breaker if N consecutive losses."""
        closed = [t for t in self._trades if t.pnl_pct is not None]
        if len(closed) < self._consec_loss_circuit:
            return
        recent = closed[-self._consec_loss_circuit:]
        if all(t.pnl_pct < 0 for t in recent):
            last_time = recent[-1].timestamp_utc
            self._circuit_breaker_until = last_time + timedelta(
                minutes=self._circuit_cooldown_min
            )
            if self._alert:
                self._alert.alert_consecutive_losses(
                    self._consec_loss_circuit, self._circuit_cooldown_min)

    def check(self, pair: str, direction: str, utc_now: datetime) -> CooldownResult:
        """Check all cooldown conditions.

        Returns:
            CooldownResult — is_ok=True if no cooldown active.
        """
        # Circuit breaker (consecutive losses)
        if self._circuit_breaker_until and utc_now < self._circuit_breaker_until:
            remaining = (self._circuit_breaker_until - utc_now).total_seconds() / 60
            return CooldownResult(
                False, "consec_loss_circuit",
                {"remaining_min": round(remaining, 1),
                 "until": self._circuit_breaker_until.isoformat()},
            )

        # Daily loss limit
        daily_pnl = self._daily_pnl(utc_now)
        if daily_pnl <= -self._daily_loss_limit:
            if self._alert:
                self._alert.alert_daily_loss(daily_pnl, self._daily_loss_limit)
            return CooldownResult(
                False, "daily_loss_limit",
                {"daily_pnl_pct": round(daily_pnl, 4),
                 "limit": self._daily_loss_limit},
            )

        # Same pair + direction cooldown
        last_same = self._last_trade_same_pair_dir(pair, direction)
        if last_same:
            elapsed = (utc_now - last_same.timestamp_utc).total_seconds() / 60
            if elapsed < self._cooldown_min:
                return CooldownResult(
                    False, "cooldown_active",
                    {"elapsed_min": round(elapsed, 1),
                     "required_min": self._cooldown_min},
                )

        # Trades per hour (pair)
        pair_trades_1h = self._count_trades_in_window(utc_now, 60, pair=pair)
        if pair_trades_1h >= self._trades_per_hour_pair:
            return CooldownResult(
                False, "max_trades_per_hour_pair",
                {"count": pair_trades_1h, "limit": self._trades_per_hour_pair},
            )

        # Trades per hour (total)
        total_trades_1h = self._count_trades_in_window(utc_now, 60)
        if total_trades_1h >= self._trades_per_hour_total:
            return CooldownResult(
                False, "max_trades_per_hour_total",
                {"count": total_trades_1h, "limit": self._trades_per_hour_total},
            )

        return CooldownResult(True)

    def _last_trade_same_pair_dir(self, pair: str, direction: str) -> Optional[TradeRecord]:
        """Find the most recent trade on the same pair and direction."""
        for trade in reversed(self._trades):
            if trade.pair == pair and trade.direction == direction:
                return trade
        return None

    def _count_trades_in_window(self, utc_now: datetime, minutes: int,
                                pair: Optional[str] = None) -> int:
        """Count trades within the last N minutes."""
        cutoff = utc_now - timedelta(minutes=minutes)
        count = 0
        for trade in self._trades:
            if trade.timestamp_utc >= cutoff:
                if pair is None or trade.pair == pair:
                    count += 1
        return count

    def _daily_pnl(self, utc_now: datetime) -> float:
        """Sum of PnL % for all closed trades today (UTC)."""
        today_start = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        total = 0.0
        for trade in self._trades:
            if trade.timestamp_utc >= today_start and trade.pnl_pct is not None:
                total += trade.pnl_pct
        return total

    def reset_daily(self) -> None:
        """Clear state for a new trading day."""
        self._circuit_breaker_until = None

    @property
    def circuit_breaker_active(self) -> bool:
        return (self._circuit_breaker_until is not None
                and datetime.now(timezone.utc) < self._circuit_breaker_until)
