"""Tiered risk sizing for small accounts.

Base risk 2%, boost to 3% on high-conviction setups, reduce to 1% on
Friday or after a losing day. Hard daily loss cap at 6%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class DailySizingState:
    """Tracks intraday P&L for daily loss cap."""
    current_date: date = field(default_factory=lambda: date.today())
    daily_pnl_pct: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0


CONVICTION_PAIRS = {"NZD_USD", "AUD_USD", "USD_JPY", "EUR_USD"}


class DynamicSizer:
    """Computes risk_pct per trade based on context."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._base_risk = float(cfg.get("base_risk_pct", 2.0))
        self._boost_risk = float(cfg.get("boost_risk_pct", 3.0))
        self._reduced_risk = float(cfg.get("reduced_risk_pct", 1.0))
        self._daily_loss_cap = float(cfg.get("daily_loss_cap_pct", 6.0))
        self._boost_adx_min = float(cfg.get("boost_adx_min", 35))
        self._conviction_pairs = set(cfg.get("conviction_pairs", CONVICTION_PAIRS))
        self._reduce_dow = set(cfg.get("reduce_dow", [4]))  # Friday
        self._state = DailySizingState()

    def _reset_if_new_day(self, utc_now: datetime) -> None:
        today = utc_now.date()
        if today != self._state.current_date:
            self._state = DailySizingState(current_date=today)

    def compute_risk_pct(
        self,
        pair: str,
        adx: float,
        utc_now: datetime,
    ) -> float:
        """Return risk percentage for this trade (0.0 = skip)."""
        self._reset_if_new_day(utc_now)

        if self._state.daily_pnl_pct <= -self._daily_loss_cap:
            return 0.0

        dow = utc_now.weekday()

        if dow in self._reduce_dow:
            return self._reduced_risk

        if self._state.daily_pnl_pct < -2.0:
            return self._reduced_risk

        if adx >= self._boost_adx_min and pair in self._conviction_pairs:
            return self._boost_risk

        return self._base_risk

    def record_trade_result(self, pnl_pct: float) -> None:
        """Update daily state after a trade closes."""
        self._state.daily_pnl_pct += pnl_pct
        self._state.trades_today += 1
        if pnl_pct > 0:
            self._state.wins_today += 1
        elif pnl_pct < 0:
            self._state.losses_today += 1

    @property
    def daily_pnl_pct(self) -> float:
        return self._state.daily_pnl_pct

    @property
    def is_daily_capped(self) -> bool:
        return self._state.daily_pnl_pct <= -self._daily_loss_cap

    def summary(self) -> dict:
        s = self._state
        return {
            "date": str(s.current_date),
            "daily_pnl_pct": round(s.daily_pnl_pct, 2),
            "trades": s.trades_today,
            "wins": s.wins_today,
            "losses": s.losses_today,
            "capped": self.is_daily_capped,
        }
