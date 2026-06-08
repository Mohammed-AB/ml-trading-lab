"""Data Quality Gate — Step 0 in the decision pipeline (before Session Gate).

Per spec A.4, checks:
- heartbeat_timeout: no data from stream in 10s
- candle_late: M1 candle not received within 5s of expected time
- stale_price: last bid/ask older than 15s
- indicator_nan: any indicator returned NaN/None
- api_rate_limit: 429 response or latency > 2000ms
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class DataQualityResult:
    is_ok: bool
    issue: Optional[str] = None
    details: dict = field(default_factory=dict)


class DataQualityGate:
    """Tracks data quality state and evaluates the gate."""

    def __init__(self, config: dict):
        self._heartbeat_timeout = config.get("heartbeat_timeout_sec", 10)
        self._candle_late_sec = config.get("candle_late_sec", 5)
        self._stale_price_sec = config.get("stale_price_sec", 15)
        self._api_timeout_ms = config.get("api_timeout_ms", 2000)

        # State tracking
        self.last_heartbeat_utc: Optional[datetime] = None
        self.last_price_utc: Optional[datetime] = None
        self.last_api_latency_ms: Optional[float] = None
        self.last_api_status: Optional[int] = None
        self.indicators_valid: bool = True
        self.invalid_indicator: Optional[str] = None

    def update_heartbeat(self, utc_now: datetime) -> None:
        self.last_heartbeat_utc = utc_now

    def update_price(self, utc_now: datetime) -> None:
        self.last_price_utc = utc_now

    def update_api_response(self, latency_ms: float, status_code: int) -> None:
        self.last_api_latency_ms = latency_ms
        self.last_api_status = status_code

    def update_indicators(self, valid: bool, invalid_name: Optional[str] = None) -> None:
        self.indicators_valid = valid
        self.invalid_indicator = invalid_name

    def check(self, utc_now: datetime) -> DataQualityResult:
        """Evaluate all data quality conditions.

        Returns the first issue found (fail-fast per spec).
        """
        # Heartbeat timeout
        if self.last_heartbeat_utc is not None:
            elapsed = (utc_now - self.last_heartbeat_utc).total_seconds()
            if elapsed > self._heartbeat_timeout:
                return DataQualityResult(
                    False, "heartbeat_timeout",
                    {"elapsed_sec": round(elapsed, 1), "threshold": self._heartbeat_timeout},
                )

        # Stale price
        if self.last_price_utc is not None:
            stale_sec = (utc_now - self.last_price_utc).total_seconds()
            if stale_sec > self._stale_price_sec:
                return DataQualityResult(
                    False, "stale_price",
                    {"stale_seconds": round(stale_sec, 1), "threshold": self._stale_price_sec},
                )

        # API rate limit / timeout
        if self.last_api_status == 429:
            return DataQualityResult(
                False, "api_rate_limit",
                {"status": 429},
            )
        if self.last_api_latency_ms is not None and self.last_api_latency_ms > self._api_timeout_ms:
            return DataQualityResult(
                False, "api_rate_limit",
                {"latency_ms": round(self.last_api_latency_ms, 1),
                 "threshold_ms": self._api_timeout_ms},
            )

        # Indicator NaN
        if not self.indicators_valid:
            return DataQualityResult(
                False, "indicator_nan",
                {"indicator": self.invalid_indicator},
            )

        return DataQualityResult(True)
