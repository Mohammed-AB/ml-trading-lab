"""Multi-timeframe context — H1 and D1 directional bias + volatility regime.

Adds higher-timeframe alignment for the Strategy Agent. Not a full regime
engine — just a compact bias tag and an ATR percentile so the bot knows:
- Is H1 trend up / down / flat?
- Is D1 trend up / down / flat?
- Is M5 volatility in the bottom / middle / top percentile of its history?

Called from the Orchestrator once per cycle per pair, results cached for
15 minutes to save REST calls.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd


Bias = Literal["up", "down", "flat"]
VolBucket = Literal["low", "mid", "high"]


@dataclass
class MTFSnapshot:
    pair: str
    h1_bias: Bias
    h1_slope_pct: float
    d1_bias: Bias
    d1_slope_pct: float
    m5_vol_pct: float  # ATR percentile 0..1
    m5_vol_bucket: VolBucket
    updated_at: datetime


class MTFContext:
    """Fetches and caches H1/D1 bias + M5 vol percentile per pair."""

    def __init__(self, price_feeder, cache_ttl_sec: int = 900):
        self._feeder = price_feeder
        self._ttl = cache_ttl_sec
        self._cache: dict[str, MTFSnapshot] = {}

    def get(self, pair: str, utc_now: datetime) -> Optional[MTFSnapshot]:
        cached = self._cache.get(pair)
        if cached and (utc_now - cached.updated_at).total_seconds() < self._ttl:
            return cached
        try:
            snap = self._build(pair, utc_now)
            if snap:
                self._cache[pair] = snap
            return snap
        except Exception:
            return cached  # stale is fine — better than None

    def _build(self, pair: str, utc_now: datetime) -> Optional[MTFSnapshot]:
        # Fetch 50 H1 candles and 50 D candles
        h1_candles, _ = self._feeder.fetch_candles(
            pair, granularity="H1", count=50, include_incomplete=False)
        d_candles, _ = self._feeder.fetch_candles(
            pair, granularity="D", count=50, include_incomplete=False)
        # Fetch 100 M5 candles for ATR percentile
        m5_candles, _ = self._feeder.fetch_candles(
            pair, granularity="M5", count=100, include_incomplete=False)

        h1_bias, h1_slope = self._bias_from_candles(h1_candles, pair)
        d1_bias, d1_slope = self._bias_from_candles(d_candles, pair)
        m5_vol_pct, m5_vol_bucket = self._vol_percentile(m5_candles, pair)

        return MTFSnapshot(
            pair=pair,
            h1_bias=h1_bias, h1_slope_pct=h1_slope,
            d1_bias=d1_bias, d1_slope_pct=d1_slope,
            m5_vol_pct=m5_vol_pct, m5_vol_bucket=m5_vol_bucket,
            updated_at=utc_now if utc_now.tzinfo else utc_now.replace(
                tzinfo=timezone.utc),
        )

    def _bias_from_candles(self, candles: list, pair: str) -> tuple[Bias, float]:
        if not candles or len(candles) < 30:
            return ("flat", 0.0)
        closes = [c.close for c in candles]
        # Compare EMA20 vs EMA50 as a compact trend proxy
        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50) if len(closes) >= 50 else _ema(closes, len(closes))
        if ema50 == 0:
            return ("flat", 0.0)
        slope_pct = (ema20 - ema50) / ema50 * 100.0
        # Threshold: H1/D1 considered trending if >0.10% separation
        if slope_pct > 0.10:
            return ("up", slope_pct)
        if slope_pct < -0.10:
            return ("down", slope_pct)
        return ("flat", slope_pct)

    def _vol_percentile(self, candles: list, pair: str
                        ) -> tuple[float, VolBucket]:
        if not candles or len(candles) < 20:
            return (0.5, "mid")
        ranges = [abs(c.high - c.low) for c in candles]
        current = ranges[-1]
        sorted_ranges = sorted(ranges)
        try:
            idx = sorted_ranges.index(current)
        except ValueError:
            idx = len(sorted_ranges) // 2
        pct = idx / max(len(sorted_ranges) - 1, 1)
        if pct < 0.33:
            bucket = "low"
        elif pct > 0.67:
            bucket = "high"
        else:
            bucket = "mid"
        return (pct, bucket)


def _ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    if len(values) < period:
        period = len(values)
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def format_snapshot(snap: Optional[MTFSnapshot]) -> str:
    if snap is None:
        return "MTF unavailable"
    return (
        f"H1:{snap.h1_bias}({snap.h1_slope_pct:+.2f}%) "
        f"D1:{snap.d1_bias}({snap.d1_slope_pct:+.2f}%) "
        f"vol:{snap.m5_vol_bucket}({snap.m5_vol_pct*100:.0f}%ile)"
    )
