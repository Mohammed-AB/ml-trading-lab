"""Shared config for strategy arena (V2 OANDA universe)."""

from __future__ import annotations

# Six-pair V2 universe (must match training / ml_features)
V2_PAIRS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CAD",
    "AUD_USD",
    "NZD_USD",
]

# Time splits (UTC) — same as ml_train / ml_backtest
OOS_START = "2026-03-01 00:00:00+00:00"
FETCH_MONTHS = 12

DEFAULT_DATA_RAW = "data/raw"
