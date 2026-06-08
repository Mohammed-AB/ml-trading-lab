"""Fetch historical M1 candle data from OANDA for backtesting / ML training.

Usage:
    # Requires .env with OANDA credentials (live token for fxtrade)
    python scripts/fetch_historical.py --pair EUR_USD --months 12 --output data/raw

    # Fetch all V2 ML pairs (6), 12 months, live API by default
    python scripts/fetch_historical.py --v2 --months 12 --output data/raw
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Legacy three-pair set
V1_PAIRS = ["EUR_USD", "USD_JPY", "GBP_USD"]
# ML Model V2 — wider spreads, excludes EUR_GBP / USD_CHF
V2_PAIRS = [
    "EUR_USD",
    "USD_JPY",
    "GBP_USD",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
]


def _pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair.upper() else 0.0001


def fetch_candles(
    base_url: str,
    token: str,
    pair: str,
    granularity: str,
    from_time: str,
    to_time: str,
) -> list[dict]:
    """Fetch candles from OANDA REST API (mid + bid + ask)."""
    url = f"{base_url}/v3/instruments/{pair}/candles"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    params = {
        "granularity": granularity,
        "from": from_time,
        "to": to_time,
        "price": "MBA",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json().get("candles", [])


def fetch_pair_history(
    base_url: str,
    token: str,
    pair: str,
    months: int,
    output_dir: Path,
) -> Path:
    """Fetch M1 candles for a pair over N calendar months, paginated in 3-day chunks."""
    end_ts = pd.Timestamp.now(tz="UTC")
    start_ts = end_ts - pd.DateOffset(months=int(months))

    all_rows: list[dict] = []
    current_from = start_ts
    pip = _pip_size(pair)

    print(f"  Fetching {pair} from {start_ts.date()} to {end_ts.date()}...")

    while current_from < end_ts:
        current_to = min(current_from + pd.Timedelta(days=3), end_ts)

        from_str = current_from.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        to_str = current_to.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        try:
            candles = fetch_candles(base_url, token, pair, "M1", from_str, to_str)
            n_ok = 0
            for c in candles:
                if not c.get("complete", False):
                    continue
                mid = c.get("mid") or {}
                bid = c.get("bid") or {}
                ask = c.get("ask") or {}
                try:
                    o = float(mid["o"])
                    hi = float(mid["h"])
                    lo = float(mid["l"])
                    cl = float(mid["c"])
                except (KeyError, TypeError, ValueError):
                    continue
                spread_pips = None
                try:
                    bc = float(bid["c"])
                    ac = float(ask["c"])
                    spread_pips = (ac - bc) / pip
                except (KeyError, TypeError, ValueError):
                    pass
                row = {
                    "timestamp": c["time"],
                    "open": o,
                    "high": hi,
                    "low": lo,
                    "close": cl,
                    "volume": int(c.get("volume", 0)),
                }
                if spread_pips is not None:
                    row["spread_pips"] = float(spread_pips)
                all_rows.append(row)
                n_ok += 1
            print(
                f"    {current_from.date()} → {current_to.date()}: "
                f"{len(candles)} candles ({n_ok} complete, total rows {len(all_rows)})"
            )
        except requests.exceptions.RequestException as e:
            print(f"    Error at {current_from.date()}: {e}. Retrying in 2s...")
            time.sleep(2)
            continue

        current_from = current_to
        time.sleep(0.5)

    if not all_rows:
        print(f"  WARNING: No candles fetched for {pair}")
        return Path()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{pair}_M1_{months}m.csv"
    df.to_csv(filepath, index=False)
    print(f"  Saved {len(df)} candles to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Fetch OANDA historical data")
    parser.add_argument("--pair", type=str, help="Single pair (e.g., EUR_USD)")
    parser.add_argument("--all", action="store_true", help="Fetch legacy V1 pairs (3)")
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Fetch ML V2 pairs (6): EUR_USD USD_JPY GBP_USD AUD_USD USD_CAD NZD_USD",
    )
    parser.add_argument("--months", type=int, default=12, help="Calendar months of history")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    args = parser.parse_args()

    load_dotenv(args.env)
    token = os.environ.get("OANDA_API_TOKEN")
    if not token:
        print("ERROR: OANDA_API_TOKEN not set. Copy .env.example to .env and fill in.")
        sys.exit(1)

    # Default to the OANDA practice (demo) host; override with OANDA_BASE_URL.
    base_url = os.environ.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")

    if args.v2:
        pairs = V2_PAIRS
    elif args.all:
        pairs = V1_PAIRS
    elif args.pair:
        pairs = [args.pair]
    else:
        pairs = V2_PAIRS

    output_dir = Path(args.output)

    print(f"Base URL: {base_url}")
    print(f"Fetching {args.months} month(s) of M1 data for: {pairs}")
    for pair in pairs:
        fetch_pair_history(base_url, token, pair, args.months, output_dir)

    print("\nDone! Example feature build:")
    print(
        "  python3 ml_features.py --data-dir data/raw --fetch-months 12 "
        "--pairs EUR_USD GBP_USD USD_JPY USD_CAD AUD_USD NZD_USD"
    )


if __name__ == "__main__":
    main()
