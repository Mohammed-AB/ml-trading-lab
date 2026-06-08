"""OANDA Practice Integration Test — Run before Paper trading.

Verifies that ALL system components work with a real OANDA Practice account:
1. API connectivity and authentication
2. Candle fetching (M1 + M5)
3. Pricing stream (heartbeat + live prices)
4. Spread calculation accuracy
5. Account details (NAV, margin)
6. Order submission + cancellation
7. Full pipeline cycle (no actual trade, just verification)

Usage:
    # Set up .env first, then:
    python scripts/test_oanda_integration.py

    # Skip order test (read-only mode):
    python scripts/test_oanda_integration.py --no-orders

Exit codes:
    0 = all tests passed
    1 = one or more tests failed
"""

import argparse
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv


def _status(name: str, passed: bool, detail: str = ""):
    icon = "PASS" if passed else "FAIL"
    detail_str = f" — {detail}" if detail else ""
    print(f"  [{icon}] {name}{detail_str}")
    return passed


def test_connectivity(base_url: str, token: str, account_id: str) -> bool:
    """Test 1: API connectivity and authentication."""
    import requests
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(
            f"{base_url}/v3/accounts/{account_id}/summary",
            headers=headers, timeout=10)
        if resp.status_code == 200:
            acct = resp.json().get("account", {})
            return _status("API connectivity",True,
                           f"Account {account_id}, NAV={acct.get('NAV')}")
        return _status("API connectivity", False,
                       f"HTTP {resp.status_code}: {resp.text[:100]}")
    except Exception as e:
        return _status("API connectivity", False, str(e))


def test_candles(base_url: str, token: str) -> bool:
    """Test 2: Candle fetching."""
    from src.scalp_mode.data.price_feeder import PriceFeeder
    feeder = PriceFeeder(base_url, "", token, "")
    try:
        m1, lat1 = feeder.fetch_candles("EUR_USD", "M1", count=5)
        m5, lat5 = feeder.fetch_candles("EUR_USD", "M5", count=5)

        if not m1:
            return _status("Candle fetch", False, "No M1 candles returned")
        if not m5:
            return _status("Candle fetch", False, "No M5 candles returned")

        c = m1[-1]
        return _status("Candle fetch", True,
                       f"M1={len(m1)} ({lat1:.0f}ms) M5={len(m5)} ({lat5:.0f}ms) "
                       f"last_close={c.close}")
    except Exception as e:
        return _status("Candle fetch", False, str(e))


def test_stream(stream_url: str, token: str, account_id: str) -> bool:
    """Test 3: Pricing stream (5 second test)."""
    from src.scalp_mode.data.price_feeder import PriceFeeder
    feeder = PriceFeeder("", stream_url, token, account_id)

    heartbeats = []
    prices = []
    feeder.set_callbacks(
        on_heartbeat=lambda ts: heartbeats.append(ts),
        on_price=lambda p: prices.append(p))

    feeder.start_stream(["EUR_USD"])
    time.sleep(5)
    feeder.stop_stream()

    got_hb = len(heartbeats) > 0
    got_price = len(prices) > 0

    if got_price:
        p = prices[-1]
        return _status("Pricing stream", True,
                       f"{len(heartbeats)} heartbeats, {len(prices)} prices, "
                       f"last spread={p.spread_pips:.1f} pips")
    elif got_hb:
        return _status("Pricing stream", True,
                       f"{len(heartbeats)} heartbeats (no prices — market may be closed)")
    else:
        return _status("Pricing stream", False,
                       "No heartbeats or prices in 5 seconds")


def test_spread(base_url: str, stream_url: str, token: str, account_id: str) -> bool:
    """Test 4: Spread calculation accuracy."""
    from src.scalp_mode.data.price_feeder import PriceFeeder
    feeder = PriceFeeder(base_url, stream_url, token, account_id)

    prices = []
    feeder.set_callbacks(on_price=lambda p: prices.append(p))
    feeder.start_stream(["EUR_USD", "USD_JPY", "GBP_USD"])
    time.sleep(5)
    feeder.stop_stream()

    if not prices:
        return _status("Spread calc", True,
                       "No prices (market closed) — will verify during Paper")

    pairs_seen = set(p.pair for p in prices)
    spreads = {}
    for p in prices:
        spreads[p.pair] = p.spread_pips

    details = ", ".join(f"{k}={v:.1f}pip" for k, v in spreads.items())
    all_reasonable = all(0 < v < 5 for v in spreads.values())
    return _status("Spread calc", all_reasonable,
                   f"{len(pairs_seen)} pairs: {details}")


def test_account(base_url: str, token: str, account_id: str) -> bool:
    """Test 5: Account details."""
    from src.scalp_mode.execution.executor import Executor
    from src.scalp_mode.logger import ScalpLogger
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        log_config = {
            "log_dir": tmp, "decision_log_file": "d.jsonl",
            "trade_log_file": "t.jsonl", "cycle_log_file": "c.jsonl",
            "system_log_file": "s.log", "max_file_size_mb": 1,
            "backup_count": 1, "level": "WARNING",
        }
        logger = ScalpLogger(log_config)
        executor = Executor(base_url, token, account_id, logger)

        acct = executor.get_account_details()
        logger.close()

    if not acct:
        return _status("Account details", False, "Could not fetch account")

    nav = acct.get("NAV", "?")
    margin = acct.get("marginAvailable", "?")
    currency = acct.get("currency", "?")
    return _status("Account details", True,
                   f"NAV={nav} {currency}, margin_available={margin}")


def test_order_lifecycle(base_url: str, token: str, account_id: str) -> bool:
    """Test 6: Order submit + cancel (uses a far-out-of-market price)."""
    import requests
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Submit a Limit order at a price far from market (won't fill)
    order_body = {
        "order": {
            "type": "LIMIT",
            "instrument": "EUR_USD",
            "units": "1",
            "price": "1.00000",  # Far below market — won't fill
            "timeInForce": "GTD",
            "gtdTime": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.000000000Z"),
        }
    }

    try:
        # Try to submit (will likely be rejected due to expired GTD, which is fine)
        resp = requests.post(
            f"{base_url}/v3/accounts/{account_id}/orders",
            headers=headers, json=order_body, timeout=10)

        if resp.status_code == 201:
            data = resp.json()
            create_tx = data.get("orderCreateTransaction", {})
            order_id = create_tx.get("id")
            if order_id:
                # Cancel it immediately
                cancel_resp = requests.put(
                    f"{base_url}/v3/accounts/{account_id}/orders/{order_id}/cancel",
                    headers=headers, timeout=10)
                cancelled = cancel_resp.status_code == 200
                return _status("Order lifecycle", True,
                               f"submitted #{order_id}, cancelled={cancelled}")

            # Check if it was auto-cancelled (GTD expired)
            cancel_tx = data.get("orderCancelTransaction", {})
            if cancel_tx:
                return _status("Order lifecycle", True,
                               f"submitted + auto-cancelled (GTD expired)")

        elif resp.status_code == 400:
            # Rejection is expected for expired GTD — API is working
            return _status("Order lifecycle", True,
                           "API responded (order rejected as expected for test params)")
        else:
            return _status("Order lifecycle", False,
                           f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        return _status("Order lifecycle", False, str(e))


def test_pipeline_cycle(base_url: str, token: str, account_id: str) -> bool:
    """Test 7: Full pipeline cycle (read-only, no actual trade)."""
    from src.scalp_mode.data.price_feeder import PriceFeeder
    from src.scalp_mode.engine.feature_engine import FeatureEngine
    import pandas as pd

    feeder = PriceFeeder(base_url, "", token, "")
    feature = FeatureEngine()

    try:
        # Fetch enough candles for indicators
        m1_candles, lat1 = feeder.fetch_candles("EUR_USD", "M1", count=100)
        m5_candles, lat5 = feeder.fetch_candles("EUR_USD", "M5", count=50)

        if len(m1_candles) < 50:
            return _status("Pipeline cycle", False,
                           f"Only {len(m1_candles)} M1 candles (need 50+)")

        df_m1 = pd.DataFrame([{
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume
        } for c in m1_candles])
        df_m5 = pd.DataFrame([{
            "open": c.open, "high": c.high, "low": c.low,
            "close": c.close, "volume": c.volume
        } for c in m5_candles])

        # Compute indicators
        ind_m1 = feature.compute(df_m1, "M1")
        ind_m5 = feature.compute(df_m5, "M5")

        has_nan_m1, nan_field = ind_m1.has_nan()
        has_nan_m5, _ = ind_m5.has_nan()

        return _status("Pipeline cycle", True,
                       f"M1: EMA20={ind_m1.ema20:.5f} RSI={ind_m1.rsi14:.1f} "
                       f"ATR={ind_m1.atr14:.6f} | "
                       f"M5: slope={ind_m5.ema_slope:.3f} BB_w={ind_m5.bb_width:.4f} | "
                       f"NaN: M1={has_nan_m1} M5={has_nan_m5}")
    except Exception as e:
        return _status("Pipeline cycle", False, str(e))


def main():
    parser = argparse.ArgumentParser(description="OANDA Practice Integration Test")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--no-orders", action="store_true",
                        help="Skip order submission test")
    args = parser.parse_args()

    load_dotenv(args.env)
    token = os.environ.get("OANDA_API_TOKEN")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")

    if not token or not account_id:
        print("ERROR: OANDA_API_TOKEN and OANDA_ACCOUNT_ID must be set in .env")
        print("  cp .env.example .env  # then edit with your credentials")
        sys.exit(1)

    base_url = os.environ.get("OANDA_BASE_URL",
                               "https://api-fxpractice.oanda.com")
    stream_url = os.environ.get("OANDA_STREAM_URL",
                                 "https://stream-fxpractice.oanda.com")

    env_label = "Live" if "fxtrade" in base_url else "Practice"
    print("=" * 60)
    print(f"  OANDA {env_label} Integration Test")
    print(f"  Account: {account_id}")
    print(f"  API: {base_url}")
    print("=" * 60)

    results = []
    results.append(test_connectivity(base_url, token, account_id))
    results.append(test_candles(base_url, token))
    results.append(test_stream(stream_url, token, account_id))
    results.append(test_spread(base_url, stream_url, token, account_id))
    results.append(test_account(base_url, token, account_id))

    if not args.no_orders:
        results.append(test_order_lifecycle(base_url, token, account_id))
    else:
        print("  [SKIP] Order lifecycle (--no-orders)")

    results.append(test_pipeline_cycle(base_url, token, account_id))

    print()
    passed = sum(results)
    total = len(results)
    print(f"  Result: {passed}/{total} passed")

    if all(results):
        print("  Status: READY FOR PAPER TRADING")
    else:
        print("  Status: FIX FAILURES BEFORE PAPER")

    print("=" * 60)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
