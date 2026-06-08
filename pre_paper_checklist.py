"""Pre-Paper Trading Checklist — Tests ALL components before going live.

Run this BEFORE starting Paper Trading to verify everything works.

Usage:
    python pre_paper_checklist.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []


def test(name, func):
    try:
        ok, msg = func()
        status = PASS if ok else FAIL
        results.append((name, ok, msg))
        print(f"  {status} {name} — {msg}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} {name} — {e}")


def main():
    print("=" * 60)
    print("  PRE-PAPER TRADING CHECKLIST")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # ============================================================
    # 1. CONFIG VALIDATION
    # ============================================================
    print("\n--- 1. Configuration ---")

    def test_config_loads():
        from src.scalp_mode.config import Config
        cfg = Config("config/settings.yaml")
        sc = cfg.scalp
        assert sc["model_a"]["compression_atr_mult"] > 0
        assert sc["model_a"]["tp_R"] > 0
        return True, f"compression={sc['model_a']['compression_atr_mult']}, tp_R={sc['model_a']['tp_R']}"
    test("Config loads", test_config_loads)

    def test_model_b_disabled():
        from src.scalp_mode.config import Config
        cfg = Config("config/settings.yaml")
        enabled = cfg.scalp.get("model_b", {}).get("enabled", True)
        return not enabled, f"model_b.enabled={enabled}"
    test("Model B disabled", test_model_b_disabled)

    def test_ai_disabled():
        from src.scalp_mode.config import Config
        cfg = Config("config/settings.yaml")
        ai = cfg.scalp.get("ai", {})
        all_off = all(
            not m.get("enabled", False)
            for m in [ai.get("regime_classifier", {}),
                      ai.get("borderline", {}),
                      ai.get("post_trade", {})]
        )
        return all_off, "All AI modules disabled" if all_off else "AI module(s) enabled!"
    test("AI modules disabled", test_ai_disabled)

    def test_risk_params():
        from src.scalp_mode.config import Config
        cfg = Config("config/settings.yaml")
        risk = cfg.scalp.get("risk", {})
        ok = (risk.get("risk_pct", 0) == 0.0025 and
              risk.get("max_concurrent", 0) == 2 and
              risk.get("daily_loss", 0) == 0.01 and
              risk.get("consec_loss_circuit", 0) == 3)
        return ok, (f"risk={risk.get('risk_pct')}, max_concurrent={risk.get('max_concurrent')}, "
                     f"daily_loss={risk.get('daily_loss')}, circuit={risk.get('consec_loss_circuit')}")
    test("Risk parameters", test_risk_params)

    # ============================================================
    # 2. ENGINE COMPONENTS
    # ============================================================
    print("\n--- 2. Engine Components ---")

    def test_feature_engine():
        import pandas as pd
        import numpy as np
        from src.scalp_mode.engine.feature_engine import FeatureEngine
        fe = FeatureEngine()
        n = 100
        close = pd.Series(np.cumsum(np.random.randn(n) * 0.0001) + 1.1000)
        df = pd.DataFrame({
            "open": close - 0.00005,
            "high": close + 0.0001,
            "low": close - 0.0001,
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        })
        ind = fe.compute(df, "M1")
        has_nan, field = ind.has_nan()
        series = fe.compute_series(df, "M1")
        return (not has_nan and len(series) > 5,
                f"EMA20={ind.ema20:.5f} RSI={ind.rsi14:.1f} series_keys={len(series)}")
    test("Feature Engine", test_feature_engine)

    def test_regime_engine():
        from src.scalp_mode.engine.feature_engine import IndicatorSet
        from src.scalp_mode.engine.regime_engine import RegimeEngine, Regime
        re = RegimeEngine({"trend": {"ema_slope_thr": 0.15, "rsi_min": 52, "rsi_max": 78},
                           "range": {"bb_width_thr": 0.004}})
        ind = IndicatorSet(ema20=1.10, ema50=1.09, ema_slope=0.25, rsi14=60,
                           bb_width=0.005, bb_upper=1.11, bb_mid=1.10, bb_lower=1.09, atr14=0.001)
        result = re.evaluate(ind, 1.10)
        return result.regime == Regime.TREND_UP, f"regime={result.regime.value}"
    test("Regime Engine", test_regime_engine)

    def test_model_a():
        from src.scalp_mode.engine.model_a import ModelATrigger
        trigger = ModelATrigger({
            "compression_N": 8, "compression_atr_mult": 2.0,
            "breakout_buffer_atr": 0.10, "retest_timeout": 3,
            "retest_tolerance_atr": 0.15, "body_ratio_min": 0.55,
            "rsi_min_long": 55, "sl_atr": 0.8, "tp_R": 1.7,
            "time_stop_min": 6, "sl_move_threshold_R": 0.8,
            "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4],
        })
        return True, "ModelATrigger initialized"
    test("Model A Trigger", test_model_a)

    def test_cooldown():
        from src.scalp_mode.engine.cooldown import CooldownManager, TradeRecord
        cm = CooldownManager({"cooldown_same_pair_dir_min": 10,
                               "consec_loss_circuit": 3, "cooldown_minutes": 60,
                               "trades_per_hour_pair": 3, "trades_per_hour_total": 6,
                               "daily_loss": 0.01})
        now = datetime.now(timezone.utc)
        result = cm.check("EUR_USD", "long", now)
        return result.is_ok, f"is_ok={result.is_ok}"
    test("Cooldown Manager", test_cooldown)

    def test_pip_utils():
        from src.scalp_mode.utils.pip_utils import pip_value, price_to_pips, pips_to_price
        pv_eur = pip_value("EUR_USD")
        pv_jpy = pip_value("USD_JPY")
        pips = price_to_pips(0.0010, "EUR_USD")
        ok = pv_eur == 0.0001 and pv_jpy == 0.01 and pips == 10.0
        return ok, f"EUR=0.0001({pv_eur}), JPY=0.01({pv_jpy}), 0.001→{pips}pips"
    test("Pip Utils (JPY check)", test_pip_utils)

    # ============================================================
    # 3. NEWS FETCHER
    # ============================================================
    print("\n--- 3. News Fetcher ---")

    def test_news_fetcher():
        from src.scalp_mode.gates.news_fetcher import NewsCalendarFetcher
        fetcher = NewsCalendarFetcher(output_path="data/news_events_test.json")
        events = fetcher.fetch_events()
        if events:
            high = [e for e in events if e.get("impact") == "high"]
            return True, f"{len(events)} events ({len(high)} high-impact)"
        else:
            return False, "No events fetched (weekend or source down)"
    test("News Fetcher (FF mirror)", test_news_fetcher)

    def test_news_gate():
        from src.scalp_mode.gates.news_gate import NewsGate, NewsEvent
        gate = NewsGate(pre_minutes=10, post_minutes=5)
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        gate.set_events([
            NewsEvent(timestamp_utc=future, currency="USD",
                      impact="high", title="Test NFP")
        ])
        result = gate.check("EUR_USD", datetime.now(timezone.utc))
        return not result.is_safe, f"Blocked by upcoming event: {result.next_event_minutes}min away"
    test("News Gate (freeze logic)", test_news_gate)

    # ============================================================
    # 4. KILL SWITCHES
    # ============================================================
    print("\n--- 4. Kill Switches ---")

    def test_consec_loss_circuit():
        from src.scalp_mode.engine.cooldown import CooldownManager, TradeRecord
        cm = CooldownManager({"cooldown_same_pair_dir_min": 10,
                               "consec_loss_circuit": 3, "cooldown_minutes": 60,
                               "trades_per_hour_pair": 3, "trades_per_hour_total": 6,
                               "daily_loss": 0.01})
        now = datetime.now(timezone.utc)
        for j in range(3):
            cm.record_trade(TradeRecord(
                pair="EUR_USD", direction="long",
                timestamp_utc=now + timedelta(seconds=j),
                pnl_pct=-0.002))
        result = cm.check("EUR_USD", "long", now + timedelta(seconds=5))
        return not result.is_ok, f"Blocked after 3 losses: {result.reason}"
    test("3 consec losses → circuit breaker", test_consec_loss_circuit)

    def test_daily_loss_limit():
        from src.scalp_mode.engine.cooldown import CooldownManager, TradeRecord
        cm = CooldownManager({"cooldown_same_pair_dir_min": 10,
                               "consec_loss_circuit": 3, "cooldown_minutes": 60,
                               "trades_per_hour_pair": 3, "trades_per_hour_total": 6,
                               "daily_loss": 0.01})
        now = datetime.now(timezone.utc)
        cm.record_trade(TradeRecord(
            pair="EUR_USD", direction="long",
            timestamp_utc=now, pnl_pct=-0.011))
        result = cm.check("EUR_USD", "long", now + timedelta(seconds=1))
        return not result.is_ok, f"Blocked: {result.reason}"
    test("Daily loss -1% → halt", test_daily_loss_limit)

    def test_pair_cooldown():
        from src.scalp_mode.engine.cooldown import CooldownManager, TradeRecord
        cm = CooldownManager({"cooldown_same_pair_dir_min": 10,
                               "consec_loss_circuit": 3, "cooldown_minutes": 60,
                               "trades_per_hour_pair": 3, "trades_per_hour_total": 6,
                               "daily_loss": 0.01})
        now = datetime.now(timezone.utc)
        cm.record_trade(TradeRecord(
            pair="EUR_USD", direction="long",
            timestamp_utc=now, pnl_pct=0.001))
        result = cm.check("EUR_USD", "long", now + timedelta(minutes=5))
        return not result.is_ok, f"Blocked: {result.reason}"
    test("Same pair+dir 10min cooldown", test_pair_cooldown)

    # ============================================================
    # 5. MONITORING & ALERTS
    # ============================================================
    print("\n--- 5. Monitoring ---")

    def test_alert_manager():
        from src.scalp_mode.monitoring import AlertManager
        am = AlertManager(log_dir="logs/test_alerts")
        am.alert_kill_switch("test_circuit", {"test": True})
        am.alert_daily_loss(-0.012, 0.01)
        am.alert_consecutive_losses(3, 60)
        log_file = Path("logs/test_alerts/alerts.jsonl")
        if log_file.exists():
            lines = log_file.read_text().strip().split("\n")
            return len(lines) >= 3, f"{len(lines)} alerts logged to file"
        return False, "Alert file not created"
    test("AlertManager (file logging)", test_alert_manager)

    def test_alert_webhook_config():
        webhook = os.environ.get("SCALP_WEBHOOK_URL")
        if webhook and "<TOKEN>" not in webhook:
            return True, f"Webhook configured: {webhook[:30]}..."
        elif webhook and "<TOKEN>" in webhook:
            print(f"  {WARN} Webhook has placeholder <TOKEN> — replace with real Telegram token")
            return True, "Webhook placeholder (optional)"
        else:
            print(f"  {WARN} No SCALP_WEBHOOK_URL env var — Telegram/Slack alerts won't work")
            return True, "No webhook (optional — console+file alerts still work)"
    test("Webhook config", test_alert_webhook_config)

    # ============================================================
    # 6. EXECUTION LAYER
    # ============================================================
    print("\n--- 6. Execution Layer ---")

    def test_executor_import():
        from src.scalp_mode.execution.executor import Executor
        return True, "Executor importable"
    test("Executor import", test_executor_import)

    def test_trade_manager_import():
        from src.scalp_mode.execution.trade_manager import TradeManager, ExitReason
        reasons = [ExitReason.TP_HIT, ExitReason.SL_HIT, ExitReason.TIME_STOP,
                   ExitReason.KILL_SWITCH]
        return len(reasons) == 4, f"Exit reasons: {[r.value for r in reasons]}"
    test("TradeManager import", test_trade_manager_import)

    def test_order_builder_import():
        from src.scalp_mode.execution.order_builder import OrderBuilder
        return True, "OrderBuilder importable"
    test("OrderBuilder import", test_order_builder_import)

    def test_pending_manager_import():
        from src.scalp_mode.execution.pending_manager import PendingOrderManager
        return True, "PendingOrderManager importable"
    test("PendingOrderManager import", test_pending_manager_import)

    def test_risk_manager_import():
        from src.scalp_mode.execution.risk_manager import RiskManager
        return True, "RiskManager importable"
    test("RiskManager import", test_risk_manager_import)

    # ============================================================
    # 7. SESSION GATE
    # ============================================================
    print("\n--- 7. Session Gate ---")

    def test_session_gate():
        from src.scalp_mode.gates.session_gate import is_session_allowed
        monday_overlap = datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc)
        result = is_session_allowed(monday_overlap)
        return result.allowed, f"Monday 14:00 UTC: allowed={result.allowed}, window={result.window_name}"
    test("Session Gate (overlap)", test_session_gate)

    def test_session_blocked():
        from src.scalp_mode.gates.session_gate import is_session_allowed
        witching = datetime(2026, 3, 30, 23, 0, tzinfo=timezone.utc)
        result = is_session_allowed(witching)
        return not result.allowed, f"Monday 23:00 UTC: allowed={result.allowed} (should be blocked)"
    test("Session Gate (witching blocked)", test_session_blocked)

    # ============================================================
    # 8. BACKTESTER (quick sanity)
    # ============================================================
    print("\n--- 8. Backtester ---")

    def test_backtester_quick():
        import pandas as pd
        import numpy as np
        from src.scalp_mode.backtest.backtester import Backtester, BacktestConfig
        n = 5000
        close = pd.Series(np.cumsum(np.random.randn(n) * 0.0002) + 1.1000)
        df = pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + abs(np.random.randn(n) * 0.0001),
            "low": close - abs(np.random.randn(n) * 0.0001),
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        })
        bt = Backtester(
            {"regime": {"trend": {"ema_slope_thr": 0.15, "rsi_min": 52, "rsi_max": 78},
                        "range": {"bb_width_thr": 0.004}},
             "model_a": {"compression_N": 8, "compression_atr_mult": 2.0,
                         "breakout_buffer_atr": 0.10, "retest_timeout": 3,
                         "retest_tolerance_atr": 0.15, "body_ratio_min": 0.55,
                         "rsi_min_long": 55, "sl_atr": 0.8, "tp_R": 1.7,
                         "time_stop_min": 6, "sl_move_threshold_R": 0.8,
                         "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]},
             "risk": {"risk_pct": 0.0025, "max_concurrent": 2,
                      "cooldown_same_pair_dir_min": 10, "consec_loss_circuit": 3,
                      "cooldown_minutes": 60, "trades_per_hour_pair": 3,
                      "trades_per_hour_total": 6, "daily_loss": 0.01,
                      "max_margin_pct": 0.08},
             "orders": {"limit_ttl_seconds": 180, "fallback_market": True,
                        "fallback_max_atr_distance": 0.3, "price_bound_slippage": 0.2},
             "costs": {"max_spread_pips": {"EUR_USD": 0.8}}},
            BacktestConfig(check_sessions=False, fixed_spread_pips=0.3))
        t0 = time.time()
        trades = bt.run("EUR_USD", df)
        elapsed = time.time() - t0
        return True, f"{len(trades)} trades on 5K bars in {elapsed:.1f}s"
    test("Backtester (5K bars)", test_backtester_quick)

    # ============================================================
    # 9. OANDA CONNECTIVITY (if credentials set)
    # ============================================================
    print("\n--- 9. OANDA Connectivity ---")

    def test_oanda_env():
        acct = os.environ.get("OANDA_ACCOUNT_ID")
        token = os.environ.get("OANDA_API_TOKEN")
        if acct and token:
            return True, f"Account: {acct}"
        return False, "Set OANDA_ACCOUNT_ID and OANDA_API_TOKEN env vars"
    test("OANDA env vars", test_oanda_env)

    if os.environ.get("OANDA_ACCOUNT_ID") and os.environ.get("OANDA_API_TOKEN"):
        def test_oanda_api():
            import requests
            acct = os.environ["OANDA_ACCOUNT_ID"]
            token = os.environ["OANDA_API_TOKEN"]
            base = os.environ.get(
                "OANDA_BASE_URL", "https://api-fxpractice.oanda.com")
            url = f"{base.rstrip('/')}/v3/accounts/{acct}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=10)
            if resp.status_code == 200:
                nav = resp.json()["account"]["NAV"]
                return True, f"NAV={nav}"
            return False, f"HTTP {resp.status_code}"
        test("OANDA API connection", test_oanda_api)

        def test_oanda_candles():
            import requests
            token = os.environ["OANDA_API_TOKEN"]
            base = os.environ.get(
                "OANDA_BASE_URL", "https://api-fxpractice.oanda.com")
            url = f"{base.rstrip('/')}/v3/instruments/EUR_USD/candles"
            t0 = time.time()
            resp = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                                params={"count": 5, "granularity": "M1"}, timeout=10)
            latency = (time.time() - t0) * 1000
            if resp.status_code == 200:
                candles = resp.json().get("candles", [])
                return True, f"{len(candles)} candles in {latency:.0f}ms"
            return False, f"HTTP {resp.status_code}"
        test("OANDA candle fetch", test_oanda_candles)
    else:
        print(f"  {WARN} Skipping OANDA tests — credentials not set")

    # ============================================================
    # 10. FILE STRUCTURE
    # ============================================================
    print("\n--- 10. File Structure ---")

    def test_directories():
        dirs = ["logs", "data", "config"]
        missing = [d for d in dirs if not Path(d).exists()]
        if missing:
            for d in missing:
                Path(d).mkdir(parents=True, exist_ok=True)
            return True, f"Created: {missing}"
        return True, "All directories exist"
    test("Required directories", test_directories)

    def test_data_files():
        files_3m = list(Path("data").glob("*_M1_3m.csv"))
        files_12m = list(Path("data").glob("*_M1_12m.csv"))
        return len(files_3m) >= 3 or len(files_12m) >= 3, f"3m: {len(files_3m)} files, 12m: {len(files_12m)} files"
    test("Data files", test_data_files)

    # ============================================================
    #  SUMMARY
    # ============================================================
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    failed = [(name, msg) for name, ok, msg in results if not ok]

    print(f"\n{'='*60}")
    print(f"  RESULT: {passed}/{total} passed")
    if failed:
        print(f"\n  FAILURES:")
        for name, msg in failed:
            print(f"    - {name}: {msg}")
        print(f"\n  Status: FIX FAILURES BEFORE PAPER")
    else:
        print(f"\n  Status: ALL CLEAR — Ready for Paper Trading!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
