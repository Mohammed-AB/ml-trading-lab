"""Scalp Mode V1 — Main entry point.

Usage:
    # Backtest mode (default)
    python main.py backtest --pair EUR_USD --data path/to/candles.csv

    # Paper trading (requires OANDA Practice account)
    python main.py paper

    # Live trading (requires OANDA Live account)
    python main.py live
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.scalp_mode.config import Config
from src.scalp_mode.logger import ScalpLogger
from src.scalp_mode.data.price_feeder import PriceFeeder
from src.scalp_mode.engine.feature_engine import FeatureEngine
from src.scalp_mode.engine.regime_engine import RegimeEngine
from src.scalp_mode.engine.model_a import ModelATrigger
from src.scalp_mode.engine.model_b import ModelBTrigger
from src.scalp_mode.ai.post_trade_analyst import PostTradeAnalyst
from src.scalp_mode.ai.borderline_reviewer import AIBorderlineReviewer
from src.scalp_mode.ai.regime_classifier import AIRegimeClassifier
from src.scalp_mode.engine.cooldown import CooldownManager
from src.scalp_mode.engine.decision_pipeline import DecisionPipeline
from src.scalp_mode.execution.risk_manager import RiskManager
from src.scalp_mode.execution.order_builder import OrderBuilder
from src.scalp_mode.execution.executor import (
    Executor,
    parse_account_nav,
    parse_oanda_decimal,
)
from src.scalp_mode.execution.trade_manager import TradeManager
from src.scalp_mode.monitoring import AlertManager
from src.scalp_mode.gates.data_quality_gate import DataQualityGate
from src.scalp_mode.gates.news_gate import NewsGate
from src.scalp_mode.gates.news_fetcher import NewsCalendarFetcher
from src.scalp_mode.utils.pip_utils import price_to_pips
from src.scalp_mode.execution.pending_manager import (
    PendingOrderManager, PendingOrder, BrokerReconciler,
)
from src.scalp_mode.backtest.backtester import Backtester, BacktestConfig
from src.scalp_mode.backtest.performance import PerformanceAnalyzer
from src.scalp_mode.backtest.go_nogo import GoNoGoEvaluator
from src.scalp_mode.ai.pilot import AIPilot, PilotContext
from src.scalp_mode.ai.pilot_journal import PilotJournal
from src.scalp_mode.ai.pilot_news_intel import NewsIntelligence
from src.scalp_mode.engine.pilot_pipeline import PilotPipeline
from src.scalp_mode.engine.model_c import ModelCTrigger
from src.scalp_mode.engine.model_d import ModelDTrigger
from src.scalp_mode.engine.model_e import ModelETrigger
from src.scalp_mode.agents.brain import Brain
from src.scalp_mode.agents.research import ResearchAgent
from src.scalp_mode.agents.strategy import StrategyAgent
from src.scalp_mode.agents.risk import RiskAgent
from src.scalp_mode.agents.portfolio import PortfolioAgent
from src.scalp_mode.agents.learning import LearningAgent
from src.scalp_mode.agents.orchestrator import Orchestrator


def run_backtest(args, config: Config):
    """Run backtest on historical data."""
    logger = ScalpLogger(config.logging_config)
    logger.info("Starting backtest...")

    bt_config = BacktestConfig(
        initial_nav=args.nav,
        fixed_spread_pips=args.spread,
        slippage_pips=args.slippage,
        check_sessions=not args.no_sessions,
        warmup_bars=60,
    )
    backtester = Backtester(config.scalp, bt_config)

    # Load candle data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        logger.error(f"Data must have columns: {required_cols}")
        sys.exit(1)

    timestamps = None
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], utc=True)

    logger.info(f"Loaded {len(df)} candles for {args.pair}")

    trades = backtester.run(args.pair, df, timestamps)
    logger.info(f"Backtest completed: {len(trades)} trades")

    # Analyze performance
    analyzer = PerformanceAnalyzer()
    # Calculate trading days from actual data
    if args.days:
        days = args.days
    elif timestamps is not None and len(timestamps) > 0:
        days = max(timestamps.dt.date.nunique(), 1)
    else:
        days = max(len(df) // 1300, 1)  # ~1300 M1 bars per forex trading day
    metrics = analyzer.compute(trades, trading_days=days)

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Pair:              {args.pair}")
    print(f"  Candles:           {len(df)}")
    print(f"  Trading days:      {days}")
    print(f"  Total trades:      {metrics.total_trades}")
    print(f"  Win rate:          {metrics.win_rate:.1%}")
    print(f"  Sharpe ratio:      {metrics.sharpe_ratio:.2f}")
    print(f"  Max drawdown:      {metrics.max_drawdown_pct:.2f}%")
    print(f"  Profit factor:     {metrics.profit_factor:.2f}")
    print(f"  Total PnL (pips):  {metrics.total_pnl_pips:.1f}")
    print(f"  Avg PnL (pips):    {metrics.avg_pnl_pips:.2f}")
    print(f"  Avg winner:        {metrics.avg_winner_pips:.2f} pips")
    print(f"  Avg loser:         {metrics.avg_loser_pips:.2f} pips")
    print(f"  Max consec losses: {metrics.max_consecutive_losses}")
    print(f"  Slippage impact:   {metrics.slippage_impact_pct:.1f}%")
    print(f"  TP/SL/Time:        {metrics.tp_hit_count}/{metrics.sl_hit_count}/{metrics.time_stop_count}")
    print(f"  Borderline trades: {metrics.borderline_count}")

    # Model breakdown
    model_a_trades = [t for t in trades if t.model == "A"]
    model_b_trades = [t for t in trades if t.model == "B"]
    if model_a_trades or model_b_trades:
        print()
        print("  --- Model Breakdown ---")
        for label, subset in [("Model A (Trend)", model_a_trades), ("Model B (Range)", model_b_trades)]:
            if not subset:
                print(f"  {label:20s}: 0 trades")
                continue
            count = len(subset)
            wins = sum(1 for t in subset if t.pnl_pips > 0)
            wr = wins / count if count > 0 else 0
            total_pnl = sum(t.pnl_pips for t in subset)
            avg_pnl = total_pnl / count if count > 0 else 0
            winners = [t.pnl_pips for t in subset if t.pnl_pips > 0]
            losers = [t.pnl_pips for t in subset if t.pnl_pips <= 0]
            avg_win = sum(winners) / len(winners) if winners else 0
            avg_loss = sum(losers) / len(losers) if losers else 0
            bl_count = sum(1 for t in subset if t.is_borderline)
            print(f"  {label:20s}: {count:3d} trades | WR={wr:.0%} | "
                  f"PnL={total_pnl:+.1f} pips | Avg={avg_pnl:+.2f} | "
                  f"AvgW={avg_win:+.2f} AvgL={avg_loss:+.2f} | "
                  f"BL={bl_count}")
    print()

    # Go/No-Go
    evaluator = GoNoGoEvaluator()
    result = evaluator.backtest_to_paper(metrics)
    print(f"  Go/No-Go verdict:  {result.verdict.value}")
    if result.failed_criteria:
        print(f"  Failed:            {', '.join(result.failed_criteria)}")
    if result.stop_criteria:
        print(f"  STOP:              {', '.join(result.stop_criteria)}")
    print("=" * 60)

    # Export results
    log_dir = Path(config.logging_config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    if trades:
        trades_csv = log_dir / "backtest_trades.csv"
        import csv
        with open(trades_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "pair", "direction", "model", "entry_time", "exit_time",
                "entry_price", "exit_price", "sl_price", "tp_price",
                "units", "pnl_pips", "pnl_pct", "exit_reason",
                "hold_time_seconds", "spread_at_entry", "slippage_pips",
                "is_borderline", "borderline_flags",
            ])
            writer.writeheader()
            for t in trades:
                writer.writerow(t.__dict__)
        print(f"\n  Trades exported to: {trades_csv}")

    summary = {
        "pair": args.pair, "candles": len(df), "trading_days": days,
        **metrics.__dict__,
        "go_nogo": result.verdict.value,
        "failed_criteria": result.failed_criteria,
    }
    summary_path = log_dir / "backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary exported to: {summary_path}")

    logger.close()


def run_walkforward(args, config: Config):
    """Run walk-forward backtest."""
    from src.scalp_mode.backtest.walk_forward import run_walk_forward

    logger = ScalpLogger(config.logging_config)
    logger.info("Starting walk-forward backtest...")

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    timestamps = None
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], utc=True)

    bt_config = BacktestConfig(
        fixed_spread_pips=args.spread,
        slippage_pips=0.1,
        check_sessions=not args.no_sessions,
        warmup_bars=60,
    )

    result = run_walk_forward(
        pair=args.pair, df_m1=df,
        scalp_config=config.scalp,
        bt_config=bt_config,
        n_windows=args.windows,
        timestamps=timestamps,
    )

    print("\n" + "=" * 60)
    print("  WALK-FORWARD RESULTS")
    print("=" * 60)
    print(f"  Pair:              {args.pair}")
    print(f"  Windows:           {result.total_windows}")
    print(f"  Profitable:        {result.profitable_windows}")
    print(f"  Win %:             {result.win_pct:.0%}")
    print()
    for w in result.windows:
        m = w.metrics
        status = "WIN" if w.is_profitable else "LOSS"
        print(f"  Window {w.window_idx}: [{status}] "
              f"trades={m.total_trades} WR={m.win_rate:.0%} "
              f"PnL={m.total_pnl_pips:+.1f}pips Sharpe={m.sharpe_ratio:.2f}")
    print()
    agg = result.aggregate_metrics
    print(f"  Aggregate: trades={agg.total_trades} WR={agg.win_rate:.0%} "
          f"Sharpe={agg.sharpe_ratio:.2f} PF={agg.profit_factor:.2f} "
          f"DD={agg.max_drawdown_pct:.2f}%")
    print()
    threshold = "PASS" if result.win_pct >= 0.60 else "FAIL"
    print(f"  Walk-forward >= 60%: {threshold} ({result.win_pct:.0%})")
    print("=" * 60)

    logger.close()


def run_paper(args, config: Config, is_live: bool = False):
    """Run paper/live trading against OANDA account."""
    mode_name = "LIVE" if is_live else "PAPER"
    logger = ScalpLogger(config.logging_config)
    logger.info(f"Starting {mode_name} trading...")

    # Live safeguards (#10)
    # NOTE: This is an educational/research project and defaults to OANDA
    # practice (demo) mode via config/settings.yaml. The auto-confirm bypass
    # has been intentionally removed so the interactive human confirmation
    # below can never be silently skipped.
    if is_live:
        logger.info("=" * 50)
        logger.info("  LIVE MODE — Real money at risk!")
        logger.info("  First-week safeguard: max 50% of normal trade rate")
        logger.info("=" * 50)
        print(f"\n{'='*50}")
        print(f"  LIVE TRADING MODE — Real money at risk!")
        print(f"  This research project ships in PRACTICE mode by default.")
        print(f"  Press Enter to confirm, or Ctrl+C to abort...")
        print(f"{'='*50}")
        try:
            input()
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    # Build all components
    feeder = PriceFeeder(
        config.oanda_base_url, config.oanda_stream_url,
        config.oanda_api_token, config.oanda_account_id)
    feature = FeatureEngine()
    regime = RegimeEngine(config.regime, config.borderline)
    trigger = ModelATrigger(config.model_a, config.borderline)
    model_b_cfg = config.scalp.get("model_b", {})
    trigger_b = None
    if model_b_cfg.get("enabled", False):
        trigger_b = ModelBTrigger(model_b_cfg, config.borderline)
        logger.info("Model B (Range reversal) enabled")

    # AI modules
    ai_config = config.scalp.get("ai", {})
    borderline_reviewer = None
    bl_cfg = ai_config.get("borderline", {})
    if bl_cfg.get("enabled", False):
        borderline_reviewer = AIBorderlineReviewer(bl_cfg)
        logger.info("AI Borderline Reviewer enabled")

    ai_regime_classifier = None
    rc_cfg = ai_config.get("regime_classifier", {})
    if rc_cfg.get("enabled", False):
        ai_regime_classifier = AIRegimeClassifier(rc_cfg, regime)
        logger.info("AI Regime Classifier enabled")

    post_analyst = PostTradeAnalyst(ai_config.get("post_trade", {}))
    if post_analyst.enabled:
        logger.info("AI Post-trade Analyst enabled")

    # Monitoring & alerts
    webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
    webhook_type = os.environ.get("ALERT_WEBHOOK_TYPE", "generic")
    alert_mgr = AlertManager(
        log_dir=config.logging_config.get("log_dir", "logs"),
        webhook_url=webhook_url, webhook_type=webhook_type)
    if webhook_url:
        logger.info(f"Alerts enabled via {webhook_type} webhook")

    executor = Executor(
        config.oanda_base_url, config.oanda_api_token,
        config.oanda_account_id, logger)
    trade_mgr = TradeManager(
        config.model_a, config.oanda_base_url,
        config.oanda_api_token, config.oanda_account_id, logger)

    # Account currency + NAV: from OANDA (USD, GBP, etc.) — not hardcoded
    acct_summary = executor.get_account_details()
    risk_config = dict(config.risk)
    risk_config["account_currency"] = (
        (acct_summary or {}).get("currency")
        or os.environ.get("OANDA_ACCOUNT_CURRENCY")
        or "USD"
    ).upper()
    if acct_summary:
        logger.info(
            f"Account {config.oanda_account_id}: "
            f"currency={risk_config['account_currency']}, "
            f"NAV={acct_summary.get('NAV')}")
    else:
        logger.warning(
            "Could not fetch account summary; using "
            f"account_currency={risk_config['account_currency']} "
            "(set OANDA_ACCOUNT_CURRENCY in .env if wrong)")

    cooldown = CooldownManager(risk_config, alert_manager=alert_mgr)
    risk_mgr = RiskManager(risk_config)
    order_builder = OrderBuilder(config.orders)
    news_gate = NewsGate()
    news_fetcher = NewsCalendarFetcher(output_path="data/news_events.json")
    news_count = news_fetcher.update_gate(news_gate)
    logger.info(f"Loaded {news_count} news events")
    news_fetcher.start_auto_update(news_gate)  # Refresh every 6 hours
    dq_gate = DataQualityGate(config.data_quality)

    # Pipeline with TradeManager wired in (#1)
    pipeline = DecisionPipeline(
        config=config, logger=logger,
        feature_engine=feature, regime_engine=regime,
        trigger=trigger, news_gate=news_gate,
        data_quality_gate=dq_gate, cooldown_manager=cooldown,
        risk_manager=risk_mgr, order_builder=order_builder,
        executor=executor, trade_manager=trade_mgr,
        trigger_b=trigger_b,
        borderline_reviewer=borderline_reviewer,
        ai_regime=ai_regime_classifier)

    # AI Pilot mode (autonomous AI decision-maker)
    pilot_cfg = ai_config.get("pilot", {})
    use_pilot = pilot_cfg.get("enabled", False)
    pilot_shadow = pilot_cfg.get("shadow_mode", False)
    pilot_pipeline = None
    pilot_journal = None
    pilot_news_intel = None

    if use_pilot or pilot_shadow:
        pilot_journal = PilotJournal(log_dir="data/logs")
        pilot_news_intel = NewsIntelligence(
            model=pilot_cfg.get("model", "claude-opus-4-20250514"),
            cache_dir="data/news_briefings")
        pilot = AIPilot(pilot_cfg, pilot_journal, pilot_news_intel)
        trigger_c = ModelCTrigger(config.scalp.get("model_c", {}))
        logger.info("Model C (EMA Crossover, backtest-validated) enabled")

        pilot_pipeline = PilotPipeline(
            pilot=pilot, journal=pilot_journal,
            news_intel=pilot_news_intel,
            feature_engine=feature, regime_engine=regime,
            order_builder=order_builder, executor=executor,
            trade_manager=trade_mgr, news_gate=news_gate,
            config=config.scalp, logger=logger,
            model_a=trigger, model_b=trigger_b, model_c=trigger_c)
        pilot_news_intel.start_background()
        if use_pilot and not pilot_shadow:
            logger.info("AI Pilot mode ACTIVE — autonomous trading enabled")
        elif pilot_shadow:
            logger.info("AI Pilot SHADOW mode — logging only, rules execute")

    # Multi-agent system
    ma_cfg = ai_config.get("multi_agent", {})
    use_multi_agent = ma_cfg.get("enabled", False)
    orchestrator = None
    ma_brain = None
    if use_multi_agent:
        trigger_c_ma = ModelCTrigger(config.scalp.get("model_c", {}))
        model_d_cfg = config.scalp.get("model_d", {})
        trigger_d_ma = ModelDTrigger(model_d_cfg) if model_d_cfg.get(
            "enabled", True) else None
        if trigger_d_ma:
            logger.info("Model D (Strategy P / hour mean-reversion) enabled")
        model_e_cfg = config.scalp.get("model_e", {})
        trigger_e_ma = ModelETrigger(model_e_cfg) if model_e_cfg.get(
            "enabled", True) else None
        if trigger_e_ma:
            logger.info("Model E (VWAP reversion) enabled")
        model_f_cfg = config.scalp.get("model_f", {})
        trigger_f_ma = None
        if model_f_cfg.get("enabled", False):
            from src.scalp_mode.engine.model_tky_ldn import ModelTkyLdnTrigger
            trigger_f_ma = ModelTkyLdnTrigger(model_f_cfg)
            logger.info("Model F (TKY_LDN Asia→London breakout) enabled")
        model_g_cfg = config.scalp.get("model_g", {})
        trigger_g_ma = None
        if model_g_cfg.get("enabled", False):
            from src.scalp_mode.engine.model_ha_adx import ModelHaAdxTrigger
            trigger_g_ma = ModelHaAdxTrigger(model_g_cfg)
            logger.info("Model G (HA_ADX trend entry) enabled")
        model_h_cfg = config.scalp.get("model_h", {})
        trigger_h_ma = None
        if model_h_cfg.get("enabled", False):
            from src.scalp_mode.engine.model_nr7 import ModelNr7Trigger
            trigger_h_ma = ModelNr7Trigger(model_h_cfg)
            logger.info("Model H (NR7 breakout) enabled")
        brain = Brain(ma_cfg.get("brain_dir", "data/brain"))
        ma_brain = brain
        ma_model = ma_cfg.get("model", "claude-opus-4-20250514")
        _ma_risk = config.scalp.get("risk", {})
        ml_gate = None
        _ml_cfg = ma_cfg.get("ml") or {}
        if _ml_cfg.get("enabled", False):
            try:
                from src.scalp_mode.ml.ml_gate import MLGate
                ml_gate = MLGate(_ml_cfg)
                logger.info(
                    "ML Trading Engine gate ENABLED "
                    f"(threshold={ml_gate.threshold})"
                )
            except Exception as e_ml:
                logger.error(f"ML gate failed to load (disabling ML): {e_ml}")
                ml_gate = None
        orchestrator = Orchestrator(
            brain=brain,
            research=ResearchAgent(brain, ma_model),
            strategy=StrategyAgent(brain, ma_model),
            risk=RiskAgent(
                brain, ma_cfg.get("account_floor_usd", 200), ma_model,
                max_positions=int(_ma_risk.get("max_concurrent", 2)),
                alert_manager=alert_mgr,
                log_dir=config.logging_config.get("log_dir", "logs")),
            portfolio=PortfolioAgent(brain, ma_model),
            learning=LearningAgent(brain, ma_model),
            order_builder=order_builder, executor=executor,
            trade_manager=trade_mgr, news_gate=news_gate,
            feature_engine=feature, regime_engine=regime,
            model_a=trigger, model_b=trigger_b, model_c=trigger_c_ma,
            model_d=trigger_d_ma, model_e=trigger_e_ma, model_f=trigger_f_ma,
            model_g=trigger_g_ma, model_h=trigger_h_ma,
            price_feeder=feeder,
            ml_gate=ml_gate,
            config=config.scalp, logger=logger,
            max_trades=ma_cfg.get("max_trades", 20))
        logger.info("Multi-Agent system ACTIVE (5 agents + shared brain)")

    # Pending order manager + broker reconciler
    pending_mgr = PendingOrderManager(
        executor, order_builder, trade_mgr, logger, poll_interval_sec=10)
    reconciler = BrokerReconciler(executor, trade_mgr, logger, reconcile_interval_sec=30)

    if orchestrator is not None:
        orchestrator.set_pending_orders_provider(
            pending_mgr.snapshot_for_risk)

    # Wire data quality callbacks
    feeder.set_callbacks(
        on_heartbeat=dq_gate.update_heartbeat,
        on_price=lambda p: dq_gate.update_price(p.timestamp_utc))

    # Graceful shutdown
    running = True

    def shutdown_handler(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received...")
        running = False

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Start price stream
    instruments = config.instruments
    feeder.start_stream(instruments)
    logger.info(f"Streaming prices for: {instruments}")

    # State recovery (spec A.6): check existing open trades
    existing_trades = executor.get_open_trades()
    if existing_trades:
        logger.info(f"Found {len(existing_trades)} existing open trades. "
                     f"Resuming Trade Manager for them.")
        from src.scalp_mode.execution.trade_manager import ManagedTrade
        for t in existing_trades:
            try:
                # Use broker's openTime for accurate time stop / SL move timing
                from src.scalp_mode.utils.datetime_utils import parse_oanda_timestamp
                try:
                    open_time = parse_oanda_timestamp(t.get("openTime"))
                except (ValueError, TypeError):
                    open_time = datetime.now(timezone.utc)

                entry = float(t.get("price", 0))
                sl_raw = float((t.get("stopLossOrder") or {}).get("price", 0))
                tp_raw = float((t.get("takeProfitOrder") or {}).get("price", 0))
                pair = t["instrument"]
                direction = "long" if int(t.get("currentUnits", 0)) > 0 else "short"

                # Safety net: if broker has no SL, apply a 30-pip emergency SL
                if sl_raw == 0 and entry > 0:
                    pip_val = 0.01 if "JPY" in pair else 0.0001
                    safety_dist = 30 * pip_val
                    if direction == "long":
                        sl_raw = round(entry - safety_dist, 5 if "JPY" not in pair else 3)
                    else:
                        sl_raw = round(entry + safety_dist, 5 if "JPY" not in pair else 3)
                    logger.warning(
                        f"  Trade {t['id']} has no SL on broker — "
                        f"setting 30-pip safety SL at {sl_raw}")
                    try:
                        sl_url = (f"{executor._base_url}/v3/accounts/"
                                  f"{executor._account_id}/trades/{t['id']}/orders")
                        sl_body = {"stopLoss": {"price": str(sl_raw),
                                                "timeInForce": "GTC"}}
                        import requests as _req
                        _req.put(sl_url, headers=executor._headers,
                                 json=sl_body, timeout=5)
                    except Exception as e_sl:
                        logger.error(f"  Failed to set safety SL: {e_sl}")

                managed = ManagedTrade(
                    trade_id=t["id"],
                    pair=pair,
                    direction=direction,
                    entry_price=entry,
                    sl_price=sl_raw,
                    tp_price=tp_raw,
                    units=abs(int(t.get("currentUnits", 0))),
                    open_time=open_time,
                    risk_amount=abs(entry - sl_raw) if sl_raw else entry * 0.003,
                    model="AI_PILOT" if (use_pilot or use_multi_agent) else "",
                )
                trade_mgr.add_trade(managed)
                logger.info(
                    f"  Resumed: {managed.trade_id} {managed.pair} "
                    f"{managed.direction} SL={managed.sl_price} TP={managed.tp_price}")
            except (KeyError, ValueError) as e:
                logger.error(f"  Failed to resume trade {t.get('id')}: {e}")

    # Stabilization period (spec A.6)
    stab_sec = config.data_quality.get("stabilization_sec", 60)
    logger.info(f"Stabilization: waiting {stab_sec}s before opening new trades...")
    stabilization_end = time.monotonic() + stab_sec
    # One pipeline run per calendar minute (UTC) at second==0 — not `last_second`,
    # which stays 0 and would skip every future minute (bug).

    last_pipeline_minute_key: tuple | None = None

    def _is_market_closed(dt):
        """Full shutdown hours (no processing at all):
        - Saturday all day (weekday==5)
        - Sunday before 22:00 UTC (Sunday 5pm ET = market reopen)
        - Friday after 21:00 UTC (5pm ET market close)
        """
        wd = dt.weekday()
        h = dt.hour
        if wd == 5:
            return True
        if wd == 6 and h < 22:
            return True
        if wd == 4 and h >= 21:
            return True
        return False

    def _seconds_until_market_open(dt):
        """How many seconds until market reopens. Sunday 22:00 UTC."""
        wd = dt.weekday()
        # Find next Sunday 22:00 UTC or Monday if we're Monday already closed
        from datetime import timedelta as _td
        target = dt.replace(hour=22, minute=0, second=0, microsecond=0)
        if wd == 4:
            # From Friday, jump to Sunday 22:00 UTC
            target = target + _td(days=2)
        elif wd == 5:
            target = target + _td(days=1)
        elif wd == 6:
            if dt.hour >= 22:
                return 60  # shouldn't happen, already open
            # Today at 22:00
            pass
        return max(60, int((target - dt).total_seconds()))

    try:
        while running:
            now = datetime.now(timezone.utc)

            # If market is closed, fully idle until it reopens.
            if _is_market_closed(now):
                sleep_s = _seconds_until_market_open(now)
                hrs = sleep_s // 3600
                mins = (sleep_s % 3600) // 60
                logger.info(
                    f"Market closed — idle until next open "
                    f"(~{hrs}h {mins}m). No processing, no API calls.")
                # Sleep in 60s chunks so SIGTERM/SIGINT stays responsive
                remaining = sleep_s
                while remaining > 0 and running:
                    chunk = min(60, remaining)
                    time.sleep(chunk)
                    remaining -= chunk
                continue

            # Manage open trades every second (not just at minute boundaries)
            live_prices_mid = {}
            live_prices_ba = {}
            for pair in instruments:
                live = feeder.get_live_price(pair)
                if live:
                    live_prices_mid[pair] = (live.bid + live.ask) / 2
                    live_prices_ba[pair] = (live.bid, live.ask)
            if live_prices_mid and trade_mgr.open_trades:
                actions = trade_mgr.evaluate_all(now, live_prices_mid)
                for trade_id, action in actions:
                    if action.action == "move_sl" and action.new_sl:
                        trade_mgr.execute_sl_move(trade_id, action.new_sl)
                        alert_mgr.alert_info(
                            "SL_MOVED",
                            f"Trade {trade_id}: SL moved to {action.new_sl:.5f}",
                        )
                    elif action.action == "close" and action.exit_reason:
                        trade = trade_mgr._trades.get(trade_id)
                        exit_price = live_prices_mid.get(trade.pair) if trade else None
                        trade_mgr.execute_close(trade_id, action.exit_reason)
                        alert_mgr.alert_info(
                            "TRADE_CLOSED",
                            f"Trade {trade_id} closed: {action.exit_reason}",
                        )
                        if trade and exit_price:
                            if trade.direction == "long":
                                pnl_price = exit_price - trade.entry_price
                            else:
                                pnl_price = trade.entry_price - exit_price
                            pnl_pips = round(price_to_pips(pnl_price, trade.pair), 2)
                            hold_sec = int((now - trade.open_time).total_seconds())
                            logger.log_trade({
                                "trade_id": trade_id,
                                "pair": trade.pair,
                                "direction": trade.direction,
                                "order_type": "CLOSE",
                                "entry_price": trade.entry_price,
                                "exit_price": exit_price,
                                "pnl_pips": pnl_pips,
                                "exit_reason": action.exit_reason.value,
                                "hold_time_seconds": hold_sec,
                                "sl_price": trade.sl_price,
                                "tp_price": trade.tp_price,
                                "units": trade.units,
                                "broker_status": "closed",
                            })

            # Live FX for pip value in account currency (USD/GBP/JPY crosses)
            if live_prices_mid:
                risk_mgr.update_rates({
                    "GBP_USD": live_prices_mid.get("GBP_USD", 1.25),
                    "USD_JPY": live_prices_mid.get("USD_JPY", 150.0),
                })

            # Poll pending orders (Limit→fill/expire→fallback)
            if pending_mgr.pending_count > 0:
                pending_mgr.poll_all(now, live_prices_ba)

            # Reconcile with broker (detect TP/SL hit by broker)
            recon = reconciler.reconcile(now)
            for tid in recon.get("closed_by_broker", []):
                trade = recon.get("closed_trades", {}).get(tid)
                mid = None
                if trade:
                    mid = live_prices_mid.get(trade.pair)
                # Infer TP vs SL from geometry (which limit is closer to current mid)
                exit_reason = "broker_close"
                exit_price = trade.tp_price if trade else 0.0
                if trade and mid is not None:
                    if trade.direction == "long":
                        d_tp = abs(mid - trade.tp_price)
                        d_sl = abs(mid - trade.sl_price)
                        if d_tp <= d_sl:
                            exit_reason = "tp_hit"
                            exit_price = trade.tp_price
                        else:
                            exit_reason = "sl_hit"
                            exit_price = trade.sl_price
                    else:
                        d_tp = abs(mid - trade.tp_price)
                        d_sl = abs(mid - trade.sl_price)
                        if d_tp <= d_sl:
                            exit_reason = "tp_hit"
                            exit_price = trade.tp_price
                        else:
                            exit_reason = "sl_hit"
                            exit_price = trade.sl_price
                alert_msg = (
                    f"Trade {tid} closed by broker ({exit_reason})"
                    if trade else f"Trade {tid} closed by broker")
                alert_mgr.alert_info("TRADE_CLOSED_BROKER", alert_msg)
                if trade:
                    if trade.direction == "long":
                        pnl_price = exit_price - trade.entry_price
                    else:
                        pnl_price = trade.entry_price - exit_price
                    pnl_pips = round(price_to_pips(pnl_price, trade.pair), 2)
                    hold_sec = int((now - trade.open_time).total_seconds())
                    logger.log_trade({
                        "trade_id": tid,
                        "pair": trade.pair,
                        "direction": trade.direction,
                        "order_type": "CLOSE",
                        "entry_price": trade.entry_price,
                        "exit_price": exit_price,
                        "pnl_pips": pnl_pips,
                        "exit_reason": exit_reason,
                        "hold_time_seconds": hold_sec,
                        "sl_price": trade.sl_price,
                        "tp_price": trade.tp_price,
                        "units": trade.units,
                        "broker_status": "closed_by_broker",
                    })
                    if ma_brain is not None:
                        _outcome = {
                            "trade_id": tid,
                            "pair": trade.pair,
                            "direction": trade.direction,
                            "pnl_pips": pnl_pips,
                            "exit_reason": exit_reason,
                            "source": "broker_reconcile",
                            "entry_price": trade.entry_price,
                            "exit_price": exit_price,
                            "sl_price": trade.sl_price,
                            "tp_price": trade.tp_price,
                            "hold_time_seconds": hold_sec,
                            "model_id": getattr(trade, "model", "") or "",
                            "cluster_id": getattr(trade, "cluster_id", "") or "",
                        }
                        ma_brain.log_outcome(_outcome)
                        # Post-mortem (best-effort)
                        try:
                            if orchestrator is not None:
                                orchestrator._postmortem.analyze(_outcome, {
                                    "pair_stats": (
                                        ma_brain.format_pair_stats_summary(
                                            min_trades=1)),
                                    "recent_lessons": [
                                        l.get("pattern", "") for l in
                                        ma_brain.read_lessons(5)
                                    ],
                                })
                        except Exception as _e:
                            logger.warning(f"Post-mortem trigger error: {_e}")

            # Run pipeline at each minute boundary only (once per UTC minute at :00)
            if now.second != 0:
                time.sleep(0.5)
                continue

            if time.monotonic() < stabilization_end:
                time.sleep(0.5)
                continue

            minute_key = (now.year, now.month, now.day, now.hour, now.minute)
            if minute_key == last_pipeline_minute_key:
                time.sleep(0.5)
                continue
            last_pipeline_minute_key = minute_key

            # Update pilot news intel with current events
            if pilot_news_intel and news_gate._events:
                pilot_news_intel.set_events(news_gate._events)

            # Multi-Agent mode: run orchestrator instead of single pilot or rule-based
            if orchestrator and use_multi_agent:
                ma_candle_data = {}
                ma_live_prices = {}
                _m1_count = 500 if ml_gate else 100
                for pair in instruments:
                    try:
                        m1c, api_lat = feeder.fetch_candles(pair, "M1", count=_m1_count)
                        m5c, _ = feeder.fetch_candles(pair, "M5", count=200)
                        dq_gate.update_api_response(api_lat, 200)
                        if not m1c or not m5c:
                            continue
                        df_m1 = pd.DataFrame([{
                            "timestamp": c.timestamp_utc,
                            "open": c.open, "high": c.high, "low": c.low,
                            "close": c.close, "volume": c.volume
                        } for c in m1c])
                        df_m5 = pd.DataFrame([{
                            "timestamp": c.timestamp_utc,
                            "open": c.open, "high": c.high, "low": c.low,
                            "close": c.close, "volume": c.volume
                        } for c in m5c])
                        h1c, _ = feeder.fetch_candles(pair, "H1", count=100)
                        df_h1 = None
                        if h1c:
                            df_h1 = pd.DataFrame([{
                                "timestamp": c.timestamp_utc,
                                "open": c.open, "high": c.high, "low": c.low,
                                "close": c.close, "volume": c.volume
                            } for c in h1c])
                        ma_candle_data[pair] = {"m1": df_m1, "m5": df_m5, "h1": df_h1}
                        live = feeder.get_live_price(pair)
                        if live:
                            ma_live_prices[pair] = (live.bid, live.ask)
                    except Exception as e:
                        logger.error(f"Multi-agent data fetch error {pair}: {e}")

                account = executor.get_account_details()
                ma_nav = parse_account_nav(account) if account else None
                ma_margin = (parse_oanda_decimal(account.get("marginAvailable"))
                             if account else 0) or 0
                if ma_nav and ma_nav > 0:
                    orchestrator.update_rates({
                        "GBP_USD": live_prices_mid.get("GBP_USD", 1.25),
                        "USD_JPY": live_prices_mid.get("USD_JPY", 150.0),
                    })
                    cycle_result = orchestrator.run_cycle(
                        instruments=instruments,
                        candle_data=ma_candle_data,
                        live_prices=ma_live_prices,
                        nav=ma_nav, margin_available=ma_margin,
                        utc_now=now, events=news_gate._events)
                    if cycle_result.get("trades_opened", 0) > 0:
                        alert_mgr.alert_info(
                            "MULTI_AGENT_TRADE",
                            f"Multi-agent opened {cycle_result['trades_opened']} trade(s)")
                else:
                    logger.warning("Multi-agent: no valid NAV, skipping cycle")
                continue  # Skip rule-based and pilot pipelines

            # AI Pilot mode: gather all data first, call AI once for all pairs
            if pilot_pipeline and use_pilot and not pilot_shadow:
                candle_data = {}
                pilot_live_prices = {}
                pilot_nav = None
                pilot_margin = 0.0
                for pair in instruments:
                    try:
                        m1c, api_lat = feeder.fetch_candles(pair, "M1", count=100)
                        m5c, _ = feeder.fetch_candles(pair, "M5", count=50)
                        dq_gate.update_api_response(api_lat, 200)
                        if not m1c or not m5c:
                            continue
                        df_m1 = pd.DataFrame([{
                            "open": c.open, "high": c.high, "low": c.low,
                            "close": c.close, "volume": c.volume
                        } for c in m1c])
                        df_m5 = pd.DataFrame([{
                            "open": c.open, "high": c.high, "low": c.low,
                            "close": c.close, "volume": c.volume
                        } for c in m5c])
                        candle_data[pair] = {"m1": df_m1, "m5": df_m5}
                        live = feeder.get_live_price(pair)
                        if live:
                            pilot_live_prices[pair] = (live.bid, live.ask)
                    except Exception as e:
                        logger.error(f"Pilot data fetch error {pair}: {e}")

                # Get account once
                account = executor.get_account_details()
                pilot_nav = parse_account_nav(account) if account else None
                if pilot_nav and pilot_nav > 0:
                    pilot_margin = (
                        parse_oanda_decimal(account.get("marginAvailable"))
                        if account else 0.0) or 0.0
                    pilot_pipeline.update_rates({
                        "GBP_USD": live_prices_mid.get("GBP_USD", 1.25),
                        "USD_JPY": live_prices_mid.get("USD_JPY", 150.0),
                    })
                    pilot_result = pilot_pipeline.run(
                        instruments=instruments,
                        candle_data=candle_data,
                        live_prices=pilot_live_prices,
                        nav=pilot_nav,
                        margin_available=pilot_margin,
                        utc_now=now)
                    for _ in range(pilot_result.trades_opened):
                        alert_mgr.alert_info(
                            "PILOT_TRADE",
                            f"AI Pilot opened a trade ({pilot_result.trades_opened} total)")
                else:
                    logger.warning("Pilot: no valid NAV from OANDA, skipping cycle")
                continue  # Skip rule-based pipeline when pilot is active

            # Fetch candles and run pipeline for each instrument
            candle_data_cache = {}  # for shadow mode reuse
            for pair in instruments:
                try:
                    m1_candles, api_lat = feeder.fetch_candles(pair, "M1", count=100)
                    m5_candles, _ = feeder.fetch_candles(pair, "M5", count=50)
                    dq_gate.update_api_response(api_lat, 200)

                    if not m1_candles or not m5_candles:
                        continue

                    df_m1 = pd.DataFrame([{
                        "open": c.open, "high": c.high, "low": c.low,
                        "close": c.close, "volume": c.volume
                    } for c in m1_candles])
                    df_m5 = pd.DataFrame([{
                        "open": c.open, "high": c.high, "low": c.low,
                        "close": c.close, "volume": c.volume
                    } for c in m5_candles])

                    candle_data_cache[pair] = {"m1": df_m1, "m5": df_m5}

                    live = feeder.get_live_price(pair)
                    if not live:
                        continue

                    # Get account details (never default NAV — wrong NAV oversizes)
                    account = executor.get_account_details()
                    nav = parse_account_nav(account)
                    if nav is None or nav <= 0:
                        logger.warning(
                            f"[{pair}] Skipping: no valid NAV/balance from OANDA "
                            f"(account fetch {'failed' if not account else 'ok'}). "
                            "Check API token, account id, and /summary response."
                        )
                        continue
                    margin_avail = (
                        parse_oanda_decimal(account.get("marginAvailable"))
                        if account else None
                    )
                    margin_avail = margin_avail if margin_avail is not None else 0.0
                    if margin_avail < 0:
                        margin_avail = 0.0

                    # Build open positions for risk manager
                    from src.scalp_mode.execution.risk_manager import OpenPosition
                    open_pos = [
                        OpenPosition(t.pair, t.direction, t.units, 0)
                        for t in trade_mgr.open_trades
                    ]

                    result = pipeline.run(
                        pair=pair, df_m1=df_m1, df_m5=df_m5,
                        bid=live.bid, ask=live.ask,
                        nav=nav, margin_available=margin_avail,
                        open_positions=open_pos,
                        utc_now=now)

                    logger.info(
                        f"[{pair}] {result.final_decision} "
                        f"({result.no_trade_reason or 'OK'}) "
                        f"{result.pipeline_latency_ms}ms "
                        f"open={len(trade_mgr.open_trades)} "
                        f"pending={pending_mgr.pending_count}")

                    if result.final_decision == "SIGNAL_SENT":
                        t = result.trigger
                        direction = t.direction.value if t and t.direction else "?"
                        entry = f"{t.entry_price:.5f}" if t and t.entry_price else "?"
                        sl = f"{t.sl_price:.5f}" if t and t.sl_price else "?"
                        tp = f"{t.tp_price:.5f}" if t and t.tp_price else "?"
                        units = result.risk.units if result.risk else "?"
                        alert_mgr.alert_info(
                            "TRADE_SIGNAL",
                            f"{pair} {direction} | entry={entry} sl={sl} tp={tp} units={units}",
                        )

                    # Track pending Limit orders for lifecycle management
                    if (result.final_decision == "SIGNAL_SENT"
                            and hasattr(result, '_pending_info')
                            and result._pending_info):
                        pi = result._pending_info
                        pending_mgr.track(PendingOrder(
                            order_id=pi["order_id"],
                            signal_id=pi["signal_id"],
                            pair=pi["pair"],
                            direction=pi["direction"],
                            units=pi["units"],
                            entry_price=pi["entry_price"],
                            sl_price=pi["sl_price"],
                            tp_price=pi["tp_price"],
                            submitted_at=now,
                            ttl_seconds=config.orders.get("limit_ttl_seconds", 180),
                            atr=pi["atr"],
                            spread_at_signal=pi["spread_at_signal"],
                            max_spread=pi["max_spread"],
                        ))

                except Exception as e:
                    logger.error(f"Error processing {pair}: {e}")

            # Shadow mode: after rule-based executes, also run AI Pilot (log only)
            if pilot_pipeline and pilot_shadow:
                try:
                    shadow_candles = {}
                    shadow_prices = {}
                    for pair in instruments:
                        if pair in candle_data_cache:
                            shadow_candles[pair] = candle_data_cache[pair]
                        live = feeder.get_live_price(pair)
                        if live:
                            shadow_prices[pair] = (live.bid, live.ask)
                    if shadow_candles:
                        shadow_account = executor.get_account_details()
                        s_nav = parse_account_nav(shadow_account) if shadow_account else 0
                        s_margin = (parse_oanda_decimal(
                            shadow_account.get("marginAvailable"))
                            if shadow_account else 0) or 0
                        if s_nav and s_nav > 0:
                            pilot_pipeline.update_rates({
                                "GBP_USD": live_prices_mid.get("GBP_USD", 1.25),
                                "USD_JPY": live_prices_mid.get("USD_JPY", 150.0),
                            })
                            shadow_ctx = PilotContext(
                                utc_now=now, nav=s_nav,
                                margin_available=s_margin,
                                account_floor=pilot_pipeline._pilot.account_floor,
                                instruments=instruments,
                                pair_data={},
                                news_intel=pilot_news_intel.get_relevant_briefings(now)
                                    if pilot_news_intel else "",
                            )
                            shadow_actions = pilot_pipeline._pilot.evaluate(shadow_ctx)
                            pilot_pipeline.log_shadow(
                                shadow_actions, {"note": "shadow_mode"}, now)
                            logger.info(
                                f"Pilot shadow: {len(shadow_actions)} actions logged")
                except Exception as e:
                    logger.error(f"Pilot shadow error: {e}")

    finally:
        logger.info("Shutting down...")
        news_fetcher.stop_auto_update()
        if pilot_news_intel:
            pilot_news_intel.stop_background()
        feeder.stop_stream()
        cancelled = executor.cancel_pending_orders()
        if cancelled:
            logger.info(f"Cancelled pending orders: {cancelled}")
        open_count = len(trade_mgr.open_trades)
        logger.info(f"Shutdown complete. {open_count} open trades retained (have SL/TP).")
        logger.close()


def main():
    parser = argparse.ArgumentParser(description="Scalp Mode V1")
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Path to config YAML")

    sub = parser.add_subparsers(dest="mode", required=True)

    # Backtest
    bt = sub.add_parser("backtest", help="Run backtest on historical data")
    bt.add_argument("--pair", default="EUR_USD")
    bt.add_argument("--data", required=True, help="Path to CSV with OHLCV data")
    bt.add_argument("--nav", type=float, default=10000, help="Initial NAV")
    bt.add_argument("--spread", type=float, default=0.5, help="Fixed spread pips")
    bt.add_argument("--slippage", type=float, default=0.1, help="Slippage pips")
    bt.add_argument("--days", type=int, default=None, help="Trading days in data")
    bt.add_argument("--no-sessions", action="store_true",
                    help="Disable session filter")

    # Analyze (post-trade)
    an = sub.add_parser("analyze", help="Run daily post-trade analysis")
    an.add_argument("--date", default=None, help="Date YYYY-MM-DD (default: today)")
    an.add_argument("--log-dir", default=None, help="Log directory")

    # Walk-forward
    wf = sub.add_parser("walkforward", help="Run walk-forward backtest")
    wf.add_argument("--pair", default="EUR_USD")
    wf.add_argument("--data", required=True, help="Path to CSV with OHLCV data")
    wf.add_argument("--windows", type=int, default=5, help="Number of windows")
    wf.add_argument("--spread", type=float, default=0.3, help="Fixed spread pips")
    wf.add_argument("--no-sessions", action="store_true",
                    help="Disable session filter")

    # Paper
    sub.add_parser(
        "paper",
        help="Run bot against OANDA URLs in config/settings.yaml (no confirm prompt)",
    )

    # Live
    sub.add_parser(
        "live",
        help="Same as paper plus extra confirmation prompt (URLs still from settings.yaml)",
    )

    args = parser.parse_args()

    # Load config
    resolve_env = args.mode in ("paper", "live")
    config = Config(args.config, resolve_env=resolve_env)

    if args.mode == "backtest":
        run_backtest(args, config)
    elif args.mode == "analyze":
        analyst = PostTradeAnalyst(config.scalp.get("ai", {}).get("post_trade", {}))
        date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_dir = args.log_dir or config.logging_config.get("log_dir", "logs")
        report = analyst.analyze_day(log_dir, date)
        print(f"\nDaily Report — {date}")
        print(f"  Trades: {report.total_trades} (W:{report.wins} L:{report.losses})")
        print(f"  Win Rate: {report.win_rate:.0%}")
        print(f"  PnL: {report.total_pnl_pips:.1f} pips")
        print(f"  Model A: {report.model_a_trades} trades, Model B: {report.model_b_trades}")
        if report.suggestions:
            print(f"  Suggestions: {'; '.join(report.suggestions)}")
        if report.ai_summary:
            print(f"  AI: {report.ai_summary}")
    elif args.mode == "walkforward":
        run_walkforward(args, config)
    elif args.mode == "paper":
        run_paper(args, config, is_live=False)
    elif args.mode == "live":
        run_paper(args, config, is_live=True)


if __name__ == "__main__":
    main()
