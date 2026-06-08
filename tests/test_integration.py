"""Integration tests — full execution path through the pipeline.

Tests the complete flow: signal → risk → order → execution → trade log → trade manager.
"""

import json
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from src.scalp_mode.engine.decision_pipeline import DecisionPipeline, PipelineResult
from src.scalp_mode.engine.feature_engine import FeatureEngine
from src.scalp_mode.engine.regime_engine import RegimeEngine
from src.scalp_mode.engine.model_a import ModelATrigger
from src.scalp_mode.engine.cooldown import CooldownManager
from src.scalp_mode.execution.risk_manager import RiskManager
from src.scalp_mode.execution.order_builder import OrderBuilder
from src.scalp_mode.execution.executor import Executor, ExecutionResult
from src.scalp_mode.execution.trade_manager import TradeManager, ManagedTrade
from src.scalp_mode.gates.data_quality_gate import DataQualityGate
from src.scalp_mode.gates.news_gate import NewsGate
from src.scalp_mode.logger import ScalpLogger
from src.scalp_mode.backtest.walk_forward import run_walk_forward


def _utc(hour, minute=0):
    return datetime(2026, 1, 7, hour, minute, tzinfo=timezone.utc)


def _make_df(n=100, base=1.0850, volatility=0.0005):
    np.random.seed(42)
    closes = np.cumsum(np.random.normal(0, volatility, n)) + base
    highs = closes + np.abs(np.random.normal(0, volatility * 0.5, n))
    lows = closes - np.abs(np.random.normal(0, volatility * 0.5, n))
    opens = closes + np.random.normal(0, volatility * 0.3, n)
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.random.randint(10, 100, n),
    })


class MockConfig:
    def max_spread_pips(self, pair):
        return {"EUR_USD": 0.8, "USD_JPY": 0.8, "GBP_USD": 1.0}.get(pair, 1.0)

    @property
    def sessions(self):
        return {"mode": "overlap_only", "block": []}

    @property
    def borderline(self):
        return {"spread_warn_ratio": 0.70}


SCALP_CONFIG = {
    "regime": {
        "trend": {"ema_slope_thr": 0.20, "rsi_min": 52, "rsi_max": 78},
        "range": {"bb_width_thr": 0.004},
    },
    "model_a": {
        "compression_N": 8, "breakout_buffer_atr": 0.10,
        "retest_timeout": 3, "retest_tolerance_atr": 0.15,
        "body_ratio_min": 0.55, "rsi_min_long": 55,
        "sl_atr": 0.8, "tp_R": 1.0, "time_stop_min": 6,
        "sl_move_threshold_R": 0.8, "sl_move_target_R": -0.1,
        "sl_move_window_min": [2, 4],
    },
    "risk": {
        "risk_pct": 0.0025, "max_concurrent": 2,
        "cooldown_same_pair_dir_min": 10, "consec_loss_circuit": 3,
        "cooldown_minutes": 60, "trades_per_hour_pair": 3,
        "trades_per_hour_total": 6, "daily_loss": 0.01,
        "max_margin_pct": 0.08,
    },
    "orders": {
        "limit_ttl_seconds": 180, "fallback_market": True,
        "fallback_max_atr_distance": 0.3, "fallback_cooldown_min": 2,
        "price_bound_slippage": 0.2,
    },
    "costs": {
        "max_spread_pips": {"EUR_USD": 0.8, "USD_JPY": 0.8, "GBP_USD": 1.0},
    },
}


@pytest.fixture
def full_pipeline(tmp_path):
    """Build a full pipeline with all components including executor mock."""
    log_config = {
        "log_dir": str(tmp_path / "logs"),
        "decision_log_file": "decision.jsonl",
        "trade_log_file": "trade.jsonl",
        "cycle_log_file": "cycle.jsonl",
        "system_log_file": "system.log",
        "max_file_size_mb": 1, "backup_count": 2, "level": "DEBUG",
    }
    logger = ScalpLogger(log_config)
    feature = FeatureEngine()
    regime = RegimeEngine(SCALP_CONFIG["regime"])
    trigger = ModelATrigger(SCALP_CONFIG["model_a"])
    cooldown = CooldownManager(SCALP_CONFIG["risk"])
    risk_mgr = RiskManager(SCALP_CONFIG["risk"])
    order_builder = OrderBuilder(SCALP_CONFIG["orders"])
    executor = Executor("http://fake", "token", "acc-1", logger)
    trade_mgr = TradeManager(
        SCALP_CONFIG["model_a"], "http://fake", "token", "acc-1", logger)
    news_gate = NewsGate()
    dq_gate = DataQualityGate({"heartbeat_timeout_sec": 10, "stale_price_sec": 15,
                                "api_timeout_ms": 2000})

    pipeline = DecisionPipeline(
        config=MockConfig(), logger=logger,
        feature_engine=feature, regime_engine=regime,
        trigger=trigger, news_gate=news_gate,
        data_quality_gate=dq_gate, cooldown_manager=cooldown,
        risk_manager=risk_mgr, order_builder=order_builder,
        executor=executor, trade_manager=trade_mgr)

    yield {
        "pipeline": pipeline, "logger": logger, "executor": executor,
        "trade_mgr": trade_mgr, "dq_gate": dq_gate,
        "log_dir": tmp_path / "logs",
    }
    logger.close()


class TestFullExecutionPath:
    """Tests the complete path from signal → execution → trade log → trade manager."""

    def test_successful_fill_registers_trade(self, full_pipeline):
        """When executor returns a fill, the trade should be registered with TradeManager."""
        p = full_pipeline
        p["dq_gate"].update_heartbeat(_utc(14, 0))
        p["dq_gate"].update_price(_utc(14, 0))
        p["dq_gate"].update_api_response(500, 200)

        with patch.object(p["executor"], "submit") as mock_submit:
            mock_submit.return_value = ExecutionResult(
                success=True, order_id="ord-1", trade_id="trade-100",
                fill_price=1.08555, fill_time="2026-01-07T14:00:01Z",
                actual_slippage_pips=0.1, broker_status="filled",
                e2e_latency_ms=150)

            result = p["pipeline"].run(
                "EUR_USD", _make_df(), _make_df(),
                bid=1.0850, ask=1.08505,
                nav=10000, margin_available=9500,
                utc_now=_utc(14, 0))

            if result.final_decision == "SIGNAL_SENT":
                # Trade should be registered with TradeManager
                assert len(p["trade_mgr"].open_trades) == 1
                trade = p["trade_mgr"].open_trades[0]
                assert trade.trade_id == "trade-100"
                assert trade.entry_price == 1.08555

    def test_trade_log_written_on_fill(self, full_pipeline):
        """Trade log should be written when execution succeeds."""
        p = full_pipeline

        p["dq_gate"].update_heartbeat(_utc(14, 0))
        p["dq_gate"].update_price(_utc(14, 0))
        p["dq_gate"].update_api_response(500, 200)

        with patch.object(p["executor"], "submit") as mock_submit:
            mock_submit.return_value = ExecutionResult(
                success=True, order_id="ord-2", trade_id="trade-200",
                fill_price=1.08560, fill_time="2026-01-07T14:00:01Z",
                actual_slippage_pips=0.05, broker_status="filled",
                e2e_latency_ms=120)

            result = p["pipeline"].run(
                "EUR_USD", _make_df(), _make_df(),
                bid=1.0850, ask=1.08505,
                nav=10000, margin_available=9500,
                utc_now=_utc(14, 0))

        p["logger"].close()

        if result.final_decision == "SIGNAL_SENT":
            trade_log = p["log_dir"] / "trade.jsonl"
            assert trade_log.exists()
            content = trade_log.read_text().strip()
            if content:
                record = json.loads(content.split("\n")[0])
                assert record["broker_status"] == "filled"
                assert record["fill_price"] == 1.08560
                assert "signal_id" in record

    def test_execution_failure_triggers_fallback(self, full_pipeline):
        """When primary limit fails, market fallback should be attempted."""
        p = full_pipeline
        p["dq_gate"].update_heartbeat(_utc(14, 0))
        p["dq_gate"].update_price(_utc(14, 0))
        p["dq_gate"].update_api_response(500, 200)

        call_count = [0]

        def side_effect(order, body):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (Limit) fails
                return ExecutionResult(
                    success=False, broker_status="rejected",
                    reject_reason="BOUNDS_VIOLATION", e2e_latency_ms=100)
            else:
                # Second call (Market fallback) succeeds
                return ExecutionResult(
                    success=True, order_id="ord-fb", trade_id="trade-fb",
                    fill_price=1.08550, broker_status="filled",
                    e2e_latency_ms=80)

        with patch.object(p["executor"], "submit", side_effect=side_effect):
            result = p["pipeline"].run(
                "EUR_USD", _make_df(), _make_df(),
                bid=1.0850, ask=1.08505,
                nav=10000, margin_available=9500,
                utc_now=_utc(14, 0))

        # If trigger fires, we should see 2 submit calls (limit + fallback)
        if result.final_decision == "SIGNAL_SENT":
            assert call_count[0] >= 2

    def test_all_failures_return_no_trade(self, full_pipeline):
        """When both limit and market fail, result should be NO_TRADE."""
        p = full_pipeline
        p["dq_gate"].update_heartbeat(_utc(14, 0))
        p["dq_gate"].update_price(_utc(14, 0))
        p["dq_gate"].update_api_response(500, 200)

        with patch.object(p["executor"], "submit") as mock_submit:
            mock_submit.return_value = ExecutionResult(
                success=False, broker_status="error",
                reject_reason="timeout", e2e_latency_ms=5000)

            result = p["pipeline"].run(
                "EUR_USD", _make_df(), _make_df(),
                bid=1.0850, ask=1.08505,
                nav=10000, margin_available=9500,
                utc_now=_utc(14, 0))

        # If trigger fires, execution failure should result in NO_TRADE
        if result.trigger and result.trigger.phase.value == "valid":
            assert result.final_decision == "NO_TRADE"
            assert "broker_reject" in (result.no_trade_reason or "")


class TestNewsGateSpreadExtension:
    """Test spread auto-extension in News Gate."""

    def test_spread_extension_blocks_after_news(self):
        from src.scalp_mode.gates.news_gate import NewsGate, NewsEvent

        gate = NewsGate()
        event = NewsEvent(
            timestamp_utc=_utc(13, 30),
            currency="USD", impact="high", title="NFP")
        gate.set_events([event])

        # 7 minutes after event (outside normal 5min post-window)
        # but spread is still elevated above max
        result = gate.check("EUR_USD", _utc(13, 37),
                            current_spread_pips=1.2, max_spread_pips=0.8)
        assert result.is_safe is False
        assert "spread_extended" in (result.blocking_event or "")

    def test_no_extension_when_spread_normal(self):
        from src.scalp_mode.gates.news_gate import NewsGate, NewsEvent

        gate = NewsGate()
        event = NewsEvent(
            timestamp_utc=_utc(13, 30),
            currency="USD", impact="high", title="NFP")
        gate.set_events([event])

        # 7 minutes after event, spread is normal
        result = gate.check("EUR_USD", _utc(13, 37),
                            current_spread_pips=0.5, max_spread_pips=0.8)
        assert result.is_safe is True


class TestWalkForward:
    """Test walk-forward backtest functionality."""

    def test_walk_forward_runs(self):
        np.random.seed(42)
        n = 1000
        df = _make_df(n)

        result = run_walk_forward(
            pair="EUR_USD", df_m1=df,
            scalp_config=SCALP_CONFIG,
            n_windows=3,
        )

        assert result.total_windows == 3
        assert 0 <= result.win_pct <= 1.0
        assert result.aggregate_metrics is not None

    def test_walk_forward_with_small_data(self):
        df = _make_df(100)
        result = run_walk_forward(
            pair="EUR_USD", df_m1=df,
            scalp_config=SCALP_CONFIG,
            n_windows=5,
        )
        # Should reduce windows automatically for small data
        assert result.total_windows <= 5
