"""Tests for Decision Pipeline — 13-step orchestrator."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.scalp_mode.engine.decision_pipeline import DecisionPipeline, PipelineResult
from src.scalp_mode.engine.feature_engine import FeatureEngine, IndicatorSet
from src.scalp_mode.engine.regime_engine import RegimeEngine, Regime
from src.scalp_mode.engine.model_a import ModelATrigger, TriggerSignal, TriggerPhase
from src.scalp_mode.engine.cooldown import CooldownManager
from src.scalp_mode.gates.data_quality_gate import DataQualityGate, DataQualityResult
from src.scalp_mode.gates.news_gate import NewsGate
from src.scalp_mode.logger import ScalpLogger


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


@pytest.fixture
def pipeline_components(tmp_path):
    log_config = {
        "log_dir": str(tmp_path / "logs"),
        "decision_log_file": "decision.jsonl",
        "trade_log_file": "trade.jsonl",
        "cycle_log_file": "cycle.jsonl",
        "system_log_file": "system.log",
        "max_file_size_mb": 1, "backup_count": 2, "level": "DEBUG",
    }
    logger = ScalpLogger(log_config)
    feature_engine = FeatureEngine()
    regime_config = {
        "trend": {"ema_slope_thr": 0.20, "rsi_min": 52, "rsi_max": 78},
        "range": {"bb_width_thr": 0.004},
    }
    regime_engine = RegimeEngine(regime_config)
    model_config = {
        "compression_N": 8, "breakout_buffer_atr": 0.10,
        "retest_timeout": 3, "retest_tolerance_atr": 0.15,
        "body_ratio_min": 0.55, "rsi_min_long": 55,
        "sl_atr": 0.8, "tp_R": 1.0, "time_stop_min": 6,
        "sl_move_threshold_R": 0.8, "sl_move_target_R": -0.1,
        "sl_move_window_min": [2, 4],
    }
    trigger = ModelATrigger(model_config)
    news_gate = NewsGate()
    dq_gate = DataQualityGate({"heartbeat_timeout_sec": 10, "stale_price_sec": 15,
                                "api_timeout_ms": 2000})
    cooldown = CooldownManager({
        "cooldown_same_pair_dir_min": 10, "consec_loss_circuit": 3,
        "cooldown_minutes": 60, "trades_per_hour_pair": 3,
        "trades_per_hour_total": 6, "daily_loss": 0.01,
    })

    pipeline = DecisionPipeline(
        config=MockConfig(), logger=logger,
        feature_engine=feature_engine, regime_engine=regime_engine,
        trigger=trigger, news_gate=news_gate,
        data_quality_gate=dq_gate, cooldown_manager=cooldown,
    )
    return pipeline, dq_gate, logger


class TestPipelineGates:
    def test_data_quality_blocks(self, pipeline_components):
        pipeline, dq_gate, logger = pipeline_components
        # Force heartbeat timeout
        dq_gate.update_heartbeat(_utc(13, 0))

        result = pipeline.run(
            "EUR_USD", _make_df(), _make_df(),
            bid=1.0850, ask=1.0851,
            utc_now=_utc(14, 0),  # 1 hour after heartbeat → timeout
        )
        assert result.final_decision == "NO_TRADE"
        assert "data_quality" in result.no_trade_reason
        logger.close()

    def test_session_blocks_outside_overlap(self, pipeline_components):
        pipeline, dq_gate, logger = pipeline_components
        # 03:00 UTC — way outside overlap
        result = pipeline.run(
            "EUR_USD", _make_df(), _make_df(),
            bid=1.0850, ask=1.0851,
            utc_now=_utc(3, 0),
        )
        assert result.final_decision == "NO_TRADE"
        assert "session_blocked" in result.no_trade_reason
        logger.close()

    def test_spread_blocks_when_wide(self, pipeline_components):
        pipeline, dq_gate, logger = pipeline_components
        # Very wide spread: 2 pips on EUR_USD (limit 0.8)
        result = pipeline.run(
            "EUR_USD", _make_df(), _make_df(),
            bid=1.0850, ask=1.0852,  # 2 pips
            utc_now=_utc(14, 0),  # Within overlap
        )
        assert result.final_decision == "NO_TRADE"
        assert result.no_trade_reason == "spread_too_wide"
        logger.close()

    def test_pipeline_reaches_trigger(self, pipeline_components):
        """With all gates passing, pipeline should reach trigger evaluation."""
        pipeline, dq_gate, logger = pipeline_components
        # Update DQ gate so it's healthy
        now = _utc(14, 0)
        dq_gate.update_heartbeat(now)
        dq_gate.update_price(now)
        dq_gate.update_api_response(500, 200)

        result = pipeline.run(
            "EUR_USD", _make_df(), _make_df(),
            bid=1.0850, ask=1.08505,  # 0.5 pip spread
            utc_now=now,
        )
        # Should pass gates and reach regime or trigger stage
        assert result.final_decision == "NO_TRADE"
        # The reason should NOT be any gate failure
        assert result.no_trade_reason not in (
            None, "spread_too_wide",
        )
        assert not result.no_trade_reason.startswith("session_blocked")
        assert not result.no_trade_reason.startswith("data_quality")
        logger.close()


class TestPipelineLogging:
    def test_decision_logged(self, pipeline_components, tmp_path):
        pipeline, dq_gate, logger = pipeline_components
        pipeline.run(
            "EUR_USD", _make_df(), _make_df(),
            bid=1.0850, ask=1.0851,
            utc_now=_utc(3, 0),
        )
        logger.close()

        log_file = tmp_path / "logs" / "decision.jsonl"
        assert log_file.exists()
        content = log_file.read_text().strip()
        assert len(content) > 0
        import json
        record = json.loads(content)
        assert record["pair"] == "EUR_USD"
        assert record["final_decision"] == "NO_TRADE"

    def test_latency_recorded(self, pipeline_components):
        pipeline, dq_gate, logger = pipeline_components
        result = pipeline.run(
            "EUR_USD", _make_df(), _make_df(),
            bid=1.0850, ask=1.0851,
            utc_now=_utc(3, 0),
        )
        assert result.pipeline_latency_ms >= 0
        logger.close()


class TestPipelineResult:
    def test_result_has_all_fields(self, pipeline_components):
        pipeline, dq_gate, logger = pipeline_components
        result = pipeline.run(
            "EUR_USD", _make_df(), _make_df(),
            bid=1.0850, ask=1.0851,
            utc_now=_utc(14, 0),
        )
        assert isinstance(result, PipelineResult)
        assert result.pair == "EUR_USD"
        assert result.final_decision in ("NO_TRADE", "SIGNAL_SENT")
        assert result.pipeline_latency_ms >= 0
        logger.close()
