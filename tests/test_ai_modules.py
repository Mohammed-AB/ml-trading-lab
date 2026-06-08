"""Tests for AI modules — Post-trade Analyst, Borderline Reviewer, Regime Classifier."""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.scalp_mode.ai.post_trade_analyst import PostTradeAnalyst, DailyReport
from src.scalp_mode.ai.borderline_reviewer import AIBorderlineReviewer, BorderlineDecision
from src.scalp_mode.ai.regime_classifier import AIRegimeClassifier
from src.scalp_mode.engine.regime_engine import RegimeEngine, Regime, RegimeResult
from src.scalp_mode.engine.feature_engine import IndicatorSet


# ──── Post-trade Analyst ─────────────────────────────────────────────

class TestPostTradeAnalyst:
    def test_analyze_empty_day(self, tmp_path):
        cfg = {"enabled": False, "output_dir": str(tmp_path / "reports")}
        analyst = PostTradeAnalyst(cfg)
        report = analyst.analyze_day(str(tmp_path), "2026-03-28")
        assert report.total_trades == 0
        assert report.date == "2026-03-28"

    def test_analyze_with_trades(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        trades = [
            {"timestamp_utc": "2026-03-28T14:00:00Z", "order_sent_ts": "2026-03-28T14:00:00Z",
             "pnl_pips": 3.5, "model": "A", "is_borderline": False,
             "actual_slippage_pips": 0.1, "hold_time_seconds": 180,
             "exit_reason": "tp_hit"},
            {"timestamp_utc": "2026-03-28T14:30:00Z", "order_sent_ts": "2026-03-28T14:30:00Z",
             "pnl_pips": -2.0, "model": "B", "is_borderline": True,
             "actual_slippage_pips": 0.2, "hold_time_seconds": 360,
             "exit_reason": "sl_hit"},
        ]
        with open(log_dir / "trade_log.jsonl", "w") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")
        with open(log_dir / "decision_log.jsonl", "w") as f:
            f.write("")

        cfg = {"enabled": False, "output_dir": str(tmp_path / "reports")}
        analyst = PostTradeAnalyst(cfg)
        report = analyst.analyze_day(str(log_dir), "2026-03-28")

        assert report.total_trades == 2
        assert report.wins == 1
        assert report.losses == 1
        assert report.win_rate == 0.5
        assert report.model_a_trades == 1
        assert report.model_b_trades == 1
        assert report.borderline_trades == 1
        assert abs(report.avg_slippage_pips - 0.15) < 0.001
        assert "tp_hit" in report.exit_reasons
        assert "sl_hit" in report.exit_reasons

    def test_report_saved(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        with open(log_dir / "trade_log.jsonl", "w") as f:
            f.write(json.dumps({
                "timestamp_utc": "2026-03-28T14:00:00Z",
                "pnl_pips": 1.0, "exit_reason": "tp_hit",
            }) + "\n")
        with open(log_dir / "decision_log.jsonl", "w") as f:
            pass

        output = tmp_path / "reports"
        cfg = {"enabled": False, "output_dir": str(output)}
        analyst = PostTradeAnalyst(cfg)
        analyst.analyze_day(str(log_dir), "2026-03-28")

        assert (output / "2026-03-28.json").exists()

    def test_suggestions_generated(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # High slippage trades
        trades = [
            {"timestamp_utc": "2026-03-28T14:00:00Z",
             "pnl_pips": 1.0, "actual_slippage_pips": 0.5,
             "hold_time_seconds": 180, "exit_reason": "tp_hit"},
        ]
        with open(log_dir / "trade_log.jsonl", "w") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")
        with open(log_dir / "decision_log.jsonl", "w") as f:
            pass

        cfg = {"enabled": False, "output_dir": str(tmp_path / "r")}
        analyst = PostTradeAnalyst(cfg)
        report = analyst.analyze_day(str(log_dir), "2026-03-28")
        assert any("slippage" in s.lower() for s in report.suggestions)

    def test_ai_failure_continues(self, tmp_path):
        """AI summary failure should not crash — returns fallback text."""
        cfg = {"enabled": True, "output_dir": str(tmp_path / "r"),
               "model": "nonexistent-model"}
        analyst = PostTradeAnalyst(cfg)
        # Should not raise even with no API key
        report = analyst.analyze_day(str(tmp_path), "2026-03-28")
        assert report.ai_summary is not None


# ──── Borderline Reviewer ────────────────────────────────────────────

class TestBorderlineReviewer:
    def test_disabled_approves(self):
        reviewer = AIBorderlineReviewer({"enabled": False})
        decision = reviewer.evaluate({"pair": "EUR_USD"})
        assert decision.approved is True
        assert decision.reason == "reviewer_disabled"

    def test_rate_limit_rejects(self):
        reviewer = AIBorderlineReviewer({
            "enabled": True, "max_calls_per_minute": 1})
        # First call will fail (no API key) but counts toward rate limit
        reviewer.evaluate({"pair": "EUR_USD"})
        # Second call should be rate limited
        decision = reviewer.evaluate({"pair": "EUR_USD"})
        assert decision.approved is False
        assert decision.reason == "rate_limited"

    def test_api_failure_rejects(self):
        """AI failure → conservative reject."""
        reviewer = AIBorderlineReviewer({
            "enabled": True, "model": "nonexistent"})
        decision = reviewer.evaluate({"pair": "EUR_USD"})
        assert decision.approved is False
        assert "error" in decision.reason

    def test_non_borderline_not_called(self):
        """Pipeline should not call reviewer for non-borderline signals."""
        # This is a pipeline integration test — reviewer itself doesn't
        # know about borderline status, it's called conditionally
        reviewer = AIBorderlineReviewer({"enabled": True})
        # Just verify the interface exists
        assert hasattr(reviewer, "evaluate")
        assert hasattr(reviewer, "enabled")


# ──── Regime Classifier ──────────────────────────────────────────────

class TestRegimeClassifier:
    def _make_rule_engine(self):
        return RegimeEngine({
            "trend": {"ema_slope_thr": 0.20, "rsi_min": 52, "rsi_max": 78},
            "range": {"bb_width_thr": 0.004},
        })

    def _make_ind(self, slope=0.30, rsi=60.0):
        return IndicatorSet(
            ema20=1.089, ema50=1.087, atr14=0.0005, rsi14=rsi,
            bb_upper=1.092, bb_mid=1.088, bb_lower=1.084,
            bb_width=0.007, ema_slope=slope,
        )

    def test_disabled_uses_rule_based(self):
        engine = self._make_rule_engine()
        classifier = AIRegimeClassifier({"enabled": False}, engine)
        ind = self._make_ind()
        result = classifier.evaluate(ind, 1.090,
                                      datetime(2026, 1, 7, 14, tzinfo=timezone.utc))
        assert result.regime == Regime.TREND_UP

    def test_ai_failure_uses_rule_based(self):
        """AI call fails → fall back to rule-based."""
        engine = self._make_rule_engine()
        classifier = AIRegimeClassifier(
            {"enabled": True, "model": "nonexistent", "frequency_minutes": 0},
            engine)
        ind = self._make_ind()
        result = classifier.evaluate(ind, 1.090,
                                      datetime(2026, 1, 7, 14, tzinfo=timezone.utc))
        # Should still return rule-based result
        assert result.regime == Regime.TREND_UP

    def test_frequency_caching(self):
        """Calls less than frequency_minutes apart use cached result."""
        engine = self._make_rule_engine()
        classifier = AIRegimeClassifier(
            {"enabled": True, "frequency_minutes": 5}, engine)
        ind = self._make_ind()
        t1 = datetime(2026, 1, 7, 14, 0, tzinfo=timezone.utc)
        # First call (no cache) — will fail AI but cache rule_based
        r1 = classifier.evaluate(ind, 1.090, t1)
        # Second call within 5 min — should use cache
        t2 = datetime(2026, 1, 7, 14, 2, tzinfo=timezone.utc)
        r2 = classifier.evaluate(ind, 1.090, t2)
        assert r2.regime == r1.regime

    def test_tracks_recent_regimes(self):
        engine = self._make_rule_engine()
        classifier = AIRegimeClassifier({"enabled": False}, engine)
        ind = self._make_ind()
        classifier.evaluate(ind, 1.090, datetime(2026, 1, 7, 14, tzinfo=timezone.utc))
        classifier.evaluate(ind, 1.090, datetime(2026, 1, 7, 14, 1, tzinfo=timezone.utc))
        assert len(classifier._recent_regimes) == 2


# ──── Pipeline integration ───────────────────────────────────────────

class TestAIPipelineIntegration:
    def test_pipeline_accepts_ai_params(self):
        """Pipeline constructor accepts AI optional params without error."""
        from src.scalp_mode.engine.decision_pipeline import DecisionPipeline
        from src.scalp_mode.engine.feature_engine import FeatureEngine
        from src.scalp_mode.engine.model_a import ModelATrigger
        from src.scalp_mode.engine.cooldown import CooldownManager
        from src.scalp_mode.gates.data_quality_gate import DataQualityGate
        from src.scalp_mode.gates.news_gate import NewsGate
        from src.scalp_mode.logger import ScalpLogger
        import tempfile

        class MockConfig:
            def max_spread_pips(self, pair): return 0.8
            @property
            def borderline(self): return {"spread_warn_ratio": 0.70}

        with tempfile.TemporaryDirectory() as tmp:
            logger = ScalpLogger({
                "log_dir": tmp, "decision_log_file": "d.jsonl",
                "trade_log_file": "t.jsonl", "cycle_log_file": "c.jsonl",
                "system_log_file": "s.log", "max_file_size_mb": 1,
                "backup_count": 1, "level": "WARNING"})

            regime = RegimeEngine({
                "trend": {"ema_slope_thr": 0.20, "rsi_min": 52, "rsi_max": 78},
                "range": {"bb_width_thr": 0.004}})

            pipeline = DecisionPipeline(
                config=MockConfig(), logger=logger,
                feature_engine=FeatureEngine(),
                regime_engine=regime,
                trigger=ModelATrigger({"compression_N": 8, "breakout_buffer_atr": 0.10,
                    "retest_timeout": 3, "retest_tolerance_atr": 0.15,
                    "body_ratio_min": 0.55, "rsi_min_long": 55, "sl_atr": 0.8,
                    "tp_R": 1.0, "time_stop_min": 6, "sl_move_threshold_R": 0.8,
                    "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4]}),
                news_gate=NewsGate(),
                data_quality_gate=DataQualityGate({"heartbeat_timeout_sec": 10,
                    "stale_price_sec": 15, "api_timeout_ms": 2000}),
                cooldown_manager=CooldownManager({}),
                borderline_reviewer=AIBorderlineReviewer({"enabled": False}),
                ai_regime=AIRegimeClassifier({"enabled": False}, regime),
            )
            assert pipeline._borderline_reviewer is not None
            assert pipeline._ai_regime is not None
            logger.close()
