"""Tests for Config loader."""

import os
import pytest
from pathlib import Path
from src.scalp_mode.config import Config


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


class TestConfigLoader:
    def test_load_without_env_resolution(self):
        cfg = Config(CONFIG_PATH, resolve_env=False)
        assert cfg.scalp["enabled"] is True
        assert cfg.scalp["version"] == "V2"

    def test_instruments(self):
        cfg = Config(CONFIG_PATH, resolve_env=False)
        assert cfg.instruments[:3] == ["EUR_USD", "USD_JPY", "GBP_USD"]
        assert "AUD_USD" in cfg.instruments
        assert "NZD_USD" in cfg.instruments
        assert len(cfg.instruments) == 6
        assert "EUR_GBP" not in cfg.instruments
        assert "USD_CHF" not in cfg.instruments

    def test_max_spread_pips(self):
        cfg = Config(CONFIG_PATH, resolve_env=False)
        assert cfg.max_spread_pips("EUR_USD") == 2.0
        assert cfg.max_spread_pips("GBP_USD") == 2.5

    def test_risk_config(self):
        cfg = Config(CONFIG_PATH, resolve_env=False)
        risk = cfg.risk
        assert risk["account_currency"] == "USD"
        assert risk["risk_pct"] == 0.25
        assert risk["max_concurrent"] == 50
        assert risk["daily_loss"] == 0.28
        assert risk["leverage"] == 33
        assert risk["margin_cap_safety"] == 0.90
        assert risk["consec_loss_circuit"] == 3

    def test_model_a_config(self):
        cfg = Config(CONFIG_PATH, resolve_env=False)
        ma = cfg.model_a
        assert ma["compression_N"] == 8
        assert ma["breakout_buffer_atr"] == 0.10
        assert ma["retest_timeout"] == 3
        assert ma["sl_atr"] == 2.0
        assert ma["tp_R"] == 2.0  # Calibrated (was 1.7)
        assert ma["time_stop_min"] == 60

    def test_dot_notation_access(self):
        cfg = Config(CONFIG_PATH, resolve_env=False)
        assert cfg.get("scalp_mode.risk.risk_pct") == 0.25
        assert cfg.get("scalp_mode.nonexistent", "default") == "default"

    def test_env_resolution_raises_when_unset(self):
        with pytest.raises(EnvironmentError, match="OANDA_ACCOUNT_ID"):
            Config(CONFIG_PATH, resolve_env=True)

    def test_env_resolution_works(self, monkeypatch):
        monkeypatch.setenv("OANDA_ACCOUNT_ID", "test-123")
        monkeypatch.setenv("OANDA_API_TOKEN", "test-token")
        cfg = Config(CONFIG_PATH, resolve_env=True)
        assert cfg.oanda_account_id == "test-123"
        assert cfg.oanda_api_token == "test-token"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path.yaml")

    def test_sessions_config(self):
        cfg = Config(CONFIG_PATH, resolve_env=False)
        sessions = cfg.sessions
        assert sessions["mode"] == "weekday_extended"
        assert "witching_hour" in sessions["block"]

    def test_data_quality_config(self):
        cfg = Config(CONFIG_PATH, resolve_env=False)
        dq = cfg.data_quality
        assert dq["heartbeat_timeout_sec"] == 10
        assert dq["stale_price_sec"] == 15

    def test_borderline_config(self):
        cfg = Config(CONFIG_PATH, resolve_env=False)
        bl = cfg.borderline
        assert bl["ema_slope_low"] == 0.15
        assert bl["ema_slope_high"] == 0.25
