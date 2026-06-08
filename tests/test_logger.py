"""Tests for ScalpLogger."""

import json
import pytest
from pathlib import Path
from src.scalp_mode.logger import ScalpLogger, generate_signal_id


class TestSignalId:
    def test_unique(self):
        ids = {generate_signal_id() for _ in range(100)}
        assert len(ids) == 100

    def test_format(self):
        sid = generate_signal_id()
        assert len(sid) == 36  # UUID4 format
        assert sid.count("-") == 4


class TestScalpLogger:
    def setup_method(self, method):
        self._tmp = None

    def _make_logger(self, tmp_path):
        log_dir = tmp_path / "logs"
        config = {
            "log_dir": str(log_dir),
            "decision_log_file": "decision_log.jsonl",
            "trade_log_file": "trade_log.jsonl",
            "cycle_log_file": "cycle_log.jsonl",
            "system_log_file": "system.log",
            "max_file_size_mb": 1,
            "backup_count": 2,
            "level": "DEBUG",
        }
        return ScalpLogger(config), log_dir

    def test_decision_log_writes(self, tmp_path):
        logger, log_dir = self._make_logger(tmp_path)
        logger.log_decision({
            "pair": "EUR_USD",
            "final_decision": "NO_TRADE",
            "no_trade_reason": "session_blocked",
        })
        logger.close()

        log_file = log_dir / "decision_log.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["pair"] == "EUR_USD"
        assert record["final_decision"] == "NO_TRADE"
        assert "timestamp_utc" in record

    def test_trade_log_writes(self, tmp_path):
        logger, log_dir = self._make_logger(tmp_path)
        logger.log_trade({
            "order_type": "Limit",
            "expected_entry_price": 1.08500,
            "broker_status": "filled",
        })
        logger.close()

        log_file = log_dir / "trade_log.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["order_type"] == "Limit"
        assert "trade_id" in record
        assert "order_sent_ts" in record

    def test_cycle_log_writes(self, tmp_path):
        logger, log_dir = self._make_logger(tmp_path)
        logger.log_cycle({"heartbeat": True, "latency_ms": 150})
        logger.close()

        log_file = log_dir / "cycle_log.jsonl"
        assert log_file.exists()

    def test_system_log_writes(self, tmp_path):
        logger, log_dir = self._make_logger(tmp_path)
        logger.info("System started")
        logger.warning("High spread detected")
        logger.error("Connection lost")
        logger.close()

        log_file = log_dir / "system.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "System started" in content
        assert "High spread detected" in content
        assert "Connection lost" in content

    def test_multiple_decision_logs(self, tmp_path):
        logger, log_dir = self._make_logger(tmp_path)
        for i in range(10):
            logger.log_decision({"pair": "EUR_USD", "cycle": i})
        logger.close()

        log_file = log_dir / "decision_log.jsonl"
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 10
