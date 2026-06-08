"""Structured logging for Scalp Mode V1.

Implements the logging schema from spec section 0.6:
- decision_log: every M1 close evaluation (JSONL)
- trade_log: every order sent (JSONL)
- cycle_log: system health/cycle metrics (JSONL)
- system_log: general application log (text)

All logs use UTC timestamps. No log write failure should crash the system.
"""

import json
import logging
import logging.handlers
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def generate_signal_id() -> str:
    """Generate a unique signal ID (UUID4) per spec A.6 idempotency requirement."""
    return str(uuid.uuid4())


class JsonlWriter:
    """Append-only JSONL file writer with rotation."""

    def __init__(self, filepath: Path, max_bytes: int, backup_count: int):
        self._filepath = filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._handler = logging.handlers.RotatingFileHandler(
            filename=str(filepath),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

    def write(self, record: dict) -> None:
        """Write a single JSON record as one line."""
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            self._handler.emit(
                logging.LogRecord(
                    name="jsonl",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=line,
                    args=None,
                    exc_info=None,
                )
            )
        except Exception as e:
            # Per spec: no log failure should crash the system
            logging.getLogger("scalp_mode").error(f"Failed to write log: {e}")

    def close(self) -> None:
        self._handler.close()


class ScalpLogger:
    """Central logger for Scalp Mode V1.

    Usage:
        logger = ScalpLogger(config.logging_config)
        logger.log_decision({...})
        logger.log_trade({...})
    """

    def __init__(self, log_config: dict):
        log_dir = Path(log_config.get("log_dir", "logs"))
        max_bytes = log_config.get("max_file_size_mb", 50) * 1024 * 1024
        backup_count = log_config.get("backup_count", 10)

        self._decision_writer = JsonlWriter(
            log_dir / log_config.get("decision_log_file", "decision_log.jsonl"),
            max_bytes, backup_count,
        )
        self._trade_writer = JsonlWriter(
            log_dir / log_config.get("trade_log_file", "trade_log.jsonl"),
            max_bytes, backup_count,
        )
        self._cycle_writer = JsonlWriter(
            log_dir / log_config.get("cycle_log_file", "cycle_log.jsonl"),
            max_bytes, backup_count,
        )

        # System log (text)
        self._sys_logger = logging.getLogger("scalp_mode")
        self._sys_logger.setLevel(getattr(logging, log_config.get("level", "INFO")))
        if not self._sys_logger.handlers:
            sys_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_dir / log_config.get("system_log_file", "system.log")),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            sys_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            ))
            self._sys_logger.addHandler(sys_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
            self._sys_logger.addHandler(console_handler)

    # --- Decision Log (every M1 close) ---

    def log_decision(self, record: dict) -> None:
        """Log a decision pipeline evaluation.

        Expected fields (from spec 0.6 decision_log):
            timestamp_utc, pair, session_allowed, news_safe,
            spread_pips_at_signal, spread_ok, regime, regime_values,
            trigger_result, trigger_values, is_borderline, borderline_flags,
            risk_approved, final_decision, no_trade_reason,
            pipeline_latency_ms, data_quality_ok, data_quality_issue
        """
        record.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        self._decision_writer.write(record)

    # --- Trade Log (every order sent) ---

    def log_trade(self, record: dict) -> None:
        """Log a trade execution.

        Expected fields (from spec 0.6 trade_log):
            trade_id, decision_log_ref, order_type, expected_entry_price,
            price_bound, fill_price, actual_slippage_pips,
            spread_at_signal, spread_at_fill, order_sent_ts,
            fill_received_ts, e2e_latency_ms, broker_status,
            reject_reason, sl_price, tp_price, exit_price, exit_reason,
            hold_time_seconds, pnl_pips, pnl_pct_nav, realized_rr
        """
        record.setdefault("trade_id", generate_signal_id())
        record.setdefault("order_sent_ts", datetime.now(timezone.utc).isoformat())
        self._trade_writer.write(record)

    # --- Cycle Log (system health) ---

    def log_cycle(self, record: dict) -> None:
        """Log a cycle health record (heartbeat, latency, etc.)."""
        record.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        self._cycle_writer.write(record)

    # --- System Log (general) ---

    def info(self, msg: str, **kwargs: Any) -> None:
        self._sys_logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._sys_logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._sys_logger.error(msg, **kwargs)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._sys_logger.debug(msg, **kwargs)

    # --- Cleanup ---

    def close(self) -> None:
        self._decision_writer.close()
        self._trade_writer.close()
        self._cycle_writer.close()
        for handler in self._sys_logger.handlers[:]:
            handler.close()
            self._sys_logger.removeHandler(handler)
