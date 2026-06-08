"""Monitoring & Alerts — Kill switch and daily loss notifications.

Provides alerting for critical trading events:
- Kill switch activation (consecutive loss circuit breaker)
- Daily loss limit hit
- Heartbeat timeout (prolonged disconnection)
- Manual trade closure (time stop failures)

Supports:
- Console logging (always on)
- File-based alert log (always on)
- Webhook (configurable: Telegram, Slack, etc.)

Usage:
    monitor = AlertManager(logger, webhook_url="https://...")
    monitor.alert_kill_switch(reason, details)
    monitor.alert_daily_loss(pnl_pct, limit)
"""

import html
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

_log = logging.getLogger("scalp_mode")


class AlertLevel:
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertManager:
    """Sends alerts for critical trading events."""

    def __init__(self, log_dir: str = "logs",
                 webhook_url: Optional[str] = None,
                 webhook_type: str = "generic"):
        """Initialize alert manager.

        Args:
            log_dir: Directory for alert log file
            webhook_url: Optional webhook URL for external notifications
                         (Telegram bot, Slack incoming webhook, generic POST)
            webhook_type: "telegram", "slack", or "generic"
        """
        self._webhook_url = webhook_url
        self._webhook_type = webhook_type

        # Alert log file (append-only)
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self._alert_file = log_path / "alerts.jsonl"

    def alert_kill_switch(self, reason: str, details: Optional[dict] = None) -> None:
        """Alert: kill switch activated (consecutive losses or daily limit)."""
        self._send(
            level=AlertLevel.CRITICAL,
            event="KILL_SWITCH",
            message=f"Kill switch activated: {reason}",
            details=details,
        )

    def alert_daily_loss(self, daily_pnl_pct: float, limit: float) -> None:
        """Alert: daily loss limit reached."""
        self._send(
            level=AlertLevel.CRITICAL,
            event="DAILY_LOSS_LIMIT",
            message=(f"Daily loss limit hit: {daily_pnl_pct:.2%} "
                     f"(limit: -{limit:.1%})"),
            details={"daily_pnl_pct": daily_pnl_pct, "limit": limit},
        )

    def alert_consecutive_losses(self, count: int, cooldown_min: int) -> None:
        """Alert: consecutive loss circuit breaker."""
        self._send(
            level=AlertLevel.WARNING,
            event="CONSECUTIVE_LOSSES",
            message=(f"{count} consecutive losses — "
                     f"circuit breaker active for {cooldown_min} min"),
            details={"count": count, "cooldown_minutes": cooldown_min},
        )

    def alert_heartbeat_timeout(self, elapsed_sec: float) -> None:
        """Alert: prolonged heartbeat/connection timeout."""
        self._send(
            level=AlertLevel.WARNING,
            event="HEARTBEAT_TIMEOUT",
            message=f"No data from broker for {elapsed_sec:.0f}s",
            details={"elapsed_seconds": elapsed_sec},
        )

    def alert_trade_close_failed(self, trade_id: str, pair: str,
                                  reason: str) -> None:
        """Alert: failed to close a trade (time stop / kill switch)."""
        self._send(
            level=AlertLevel.CRITICAL,
            event="CLOSE_FAILED",
            message=f"Failed to close trade {trade_id} ({pair}): {reason}",
            details={"trade_id": trade_id, "pair": pair, "reason": reason},
        )

    def alert_info(self, event: str, message: str,
                   details: Optional[dict] = None) -> None:
        """General info alert."""
        self._send(AlertLevel.INFO, event, message, details)

    def _send(self, level: str, event: str, message: str,
              details: Optional[dict] = None) -> None:
        """Send alert to all channels."""
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "message": message,
            "details": details,
        }

        # Console
        if level == AlertLevel.CRITICAL:
            _log.critical(f"[ALERT] {message}")
        elif level == AlertLevel.WARNING:
            _log.warning(f"[ALERT] {message}")
        else:
            _log.info(f"[ALERT] {message}")

        # File
        try:
            with open(self._alert_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        except IOError:
            pass

        # Webhook (async to not block trading)
        if self._webhook_url:
            threading.Thread(
                target=self._send_webhook, args=(record,),
                daemon=True).start()

    def _send_webhook(self, record: dict) -> None:
        """Send alert via webhook (runs in background thread)."""
        try:
            if self._webhook_type == "telegram":
                self._send_telegram(record)
            elif self._webhook_type == "slack":
                self._send_slack(record)
            else:
                self._send_generic(record)
        except Exception as e:
            _log.error(f"Webhook alert failed: {e}")

    def _send_telegram(self, record: dict) -> None:
        """Send via Telegram Bot API.

        Uses HTML so underscores and special chars in event names do not break
        Telegram's legacy Markdown parser (e.g. MANUAL_TEST).

        webhook_url format: https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<CHAT_ID>
        """
        level = html.escape(str(record["level"]))
        event = html.escape(str(record["event"]))
        message = html.escape(str(record["message"]))
        text = f"<b>{level}</b> | {event}\n{message}"
        requests.post(self._webhook_url,
                      json={"text": text, "parse_mode": "HTML"},
                      timeout=10)

    def _send_slack(self, record: dict) -> None:
        """Send via Slack Incoming Webhook."""
        icon = {"CRITICAL": ":rotating_light:", "WARNING": ":warning:"}.get(
            record["level"], ":information_source:")
        text = f"{icon} *{record['event']}*: {record['message']}"
        requests.post(self._webhook_url, json={"text": text},
                      timeout=10)

    def _send_generic(self, record: dict) -> None:
        """Send via generic POST webhook."""
        requests.post(self._webhook_url, json=record, timeout=10)
