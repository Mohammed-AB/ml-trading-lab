"""Tests for Risk Agent repeat-signal guards (dedup, SL cooldown, helpers)."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.scalp_mode.agents.brain import Brain
from src.scalp_mode.agents.risk import (
    COOLDOWN_AFTER_LOSS_MINUTES,
    RiskAgent,
    _cooldown_after_sl,
    _last_outcome_for_pair_direction,
    _pair_direction_tier,
)
from src.scalp_mode.agents.strategy import TradeProposal


def _utc(y, m, d, hh, mm, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def test_last_outcome_for_pair_direction_picks_newest():
    outcomes = [
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-17T09:20:00+00:00", "pnl_pips": 10},
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-17T09:25:00+00:00", "pnl_pips": -24, "exit_reason": "sl_hit"},
    ]
    last = _last_outcome_for_pair_direction(outcomes, "EUR_USD", "long")
    assert last["pnl_pips"] == -24


def test_cooldown_after_sl_within_window():
    outcomes = [
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-17T09:25:00+00:00",
         "pnl_pips": -24, "exit_reason": "sl_hit"},
    ]
    now = _utc(2026, 4, 17, 9, 26, 0)
    cool, ts = _cooldown_after_sl(outcomes, "EUR_USD", "long", now)
    assert cool is True
    assert ts is not None


def test_cooldown_cleared_after_tp_win():
    outcomes = [
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-17T09:20:00+00:00", "pnl_pips": 10, "exit_reason": "tp_hit"},
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-17T09:25:00+00:00", "pnl_pips": -24, "exit_reason": "sl_hit"},
        {"pair": "EUR_USD", "direction": "long", "timestamp": "2026-04-17T09:30:00+00:00", "pnl_pips": 8, "exit_reason": "tp_hit"},
    ]
    now = _utc(2026, 4, 17, 9, 31, 0)
    cool, _ = _cooldown_after_sl(outcomes, "EUR_USD", "long", now)
    assert cool is False


def test_pair_direction_tier():
    stats = {
        "EUR_USD_long": {"trades": 6, "avg_pips": -10.0},
        "USD_CHF_short": {"trades": 5, "avg_pips": 15.0},
        "AUD_USD_long": {"trades": 2, "avg_pips": -50.0},
    }
    assert _pair_direction_tier(stats, "EUR_USD", "long") == "limited"
    assert _pair_direction_tier(stats, "USD_CHF", "short") == "preferred"
    assert _pair_direction_tier(stats, "AUD_USD", "long") == "default"


def _write_outcomes(brain_dir: Path, rows: list[dict]) -> None:
    p = brain_dir / "trade_log" / "outcomes.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


@pytest.fixture
def tmp_brain(tmp_path):
    b = Brain(base_dir=str(tmp_path / "brain"))
    return b


def test_risk_rejects_duplicate_pending(tmp_brain):
    prop = TradeProposal(
        proposal_id="p1", pair="EUR_USD", direction="long",
        risk_pct=0.01, sl_pips=10, tp_pips=15, confidence=0.85,
        reasoning="test", model_source="model_c",
    )
    ra = RiskAgent(tmp_brain, account_floor=1.0, max_positions=5)
    now = _utc(2026, 4, 17, 12, 0, 0)
    v = ra.evaluate(
        prop, nav=10000.0, margin_available=9000.0,
        open_positions=[],
        utc_now=now,
        pending_orders=[{"pair": "EUR_USD", "direction": "long"}],
    )
    assert v.approved is False
    assert v.reject_reason == "pair_direction_duplicate"


def test_risk_rejects_cooldown_after_loss(tmp_brain):
    _write_outcomes(Path(tmp_brain._dir), [{
        "trade_id": "1", "pair": "EUR_USD", "direction": "long",
        "pnl_pips": -20.0, "exit_reason": "sl_hit",
        "timestamp": "2026-04-17T09:25:00+00:00",
    }])
    prop = TradeProposal(
        proposal_id="p2", pair="EUR_USD", direction="long",
        risk_pct=0.01, sl_pips=10, tp_pips=15, confidence=0.85,
        reasoning="test", model_source="model_c",
    )
    ra = RiskAgent(tmp_brain, account_floor=1.0, max_positions=5)
    now = _utc(2026, 4, 17, 9, 26, 0)
    v = ra.evaluate(
        prop, nav=10000.0, margin_available=9000.0,
        open_positions=[],
        utc_now=now,
    )
    assert v.approved is False
    assert v.reject_reason == "cooldown_after_loss"


def test_risk_accepts_after_cooldown_window(tmp_brain):
    ts_old = (_utc(2026, 4, 17, 8, 0, 0)
              - timedelta(minutes=COOLDOWN_AFTER_LOSS_MINUTES + 1))
    _write_outcomes(Path(tmp_brain._dir), [{
        "trade_id": "1", "pair": "EUR_USD", "direction": "long",
        "pnl_pips": -20.0, "exit_reason": "sl_hit",
        "timestamp": ts_old.isoformat(),
    }])
    prop = TradeProposal(
        proposal_id="p2", pair="EUR_USD", direction="long",
        risk_pct=0.01, sl_pips=10, tp_pips=15, confidence=0.85,
        reasoning="test", model_source="model_c",
    )
    ra = RiskAgent(tmp_brain, account_floor=1.0, max_positions=5)
    now = _utc(2026, 4, 17, 12, 0, 0)
    v = ra.evaluate(
        prop, nav=10000.0, margin_available=9000.0,
        open_positions=[],
        utc_now=now,
    )
    # Should pass cooldown (and likely fail later on correlation or approve)
    assert v.reject_reason != "cooldown_after_loss"
