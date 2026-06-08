"""Tests for hour_edge_filter — regression coverage after the 2026-04-20
neutral-hour passthrough fix (see OBSERVATIONS.md).
"""
import pytest

from src.scalp_mode.engine.hour_edge_filter import (
    BOOST_ALIGNED,
    PENALTY_COUNTER,
    score_with_hour_edge,
)


# ---------- neutral hours: passthrough, no cap (the fix) ----------

@pytest.mark.parametrize("hour", [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21])
@pytest.mark.parametrize("direction", ["long", "short"])
def test_neutral_hour_passes_confidence_unchanged(hour, direction):
    """High-conviction trades at neutral UTC hours must not be capped.

    Regression: previously LOW_CONV_CAP = 0.50 made any neutral-hour trade
    arithmetically unapprovable by Risk (min conf 0.65 preferred / 0.70
    default). Neutral hours now pass through.
    """
    v = score_with_hour_edge(direction=direction, utc_hour=hour, model_conf=0.75)
    assert v.action == "pass"
    assert v.adjusted_conf == pytest.approx(0.75)
    assert v.tag == "hour_neutral"


def test_neutral_hour_preserves_low_conf_too():
    """Low-conviction neutral trades pass through as-is, rely on Risk to gate."""
    v = score_with_hour_edge(direction="long", utc_hour=13, model_conf=0.40)
    assert v.action == "pass"
    assert v.adjusted_conf == pytest.approx(0.40)


# ---------- edge hours: boost still works ----------

@pytest.mark.parametrize("hour", [0, 7, 20])
def test_long_edge_hour_boosts_long(hour):
    v = score_with_hour_edge(direction="long", utc_hour=hour, model_conf=0.60)
    assert v.action == "boost"
    assert v.adjusted_conf == pytest.approx(0.60 + BOOST_ALIGNED)


@pytest.mark.parametrize("hour", [19, 22])
def test_short_edge_hour_boosts_short(hour):
    v = score_with_hour_edge(direction="short", utc_hour=hour, model_conf=0.60)
    assert v.action == "boost"
    assert v.adjusted_conf == pytest.approx(0.60 + BOOST_ALIGNED)


def test_boost_caps_at_one():
    v = score_with_hour_edge(direction="long", utc_hour=20, model_conf=0.95)
    assert v.adjusted_conf == pytest.approx(1.0)


# ---------- counter-edge: penalty still works ----------

def test_counter_edge_short_at_long_hour_is_shrunk():
    v = score_with_hour_edge(direction="short", utc_hour=20, model_conf=0.70)
    assert v.action == "shrink"
    assert v.adjusted_conf == pytest.approx(0.70 + PENALTY_COUNTER)


def test_counter_edge_long_at_short_hour_is_shrunk():
    v = score_with_hour_edge(direction="long", utc_hour=19, model_conf=0.70)
    assert v.action == "shrink"
    assert v.adjusted_conf == pytest.approx(0.70 + PENALTY_COUNTER)


# ---------- dead hours: still block ----------

@pytest.mark.parametrize("hour", [1, 2, 3, 4, 5, 23])
def test_dead_hours_block(hour):
    v = score_with_hour_edge(direction="long", utc_hour=hour, model_conf=0.85)
    assert v.action == "block"
    assert v.tag == "hour_dead"


def test_dead_hour_allow_override():
    v = score_with_hour_edge(
        direction="long", utc_hour=3, model_conf=0.85, allow_dead_hours=True)
    assert v.action != "block"


# ---------- regression: the 13:54 USD_CHF short scenario ----------

def test_regression_usd_chf_short_hour_13_can_reach_risk_threshold():
    """Strategy-proposed 0.60 short at hour 13 + second-opinion +0.05
    must end >= 0.65 (Risk preferred-tier min). Previously capped at 0.50 + 0.05
    = 0.55 (always rejected).
    """
    v = score_with_hour_edge(direction="short", utc_hour=13, model_conf=0.60)
    second_opinion_adjusted = v.adjusted_conf + 0.05
    assert second_opinion_adjusted >= 0.65, (
        f"Conf after hour-edge + second-opinion = {second_opinion_adjusted:.2f} "
        f"must reach Risk preferred threshold 0.65")
