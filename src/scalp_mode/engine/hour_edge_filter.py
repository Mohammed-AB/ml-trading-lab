"""Hour-edge filter — per-hour directional bias gate for any model signal.

Encodes the per-hour biases discovered in the 26-year edge scan:
- LONG edge hours (UTC):  00 (68.8% WR), 07 (55.7%), 20 (71.0%)
- SHORT edge hours (UTC): 19 (63.2%), 22 (68.7%)
- DEAD hours (UTC):       01-05, 23 — no discovered mean-reversion edge

Called by the Strategy Agent on every proposal before Risk Agent. Adjusts
the proposal's confidence based on hour×direction alignment.
"""
from dataclasses import dataclass
from typing import Literal


LONG_EDGE_HOURS = {0: 0.688, 7: 0.557, 20: 0.710}
SHORT_EDGE_HOURS = {19: 0.632, 22: 0.687}
DEAD_HOURS = {1, 2, 3, 4, 5, 23}

BOOST_ALIGNED = +0.10
PENALTY_COUNTER = -0.15
# Neutral hours: pass confidence through unchanged. Previously capped at 0.50,
# but that conflicted with Risk-Agent pair-tier minimums (0.65 preferred / 0.70
# default) — any trade at a neutral UTC hour (06, 08-18, 21) became
# arithmetically unapprovable. See OBSERVATIONS.md (2026-04-20).


Action = Literal["pass", "boost", "shrink", "block"]


@dataclass
class HourEdgeVerdict:
    action: Action
    adjusted_conf: float
    tag: str
    reason: str


def score_with_hour_edge(
    direction: str,
    utc_hour: int,
    model_conf: float,
    allow_dead_hours: bool = False,
) -> HourEdgeVerdict:
    """Adjust a proposal's confidence based on historical hour bias.

    Returns a verdict where action is one of pass/boost/shrink/block.
    If the caller gets action=='block', the proposal should be dropped.
    Otherwise the caller should overwrite its confidence with adjusted_conf.
    """
    direction = (direction or "").lower()
    if direction not in ("long", "short"):
        return HourEdgeVerdict(
            action="pass", adjusted_conf=model_conf,
            tag="hour_unknown_direction",
            reason=f"direction '{direction}' not long/short")
    if not (0 <= utc_hour <= 23):
        return HourEdgeVerdict(
            action="pass", adjusted_conf=model_conf,
            tag="hour_invalid", reason=f"hour {utc_hour} out of range")

    # Dead hours — penalize but do not block (AI retains final say)
    if utc_hour in DEAD_HOURS and not allow_dead_hours:
        return HourEdgeVerdict(
            action="shrink",
            adjusted_conf=max(0.0, model_conf - 0.15),
            tag="hour_dead",
            reason=f"hour {utc_hour:02d} UTC in DEAD_HOURS — low-edge period")

    # Aligned long
    if utc_hour in LONG_EDGE_HOURS and direction == "long":
        wr = LONG_EDGE_HOURS[utc_hour]
        return HourEdgeVerdict(
            action="boost",
            adjusted_conf=min(1.0, model_conf + BOOST_ALIGNED),
            tag=f"hour_long_edge_{utc_hour:02d}",
            reason=f"long at hour {utc_hour:02d} UTC hist WR {wr:.1%}")

    # Aligned short
    if utc_hour in SHORT_EDGE_HOURS and direction == "short":
        wr = SHORT_EDGE_HOURS[utc_hour]
        return HourEdgeVerdict(
            action="boost",
            adjusted_conf=min(1.0, model_conf + BOOST_ALIGNED),
            tag=f"hour_short_edge_{utc_hour:02d}",
            reason=f"short at hour {utc_hour:02d} UTC hist WR {wr:.1%}")

    # Counter the local edge
    if utc_hour in LONG_EDGE_HOURS and direction == "short":
        wr = LONG_EDGE_HOURS[utc_hour]
        return HourEdgeVerdict(
            action="shrink",
            adjusted_conf=max(0.0, model_conf + PENALTY_COUNTER),
            tag=f"hour_counter_long_{utc_hour:02d}",
            reason=(f"short at long-edge hour {utc_hour:02d} "
                    f"(hist long WR {wr:.1%}) — counter-trend"))
    if utc_hour in SHORT_EDGE_HOURS and direction == "long":
        wr = SHORT_EDGE_HOURS[utc_hour]
        return HourEdgeVerdict(
            action="shrink",
            adjusted_conf=max(0.0, model_conf + PENALTY_COUNTER),
            tag=f"hour_counter_short_{utc_hour:02d}",
            reason=(f"long at short-edge hour {utc_hour:02d} "
                    f"(hist short WR {wr:.1%}) — counter-trend"))

    # Neutral hours — pass through unchanged (Risk tiers handle gating).
    return HourEdgeVerdict(
        action="pass",
        adjusted_conf=model_conf,
        tag="hour_neutral",
        reason=f"hour {utc_hour:02d} UTC has no strong bias")
