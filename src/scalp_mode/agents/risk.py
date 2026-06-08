"""Risk Agent — Evaluates trade proposals and enforces guardrails.

Called after each Strategy Agent proposal. Approves or rejects based
on both AI analysis and hard-coded guardrails that cannot be overridden.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .brain import Brain
from .strategy import TradeProposal

_log = logging.getLogger("scalp_mode")

MAX_POSITION_UNITS = 500000
MIN_SL_PIPS = 3.0
MAX_RISK_PCT = 0.02
MAX_PER_PAIR = 1
MAX_CONCURRENT_POSITIONS = 2

CONSEC_LOSS_PAUSE_COUNT = 3
CONSEC_LOSS_PAUSE_MINUTES = 60

# Anti flip-flop: if a pair traded the opposite direction within this
# window, reject the new proposal. Prevents noise-driven reversals that
# eat spread cost without conviction.
DIRECTION_FLIP_COOLDOWN_MINUTES = 15

# After an SL on (pair, direction), block same-direction re-entry briefly.
COOLDOWN_AFTER_LOSS_MINUTES = 30

# Dynamic pair tier thresholds (rolling avg pips/trade from pair_stats.json)
PAIR_TIER_MIN_TRADES = 5
PAIR_TIER_LIMITED_AVG_PIPS = -8.0
PAIR_TIER_PREFERRED_AVG_PIPS = 10.0
# Tuned down 2026-04-20 after diagnosing that the pre-fix thresholds
# (0.70/0.65/0.80) made approval arithmetically impossible for most of
# Strategy's natural proposal range (0.55-0.70). See OBSERVATIONS.md.
DEFAULT_MIN_CONFIDENCE = 0.60
LIMITED_MIN_CONFIDENCE = 0.75
PREFERRED_MIN_CONFIDENCE = 0.55


def _parse_outcome_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _consecutive_loss_meta(outcomes: list[dict]) -> tuple[int, Optional[datetime]]:
    """From newest outcome backward: streak of losses, timestamp of newest loss in streak."""
    streak = 0
    newest_loss_ts: Optional[datetime] = None
    for o in reversed(outcomes):
        pnl = o.get("pnl_pips")
        if pnl is None:
            break
        try:
            pnl_f = float(pnl)
        except (TypeError, ValueError):
            break
        if pnl_f < 0:
            streak += 1
            if newest_loss_ts is None:
                newest_loss_ts = _parse_outcome_ts(o.get("timestamp"))
        else:
            break
    return streak, newest_loss_ts


def _norm_pair(p: str) -> str:
    return (p or "").strip().upper()


def _norm_dir(d: str) -> str:
    return (d or "").strip().lower()


def _is_cooldown_triggering_loss(o: dict) -> bool:
    """Loss that should trigger same pair+direction cooldown (SL or negative P/L)."""
    er = (o.get("exit_reason") or "").lower()
    if er == "sl_hit":
        return True
    try:
        return float(o.get("pnl_pips", 0) or 0) < 0
    except (TypeError, ValueError):
        return False


def _last_outcome_for_pair_direction(
    outcomes: list[dict], pair: str, direction: str,
) -> Optional[dict]:
    """Most recent outcome row for this pair+direction (outcomes oldest→newest in tail)."""
    p_u = _norm_pair(pair)
    d_l = _norm_dir(direction)
    best: Optional[tuple[datetime, dict]] = None
    for o in outcomes:
        if _norm_pair(o.get("pair", "")) != p_u:
            continue
        if _norm_dir(o.get("direction", "")) != d_l:
            continue
        ts = _parse_outcome_ts(o.get("timestamp"))
        if ts is None:
            continue
        if best is None or ts > best[0]:
            best = (ts, o)
    return best[1] if best else None


def _cooldown_after_sl(
    outcomes: list[dict],
    pair: str,
    direction: str,
    utc_now: datetime,
) -> tuple[bool, Optional[datetime]]:
    """True if last outcome for pair+direction was SL (loss) within COOLDOWN_AFTER_LOSS_MINUTES."""
    last = _last_outcome_for_pair_direction(outcomes, pair, direction)
    if not last or not _is_cooldown_triggering_loss(last):
        return (False, None)
    ts = _parse_outcome_ts(last.get("timestamp"))
    if not ts:
        return (False, None)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    now = utc_now if utc_now.tzinfo else utc_now.replace(tzinfo=timezone.utc)
    if now - ts > timedelta(minutes=COOLDOWN_AFTER_LOSS_MINUTES):
        return (False, None)
    return (True, ts)


def _pair_direction_tier(pair_stats: dict, pair: str, direction: str) -> str:
    """'limited' | 'preferred' | 'default' from rolling pair_stats.json."""
    key = f"{_norm_pair(pair)}_{_norm_dir(direction)}"
    slot = pair_stats.get(key) or {}
    n = int(slot.get("trades", 0) or 0)
    if n < PAIR_TIER_MIN_TRADES:
        return "default"
    avg = float(slot.get("avg_pips", 0) or 0)
    if avg <= PAIR_TIER_LIMITED_AVG_PIPS:
        return "limited"
    if avg >= PAIR_TIER_PREFERRED_AVG_PIPS:
        return "preferred"
    return "default"


def _min_confidence_for_tier(tier: str) -> float:
    if tier == "limited":
        return LIMITED_MIN_CONFIDENCE
    if tier == "preferred":
        return PREFERRED_MIN_CONFIDENCE
    return DEFAULT_MIN_CONFIDENCE


def _log_risk_rejection_alert(log_dir: str, event: str, details: dict) -> None:
    path = Path(log_dir) / "alerts.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "event": event,
            "message": event,
            "details": details,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    except IOError:
        pass


@dataclass
class RiskVerdict:
    """Risk Agent's decision on a trade proposal."""
    proposal_id: str
    approved: bool
    adjusted_units: int = 0
    adjusted_sl_pips: float = 0.0
    adjusted_tp_pips: float = 0.0
    reject_reason: str = ""
    reasoning: str = ""


class RiskAgent:
    """Evaluates proposals with both AI judgment and hard guardrails.

    Guardrails (code-enforced, cannot be overridden):
    - Max open positions (configurable)
    - Max 1 per pair
    - Min 3 pip stop loss (spread buffer)
    - Max risk per trade (tiered with session drawdown + peak NAV)
    - Max 10,000 units per position
    - Account floor
    - Consecutive loss pause (3 losses -> 60 min pause from last loss)
    - Session drawdown tiers (5/10/15% from session-start NAV)
    """

    def __init__(self, brain: Brain, account_floor: float = 200.0,
                 model: str = "claude-opus-4-20250514",
                 max_positions: int = 2,
                 *,
                 alert_manager=None,
                 log_dir: str = "logs"):
        self._brain = brain
        self._account_floor = account_floor
        self._model = model
        self._max_positions = max(1, max_positions)
        self._alert_manager = alert_manager
        self._log_dir = log_dir

    def _emit_risk_alert(self, event: str, details: dict) -> None:
        if self._alert_manager:
            try:
                self._alert_manager.alert_info(event, event, details)
            except Exception:
                pass
        else:
            _log_risk_rejection_alert(self._log_dir, event, details)

    def evaluate(
        self,
        proposal: TradeProposal,
        nav: float,
        margin_available: float,
        open_positions: list[dict],
        utc_now: datetime,
        *,
        session_start_nav: Optional[float] = None,
        peak_nav: Optional[float] = None,
        pending_orders: Optional[list[dict]] = None,
    ) -> RiskVerdict:
        """Evaluate a trade proposal. Returns verdict."""
        pending_orders = pending_orders or []
        outcomes = self._brain.read_recent_outcomes(500)
        pair_u = _norm_pair(proposal.pair)
        dir_l = _norm_dir(proposal.direction)

        # === HARD GUARDRAILS (code-enforced) ===

        # Advisory mute list: if Learning / Chief / human ops have flagged
        # a pair+direction combo in data/brain/mutes/blacklist.json, log it
        # for observability but DO NOT reject. Strategy sees the same list
        # in its prompt and is free to weigh it against live conditions.
        # We just track how often muted combos get through so humans can
        # see the trend.
        mutes = self._brain.read_mutes()
        mute_key = f"{pair_u}:{dir_l}"
        if mute_key in mutes:
            mute_reason = mutes[mute_key].get("reason", "no_reason_given")
            self._emit_risk_alert(
                "RISK_MUTED_COMBO_APPROVED",
                {
                    "pair": proposal.pair,
                    "direction": proposal.direction,
                    "mute_reason": mute_reason[:200],
                    "proposal_confidence": proposal.confidence,
                })
            _log.info(
                f"Risk Agent: note — {pair_u} {dir_l} is on advisory mute "
                f"list ({mute_reason[:100]}); proceeding with Strategy's "
                f"conviction conf={proposal.confidence:.2f}")

        # Account floor
        if nav <= self._account_floor:
            return RiskVerdict(
                proposal_id=proposal.proposal_id,
                approved=False,
                reject_reason="account_floor",
                reasoning=f"NAV ${nav:.2f} <= floor ${self._account_floor:.2f}")

        # Consecutive loss pause — bypassed for ML trades (ML threshold IS the filter)
        if proposal.model_source != "ml_gate":
            streak_tail = outcomes[-80:] if len(outcomes) > 80 else outcomes
            streak, newest_loss_ts = _consecutive_loss_meta(streak_tail)
            if streak >= CONSEC_LOSS_PAUSE_COUNT and newest_loss_ts:
                until = newest_loss_ts + timedelta(minutes=CONSEC_LOSS_PAUSE_MINUTES)
                if utc_now.tzinfo is None:
                    utc_now = utc_now.replace(tzinfo=timezone.utc)
                if newest_loss_ts.tzinfo is None:
                    newest_loss_ts = newest_loss_ts.replace(tzinfo=timezone.utc)
                if utc_now < until:
                    return RiskVerdict(
                        proposal_id=proposal.proposal_id,
                        approved=False,
                        reject_reason="consecutive_loss_pause",
                        reasoning=(
                            f"{streak} consecutive losses; pause until "
                            f"{until.strftime('%H:%M UTC')} ({CONSEC_LOSS_PAUSE_MINUTES}m)"))

        # Tiered drawdown from session-start NAV
        max_pos_cap = self._max_positions
        risk_cap = min(proposal.risk_pct, MAX_RISK_PCT)

        # No drawdown tiers or risk caps — AI sizes freely.
        # Only hard constraint: $200 NAV floor (checked above).

        # Max concurrent positions
        if len(open_positions) >= max_pos_cap:
            return RiskVerdict(
                proposal_id=proposal.proposal_id,
                approved=False,
                reject_reason="max_positions",
                reasoning=(
                    f"{len(open_positions)} open >= max {max_pos_cap} "
                    f"(tier/cap)"))

        # Pending order dedup: no second limit/market in same pair+direction
        for pend in pending_orders:
            if (_norm_pair(str(pend.get("pair", ""))) == pair_u
                    and _norm_dir(str(pend.get("direction", ""))) == dir_l):
                det = {
                    "pair": proposal.pair,
                    "direction": proposal.direction,
                    "reason": "pair_direction_duplicate_pending",
                }
                self._emit_risk_alert("RISK_REJECT_PAIR_DIR_DUPLICATE", det)
                return RiskVerdict(
                    proposal_id=proposal.proposal_id,
                    approved=False,
                    reject_reason="pair_direction_duplicate",
                    reasoning=(
                        f"Pending order already exists for {proposal.pair} "
                        f"{proposal.direction}"))

        # Open position: same pair+direction (belt-and-suspenders with max 1/pair)
        for pos in open_positions:
            if (_norm_pair(str(pos.get("pair", ""))) == pair_u
                    and _norm_dir(str(pos.get("direction", ""))) == dir_l):
                det = {
                    "pair": proposal.pair,
                    "direction": proposal.direction,
                    "reason": "pair_direction_duplicate_open",
                }
                self._emit_risk_alert("RISK_REJECT_PAIR_DIR_DUPLICATE", det)
                return RiskVerdict(
                    proposal_id=proposal.proposal_id,
                    approved=False,
                    reject_reason="pair_direction_duplicate",
                    reasoning=(
                        f"Already open: {proposal.pair} {proposal.direction}"))

        # Max 1 per pair (any direction — blocks hedged doubles)
        pair_positions = [p for p in open_positions
                          if _norm_pair(str(p.get("pair", ""))) == pair_u]
        if len(pair_positions) >= MAX_PER_PAIR:
            return RiskVerdict(
                proposal_id=proposal.proposal_id,
                approved=False,
                reject_reason="duplicate_pair",
                reasoning=f"Already holding {proposal.pair}")

        # Loss cooldown — bypassed for ML trades (ML threshold IS the filter)
        if proposal.model_source != "ml_gate":
            cool, last_loss_ts = _cooldown_after_sl(
                outcomes, proposal.pair, proposal.direction, utc_now)
            if cool and last_loss_ts:
                det = {
                    "pair": proposal.pair,
                    "direction": proposal.direction,
                    "reason": "cooldown_after_loss",
                    "last_loss_ts": last_loss_ts.isoformat(),
                }
                self._emit_risk_alert("RISK_REJECT_COOLDOWN_AFTER_LOSS", det)
                return RiskVerdict(
                    proposal_id=proposal.proposal_id,
                    approved=False,
                    reject_reason="cooldown_after_loss",
                    reasoning=(
                        f"{proposal.pair} {proposal.direction} lost recently — "
                        f"wait {COOLDOWN_AFTER_LOSS_MINUTES}m after last SL "
                        f"(last at {last_loss_ts.strftime('%H:%M UTC')})"))

        # Anti flip-flop — bypassed for ML trades
        if proposal.model_source != "ml_gate":
            now_utc = utc_now if utc_now.tzinfo else utc_now.replace(
                tzinfo=timezone.utc)
            flip_cutoff = now_utc - timedelta(
                minutes=DIRECTION_FLIP_COOLDOWN_MINUTES)
            for o in reversed(outcomes[-30:]):
                if (o.get("pair") or "").upper() != proposal.pair.upper():
                    continue
                ts = _parse_outcome_ts(o.get("timestamp"))
                if not ts:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < flip_cutoff:
                    break
                prev_dir = (o.get("direction") or "").lower()
                if prev_dir and prev_dir != proposal.direction.lower():
                    return RiskVerdict(
                        proposal_id=proposal.proposal_id,
                        approved=False,
                        reject_reason="direction_flip_cooldown",
                        reasoning=(
                        f"{proposal.pair} just traded {prev_dir} within "
                        f"{DIRECTION_FLIP_COOLDOWN_MINUTES}m — wait before reversing"))

        # Min SL
        sl_pips = max(proposal.sl_pips, MIN_SL_PIPS)

        # Max risk (tier-adjusted)
        risk_pct = min(proposal.risk_pct, risk_cap)

        # Compute units
        from ..utils.pip_utils import pip_value_in_account_ccy
        pip_val = pip_value_in_account_ccy(proposal.pair, "USD", {})
        if pip_val <= 0:
            pip_val = 0.0001

        risk_amount = nav * risk_pct
        units = int(risk_amount / (sl_pips * pip_val))

        # Max units cap
        units = min(units, MAX_POSITION_UNITS)

        if units <= 0:
            return RiskVerdict(
                proposal_id=proposal.proposal_id,
                approved=False,
                reject_reason="units_zero",
                reasoning="Computed 0 units after caps")

        # Dynamic pair tier: min confidence from rolling pair_stats (after sizing OK).
        # ML-gated proposals bypass this check — the ML probability threshold
        # (config.scalp.multi_agent.ml.probability_threshold) IS the confidence
        # gate. Re-checking it here would double-gate and reject valid ML signals.
        if proposal.model_source != "ml_gate":
            pair_stats = self._brain.read_pair_stats()
            tier = _pair_direction_tier(pair_stats, proposal.pair, proposal.direction)
            min_conf = _min_confidence_for_tier(tier)
            if float(proposal.confidence or 0) < min_conf:
                return RiskVerdict(
                    proposal_id=proposal.proposal_id,
                    approved=False,
                    reject_reason="pair_tier_confidence",
                    reasoning=(
                        f"Pair tier {tier} requires confidence >= {min_conf:.2f}, "
                        f"got {proposal.confidence:.2f}"))

        # Correlation check
        correlated = {
            frozenset({"EUR_USD", "GBP_USD"}),
            frozenset({"EUR_USD", "EUR_GBP"}),
            frozenset({"AUD_USD", "NZD_USD"}),
        }
        for pos in open_positions:
            pair_set = frozenset({proposal.pair, pos.get("pair", "")})
            if pair_set in correlated and pos.get("direction") == proposal.direction:
                return RiskVerdict(
                    proposal_id=proposal.proposal_id,
                    approved=False,
                    reject_reason="correlation",
                    reasoning=f"Correlated with open {pos.get('pair')} {pos.get('direction')}")

        # === APPROVED ===
        verdict = RiskVerdict(
            proposal_id=proposal.proposal_id,
            approved=True,
            adjusted_units=units,
            adjusted_sl_pips=sl_pips,
            adjusted_tp_pips=proposal.tp_pips,
            reasoning=(
                f"Approved: {proposal.pair} {proposal.direction} "
                f"{units} units, SL={sl_pips:.1f}, TP={proposal.tp_pips:.1f}, "
                f"risk={risk_pct:.1%} of ${nav:.0f}"),
        )

        # Log to brain
        self._brain.log_verdict(asdict(verdict))
        _log.info(f"Risk Agent: {verdict.reasoning}")
        return verdict
