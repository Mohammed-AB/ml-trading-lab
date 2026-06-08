"""Strategy Agent — Proposes trades based on models, indicators, and brain knowledge.

Runs every 1 minute. Reads all models (A, B, C), indicators, regime,
Research Agent's market brief, historical stats, and lessons learned.
Proposes 0-1 trade per cycle. Does NOT execute — sends to Risk Agent.
"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from .brain import Brain

_log = logging.getLogger("scalp_mode")


@dataclass
class TradeProposal:
    """A proposed trade from the Strategy Agent."""
    proposal_id: str
    pair: str
    direction: str  # long, short
    risk_pct: float
    sl_pips: float
    tp_pips: float
    confidence: float
    reasoning: str
    model_source: str  # "model_a", "model_b", "model_c", "own_analysis"
    timestamp: str = ""
    exit_plan: str = ""  # Pre-committed rules for when to close/move SL


class StrategyAgent:
    """Analyzes markets and proposes trades.

    Does not execute. Proposals go to Risk Agent for approval.
    Limited to 1 proposal per cycle to prevent overtrading.
    """

    def __init__(self, brain: Brain, model: str = "claude-opus-4-20250514"):
        self._brain = brain
        self._model = model
        try:
            from .second_opinion import SecondOpinionAgent
            self._second_opinion = SecondOpinionAgent(brain, model)
        except Exception:
            self._second_opinion = None

    def run(self, instruments: list[str], pair_data: dict,
            open_positions: list[dict], nav: float,
            utc_now: datetime) -> Optional[TradeProposal]:
        """Analyze all pairs and propose at most 1 trade."""

        # Build context from brain
        market_state = self._brain.read_market_state()
        lessons = self._brain.read_lessons(50)
        daily_reviews = self._brain.read_daily_reviews(3)
        strategies = self._brain.read_strategies()
        pair_stats_block = self._brain.format_pair_stats_for_strategy_prompt(
            min_trades=1)
        recent_outcomes = self._brain.read_recent_outcomes(10)
        mutes = self._brain.read_mutes()

        # System prompt with accumulated knowledge
        system = (
            "You are the Strategy Agent for a forex scalping system. "
            "Your job is to analyze market data and propose at most ONE trade. "
            "You have FOUR candidate rule models (A, B, C, D) plus your own "
            "analysis. NOTE: none of these has a proven live edge — honest "
            "out-of-sample evaluation shows the rule strategies lose money, so "
            "treat them as research signals, not money-makers.\n\n"
            "MODELS:\n"
            "- Model A: Bollinger compression + breakout (trend-favored)\n"
            "- Model B: Range reversal at support/resistance (range-favored)\n"
            "- Model C: EMA 5/40 crossover\n"
            "- Model D: Hour-conditional mean reversion (Strategy P)\n"
            "  * Long at hours 00/07/20 UTC after >=10 pip drop in last 5 bars\n"
            "  * Short at hours 19/22 UTC after >=10 pip rally in last 5 bars\n"
            "- Model E: VWAP reversion during London/NY (07:00-16:00 UTC)\n"
            "  * Fires when price extends >=2.5 ATR from session VWAP\n\n"
            "CONVICTION CLUSTERING:\n"
            "- When 2+ models align on the same pair+direction, conviction is HIGH\n"
            "- When only 1 model fires, conviction is MEDIUM — consider smaller size\n"
            "- Regime contradicting the model is a risk factor, not an auto-skip\n\n"
            "GUIDANCE (use your judgment — none of these are hard rules):\n"
            "1. Propose at most 1 trade per cycle. Pick the BEST setup.\n"
            "2. Always include stop loss and take profit. Size SL to the setup — "
            "tight scalps (3-5 pip SL) are fine when geometry supports it.\n"
            "3. Size your risk freely based on conviction. The only hard floor is $200 NAV.\n"
            "4. Avoid proposing trades on pairs you already hold (Risk will block dupes).\n"
            "5. RSI alone is weak (~52% reliable) — combine with model signals.\n"
            "6. Model C on AUD_USD has historically been a strong signal.\n"
            "7. EMA crossover on GBP_USD has a negative backtest — weigh this, don't auto-skip.\n"
            "8. Solo model fires CAN be taken if geometry, regime, and context align.\n"
            "9. When Model D fires at its edge hours, it has strong backtest support.\n"
            "10. You have FULL AUTONOMY. Weigh all the context (lessons, research, "
            "regime, geometry) and make YOUR call. If a setup is genuinely good, take it.\n"
            "11. You are a SCALPER. Scalpers trade in ALL market conditions — volatile, "
            "quiet, headline-driven, uncertain. You manage risk through POSITION SIZE and "
            "STOP LOSS, not by sitting out. Skipping 5+ consecutive cycles is a failure. "
            "If macro is scary, SIZE DOWN (0.5% risk, tight SL) but still take the best "
            "available setup. News headlines are ALREADY priced into the candles you see.\n"
            "12. Geopolitical events (wars, tariffs, ceasefire deadlines) create VOLATILITY "
            "which is OPPORTUNITY for scalpers, not a reason to hide. Reduce size, tighten "
            "stops, but TRADE.\n\n"
        )

        system += pair_stats_block + "\n\n"

        if strategies:
            system += f"BACKTEST DATA:\n{strategies}\n\n"

        if lessons:
            # Lessons come sorted by confidence desc; show top ones
            system += "LESSONS LEARNED (sorted by confidence):\n"
            for l in lessons[:40]:
                conf = float(l.get("confidence", 0) or 0)
                tag = f"[{conf:.2f}]" if conf > 0 else ""
                system += f"- {tag} {l.get('pattern', '')}\n"
            system += "\n"

        if mutes:
            # Advisory list flagged by Learning Agent / Chief / human ops.
            # NOT a hard gate — you are free to override with conviction,
            # but the Risk Agent will log any mute-listed trade for review.
            system += "ADVISORY MUTE LIST (pair+direction combos recently flagged):\n"
            for key, entry in sorted(mutes.items()):
                reason = entry.get("reason", "")[:200]
                system += f"- {key}: {reason}\n"
            system += (
                "Use this list as context, not a rule. If live conditions "
                "genuinely support a muted setup, take it with reduced size "
                "and explicit reasoning for overriding.\n\n")

        if daily_reviews != "No daily reviews yet.":
            system += f"RECENT DAILY REVIEWS:\n{daily_reviews}\n\n"

        # User message with current data
        parts = [
            f"Time: {utc_now.strftime('%Y-%m-%d %H:%M UTC')} "
            f"({utc_now.strftime('%A')})",
            f"NAV: ${nav:.2f}",
        ]

        # ML gate (orchestrator injects ml_review on the scored pair)
        ml_reviews = [
            row.get("ml_review")
            for row in pair_data.values()
            if row.get("ml_review")
        ]
        if ml_reviews:
            parts.append(
                "\n=== ML SIGNAL (you are the reviewer, not the generator) ===\n"
                + "\n".join(ml_reviews)
                + "\n=== End ML signal ===\n"
            )

        # Market brief from Research Agent
        if market_state.get("analysis"):
            parts.append(f"\nMARKET BRIEF:\n{market_state['analysis']}")

        # Open positions
        if open_positions:
            parts.append("\nOPEN POSITIONS:")
            for p in open_positions:
                parts.append(
                    f"  {p.get('pair')} {p.get('direction')} "
                    f"unrealized={p.get('unrealized_pnl', 0):+.1f} pips")
        else:
            parts.append("\nNo open positions.")

        # Recent closed trades (Tier 4.11 — learn from immediate outcomes)
        if recent_outcomes:
            parts.append("\nRECENT CLOSED TRADES (last 10):")
            for o in recent_outcomes[-10:]:
                parts.append(
                    f"  {o.get('pair', '?')} {o.get('direction', '?')} "
                    f"{o.get('pnl_pips', 0):+.1f} pips "
                    f"({o.get('exit_reason', '?')})")

        # Per-pair data with model signals and historical context
        parts.append("\nMARKET DATA:")
        for pair in instruments:
            pd_entry = pair_data.get(pair, {})
            if not pd_entry:
                continue

            hist = self._brain.read_historical(pair)
            hist_summary = hist[:500] if len(hist) > 500 else hist

            ind_m5 = pd_entry.get("indicators_m5", {})
            ind_m1 = pd_entry.get("indicators_m1", {})
            regime = pd_entry.get("regime", "unknown")

            # Regime-conditional annotations (Tier 4.12)
            # Model A (Bollinger compression) favors Trend regimes.
            # Model B (Range reversal) only fires in Range — already gated.
            # Model C (EMA crossover) favors Range or early trend.
            a_sig = pd_entry.get("model_a_signal", "N/A")
            b_sig = pd_entry.get("model_b_signal", "N/A")
            c_sig = pd_entry.get("model_c_signal", "N/A")
            a_note = ""
            c_note = ""
            if "trend" not in regime.lower() and "entry=" in a_sig:
                a_note = "  [REGIME WARN: Model A favors Trend; current regime differs]"
            if "notrade" in regime.lower():
                a_note = "  [REGIME: NoTrade — low confidence environment, use caution]"
                c_note = "  [REGIME: NoTrade — low confidence environment, use caution]"

            d_sig = pd_entry.get("model_d_signal", "N/A")
            e_sig = pd_entry.get("model_e_signal", "N/A")
            mtf = pd_entry.get("mtf_context") or ""
            pair_brief = market_state.get(
                "per_pair_briefs", {}).get(pair, {})
            pair_brief_str = ""
            if pair_brief:
                pair_brief_str = (
                    f"Research bias: {pair_brief.get('bias', '?')} "
                    f"(conviction {pair_brief.get('conviction', '?')}) "
                    f"— {pair_brief.get('key_driver', '')[:120]}\n")

            news_risk = pd_entry.get("news_elevated_risk", "")
            news_warn = (
                f"NEWS RISK WINDOW: {news_risk} — reduce size or skip\n"
                if news_risk else "")

            parts.append(
                f"\n--- {pair} ---\n"
                f"Bid={pd_entry.get('bid', 0):.5f} Ask={pd_entry.get('ask', 0):.5f} "
                f"Spread={pd_entry.get('spread_pips', 0):.1f} pips\n"
                f"Regime: {regime}\n"
                + (f"MTF: {mtf}\n" if mtf else "")
                + pair_brief_str
                + news_warn
                + f"M5: RSI={ind_m5.get('rsi14', 0):.1f} "
                f"EMA20={ind_m5.get('ema20', 0):.5f} "
                f"Slope={ind_m5.get('ema_slope', 0):.4f} "
                f"ATR={ind_m5.get('atr14', 0):.6f}\n"
                f"M1: RSI={ind_m1.get('rsi14', 0):.1f}\n"
                f"Model A: {a_sig}{a_note}\n"
                f"Model B: {b_sig}\n"
                f"Model C: {c_sig}{c_note}\n"
                f"Model D (hour mean-reversion): {d_sig}\n"
                f"Model E (VWAP reversion): {e_sig}\n"
                f"Historical: {hist_summary[:200]}"
            )

        # Instructions
        parts.append(
            "\nRESPOND WITH EXACTLY ONE OF:\n"
            "TRADE:\n"
            "  PAIR: <pair>\n"
            "  DIRECTION: LONG | SHORT\n"
            "  RISK_PCT: <0.005-0.02>\n"
            "  SL_PIPS: <3-50>\n"
            "  TP_PIPS: <3-50>\n"
            "  CONFIDENCE: <0.5-1.0>\n"
            "  MODEL: model_a | model_b | model_c | model_d | model_e | own_analysis\n"
            "  REASONING: <why this is the best trade right now>\n"
            "  EXIT_PLAN: <specific rules, e.g. \"close if breaks 1.1775, "
            "move SL to BE at 50% TP, take partial at +5 pips\">\n\n"
            "OR:\n"
            "SKIP:\n"
            "  REASONING: <why no good setups exist>\n"
        )

        user_message = "\n".join(parts)

        try:
            import anthropic
            import httpx
            client = anthropic.Anthropic(
                timeout=httpx.Timeout(45.0, connect=10.0))
            response = client.messages.create(
                model=self._model,
                max_tokens=1200,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text.strip()
            _log.info(f"Strategy Agent:\n{text}")
        except Exception as e:
            _log.error(f"Strategy Agent failed: {e}")
            return None

        # Parse response
        proposal = self._parse_response(text, utc_now)

        # Apply hour edge filter (Wave 1b)
        if proposal:
            try:
                from ..engine.hour_edge_filter import score_with_hour_edge
                verdict = score_with_hour_edge(
                    direction=proposal.direction,
                    utc_hour=utc_now.hour,
                    model_conf=proposal.confidence)
                if verdict.action == "block":
                    _log.info(
                        f"Strategy: hour-edge BLOCK {proposal.pair} "
                        f"{proposal.direction} — {verdict.reason}")
                    # Log the skip and return None
                    self._brain.log_proposal({
                        "action": "SKIP",
                        "reasoning": f"hour_edge_block: {verdict.reason}",
                    })
                    return None
                if verdict.action in ("boost", "shrink", "pass"):
                    old_conf = proposal.confidence
                    proposal.confidence = verdict.adjusted_conf
                    proposal.reasoning = (
                        f"[{verdict.tag}] " + proposal.reasoning)
                    if abs(old_conf - verdict.adjusted_conf) > 0.01:
                        _log.info(
                            f"Strategy: hour-edge {verdict.action} "
                            f"{proposal.pair} {proposal.direction} "
                            f"conf {old_conf:.2f} -> {verdict.adjusted_conf:.2f} "
                            f"({verdict.reason})")
            except Exception as e:
                _log.warning(f"hour_edge_filter error: {e}")

        # Second opinion disabled — Strategy has full autonomy.
        # The hour-edge filter already adjusts confidence for timing.

        # Log to brain
        if proposal:
            self._brain.log_proposal(asdict(proposal))
        else:
            self._brain.log_proposal({
                "action": "SKIP",
                "reasoning": text[:200],
            })

        return proposal

    def _parse_response(self, text: str, utc_now: datetime) -> Optional[TradeProposal]:
        if "SKIP" in text.upper() and "TRADE" not in text.split("SKIP")[0]:
            return None

        fields = {}
        for line in text.split("\n"):
            stripped = line.strip()
            if ":" in stripped:
                key, _, value = stripped.partition(":")
                key = key.strip().upper()
                value = value.strip()
                if key in ("PAIR", "DIRECTION", "RISK_PCT", "SL_PIPS",
                           "TP_PIPS", "CONFIDENCE", "MODEL", "REASONING",
                           "EXIT_PLAN"):
                    fields[key] = value

        if not fields.get("PAIR") or not fields.get("DIRECTION"):
            return None

        try:
            sl_pips = float(fields.get("SL_PIPS", 12))
            tp_pips = float(fields.get("TP_PIPS", 18))
            risk_pct = float(fields.get("RISK_PCT", 0.015))
            confidence = float(fields.get("CONFIDENCE", 0.6))
        except (ValueError, TypeError):
            sl_pips, tp_pips, risk_pct, confidence = 12, 18, 0.015, 0.6

        sl_pips = max(sl_pips, 3.0)
        risk_pct = min(max(risk_pct, 0.005), 0.02)

        return TradeProposal(
            proposal_id=f"prop-{uuid.uuid4().hex[:8]}",
            pair=fields["PAIR"].upper().replace("/", "_"),
            direction=fields["DIRECTION"].lower(),
            risk_pct=risk_pct,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            confidence=confidence,
            reasoning=fields.get("REASONING", ""),
            model_source=fields.get("MODEL", "own_analysis").lower(),
            timestamp=utc_now.isoformat(),
            exit_plan=fields.get("EXIT_PLAN", ""),
        )
