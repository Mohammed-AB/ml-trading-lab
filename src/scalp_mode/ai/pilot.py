"""AI Pilot — Autonomous trading decision-maker.

Receives all market data, account state, trade history, and news
intelligence, then returns one or more actions: TRADE, SKIP, CLOSE,
or MODIFY_SL.  Only two hard limits exist externally: news blackouts
and account floor.  Everything else is at the AI's discretion.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .pilot_knowledge import get_knowledge_base
from .pilot_journal import PilotJournal
from .pilot_news_intel import NewsIntelligence

_log = logging.getLogger("scalp_mode")


@dataclass
class PilotAction:
    """A single action the AI wants to take."""
    decision: str           # TRADE, SKIP, CLOSE, MODIFY_SL
    pair: str = ""
    direction: str = ""     # long, short
    risk_pct: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    trade_id: str = ""      # for CLOSE / MODIFY_SL
    new_sl_price: float = 0.0  # for MODIFY_SL
    signal_id: str = ""     # auto-generated


@dataclass
class PilotContext:
    """Full context passed to the AI each minute."""
    utc_now: datetime
    nav: float
    margin_available: float
    account_floor: float
    instruments: list[str] = field(default_factory=list)
    # Per-pair data: {pair: {bid, ask, spread_pips, indicators_m1, indicators_m5, regime}}
    pair_data: dict = field(default_factory=dict)
    open_positions: list[dict] = field(default_factory=list)
    session_stats: dict = field(default_factory=dict)
    recent_trades: str = ""
    journal: str = ""
    news_intel: str = ""


class AIPilot:
    """Autonomous AI decision-maker for forex scalping.

    Usage:
        pilot = AIPilot(config, journal, news_intel)
        context = pilot.build_context(...)
        actions = pilot.evaluate(context)
    """

    def __init__(self, config: dict, journal: PilotJournal,
                 news_intel: NewsIntelligence):
        self._model = config.get("model", "claude-opus-4-20250514")
        self._account_floor = config.get("account_floor_usd", 200)
        self._log_reasoning = config.get("log_reasoning", True)
        self._journal = journal
        self._news_intel = news_intel
        self._knowledge = get_knowledge_base()
        self._call_count = 0
        self._max_trades = config.get("max_trades", 0)  # 0 = unlimited
        self._trades_executed = 0

    @property
    def account_floor(self) -> float:
        return self._account_floor

    def build_system_prompt(self) -> str:
        """Build the full system prompt with knowledge + journal."""
        journal_context = self._journal.load_recent_journal(days=7)
        recent_trades = self._journal.load_recent_trades(count=20)
        session_stats = self._journal.get_session_stats()

        stats_str = (
            f"Today's stats: {session_stats['trades_opened']} trades opened, "
            f"{session_stats['trades_closed']} closed, "
            f"{session_stats['wins']} wins, {session_stats['losses']} losses, "
            f"win rate {session_stats['win_rate']:.0%}, "
            f"P/L {session_stats['total_pnl_pips']:+.1f} pips"
        )

        return (
            f"{self._knowledge}\n\n"
            f"=== YOUR TRADING JOURNAL ===\n{journal_context}\n\n"
            f"=== YOUR RECENT TRADES ===\n{recent_trades}\n\n"
            f"=== SESSION STATUS ===\n{stats_str}\n\n"
            f"CRITICAL RULES:\n"
            f"1. You MUST always set a stop loss on every trade. No exceptions.\n"
            f"2. Account floor is ${self._account_floor}. If NAV is near this, "
            f"be extremely conservative or stop trading.\n"
            f"3. Always explain your reasoning. Learn from your past trades.\n"
            f"4. Quality over quantity. Fewer high-confidence trades > many mediocre ones.\n"
            f"5. Respond ONLY in the structured format specified. No extra text.\n"
            f"6. DO NOT move stop losses in the first 3-5 minutes after entry.\n"
            f"   Only move SL to breakeven after price has moved 50%+ toward TP.\n"
            f"   Do NOT issue MODIFY_SL every minute — wait for significant progress.\n"
            f"   Premature stop-tightening kills profits. Let trades breathe.\n"
            f"7. USE MODEL SIGNALS AS YOUR PRIMARY TOOLS, NOT AS GATEKEEPERS.\n"
            f"   You have three models: A (breakout), B (reversal), C (EMA crossover).\n"
            f"   When a model shows VALID with entry/SL/TP, treat it as one research signal.\n"
            f"   None of these models has a proven live edge — out-of-sample tests show\n"
            f"   they lose money, so do NOT treat any of them as a guaranteed winner.\n"
            f"   You CAN trade without a model signal if you see a strong setup, but\n"
            f"   model-confirmed trades should be your bread and butter.\n"
            f"   Use your judgment: combine model signals with indicators, context,\n"
            f"   and your trading journal to make the best decision.\n"
            f"8. SESSION AWARENESS:\n"
            f"   - Asian session (22:00-07:00 UTC): Very low volatility. Most pairs\n"
            f"     range tightly. Be patient during Asian session.\n"
            f"   - London (07:00-16:00 UTC): Best session for breakouts and trends.\n"
            f"     If models are not firing during London, USE YOUR OWN ANALYSIS.\n"
            f"     You are a smart AI — read the indicators, find setups, trade them.\n"
            f"   - NY overlap (12:00-16:00 UTC): Peak liquidity, best setups.\n"
            f"     You should be ACTIVELY LOOKING for trades during this window.\n"
            f"   - If it is London or NY overlap and you have been skipping for 30+\n"
            f"     minutes, something is wrong. Look harder. Use your own analysis\n"
            f"     beyond just waiting for models. The models are tools, not the\n"
            f"     only way to find trades.\n"
            f"9. YOU MANAGE YOUR OWN EXITS. There is no automatic time stop on your\n"
            f"   trades. Use CLOSE when your thesis is invalidated, or let TP/SL hit.\n"
            f"   Do not hold losing trades hoping they'll recover — cut losers fast.\n"
        )

    def build_context_message(self, ctx: PilotContext) -> str:
        """Build the user message with all current market data."""
        parts = []

        # Time and account
        parts.append(
            f"=== CURRENT STATE ===\n"
            f"Time: {ctx.utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"({ctx.utc_now.strftime('%A')})\n"
            f"NAV: ${ctx.nav:.2f} | Margin Available: ${ctx.margin_available:.2f} | "
            f"Account Floor: ${ctx.account_floor:.2f}\n"
            f"Distance to floor: ${ctx.nav - ctx.account_floor:.2f}"
        )

        # Open positions
        if ctx.open_positions:
            pos_lines = ["=== OPEN POSITIONS ==="]
            for p in ctx.open_positions:
                pos_lines.append(
                    f"  {p.get('trade_id','?')}: {p.get('pair','?')} "
                    f"{p.get('direction','?')} {p.get('units',0)} units | "
                    f"entry={p.get('entry_price',0):.5f} "
                    f"SL={p.get('sl_price',0):.5f} TP={p.get('tp_price',0):.5f} | "
                    f"unrealized={p.get('unrealized_pnl',0):+.2f} pips"
                )
            parts.append("\n".join(pos_lines))
        else:
            parts.append("=== OPEN POSITIONS ===\nNone")

        # Per-pair market data
        parts.append("=== MARKET DATA (all pairs) ===")
        for pair in ctx.instruments:
            pd_data = ctx.pair_data.get(pair, {})
            if not pd_data:
                continue
            ind_m5 = pd_data.get("indicators_m5", {})
            ind_m1 = pd_data.get("indicators_m1", {})
            regime = pd_data.get("regime", "unknown")
            regime_conf = pd_data.get("regime_confidence", 0)

            model_a_sig = pd_data.get("model_a_signal", "")
            model_b_sig = pd_data.get("model_b_signal", "")
            model_c_sig = pd_data.get("model_c_signal", "")
            signal_lines = ""
            if model_a_sig:
                signal_lines += f"Model A (Trend breakout): {model_a_sig}\n"
            if model_b_sig:
                signal_lines += f"Model B (Range reversal): {model_b_sig}\n"
            if model_c_sig:
                signal_lines += f"Model C (EMA Crossover — research signal, no proven edge): {model_c_sig}\n"
            if not signal_lines:
                signal_lines = "Models: no signal evaluated\n"

            parts.append(
                f"\n--- {pair} ---\n"
                f"Bid={pd_data.get('bid',0):.5f} Ask={pd_data.get('ask',0):.5f} "
                f"Spread={pd_data.get('spread_pips',0):.1f} pips\n"
                f"Regime: {regime} (confidence={regime_conf:.2f})\n"
                f"M5: EMA20={ind_m5.get('ema20',0):.5f} EMA50={ind_m5.get('ema50',0):.5f} "
                f"Slope={ind_m5.get('ema_slope',0):.4f} "
                f"RSI={ind_m5.get('rsi14',0):.1f} "
                f"ATR={ind_m5.get('atr14',0):.6f} "
                f"BB_width={ind_m5.get('bb_width',0):.5f}\n"
                f"M1: EMA20={ind_m1.get('ema20',0):.5f} "
                f"RSI={ind_m1.get('rsi14',0):.1f} "
                f"ATR={ind_m1.get('atr14',0):.6f}\n"
                f"{signal_lines}"
                f"Recent M5 closes: {pd_data.get('recent_closes', [])}"
            )

        # News intelligence
        parts.append(f"\n{ctx.news_intel}")

        # Instructions
        parts.append(
            "\n=== YOUR DECISION ===\n"
            "Analyze all the data above. For each pair, decide what to do.\n"
            "You may issue MULTIPLE actions in one response.\n"
            "You may also CLOSE or MODIFY_SL on open positions.\n\n"
            "Format each action as:\n"
            "ACTION:\n"
            "  DECISION: TRADE | SKIP | CLOSE | MODIFY_SL\n"
            "  PAIR: <pair>                    (required for TRADE/SKIP)\n"
            "  DIRECTION: LONG | SHORT         (required for TRADE)\n"
            "  RISK_PCT: <0.005-0.03>          (required for TRADE, fraction of NAV)\n"
            "  SL_PIPS: <number>               (required for TRADE)\n"
            "  TP_PIPS: <number>               (required for TRADE)\n"
            "  CONFIDENCE: <0.0-1.0>           (required for TRADE)\n"
            "  TRADE_ID: <id>                  (required for CLOSE/MODIFY_SL)\n"
            "  NEW_SL_PRICE: <price>           (required for MODIFY_SL)\n"
            "  REASONING: <1-2 sentences>\n\n"
            "If no good setups exist, output a single SKIP with reasoning.\n\n"
            "YOU HAVE THREE SIGNAL MODELS AS TOOLS:\n"
            "  Model A (Trend breakout): fires in trend regimes on compression+breakout+retest\n"
            "  Model B (Range reversal): fires in range regimes on failed breakout+wick rejection\n"
            "  Model C (EMA Crossover): fires on EMA 5/40 crossover with spread filter\n"
            "    -> No proven live edge; out-of-sample evaluation shows it loses money.\n"
            "    -> Treat as a research signal only, never a guaranteed winner.\n\n"
            "These models are ADVISORS, not gatekeepers. Use them as your primary signals\n"
            "but you are ABSOLUTELY FREE to trade your own setups too.\n\n"
            "IMPORTANT: You are an intelligent AI with deep forex knowledge.\n"
            "You do NOT need a model to show VALID to enter a trade.\n"
            "If you see a strong setup using your own analysis of the indicators,\n"
            "price action, regime, and market context — TAKE THE TRADE.\n"
            "Models are helpful when they fire, but they don't fire often.\n"
            "During London and NY overlap, you should be finding 2-5 good setups\n"
            "per hour using your own analysis. If you've been skipping for 30+\n"
            "minutes during active hours, you are being too passive.\n\n"
            "When a model DOES fire (especially Model C on AUD/USD), that is your\n"
            "highest conviction setup. But don't wait around doing nothing when\n"
            "models are quiet. Use your knowledge base strategies:\n"
            "- EMA crossover momentum\n"
            "- Bollinger Band bounce\n"
            "- RSI divergence\n"  
            "- Momentum continuation\n"
            "- Breakout-retest\n"
            "- Failed breakout reversal\n\n"
            "You are the decision maker. Be disciplined and risk-aware — this is "
            "a research exercise in PRACTICE mode, not a profit mandate."
        )

        return "\n".join(parts)

    def evaluate(self, ctx: PilotContext) -> list[PilotAction]:
        """Call Claude and get trading decisions.

        Returns empty list if account is below floor.
        """
        if ctx.nav <= self._account_floor:
            _log.warning(
                f"AI Pilot: NAV ${ctx.nav:.2f} <= floor ${self._account_floor:.2f}. "
                f"No trading.")
            return []

        if self._max_trades > 0 and self._trades_executed >= self._max_trades:
            _log.info(
                f"AI Pilot: trade limit reached ({self._trades_executed}/{self._max_trades}). "
                f"Paused for review. Restart to reset counter.")
            return []

        try:
            import anthropic
            import httpx

            system_prompt = self.build_system_prompt()
            user_message = self.build_context_message(ctx)

            client = anthropic.Anthropic(
                timeout=httpx.Timeout(60.0, connect=10.0))
            response = client.messages.create(
                model=self._model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            raw_text = response.content[0].text.strip()
            self._call_count += 1

            if self._log_reasoning:
                _log.info(f"AI Pilot response (call #{self._call_count}):\n{raw_text}")

            actions = self._parse_response(raw_text)
            trade_count = sum(1 for a in actions if a.decision == "TRADE")
            self._trades_executed += trade_count
            if trade_count > 0 and self._max_trades > 0:
                _log.info(
                    f"AI Pilot: {self._trades_executed}/{self._max_trades} "
                    f"trades used")
            return actions

        except Exception as e:
            _log.error(f"AI Pilot evaluation failed: {e}")
            return []

    def _parse_response(self, text: str) -> list[PilotAction]:
        """Parse Claude's structured response into PilotAction objects."""
        actions = []
        current: dict = {}

        for line in text.split("\n"):
            stripped = line.strip()

            if stripped.startswith("ACTION:") or stripped == "ACTION:":
                if current:
                    action = self._build_action(current)
                    if action:
                        actions.append(action)
                current = {}
                continue

            if ":" in stripped:
                key, _, value = stripped.partition(":")
                key = key.strip().upper()
                value = value.strip()
                if key in ("DECISION", "PAIR", "DIRECTION", "RISK_PCT",
                           "SL_PIPS", "TP_PIPS", "CONFIDENCE", "TRADE_ID",
                           "NEW_SL_PRICE", "REASONING"):
                    current[key] = value

        # Don't forget the last action
        if current:
            action = self._build_action(current)
            if action:
                actions.append(action)

        # If no ACTION: headers were found, try parsing as single action
        if not actions and current:
            action = self._build_action(current)
            if action:
                actions.append(action)

        return actions

    def _build_action(self, fields: dict) -> Optional[PilotAction]:
        """Build a PilotAction from parsed fields."""
        decision = fields.get("DECISION", "").upper()
        if decision not in ("TRADE", "SKIP", "CLOSE", "MODIFY_SL"):
            return None

        action = PilotAction(
            decision=decision,
            pair=fields.get("PAIR", "").upper().replace("/", "_"),
            direction=fields.get("DIRECTION", "").lower(),
            reasoning=fields.get("REASONING", ""),
            trade_id=fields.get("TRADE_ID", ""),
            signal_id=f"pilot-{uuid.uuid4().hex[:12]}",
        )

        try:
            action.risk_pct = float(fields.get("RISK_PCT", 0))
        except (ValueError, TypeError):
            action.risk_pct = 0.01

        try:
            action.sl_pips = float(fields.get("SL_PIPS", 0))
        except (ValueError, TypeError):
            action.sl_pips = 0

        try:
            action.tp_pips = float(fields.get("TP_PIPS", 0))
        except (ValueError, TypeError):
            action.tp_pips = 0

        try:
            action.confidence = float(fields.get("CONFIDENCE", 0))
        except (ValueError, TypeError):
            action.confidence = 0.5

        try:
            action.new_sl_price = float(fields.get("NEW_SL_PRICE", 0))
        except (ValueError, TypeError):
            action.new_sl_price = 0

        # Validation
        if decision == "TRADE":
            if not action.pair or not action.direction:
                _log.warning(f"AI Pilot: TRADE missing pair/direction: {fields}")
                return None
            if action.sl_pips <= 0:
                _log.warning(f"AI Pilot: TRADE with no stop loss rejected: {fields}")
                return None
            if action.direction not in ("long", "short"):
                _log.warning(f"AI Pilot: invalid direction '{action.direction}'")
                return None
            # Cap risk_pct to prevent catastrophic sizing
            action.risk_pct = min(max(action.risk_pct, 0.001), 0.05)

        return action
