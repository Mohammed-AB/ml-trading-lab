"""Pilot Pipeline — Autonomous AI trading pipeline.

Replaces the rule-based DecisionPipeline when ai.pilot.enabled is true.
Only two hard limits: news blackouts and account floor.
The original DecisionPipeline is never modified.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..ai.pilot import AIPilot, PilotAction, PilotContext
from ..ai.pilot_journal import PilotJournal, PilotTradeRecord
from ..ai.pilot_news_intel import NewsIntelligence
from ..engine.feature_engine import FeatureEngine, IndicatorSet
from ..engine.regime_engine import RegimeEngine
from ..engine.model_a import ModelATrigger
from ..engine.model_b import ModelBTrigger
from ..engine.model_c import ModelCTrigger
from ..execution.order_builder import OrderBuilder
from ..execution.executor import Executor
from ..execution.trade_manager import TradeManager
from ..gates.news_gate import NewsGate
from ..utils.pip_utils import pips_to_price, pip_value_in_account_ccy

_log = logging.getLogger("scalp_mode")


@dataclass
class PilotPipelineResult:
    actions_taken: int = 0
    actions_skipped: int = 0
    trades_opened: int = 0
    trades_closed: int = 0
    sl_modified: int = 0
    errors: list = None
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class PilotPipeline:
    """Autonomous AI trading pipeline.

    Flow:
    1. Check account floor (hard limit)
    2. Gather data for all pairs (indicators, prices, regime)
    3. Filter pairs blocked by news (hard limit)
    4. Build full context for AI
    5. Call AIPilot.evaluate() -> list of actions
    6. Execute each action (TRADE, CLOSE, MODIFY_SL)
    7. Log everything
    """

    def __init__(self, pilot: AIPilot, journal: PilotJournal,
                 news_intel: NewsIntelligence,
                 feature_engine: FeatureEngine,
                 regime_engine: RegimeEngine,
                 order_builder: OrderBuilder,
                 executor: Executor,
                 trade_manager: TradeManager,
                 news_gate: NewsGate,
                 config: dict,
                 logger=None,
                 model_a: ModelATrigger = None,
                 model_b: ModelBTrigger = None,
                 model_c: ModelCTrigger = None):
        self._pilot = pilot
        self._journal = journal
        self._news_intel = news_intel
        self._feature = feature_engine
        self._regime = regime_engine
        self._order_builder = order_builder
        self._executor = executor
        self._trade_mgr = trade_manager
        self._news_gate = news_gate
        self._config = config
        self._logger = logger
        self._model_a = model_a
        self._model_b = model_b
        self._model_c = model_c
        self._account_ccy = config.get("risk", {}).get("account_currency", "USD")
        self._live_rates: dict = {}
        self._log_dir = Path(config.get("logging", {}).get("log_dir", "logs"))
        self._shadow_log = self._log_dir / "pilot_shadow.jsonl"

    def update_rates(self, rates: dict) -> None:
        self._live_rates = rates

    def run(self, instruments: list[str],
            candle_data: dict,  # {pair: {"m1": df, "m5": df}}
            live_prices: dict,  # {pair: (bid, ask)}
            nav: float, margin_available: float,
            utc_now: datetime) -> PilotPipelineResult:
        """Run one cycle of the AI Pilot pipeline."""
        start = time.monotonic()
        result = PilotPipelineResult()

        # Hard limit 1: Account floor
        if nav <= self._pilot.account_floor:
            _log.warning(
                f"Pilot: NAV ${nav:.2f} <= floor ${self._pilot.account_floor:.2f}. "
                f"All trading halted.")
            result.latency_ms = (time.monotonic() - start) * 1000
            return result

        # Gather per-pair data
        pair_data = {}
        available_pairs = []
        for pair in instruments:
            cd = candle_data.get(pair)
            lp = live_prices.get(pair)
            if not cd or not lp:
                continue
            bid, ask = lp
            spread_pips = self._compute_spread_pips(pair, bid, ask)

            # Hard limit 2: News blackout
            news_result = self._news_gate.check(
                pair, utc_now, spread_pips,
                self._config.get("costs", {}).get("max_spread_pips", {}).get(pair, 2.0))
            news_blocked = not news_result.is_safe

            # Compute indicators
            ind_m5 = self._safe_indicators(cd.get("m5"), "M5")
            ind_m1 = self._safe_indicators(cd.get("m1"), "M1")

            # Compute regime
            regime_result = None
            if ind_m5:
                close_m5 = float(cd["m5"]["close"].iloc[-1]) if len(cd["m5"]) > 0 else 0
                regime_result = self._regime.evaluate(ind_m5, close_m5)

            recent_closes = []
            if cd.get("m5") is not None and len(cd["m5"]) >= 5:
                recent_closes = [round(float(c), 5)
                                 for c in cd["m5"]["close"].iloc[-5:].tolist()]

            # Run Model A and Model B to get their signals
            model_a_signal = ""
            model_b_signal = ""
            if regime_result and ind_m1:
                if self._model_a:
                    try:
                        sig_a = self._model_a.evaluate(
                            cd["m1"], ind_m1, regime_result.regime, pair)
                        model_a_signal = (
                            f"phase={sig_a.phase.value}, "
                            f"direction={sig_a.direction.value if sig_a.direction else 'none'}"
                        )
                        if sig_a.entry_price:
                            model_a_signal += (
                                f", entry={sig_a.entry_price:.5f}, "
                                f"SL={sig_a.sl_price:.5f}, TP={sig_a.tp_price:.5f}, "
                                f"risk_pips={sig_a.risk_pips:.1f}"
                            )
                        if sig_a.values:
                            reason = sig_a.values.get("reason", "")
                            if reason:
                                model_a_signal += f", reason={reason}"
                    except Exception as e:
                        model_a_signal = f"error: {e}"

                if self._model_b and regime_result.regime.value.lower().startswith("range"):
                    try:
                        sig_b = self._model_b.evaluate(
                            cd["m1"], cd["m5"], ind_m1, ind_m5,
                            regime_result.regime, pair, spread_pips)
                        model_b_signal = (
                            f"phase={sig_b.phase.value}, "
                            f"direction={sig_b.direction.value if sig_b.direction else 'none'}"
                        )
                        if sig_b.entry_price:
                            model_b_signal += (
                                f", entry={sig_b.entry_price:.5f}, "
                                f"SL={sig_b.sl_price:.5f}, TP={sig_b.tp_price:.5f}, "
                                f"risk_pips={sig_b.risk_pips:.1f}"
                            )
                        if sig_b.values:
                            reason = sig_b.values.get("reason", "")
                            if reason:
                                model_b_signal += f", reason={reason}"
                    except Exception as e:
                        model_b_signal = f"error: {e}"

            # Run Model C (EMA Crossover)
            model_c_signal = ""
            if self._model_c and ind_m1:
                try:
                    sig_c = self._model_c.evaluate(cd["m1"], ind_m1, pair, spread_pips)
                    model_c_signal = (
                        f"phase={sig_c.phase.value}, "
                        f"direction={sig_c.direction.value if sig_c.direction else 'none'}"
                    )
                    if sig_c.entry_price:
                        model_c_signal += (
                            f", entry={sig_c.entry_price:.5f}, "
                            f"SL={sig_c.sl_price:.5f}, TP={sig_c.tp_price:.5f}, "
                            f"risk_pips={sig_c.risk_pips:.1f}"
                        )
                    if sig_c.values:
                        for k in ("backtest_win_rate", "backtest_sharpe", "reason"):
                            v = sig_c.values.get(k)
                            if v:
                                model_c_signal += f", {k}={v}"
                except Exception as e:
                    model_c_signal = f"error: {e}"

            pd_entry = {
                "bid": bid,
                "ask": ask,
                "spread_pips": spread_pips,
                "news_blocked": news_blocked,
                "news_event": news_result.blocking_event if news_blocked else None,
                "indicators_m5": self._ind_to_dict(ind_m5) if ind_m5 else {},
                "indicators_m1": self._ind_to_dict(ind_m1) if ind_m1 else {},
                "regime": regime_result.regime.value if regime_result else "unknown",
                "regime_confidence": (
                    regime_result.values.get("ai_confidence", 0.5)
                    if regime_result else 0),
                "recent_closes": recent_closes,
                "model_a_signal": model_a_signal,
                "model_b_signal": model_b_signal,
                "model_c_signal": model_c_signal,
            }
            pair_data[pair] = pd_entry

            if not news_blocked:
                available_pairs.append(pair)

        # Build open positions info
        open_positions = []
        for t in self._trade_mgr.open_trades:
            mid = live_prices.get(t.pair, (0, 0))
            mid_price = (mid[0] + mid[1]) / 2 if mid[0] > 0 else 0
            if t.direction == "long":
                unrealized = self._price_to_pips(mid_price - t.entry_price, t.pair)
            else:
                unrealized = self._price_to_pips(t.entry_price - mid_price, t.pair)

            open_positions.append({
                "trade_id": t.trade_id,
                "pair": t.pair,
                "direction": t.direction,
                "units": t.units,
                "entry_price": t.entry_price,
                "sl_price": t.sl_price,
                "tp_price": t.tp_price,
                "unrealized_pnl": round(unrealized, 2),
            })

        # Build context
        ctx = PilotContext(
            utc_now=utc_now,
            nav=nav,
            margin_available=margin_available,
            account_floor=self._pilot.account_floor,
            instruments=available_pairs,
            pair_data=pair_data,
            open_positions=open_positions,
            session_stats=self._journal.get_session_stats(),
            news_intel=self._news_intel.get_relevant_briefings(utc_now),
        )

        # Call AI
        actions = self._pilot.evaluate(ctx)

        # Execute actions
        for action in actions:
            try:
                if action.decision == "TRADE":
                    if action.pair not in available_pairs:
                        _log.warning(
                            f"Pilot: AI requested trade on news-blocked pair "
                            f"{action.pair}, skipping")
                        result.actions_skipped += 1
                        continue
                    success = self._execute_trade(action, nav, margin_available,
                                                   live_prices, utc_now)
                    if success:
                        result.trades_opened += 1
                    result.actions_taken += 1

                elif action.decision == "CLOSE":
                    self._execute_close(action, utc_now, live_prices)
                    result.trades_closed += 1
                    result.actions_taken += 1

                elif action.decision == "MODIFY_SL":
                    self._execute_modify_sl(action)
                    result.sl_modified += 1
                    result.actions_taken += 1

                elif action.decision == "SKIP":
                    result.actions_skipped += 1
                    self._journal.record_trade(PilotTradeRecord(
                        timestamp_utc=utc_now.isoformat(),
                        pair=action.pair or "ALL",
                        direction="",
                        action="SKIP",
                        reasoning=action.reasoning,
                    ))

            except Exception as e:
                _log.error(f"Pilot action execution error: {e}")
                result.errors.append(str(e))

        result.latency_ms = (time.monotonic() - start) * 1000
        _log.info(
            f"Pilot cycle: {result.actions_taken} actions, "
            f"{result.trades_opened} trades opened, "
            f"{result.trades_closed} closed, "
            f"{result.sl_modified} SL modified, "
            f"{result.actions_skipped} skipped | "
            f"{result.latency_ms:.0f}ms")

        return result

    # --- Action executors ---

    def _execute_trade(self, action: PilotAction, nav: float,
                       margin_available: float,
                       live_prices: dict, utc_now: datetime) -> bool:
        """Execute a TRADE action from the AI."""
        pair = action.pair
        lp = live_prices.get(pair)
        if not lp:
            _log.warning(f"Pilot: no live price for {pair}")
            return False
        bid, ask = lp
        mid = (bid + ask) / 2

        # Enforce minimum SL/TP distance (OANDA requires > spread)
        spread_pips = self._compute_spread_pips(pair, bid, ask)
        min_distance = max(spread_pips * 1.5, 3.0)
        sl_pips = max(action.sl_pips, min_distance)
        tp_pips = max(action.tp_pips, min_distance)

        # Position sizing from AI's risk_pct
        risk_amount = nav * action.risk_pct
        pip_val_acct = pip_value_in_account_ccy(
            pair, self._account_ccy, self._live_rates)
        if sl_pips <= 0 or pip_val_acct <= 0:
            _log.warning(f"Pilot: invalid SL or pip value for {pair}")
            return False

        units = int(risk_amount / (sl_pips * pip_val_acct))
        if units <= 0:
            _log.warning(f"Pilot: computed 0 units for {pair}")
            return False

        sl_distance = pips_to_price(sl_pips, pair)
        tp_distance = pips_to_price(tp_pips, pair)

        if action.direction == "long":
            sl_price = bid - sl_distance
            tp_price = ask + tp_distance
        else:
            sl_price = ask + sl_distance
            tp_price = bid - tp_distance

        # Build and submit order
        order = self._order_builder.build_market(
            pair=pair,
            direction=action.direction,
            units=units,
            current_price=mid,
            sl_price=sl_price,
            tp_price=tp_price,
            signal_id=action.signal_id,
        )
        oanda_body = self._order_builder.to_oanda_order(order)
        exec_result = self._executor.submit(order, oanda_body)

        if exec_result.success:
            trade_id = exec_result.trade_id or exec_result.order_id or ""
            fill_price = exec_result.fill_price or mid
            status_msg = exec_result.broker_status
            _log.info(
                f"Pilot TRADE {status_msg}: {pair} {action.direction} "
                f"{units} units, SL={sl_price:.5f} TP={tp_price:.5f} "
                f"(conf={action.confidence:.2f})")

            from ..execution.trade_manager import ManagedTrade
            managed = ManagedTrade(
                trade_id=trade_id,
                pair=pair,
                direction=action.direction,
                entry_price=float(fill_price),
                sl_price=sl_price,
                tp_price=tp_price,
                units=units,
                open_time=utc_now,
                risk_amount=risk_amount,
                model="AI_PILOT",
            )
            self._trade_mgr.add_trade(managed)

            self._journal.record_trade(PilotTradeRecord(
                timestamp_utc=utc_now.isoformat(),
                pair=pair,
                direction=action.direction,
                action="TRADE",
                risk_pct=action.risk_pct,
                sl_pips=action.sl_pips,
                tp_pips=action.tp_pips,
                units=units,
                reasoning=action.reasoning,
                confidence=action.confidence,
                trade_id=trade_id,
            ))

            if self._logger:
                self._logger.log_trade({
                    "trade_id": trade_id,
                    "pair": pair,
                    "direction": action.direction,
                    "order_type": "MARKET",
                    "units": units,
                    "entry_price": fill_price,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "broker_status": "filled",
                    "model": "AI_PILOT",
                    "confidence": action.confidence,
                    "reasoning": action.reasoning,
                })
            return True

        _log.warning(
            f"Pilot TRADE not filled for {pair}: "
            f"status={exec_result.broker_status}, "
            f"reason={exec_result.reject_reason or 'unknown'}, "
            f"units={units}, SL={sl_price:.5f}, TP={tp_price:.5f}, "
            f"sl_pips={sl_pips:.1f}, tp_pips={tp_pips:.1f}, spread={spread_pips:.1f}")
        return False

    def _execute_close(self, action: PilotAction, utc_now: datetime,
                       live_prices: dict) -> None:
        """Execute a CLOSE action on an open trade."""
        trade_id = action.trade_id
        if not trade_id:
            _log.warning("Pilot CLOSE: no trade_id specified")
            return

        trade = self._trade_mgr._trades.get(trade_id)
        if not trade:
            _log.warning(f"Pilot CLOSE: trade {trade_id} not found")
            return

        from ..execution.trade_manager import ExitReason
        self._trade_mgr.execute_close(trade_id, ExitReason.AI_PILOT_CLOSE)
        _log.info(f"Pilot CLOSE: {trade_id} ({action.reasoning})")

        # Record in journal
        mid = live_prices.get(trade.pair, (0, 0))
        exit_price = (mid[0] + mid[1]) / 2 if mid[0] > 0 else 0
        if trade.direction == "long":
            pnl_pips = self._price_to_pips(exit_price - trade.entry_price, trade.pair)
        else:
            pnl_pips = self._price_to_pips(trade.entry_price - exit_price, trade.pair)

        self._journal.record_trade(PilotTradeRecord(
            timestamp_utc=utc_now.isoformat(),
            pair=trade.pair,
            direction=trade.direction,
            action="CLOSE",
            reasoning=action.reasoning,
            trade_id=trade_id,
            pnl_pips=round(pnl_pips, 2),
            exit_reason="ai_pilot_close",
        ))

    def _execute_modify_sl(self, action: PilotAction) -> None:
        """Execute a MODIFY_SL action on an open trade."""
        trade_id = action.trade_id
        new_sl = action.new_sl_price
        if not trade_id or new_sl <= 0:
            _log.warning(f"Pilot MODIFY_SL: missing trade_id or price")
            return

        self._trade_mgr.execute_sl_move(trade_id, new_sl)
        _log.info(
            f"Pilot MODIFY_SL: {trade_id} -> {new_sl:.5f} "
            f"({action.reasoning})")

    # --- Shadow mode ---

    def log_shadow(self, actions: list[PilotAction],
                   rule_based_result: dict, utc_now: datetime) -> None:
        """Log AI decisions alongside rule-based decisions for comparison."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp_utc": utc_now.isoformat(),
                "pilot_actions": [
                    {"decision": a.decision, "pair": a.pair,
                     "direction": a.direction, "confidence": a.confidence,
                     "reasoning": a.reasoning}
                    for a in actions
                ],
                "rule_based": rule_based_result,
            }
            with open(self._shadow_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except IOError as e:
            _log.error(f"Shadow log write failed: {e}")

    # --- Helpers ---

    def _safe_indicators(self, df, timeframe: str) -> Optional[IndicatorSet]:
        if df is None or len(df) < 20:
            return None
        try:
            return self._feature.compute(df, timeframe)
        except Exception:
            return None

    def _ind_to_dict(self, ind: IndicatorSet) -> dict:
        return {
            "ema20": ind.ema20,
            "ema50": ind.ema50,
            "ema_slope": ind.ema_slope,
            "rsi14": ind.rsi14,
            "atr14": ind.atr14,
            "bb_upper": ind.bb_upper,
            "bb_lower": ind.bb_lower,
            "bb_width": ind.bb_width,
            "macd_line": getattr(ind, "macd_line", 0),
            "macd_signal": getattr(ind, "macd_signal", 0),
        }

    def _compute_spread_pips(self, pair: str, bid: float, ask: float) -> float:
        if "JPY" in pair.upper():
            return (ask - bid) * 100
        return (ask - bid) * 10000

    def _price_to_pips(self, price_diff: float, pair: str) -> float:
        if "JPY" in pair.upper():
            return price_diff * 100
        return price_diff * 10000
