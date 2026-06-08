"""Decision Pipeline — 13-step orchestrator per spec 0.4 / A.4.

Executes the full decision sequence on each M1 close:
  Step  0: Data Quality Gate
  Step  1: Session Gate
  Step  2: News Gate
  Step  3: Spread Filter
  Step  4: Regime Engine
  Step  5: Cooldown Check
  Step  6: Model A Trigger
  Step  7: Retest Wait
  Step  8: Risk Manager
  Step  9: Order Builder
  Step 10: Send Order (requires live broker connection)
  Step 11: Trade Manager (runs asynchronously on open trades)
  Step 12: Logger

Any failure at any step → NO_TRADE with reason logged.
Step 12 (Logger) always runs.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..gates.data_quality_gate import DataQualityGate, DataQualityResult
from ..gates.session_gate import SessionResult, is_session_allowed
from ..gates.news_gate import NewsGate, NewsGateResult
from ..gates.spread_filter import SpreadResult, check_spread
from ..logger import ScalpLogger, generate_signal_id
from ..execution.risk_manager import RiskManager, RiskResult, OpenPosition
from ..execution.order_builder import OrderBuilder, OrderSpec
from ..execution.executor import Executor, ExecutionResult
from ..execution.trade_manager import TradeManager, ManagedTrade
from .feature_engine import FeatureEngine, IndicatorSet
from .regime_engine import RegimeEngine, Regime, RegimeResult
from .model_a import ModelATrigger, TriggerSignal, TriggerPhase
from .model_b import ModelBTrigger
from .cooldown import CooldownManager, CooldownResult
from ..ai.borderline_reviewer import AIBorderlineReviewer
from ..ai.regime_classifier import AIRegimeClassifier


@dataclass
class PipelineResult:
    """Full result of one decision cycle."""
    pair: str
    final_decision: str  # "NO_TRADE" or "SIGNAL_SENT"
    no_trade_reason: Optional[str] = None
    pipeline_latency_ms: int = 0

    # Step results
    data_quality: Optional[DataQualityResult] = None
    session: Optional[SessionResult] = None
    news: Optional[NewsGateResult] = None
    spread: Optional[SpreadResult] = None
    regime: Optional[RegimeResult] = None
    cooldown: Optional[CooldownResult] = None
    trigger: Optional[TriggerSignal] = None
    risk: Optional[RiskResult] = None
    order: Optional[OrderSpec] = None
    execution: Optional[ExecutionResult] = None
    indicators_m5: Optional[IndicatorSet] = None
    indicators_m1: Optional[IndicatorSet] = None

    # Borderline
    is_borderline: bool = False
    borderline_flags: Optional[list[str]] = None

    # Pending order info (for PendingOrderManager to track)
    _pending_info: Optional[dict] = None


class DecisionPipeline:
    """Orchestrates the 13-step decision sequence for one pair.

    Usage:
        pipeline = DecisionPipeline(
            config=config, logger=logger,
            feature_engine=feature_engine, regime_engine=regime_engine,
            trigger=model_a_trigger, news_gate=news_gate,
            data_quality_gate=data_quality_gate, cooldown_manager=cooldown_manager,
            risk_manager=risk_manager, order_builder=order_builder,
            executor=executor,  # Optional: None for backtest/paper mode
        )
        result = pipeline.run("EUR_USD", df_m1, df_m5, bid, ask, nav, margin,
                              open_positions, utc_now)
    """

    def __init__(self, config, logger: ScalpLogger,
                 feature_engine: FeatureEngine,
                 regime_engine: RegimeEngine,
                 trigger: ModelATrigger,
                 news_gate: NewsGate,
                 data_quality_gate: DataQualityGate,
                 cooldown_manager: CooldownManager,
                 risk_manager: Optional[RiskManager] = None,
                 order_builder: Optional[OrderBuilder] = None,
                 executor: Optional[Executor] = None,
                 trade_manager: Optional[TradeManager] = None,
                 trigger_b: Optional[ModelBTrigger] = None,
                 borderline_reviewer: Optional[AIBorderlineReviewer] = None,
                 ai_regime: Optional[AIRegimeClassifier] = None):
        self._config = config
        self._logger = logger
        self._feature = feature_engine
        self._regime = regime_engine
        self._trigger = trigger
        self._trigger_b = trigger_b
        self._borderline_reviewer = borderline_reviewer
        self._ai_regime = ai_regime
        self._news_gate = news_gate
        self._dq_gate = data_quality_gate
        self._cooldown = cooldown_manager
        self._risk = risk_manager
        self._order_builder = order_builder
        self._executor = executor
        self._trade_mgr = trade_manager

    def run(self, pair: str, df_m1: pd.DataFrame, df_m5: pd.DataFrame,
            bid: float, ask: float,
            nav: float = 0, margin_available: float = 0,
            open_positions: Optional[list[OpenPosition]] = None,
            utc_now: Optional[datetime] = None) -> PipelineResult:
        """Execute the full decision pipeline for one pair.

        Args:
            pair: Instrument (e.g., "EUR_USD")
            df_m1: M1 candle DataFrame (enough for indicator warmup)
            df_m5: M5 candle DataFrame
            bid: Current bid price (from stream)
            ask: Current ask price (from stream)
            utc_now: Current UTC time (default: now)

        Returns:
            PipelineResult with the decision and all intermediate results.
        """
        start_time = time.monotonic()
        utc_now = utc_now or datetime.now(timezone.utc)
        open_positions = open_positions or []

        result = PipelineResult(pair=pair, final_decision="NO_TRADE")
        all_borderline = []

        # --- Step 0: Data Quality Gate ---
        dq = self._dq_gate.check(utc_now)
        result.data_quality = dq
        if not dq.is_ok:
            result.no_trade_reason = f"data_quality:{dq.issue}"
            self._finalize(result, start_time, utc_now)
            return result

        # --- Step 1: Session Gate ---
        sess_cfg = self._config.sessions
        session = is_session_allowed(
            utc_now,
            mode=sess_cfg.get("mode", "overlap_only"),
            block=sess_cfg.get("block"),
        )
        result.session = session
        if not session.allowed:
            result.no_trade_reason = f"session_blocked:{session.window_name}"
            self._finalize(result, start_time, utc_now)
            return result

        # --- Step 2: News Gate ---
        # Pass current spread for auto-extension check (spec 2.3)
        from ..utils.pip_utils import price_to_pips
        current_spread_pips = price_to_pips(ask - bid, pair)
        news = self._news_gate.check(pair, utc_now,
                                      current_spread_pips=current_spread_pips,
                                      max_spread_pips=self._config.max_spread_pips(pair))
        result.news = news
        if not news.is_safe:
            result.no_trade_reason = f"news_freeze:{news.blocking_event}"
            self._finalize(result, start_time, utc_now)
            return result

        # --- Step 3: Spread Filter ---
        max_spread = self._config.max_spread_pips(pair)
        spread = check_spread(bid, ask, pair, max_spread)
        result.spread = spread
        if not spread.is_ok:
            result.no_trade_reason = "spread_too_wide"
            self._finalize(result, start_time, utc_now)
            return result

        # Check borderline B4 (spread close to limit)
        bl_warn_ratio = self._config.borderline.get("spread_warn_ratio", 0.70)
        if spread.spread_pips > bl_warn_ratio * max_spread:
            all_borderline.append("B4")

        # --- Compute Indicators ---
        ind_m5 = self._feature.compute(df_m5, "M5")
        ind_m1 = self._feature.compute(df_m1, "M1")
        result.indicators_m5 = ind_m5
        result.indicators_m1 = ind_m1

        # Check indicator NaN (feeds back to Data Quality Gate)
        has_nan, nan_field = ind_m5.has_nan()
        if has_nan:
            self._dq_gate.update_indicators(False, f"M5_{nan_field}")
            result.no_trade_reason = f"indicator_nan:M5_{nan_field}"
            self._finalize(result, start_time, utc_now)
            return result

        has_nan, nan_field = ind_m1.has_nan()
        if has_nan:
            self._dq_gate.update_indicators(False, f"M1_{nan_field}")
            result.no_trade_reason = f"indicator_nan:M1_{nan_field}"
            self._finalize(result, start_time, utc_now)
            return result

        self._dq_gate.update_indicators(True)

        # --- Step 4: Regime Engine (rule-based, optionally AI-enhanced) ---
        close_m5 = float(df_m5.iloc[-1]["close"])
        if self._ai_regime and self._ai_regime.enabled:
            regime_result = self._ai_regime.evaluate(ind_m5, close_m5, utc_now)
        else:
            regime_result = self._regime.evaluate(ind_m5, close_m5)
        result.regime = regime_result

        if regime_result.regime == Regime.NO_TRADE:
            result.no_trade_reason = "no_regime"
            self._finalize(result, start_time, utc_now)
            return result

        if regime_result.is_borderline and regime_result.borderline_flags:
            all_borderline.extend(regime_result.borderline_flags)

        # --- Step 5: Cooldown Check ---
        # For Trend: direction is known → check now.
        # For Range: direction depends on Model B → defer to after trigger.
        if regime_result.regime == Regime.TREND_UP:
            direction = "long"
            cooldown_result = self._cooldown.check(pair, direction, utc_now)
            result.cooldown = cooldown_result
            if not cooldown_result.is_ok:
                result.no_trade_reason = f"cooldown:{cooldown_result.reason}"
                self._finalize(result, start_time, utc_now)
                return result
        elif regime_result.regime == Regime.TREND_DOWN:
            direction = "short"
            cooldown_result = self._cooldown.check(pair, direction, utc_now)
            result.cooldown = cooldown_result
            if not cooldown_result.is_ok:
                result.no_trade_reason = f"cooldown:{cooldown_result.reason}"
                self._finalize(result, start_time, utc_now)
                return result

        # --- Step 6 & 7: Model A (Trend) or Model B (Range) ---
        if regime_result.regime in (Regime.TREND_UP, Regime.TREND_DOWN):
            trigger_signal = self._trigger.evaluate(
                df_m1, ind_m1, regime_result.regime, pair)
        elif regime_result.regime == Regime.RANGE and self._trigger_b:
            trigger_signal = self._trigger_b.evaluate(
                df_m1, df_m5, ind_m1, ind_m5, regime_result.regime, pair,
                spread_pips=spread.spread_pips)
        else:
            trigger_signal = TriggerSignal(
                phase=TriggerPhase.NO_COMPRESSION,
                values={"reason": "no_model_for_regime"})
        result.trigger = trigger_signal

        if trigger_signal.phase == TriggerPhase.WAITING_RETEST:
            result.no_trade_reason = "waiting_retest"
            self._finalize(result, start_time, utc_now)
            return result

        if trigger_signal.phase != TriggerPhase.VALID:
            result.no_trade_reason = f"no_trigger:{trigger_signal.phase.value}"
            self._finalize(result, start_time, utc_now)
            return result

        if trigger_signal.borderline_flags:
            all_borderline.extend(trigger_signal.borderline_flags)

        # --- Step 5b: Deferred Cooldown for Range (direction now known) ---
        if regime_result.regime == Regime.RANGE and trigger_signal.direction:
            direction = trigger_signal.direction.value
            cooldown_result = self._cooldown.check(pair, direction, utc_now)
            result.cooldown = cooldown_result
            if not cooldown_result.is_ok:
                result.no_trade_reason = f"cooldown:{cooldown_result.reason}"
                self._finalize(result, start_time, utc_now)
                return result

        # --- Step 7.5: AI Borderline Review (between trigger and risk) ---
        if all_borderline and self._borderline_reviewer and self._borderline_reviewer.enabled:
            direction = trigger_signal.direction.value if trigger_signal.direction else "long"
            summary = {
                "pair": pair,
                "direction": direction,
                "model": "B" if regime_result.regime == Regime.RANGE else "A",
                "regime": regime_result.regime.value,
                "regime_values": regime_result.values,
                "trigger_values": trigger_signal.values,
                "borderline_flags": list(set(all_borderline)),
                "spread_pips": spread.spread_pips,
            }
            decision = self._borderline_reviewer.evaluate(summary)
            if not decision.approved:
                result.no_trade_reason = f"borderline_rejected:{decision.reason}"
                result.is_borderline = True
                result.borderline_flags = list(set(all_borderline))
                self._finalize(result, start_time, utc_now)
                return result

        # --- Step 8: Risk Manager ---
        if self._risk:
            direction = trigger_signal.direction.value if trigger_signal.direction else "long"
            stop_pips = trigger_signal.risk_pips or 0

            risk_result = self._risk.evaluate(
                pair=pair, direction=direction, stop_pips=stop_pips,
                nav=nav, margin_available=margin_available,
                open_positions=open_positions,
                mid_price=(bid + ask) / 2,
            )
            result.risk = risk_result

            if not risk_result.approved:
                result.no_trade_reason = f"risk_limit:{risk_result.reject_reason}"
                self._finalize(result, start_time, utc_now)
                return result
        else:
            # No risk manager (backtest mode) — use a default unit size
            risk_result = RiskResult(approved=True, units=1000)
            result.risk = risk_result

        # --- Step 9: Order Builder ---
        signal_id = generate_signal_id()
        direction = trigger_signal.direction.value if trigger_signal.direction else "long"

        if self._order_builder and trigger_signal.entry_price:
            if self._order_builder.use_market_primary:
                order = self._order_builder.build_market(
                    pair=pair, direction=direction,
                    units=risk_result.units,
                    current_price=(bid + ask) / 2,
                    sl_price=trigger_signal.sl_price,
                    tp_price=trigger_signal.tp_price,
                    signal_id=signal_id,
                )
            else:
                order = self._order_builder.build_limit(
                    pair=pair, direction=direction,
                    units=risk_result.units,
                    entry_price=trigger_signal.entry_price,
                    sl_price=trigger_signal.sl_price,
                    tp_price=trigger_signal.tp_price,
                    signal_id=signal_id, utc_now=utc_now,
                )
            result.order = order

            # --- Step 10: Send Order ---
            if self._executor:
                oanda_body = self._order_builder.to_oanda_order(order)
                exec_result = self._executor.submit(order, oanda_body)
                result.execution = exec_result

                if not exec_result.success:
                    # Immediate failure → try Market fallback (spec A.3 step 2)
                    fallback_order = self._order_builder.build_market_fallback(
                        pair=pair, direction=direction,
                        units=risk_result.units,
                        entry_price=trigger_signal.entry_price,
                        sl_price=trigger_signal.sl_price,
                        tp_price=trigger_signal.tp_price,
                        current_price=(bid + ask) / 2,
                        atr=ind_m1.atr14 or 0,
                        spread_pips=spread.spread_pips,
                        max_spread_pips=max_spread,
                        signal_id=signal_id + "-fallback",
                    )
                    if fallback_order:
                        fb_body = self._order_builder.to_oanda_order(fallback_order)
                        exec_result = self._executor.submit(fallback_order, fb_body)
                        result.execution = exec_result
                        result.order = fallback_order

                    if not exec_result.success:
                        reason = exec_result.reject_reason or exec_result.broker_status
                        result.no_trade_reason = f"broker_reject:{reason}"
                        self._finalize(result, start_time, utc_now)
                        return result

                # Pending Limit order → tracked by PendingOrderManager in main.py
                if exec_result.broker_status == "pending":
                    # Store pending info on result for caller to track
                    result._pending_info = {
                        "order_id": exec_result.order_id,
                        "signal_id": signal_id,
                        "pair": pair,
                        "direction": direction,
                        "units": risk_result.units,
                        "entry_price": trigger_signal.entry_price,
                        "sl_price": trigger_signal.sl_price,
                        "tp_price": trigger_signal.tp_price,
                        "atr": ind_m1.atr14 or 0,
                        "spread_at_signal": spread.spread_pips,
                        "max_spread": max_spread,
                    }

                # Log trade on successful execution (filled immediately)
                self._logger.log_trade({
                    "trade_id": exec_result.trade_id or signal_id,
                    "decision_log_ref": signal_id,
                    "pair": pair,
                    "direction": direction,
                    "order_type": result.order.order_type.value,
                    "expected_entry_price": trigger_signal.entry_price,
                    "price_bound": result.order.price_bound,
                    "fill_price": exec_result.fill_price,
                    "actual_slippage_pips": exec_result.actual_slippage_pips,
                    "spread_at_signal": spread.spread_pips,
                    "spread_at_fill": None,
                    "order_sent_ts": utc_now.isoformat(),
                    "fill_received_ts": exec_result.fill_time,
                    "e2e_latency_ms": exec_result.e2e_latency_ms,
                    "broker_status": exec_result.broker_status,
                    "reject_reason": exec_result.reject_reason,
                    "sl_price": trigger_signal.sl_price,
                    "tp_price": trigger_signal.tp_price,
                    "units": risk_result.units,
                    "signal_id": signal_id,
                    "is_borderline": bool(all_borderline),
                    "borderline_flags": list(set(all_borderline)) if all_borderline else None,
                })

                # --- Step 11: Register with Trade Manager ---
                if (self._trade_mgr
                        and exec_result.broker_status == "filled"
                        and exec_result.fill_price):
                    managed = ManagedTrade(
                        trade_id=exec_result.trade_id or signal_id,
                        pair=pair,
                        direction=direction,
                        entry_price=exec_result.fill_price,
                        sl_price=trigger_signal.sl_price,
                        tp_price=trigger_signal.tp_price,
                        units=risk_result.units,
                        open_time=utc_now,
                        risk_amount=abs(exec_result.fill_price - trigger_signal.sl_price),
                    )
                    self._trade_mgr.add_trade(managed)

        result.final_decision = "SIGNAL_SENT"
        result.no_trade_reason = None

        # --- Borderline summary ---
        if all_borderline:
            result.is_borderline = True
            result.borderline_flags = list(set(all_borderline))

        self._finalize(result, start_time, utc_now)
        return result

    def _finalize(self, result: PipelineResult, start_time: float,
                  utc_now: datetime) -> None:
        """Log the decision and compute latency."""
        result.pipeline_latency_ms = int((time.monotonic() - start_time) * 1000)

        # Build log record (spec 0.6 Table 5 — all fields)
        trigger = result.trigger
        log_record = {
            "timestamp_utc": utc_now.isoformat(),
            "pair": result.pair,
            "signal_id": result.order.signal_id if result.order else None,
            "data_quality_ok": result.data_quality.is_ok if result.data_quality else None,
            "data_quality_issue": result.data_quality.issue if result.data_quality else None,
            "session_allowed": result.session.allowed if result.session else None,
            "news_safe": result.news.is_safe if result.news else None,
            "next_event_min": result.news.next_event_minutes if result.news else None,
            "spread_pips_at_signal": result.spread.spread_pips if result.spread else None,
            "spread_ok": result.spread.is_ok if result.spread else None,
            "regime": result.regime.regime.value if result.regime else None,
            "regime_values": result.regime.values if result.regime else None,
            "trigger_result": trigger.phase.value if trigger else None,
            "trigger_values": trigger.values if trigger else None,
            "direction": trigger.direction.value if trigger and trigger.direction else None,
            "entry_price": trigger.entry_price if trigger else None,
            "sl_price": trigger.sl_price if trigger else None,
            "tp_price": trigger.tp_price if trigger else None,
            "is_borderline": result.is_borderline,
            "borderline_flags": result.borderline_flags,
            "cooldown_ok": result.cooldown.is_ok if result.cooldown else None,
            "risk_approved": result.risk.approved if result.risk else None,
            "units": result.risk.units if result.risk else None,
            "order_type": result.order.order_type.value if result.order else None,
            "final_decision": result.final_decision,
            "no_trade_reason": result.no_trade_reason,
            "pipeline_latency_ms": result.pipeline_latency_ms,
        }

        self._logger.log_decision(log_record)
