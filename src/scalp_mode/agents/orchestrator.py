"""Orchestrator — Coordinates all 5 agents on their schedules.

Main loop:
    every 1 min: Portfolio Agent (if open positions) + Strategy Agent -> Risk Agent
    every 15 min: Research Agent
    every 30 min: Learning Agent
    end of day: Learning Agent daily review
"""

import json
import logging
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from ..ml.ml_gate import MLGate


from .brain import Brain
from .research import ResearchAgent
from .strategy import StrategyAgent, TradeProposal
from .risk import RiskAgent, RiskVerdict
from .portfolio import PortfolioAgent, PortfolioAction
from .learning import LearningAgent
from .chief import ChiefAgent
from .postmortem import PostMortemAnalyst

_log = logging.getLogger("scalp_mode")


class Orchestrator:
    """Coordinates all agents and manages the execution pipeline.

    Flow each minute:
    1. Portfolio Agent evaluates open positions
    2. Strategy Agent proposes 0-1 trade
    3. Risk Agent approves or rejects
    4. If approved: execute via Order Builder + Executor
    5. Periodically: Research Agent, Learning Agent
    """

    def __init__(self, brain: Brain,
                 research: ResearchAgent,
                 strategy: StrategyAgent,
                 risk: RiskAgent,
                 portfolio: PortfolioAgent,
                 learning: LearningAgent,
                 order_builder, executor, trade_manager,
                 news_gate, feature_engine, regime_engine,
                 model_a=None, model_b=None, model_c=None, model_d=None,
                 model_e=None, model_f=None, model_g=None, model_h=None,
                 price_feeder=None,
                 ml_gate: Optional["MLGate"] = None,
                 config: dict = None, logger=None,
                 max_trades: int = 20):
        self._brain = brain
        self._research = research
        self._strategy = strategy
        self._risk = risk
        self._portfolio = portfolio
        self._learning = learning
        # Chief + Post-mortem share the Learning model. Can be separately
        # configured later.
        _meta_model = getattr(learning, "_model", "claude-opus-4-20250514")
        self._chief = ChiefAgent(brain, _meta_model)
        self._postmortem = PostMortemAnalyst(brain, _meta_model)
        self._order_builder = order_builder
        self._executor = executor
        self._trade_mgr = trade_manager
        self._news_gate = news_gate
        self._feature = feature_engine
        self._regime = regime_engine
        self._model_a = model_a
        self._model_b = model_b
        self._model_c = model_c
        self._model_d = model_d
        self._model_e = model_e
        self._model_f = model_f
        self._model_g = model_g
        self._model_h = model_h
        # Multi-timeframe context (H1/D1 bias + vol percentile)
        self._mtf = None
        if price_feeder is not None:
            try:
                from ..engine.mtf_context import MTFContext
                self._mtf = MTFContext(price_feeder, cache_ttl_sec=900)
            except Exception:
                self._mtf = None
        self._config = config or {}
        self._logger = logger
        self._max_trades = max_trades
        self._account_ccy = config.get("risk", {}).get("account_currency", "USD") if config else "USD"
        self._live_rates: dict = {}
        self._last_daily_review: Optional[str] = None

        # Pending orders for Risk dedup (set from main after PendingOrderManager exists)
        self._pending_orders_fn: Optional[Callable[[], list[dict]]] = None
        # cluster_id forensics: (PAIR|dir) -> (cluster_uuid, last_open_time)
        self._cluster_memory: dict[str, tuple[str, datetime]] = {}

        # Session / peak NAV (capital protection)
        self._session_start_nav: Optional[float] = None
        self._peak_nav: float = 0.0
        self._load_peak_nav_file()

        # Persistent daily trade counter (survives restarts)
        self._trades_executed = self._load_trade_count_for_date(
            datetime.now(timezone.utc))

        # API cost optimization
        self._flat_cycle_streak = 0  # no model entry= across pairs
        self._strategy_skip_streak = 0  # Claude returned SKIP while models had signals

        # Portfolio batching (skip redundant Claude calls)
        self._last_portfolio_all_hold = False
        self._last_portfolio_progress: dict[str, float] = {}

        # LightGBM ML gate (optional): when set, Strategy Agent runs only on ML hits.
        self._ml_gate: Optional["MLGate"] = ml_gate

    def _peak_nav_path(self) -> Path:
        return Path("data/peak_nav.json")

    def _trade_count_path(self, utc_now: datetime) -> Path:
        return Path(f"data/trade_count_{utc_now.strftime('%Y-%m-%d')}.json")

    def _load_peak_nav_file(self) -> None:
        p = self._peak_nav_path()
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                v = float(d.get("peak_nav", 0) or 0)
                if v > 0:
                    self._peak_nav = v
            except (json.JSONDecodeError, IOError, ValueError):
                pass

    def _save_peak_nav_file(self) -> None:
        if self._peak_nav <= 0:
            return
        p = self._peak_nav_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"peak_nav": self._peak_nav}), encoding="utf-8")
        except IOError as e:
            _log.warning(f"Orchestrator: could not save peak NAV: {e}")

    def _load_trade_count_for_date(self, utc_now: datetime) -> int:
        p = self._trade_count_path(utc_now)
        if not p.exists():
            return 0
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return int(d.get("count", 0))
        except (json.JSONDecodeError, IOError, ValueError, TypeError):
            return 0

    def _save_trade_count_file(self, utc_now: datetime) -> None:
        p = self._trade_count_path(utc_now)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps({
                    "count": self._trades_executed,
                    "date": utc_now.strftime("%Y-%m-%d"),
                }),
                encoding="utf-8",
            )
        except IOError as e:
            _log.warning(f"Orchestrator: could not save trade count: {e}")

    def _update_nav_tracking(self, nav: float, utc_now: datetime) -> None:
        if nav <= 0:
            return
        if self._session_start_nav is None:
            self._session_start_nav = nav
        if nav > self._peak_nav:
            self._peak_nav = nav
            self._save_peak_nav_file()
        elif self._peak_nav <= 0:
            self._peak_nav = nav
            self._save_peak_nav_file()

    @staticmethod
    def _has_model_entry(pair_data: dict) -> bool:
        for row in pair_data.values():
            for key in ("model_a_signal", "model_b_signal", "model_c_signal",
                        "model_d_signal", "model_e_signal", "model_f_signal",
                        "model_g_signal", "model_h_signal"):
                s = row.get(key) or ""
                if "entry=" in s:
                    return True
        return False

    @staticmethod
    def _progress_towards_tp_frac(trade: dict, mid: float) -> float:
        """0..1 progress from entry toward TP (signed by direction)."""
        entry = float(trade.get("entry_price") or 0)
        tp = float(trade.get("tp_price") or 0)
        if entry == 0 or tp == 0 or mid == 0:
            return 0.0
        if trade.get("direction") == "long":
            denom = tp - entry
            if abs(denom) < 1e-9:
                return 0.0
            return (mid - entry) / denom
        denom = entry - tp
        if abs(denom) < 1e-9:
            return 0.0
        return (entry - mid) / denom

    def _portfolio_eligible(self, open_dicts: list[dict], utc_now: datetime) -> list[dict]:
        out = []
        for t in open_dicts:
            ot = t.get("open_time")
            if isinstance(ot, datetime):
                mins = (utc_now - ot).total_seconds() / 60.0
            else:
                mins = 999.0
            if mins >= 5.0:
                out.append(t)
        return out

    def _should_skip_portfolio_claude(
        self, eligible: list[dict], live_prices: dict,
    ) -> bool:
        if not eligible:
            return True
        if not self._last_portfolio_all_hold:
            return False
        for t in eligible:
            tid = str(t.get("trade_id", ""))
            pair = t.get("pair", "")
            lp = live_prices.get(pair)
            if not lp or not isinstance(lp, tuple):
                return False
            mid = (lp[0] + lp[1]) / 2
            new_p = self._progress_towards_tp_frac(t, mid)
            old_p = self._last_portfolio_progress.get(tid)
            if old_p is None:
                return False
            if abs(new_p - old_p) >= 0.20:
                return False
        return True

    def update_rates(self, rates: dict) -> None:
        self._live_rates = rates

    def set_pending_orders_provider(
            self, fn: Optional[Callable[[], list[dict]]]) -> None:
        """Wire PendingOrderManager.snapshot_for_risk() from main."""
        self._pending_orders_fn = fn

    @staticmethod
    def _model_id_from_source(model_source: str) -> str:
        m = (model_source or "").lower()
        if "model_a" in m:
            return "A"
        if "model_b" in m:
            return "B"
        if "model_c" in m:
            return "C"
        if "model_d" in m:
            return "D"
        if "model_e" in m:
            return "E"
        if "model_f" in m or "tky_ldn" in m:
            return "F"
        if "model_g" in m or "ha_adx" in m:
            return "G"
        if "model_h" in m or "nr7" in m:
            return "H"
        return "M"

    def _cluster_id_for_new_trade(
            self, pair: str, direction: str, utc_now: datetime) -> str:
        """Reuse cluster_id if same pair+direction opened within 15 minutes."""
        p = pair.strip().upper()
        d = direction.strip().lower()
        key = f"{p}|{d}"
        prev = self._cluster_memory.get(key)
        if prev:
            cid, last_ts = prev
            delta = (utc_now - last_ts).total_seconds()
            if delta <= 15 * 60:
                self._cluster_memory[key] = (cid, utc_now)
                return cid
        cid = str(uuid.uuid4())
        self._cluster_memory[key] = (cid, utc_now)
        return cid

    # Trading allowed 07:00–22:00 UTC (London through Late NY).
    # Asian session (22:00–07:00 UTC) is observation-only.
    # Friday ends at 21:00 UTC to match NY forex market close (5 PM ET during DST).
    _TRADE_START_HOUR = 0   # 24/7 — ML gate handles session filtering via features
    _TRADE_END_HOUR = 24    # 24/7 — ML gate handles session filtering via features
    _FRIDAY_END_HOUR = 21   # NY market close on Friday (5 PM ET)

    @staticmethod
    def _is_trading_session(utc_now: datetime) -> bool:
        h = utc_now.hour
        # Saturday/Sunday — always closed
        if utc_now.weekday() >= 5:
            return False
        # Friday — stop an hour earlier to align with NY market close
        if utc_now.weekday() == 4:
            return Orchestrator._TRADE_START_HOUR <= h < Orchestrator._FRIDAY_END_HOUR
        return Orchestrator._TRADE_START_HOUR <= h < Orchestrator._TRADE_END_HOUR

    def run_cycle(self, instruments: list[str],
                  candle_data: dict,
                  live_prices: dict,
                  nav: float, margin_available: float,
                  utc_now: datetime,
                  events: list = None) -> dict:
        """Run one orchestration cycle. Called every minute from main loop."""
        start = time.monotonic()
        result = {
            "trades_opened": 0, "trades_closed": 0,
            "sl_modified": 0, "proposals": 0, "skips": 0,
        }

        # Persisted daily trade counter (new UTC day resets via new file)
        self._trades_executed = self._load_trade_count_for_date(utc_now)

        trading_allowed = self._is_trading_session(utc_now)
        trade_limit_hit = self._trades_executed >= self._max_trades

        self._update_nav_tracking(nav, utc_now)

        # === AI Agents (Research/Learning/Chief) — disabled when ML gate active ===
        # These burn Anthropic credits but don't influence ML trade decisions.
        if not self._ml_gate:
            research_interval = 900 if trading_allowed else 3600
            if self._research.should_run(utc_now, min_interval_sec=research_interval):
                try:
                    price_snapshot = {}
                    for pair in instruments:
                        lp = live_prices.get(pair)
                        if lp:
                            price_snapshot[pair] = (lp[0] + lp[1]) / 2
                    self._research.run(events or [], price_snapshot, utc_now)
                except Exception as e:
                    _log.error(f"Research Agent error: {e}")

            if trading_allowed and self._learning.should_run(utc_now):
                try:
                    self._learning.run(utc_now)
                except Exception as e:
                    _log.error(f"Learning Agent error: {e}")

            if trading_allowed and self._chief.should_run(utc_now):
                try:
                    self._chief.run(utc_now)
                except Exception as e:
                    _log.error(f"Chief Agent error: {e}")

            today = utc_now.strftime("%Y-%m-%d")
            if utc_now.hour >= 21 and self._last_daily_review != today:
                try:
                    self._learning.write_daily_review(today)
                    self._last_daily_review = today
                except Exception as e:
                    _log.error(f"Daily review error: {e}")

        # === Auto SL management (mechanical, pre-Portfolio) ===
        # Breakeven + trailing stop triggers without burning Claude calls.
        if self._trade_mgr.open_trades and trading_allowed:
            try:
                auto_actions = self._trade_mgr.evaluate_auto_management(
                    utc_now, live_prices)
                for trade_id, act in auto_actions:
                    if act.action == "move_sl" and act.new_sl:
                        if self._trade_mgr.execute_sl_move(trade_id, act.new_sl):
                            _log.info(
                                f"Auto SL: {trade_id} -> {act.new_sl:.5f} "
                                f"({act.details.get('reason', '')})")
                            result["sl_modified"] += 1
            except Exception as e:
                _log.warning(f"Auto SL management error: {e}")

        # === Portfolio Agent (manage open positions) — only during trading sessions ===
        # Disabled when ML gate active: ML trades have fixed SL/TP, no AI management needed.
        open_trades_raw = self._trade_mgr.open_trades
        if open_trades_raw and trading_allowed and not self._ml_gate:
            try:
                open_dicts = [
                    {
                        "trade_id": t.trade_id,
                        "pair": t.pair,
                        "direction": t.direction,
                        "entry_price": t.entry_price,
                        "sl_price": t.sl_price,
                        "tp_price": t.tp_price,
                        "units": t.units,
                        "open_time": t.open_time,
                        "risk_amount": t.risk_amount,
                        "last_sl_move_time": t.last_sl_move_time,
                        "exit_plan": getattr(t, "exit_plan", "") or "",
                    }
                    for t in open_trades_raw
                ]
                eligible = self._portfolio_eligible(open_dicts, utc_now)
                skip_claude = self._should_skip_portfolio_claude(
                    eligible, live_prices)
                actions = []
                if not eligible:
                    self._last_portfolio_all_hold = True
                elif skip_claude:
                    self._last_portfolio_all_hold = True
                else:
                    actions = self._portfolio.run(eligible, live_prices, utc_now)
                    self._last_portfolio_all_hold = len(actions) == 0
                    if self._last_portfolio_all_hold:
                        for t in eligible:
                            tid = str(t.get("trade_id", ""))
                            pair = t.get("pair", "")
                            lp = live_prices.get(pair)
                            if lp and isinstance(lp, tuple):
                                mid = (lp[0] + lp[1]) / 2
                                self._last_portfolio_progress[tid] = (
                                    self._progress_towards_tp_frac(t, mid))
                for action in actions:
                    self._execute_portfolio_action(action, live_prices, utc_now)
                    if action.action == "close":
                        result["trades_closed"] += 1
                    elif action.action in ("modify_sl", "set_sl"):
                        result["sl_modified"] += 1
            except Exception as e:
                _log.error(f"Portfolio Agent error: {e}")

        # === Strategy Agent (propose trade) — only during trading sessions ===
        if not trading_allowed:
            result["latency_ms"] = (time.monotonic() - start) * 1000
            _log.info(
                f"Orchestrator cycle: session closed (Asian) — "
                f"observation only | {result['latency_ms']:.0f}ms")
            return result

        if trade_limit_hit:
            _log.info(f"Orchestrator: trade limit {self._trades_executed}/{self._max_trades} — skipping strategy, portfolio still active")
            result["latency_ms"] = (time.monotonic() - start) * 1000
            return result

        try:
            pair_data = self._build_pair_data(instruments, candle_data,
                                                live_prices, utc_now)

            proposal = None

            # Check Models F/G/H first — they fire independently of ML gate
            has_fgh = False
            for row in pair_data.values():
                for key in ("model_f_signal", "model_g_signal", "model_h_signal"):
                    s = row.get(key) or ""
                    if "entry=" in s:
                        has_fgh = True
                        break
                if has_fgh:
                    break

            if has_fgh:
                self._flat_cycle_streak = 0
                risk_pct = self._config.get("risk", {}).get("risk_pct", 5.0)
                sizing_cfg = self._config.get("risk_sizing", {})
                base_risk = sizing_cfg.get("base_risk_pct", risk_pct)

                fgh_proposal = None
                for fgh_pair, row in pair_data.items():
                    for model_key, model_letter in [
                        ("model_f_signal", "model_f"),
                        ("model_g_signal", "model_g"),
                        ("model_h_signal", "model_h"),
                    ]:
                        sig_str = row.get(model_key) or ""
                        if "entry=" not in sig_str:
                            continue
                        parts = sig_str.split(", ")
                        sig_dir = None
                        sig_entry = sig_sl = sig_tp = 0.0
                        for part in parts:
                            p = part.strip()
                            if p == "long":
                                sig_dir = "long"
                            elif p == "short":
                                sig_dir = "short"
                            elif p.startswith("entry="):
                                sig_entry = float(p.split("=")[1])
                            elif p.startswith("SL="):
                                sig_sl = float(p.split("=")[1])
                            elif p.startswith("TP="):
                                sig_tp = float(p.split("=")[1])
                        if not sig_dir or sig_entry == 0:
                            continue
                        pip = 0.01 if "JPY" in fgh_pair else 0.0001
                        sl_pips = abs(sig_entry - sig_sl) / pip
                        tp_pips = abs(sig_tp - sig_entry) / pip
                        if sl_pips < 1 or tp_pips < 1:
                            continue
                        fgh_proposal = TradeProposal(
                            proposal_id=f"fgh-{__import__('uuid').uuid4().hex[:8]}",
                            pair=fgh_pair,
                            direction=sig_dir,
                            risk_pct=base_risk,
                            sl_pips=round(sl_pips, 1),
                            tp_pips=round(tp_pips, 1),
                            confidence=0.70,
                            model_source=model_letter,
                            reasoning=f"Mechanical {model_letter.upper()}: {sig_str}",
                            exit_plan=f"Fixed SL={sl_pips:.0f}p TP={tp_pips:.0f}p — no LLM override",
                        )
                        _log.info(
                            f"FGH MECHANICAL: {fgh_pair} {sig_dir} "
                            f"SL={sl_pips:.0f}p TP={tp_pips:.0f}p "
                            f"risk={base_risk}% via {model_letter}")
                        break
                    if fgh_proposal:
                        break
                proposal = fgh_proposal
            elif self._ml_gate is not None:
                ok_ins = [
                    p for p in instruments
                    if candle_data.get(p) and live_prices.get(p)
                ]
                sig = self._ml_gate.best_signal(ok_ins, candle_data)
                if sig is None:
                    self._flat_cycle_streak += 1
                    self._strategy_skip_streak = 0
                    result["latency_ms"] = (time.monotonic() - start) * 1000
                    _log.info(
                        "Orchestrator: ML gate — no score above threshold, "
                        "skipping Strategy Agent | "
                        f"{result['latency_ms']:.0f}ms")
                    return result
                self._flat_cycle_streak = 0
                (sig_pair, sig_dir, sig_prob, sig_pl, sig_ps, sig_summ) = sig
                _log.info(
                    f"ML best: {sig_pair} {sig_dir} p={sig_prob:.3f} "
                    f"(L={sig_pl:.3f} S={sig_ps:.3f})")
                risk_pct = self._config.get("risk", {}).get("risk_pct", 0.07)
                proposal = TradeProposal(
                    proposal_id=f"ml-{__import__('uuid').uuid4().hex[:8]}",
                    pair=sig_pair,
                    direction=sig_dir,
                    risk_pct=risk_pct,
                    sl_pips=10.0,
                    tp_pips=15.0,
                    confidence=sig_prob,
                    model_source="ml_gate",
                    reasoning=(
                        f"ML gate: {sig_pair} {sig_dir} p={sig_prob:.1%}. "
                        f"{sig_summ}"),
                    exit_plan="SL=10 TP=15 fixed from ML V2 training labels",
                )
            else:
                has_sig = self._has_model_entry(pair_data)
                if not has_sig:
                    self._flat_cycle_streak += 1
                    self._strategy_skip_streak = 0
                else:
                    self._flat_cycle_streak = 0

                open_pos_dicts = [
                    {"pair": t.pair, "direction": t.direction,
                     "unrealized_pnl": 0}
                    for t in self._trade_mgr.open_trades
                ]
                proposal = self._strategy.run(
                    instruments, pair_data, open_pos_dicts, nav, utc_now)

            open_pos_dicts = [
                {"pair": t.pair, "direction": t.direction,
                 "unrealized_pnl": 0}
                for t in self._trade_mgr.open_trades
            ]
            pending_for_risk = (
                self._pending_orders_fn()
                if self._pending_orders_fn else [])

            if proposal:
                self._strategy_skip_streak = 0
                result["proposals"] += 1
                verdict = self._risk.evaluate(
                    proposal, nav, margin_available,
                    open_pos_dicts, utc_now,
                    session_start_nav=self._session_start_nav,
                    peak_nav=self._peak_nav if self._peak_nav > 0 else None,
                    pending_orders=pending_for_risk,
                )

                if verdict.approved:
                    success = self._execute_trade(
                        proposal, verdict, live_prices, utc_now)
                    if success:
                        result["trades_opened"] += 1
                        self._trades_executed += 1
                        self._save_trade_count_file(utc_now)
                        _log.info(
                            f"Orchestrator: trade {self._trades_executed}/"
                            f"{self._max_trades}")
                else:
                    _log.info(
                        f"Risk Agent rejected: {verdict.reject_reason}")
            else:
                self._strategy_skip_streak += 1
                result["skips"] += 1
        except Exception as e:
            _log.error(f"Strategy/Risk pipeline error: {e}")

        result["latency_ms"] = (time.monotonic() - start) * 1000
        _log.info(
            f"Orchestrator cycle: {result['trades_opened']} opened, "
            f"{result['trades_closed']} closed, "
            f"{result['sl_modified']} SL modified, "
            f"{result['proposals']} proposals, "
            f"{result['skips']} skips | "
            f"{result['latency_ms']:.0f}ms")
        return result

    def _execute_trade(self, proposal: TradeProposal,
                       verdict: RiskVerdict,
                       live_prices: dict,
                       utc_now: datetime) -> bool:
        """Execute an approved trade."""
        pair = proposal.pair
        lp = live_prices.get(pair)
        if not lp:
            return False
        bid, ask = lp
        mid = (bid + ask) / 2

        from ..utils.pip_utils import pips_to_price
        sl_distance = pips_to_price(verdict.adjusted_sl_pips, pair)
        tp_distance = pips_to_price(verdict.adjusted_tp_pips, pair)

        if proposal.direction == "long":
            sl_price = bid - sl_distance
            tp_price = ask + tp_distance
        else:
            sl_price = ask + sl_distance
            tp_price = bid - tp_distance

        order = self._order_builder.build_market(
            pair=pair, direction=proposal.direction,
            units=verdict.adjusted_units,
            current_price=mid,
            sl_price=sl_price, tp_price=tp_price,
            signal_id=proposal.proposal_id)

        oanda_body = self._order_builder.to_oanda_order(order)
        exec_result = self._executor.submit(order, oanda_body)

        if exec_result.success:
            trade_id = exec_result.trade_id or exec_result.order_id or ""
            fill_price = exec_result.fill_price or mid
            mid_letter = self._model_id_from_source(proposal.model_source)
            cluster_id = self._cluster_id_for_new_trade(
                pair, proposal.direction, utc_now)

            from ..execution.trade_manager import ManagedTrade
            managed = ManagedTrade(
                trade_id=trade_id, pair=pair,
                direction=proposal.direction,
                entry_price=float(fill_price),
                sl_price=sl_price, tp_price=tp_price,
                units=verdict.adjusted_units,
                open_time=utc_now,
                risk_amount=sl_distance,
                model=mid_letter,
                cluster_id=cluster_id,
                exit_plan=getattr(proposal, "exit_plan", "") or "")
            self._trade_mgr.add_trade(managed)

            _log.info(
                f"TRADE FILLED: {pair} {proposal.direction} "
                f"{verdict.adjusted_units} units | "
                f"SL={sl_price:.5f} TP={tp_price:.5f} | "
                f"conf={proposal.confidence:.2f} | "
                f"model={proposal.model_source}")

            if self._logger:
                self._logger.log_trade({
                    "trade_id": trade_id, "pair": pair,
                    "direction": proposal.direction,
                    "order_type": "MARKET",
                    "units": verdict.adjusted_units,
                    "entry_price": fill_price,
                    "sl_price": sl_price, "tp_price": tp_price,
                    "broker_status": exec_result.broker_status,
                    "model": "MULTI_AGENT",
                    "confidence": proposal.confidence,
                    "reasoning": proposal.reasoning,
                    "model_source": proposal.model_source,
                })
            return True

        _log.warning(
            f"Trade not filled: {pair} {exec_result.reject_reason}")
        return False

    def _execute_portfolio_action(self, action: PortfolioAction,
                                   live_prices: dict,
                                   utc_now: datetime) -> None:
        """Execute a Portfolio Agent action."""
        if action.action == "close":
            from ..execution.trade_manager import ExitReason
            self._trade_mgr.execute_close(
                action.trade_id, ExitReason.AI_PILOT_CLOSE)
            _log.info(f"Portfolio CLOSE: {action.trade_id} ({action.reasoning})")

            # Log outcome to brain
            trade = self._trade_mgr._trades.get(action.trade_id)
            if trade:
                lp = live_prices.get(trade.pair, (0, 0))
                exit_price = (lp[0] + lp[1]) / 2 if isinstance(lp, tuple) else 0
                mult = 100 if "JPY" in trade.pair else 10000
                if trade.direction == "long":
                    pnl = (exit_price - trade.entry_price) * mult
                else:
                    pnl = (trade.entry_price - exit_price) * mult
                outcome_dict = {
                    "trade_id": action.trade_id,
                    "pair": trade.pair, "direction": trade.direction,
                    "pnl_pips": round(pnl, 2),
                    "exit_reason": "ai_pilot_close",
                    "reasoning": action.reasoning,
                    "entry_price": trade.entry_price,
                    "exit_price": exit_price,
                    "sl_price": trade.sl_price,
                    "tp_price": trade.tp_price,
                    "model_id": getattr(trade, "model", "") or "",
                    "cluster_id": getattr(trade, "cluster_id", "") or "",
                }
                self._brain.log_outcome(outcome_dict)
                # Fire post-mortem (best-effort, non-blocking conceptually)
                try:
                    self._postmortem.analyze(outcome_dict, {
                        "pair_stats": self._brain.format_pair_stats_summary(
                            min_trades=1),
                        "recent_lessons": [
                            l.get("pattern", "") for l in
                            self._brain.read_lessons(5)
                        ],
                    })
                except Exception as e:
                    _log.warning(f"Post-mortem error: {e}")

        elif action.action == "modify_sl" and action.new_sl_price > 0:
            self._trade_mgr.execute_sl_move(
                action.trade_id, action.new_sl_price)
            _log.info(
                f"Portfolio SL: {action.trade_id} -> "
                f"{action.new_sl_price:.5f} ({action.reasoning})")

        elif action.action == "set_sl" and action.new_sl_price > 0:
            # Fresh SL installation (trade had no SL). Same broker call
            # path as a move — just makes it clear in logs.
            self._trade_mgr.execute_sl_move(
                action.trade_id, action.new_sl_price)
            _log.info(
                f"Portfolio SET_SL: {action.trade_id} -> "
                f"{action.new_sl_price:.5f} ({action.reasoning})")

    def _build_pair_data(self, instruments, candle_data, live_prices, utc_now):
        """Build per-pair data dict with indicators and model signals."""
        pair_data = {}
        for pair in instruments:
            cd = candle_data.get(pair)
            lp = live_prices.get(pair)
            if not cd or not lp:
                continue
            bid, ask = lp
            spread = (ask - bid) * (100 if "JPY" in pair else 10000)

            # Indicators
            ind_m5, ind_m1 = None, None
            try:
                if cd.get("m5") is not None and len(cd["m5"]) >= 20:
                    ind_m5 = self._feature.compute(cd["m5"], "M5")
                if cd.get("m1") is not None and len(cd["m1"]) >= 20:
                    ind_m1 = self._feature.compute(cd["m1"], "M1")
            except Exception:
                pass

            # Regime
            regime_str = "unknown"
            if ind_m5:
                try:
                    close_m5 = float(cd["m5"]["close"].iloc[-1])
                    regime_result = self._regime.evaluate(ind_m5, close_m5)
                    regime_str = regime_result.regime.value
                except Exception:
                    pass

            # Model signals
            (model_a_sig, model_b_sig, model_c_sig,
             model_d_sig, model_e_sig, model_f_sig,
             model_g_sig, model_h_sig) = "", "", "", "", "", "", "", ""
            if ind_m1 and ind_m5:
                try:
                    if self._model_a:
                        from ..engine.regime_engine import Regime
                        try:
                            regime_enum = Regime(regime_str)
                        except ValueError:
                            regime_enum = Regime.RANGE
                        sig = self._model_a.evaluate(cd["m1"], ind_m1, regime_enum, pair)
                        model_a_sig = f"phase={sig.phase.value}"
                        if sig.entry_price:
                            dir_str = sig.direction.value if sig.direction else "?"
                            model_a_sig += (
                                f", {dir_str}, entry={sig.entry_price:.5f}, "
                                f"SL={sig.sl_price:.5f}, TP={sig.tp_price:.5f}")
                except Exception:
                    pass
                try:
                    if self._model_b and "range" in regime_str.lower():
                        sig = self._model_b.evaluate(
                            cd["m1"], cd["m5"], ind_m1, ind_m5,
                            Regime(regime_str) if regime_str != "unknown" else Regime.RANGE,
                            pair, spread)
                        model_b_sig = f"phase={sig.phase.value}"
                        if sig.entry_price:
                            dir_str = sig.direction.value if sig.direction else "?"
                            model_b_sig += (
                                f", {dir_str}, entry={sig.entry_price:.5f}, "
                                f"SL={sig.sl_price:.5f}, TP={sig.tp_price:.5f}")
                except Exception:
                    pass
                try:
                    if self._model_c:
                        sig = self._model_c.evaluate(cd["m1"], ind_m1, pair, spread)
                        model_c_sig = f"phase={sig.phase.value}"
                        if sig.entry_price:
                            dir_str = sig.direction.value if sig.direction else "?"
                            model_c_sig += (
                                f", {dir_str}, entry={sig.entry_price:.5f}, "
                                f"SL={sig.sl_price:.5f}, TP={sig.tp_price:.5f}")
                except Exception:
                    pass
                try:
                    if self._model_d:
                        sig_d = self._model_d.evaluate(
                            cd["m1"], ind_m5, pair, spread,
                            utc_now.hour, utc_now.weekday())
                        model_d_sig = f"phase={sig_d.phase.value}"
                        if sig_d.entry_price:
                            model_d_sig += (
                                f", {sig_d.direction.value if sig_d.direction else ''}"
                                f", entry={sig_d.entry_price:.5f}, "
                                f"SL={sig_d.sl_price:.5f}, TP={sig_d.tp_price:.5f}")
                            if sig_d.values and "hist_wr" in sig_d.values:
                                model_d_sig += (
                                    f", hist_WR={sig_d.values['hist_wr']:.1%}")
                except Exception as e:
                    _log.debug(f"Model D evaluate error: {e}")
                try:
                    if self._model_e:
                        sig_e = self._model_e.evaluate(
                            cd["m5"], ind_m5, pair, spread, utc_now.hour,
                            regime=regime_str)
                        model_e_sig = f"phase={sig_e.phase.value}"
                        if sig_e.entry_price:
                            model_e_sig += (
                                f", {sig_e.direction.value if sig_e.direction else ''}"
                                f", entry={sig_e.entry_price:.5f}, "
                                f"SL={sig_e.sl_price:.5f}, TP={sig_e.tp_price:.5f}")
                            if sig_e.values and "deviation_atr" in sig_e.values:
                                model_e_sig += (
                                    f", dev={sig_e.values['deviation_atr']:+.2f}atr")
                except Exception as e:
                    _log.debug(f"Model E evaluate error: {e}")
                try:
                    df_h1 = cd.get("h1")
                    if getattr(self, "_model_f", None) and df_h1 is not None and len(df_h1) >= 30:
                        sig_f = self._model_f.evaluate(
                            df_h1, ind_m5, pair, spread,
                            utc_now.hour, utc_now.weekday())
                        model_f_sig = f"phase={sig_f.phase.value}"
                        if sig_f.entry_price:
                            model_f_sig += (
                                f", {sig_f.direction.value if sig_f.direction else ''}"
                                f", entry={sig_f.entry_price:.5f}, "
                                f"SL={sig_f.sl_price:.5f}, TP={sig_f.tp_price:.5f}")
                            if sig_f.values:
                                model_f_sig += (
                                    f", asia_range={sig_f.values.get('asia_range_pips', 0):.0f}p"
                                    f", adx={sig_f.values.get('adx', 0):.0f}")
                    if sig_f.values and sig_f.values.get("reason"):
                        _log.info(f"Model F ({pair}): {sig_f.values['reason']}")
                except Exception as e:
                    _log.info(f"Model F (TKY_LDN) evaluate error: {e}")
                try:
                    if getattr(self, "_model_g", None) and df_h1 is not None and len(df_h1) >= 30:
                        sig_g = self._model_g.evaluate(
                            df_h1, ind_m5, pair, spread,
                            utc_now.hour, utc_now.weekday())
                        model_g_sig = f"phase={sig_g.phase.value}"
                        if sig_g.entry_price:
                            model_g_sig += (
                                f", {sig_g.direction.value if sig_g.direction else ''}"
                                f", entry={sig_g.entry_price:.5f}, "
                                f"SL={sig_g.sl_price:.5f}, TP={sig_g.tp_price:.5f}")
                            if sig_g.values:
                                model_g_sig += f", adx={sig_g.values.get('adx', 0):.0f}"
                    if sig_g.values and sig_g.values.get("reason"):
                        _log.info(f"Model G ({pair}): {sig_g.values['reason']}")
                except Exception as e:
                    _log.info(f"Model G (HA_ADX) evaluate error: {e}")
                try:
                    if getattr(self, "_model_h", None) and df_h1 is not None and len(df_h1) >= 30:
                        sig_h = self._model_h.evaluate(
                            df_h1, ind_m5, pair, spread,
                            utc_now.hour, utc_now.weekday())
                        model_h_sig = f"phase={sig_h.phase.value}"
                        if sig_h.entry_price:
                            model_h_sig += (
                                f", {sig_h.direction.value if sig_h.direction else ''}"
                                f", entry={sig_h.entry_price:.5f}, "
                                f"SL={sig_h.sl_price:.5f}, TP={sig_h.tp_price:.5f}")
                            if sig_h.values:
                                model_h_sig += f", nr_range={sig_h.values.get('nr_range_pips', 0):.0f}p"
                    if sig_h.values and sig_h.values.get("reason"):
                        _log.info(f"Model H ({pair}): {sig_h.values['reason']}")
                except Exception as e:
                    _log.info(f"Model H (NR7) evaluate error: {e}")

            def ind_dict(ind):
                if not ind:
                    return {}
                return {
                    "ema20": ind.ema20, "ema50": ind.ema50,
                    "ema_slope": ind.ema_slope, "rsi14": ind.rsi14,
                    "atr14": ind.atr14, "bb_width": ind.bb_width,
                }

            mtf_str = ""
            mtf_snap = None
            if self._mtf is not None:
                try:
                    mtf_snap = self._mtf.get(pair, utc_now)
                    from ..engine.mtf_context import format_snapshot
                    mtf_str = format_snapshot(mtf_snap)
                except Exception:
                    mtf_str = ""

            news_risk = ""
            try:
                risk_title = self._news_gate.check_elevated_risk(pair, utc_now)
                if risk_title:
                    news_risk = risk_title[:60]
            except Exception:
                news_risk = ""

            pair_data[pair] = {
                "bid": bid, "ask": ask, "spread_pips": spread,
                "regime": regime_str,
                "indicators_m5": ind_dict(ind_m5),
                "indicators_m1": ind_dict(ind_m1),
                "model_a_signal": model_a_sig,
                "model_b_signal": model_b_sig,
                "model_c_signal": model_c_sig,
                "model_d_signal": model_d_sig,
                "model_e_signal": model_e_sig,
                "model_f_signal": model_f_sig,
                "model_g_signal": model_g_sig,
                "model_h_signal": model_h_sig,
                "mtf_context": mtf_str,
                "mtf_snapshot": mtf_snap,
                "news_elevated_risk": news_risk,
            }
        return pair_data
