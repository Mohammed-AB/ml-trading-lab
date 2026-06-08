"""Backtester — Replays historical candles through the decision pipeline.

Simulates the full V1 decision sequence bar-by-bar on M1 candles,
with M5 context resampled from M1 data. Models execution costs
(spread + slippage) per spec section 6.

Outputs a list of simulated trades for performance analysis.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from ..engine.feature_engine import FeatureEngine, IndicatorSet
from ..engine.regime_engine import RegimeEngine, Regime
from ..engine.model_a import ModelATrigger, TriggerPhase, Direction
from ..engine.model_b import ModelBTrigger
from ..engine.cooldown import CooldownManager, TradeRecord
from ..gates.session_gate import is_session_allowed
from ..gates.news_gate import NewsGate
from ..gates.spread_filter import check_spread
from ..utils.pip_utils import price_to_pips, pips_to_price


@dataclass
class BacktestTrade:
    """A completed simulated trade."""
    pair: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    units: int
    pnl_pips: float
    pnl_pct: float          # % of NAV at entry
    exit_reason: str         # tp_hit, sl_hit, time_stop
    hold_time_seconds: int
    spread_at_entry: float   # pips
    slippage_pips: float
    is_borderline: bool = False
    borderline_flags: Optional[list[str]] = None
    model: str = "A"  # "A" (Trend) or "B" (Range)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_nav: float = 10000.0
    spread_model: str = "session_variable"  # "fixed", "session_variable", or "from_data"
    fixed_spread_pips: float = 0.5     # Used when spread_model="fixed"
    slippage_pips: float = 0.1         # Simulated slippage per trade
    reject_rate: float = 0.0           # Simulated execution rejection rate (0.0-1.0)
    warmup_bars: int = 60              # Bars to skip for indicator warmup
    m5_resample: bool = True           # Resample M5 from M1
    check_sessions: bool = True        # Apply session gate
    check_news: bool = False           # Apply news gate (needs events)


class Backtester:
    """Runs a backtest on historical M1 candle data.

    Usage:
        bt = Backtester(config, backtest_config)
        trades = bt.run(pair, df_m1)
        # Analyze trades with PerformanceAnalyzer
    """

    def __init__(self, scalp_config: dict, bt_config: Optional[BacktestConfig] = None):
        self._cfg = scalp_config
        self._bt = bt_config or BacktestConfig()

        # Build engines
        self._feature = FeatureEngine()
        self._regime = RegimeEngine(
            scalp_config.get("regime", {
                "trend": {"ema_slope_thr": 0.20, "rsi_min": 52, "rsi_max": 78},
                "range": {"bb_width_thr": 0.004},
            }),
            scalp_config.get("borderline"),
        )
        self._trigger = ModelATrigger(
            scalp_config.get("model_a", {
                "compression_N": 8, "breakout_buffer_atr": 0.10,
                "retest_timeout": 3, "retest_tolerance_atr": 0.15,
                "body_ratio_min": 0.55, "rsi_min_long": 55,
                "sl_atr": 0.8, "tp_R": 1.0, "time_stop_min": 6,
                "sl_move_threshold_R": 0.8, "sl_move_target_R": -0.1,
                "sl_move_window_min": [2, 4],
            }),
            scalp_config.get("borderline"),
        )
        model_b_cfg = scalp_config.get("model_b", {})
        self._trigger_b = None
        if model_b_cfg.get("enabled", False):
            self._trigger_b = ModelBTrigger(model_b_cfg, scalp_config.get("borderline"))
        risk_cfg = scalp_config.get("risk", {
            "cooldown_same_pair_dir_min": 10, "consec_loss_circuit": 3,
            "cooldown_minutes": 60, "trades_per_hour_pair": 3,
            "trades_per_hour_total": 6, "daily_loss": 0.01,
        })
        self._cooldown = CooldownManager(risk_cfg)
        self._news_gate = NewsGate()

        self._risk_pct = risk_cfg.get("risk_pct", 0.0025)
        self._max_concurrent = risk_cfg.get("max_concurrent", 2)
        self._time_stop_min = scalp_config.get("model_a", {}).get("time_stop_min", 6)
        self._sl_move_R = scalp_config.get("model_a", {}).get("sl_move_threshold_R", 0.8)
        self._sl_move_target = scalp_config.get("model_a", {}).get("sl_move_target_R", -0.1)
        self._sl_move_window = scalp_config.get("model_a", {}).get("sl_move_window_min", [2, 4])

    def run(self, pair: str, df_m1: pd.DataFrame,
            timestamps: Optional[pd.Series] = None) -> list[BacktestTrade]:
        """Run backtest on M1 candle data.

        Args:
            pair: Instrument (e.g., "EUR_USD")
            df_m1: DataFrame with columns: open, high, low, close, volume.
                   Optionally a 'timestamp' column for session filtering.
            timestamps: Optional Series of UTC datetimes for each bar.
                       If not provided, session gate is skipped.

        Returns:
            List of BacktestTrade results.
        """
        trades: list[BacktestTrade] = []
        nav = self._bt.initial_nav
        open_trades: list[dict] = []  # Active simulated trades

        # Resample M5 from M1
        if self._bt.m5_resample and len(df_m1) >= 5:
            df_m5 = self._resample_m5(df_m1)
        else:
            df_m5 = df_m1.copy()

        max_spread = self._cfg.get("costs", {}).get(
            "max_spread_pips", {}).get(pair, 0.8)

        # --- Pre-compute ALL indicators once (massive speedup) ---
        m1_series = self._feature.compute_series(df_m1, "M1")
        m5_series = self._feature.compute_series(df_m5, "M5")

        def _val(series_dict, key, idx):
            s = series_dict.get(key)
            if s is None or idx >= len(s):
                return None
            v = s.iloc[idx]
            return None if pd.isna(v) else float(v)

        def _make_ind(series_dict, idx, timeframe="M1"):
            ind = IndicatorSet(
                ema20=_val(series_dict, "ema20", idx),
                ema50=_val(series_dict, "ema50", idx),
                atr14=_val(series_dict, "atr14", idx),
                rsi14=_val(series_dict, "rsi14", idx),
                bb_upper=_val(series_dict, "bb_upper", idx),
                bb_mid=_val(series_dict, "bb_mid", idx),
                bb_lower=_val(series_dict, "bb_lower", idx),
                bb_width=_val(series_dict, "bb_width", idx),
                ema_slope=_val(series_dict, "ema_slope", idx),
            )
            if timeframe == "M1" and "macd_hist" in series_dict:
                ind.macd_hist = _val(series_dict, "macd_hist", idx)
                ind.macd_hist_prev = _val(series_dict, "macd_hist", idx - 1) if idx >= 1 else None
                ind.macd_hist_prev2 = _val(series_dict, "macd_hist", idx - 2) if idx >= 2 else None
            return ind

        for i in range(self._bt.warmup_bars, len(df_m1)):
            bar = df_m1.iloc[i]
            bar_time = None
            if timestamps is not None and i < len(timestamps):
                bar_time = timestamps.iloc[i]
                if not isinstance(bar_time, datetime):
                    try:
                        bar_time = pd.Timestamp(bar_time).to_pydatetime()
                        if bar_time.tzinfo is None:
                            bar_time = bar_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        bar_time = None

            # --- Manage open trades first ---
            closed = []
            for ot in open_trades:
                result = self._manage_trade(ot, bar, i, bar_time)
                if result is not None:
                    result_trade = self._close_trade(ot, result, bar_time or
                                                      datetime(2026, 1, 1, tzinfo=timezone.utc),
                                                      nav)
                    trades.append(result_trade)
                    nav += result_trade.pnl_pct * nav  # Approximate compounding
                    self._cooldown.record_trade(TradeRecord(
                        pair=ot["pair"], direction=ot["direction"],
                        timestamp_utc=result_trade.exit_time,
                        pnl_pct=result_trade.pnl_pct,
                    ))
                    closed.append(ot)
            for c in closed:
                open_trades.remove(c)

            # --- Skip if at max concurrent ---
            if len(open_trades) >= self._max_concurrent:
                continue

            # --- Session gate ---
            if self._bt.check_sessions and bar_time:
                sess = self._cfg.get("sessions", {})
                session = is_session_allowed(
                    bar_time,
                    mode=sess.get("mode", "overlap_only"),
                    block=sess.get("block"),
                )
                if not session.allowed:
                    continue

            # --- Spread filter ---
            spread_pips = self._get_spread(pair, bar_time)
            spread_result = check_spread(
                bid=float(bar["close"]) - pips_to_price(spread_pips / 2, pair),
                ask=float(bar["close"]) + pips_to_price(spread_pips / 2, pair),
                pair=pair, max_spread_pips=max_spread)
            if not spread_result.is_ok:
                continue

            # --- Cooldown ---
            direction_guess = "long"  # Will be refined by regime
            if bar_time:
                cd = self._cooldown.check(pair, direction_guess, bar_time)
                if not cd.is_ok:
                    continue

            # --- Indicators (pre-computed, just lookup) ---
            if i < 50:
                continue

            ind_m1 = _make_ind(m1_series, i, "M1")

            m5_idx = i // 5
            if m5_idx < 20:
                continue
            ind_m5 = _make_ind(m5_series, m5_idx, "M5")

            # NaN check
            has_nan, _ = ind_m5.has_nan()
            if has_nan:
                continue
            has_nan, _ = ind_m1.has_nan()
            if has_nan:
                continue

            # Lookbacks for trigger (still needed for pattern detection)
            lookback_m1 = df_m1.iloc[max(0, i - 99):i + 1]
            lookback_m5 = df_m5.iloc[max(0, m5_idx - 49):m5_idx + 1]

            # --- Regime ---
            close_m5 = float(lookback_m5.iloc[-1]["close"])
            regime_result = self._regime.evaluate(ind_m5, close_m5)
            if regime_result.regime == Regime.NO_TRADE:
                continue

            # --- Trigger ---
            if regime_result.regime in (Regime.TREND_UP, Regime.TREND_DOWN):
                trigger = self._trigger.evaluate(
                    lookback_m1, ind_m1, regime_result.regime, pair)
                trade_model = "A"
            elif regime_result.regime == Regime.RANGE and self._trigger_b:
                trigger = self._trigger_b.evaluate(
                    lookback_m1, lookback_m5, ind_m1, ind_m5,
                    regime_result.regime, pair, spread_pips=spread_pips)
                trade_model = "B"
            else:
                continue
            if trigger.phase != TriggerPhase.VALID:
                continue

            # --- Position sizing ---
            stop_pips = trigger.risk_pips or 0
            if stop_pips <= 0:
                continue
            from ..utils.pip_utils import pip_value
            risk_amount = nav * self._risk_pct
            units = int(risk_amount / (stop_pips * pip_value(pair)))
            if units <= 0:
                continue

            # --- Simulate execution rejection ---
            if self._bt.reject_rate > 0 and np.random.random() < self._bt.reject_rate:
                continue

            # --- Simulate entry with costs ---
            entry_price = trigger.entry_price
            if trigger.direction == Direction.LONG:
                entry_price += pips_to_price(self._bt.slippage_pips, pair)
            else:
                entry_price -= pips_to_price(self._bt.slippage_pips, pair)

            open_trades.append({
                "pair": pair,
                "direction": trigger.direction.value,
                "model": trade_model,
                "entry_price": entry_price,
                "sl_price": trigger.sl_price,
                "tp_price": trigger.tp_price,
                "units": units,
                "entry_bar": i,
                "entry_time": bar_time,
                "risk_amount": abs(entry_price - trigger.sl_price),
                "spread_at_entry": spread_pips,
                "slippage": self._bt.slippage_pips,
                "sl_moved": False,
                "is_borderline": regime_result.is_borderline or bool(trigger.borderline_flags),
                "borderline_flags": list(set(
                    (regime_result.borderline_flags or []) +
                    (trigger.borderline_flags or [])
                )) or None,
            })

        # Close any remaining open trades at the last bar
        last_bar = df_m1.iloc[-1]
        last_time = None
        if timestamps is not None and len(timestamps) > 0:
            last_time = timestamps.iloc[-1]
            if not isinstance(last_time, datetime):
                try:
                    last_time = pd.Timestamp(last_time).to_pydatetime()
                    if last_time.tzinfo is None:
                        last_time = last_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    last_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        for ot in open_trades:
            result_trade = self._close_trade(
                ot, {"exit_price": float(last_bar["close"]), "reason": "end_of_data"},
                last_time or datetime(2026, 1, 1, tzinfo=timezone.utc), nav)
            trades.append(result_trade)

        return trades

    def _manage_trade(self, ot: dict, bar: pd.Series, bar_idx: int,
                      bar_time: Optional[datetime]) -> Optional[dict]:
        """Check if an open trade should be closed or SL moved.

        Returns dict with exit_price and reason if closed, None if held.
        """
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        bars_held = bar_idx - ot["entry_bar"]

        if ot["direction"] == "long":
            # SL hit
            if low <= ot["sl_price"]:
                return {"exit_price": ot["sl_price"], "reason": "sl_hit"}
            # TP hit
            if high >= ot["tp_price"]:
                return {"exit_price": ot["tp_price"], "reason": "tp_hit"}

            # SL move check
            risk = ot["risk_amount"]
            current_R = (close - ot["entry_price"]) / risk if risk > 0 else 0
            if (not ot["sl_moved"]
                    and self._sl_move_window[0] <= bars_held <= self._sl_move_window[1]
                    and current_R >= self._sl_move_R):
                sl_offset = abs(self._sl_move_target) * risk
                ot["sl_price"] = ot["entry_price"] - sl_offset  # Below entry for long
                ot["sl_moved"] = True

        else:  # short
            if high >= ot["sl_price"]:
                return {"exit_price": ot["sl_price"], "reason": "sl_hit"}
            if low <= ot["tp_price"]:
                return {"exit_price": ot["tp_price"], "reason": "tp_hit"}

            risk = ot["risk_amount"]
            current_R = (ot["entry_price"] - close) / risk if risk > 0 else 0
            if (not ot["sl_moved"]
                    and self._sl_move_window[0] <= bars_held <= self._sl_move_window[1]
                    and current_R >= self._sl_move_R):
                sl_offset = abs(self._sl_move_target) * risk
                ot["sl_price"] = ot["entry_price"] + sl_offset  # Above entry for short
                ot["sl_moved"] = True

        # Time stop
        if bars_held >= self._time_stop_min:
            risk = ot["risk_amount"]
            if ot["direction"] == "long":
                current_R = (close - ot["entry_price"]) / risk if risk > 0 else 0
            else:
                current_R = (ot["entry_price"] - close) / risk if risk > 0 else 0
            if current_R < 0.5:
                return {"exit_price": close, "reason": "time_stop"}

        return None

    def _close_trade(self, ot: dict, result: dict, exit_time: datetime,
                     nav: float) -> BacktestTrade:
        """Create a BacktestTrade from an open trade and exit result."""
        exit_price = result["exit_price"]
        if ot["direction"] == "long":
            pnl_price = exit_price - ot["entry_price"]
        else:
            pnl_price = ot["entry_price"] - exit_price

        pnl_pips = price_to_pips(pnl_price, ot["pair"])
        pnl_pct = (pnl_price * ot["units"]) / nav if nav > 0 else 0

        entry_time = ot.get("entry_time") or datetime(2026, 1, 1, tzinfo=timezone.utc)
        hold_seconds = int((exit_time - entry_time).total_seconds()) if entry_time else 0

        return BacktestTrade(
            pair=ot["pair"],
            direction=ot["direction"],
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=ot["entry_price"],
            exit_price=exit_price,
            sl_price=ot["sl_price"],
            tp_price=ot["tp_price"],
            units=ot["units"],
            pnl_pips=round(pnl_pips, 2),
            pnl_pct=round(pnl_pct, 6),
            exit_reason=result["reason"],
            hold_time_seconds=max(hold_seconds, 0),
            spread_at_entry=ot.get("spread_at_entry", 0),
            slippage_pips=ot.get("slippage", 0),
            is_borderline=ot.get("is_borderline", False),
            borderline_flags=ot.get("borderline_flags"),
            model=ot.get("model", "A"),
        )

    def _get_spread(self, pair: str, bar_time: Optional[datetime]) -> float:
        """Get spread for a bar based on spread model.

        Models:
        - "fixed": constant spread (default)
        - "session_variable": wider spread outside peak overlap, tighter during overlap
        """
        base = self._bt.fixed_spread_pips

        if self._bt.spread_model == "session_variable" and bar_time:
            hour = bar_time.hour
            # Peak overlap (12-16 UTC): base spread
            # London/NY solo (8-12, 16-22): +30% wider
            # Asian/off-hours (22-8): +80% wider
            if 12 <= hour < 16:
                return base  # Tightest during overlap
            elif 8 <= hour < 12 or 16 <= hour < 22:
                return base * 1.3
            else:
                return base * 1.8

        return base

    @staticmethod
    def _resample_m5(df_m1: pd.DataFrame) -> pd.DataFrame:
        """Resample M1 candles to M5 by grouping every 5 bars."""
        n = len(df_m1)
        groups = n // 5
        if groups == 0:
            return df_m1.copy()

        trimmed = df_m1.iloc[:groups * 5].copy()
        trimmed["group"] = np.repeat(range(groups), 5)

        m5 = trimmed.groupby("group").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).reset_index(drop=True)

        return m5
