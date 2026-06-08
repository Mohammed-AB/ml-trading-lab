"""Model C: EMA Crossover — research strategy (no proven edge).

Surfaced via a large Optuna parameter sweep over historical data.
Example config: EMA fast=5, slow=40, ATR mult=1.35, R:R=0.5, spread filter.

NOTE: in-sample sweep metrics for strategies like this are not predictive —
honest out-of-sample evaluation (see docs/ARENA_LEADERBOARD.md) shows these
rule strategies lose money. Treat any historical win-rate figure as an
overfit artifact, not a live expectation. Educational/research use only.

Uses the same TriggerSignal/TriggerPhase as Model A for pipeline compatibility.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .model_a import TriggerSignal, TriggerPhase, Direction
from .feature_engine import IndicatorSet
from ..utils.pip_utils import pips_to_price


class ModelCTrigger:
    """EMA Crossover strategy with ATR-based stops.

    Enters when fast EMA crosses slow EMA, with spread filter
    and ATR-based SL/TP.

    Usage:
        trigger_c = ModelCTrigger(config)
        signal = trigger_c.evaluate(df_m1, ind_m1, pair, spread_pips)
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._ema_fast = cfg.get("ema_fast", 5)
        self._ema_slow = cfg.get("ema_slow", 40)
        self._atr_mult = cfg.get("atr_mult", 1.3549)
        self._rr = cfg.get("rr", 0.5009)
        self._max_spread_pips = cfg.get("max_spread_pips", 2.0)
        self._use_spread_filter = cfg.get("spread_filter", True)

    def evaluate(self, df_m1: pd.DataFrame, ind_m1: IndicatorSet,
                 pair: str, spread_pips: float = 0) -> TriggerSignal:
        """Evaluate EMA crossover on M1 data.

        Returns TriggerSignal with VALID when a crossover is detected.
        """
        min_rows = self._ema_slow + 2
        if len(df_m1) < min_rows:
            return TriggerSignal(
                phase=TriggerPhase.NO_COMPRESSION,
                values={"reason": "insufficient_data",
                        "rows": len(df_m1), "needed": min_rows})

        close = df_m1["close"]
        ema_fast = close.ewm(span=self._ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self._ema_slow, adjust=False).mean()

        fast_now = float(ema_fast.iloc[-1])
        fast_prev = float(ema_fast.iloc[-2])
        slow_now = float(ema_slow.iloc[-1])
        slow_prev = float(ema_slow.iloc[-2])

        # Detect crossover
        cross_up = fast_prev <= slow_prev and fast_now > slow_now
        cross_down = fast_prev >= slow_prev and fast_now < slow_now

        if not cross_up and not cross_down:
            return TriggerSignal(
                phase=TriggerPhase.NO_COMPRESSION,
                values={"reason": "no_crossover",
                        "ema_fast": round(fast_now, 6),
                        "ema_slow": round(slow_now, 6),
                        "gap": round(fast_now - slow_now, 6)})

        # Spread filter
        if self._use_spread_filter and spread_pips > self._max_spread_pips:
            return TriggerSignal(
                phase=TriggerPhase.NO_COMPRESSION,
                values={"reason": "spread_too_wide",
                        "spread": spread_pips,
                        "max": self._max_spread_pips})

        direction = Direction.LONG if cross_up else Direction.SHORT
        atr = ind_m1.atr14 if ind_m1.atr14 and ind_m1.atr14 > 0 else 0.0005
        current_price = float(close.iloc[-1])

        # ATR-based SL and TP
        sl_distance = atr * self._atr_mult
        tp_distance = sl_distance * self._rr

        if direction == Direction.LONG:
            entry_price = current_price
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            entry_price = current_price
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        sl_pips = sl_distance * (100 if "JPY" in pair else 10000)
        tp_pips = tp_distance * (100 if "JPY" in pair else 10000)

        return TriggerSignal(
            phase=TriggerPhase.VALID,
            direction=direction,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_pips=round(sl_pips, 1),
            values={
                "model": "C_EMA_CROSSOVER",
                "ema_fast": round(fast_now, 6),
                "ema_slow": round(slow_now, 6),
                "atr": round(atr, 6),
                "sl_pips": round(sl_pips, 1),
                "tp_pips": round(tp_pips, 1),
                "rr_ratio": round(self._rr, 3),
                "spread_pips": round(spread_pips, 1),
                "edge_note": "research signal — no proven out-of-sample edge",
            },
        )
