"""Model D — Hour-conditional mean-reversion scalp ("Strategy P").

Surfaced via a multi-year, multi-pair, 1-minute parameter sweep.
NOTE: in-sample sweep metrics overfit and are NOT predictive — honest
out-of-sample evaluation shows the rule strategies lose money. Treat this
as a research signal only, not a profitable edge. Educational use only.

Entry rules:
- LONG hours (UTC): 00, 07, 20 — after prev 5-bar drop >= 10 pips
- SHORT hours (UTC): 19, 22 — after prev 5-bar rally >= 10 pips
- Requires ATR(14) on M5 >= 3 pips
- Weekdays only, spread <= 30% of TP
- TP=4 pips, SL=5 pips

Matches the Model A/B/C interface so Strategy Agent treats it uniformly.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .model_a import TriggerSignal, TriggerPhase, Direction
from .feature_engine import IndicatorSet
from ..utils.pip_utils import price_to_pips


# Configuration (will be loaded from settings.yaml in production)
LONG_HOURS = frozenset({0, 7, 20})
SHORT_HOURS = frozenset({19, 22})
TRIGGER_MOVE_PIPS = 10.0
ATR_MIN_PIPS = 3.0
TP_PIPS = 4.0
SL_PIPS = 5.0
MAX_HOLD_BARS = 15
SUPPORTED_PAIRS = frozenset({
    "EUR_USD",
    "GBP_USD",
    "AUD_USD",
    "USD_CAD",
    "NZD_USD",
})

# Per-hour historical WR (used for confidence scoring)
HOUR_WR = {
    0: 0.688, 7: 0.557, 20: 0.710,
    19: 0.632, 22: 0.687,
}


class ModelDTrigger:
    """Hour-conditional mean reversion. Returns a TriggerSignal."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._long_hours = frozenset(cfg.get("long_hours", list(LONG_HOURS)))
        self._short_hours = frozenset(cfg.get("short_hours", list(SHORT_HOURS)))
        self._trigger_pips = float(cfg.get("trigger_move_pips", TRIGGER_MOVE_PIPS))
        self._atr_min = float(cfg.get("atr_min_pips", ATR_MIN_PIPS))
        self._tp_pips = float(cfg.get("tp_pips", TP_PIPS))
        self._sl_pips = float(cfg.get("sl_pips", SL_PIPS))
        self._pairs = frozenset(cfg.get("pairs", list(SUPPORTED_PAIRS)))

    def evaluate(
        self,
        df_m1: pd.DataFrame,
        ind_m5: IndicatorSet,
        pair: str,
        spread_pips: float,
        current_utc_hour: int,
        day_of_week: int,
    ) -> TriggerSignal:
        """Scan for a Model D setup.

        Returns TriggerSignal(VALID) if conditions met; otherwise NO_BREAKOUT.
        """
        # Gate: weekdays only
        if day_of_week >= 5:
            return TriggerSignal(phase=TriggerPhase.NO_BREAKOUT,
                                 values={"reason": "weekend"})

        # Gate: supported pairs
        if pair not in self._pairs:
            return TriggerSignal(phase=TriggerPhase.NO_BREAKOUT,
                                 values={"reason": "pair_excluded"})

        # Gate: hour must be in edge window
        is_long_hour = current_utc_hour in self._long_hours
        is_short_hour = current_utc_hour in self._short_hours
        if not (is_long_hour or is_short_hour):
            return TriggerSignal(
                phase=TriggerPhase.NO_BREAKOUT,
                values={"reason": f"hour_{current_utc_hour}_no_edge"})

        # Gate: ATR must be high enough
        pip_size = 0.01 if "JPY" in pair else 0.0001
        atr_pips = (ind_m5.atr14 or 0) / pip_size if ind_m5 else 0
        if atr_pips < self._atr_min:
            return TriggerSignal(
                phase=TriggerPhase.NO_BREAKOUT,
                values={"reason": f"atr_too_low_{atr_pips:.1f}"})

        # Gate: spread limit
        if spread_pips > 0.30 * self._tp_pips:
            return TriggerSignal(
                phase=TriggerPhase.NO_BREAKOUT,
                values={"reason": f"spread_{spread_pips:.1f}_too_wide"})

        # Need at least 6 M1 candles for prev-5 calc
        if df_m1 is None or len(df_m1) < 6:
            return TriggerSignal(phase=TriggerPhase.NO_BREAKOUT,
                                 values={"reason": "insufficient_data"})

        closes = df_m1["close"].iloc[-6:].tolist()
        prev5_price = closes[-1] - closes[0]
        prev5_pips = prev5_price / pip_size

        current_price = float(df_m1["close"].iloc[-1])

        # Long setup
        if is_long_hour and prev5_pips <= -self._trigger_pips:
            entry = current_price
            sl = entry - self._sl_pips * pip_size
            tp = entry + self._tp_pips * pip_size
            wr = HOUR_WR.get(current_utc_hour, 0.55)
            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=Direction.LONG,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                risk_pips=self._sl_pips,
                values={
                    "prev5_pips": round(prev5_pips, 2),
                    "hour": current_utc_hour,
                    "atr_pips": round(atr_pips, 2),
                    "hist_wr": wr,
                    "reason": (
                        f"long-mean-revert h{current_utc_hour:02d} "
                        f"after {prev5_pips:+.1f} prev5"),
                },
            )

        # Short setup
        if is_short_hour and prev5_pips >= self._trigger_pips:
            entry = current_price
            sl = entry + self._sl_pips * pip_size
            tp = entry - self._tp_pips * pip_size
            wr = HOUR_WR.get(current_utc_hour, 0.55)
            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=Direction.SHORT,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                risk_pips=self._sl_pips,
                values={
                    "prev5_pips": round(prev5_pips, 2),
                    "hour": current_utc_hour,
                    "atr_pips": round(atr_pips, 2),
                    "hist_wr": wr,
                    "reason": (
                        f"short-mean-revert h{current_utc_hour:02d} "
                        f"after {prev5_pips:+.1f} prev5"),
                },
            )

        return TriggerSignal(
            phase=TriggerPhase.NO_BREAKOUT,
            values={
                "reason": f"no_trigger_move prev5={prev5_pips:+.1f}",
                "hour": current_utc_hour,
            },
        )
