"""Model H -- NR7 Expansion Breakout.

Detects the bar with the smallest range in the last 7 bars.
On the next bar, enters long above NR7 high or short below NR7 low
with ATR offset for confirmation.

Backtest OOS: PF 1.61, 63 trades, +395 pips.
Best pairs: EUR_USD (PF 4.14), AUD_USD (PF 3.09).
Best hours: H18-19. Best day: Wednesday.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .model_a import Direction, TriggerPhase, TriggerSignal
from .feature_engine import IndicatorSet


class ModelNr7Trigger:
    """NR7 narrowest-range-7 breakout on M5."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._lookback = int(cfg.get("lookback", 7))
        self._offset_atr_mult = float(cfg.get("offset_atr_mult", 0.2))
        self._tp_risk_mult = float(cfg.get("tp_risk_mult", 3.0))
        self._min_adx = float(cfg.get("min_adx", 25))
        self._active_hours = set(cfg.get("active_hours", list(range(7, 20))))
        self._skip_dow = set(cfg.get("skip_dow", []))
        self._skip_pairs = set(cfg.get("skip_pairs", ["GBP_USD"]))

    def evaluate(
        self,
        df_m5: pd.DataFrame,
        ind_m5: IndicatorSet | None,
        pair: str,
        spread_pips: float,
        current_utc_hour: int,
        current_dow: int,
    ) -> TriggerSignal:
        no = lambda reason: TriggerSignal(phase=TriggerPhase.NO_BREAKOUT, values={"reason": reason})

        if pair in self._skip_pairs:
            return no("pair_skipped")
        if current_utc_hour not in self._active_hours:
            return no(f"hour_{current_utc_hour}_inactive")
        if current_dow in self._skip_dow:
            return no("dow_skipped")
        if df_m5 is None or len(df_m5) < self._lookback + 5:
            return no("insufficient_data")

        atr = (ind_m5.atr14 or 0) if ind_m5 else 0
        if atr <= 0:
            return no("no_atr")

        pip = 0.01 if "JPY" in pair else 0.0001
        h = df_m5["high"].values
        l = df_m5["low"].values
        c = df_m5["close"].values

        n = len(df_m5)
        i = n - 1

        adx_val = self._compute_adx_last(h, l, c)
        if adx_val < self._min_adx:
            return no(f"adx_{adx_val:.0f}_low")

        ranges = h[i - self._lookback:i] - l[i - self._lookback:i]
        if len(ranges) < self._lookback:
            return no("not_enough_range_bars")

        nr_idx = int(np.argmin(ranges))
        abs_nr_idx = i - self._lookback + nr_idx

        if abs_nr_idx != i - 1:
            return no("nr7_not_prev_bar")

        nr_high = float(h[abs_nr_idx])
        nr_low = float(l[abs_nr_idx])
        nr_range = nr_high - nr_low
        offset = self._offset_atr_mult * atr

        bar_close = float(c[i])
        bar_high = float(h[i])
        bar_low = float(l[i])

        if bar_close > nr_high + offset:
            entry = bar_close
            sl = nr_low
            risk = entry - sl
            if risk < pip:
                return no("risk_too_small")
            tp = entry + self._tp_risk_mult * risk
            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=Direction.LONG,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                risk_pips=round(risk / pip, 1),
                values={"nr_range_pips": round(nr_range / pip, 1), "adx": round(adx_val, 1)},
            )

        if bar_close < nr_low - offset:
            entry = bar_close
            sl = nr_high
            risk = sl - entry
            if risk < pip:
                return no("risk_too_small")
            tp = entry - self._tp_risk_mult * risk
            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=Direction.SHORT,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                risk_pips=round(risk / pip, 1),
                values={"nr_range_pips": round(nr_range / pip, 1), "adx": round(adx_val, 1)},
            )

        return no("no_breakout")

    @staticmethod
    def _compute_adx_last(h, l, c, period=14):
        n = len(h)
        if n < period + 2:
            return 0.0
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        pdm = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                       np.maximum(h[1:] - h[:-1], 0), 0.0)
        mdm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                       np.maximum(l[:-1] - l[1:], 0), 0.0)
        alpha = 1.0 / period
        atr_s = float(tr[:period].mean())
        pdm_s = float(pdm[:period].mean())
        mdm_s = float(mdm[:period].mean())
        for i in range(period, len(tr)):
            atr_s = atr_s * (1 - alpha) + float(tr[i]) * alpha
            pdm_s = pdm_s * (1 - alpha) + float(pdm[i]) * alpha
            mdm_s = mdm_s * (1 - alpha) + float(mdm[i]) * alpha
        if atr_s < 1e-15:
            return 0.0
        pdi = 100 * pdm_s / atr_s
        mdi = 100 * mdm_s / atr_s
        denom = pdi + mdi
        if denom < 1e-15:
            return 0.0
        return 100 * abs(pdi - mdi) / denom
