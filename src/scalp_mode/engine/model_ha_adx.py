"""Model G -- Heikin-Ashi + ADX trend entry (HA_ADX).

Enter when all three align on M5 data:
  1. HA candle is bullish (ha_close > ha_open) or bearish
  2. ADX > min_adx (trend strength)
  3. +DI > -DI (long) or -DI > +DI (short)

Only fires on the FIRST bar of alignment (transition).

Backtest OOS: PF 1.33, 158 trades, +626 pips, survives 2x spread.
Best hours: H07, H10-11, H18-19. Best day: Monday.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .model_a import Direction, TriggerPhase, TriggerSignal
from .feature_engine import IndicatorSet


class ModelHaAdxTrigger:
    """HA color + ADX + DI alignment trigger on M5."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._min_adx = float(cfg.get("min_adx", 25))
        self._tp_risk_mult = float(cfg.get("tp_risk_mult", 1.5))
        self._sl_atr_mult = float(cfg.get("sl_atr_mult", 1.5))
        self._active_hours = set(cfg.get("active_hours", list(range(7, 20))))
        self._skip_dow = set(cfg.get("skip_dow", [5, 6]))

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

        if current_utc_hour not in self._active_hours:
            return no(f"hour_{current_utc_hour}_inactive")
        if current_dow in self._skip_dow:
            return no("dow_skipped")
        if df_m5 is None or len(df_m5) < 30:
            return no("insufficient_data")

        atr = (ind_m5.atr14 or 0) if ind_m5 else 0
        if atr <= 0:
            return no("no_atr")

        pip = 0.01 if "JPY" in pair else 0.0001
        h = df_m5["high"].values
        l = df_m5["low"].values
        c = df_m5["close"].values
        o = df_m5["open"].values

        ha_c, ha_o = self._compute_ha(o, h, l, c)
        adx, pdi, mdi = self._compute_adx_di(h, l, c)

        n = len(df_m5)
        i = n - 1
        if i < 2:
            return no("too_few_bars")

        cur_bull = ha_c[i] > ha_o[i]
        cur_bear = ha_c[i] < ha_o[i]
        prev_bull = ha_c[i-1] > ha_o[i-1]
        prev_bear = ha_c[i-1] < ha_o[i-1]

        cur_adx = adx[i] if i < len(adx) else 0
        cur_pdi = pdi[i] if i < len(pdi) else 0
        cur_mdi = mdi[i] if i < len(mdi) else 0

        if cur_adx < self._min_adx:
            return no(f"adx_{cur_adx:.0f}_low")

        entry = float(c[i])
        risk = self._sl_atr_mult * atr

        if cur_bull and cur_pdi > cur_mdi and not (prev_bull and pdi[i-1] > mdi[i-1]):
            sl = entry - risk
            tp = entry + self._tp_risk_mult * risk
            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=Direction.LONG,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                risk_pips=round(risk / pip, 1),
                values={"adx": round(cur_adx, 1), "pdi": round(cur_pdi, 1), "mdi": round(cur_mdi, 1)},
            )

        if cur_bear and cur_mdi > cur_pdi and not (prev_bear and mdi[i-1] > pdi[i-1]):
            sl = entry + risk
            tp = entry - self._tp_risk_mult * risk
            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=Direction.SHORT,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                risk_pips=round(risk / pip, 1),
                values={"adx": round(cur_adx, 1), "pdi": round(cur_pdi, 1), "mdi": round(cur_mdi, 1)},
            )

        return no("no_transition")

    @staticmethod
    def _compute_ha(o, h, l, c):
        n = len(c)
        ha_c = (o + h + l + c) / 4.0
        ha_o = np.empty(n, dtype=np.float64)
        ha_o[0] = (o[0] + c[0]) / 2.0
        for i in range(1, n):
            ha_o[i] = (ha_o[i-1] + ha_c[i-1]) / 2.0
        return ha_c, ha_o

    @staticmethod
    def _compute_adx_di(h, l, c, period=14):
        n = len(h)
        adx = np.zeros(n, dtype=np.float64)
        pdi = np.zeros(n, dtype=np.float64)
        mdi = np.zeros(n, dtype=np.float64)
        if n < period + 2:
            return adx, pdi, mdi

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
            j = i + 1
            if atr_s > 1e-15:
                pdi[j] = 100 * pdm_s / atr_s
                mdi[j] = 100 * mdm_s / atr_s
                denom = pdi[j] + mdi[j]
                if denom > 1e-15:
                    adx[j] = 100 * abs(pdi[j] - mdi[j]) / denom

        return adx, pdi, mdi
