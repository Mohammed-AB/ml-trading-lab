"""Model F — Tokyo-to-London Range Expansion (TKY_LDN).

Computes the Asia session range (UTC hours 0-6) from M5 bars.
When London session (hour >= 7) breaks the Asia range with a
conviction candle (body > 50% of bar range), enter in the break direction.

SL: opposite Asia extreme.
TP: 2x risk (entry-to-SL distance).

Filters:
  - ADX(14) > min_adx (default 25) — trend confirmation
  - Skip Thursday (DOW 3) — historically weak
  - Skip USD_CAD — historically negative on this strategy
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .model_a import Direction, TriggerPhase, TriggerSignal
from .feature_engine import IndicatorSet


class ModelTkyLdnTrigger:
    """Asia range → London breakout trigger on M5 data."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._config = cfg
        self._asia_start_hour = int(cfg.get("asia_start_hour", 0))
        self._asia_end_hour = int(cfg.get("asia_end_hour", 6))
        self._london_start_hour = int(cfg.get("london_start_hour", 7))
        self._london_end_hour = int(cfg.get("london_end_hour", 12))
        self._min_adx = float(cfg.get("min_adx", 25))
        self._conviction_body_pct = float(cfg.get("conviction_body_pct", 0.50))
        self._tp_risk_mult = float(cfg.get("tp_risk_mult", 2.0))
        self._skip_pairs = set(cfg.get("skip_pairs", ["USD_CAD"]))
        self._skip_dow = set(cfg.get("skip_dow", [3]))  # Thursday
        self._min_range_atr_pct = float(cfg.get("min_range_atr_pct", 0.3))

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
        if current_dow in self._skip_dow:
            return no("dow_skipped")

        ny_start = int(self._config.get("ny_start_hour", 12)) if hasattr(self, "_config") else 12
        ny_end = int(self._config.get("ny_end_hour", 17)) if hasattr(self, "_config") else 17

        in_london = self._london_start_hour <= current_utc_hour <= self._london_end_hour
        in_ny = ny_start <= current_utc_hour <= ny_end

        if not in_london and not in_ny:
            return no(f"hour_{current_utc_hour}_outside_sessions")

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

        if "timestamp" in df_m5.columns:
            ts = pd.to_datetime(df_m5["timestamp"])
        elif isinstance(df_m5.index, pd.DatetimeIndex):
            ts = df_m5.index
        else:
            return no("no_timestamps")

        hours = ts.hour if hasattr(ts, 'hour') else ts.map(lambda x: x.hour)

        today_mask = ts >= (ts.iloc[-1].floor("D") if hasattr(ts, 'iloc') else ts[-1].floor("D"))

        if in_london:
            range_mask = today_mask & (hours >= self._asia_start_hour) & (hours <= self._asia_end_hour)
            session_name = "asia"
        else:
            range_mask = today_mask & (hours >= self._london_start_hour) & (hours <= self._london_end_hour)
            session_name = "london"

        if range_mask.sum() < 3:
            return no(f"no_{session_name}_bars")

        asia_high = float(h[range_mask].max())
        asia_low = float(l[range_mask].min())
        asia_range = asia_high - asia_low

        if asia_range < self._min_range_atr_pct * atr:
            return no("asia_range_too_small")

        last_bar = len(df_m5) - 1
        bar_close = float(c[last_bar])
        bar_open = float(o[last_bar])
        bar_high = float(h[last_bar])
        bar_low = float(l[last_bar])
        bar_range = bar_high - bar_low
        bar_body = abs(bar_close - bar_open)

        if bar_range < pip * 0.5:
            return no("flat_bar")

        conviction = bar_body / bar_range >= self._conviction_body_pct

        # ADX filter via IndicatorSet (compute from M5 if available)
        adx_val = self._compute_adx(df_m5)
        if adx_val < self._min_adx:
            return no(f"adx_{adx_val:.1f}_below_{self._min_adx}")

        if bar_close > asia_high and conviction:
            entry = bar_close
            sl = asia_low
            risk = entry - sl
            tp = entry + self._tp_risk_mult * risk
            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=Direction.LONG,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                risk_pips=round(risk / pip, 1),
                values={
                    "asia_high": asia_high, "asia_low": asia_low,
                    "asia_range_pips": round(asia_range / pip, 1),
                    "adx": round(adx_val, 1),
                    "conviction": conviction,
                },
            )

        if bar_close < asia_low and conviction:
            entry = bar_close
            sl = asia_high
            risk = sl - entry
            tp = entry - self._tp_risk_mult * risk
            return TriggerSignal(
                phase=TriggerPhase.VALID,
                direction=Direction.SHORT,
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                risk_pips=round(risk / pip, 1),
                values={
                    "asia_high": asia_high, "asia_low": asia_low,
                    "asia_range_pips": round(asia_range / pip, 1),
                    "adx": round(adx_val, 1),
                    "conviction": conviction,
                },
            )

        return no("no_breakout")

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 5:
            return 0.0
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)
        n = len(df)

        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]),
                                   np.abs(l[1:] - c[:-1])))

        plus_dm = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]),
                           np.maximum(h[1:] - h[:-1], 0), 0.0)
        minus_dm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]),
                            np.maximum(l[:-1] - l[1:], 0), 0.0)

        alpha = 1.0 / period
        atr_s = float(tr[:period].mean())
        pdm_s = float(plus_dm[:period].mean())
        mdm_s = float(minus_dm[:period].mean())

        for i in range(period, len(tr)):
            atr_s = atr_s * (1 - alpha) + float(tr[i]) * alpha
            pdm_s = pdm_s * (1 - alpha) + float(plus_dm[i]) * alpha
            mdm_s = mdm_s * (1 - alpha) + float(minus_dm[i]) * alpha

        if atr_s < 1e-15:
            return 0.0
        pdi = 100 * pdm_s / atr_s
        mdi = 100 * mdm_s / atr_s
        denom = pdi + mdi
        if denom < 1e-15:
            return 0.0
        dx = 100 * abs(pdi - mdi) / denom
        return dx
