"""Model E — Session VWAP reversion.

When price extends >= 2.5 M5 ATR above/below the session VWAP during
London or NY Overlap, fade it back to the VWAP.

Session definition (UTC):
    London:       07:00-12:00
    NY Overlap:   12:00-16:00
VWAP resets at session open. Uses cumulative (price*volume)/volume.

Entry: fade back toward VWAP when price is extended.
SL: beyond the extreme (ATR * 0.5)
TP: VWAP itself

Skips JPY pairs during low-vol hours (USD_JPY scalps poorly on VWAP).
"""
from typing import Optional

import pandas as pd

from .model_a import TriggerSignal, TriggerPhase, Direction
from .feature_engine import IndicatorSet


class ModelETrigger:
    """VWAP reversion scalp."""

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._extension_atr = float(cfg.get("extension_atr_mult", 2.5))
        self._min_session_bars = int(cfg.get("min_session_bars", 10))
        self._sl_atr_mult = float(cfg.get("sl_atr_mult", 0.5))
        self._active_hours = set(cfg.get("active_hours", list(range(7, 16))))

    def evaluate(
        self,
        df_m5: pd.DataFrame,
        ind_m5: IndicatorSet,
        pair: str,
        spread_pips: float,
        current_utc_hour: int,
        regime: str = "",
    ) -> TriggerSignal:
        if current_utc_hour not in self._active_hours:
            return TriggerSignal(
                phase=TriggerPhase.NO_BREAKOUT,
                values={"reason": f"hour_{current_utc_hour}_outside_active"})

        if df_m5 is None or len(df_m5) < self._min_session_bars:
            return TriggerSignal(
                phase=TriggerPhase.NO_BREAKOUT,
                values={"reason": "insufficient_data"})

        atr = (ind_m5.atr14 or 0) if ind_m5 else 0
        if atr <= 0:
            return TriggerSignal(
                phase=TriggerPhase.NO_BREAKOUT,
                values={"reason": "no_atr"})

        pip_size = 0.01 if "JPY" in pair else 0.0001

        # Build session VWAP — restart at 07:00 UTC (or earliest bar today)
        last_ts = pd.to_datetime(df_m5.index[-1]) if len(df_m5) > 0 else None
        if last_ts is None:
            return TriggerSignal(phase=TriggerPhase.NO_BREAKOUT,
                                 values={"reason": "no_timestamp"})

        # Pick session open: today's 07:00 UTC if we're past it, else yesterday
        try:
            sess_start = last_ts.floor("D") + pd.Timedelta(hours=7)
            if last_ts < sess_start:
                sess_start = sess_start - pd.Timedelta(days=1)
            session_df = df_m5[df_m5.index >= sess_start]
            if len(session_df) < self._min_session_bars:
                return TriggerSignal(
                    phase=TriggerPhase.NO_BREAKOUT,
                    values={"reason": "short_session"})
        except Exception:
            session_df = df_m5.tail(60)

        if "volume" in session_df.columns and session_df["volume"].sum() > 0:
            typical = (session_df["high"] + session_df["low"]
                       + session_df["close"]) / 3.0
            vwap = (typical * session_df["volume"]).sum() / session_df["volume"].sum()
        else:
            vwap = session_df["close"].mean()

        current = float(df_m5["close"].iloc[-1])
        deviation = current - vwap
        deviation_atr = deviation / atr if atr > 0 else 0
        deviation_pips = deviation / pip_size

        # Need > threshold * ATR to fade
        if abs(deviation_atr) < self._extension_atr:
            return TriggerSignal(
                phase=TriggerPhase.NO_BREAKOUT,
                values={
                    "reason": (
                        f"deviation_{deviation_atr:+.2f}atr "
                        f"below {self._extension_atr}"),
                    "vwap": round(vwap, 5),
                    "current": round(current, 5),
                })

        # Reversion setup
        # Note: regime is surfaced in `values["regime"]` so Strategy can see
        # whether this fire aligns with or fades the prevailing trend. We do
        # NOT hard-gate counter-trend fades here — Strategy decides.
        sl_dist = atr * self._sl_atr_mult
        if deviation > 0:
            # Price above VWAP — fade short toward VWAP
            entry = current
            sl = entry + sl_dist
            tp = float(vwap)
            direction = Direction.SHORT
            risk_pips = sl_dist / pip_size
        else:
            entry = current
            sl = entry - sl_dist
            tp = float(vwap)
            direction = Direction.LONG
            risk_pips = sl_dist / pip_size

        regime_l = (regime or "").lower()
        return TriggerSignal(
            phase=TriggerPhase.VALID,
            direction=direction,
            entry_price=entry,
            sl_price=sl,
            tp_price=tp,
            risk_pips=risk_pips,
            values={
                "reason": f"vwap_fade dev={deviation_atr:+.2f}atr ({deviation_pips:+.1f}p)",
                "vwap": round(vwap, 5),
                "deviation_atr": round(deviation_atr, 2),
                "regime": regime_l or "unknown",
                "sl_pips": round(risk_pips, 2),
            },
        )
