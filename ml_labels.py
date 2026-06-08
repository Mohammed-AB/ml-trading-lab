"""Forward TP/SL labels for ML (120 M1 bars, 15/10 pip targets, spread at entry)."""

from __future__ import annotations

import numpy as np

N_FUTURE = 120
TP_LONG_PIPS = 15.0
SL_LONG_PIPS = 10.0
TP_SHORT_PIPS = 15.0
SL_SHORT_PIPS = 10.0

# ATR-scaled horizons (price multiples on bar *i* ATR14)
ATR_SL_MULT = 1.0
ATR_TP_MULT = 1.5
ATR_TP2_MULT = 2.0


def compute_labels_python(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    spread_half: float,
    pip: float,
    n_future: int = N_FUTURE,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential first-touch labels (SL before TP when both possible same bar)."""
    n = len(close)
    long_l = np.zeros(n, dtype=np.int8)
    short_l = np.zeros(n, dtype=np.int8)
    nf = int(n_future)
    for i in range(n - nf - 1):
        ent = close[i] + spread_half
        tp = ent + float(TP_LONG_PIPS) * pip
        sl = ent - float(SL_LONG_PIPS) * pip
        lab = 0
        for k in range(1, nf + 1):
            j = i + k
            hit_sl = low[j] <= sl
            hit_tp = high[j] >= tp
            # Conservative: both touched same bar → treat as SL (no TP win).
            if hit_sl and hit_tp:
                break
            if hit_sl:
                break
            if hit_tp:
                lab = 1
                break
        long_l[i] = lab

        ent_s = close[i] - spread_half
        tp_s = ent_s - float(TP_SHORT_PIPS) * pip
        sl_s = ent_s + float(SL_SHORT_PIPS) * pip
        lab_s = 0
        for k in range(1, nf + 1):
            j = i + k
            hit_sl = high[j] >= sl_s
            hit_tp = low[j] <= tp_s
            if hit_sl and hit_tp:
                break
            if hit_sl:
                break
            if hit_tp:
                lab_s = 1
                break
        short_l[i] = lab_s
    return long_l, short_l


def _try_numba():
    try:
        from numba import njit

        @njit(cache=True)
        def _compute_labels_numba(
            high, low, close, spread_half, pip, n_future
        ):
            n = len(close)
            long_l = np.zeros(n, dtype=np.int8)
            short_l = np.zeros(n, dtype=np.int8)
            for i in range(n - n_future - 1):
                ent = close[i] + spread_half
                tp = ent + 15.0 * pip
                sl = ent - 10.0 * pip
                lab = 0
                for k in range(1, n_future + 1):
                    j = i + k
                    hit_sl = low[j] <= sl
                    hit_tp = high[j] >= tp
                    if hit_sl and hit_tp:
                        break
                    if hit_sl:
                        break
                    if hit_tp:
                        lab = 1
                        break
                long_l[i] = lab

                ent_s = close[i] - spread_half
                tp_s = ent_s - 15.0 * pip
                sl_s = ent_s + 10.0 * pip
                lab_s = 0
                for k in range(1, n_future + 1):
                    j = i + k
                    hit_sl = high[j] >= sl_s
                    hit_tp = low[j] <= tp_s
                    if hit_sl and hit_tp:
                        break
                    if hit_sl:
                        break
                    if hit_tp:
                        lab_s = 1
                        break
                short_l[i] = lab_s
            return long_l, short_l

        return _compute_labels_numba
    except Exception:
        return None


_NUMBA_FN = _try_numba()


def compute_labels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    spread_half: float,
    pip: float,
    n_future: int = N_FUTURE,
) -> tuple[np.ndarray, np.ndarray]:
    high = np.ascontiguousarray(high, dtype=np.float64)
    low = np.ascontiguousarray(low, dtype=np.float64)
    close = np.ascontiguousarray(close, dtype=np.float64)
    if _NUMBA_FN is not None:
        return _NUMBA_FN(high, low, close, float(spread_half), float(pip), int(n_future))
    return compute_labels_python(high, low, close, spread_half, pip, n_future)


def compute_labels_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    spread_half: float,
    pip: float,
    n_future: int = N_FUTURE,
    tp_mult: float = ATR_TP_MULT,
    sl_mult: float = ATR_SL_MULT,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as fixed-pip labels but SL/TP distances are ``sl_mult * atr[i]`` and ``tp_mult * atr[i]`` (price)."""
    high = np.ascontiguousarray(high, dtype=np.float64)
    low = np.ascontiguousarray(low, dtype=np.float64)
    close = np.ascontiguousarray(close, dtype=np.float64)
    atr = np.ascontiguousarray(atr, dtype=np.float64)
    n = len(close)
    long_l = np.zeros(n, dtype=np.int8)
    short_l = np.zeros(n, dtype=np.int8)
    nf = int(n_future)
    atr_min = max(pip * 0.05, 1e-12)
    for i in range(n - nf - 1):
        a = float(atr[i]) if np.isfinite(atr[i]) else atr_min
        a = max(a, atr_min)
        sl_px = sl_mult * a
        tp_px = tp_mult * a

        ent = close[i] + spread_half
        tp = ent + tp_px
        sl = ent - sl_px
        lab = 0
        for k in range(1, nf + 1):
            j = i + k
            hit_sl = low[j] <= sl
            hit_tp = high[j] >= tp
            if hit_sl and hit_tp:
                break
            if hit_sl:
                break
            if hit_tp:
                lab = 1
                break
        long_l[i] = lab

        ent_s = close[i] - spread_half
        tp_s = ent_s - tp_px
        sl_s = ent_s + sl_px
        lab_s = 0
        for k in range(1, nf + 1):
            j = i + k
            hit_sl = high[j] >= sl_s
            hit_tp = low[j] <= tp_s
            if hit_sl and hit_tp:
                break
            if hit_sl:
                break
            if hit_tp:
                lab_s = 1
                break
        short_l[i] = lab_s
    return long_l, short_l


def compute_labels_atr_tp2(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    spread_half: float,
    pip: float,
    n_future: int = N_FUTURE,
) -> tuple[np.ndarray, np.ndarray]:
    """ATR labels with wider TP (2.0 × ATR vs 1.5)."""
    return compute_labels_atr(
        high,
        low,
        close,
        atr,
        spread_half,
        pip,
        n_future=n_future,
        tp_mult=ATR_TP2_MULT,
        sl_mult=ATR_SL_MULT,
    )
