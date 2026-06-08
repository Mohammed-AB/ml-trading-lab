"""Research-sourced strategies (R1–R7) — same contract as backtest_strategies signal fns.

Each returns: indices, dirs (1 long / -1 short), entries, sls, tps  (all lists, price space)
or empty lists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _tr(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = len(c)
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr[1:] = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - pc[1:]), np.abs(l[1:] - pc[1:])),
    )
    return tr


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().values


def _atr(h, l, c, period: int) -> np.ndarray:
    tr = _tr(h, l, c)
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


def _supertrend(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int, mult: float
) -> tuple[np.ndarray, np.ndarray]:
    """ATR stop bands; direction 1=long / -1=short (close vs lower band in uptrend)."""
    n = len(c)
    atr = np.nan_to_num(_atr(h, l, c, period), nan=0.0)
    m = (h + l) * 0.5
    b_u = m + mult * atr
    b_d = m - mult * atr
    st = np.zeros(n)
    d = np.zeros(n, dtype=np.int8)
    st[0] = b_d[0]
    d[0] = 1
    for i in range(1, n):
        if c[i] > st[i - 1]:
            st[i] = max(b_d[i], st[i - 1])
            d[i] = 1
        else:
            st[i] = min(b_u[i], st[i - 1] if c[i] < b_u[i] else b_u[i])
            d[i] = -1
    return st, d


def _heikin_ashi(o, h, l, c) -> tuple[np.ndarray, ...]:
    n = len(c)
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.zeros(n)
    ha_h = np.zeros(n)
    ha_l = np.zeros(n)
    ha_o[0] = (o[0] + c[0]) * 0.5
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) * 0.5
    ha_h = np.maximum(h, np.maximum(ha_o, ha_c))
    ha_l = np.minimum(l, np.minimum(ha_o, ha_c))
    return ha_o, ha_h, ha_l, ha_c


# --- R1 TTM Squeeze (BB inside KC, trade expansion) ---------------------------------


def signals_r1_ttm_squeeze(df: pd.DataFrame, pair: str, pip: float, spread: float):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    n = len(c)
    atr10 = _atr(h, l, c, 10)
    ema20 = _ema(c, 20)
    std20 = pd.Series(c).rolling(20).std().values
    bb_u = ema20 + 2.0 * std20
    bb_d = ema20 - 2.0 * std20
    kc_u = ema20 + 1.5 * np.nan_to_num(atr10, nan=0.0)
    kc_d = ema20 - 1.5 * np.nan_to_num(atr10, nan=0.0)
    squeeze = (bb_u < kc_u) & (bb_d > kc_d)
    mom = c - (h + l) * 0.5
    mom2 = _ema(mom, 20)

    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -10
    for i in range(30, n - 2):
        if i - last < 10:
            continue
        if not (squeeze[i - 1] or squeeze[i - 2] or squeeze[i - 3]):
            continue
        if squeeze[i]:
            continue
        if mom2[i] > mom2[i - 1] and mom2[i] > 0 and c[i] > ema20[i]:
            e = c[i] + spread
            s = e - 8 * pip
            t = e + 12 * pip
            indices.append(i)
            dirs.append(1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
        elif mom2[i] < mom2[i - 1] and mom2[i] < 0 and c[i] < ema20[i]:
            e = c[i] - spread
            s = e + 8 * pip
            t = e - 12 * pip
            indices.append(i)
            dirs.append(-1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
    return indices, dirs, entries, sls, tps


# --- R2 Simplified "exhaustion fade" (proxy for DC / backlash) --------------------


def signals_r2_exhaustion_fade(df: pd.DataFrame, pair: str, pip: float, spread: float):
    """After strong 8-bar move, fade the next 3 bars' extreme."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(c)
    atr = df["atr14"].values if "atr14" in df.columns else _atr(h, l, c, 14)
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -5
    for i in range(20, n - 2):
        if i - last < 5:
            continue
        move = c[i - 1] - c[i - 8]
        th = 12 * max(pip, atr[i] * 0.5) if not np.isnan(atr[i]) else 12 * pip
        if move > th:
            e = c[i] - spread
            s = h[i] + 5 * pip
            t = e - 10 * pip
            if s - e < 2 * pip:
                continue
            indices.append(i)
            dirs.append(-1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
        elif move < -th:
            e = c[i] + spread
            s = l[i] - 5 * pip
            t = e + 10 * pip
            if e - s < 2 * pip:
                continue
            indices.append(i)
            dirs.append(1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
    return indices, dirs, entries, sls, tps


# --- R3 Dual Supertrend -----------------------------------------------------------


def signals_r3_dual_supertrend(df: pd.DataFrame, pair: str, pip: float, spread: float):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(c)
    st_f, d_fast = _supertrend(h, l, c, 7, 2.0)
    _, d_slow = _supertrend(h, l, c, 14, 3.0)
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -3
    for i in range(2, n - 1):
        if i - last < 3:
            continue
        hr = int(df["hour"].values[i]) if "hour" in df else 0
        if not (7 <= hr <= 16):
            continue
        if d_fast[i] == 1 and d_slow[i] == 1 and d_fast[i - 1] == -1:
            e = c[i] + spread
            s = st_f[i] - 3 * pip
            t = e + 10 * pip
            indices.append(i)
            dirs.append(1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
        elif d_fast[i] == -1 and d_slow[i] == -1 and d_fast[i - 1] == 1:
            e = c[i] - spread
            s = st_f[i] + 3 * pip
            t = e - 10 * pip
            indices.append(i)
            dirs.append(-1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
    return indices, dirs, entries, sls, tps


# --- R4 Heikin-Ashi + Stochastic --------------------------------------------------


def signals_r4_ha_stoch(df: pd.DataFrame, pair: str, pip: float, spread: float):
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    ha_o, ha_h, ha_l, ha_c = _heikin_ashi(o, h, l, c)
    em5 = _ema(ha_c, 5)
    n = len(c)
    ll = np.minimum(ha_l, np.minimum(ha_o, ha_c))
    hh = np.maximum(ha_h, np.maximum(ha_o, ha_c))
    lowest = pd.Series(ha_c).rolling(5).min().values
    highest = pd.Series(ha_c).rolling(5).max().values
    st_k = 100.0 * (ha_c - lowest) / np.maximum(highest - lowest, 1e-12)
    st_d = pd.Series(st_k).rolling(3).mean().values

    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -5
    for i in range(10, n - 1):
        if i - last < 5:
            continue
        hr = int(df["hour"].values[i]) if "hour" in df else 12
        if not (7 <= hr <= 20):
            continue
        b1 = ha_c[i] > em5[i] and ha_c[i - 1] > em5[i - 1] and (ha_c[i] - ha_o[i]) > abs(ha_c[i] - ha_l[i])
        xup = st_k[i] > st_d[i] and st_k[i - 1] <= st_d[i - 1] and st_k[i] < 50
        if b1 and xup:
            e = c[i] + spread
            s = l[i] - 5 * pip
            t = e + 8 * pip
            indices.append(i)
            dirs.append(1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
        b2 = ha_c[i] < em5[i] and ha_c[i - 1] < em5[i - 1] and (ha_o[i] - ha_c[i]) > abs(ha_h[i] - ha_c[i])
        xdn = st_k[i] < st_d[i] and st_k[i - 1] >= st_d[i - 1] and st_k[i] > 50
        if b2 and xdn:
            e = c[i] - spread
            s = h[i] + 5 * pip
            t = e - 8 * pip
            indices.append(i)
            dirs.append(-1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
    return indices, dirs, entries, sls, tps


# --- R5 Session opening range (Asian 00–07 UTC, trade London break) ----------------


def signals_r5_asian_orb(df: pd.DataFrame, pair: str, pip: float, spread: float):
    """Asian range 00--06 UTC; trade first London break 07--11 (M5 or M1 with hour)."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(c)
    day = ts.dt.date.values
    hour = ts.dt.hour.values
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last_i = -100
    unique_days = np.unique([str(x) for x in day])
    for ds in unique_days:
        idx = np.where(np.array([str(x) for x in day]) == ds)[0]
        if len(idx) < 20:
            continue
        as_m = idx[hour[idx] < 7]
        if len(as_m) < 5:
            continue
        ah = float(h[as_m].max())
        al = float(l[as_m].min())
        r_p = (ah - al) / pip
        if r_p > 60.0 or r_p < 4.0:
            continue
        brk = idx[(hour[idx] >= 7) & (hour[idx] <= 11)]
        for j in brk:
            if j < 1 or j <= last_i + 5:
                continue
            if c[j] > ah + 3.0 * pip and c[j - 1] <= ah + 3.0 * pip:
                e = c[j] + spread
                s = al - 5.0 * pip
                t = e + min((ah - al), 15.0 * pip)
                indices.append(j)
                dirs.append(1)
                entries.append(e)
                sls.append(s)
                tps.append(t)
                last_i = j
            elif c[j] < al - 3.0 * pip and c[j - 1] >= al - 3.0 * pip:
                e = c[j] - spread
                s = ah + 5.0 * pip
                t = e - min((ah - al), 15.0 * pip)
                indices.append(j)
                dirs.append(-1)
                entries.append(e)
                sls.append(s)
                tps.append(t)
                last_i = j
    return indices, dirs, entries, sls, tps


# --- R6 CCI + Keltner -------------------------------------------------------------


def signals_r6_cci_keltner(df: pd.DataFrame, pair: str, pip: float, spread: float):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(c)
    ema = _ema(c, 20)
    atr = _atr(h, l, c, 10)
    kc_u = ema + 1.5 * np.nan_to_num(atr, nan=0.0)
    kc_d = ema - 1.5 * np.nan_to_num(atr, nan=0.0)
    tp_ = (h + l + c) / 3.0
    sma_tp = pd.Series(tp_).rolling(14).mean().values
    md = (pd.Series(tp_) - sma_tp).abs().rolling(14).mean().values
    cci = (tp_ - sma_tp) / (0.015 * np.maximum(md, 1e-12))

    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -3
    for i in range(5, n - 1):
        if i - last < 3:
            continue
        lo100 = np.min(cci[max(0, i - 5) : i]) < -100
        hi100 = np.max(cci[max(0, i - 5) : i]) > 100
        if c[i] > ema[i] and cci[i - 1] < 0 and cci[i] >= 0 and lo100:
            e = c[i] + spread
            s = kc_d[i] - 2 * pip
            t = kc_u[i] + 2 * pip
            indices.append(i)
            dirs.append(1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
        if c[i] < ema[i] and cci[i - 1] > 0 and cci[i] <= 0 and hi100:
            e = c[i] - spread
            s = kc_u[i] + 2 * pip
            t = kc_d[i] - 2 * pip
            indices.append(i)
            dirs.append(-1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
    return indices, dirs, entries, sls, tps


# --- R7 Linear regression channel reversion ----------------------------------------


def signals_r7_linreg_channel(df: pd.DataFrame, pair: str, pip: float, spread: float):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    n = len(c)
    win = 50
    indices, dirs, entries, sls, tps = [], [], [], [], []
    last = -5
    x = np.arange(win, dtype=np.float64)
    for i in range(win, n - 1):
        if i - last < 5:
            continue
        y = c[i - win : i]
        if np.any(np.isnan(y)):
            continue
        xm = x.mean()
        ym = y.mean()
        b = ((x - xm) * (y - ym)).sum() / max(((x - xm) ** 2).sum(), 1e-12)
        a = ym - b * xm
        pred = a + b * (win - 1)
        resid = y - (a + b * x)
        std = max(np.std(resid), 1e-12)
        up = pred + 2 * std
        lo = pred - 2 * std
        slope = b
        if abs(slope) < 1e-7 * c[i]:
            continue
        if slope > 0 and l[i] <= lo + 2 * pip and c[i] > lo:
            e = c[i] + spread
            s = l[i] - 4 * pip
            t = pred + 2 * pip
            indices.append(i)
            dirs.append(1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
        elif slope < 0 and h[i] >= up - 2 * pip and c[i] < up:
            e = c[i] - spread
            s = h[i] + 4 * pip
            t = pred - 2 * pip
            indices.append(i)
            dirs.append(-1)
            entries.append(e)
            sls.append(s)
            tps.append(t)
            last = i
    return indices, dirs, entries, sls, tps


RESEARCH_STRATEGIES = [
    ("R1_TTM", signals_r1_ttm_squeeze, 20),
    ("R2_EXH", signals_r2_exhaustion_fade, 20),
    ("R3_DST", signals_r3_dual_supertrend, 20),
    ("R4_HA", signals_r4_ha_stoch, 15),
    ("R5_ORB", signals_r5_asian_orb, 30),
    ("R6_CCI", signals_r6_cci_keltner, 20),
    ("R7_LRC", signals_r7_linreg_channel, 30),
]
