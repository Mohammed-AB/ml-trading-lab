"""PDF-sourced M5 scalping strategies (R23–R37). Same contract as research.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _hour_ok(h: int) -> bool:
    return 7 <= h <= 20


def _prev_day_hl(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row previous calendar day high / low (UTC)."""
    ts = pd.to_datetime(df["timestamp"], utc=True)
    day = ts.dt.floor("D")
    daily_hi = df.groupby(day, sort=False)["high"].max()
    daily_lo = df.groupby(day, sort=False)["low"].min()
    pdh = day.map(daily_hi.shift(1)).astype(float).values
    pdl = day.map(daily_lo.shift(1)).astype(float).values
    return pdh, pdl


# --- R23 Stochastic divergence (simplified hidden div) ---------------------------


def signals_r23_stoch_div(df: pd.DataFrame, pair: str, pip: float, spread: float):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    sk = df["stoch_k"].values
    sd = df["stoch_d"].values
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(25, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        lo_p = np.min(l[i - 20 : i])
        hi_p = np.max(h[i - 20 : i])
        lo_sk = np.min(sk[i - 20 : i])
        hi_sk = np.max(sk[i - 20 : i])
        hl_price = l[i - 1] > lo_p + 1e-12
        ll_stoch = sk[i - 1] < lo_sk - 1e-6
        cross_up = sk[i - 2] < sd[i - 2] and sk[i - 1] > sd[i - 1]
        if sk[i - 1] < 20 and cross_up and hl_price and ll_stoch:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 8 * pip)
            tps.append(e + 12 * pip)
            last = i
            continue
        lh_price = h[i - 1] < hi_p - 1e-12
        hh_stoch = sk[i - 1] > hi_sk + 1e-6
        cross_dn = sk[i - 2] > sd[i - 2] and sk[i - 1] < sd[i - 1]
        if sk[i - 1] > 80 and cross_dn and lh_price and hh_stoch:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 8 * pip)
            tps.append(e - 12 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R24 ATR expansion ---------------------------------------------------------


def signals_r24_atr_exp(df: pd.DataFrame, pair: str, pip: float, spread: float):
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    atr = df["atr14"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(15, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        avg_atr = np.nanmean(atr[i - 10 : i])
        if not np.isfinite(avg_atr) or avg_atr <= 0:
            continue
        if atr[i] <= 1.5 * avg_atr:
            continue
        rng = h[i] - l[i]
        if rng <= 0:
            continue
        body_top = max(o[i], c[i])
        body_bot = min(o[i], c[i])
        bull_strong = c[i] > o[i] and (h[i] - body_top) < 0.25 * rng
        bear_strong = c[i] < o[i] and (body_bot - l[i]) < 0.25 * rng
        sl_dist = min(float(atr[i]), 15 * pip)
        tp_dist = min(sl_dist * 1.5, 20 * pip)
        if bull_strong:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - sl_dist)
            tps.append(e + tp_dist)
            last = i
        elif bear_strong:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + sl_dist)
            tps.append(e - tp_dist)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R25 Three-bar momentum ----------------------------------------------------


def signals_r25_3bar_mom(df: pd.DataFrame, pair: str, pip: float, spread: float):
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(3, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        r2 = h[i - 2] - l[i - 2]
        r1 = h[i - 1] - l[i - 1]
        r0 = h[i] - l[i]
        if r2 <= 0 or r1 <= 0 or r0 <= 0:
            continue
        b1 = c[i - 2] > o[i - 2] and (c[i - 2] - o[i - 2]) > 0.5 * r2
        b2 = c[i - 1] > o[i - 1] and c[i - 1] > c[i - 2]
        b3 = c[i] > o[i] and c[i] > c[i - 1]
        s1 = c[i - 2] < o[i - 2] and (o[i - 2] - c[i - 2]) > 0.5 * r2
        s2 = c[i - 1] < o[i - 1] and c[i - 1] < c[i - 2]
        s3 = c[i] < o[i] and c[i] < c[i - 1]
        if b1 and b2 and b3:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 12 * pip)
            tps.append(e + 10 * pip)
            last = i
        elif s1 and s2 and s3:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 12 * pip)
            tps.append(e - 10 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R26 RSI mean reversion (trend + cross back from 30/70) --------------------


def signals_r26_rsi_mr(df: pd.DataFrame, pair: str, pip: float, spread: float):
    c = df["close"].values
    rsi = df["rsi14"].values
    ema40 = df["ema40"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(2, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        if c[i] > ema40[i] and rsi[i - 1] < 30 and rsi[i] >= 30:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 10 * pip)
            tps.append(e + 10 * pip)
            last = i
        elif c[i] < ema40[i] and rsi[i - 1] > 70 and rsi[i] <= 70:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 10 * pip)
            tps.append(e - 10 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R27 MACD signal line cross ------------------------------------------------


def signals_r27_macd_sig(df: pd.DataFrame, pair: str, pip: float, spread: float):
    c = df["close"].values
    m = df["macd"].values
    s = df["macd_signal"].values
    ema40 = df["ema40"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(2, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        if c[i] > ema40[i] and m[i - 1] < s[i - 1] and m[i] >= s[i] and m[i] < 0:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 10 * pip)
            tps.append(e + 15 * pip)
            last = i
        elif c[i] < ema40[i] and m[i - 1] > s[i - 1] and m[i] <= s[i] and m[i] > 0:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 10 * pip)
            tps.append(e - 15 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R28 Liquidity sweep -------------------------------------------------------


def signals_r28_liq_swp(df: pd.DataFrame, pair: str, pip: float, spread: float):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(22, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        rh = float(np.max(h[i - 20 : i]))
        rl = float(np.min(l[i - 20 : i]))
        if l[i] < rl and c[i] > rl:
            e = c[i] + spread
            sl_p = max(l[i] - 2 * pip, e - 12 * pip)
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(sl_p)
            tps.append(e + 15 * pip)
            last = i
        elif h[i] > rh and c[i] < rh:
            e = c[i] - spread
            sl_p = min(h[i] + 2 * pip, e + 12 * pip)
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(sl_p)
            tps.append(e - 15 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R29 Round number fade -----------------------------------------------------


def signals_r29_rnd_num(df: pd.DataFrame, pair: str, pip: float, spread: float):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(25, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        pips_val = c[i] / pip
        rnd = round(pips_val / 50.0) * 50.0 * pip
        touched = False
        for j in range(i - 20, i):
            if j < 0:
                continue
            if l[j] <= rnd <= h[j]:
                touched = True
                break
        if touched:
            continue
        if l[i] <= rnd <= h[i] and c[i] > rnd and c[i] > l[i] + 0.1 * pip:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 8 * pip)
            tps.append(e + 12 * pip)
            last = i
        elif l[i] <= rnd <= h[i] and c[i] < rnd and c[i] < h[i] - 0.1 * pip:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 8 * pip)
            tps.append(e - 12 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R30 CHOCH simplified ------------------------------------------------------


def signals_r30_choch(df: pd.DataFrame, pair: str, pip: float, spread: float):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    ema40 = df["ema40"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    w = 10
    for i in range(w + 2, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        seg_h = h[i - w : i]
        seg_l = l[i - w : i]
        li = int(np.argmin(seg_l))
        hi = int(np.argmax(seg_h))
        if 0 < li < w - 2:
            swing_hi = float(np.max(seg_h[:li]))
            if c[i - 1] <= swing_hi and c[i] > swing_hi and c[i] > ema40[i]:
                e = c[i] + spread
                out_i.append(i)
                out_d.append(1)
                ent.append(e)
                sls.append(e - 10 * pip)
                tps.append(e + 15 * pip)
                last = i
                continue
        if 0 < hi < w - 2:
            swing_lo = float(np.min(seg_l[:hi]))
            if c[i - 1] >= swing_lo and c[i] < swing_lo and c[i] < ema40[i]:
                e = c[i] - spread
                out_i.append(i)
                out_d.append(-1)
                ent.append(e)
                sls.append(e + 10 * pip)
                tps.append(e - 15 * pip)
                last = i
    return out_i, out_d, ent, sls, tps


# --- R31 Previous day high/low fade --------------------------------------------


def signals_r31_pd_fade(df: pd.DataFrame, pair: str, pip: float, spread: float):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    pdh, pdl = _prev_day_hl(df)
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(2, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        if not np.isfinite(pdh[i]) or not np.isfinite(pdl[i]):
            continue
        if l[i] < pdl[i] and c[i] > pdl[i]:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 12 * pip)
            tps.append(e + 20 * pip)
            last = i
        elif h[i] > pdh[i] and c[i] < pdh[i]:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 12 * pip)
            tps.append(e - 20 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R32 ATR band breakout -----------------------------------------------------


def signals_r32_atr_ts_bo(df: pd.DataFrame, pair: str, pip: float, spread: float):
    c = df["close"].values
    ema20 = df["ema20"].values
    atr = df["atr14"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(2, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        upb = ema20[i - 1] + 2 * atr[i - 1]
        lob = ema20[i - 1] - 2 * atr[i - 1]
        if c[i - 1] <= upb and c[i] > upb:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 12 * pip)
            tps.append(e + 15 * pip)
            last = i
        elif c[i - 1] >= lob and c[i] < lob:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 12 * pip)
            tps.append(e - 15 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R33 Stochastic pop --------------------------------------------------------


def signals_r33_stoch_pop(df: pd.DataFrame, pair: str, pip: float, spread: float):
    c = df["close"].values
    sk = df["stoch_k"].values
    ema40 = df["ema40"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(2, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        if c[i] > ema40[i] and sk[i - 1] < 20 and sk[i] >= 20:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 10 * pip)
            tps.append(e + 12 * pip)
            last = i
        elif c[i] < ema40[i] and sk[i - 1] > 80 and sk[i] <= 80:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 10 * pip)
            tps.append(e - 12 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R34 Inside bar failure ----------------------------------------------------


def signals_r34_ib_fail(df: pd.DataFrame, pair: str, pip: float, spread: float):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(3, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        inside = h[i - 1] < h[i - 2] and l[i - 1] > l[i - 2]
        if not inside:
            continue
        ibh, ibl = h[i - 1], l[i - 1]
        if l[i] < ibl and c[i] > ibh:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 12 * pip)
            tps.append(e + 18 * pip)
            last = i
        elif h[i] > ibh and c[i] < ibl:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 12 * pip)
            tps.append(e - 18 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R35 EMA 9/20 cross pullback -----------------------------------------------


def signals_r35_ema_x_pb(df: pd.DataFrame, pair: str, pip: float, spread: float):
    e9 = df["ema9"].values
    e20 = df["ema20"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    trend_dir = 0
    cross_i = -999
    for i in range(2, n - 1):
        if not _hour_ok(int(hour[i])):
            continue
        if e9[i - 1] <= e20[i - 1] and e9[i] > e20[i]:
            trend_dir = 1
            cross_i = i
        elif e9[i - 1] >= e20[i - 1] and e9[i] < e20[i]:
            trend_dir = -1
            cross_i = i
        if i - last < 5:
            continue
        if trend_dir == 1 and i > cross_i and l[i] <= e20[i] and c[i] > e20[i]:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 10 * pip)
            tps.append(e + 15 * pip)
            last = i
            trend_dir = 0
        elif trend_dir == -1 and i > cross_i and h[i] >= e20[i] and c[i] < e20[i]:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 10 * pip)
            tps.append(e - 15 * pip)
            last = i
            trend_dir = 0
    return out_i, out_d, ent, sls, tps


# --- R36 RSI 50 continuation (simpler than R17) ------------------------------


def signals_r36_rsi_50c(df: pd.DataFrame, pair: str, pip: float, spread: float):
    c = df["close"].values
    rsi = df["rsi14"].values
    ema40 = df["ema40"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(2, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        if c[i] > ema40[i] and rsi[i - 1] < 50 and rsi[i] >= 50:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 10 * pip)
            tps.append(e + 12 * pip)
            last = i
        elif c[i] < ema40[i] and rsi[i - 1] > 50 and rsi[i] <= 50:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 10 * pip)
            tps.append(e - 12 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


# --- R37 EMA 9/20 squeeze expansion --------------------------------------------


def signals_r37_ema_rib_sqz(df: pd.DataFrame, pair: str, pip: float, spread: float):
    e9 = df["ema9"].values
    e20 = df["ema20"].values
    c = df["close"].values
    o = df["open"].values
    hour = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    n = len(c)
    out_i, out_d, ent, sls, tps = [], [], [], [], []
    last = -5
    for i in range(2, n - 1):
        if i - last < 5 or not _hour_ok(int(hour[i])):
            continue
        prev_d = abs(e9[i - 1] - e20[i - 1]) / pip
        curr_d = abs(e9[i] - e20[i]) / pip
        bull = c[i] > o[i]
        bear = c[i] < o[i]
        if prev_d < 1.0 and curr_d > 1.5 and e9[i] > e20[i] and bull:
            e = c[i] + spread
            out_i.append(i)
            out_d.append(1)
            ent.append(e)
            sls.append(e - 10 * pip)
            tps.append(e + 15 * pip)
            last = i
        elif prev_d < 1.0 and curr_d > 1.5 and e9[i] < e20[i] and bear:
            e = c[i] - spread
            out_i.append(i)
            out_d.append(-1)
            ent.append(e)
            sls.append(e + 10 * pip)
            tps.append(e - 15 * pip)
            last = i
    return out_i, out_d, ent, sls, tps


RESEARCH_PDF_STRATEGIES = [
    ("R23_STDV", signals_r23_stoch_div, 20),
    ("R24_ATRX", signals_r24_atr_exp, 20),
    ("R25_3BAR", signals_r25_3bar_mom, 20),
    ("R26_RSIM", signals_r26_rsi_mr, 20),
    ("R27_MACS", signals_r27_macd_sig, 20),
    ("R28_LIQ", signals_r28_liq_swp, 20),
    ("R29_RND", signals_r29_rnd_num, 20),
    ("R30_CHOC", signals_r30_choch, 20),
    ("R31_PDF", signals_r31_pd_fade, 20),
    ("R32_ATRB", signals_r32_atr_ts_bo, 20),
    ("R33_STPO", signals_r33_stoch_pop, 20),
    ("R34_IBF", signals_r34_ib_fail, 20),
    ("R35_EXPB", signals_r35_ema_x_pb, 20),
    ("R36_R50", signals_r36_rsi_50c, 20),
    ("R37_SQZ", signals_r37_ema_rib_sqz, 20),
]
