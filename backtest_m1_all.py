"""Full M1 backtest — all 13 strategies on native 1-minute data.

Uses numpy vectorization for signal detection to handle 8M+ bars per pair.
Simulate_trade remains a loop but is called far fewer times with pre-filtering.

Usage:
    python3 backtest_m1_all.py
    python3 backtest_m1_all.py --years 5
"""

import argparse
import json
import time as time_mod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path.home() / "Downloads" / "FX-1-Minute-Data-master" / "forex_data" / "1min"
OUTPUT_DIR = Path("data/brain/lessons")

PAIR_MAP = {
    "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY",
    "USDCHF": "USD_CHF", "USDCAD": "USD_CAD", "AUDUSD": "AUD_USD",
    "NZDUSD": "NZD_USD", "EURGBP": "EUR_GBP",
}

SPREAD_PIPS = {
    "EUR_USD": 0.4, "GBP_USD": 0.6, "USD_JPY": 0.5, "USD_CHF": 0.8,
    "USD_CAD": 0.8, "AUD_USD": 0.5, "NZD_USD": 0.7, "EUR_GBP": 0.6,
}


def load_pair(filename, pair, years=0):
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
    if years > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=years * 365)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


def add_indicators(df, pair):
    """Vectorized indicator computation for M1 data."""
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    n = len(df)
    pip = 0.01 if "JPY" in pair else 0.0001

    # EMAs
    df["ema9"] = pd.Series(c).ewm(span=9, adjust=False).mean().values
    df["ema20"] = pd.Series(c).ewm(span=20, adjust=False).mean().values
    df["ema40"] = pd.Series(c).ewm(span=40, adjust=False).mean().values

    # ATR
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    df["atr14"] = pd.Series(tr).rolling(14).mean().values

    # Bollinger
    sma20 = pd.Series(c).rolling(20).mean().values
    std20 = pd.Series(c).rolling(20).std().values
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"] = sma20
    bw = pd.Series(std20)
    bw_min = bw.rolling(100, min_periods=20).min()
    bw_max = bw.rolling(100, min_periods=20).max()
    denom = bw_max - bw_min
    df["bb_squeeze"] = np.where(denom > 0, (bw - bw_min) / denom, 0.5)

    # RSI
    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).values

    # Bar properties
    df["bar_range"] = h - l
    df["body"] = np.abs(c - o)
    df["is_bull"] = c > o
    df["is_bear"] = c < o

    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    df["pip"] = pip

    # EMA slope (over 20 bars for M1)
    ema20v = df["ema20"].values
    slope = np.zeros(n)
    slope[20:] = ema20v[20:] - ema20v[:-20]
    df["ema20_slope"] = slope

    return df


# ---------------------------------------------------------------------------
#  TRADE SIMULATOR — optimised with early exit
# ---------------------------------------------------------------------------

def sim_trades(h, l, c, indices, dirs, entries, sls, tps, max_bars, pip):
    """Simulate trades. dirs: 1=long, -1=short."""
    n = len(h)
    results = []
    for k in range(len(indices)):
        idx = indices[k]; d = dirs[k]
        entry = entries[k]; sl = sls[k]; tp = tps[k]
        end = min(idx + max_bars, n)
        ex = "time"; pnl = 0.0; bars = end - idx

        if d == 1:
            for j in range(idx + 1, end):
                if l[j] <= sl:
                    ex = "sl"; pnl = sl - entry; bars = j - idx; break
                if h[j] >= tp:
                    ex = "tp"; pnl = tp - entry; bars = j - idx; break
            else:
                pnl = c[min(end-1, n-1)] - entry
        else:
            for j in range(idx + 1, end):
                if h[j] >= sl:
                    ex = "sl"; pnl = entry - sl; bars = j - idx; break
                if l[j] <= tp:
                    ex = "tp"; pnl = entry - tp; bars = j - idx; break
            else:
                pnl = entry - c[min(end-1, n-1)]

        results.append((round(pnl / pip, 2), ex, bars))
    return results


def build_trades(name, indices, dirs, entries, sls, tps, results, hours, ts, pair, pip):
    trades = []
    for k in range(len(indices)):
        pnl_p, ex, bars = results[k]
        trades.append({
            "strategy": name, "pair": pair,
            "direction": "long" if dirs[k] == 1 else "short",
            "hour": int(hours[indices[k]]),
            "pnl_pips": pnl_p, "exit_reason": ex, "bars_held": bars,
            "sl_pips": round(abs(entries[k] - sls[k]) / pip, 1),
            "tp_pips": round(abs(entries[k] - tps[k]) / pip, 1),
            "timestamp": str(ts[indices[k]]),
        })
    return trades


# ---------------------------------------------------------------------------
#  STRATEGIES — vectorized signal masks + minimal loop for pattern validation
# ---------------------------------------------------------------------------

def strat_s1_dd(df, pair, pip, spread):
    """S1: Double Doji Break at 20 EMA (M1)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values; slope = df["ema20_slope"].values
    ema20 = df["ema20"].values; br = df["bar_range"].values
    n = len(c)

    # Vectorized pre-filter
    valid = ~np.isnan(atr) & (atr > 0) & (np.arange(n) > 30)
    near_ema = np.abs(c - ema20) < atr * 1.0
    trending = np.abs(slope) > atr * 0.03
    doji = br < atr * 0.4
    doji_prev = np.roll(doji, 1); doji_prev[0] = False
    mask = valid & near_ema & trending & doji & doji_prev

    candidates = np.where(mask)[0]

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -30
    for i in candidates:
        if i - last < 30:
            continue
        if abs(h[i] - h[i-1]) > 2 * pip and abs(l[i] - l[i-1]) > 2 * pip:
            continue

        bull = slope[i] > 0
        if bull:
            e = max(h[i], h[i-1]) + spread
            s = min(l[i], l[i-1]) - pip
            t = e + 10 * pip
        else:
            e = min(l[i], l[i-1]) - spread
            s = max(h[i], h[i-1]) + pip
            t = e - 10 * pip

        sl_d = abs(e - s)
        if sl_d > 10 * pip or sl_d < 0.5 * pip:
            continue

        idx.append(i); dirs.append(1 if bull else -1)
        ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s2_fb(df, pair, pip, spread):
    """S2: First Break after spike (M1)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values; br = df["bar_range"].values
    n = len(c)

    # 3-bar move in pips
    move3 = np.zeros(n)
    move3[3:] = (c[3:] - c[:-3]) / pip

    valid = ~np.isnan(atr) & (atr > 0) & (np.arange(n) > 5)
    big_move = np.abs(move3) > 12
    small_bar = br < atr * 0.5
    mask = valid & big_move & small_bar & (np.arange(n) < n - 60)

    candidates = np.where(mask)[0]

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -30
    for i in candidates:
        if i - last < 30:
            continue
        # Check all 3 prior bars trend same direction
        if move3[i] > 0:
            if not all(c[i-k] > o[i-k] for k in range(3)):
                continue
            e = h[i] + spread; s = l[i] - pip; t = e + 10*pip; d = 1
        else:
            if not all(c[i-k] < o[i-k] for k in range(3)):
                continue
            e = l[i] - spread; s = h[i] + pip; t = e - 10*pip; d = -1

        sl_d = abs(e - s)
        if sl_d > 10*pip or sl_d < 0.5*pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s3_sb(df, pair, pip, spread):
    """S3: Second Break / M-W at EMA (M1)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    ema20 = df["ema20"].values; atr = df["atr14"].values
    slope = df["ema20_slope"].values
    n = len(c)

    valid = ~np.isnan(atr) & (atr > 0) & (np.arange(n) > 50)
    trending = np.abs(slope) > atr * 0.03
    near_ema = np.abs(c - ema20) < atr * 1.5
    mask = valid & trending & near_ema & (np.arange(n) < n - 60)

    candidates = np.where(mask)[0]

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -50
    w = 30
    for i in candidates:
        if i - last < 50:
            continue
        bull = slope[i] > 0

        if bull:
            rl = l[i-w:i+1]
            if len(rl) < w:
                continue
            s1, s2 = np.argsort(rl)[:2]
            if abs(s1-s2) < 5:
                continue
            lv1, lv2 = rl[s1], rl[s2]
            if abs(lv1-lv2) > 2*pip:
                continue
            mid = h[i-w+min(s1,s2):i-w+max(s1,s2)+1]
            if len(mid) == 0 or np.max(mid) - max(lv1,lv2) < 3*pip:
                continue
            e = c[i]+spread; s = min(lv1,lv2)-pip; t = e+10*pip; d = 1
        else:
            rh = h[i-w:i+1]
            if len(rh) < w:
                continue
            s1, s2 = np.argsort(-rh)[:2]
            if abs(s1-s2) < 5:
                continue
            hv1, hv2 = rh[s1], rh[s2]
            if abs(hv1-hv2) > 2*pip:
                continue
            mid = l[i-w+min(s1,s2):i-w+max(s1,s2)+1]
            if len(mid) == 0 or min(hv1,hv2) - np.min(mid) < 3*pip:
                continue
            e = c[i]-spread; s = max(hv1,hv2)+pip; t = e-10*pip; d = -1

        sl_d = abs(e - s)
        if sl_d > 12*pip or sl_d < 1*pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s4_ema(df, pair, pip, spread):
    """S4: EMA Bounce (M1) — with strict cooldown."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    ema9 = df["ema9"].values; ema20 = df["ema20"].values
    n = len(c)

    # Vectorized: ema9 > ema20 for last 15 bars (bull)
    bull_streak = np.ones(n, dtype=bool)
    bear_streak = np.ones(n, dtype=bool)
    for k in range(15):
        shifted_diff = np.roll(ema9 - ema20, k)
        shifted_diff[:k] = 0
        bull_streak &= shifted_diff > 0
        bear_streak &= shifted_diff < 0

    valid = ~np.isnan(atr) & (atr > 0) & (np.arange(n) > 20)
    # Touch ema9
    bull_touch = l <= ema9 + pip
    bear_touch = h >= ema9 - pip
    # Confirm bar
    bull_confirm = (c > o) & ((c - o) > 0.4 * (h - l))
    bear_confirm = (c < o) & ((o - c) > 0.4 * (h - l))

    bull_mask = valid & bull_streak & bull_touch & bull_confirm & (c > ema20)
    bear_mask = valid & bear_streak & bear_touch & bear_confirm & (c < ema20)
    mask = (bull_mask | bear_mask) & (np.arange(n) < n - 60)

    candidates = np.where(mask)[0]

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -30
    for i in candidates:
        if i - last < 30:
            continue
        if bull_mask[i]:
            e = h[i]+spread; s = ema20[i]-atr[i]; t = e+2*abs(e-s); d = 1
        else:
            e = l[i]-spread; s = ema20[i]+atr[i]; t = e-2*abs(e-s); d = -1

        sl_d = abs(e - s)
        if sl_d > 15*pip or sl_d < 1*pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s5_bb(df, pair, pip, spread):
    """S5: Block Break / Bollinger Squeeze (M1)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values; bb_sq = df["bb_squeeze"].values
    slope = df["ema20_slope"].values
    n = len(c)

    valid = ~np.isnan(atr) & (atr > 0) & ~np.isnan(bb_sq) & (bb_sq < 0.25)
    mask = valid & (np.arange(n) > 120) & (np.arange(n) < n - 60)
    candidates = np.where(mask)[0]

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -50
    for i in candidates:
        if i - last < 50:
            continue
        blk_h = h[i-10:i+1]; blk_l = l[i-10:i+1]
        bt = np.max(blk_h); bb = np.min(blk_l)
        bh = bt - bb
        if bh > atr[i]*0.6 or bh < 2*pip:
            continue
        tt = np.sum(np.abs(blk_h - bt) < 1.5*pip)
        bt_t = np.sum(np.abs(blk_l - bb) < 1.5*pip)
        if tt < 3 or bt_t < 3:
            continue

        bull = slope[i] > 0
        if bull:
            e = bt+spread; s = bb-pip; t = e+max(10*pip, bh); d = 1
        else:
            e = bb-spread; s = bt+pip; t = e-max(10*pip, bh); d = -1

        sl_d = abs(e - s)
        if sl_d > 15*pip or sl_d < 1*pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s6_rb(df, pair, pip, spread):
    """S6: Range Break (M1)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values
    n = len(c)

    valid = ~np.isnan(atr) & (atr > 0) & (np.arange(n) > 100) & (np.arange(n) < n - 120)
    candidates = np.where(valid)[0][::60]  # sample every 60 bars to speed up

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -100
    for i in candidates:
        if i - last < 100:
            continue
        seg_h = h[i-80:i-5]; seg_l = l[i-80:i-5]
        rt = np.max(seg_h); rb = np.min(seg_l)
        rh = rt - rb
        if rh < 8*pip or rh > 60*pip:
            continue
        tt = np.sum(np.abs(seg_h - rt) < 2*pip)
        bt = np.sum(np.abs(seg_l - rb) < 2*pip)
        if tt < 4 or bt < 4:
            continue

        sq_h = h[i-5:i+1]; sq_l = l[i-5:i+1]
        if np.max(sq_h) - np.min(sq_l) > rh*0.4:
            continue

        if c[i] > rt:
            e = c[i]+spread; s = np.min(sq_l)-pip; t = rt+rh; d = 1
        elif c[i] < rb:
            e = c[i]-spread; s = np.max(sq_h)+pip; t = rb-rh; d = -1
        else:
            continue

        sl_d = abs(e - s)
        if sl_d > 20*pip or sl_d < 2*pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s9_fbr(df, pair, pip, spread):
    """S9: Failed Breakout Reversal (M1)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    n = len(c)

    # Rolling max/min for swing detection
    roll_h = pd.Series(h).rolling(60, min_periods=20).max().values
    roll_l = pd.Series(l).rolling(60, min_periods=20).min().values

    # Shift to avoid self-reference
    sw_hi = np.roll(roll_h, 5); sw_hi[:5] = np.nan
    sw_lo = np.roll(roll_l, 5); sw_lo[:5] = np.nan

    valid = ~np.isnan(atr) & (atr > 0) & ~np.isnan(sw_hi) & (np.arange(n) > 65)
    # Previous bar broke level
    prev_h = np.roll(h, 1); prev_l = np.roll(l, 1)
    broke_hi = (prev_h > sw_hi) & (prev_h - sw_hi < 5*pip)
    broke_lo = (prev_l < sw_lo) & (sw_lo - prev_l < 5*pip)

    # Reversal bar
    bear_rev = (c < sw_hi) & (c < o) & ((o - c) > 0.3 * (h - l)) & ((h - l) > 0)
    bull_rev = (c > sw_lo) & (c > o) & ((c - o) > 0.3 * (h - l)) & ((h - l) > 0)

    short_mask = valid & broke_hi & bear_rev & (np.arange(n) < n - 60)
    long_mask = valid & broke_lo & bull_rev & (np.arange(n) < n - 60)

    candidates = np.where(short_mask | long_mask)[0]

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -50
    for i in candidates:
        if i - last < 50:
            continue
        if short_mask[i]:
            e = c[i]-spread; s = prev_h[i]+2*pip; t = e-2*abs(e-s); d = -1
        else:
            e = c[i]+spread; s = prev_l[i]-2*pip; t = e+2*abs(e-s); d = 1

        sl_d = abs(e - s)
        if sl_d > 15*pip or sl_d < 1*pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s10_wedge(df, pair, pip, spread):
    """S10: Wedge/Three-Push (M1) — sampled every 20 bars for speed."""
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    atr = df["atr14"].values; rsi = df["rsi14"].values
    n = len(c)

    valid = ~np.isnan(atr) & (atr > 0) & ~np.isnan(rsi) & (np.arange(n) > 50)
    base_candidates = np.where(valid & (np.arange(n) < n - 60))[0]
    candidates = base_candidates[::10]  # sample every 10 bars

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -80
    w = 40

    for i in candidates:
        if i - last < 80:
            continue
        seg_h = h[i-w:i+1]; seg_l = l[i-w:i+1]; seg_rsi = rsi[i-w:i+1]
        if len(seg_h) < w:
            continue

        swing_highs = []
        swing_lows = []
        for k in range(2, len(seg_h)-2):
            if seg_h[k] > max(seg_h[k-1], seg_h[k-2], seg_h[k+1], seg_h[k+2]):
                swing_highs.append((k, seg_h[k], seg_rsi[k]))
            if seg_l[k] < min(seg_l[k-1], seg_l[k-2], seg_l[k+1], seg_l[k+2]):
                swing_lows.append((k, seg_l[k], seg_rsi[k]))

        found = False
        if len(swing_highs) >= 3:
            sh = swing_highs[-3:]
            if sh[0][1] < sh[1][1] < sh[2][1]:
                p1 = sh[1][1]-sh[0][1]; p2 = sh[2][1]-sh[1][1]
                if 0 < p2 < p1 and sh[2][2] < sh[1][2]:
                    if c[i] < o[i] and (o[i]-c[i]) > 0.3*(h[i]-l[i]):
                        e = c[i]-spread; s = h[i-w+sh[2][0]]+2*pip; t = e-2*abs(e-s)
                        sl_d = abs(e-s)
                        if 1*pip < sl_d < 15*pip:
                            idx.append(i); dirs.append(-1)
                            ent.append(e); sls.append(s); tps.append(t)
                            last = i; found = True

        if not found and len(swing_lows) >= 3:
            sl_pts = swing_lows[-3:]
            if sl_pts[0][1] > sl_pts[1][1] > sl_pts[2][1]:
                p1 = sl_pts[0][1]-sl_pts[1][1]; p2 = sl_pts[1][1]-sl_pts[2][1]
                if 0 < p2 < p1 and sl_pts[2][2] > sl_pts[1][2]:
                    if c[i] > o[i] and (c[i]-o[i]) > 0.3*(h[i]-l[i]):
                        e = c[i]+spread; s = l[i-w+sl_pts[2][0]]-2*pip; t = e+2*abs(e-s)
                        sl_d = abs(e-s)
                        if 1*pip < sl_d < 15*pip:
                            idx.append(i); dirs.append(1)
                            ent.append(e); sls.append(s); tps.append(t)
                            last = i

    return idx, dirs, ent, sls, tps


def strat_s11_rsi(df, pair, pip, spread):
    """S11: RSI/BB Mean Reversion (M1)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    rsi = df["rsi14"].values
    bb_u = df["bb_upper"].values; bb_l = df["bb_lower"].values; bb_m = df["bb_mid"].values
    slope = df["ema20_slope"].values
    br = df["bar_range"].values; body = df["body"].values
    n = len(c)

    valid = ~np.isnan(atr) & (atr > 0) & ~np.isnan(rsi) & ~np.isnan(bb_u)
    indecision = np.where(br > 0, body / br, 1.0) < 0.35

    long_m = valid & (rsi < 25) & (c < bb_l) & indecision & (np.arange(n) > 30) & (np.arange(n) < n - 60)
    short_m = valid & (rsi > 75) & (c > bb_u) & indecision & (np.arange(n) > 30) & (np.arange(n) < n - 60)

    candidates = np.where(long_m | short_m)[0]

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -30
    for i in candidates:
        if i - last < 30:
            continue
        if long_m[i]:
            if abs(slope[i]) > atr[i]*0.2 and slope[i] < 0:
                continue
            e = c[i]+spread; s = l[i]-atr[i]; t = float(bb_m[i])
            if t <= e:
                continue
            d = 1
        else:
            if abs(slope[i]) > atr[i]*0.2 and slope[i] > 0:
                continue
            e = c[i]-spread; s = h[i]+atr[i]; t = float(bb_m[i])
            if t >= e:
                continue
            d = -1

        sl_d = abs(e - s)
        if sl_d > 20*pip or sl_d < 1*pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s13_abcd(df, pair, pip, spread):
    """S13: ABCD Fibonacci (M1) — sampled for speed."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    n = len(c)

    valid = ~np.isnan(atr) & (atr > 0) & (np.arange(n) > 60) & (np.arange(n) < n - 120)
    base = np.where(valid)[0]
    candidates = base[::5]  # every 5 bars

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -60

    for i in candidates:
        if i - last < 60:
            continue
        look = 40
        sh = h[i-look:i-5]; sl_seg = l[i-look:i-5]
        if len(sh) < 15:
            continue

        a_hi = int(np.argmax(sh)); a_lo = int(np.argmin(sl_seg))

        found = False
        if a_hi < len(sh) - 5:
            b_lo = int(np.argmin(sl_seg[a_hi+3:])) + a_hi + 3
            ab = sh[a_hi] - sl_seg[b_lo]
            if ab > 1.5*atr[i]:
                ret = c[i] - sl_seg[b_lo]
                pct = ret / ab if ab != 0 else 0
                if 0.38 <= pct <= 0.62 and c[i] > o[i]:
                    e = c[i]+spread; s = sl_seg[b_lo]-2*pip; t = sh[a_hi]+ab*0.618
                    sl_d = abs(e-s); tp_d = abs(e-t)
                    if 2*pip < sl_d < 25*pip and tp_d > sl_d:
                        idx.append(i); dirs.append(1)
                        ent.append(e); sls.append(s); tps.append(t)
                        last = i; found = True

        if not found and a_lo < len(sl_seg) - 5:
            b_hi = int(np.argmax(sh[a_lo+3:])) + a_lo + 3
            ab = sh[b_hi] - sl_seg[a_lo]
            if ab > 1.5*atr[i]:
                ret = sh[b_hi] - c[i]
                pct = ret / ab if ab != 0 else 0
                if 0.38 <= pct <= 0.62 and c[i] < o[i]:
                    e = c[i]-spread; s = sh[b_hi]+2*pip; t = sl_seg[a_lo]-ab*0.618
                    sl_d = abs(e-s); tp_d = abs(e-t)
                    if 2*pip < sl_d < 25*pip and tp_d > sl_d:
                        idx.append(i); dirs.append(-1)
                        ent.append(e); sls.append(s); tps.append(t)
                        last = i

    return idx, dirs, ent, sls, tps


def strat_s14_bkpb(df, pair, pip, spread):
    """S14: Breakout Pullback (M1) — sampled for speed."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    n = len(c)

    valid = ~np.isnan(atr) & (atr > 0) & (np.arange(n) > 80) & (np.arange(n) < n - 120)
    base = np.where(valid)[0]
    candidates = base[::10]

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -60

    for i in candidates:
        if i - last < 60:
            continue
        seg_h = h[i-60:i-5]; seg_l = l[i-60:i-5]
        if len(seg_h) < 30:
            continue
        rt = np.max(seg_h); rb = np.min(seg_l)
        rh = rt - rb
        if rh < 5*pip or rh > 50*pip:
            continue
        tt = np.sum(np.abs(seg_h-rt) < 2*pip)
        bt = np.sum(np.abs(seg_l-rb) < 2*pip)
        if tt < 3 or bt < 3:
            continue

        bb_seg = slice(i-5, i-1)
        bb_h = h[bb_seg]; bb_l = l[bb_seg]; bb_c = c[bb_seg]; bb_o = o[bb_seg]
        bb_r = bb_h - bb_l; bb_r[bb_r == 0] = 1e-10

        bull_bo = np.any(bb_c > rt) and np.any((bb_c-bb_o)/bb_r > 0.7)
        bear_bo = np.any(bb_c < rb) and np.any((bb_o-bb_c)/bb_r > 0.7)

        if bull_bo and not bear_bo:
            if l[i] > rt - rh*0.5 and c[i] > rt:
                e = c[i]+spread; s = rt-2*pip; t = rt+rh; d = 1
            else:
                continue
        elif bear_bo and not bull_bo:
            if h[i] < rb + rh*0.5 and c[i] < rb:
                e = c[i]-spread; s = rb+2*pip; t = rb-rh; d = -1
            else:
                continue
        else:
            continue

        sl_d = abs(e-s); tp_d = abs(e-t)
        if sl_d > 20*pip or sl_d < 1*pip or tp_d < sl_d*0.8:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s15_vwap(df, pair, pip, spread):
    """S15: VWAP Session Reversion (M1 native)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    vol = df["volume"].values if "volume" in df.columns else np.zeros(len(df))
    atr = df["atr14"].values
    hours = df["hour"].values; dates = df["date"].values
    n = len(c)
    active = set(range(7, 16))

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -60
    cur_date = None; sess_start = 0

    for i in range(100, n - 300):
        if i - last < 60:
            continue
        hr = hours[i]
        if hr not in active:
            cur_date = None; continue
        d = dates[i]
        if d != cur_date:
            sess_start = i; cur_date = d
        if i - sess_start < 50:
            continue
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        sess_c = c[sess_start:i+1]
        sess_h = h[sess_start:i+1]; sess_l = l[sess_start:i+1]
        sess_v = vol[sess_start:i+1]
        if sess_v.sum() > 0:
            typ = (sess_h + sess_l + sess_c) / 3.0
            vwap = (typ * sess_v).sum() / sess_v.sum()
        else:
            vwap = sess_c.mean()

        dev = c[i] - vwap
        if abs(dev / a) < 2.5:
            continue

        sl_dist = a * 0.5
        if dev > 0:
            e = c[i]-spread/2; s = e+sl_dist; t = float(vwap); direction = -1
        else:
            e = c[i]+spread/2; s = e-sl_dist; t = float(vwap); direction = 1

        if direction == 1 and t <= e:
            continue
        if direction == -1 and t >= e:
            continue

        idx.append(i); dirs.append(direction); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def strat_s16_hl2(df, pair, pip, spread):
    """S16: High 2 / Low 2 (M1)."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values; ema20 = df["ema20"].values
    slope = df["ema20_slope"].values
    n = len(c)

    valid = ~np.isnan(atr) & (atr > 0) & (np.abs(slope) > atr * 0.03)
    bull = valid & (slope > 0) & (c > ema20)
    bear = valid & (slope < 0) & (c < ema20)

    mask = (bull | bear) & (np.arange(n) > 30) & (np.arange(n) < n - 60)
    candidates = np.where(mask)[0]

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -40
    for i in candidates:
        if i - last < 40:
            continue
        if bull[i]:
            h1 = False; h2_ok = False
            for k in range(max(1, i-20), i+1):
                if h[k] > h[k-1]:
                    if h1:
                        if k == i:
                            h2_ok = True
                        break
                    else:
                        h1 = True
                        went_down = False
                        for m in range(k+1, min(k+6, i+1)):
                            if m >= 1 and l[m] < l[m-1]:
                                went_down = True; break
                        if not went_down:
                            h1 = False
            if not h2_ok:
                continue
            pb_low = np.min(l[max(0,i-20):i+1])
            e = h[i]+spread; s = pb_low-pip; t = e+2*abs(e-s); d = 1
        else:
            l1 = False; l2_ok = False
            for k in range(max(1, i-20), i+1):
                if l[k] < l[k-1]:
                    if l1:
                        if k == i:
                            l2_ok = True
                        break
                    else:
                        l1 = True
                        went_up = False
                        for m in range(k+1, min(k+6, i+1)):
                            if m >= 1 and h[m] > h[m-1]:
                                went_up = True; break
                        if not went_up:
                            l1 = False
            if not l2_ok:
                continue
            pb_hi = np.max(h[max(0,i-20):i+1])
            e = l[i]-spread; s = pb_hi+pip; t = e-2*abs(e-s); d = -1

        sl_d = abs(e - s)
        if sl_d > 15*pip or sl_d < 1*pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


# ---------------------------------------------------------------------------
#  DISPATCH
# ---------------------------------------------------------------------------

STRATEGIES = [
    ("S1_DD",    strat_s1_dd,    60),
    ("S2_FB",    strat_s2_fb,    60),
    ("S3_SB",    strat_s3_sb,    60),
    ("S4_EMA",   strat_s4_ema,   60),
    ("S5_BB",    strat_s5_bb,    60),
    ("S6_RB",    strat_s6_rb,   120),
    ("S9_FBR",   strat_s9_fbr,   60),
    ("S10_WDG",  strat_s10_wedge, 60),
    ("S11_RSI",  strat_s11_rsi,  60),
    ("S13_ABCD", strat_s13_abcd,120),
    ("S14_BKPB", strat_s14_bkpb,120),
    ("S15_VWAP", strat_s15_vwap, 300),
    ("S16_HL2",  strat_s16_hl2,  60),
]


# ---------------------------------------------------------------------------
#  REPORTING
# ---------------------------------------------------------------------------

def print_report(all_trades):
    if not all_trades:
        print("  No trades."); return

    print(f"\n{'='*90}")
    print(f"  M1 BACKTEST — {len(all_trades):,} trades")
    print(f"{'='*90}")

    total = sum(t["pnl_pips"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["pnl_pips"] > 0)
    wr = wins / len(all_trades) * 100
    gp = sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] > 0)
    gl = abs(sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] <= 0))
    pf = gp / gl if gl else 999

    print(f"\n  Overall: {wins:,}W / {len(all_trades)-wins:,}L | WR {wr:.1f}% | PnL {total:+,.0f} pips | PF {pf:.2f}")

    print(f"\n  {'Strategy':12s} {'Trades':>8s} {'WR':>7s} {'PnL':>12s} {'Avg':>8s} {'PF':>7s}")
    print(f"  {'-'*60}")
    strats = defaultdict(list)
    for t in all_trades:
        strats[t["strategy"]].append(t)
    for s in sorted(strats.keys()):
        st = strats[s]
        w = sum(1 for t in st if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in st)
        g = sum(t["pnl_pips"] for t in st if t["pnl_pips"] > 0)
        lo = abs(sum(t["pnl_pips"] for t in st if t["pnl_pips"] <= 0))
        pf_v = g/lo if lo else 999
        print(f"  {s:12s} {len(st):8,d} {w/len(st)*100:6.1f}% {p:+12,.0f} {p/len(st):+8.2f} {pf_v:7.2f}")

    print(f"\n  TOP 30 COMBOS:")
    print(f"  {'Combo':38s} {'N':>6s} {'WR':>7s} {'PnL':>10s} {'Avg':>7s} {'PF':>6s}")
    combos = defaultdict(list)
    for t in all_trades:
        combos[f"{t['strategy']} {t['pair']} {t['direction']}"].append(t)
    ranked = []
    for k, ct in combos.items():
        if len(ct) < 20:
            continue
        w = sum(1 for t in ct if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in ct)
        g = sum(t["pnl_pips"] for t in ct if t["pnl_pips"] > 0)
        lo = abs(sum(t["pnl_pips"] for t in ct if t["pnl_pips"] <= 0))
        ranked.append((k, len(ct), w/len(ct)*100, p, p/len(ct), g/lo if lo else 999))
    print("  --- BEST ---")
    for k, nn, w, p, a, pf_v in sorted(ranked, key=lambda x: -x[3])[:30]:
        print(f"  {k:38s} {nn:6d} {w:6.1f}% {p:+10,.0f} {a:+7.2f} {pf_v:6.2f}")
    print("  --- WORST ---")
    for k, nn, w, p, a, pf_v in sorted(ranked, key=lambda x: x[3])[:15]:
        print(f"  {k:38s} {nn:6d} {w:6.1f}% {p:+10,.0f} {a:+7.2f} {pf_v:6.2f}")


def save_results(all_trades):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combo = defaultdict(lambda: {"wins":0,"losses":0,"pnl":0,"trades":0})
    for t in all_trades:
        key = f"{t['strategy']}_{t['pair']}_{t['direction']}"
        s = combo[key]; s["trades"]+=1; s["pnl"]+=t["pnl_pips"]
        if t["pnl_pips"]>0: s["wins"]+=1
        else: s["losses"]+=1
    for s in combo.values():
        s["win_rate"]=round(s["wins"]/s["trades"]*100,1)
        s["avg_pips"]=round(s["pnl"]/s["trades"],2)
        s["pnl"]=round(s["pnl"],1)
    with open(OUTPUT_DIR/"m1_combo_stats.json","w") as f:
        json.dump(dict(combo),f,indent=2,sort_keys=True)

    strat_sum = {}
    strats = defaultdict(list)
    for t in all_trades:
        strats[t["strategy"]].append(t)
    for s, trades in strats.items():
        w=sum(1 for t in trades if t["pnl_pips"]>0)
        p=sum(t["pnl_pips"] for t in trades)
        g=sum(t["pnl_pips"] for t in trades if t["pnl_pips"]>0)
        lo=abs(sum(t["pnl_pips"] for t in trades if t["pnl_pips"]<=0))
        strat_sum[s]={"trades":len(trades),"wins":w,"win_rate":round(w/len(trades)*100,1),
                      "total_pips":round(p,1),"avg_pips":round(p/len(trades),2),
                      "profit_factor":round(g/lo,3) if lo else 999}
    with open(OUTPUT_DIR/"m1_strategy_summary.json","w") as f:
        json.dump(strat_sum,f,indent=2,sort_keys=True)

    lessons = []
    for key, s in sorted(combo.items(), key=lambda x:-x[1]["pnl"]):
        if s["trades"]<50: continue
        if s["win_rate"]>=53 and s["avg_pips"]>0.2:
            lessons.append({"theme":f"M1-EDGE: {key} — {s['win_rate']}% WR, {s['trades']} trades, avg {s['avg_pips']:+.2f}p",
                            "confidence":min(0.95,0.5+s["win_rate"]/200),"source":"backtest_m1_all","pinned":True})
    with open(OUTPUT_DIR/"m1_lessons.jsonl","w") as f:
        for le in lessons:
            f.write(json.dumps(le)+"\n")

    print(f"\n  Saved: m1_combo_stats.json, m1_strategy_summary.json, m1_lessons.jsonl ({len(lessons)} lessons)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=0)
    args = parser.parse_args()

    print("="*90)
    print(f"  M1 FULL BACKTEST — 13 strategies" + (f" (last {args.years}yr)" if args.years else " (full history)"))
    print("="*90)

    all_trades = []
    t0 = time_mod.time()

    for fb, pair in PAIR_MAP.items():
        fn = f"{fb}_1min.csv"
        print(f"\n  {pair} loading...", end=" ", flush=True)
        df = load_pair(fn, pair, args.years)
        if df.empty:
            print("SKIP"); continue
        print(f"{len(df):,} M1 bars", end="", flush=True)

        print(" | ind...", end="", flush=True)
        df = add_indicators(df, pair)
        pip = 0.01 if "JPY" in pair else 0.0001
        spread = SPREAD_PIPS.get(pair, 0.5) * pip

        h_arr = df["high"].values; l_arr = df["low"].values
        c_arr = df["close"].values; hours = df["hour"].values
        ts = df["timestamp"].values

        for sname, sfunc, mb in STRATEGIES:
            print(f" {sname}", end="", flush=True)
            try:
                sig = sfunc(df, pair, pip, spread)
                idx, dirs, ent, sls, tps = sig[0], sig[1], sig[2], sig[3], sig[4]
                print(f":{len(idx)}", end="", flush=True)
                if idx:
                    results = sim_trades(h_arr, l_arr, c_arr, idx, dirs, ent, sls, tps, mb, pip)
                    trades = build_trades(sname, idx, dirs, ent, sls, tps, results, hours, ts, pair, pip)
                    all_trades.extend(trades)
            except Exception as e:
                print(f":ERR({e})", end="", flush=True)

        elapsed = time_mod.time() - t0
        print(f" | {elapsed:.0f}s")

    total_time = time_mod.time() - t0
    print(f"\n  Total: {len(all_trades):,} trades in {total_time:.0f}s ({total_time/60:.1f}min)")

    print_report(all_trades)
    save_results(all_trades)
    print("\n  Done.")


if __name__ == "__main__":
    main()
