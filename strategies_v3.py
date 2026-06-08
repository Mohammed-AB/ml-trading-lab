from __future__ import annotations

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# GROUP 1 — ICT / Market Structure  (7 strategies)
# ═══════════════════════════════════════════════════════════════════════════


def signals_v3_fvg(df, pair, pip, spread):
    """Fair Value Gap – enter on retest of a 3-candle imbalance zone."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    atr = df["atr14"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(4, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 8:
            continue
        a = atr[i]

        bull_gap_top = l[i - 2]
        bull_gap_bot = h[i]
        if bull_gap_top > bull_gap_bot + 0.2 * a:
            gap_mid = (bull_gap_top + bull_gap_bot) / 2.0
            if l[i] <= bull_gap_top and c[i] > gap_mid and c[i - 1] > c[i - 2]:
                e = gap_mid + spread
                sl = bull_gap_bot - 0.5 * a
                risk = abs(e - sl)
                if 0.5 * a < risk < 4 * a:
                    idx.append(i); dirs.append(1); entries.append(e)
                    sls.append(sl); tps.append(e + 2 * risk)
                    last = i; continue

        bear_gap_bot = h[i - 2]
        bear_gap_top = l[i]
        if bear_gap_top > bear_gap_bot + 0.2 * a:
            gap_mid = (bear_gap_top + bear_gap_bot) / 2.0
            if h[i] >= bear_gap_bot and c[i] < gap_mid and c[i - 1] < c[i - 2]:
                e = gap_mid - spread
                sl = bear_gap_top + 0.5 * a
                risk = abs(e - sl)
                if 0.5 * a < risk < 4 * a:
                    idx.append(i); dirs.append(-1); entries.append(e)
                    sls.append(sl); tps.append(e - 2 * risk)
                    last = i

    return idx, dirs, entries, sls, tps


def signals_v3_brk(df, pair, pip, spread):
    """Breaker block – last opposing candle before a strong impulse becomes entry zone on retest."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    atr = df["atr14"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(6, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 8:
            continue
        a = atr[i]

        impulse_up = c[i - 1] - o[i - 1]
        if impulse_up > 1.5 * a and c[i - 2] < o[i - 2]:
            zone_hi = o[i - 2]
            zone_lo = c[i - 2]
            if l[i] <= zone_hi and c[i] > zone_hi:
                e = zone_hi + spread
                sl = zone_lo - 0.3 * a
                risk = abs(e - sl)
                if 0.3 * a < risk < 5 * a:
                    idx.append(i); dirs.append(1); entries.append(e)
                    sls.append(sl); tps.append(e + 2 * risk)
                    last = i; continue

        impulse_dn = o[i - 1] - c[i - 1]
        if impulse_dn > 1.5 * a and c[i - 2] > o[i - 2]:
            zone_hi = c[i - 2]
            zone_lo = o[i - 2]
            if h[i] >= zone_lo and c[i] < zone_lo:
                e = zone_lo - spread
                sl = zone_hi + 0.3 * a
                risk = abs(e - sl)
                if 0.3 * a < risk < 5 * a:
                    idx.append(i); dirs.append(-1); entries.append(e)
                    sls.append(sl); tps.append(e - 2 * risk)
                    last = i

    return idx, dirs, entries, sls, tps


def signals_v3_liqswp(df, pair, pip, spread):
    """Liquidity sweep – price sweeps 20-bar high/low then reverses back inside."""
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    atr = df["atr14"].values; br = df["bar_range"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(22, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 10:
            continue
        a = atr[i]
        hi20 = np.max(h[i - 21:i - 1])
        lo20 = np.min(l[i - 21:i - 1])
        wick_up = h[i] - max(c[i], df["open"].values[i])
        wick_dn = min(c[i], df["open"].values[i]) - l[i]
        rng = br[i] if br[i] > 0 else a

        if h[i] > hi20 and c[i] < hi20 and wick_up > 0.5 * rng:
            e = c[i] - spread
            sl = h[i] + 0.5 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 5 * a:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(sl); tps.append(e - 1.5 * risk)
                last = i; continue

        if l[i] < lo20 and c[i] > lo20 and wick_dn > 0.5 * rng:
            e = c[i] + spread
            sl = l[i] - 0.5 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 5 * a:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(sl); tps.append(e + 1.5 * risk)
                last = i

    return idx, dirs, entries, sls, tps


def _swing_points(h, l, n, lookback=10):
    """Return arrays of swing-high and swing-low values (NaN where not a swing)."""
    sw_h = np.full(n, np.nan)
    sw_l = np.full(n, np.nan)
    for i in range(lookback, n - lookback):
        if h[i] == np.max(h[i - lookback:i + lookback + 1]):
            sw_h[i] = h[i]
        if l[i] == np.min(l[i - lookback:i + lookback + 1]):
            sw_l[i] = l[i]
    return sw_h, sw_l


def signals_v3_bos(df, pair, pip, spread):
    """Break of Structure continuation – enter on pullback to the broken swing level."""
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    atr = df["atr14"].values
    n = len(df)
    sw_h, sw_l = _swing_points(h, l, n, 10)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    prev_sh = np.nan
    prev_sl = np.nan
    trend = 0  # +1 up, -1 down

    for i in range(22, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        a = atr[i]

        if not np.isnan(sw_h[i - 1]):
            if not np.isnan(prev_sh) and sw_h[i - 1] > prev_sh:
                trend = 1
            prev_sh = sw_h[i - 1]
        if not np.isnan(sw_l[i - 1]):
            if not np.isnan(prev_sl) and sw_l[i - 1] < prev_sl:
                trend = -1
            prev_sl = sw_l[i - 1]

        if i - last < 10:
            continue

        if trend == 1 and not np.isnan(prev_sl):
            if l[i] <= prev_sl + 0.3 * a and c[i] > prev_sl:
                e = c[i] + spread
                sl = prev_sl - a
                risk = abs(e - sl)
                if 0.5 * a < risk < 5 * a:
                    idx.append(i); dirs.append(1); entries.append(e)
                    sls.append(sl); tps.append(e + 2 * risk)
                    last = i; continue

        if trend == -1 and not np.isnan(prev_sh):
            if h[i] >= prev_sh - 0.3 * a and c[i] < prev_sh:
                e = c[i] - spread
                sl = prev_sh + a
                risk = abs(e - sl)
                if 0.5 * a < risk < 5 * a:
                    idx.append(i); dirs.append(-1); entries.append(e)
                    sls.append(sl); tps.append(e - 2 * risk)
                    last = i

    return idx, dirs, entries, sls, tps


def signals_v3_choch(df, pair, pip, spread):
    """Change of Character – first lower-low in uptrend or higher-high in downtrend."""
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    atr = df["atr14"].values
    n = len(df)
    sw_h, sw_l = _swing_points(h, l, n, 10)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    prev_sh = np.nan; prev_sl = np.nan
    trend = 0
    choch_bar = -100; choch_dir = 0
    choch_zone_hi = np.nan; choch_zone_lo = np.nan

    for i in range(22, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        a = atr[i]

        if not np.isnan(sw_h[i - 1]):
            if trend == -1 and not np.isnan(prev_sh) and sw_h[i - 1] > prev_sh:
                choch_bar = i - 1; choch_dir = 1
                choch_zone_lo = l[i - 1]; choch_zone_hi = h[i - 1]
                trend = 1
            elif not np.isnan(prev_sh):
                if sw_h[i - 1] > prev_sh:
                    trend = 1
            prev_sh = sw_h[i - 1]

        if not np.isnan(sw_l[i - 1]):
            if trend == 1 and not np.isnan(prev_sl) and sw_l[i - 1] < prev_sl:
                choch_bar = i - 1; choch_dir = -1
                choch_zone_lo = l[i - 1]; choch_zone_hi = h[i - 1]
                trend = -1
            elif not np.isnan(prev_sl):
                if sw_l[i - 1] < prev_sl:
                    trend = -1
            prev_sl = sw_l[i - 1]

        if i - last < 10 or i - choch_bar > 20 or choch_dir == 0:
            continue

        if choch_dir == 1 and l[i] <= choch_zone_hi and c[i] > choch_zone_lo:
            e = c[i] + spread
            sl = choch_zone_lo - 0.3 * a
            risk = abs(e - sl)
            if 0.3 * a < risk < 5 * a:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(sl); tps.append(e + 1.5 * risk)
                last = i; choch_dir = 0; continue

        if choch_dir == -1 and h[i] >= choch_zone_lo and c[i] < choch_zone_hi:
            e = c[i] - spread
            sl = choch_zone_hi + 0.3 * a
            risk = abs(e - sl)
            if 0.3 * a < risk < 5 * a:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(sl); tps.append(e - 1.5 * risk)
                last = i; choch_dir = 0

    return idx, dirs, entries, sls, tps


def signals_v3_eqliq(df, pair, pip, spread):
    """Equal highs/lows stop-run – detect 2+ bars with highs within 0.3 ATR then fade the sweep."""
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    atr = df["atr14"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(22, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 10:
            continue
        a = atr[i]
        thr = 0.3 * a

        eq_hi_count = 0
        eq_hi_level = 0.0
        for j in range(i - 15, i - 1):
            if j < 1:
                continue
            for k in range(j + 1, min(j + 8, i)):
                if abs(h[j] - h[k]) < thr:
                    eq_hi_count += 1
                    eq_hi_level = max(h[j], h[k])
                    break
            if eq_hi_count >= 2:
                break

        if eq_hi_count >= 2 and h[i] > eq_hi_level and c[i] < eq_hi_level:
            e = c[i] - spread
            sl = h[i] + 0.5 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 5 * a:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(sl); tps.append(e - 2 * risk)
                last = i; continue

        eq_lo_count = 0
        eq_lo_level = 1e12
        for j in range(i - 15, i - 1):
            if j < 1:
                continue
            for k in range(j + 1, min(j + 8, i)):
                if abs(l[j] - l[k]) < thr:
                    eq_lo_count += 1
                    eq_lo_level = min(l[j], l[k])
                    break
            if eq_lo_count >= 2:
                break

        if eq_lo_count >= 2 and l[i] < eq_lo_level and c[i] > eq_lo_level:
            e = c[i] + spread
            sl = l[i] - 0.5 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 5 * a:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(sl); tps.append(e + 2 * risk)
                last = i

    return idx, dirs, entries, sls, tps


def signals_v3_sd_zone(df, pair, pip, spread):
    """Supply/demand zone – impulse move >2 ATR in 3 bars; enter on first retest of consolidation zone."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    atr = df["atr14"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []

    zones = []  # (zone_hi, zone_lo, direction, used)

    for i in range(5, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        a = atr[i]

        if i >= 5:
            move_up = c[i - 1] - l[i - 3]
            if move_up > 2 * a:
                z_hi = max(h[i - 4], h[i - 3])
                z_lo = min(l[i - 4], l[i - 3])
                if z_hi - z_lo < 2 * a:
                    zones.append([z_hi, z_lo, 1, False, i - 1])

            move_dn = h[i - 3] - c[i - 1]
            if move_dn > 2 * a:
                z_hi = max(h[i - 4], h[i - 3])
                z_lo = min(l[i - 4], l[i - 3])
                if z_hi - z_lo < 2 * a:
                    zones.append([z_hi, z_lo, -1, False, i - 1])

        active = [z for z in zones if not z[3] and i - z[4] < 60]
        zones = [z for z in zones if not z[3] and i - z[4] < 60]

        for z in active:
            z_hi, z_lo, d, used, born = z
            if d == 1 and l[i] <= z_hi and c[i] > z_lo:
                e = z_hi + spread
                sl = z_lo - 0.3 * a
                risk = abs(e - sl)
                if 0.5 * a < risk < 5 * a:
                    idx.append(i); dirs.append(1); entries.append(e)
                    sls.append(sl); tps.append(e + 2 * risk)
                    z[3] = True; break
            elif d == -1 and h[i] >= z_lo and c[i] < z_hi:
                e = z_lo - spread
                sl = z_hi + 0.3 * a
                risk = abs(e - sl)
                if 0.5 * a < risk < 5 * a:
                    idx.append(i); dirs.append(-1); entries.append(e)
                    sls.append(sl); tps.append(e - 2 * risk)
                    z[3] = True; break

    return idx, dirs, entries, sls, tps


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 2 — Pivot / Level-Based  (4 strategies)
# ═══════════════════════════════════════════════════════════════════════════


def signals_v3_cam_mr(df, pair, pip, spread):
    """Camarilla mean-reversion – fade at R4/S4 levels."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values
    cam_r4 = df["cam_r4"].values; cam_s4 = df["cam_s4"].values
    cam_r3 = df["cam_r3"].values; cam_s3 = df["cam_s3"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(22, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 8:
            continue
        if np.isnan(cam_r4[i]) or np.isnan(cam_s4[i]):
            continue
        a = atr[i]

        if h[i] >= cam_r4[i] and c[i] <= cam_r4[i]:
            e = c[i] - spread
            sl = cam_r4[i] + a
            tp = cam_r3[i] if not np.isnan(cam_r3[i]) else e - 2 * a
            risk = abs(e - sl)
            if 0.3 * a < risk < 5 * a and abs(e - tp) > 0.3 * a:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(sl); tps.append(tp)
                last = i; continue

        if l[i] <= cam_s4[i] and c[i] >= cam_s4[i]:
            e = c[i] + spread
            sl = cam_s4[i] - a
            tp = cam_s3[i] if not np.isnan(cam_s3[i]) else e + 2 * a
            risk = abs(e - sl)
            if 0.3 * a < risk < 5 * a and abs(e - tp) > 0.3 * a:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(sl); tps.append(tp)
                last = i

    return idx, dirs, entries, sls, tps


def signals_v3_cam_bo(df, pair, pip, spread):
    """Camarilla breakout – close beyond R3/S3 with EMA momentum."""
    c = df["close"].values
    atr = df["atr14"].values
    ema9 = df["ema9"].values; ema20 = df["ema20"].values
    cam_r3 = df["cam_r3"].values; cam_s3 = df["cam_s3"].values
    cam_r4 = df["cam_r4"].values; cam_s4 = df["cam_s4"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(22, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 8:
            continue
        if np.isnan(cam_r3[i]) or np.isnan(cam_s3[i]):
            continue
        a = atr[i]
        mid_rs = (cam_r3[i] + cam_s3[i]) / 2.0

        if c[i] > cam_r3[i] and c[i] > ema9[i] and ema9[i] > ema20[i]:
            e = c[i] + spread
            sl = mid_rs
            tp = cam_r4[i] if not np.isnan(cam_r4[i]) else e + 2 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 6 * a and abs(e - tp) > 0.3 * a:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(sl); tps.append(tp)
                last = i; continue

        if c[i] < cam_s3[i] and c[i] < ema9[i] and ema9[i] < ema20[i]:
            e = c[i] - spread
            sl = mid_rs
            tp = cam_s4[i] if not np.isnan(cam_s4[i]) else e - 2 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 6 * a and abs(e - tp) > 0.3 * a:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(sl); tps.append(tp)
                last = i

    return idx, dirs, entries, sls, tps


def signals_v3_wood(df, pair, pip, spread):
    """Woodie pivot bounce – touch pivot_pp with RSI filter."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values; rsi = df["rsi14"].values
    pp = df["pivot_pp"].values
    r1 = df["pivot_r1"].values; s1 = df["pivot_s1"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(22, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 8:
            continue
        if np.isnan(pp[i]) or np.isnan(rsi[i]):
            continue
        a = atr[i]
        thr = 0.3 * a

        approaching_from_below = c[i - 1] < pp[i] and l[i] <= pp[i] + thr and c[i] >= pp[i]
        if approaching_from_below and rsi[i] > 40:
            e = c[i] + spread
            sl = e - a
            tp = r1[i] if not np.isnan(r1[i]) else e + 2 * a
            risk = abs(e - sl)
            if abs(e - tp) > 0.3 * a:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(sl); tps.append(tp)
                last = i; continue

        approaching_from_above = c[i - 1] > pp[i] and h[i] >= pp[i] - thr and c[i] <= pp[i]
        if approaching_from_above and rsi[i] < 60:
            e = c[i] - spread
            sl = e + a
            tp = s1[i] if not np.isnan(s1[i]) else e - 2 * a
            risk = abs(e - sl)
            if abs(e - tp) > 0.3 * a:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(sl); tps.append(tp)
                last = i

    return idx, dirs, entries, sls, tps


def signals_v3_floor(df, pair, pip, spread):
    """Floor pivot confluence – price near pivot_pp AND near prior-day high/low."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values
    pp = df["pivot_pp"].values
    dates = df["date"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    prev_day_hi = np.nan; prev_day_lo = np.nan
    cur_day_hi = -1e12; cur_day_lo = 1e12
    cur_date = None

    for i in range(22, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        d = dates[i]
        if d != cur_date:
            if cur_date is not None:
                prev_day_hi = cur_day_hi
                prev_day_lo = cur_day_lo
            cur_date = d
            cur_day_hi = h[i]; cur_day_lo = l[i]
        else:
            cur_day_hi = max(cur_day_hi, h[i])
            cur_day_lo = min(cur_day_lo, l[i])

        if i - last < 8 or np.isnan(pp[i]) or np.isnan(prev_day_hi):
            continue
        a = atr[i]
        near_pp = abs(c[i] - pp[i]) < 0.5 * a
        near_prev_hi = abs(c[i] - prev_day_hi) < 0.5 * a
        near_prev_lo = abs(c[i] - prev_day_lo) < 0.5 * a

        if not near_pp or not (near_prev_hi or near_prev_lo):
            continue

        if c[i] > pp[i] and c[i] > c[i - 1]:
            e = c[i] + spread
            sl = e - 1.5 * a
            risk = abs(e - sl)
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(sl); tps.append(e + 2 * risk)
            last = i
        elif c[i] < pp[i] and c[i] < c[i - 1]:
            e = c[i] - spread
            sl = e + 1.5 * a
            risk = abs(e - sl)
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(sl); tps.append(e - 2 * risk)
            last = i

    return idx, dirs, entries, sls, tps


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 3 — Ichimoku  (3 strategies)
# ═══════════════════════════════════════════════════════════════════════════


def signals_v3_kumo(df, pair, pip, spread):
    """Kumo twist entry – senkou_a / senkou_b cross with cloud and TK confirmation."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values
    sa = df["ichi_senkou_a"].values; sb = df["ichi_senkou_b"].values
    tk = df["ichi_tenkan"].values; kj = df["ichi_kijun"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(54, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 10:
            continue
        if np.isnan(sa[i]) or np.isnan(sb[i]) or np.isnan(tk[i]) or np.isnan(kj[i]):
            continue
        a = atr[i]
        cloud_top = max(sa[i], sb[i])
        cloud_bot = min(sa[i], sb[i])

        bull_twist = sa[i] > sb[i] and sa[i - 1] <= sb[i - 1]
        if bull_twist and c[i] > cloud_top and tk[i] > kj[i]:
            if l[i] <= kj[i] + 0.5 * a:
                e = c[i] + spread
                sl = cloud_bot - 0.3 * a
                risk = abs(e - sl)
                if 0.5 * a < risk < 8 * a:
                    idx.append(i); dirs.append(1); entries.append(e)
                    sls.append(sl); tps.append(e + 2 * risk)
                    last = i; continue

        bear_twist = sa[i] < sb[i] and sa[i - 1] >= sb[i - 1]
        if bear_twist and c[i] < cloud_bot and tk[i] < kj[i]:
            if h[i] >= kj[i] - 0.5 * a:
                e = c[i] - spread
                sl = cloud_top + 0.3 * a
                risk = abs(e - sl)
                if 0.5 * a < risk < 8 * a:
                    idx.append(i); dirs.append(-1); entries.append(e)
                    sls.append(sl); tps.append(e - 2 * risk)
                    last = i

    return idx, dirs, entries, sls, tps


def signals_v3_kijun(df, pair, pip, spread):
    """Kijun bounce – price touches kijun in trend then bounces."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values
    sa = df["ichi_senkou_a"].values; sb = df["ichi_senkou_b"].values
    tk = df["ichi_tenkan"].values; kj = df["ichi_kijun"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(54, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 10:
            continue
        if np.isnan(sa[i]) or np.isnan(sb[i]) or np.isnan(tk[i]) or np.isnan(kj[i]):
            continue
        a = atr[i]
        cloud_top = max(sa[i], sb[i])
        cloud_bot = min(sa[i], sb[i])

        bull_trend = c[i] > cloud_top and tk[i] > kj[i]
        if bull_trend and l[i] <= kj[i] + 0.2 * a and c[i] > kj[i]:
            e = c[i] + spread
            sl = kj[i] - a
            risk = abs(e - sl)
            if 0.5 * a < risk < 6 * a:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(sl); tps.append(e + 2 * risk)
                last = i; continue

        bear_trend = c[i] < cloud_bot and tk[i] < kj[i]
        if bear_trend and h[i] >= kj[i] - 0.2 * a and c[i] < kj[i]:
            e = c[i] - spread
            sl = kj[i] + a
            risk = abs(e - sl)
            if 0.5 * a < risk < 6 * a:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(sl); tps.append(e - 2 * risk)
                last = i

    return idx, dirs, entries, sls, tps


def signals_v3_chikou(df, pair, pip, spread):
    """Chikou confirmation – chikou crosses above past price with cloud + TK alignment."""
    c = df["close"].values
    atr = df["atr14"].values
    sa = df["ichi_senkou_a"].values; sb = df["ichi_senkou_b"].values
    tk = df["ichi_tenkan"].values; kj = df["ichi_kijun"].values
    chikou = df["ichi_chikou"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(54, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 10:
            continue
        if np.isnan(sa[i]) or np.isnan(sb[i]) or np.isnan(tk[i]) or np.isnan(kj[i]):
            continue
        if np.isnan(chikou[i]) or np.isnan(chikou[i - 1]):
            continue
        if i < 27:
            continue
        a = atr[i]
        cloud_top = max(sa[i], sb[i])
        cloud_bot = min(sa[i], sb[i])

        past_price = c[i - 26] if i >= 26 else np.nan
        prev_past = c[i - 27] if i >= 27 else np.nan
        if np.isnan(past_price) or np.isnan(prev_past):
            continue

        chi_cross_up = chikou[i] > past_price and chikou[i - 1] <= prev_past
        if chi_cross_up and c[i] > cloud_top and tk[i] > kj[i]:
            e = c[i] + spread
            sl = kj[i] - 0.3 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 8 * a:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(sl); tps.append(e + 2 * risk)
                last = i; continue

        chi_cross_dn = chikou[i] < past_price and chikou[i - 1] >= prev_past
        if chi_cross_dn and c[i] < cloud_bot and tk[i] < kj[i]:
            e = c[i] - spread
            sl = kj[i] + 0.3 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 8 * a:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(sl); tps.append(e - 2 * risk)
                last = i

    return idx, dirs, entries, sls, tps


# ═══════════════════════════════════════════════════════════════════════════
# GROUP 4 — Channel / Breakout  (first strategy only)
# ═══════════════════════════════════════════════════════════════════════════


def signals_v3_donch(df, pair, pip, spread):
    """Donchian 20/10 system – breakout on 20-bar channel, SL at 10-bar opposite channel."""
    c = df["close"].values
    atr = df["atr14"].values
    dh20 = df["donch_high_20"].values; dl20 = df["donch_low_20"].values
    dh10 = df["donch_high_10"].values; dl10 = df["donch_low_10"].values
    n = len(df)
    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -10

    for i in range(22, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or i - last < 8:
            continue
        if np.isnan(dh20[i]) or np.isnan(dl20[i]):
            continue
        a = atr[i]

        if c[i] > dh20[i]:
            e = c[i] + spread
            sl = dl10[i] if not np.isnan(dl10[i]) else e - 2 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 8 * a:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(sl); tps.append(e + 2 * a)
                last = i; continue

        if c[i] < dl20[i]:
            e = c[i] - spread
            sl = dh10[i] if not np.isnan(dh10[i]) else e + 2 * a
            risk = abs(e - sl)
            if 0.5 * a < risk < 8 * a:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(sl); tps.append(e - 2 * a)
                last = i

    return idx, dirs, entries, sls, tps


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

STRATEGIES_V3_PART1 = [
    # Group 1 – ICT / Market Structure
    ("V3_FVG",     signals_v3_fvg,     40),
    ("V3_BRK",     signals_v3_brk,     40),
    ("V3_LIQSWP",  signals_v3_liqswp,  40),
    ("V3_BOS",     signals_v3_bos,     40),
    ("V3_CHOCH",   signals_v3_choch,   40),
    ("V3_EQLIQ",   signals_v3_eqliq,   40),
    ("V3_SD_ZONE", signals_v3_sd_zone, 40),
    # Group 2 – Pivot / Level-Based
    ("V3_CAM_MR",  signals_v3_cam_mr,  40),
    ("V3_CAM_BO",  signals_v3_cam_bo,  40),
    ("V3_WOOD",    signals_v3_wood,    40),
    ("V3_FLOOR",   signals_v3_floor,   40),
    # Group 3 – Ichimoku
    ("V3_KUMO",    signals_v3_kumo,    60),
    ("V3_KIJUN",   signals_v3_kijun,   60),
    ("V3_CHIKOU",  signals_v3_chikou,  60),
    # Group 4 – Channel / Breakout
    ("V3_DONCH",   signals_v3_donch,   40),
]

from strategies_v3_part2 import STRATEGIES_V3_PART2  # noqa: E402

STRATEGIES_V3 = STRATEGIES_V3_PART1 + STRATEGIES_V3_PART2
