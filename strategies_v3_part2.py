from __future__ import annotations

import numpy as np
import pandas as pd


# ===================================================================
# Group 4 continued: Channel / Breakout (3 strategies)
# ===================================================================


def signals_v3_donch_vol(df, pair, pip, spread):
    """Donchian 20/10 breakout filtered by expanding ATR volatility."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = df["atr14"].values
    dh20 = df["donch_high_20"].values
    dl20 = df["donch_low_20"].values
    dh10 = df["donch_high_10"].values
    dl10 = df["donch_low_10"].values
    n = len(df)

    atr_sma20 = np.full(n, np.nan)
    for i in range(19, n):
        atr_sma20[i] = np.mean(atr[i - 19 : i + 1])

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    for i in range(100, n):
        if i - last < 20:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(atr_sma20[i]):
            continue
        if atr[i] <= atr_sma20[i]:
            continue

        if c[i] > dh20[i - 1]:
            e = c[i] + spread
            s = dl10[i] - 0.5 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + 2.0 * risk
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t); last = i
        elif c[i] < dl20[i - 1]:
            e = c[i] - spread
            s = dh10[i] + 0.5 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - 2.0 * risk
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


def signals_v3_kelt_adx(df, pair, pip, spread):
    """Keltner band touch + ADX trend confirmation."""
    c = df["close"].values
    l = df["low"].values
    h = df["high"].values
    atr = df["atr14"].values
    adx = df["adx"].values
    pdi = df["plus_di"].values
    mdi = df["minus_di"].values
    ku = df["kelt_upper_15"].values
    kl = df["kelt_lower_15"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    for i in range(100, n):
        if i - last < 20:
            continue
        if np.isnan(adx[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if adx[i] <= 25.0:
            continue

        if pdi[i] > mdi[i] and l[i] <= kl[i]:
            e = c[i] + spread
            s = kl[i] - atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = ku[i]
            if t <= e:
                continue
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t); last = i
        elif mdi[i] > pdi[i] and h[i] >= ku[i]:
            e = c[i] - spread
            s = ku[i] + atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = kl[i]
            if t >= e:
                continue
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


def signals_v3_kelt_walk(df, pair, pip, spread):
    """Keltner band walk — pullback entry after 3+ closes beyond upper/lower band."""
    c = df["close"].values
    atr = df["atr14"].values
    ema20 = df["ema20"].values
    ku = df["kelt_upper_15"].values
    kl = df["kelt_lower_15"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    for i in range(103, n):
        if i - last < 20:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        bull_walk = all(c[i - k] > ku[i - k] for k in range(3, 6))
        if bull_walk and c[i] <= ema20[i] + 0.3 * atr[i] and c[i] >= ema20[i] - 0.3 * atr[i]:
            e = c[i] + spread
            s = kl[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + 2.0 * risk
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t); last = i
            continue

        bear_walk = all(c[i - k] < kl[i - k] for k in range(3, 6))
        if bear_walk and c[i] >= ema20[i] - 0.3 * atr[i] and c[i] <= ema20[i] + 0.3 * atr[i]:
            e = c[i] - spread
            s = ku[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - 2.0 * risk
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


# ===================================================================
# Group 5: Price Action Patterns (4 strategies)
# ===================================================================


def signals_v3_hikkake(df, pair, pip, spread):
    """Inside bar false-break (Hikkake) pattern."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    atr = df["atr14"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    for i in range(102, n):
        if i - last < 15:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        mother_h = h[i - 2]
        mother_l = l[i - 2]
        ib_h = h[i - 1]
        ib_l = l[i - 1]
        if not (ib_h < mother_h and ib_l > mother_l):
            continue

        cur_h = h[i]
        cur_l = l[i]
        cur_c = c[i]

        if cur_h > mother_h and cur_c < mother_h:
            e = cur_c - spread
            s = cur_h + 0.3 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - 2.0 * risk
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t); last = i
        elif cur_l < mother_l and cur_c > mother_l:
            e = cur_c + spread
            s = cur_l - 0.3 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + 2.0 * risk
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


def signals_v3_mstar(df, pair, pip, spread):
    """Morning star / evening star 3-candle reversal pattern."""
    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    body = df["body"].values
    br = df["bar_range"].values
    atr = df["atr14"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    for i in range(102, n):
        if i - last < 15:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        body1 = body[i - 2]
        br1 = br[i - 2]
        body2 = body[i - 1]
        br2 = br[i - 1]
        body3 = body[i]

        if br1 < 0.3 * atr[i] or br2 <= 0:
            continue
        if body2 > 0.30 * br2:
            continue
        if body3 < 0.3 * atr[i]:
            continue

        mid1 = (o[i - 2] + c[i - 2]) / 2.0

        if c[i - 2] < o[i - 2] and c[i] > o[i] and c[i] > mid1:
            e = c[i] + spread
            star_low = l[i - 1]
            s = star_low - 0.3 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + body1
            if t <= e:
                t = e + risk
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t); last = i
        elif c[i - 2] > o[i - 2] and c[i] < o[i] and c[i] < mid1:
            e = c[i] - spread
            star_high = h[i - 1]
            s = star_high + 0.3 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - body1
            if t >= e:
                t = e - risk
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


def signals_v3_nr7(df, pair, pip, spread):
    """NR7 expansion breakout — smallest range in 7 bars, enter on next bar breakout."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    br = df["bar_range"].values
    atr = df["atr14"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    for i in range(107, n):
        if i - last < 15:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        nr_bar = i - 1
        nr_range = br[nr_bar]
        if nr_range <= 0:
            continue
        is_nr7 = True
        for k in range(1, 7):
            if br[nr_bar - k] <= nr_range:
                is_nr7 = False
                break
        if not is_nr7:
            continue

        long_entry = h[nr_bar] + 0.2 * atr[nr_bar]
        short_entry = l[nr_bar] - 0.2 * atr[nr_bar]
        nr_width = h[nr_bar] - l[nr_bar]

        if h[i] >= long_entry and c[i] > long_entry:
            e = long_entry + spread
            s = l[nr_bar]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + 2.0 * nr_width
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t); last = i
        elif l[i] <= short_entry and c[i] < short_entry:
            e = short_entry - spread
            s = h[nr_bar]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - 2.0 * nr_width
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


def signals_v3_brick(df, pair, pip, spread):
    """ATR brick trend — enter on pullback after 2 consecutive same-direction bricks."""
    c = df["close"].values
    atr = df["atr14"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    brick_levels = np.full(n, np.nan)
    brick_dirs = np.zeros(n, dtype=int)

    level = c[100]
    brick_size = atr[100] if not np.isnan(atr[100]) and atr[100] > 0 else 10 * pip
    consec = 0
    prev_dir = 0

    for i in range(100, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            brick_levels[i] = level
            brick_dirs[i] = prev_dir
            continue

        brick_size = atr[i]
        move = c[i] - level

        if move >= brick_size:
            num_bricks = int(move / brick_size)
            level += num_bricks * brick_size
            if prev_dir == 1:
                consec += num_bricks
            else:
                consec = num_bricks
                prev_dir = 1
        elif move <= -brick_size:
            num_bricks = int(abs(move) / brick_size)
            level -= num_bricks * brick_size
            if prev_dir == -1:
                consec += num_bricks
            else:
                consec = num_bricks
                prev_dir = -1

        brick_levels[i] = level
        brick_dirs[i] = prev_dir

        if i - last < 20:
            continue

        if consec >= 2 and prev_dir == 1:
            pullback_target = level
            if abs(c[i] - pullback_target) < 0.5 * brick_size and c[i] <= pullback_target + 0.2 * brick_size:
                e = c[i] + spread
                s = level - brick_size
                t = level + 2.0 * brick_size
                risk = abs(e - s)
                if risk < pip:
                    continue
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(s); tps.append(t); last = i
        elif consec >= 2 and prev_dir == -1:
            pullback_target = level
            if abs(c[i] - pullback_target) < 0.5 * brick_size and c[i] >= pullback_target - 0.2 * brick_size:
                e = c[i] - spread
                s = level + brick_size
                t = level - 2.0 * brick_size
                risk = abs(e - s)
                if risk < pip:
                    continue
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


# ===================================================================
# Group 6: Statistical / Correlation (3 strategies)
# ===================================================================


def signals_v3_zscore(df, pair, pip, spread):
    """Z-score mean reversion at +/- 2.0 extremes."""
    c = df["close"].values
    atr = df["atr14"].values
    zs = df["zscore_30"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    for i in range(100, n):
        if i - last < 15:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(zs[i]):
            continue

        if zs[i] < -2.0:
            e = c[i] + spread
            s = e - 1.5 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + 1.5 * risk
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t); last = i
        elif zs[i] > 2.0:
            e = c[i] - spread
            s = e + 1.5 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - 1.5 * risk
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


def signals_v3_pair_z(df, pair, pip, spread):
    """Pair-ratio z-score mean reversion (simplified single-pair version)."""
    c = df["close"].values
    atr = df["atr14"].values
    zs = df["zscore_30"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    log_c = np.log(c + 1e-10)
    z100 = np.full(n, np.nan)
    for i in range(100, n):
        window = log_c[i - 99 : i + 1]
        mu = np.mean(window)
        std = np.std(window)
        if std > 1e-10:
            z100[i] = (log_c[i] - mu) / std

    use_z100 = pair in ("EUR_USD", "GBP_USD")

    for i in range(100, n):
        if i - last < 15:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        z_val = z100[i] if use_z100 else zs[i]
        if np.isnan(z_val):
            continue

        if z_val < -2.0:
            e = c[i] + spread
            s = e - 1.5 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + 1.5 * risk
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t); last = i
        elif z_val > 2.0:
            e = c[i] - spread
            s = e + 1.5 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - 1.5 * risk
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


def signals_v3_hurst(df, pair, pip, spread):
    """Hurst regime switch — trend mode vs. mean-revert mode."""
    c = df["close"].values
    atr = df["atr14"].values
    hurst = df["hurst"].values
    st_dir = df["supertrend_dir"].values
    zs = df["zscore_30"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    for i in range(101, n):
        if i - last < 15:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(hurst[i]):
            continue

        if hurst[i] > 0.55:
            if np.isnan(st_dir[i]) or np.isnan(st_dir[i - 1]):
                continue
            if st_dir[i] == st_dir[i - 1]:
                continue

            d = 1 if st_dir[i] > 0 else -1
            e = c[i] + spread * d
            risk = 1.5 * atr[i]
            s = e - d * risk
            t = e + d * 2.0 * risk
            if risk < pip:
                continue
            idx.append(i); dirs.append(d); entries.append(e)
            sls.append(s); tps.append(t); last = i

        elif hurst[i] < 0.45:
            if np.isnan(zs[i]):
                continue
            if zs[i] < -2.0:
                e = c[i] + spread
                s = e - 1.5 * atr[i]
                risk = abs(e - s)
                if risk < pip:
                    continue
                t = e + 1.0 * risk
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(s); tps.append(t); last = i
            elif zs[i] > 2.0:
                e = c[i] - spread
                s = e + 1.5 * atr[i]
                risk = abs(e - s)
                if risk < pip:
                    continue
                t = e - 1.0 * risk
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


# ===================================================================
# Group 7: Session / Volume (3 strategies)
# ===================================================================


def signals_v3_orb(df, pair, pip, spread):
    """Opening range breakout — London open hours 7-8 UTC."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = df["atr14"].values
    hour = df["hour"].values
    date = df["date"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []

    day_ranges = {}
    for i in range(n):
        if hour[i] in (7, 8):
            d = date[i]
            if d not in day_ranges:
                day_ranges[d] = [h[i], l[i]]
            else:
                day_ranges[d][0] = max(day_ranges[d][0], h[i])
                day_ranges[d][1] = min(day_ranges[d][1], l[i])

    long_done = set()
    short_done = set()

    for i in range(100, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if hour[i] <= 8:
            continue

        d = date[i]
        if d not in day_ranges:
            continue

        rng_h, rng_l = day_ranges[d]
        rng_width = rng_h - rng_l
        if rng_width < pip:
            continue

        if d not in long_done and c[i] > rng_h:
            e = c[i] + spread
            s = rng_l
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + 1.5 * rng_width
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t)
            long_done.add(d)

        if d not in short_done and c[i] < rng_l:
            e = c[i] - spread
            s = rng_h
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - 1.5 * rng_width
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t)
            short_done.add(d)

    return idx, dirs, entries, sls, tps


def signals_v3_tky_ldn(df, pair, pip, spread):
    """Tokyo→London expansion — break of Asia range during London session."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    atr = df["atr14"].values
    hour = df["hour"].values
    date = df["date"].values
    body = df["body"].values
    br = df["bar_range"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []

    asia_ranges = {}
    for i in range(n):
        if 0 <= hour[i] <= 6:
            d = date[i]
            if d not in asia_ranges:
                asia_ranges[d] = [h[i], l[i]]
            else:
                asia_ranges[d][0] = max(asia_ranges[d][0], h[i])
                asia_ranges[d][1] = min(asia_ranges[d][1], l[i])

    done = set()

    for i in range(100, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if hour[i] < 7 or hour[i] > 12:
            continue

        d = date[i]
        if d in done or d not in asia_ranges:
            continue

        asia_h, asia_l = asia_ranges[d]
        asia_width = asia_h - asia_l
        if asia_width < pip:
            continue
        if br[i] <= 0 or body[i] < 0.5 * br[i]:
            continue

        if c[i] > asia_h and c[i] > o[i]:
            e = c[i] + spread
            s = asia_l
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + 1.5 * asia_width
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t)
            done.add(d)
        elif c[i] < asia_l and c[i] < o[i]:
            e = c[i] - spread
            s = asia_h
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - 1.5 * asia_width
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t)
            done.add(d)

    return idx, dirs, entries, sls, tps


def signals_v3_tvol(df, pair, pip, spread):
    """Tick volume climax reversal — high volume + long rejection wick."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    vol = df["volume"].values
    atr = df["atr14"].values
    br = df["bar_range"].values
    n = len(df)

    vol_sma20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_sma20[i] = np.mean(vol[i - 19 : i + 1])

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    for i in range(100, n):
        if i - last < 15:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(vol_sma20[i]):
            continue
        if br[i] <= 0:
            continue
        if vol[i] < 2.0 * vol_sma20[i]:
            continue

        top_body = max(o[i], c[i])
        bot_body = min(o[i], c[i])
        upper_wick = h[i] - top_body
        lower_wick = bot_body - l[i]

        if lower_wick > 0.60 * br[i]:
            e = c[i] + spread
            s = l[i] - 0.3 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e + 2.0 * risk
            idx.append(i); dirs.append(1); entries.append(e)
            sls.append(s); tps.append(t); last = i
        elif upper_wick > 0.60 * br[i]:
            e = c[i] - spread
            s = h[i] + 0.3 * atr[i]
            risk = abs(e - s)
            if risk < pip:
                continue
            t = e - 2.0 * risk
            idx.append(i); dirs.append(-1); entries.append(e)
            sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


# ===================================================================
# Group 8: Heikin-Ashi / Trend (2 strategies)
# ===================================================================


def signals_v3_ha_pure(df, pair, pip, spread):
    """HA trend-only — enter on continuation after opposite-color pause."""
    ha_o = df["ha_open"].values
    ha_c = df["ha_close"].values
    ha_h = df["ha_high"].values
    ha_l = df["ha_low"].values
    atr = df["atr14"].values
    n = len(df)

    tol = pip

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -30

    for i in range(105, n):
        if i - last < 20:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        bull_run = 0
        for k in range(3, 8):
            j = i - k
            if j < 0:
                break
            if ha_c[j] > ha_o[j] and (ha_l[j] - ha_o[j]) <= tol:
                bull_run += 1
            else:
                break
        if bull_run >= 3:
            pause = i - 2
            cont = i - 1
            if ha_c[pause] < ha_o[pause] and ha_c[cont] > ha_o[cont]:
                e = ha_c[i - 1] + spread
                cluster_low = min(ha_l[i - k] for k in range(1, bull_run + 3) if i - k >= 0)
                s = cluster_low - 0.3 * atr[i]
                risk = abs(e - s)
                if risk < pip:
                    continue
                t = e + 2.0 * risk
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(s); tps.append(t); last = i
                continue

        bear_run = 0
        for k in range(3, 8):
            j = i - k
            if j < 0:
                break
            if ha_c[j] < ha_o[j] and (ha_o[j] - ha_h[j]) <= tol:
                bear_run += 1
            else:
                break
        if bear_run >= 3:
            pause = i - 2
            cont = i - 1
            if ha_c[pause] > ha_o[pause] and ha_c[cont] < ha_o[cont]:
                e = ha_c[i - 1] - spread
                cluster_high = max(ha_h[i - k] for k in range(1, bear_run + 3) if i - k >= 0)
                s = cluster_high + 0.3 * atr[i]
                risk = abs(e - s)
                if risk < pip:
                    continue
                t = e - 2.0 * risk
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(s); tps.append(t); last = i

    return idx, dirs, entries, sls, tps


def signals_v3_ha_adx(df, pair, pip, spread):
    """HA + ADX trend entry on first aligned bar transition."""
    ha_o = df["ha_open"].values
    ha_c = df["ha_close"].values
    atr = df["atr14"].values
    adx = df["adx"].values
    pdi = df["plus_di"].values
    mdi = df["minus_di"].values
    n = len(df)

    idx, dirs, entries, sls, tps = [], [], [], [], []
    last = -20

    prev_aligned_bull = False
    prev_aligned_bear = False

    for i in range(100, n):
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(adx[i]):
            prev_aligned_bull = False
            prev_aligned_bear = False
            continue

        ha_bull = ha_c[i] > ha_o[i]
        ha_bear = ha_c[i] < ha_o[i]
        strong = adx[i] > 25.0

        curr_bull = ha_bull and strong and pdi[i] > mdi[i]
        curr_bear = ha_bear and strong and mdi[i] > pdi[i]

        if curr_bull and not prev_aligned_bull and i - last >= 20:
            e = ha_c[i] + spread
            risk = 1.5 * atr[i]
            s = e - risk
            t = e + 2.0 * risk
            if risk >= pip:
                idx.append(i); dirs.append(1); entries.append(e)
                sls.append(s); tps.append(t); last = i

        elif curr_bear and not prev_aligned_bear and i - last >= 20:
            e = ha_c[i] - spread
            risk = 1.5 * atr[i]
            s = e + risk
            t = e - 2.0 * risk
            if risk >= pip:
                idx.append(i); dirs.append(-1); entries.append(e)
                sls.append(s); tps.append(t); last = i

        prev_aligned_bull = curr_bull
        prev_aligned_bear = curr_bear

    return idx, dirs, entries, sls, tps


# ===================================================================
# Registry
# ===================================================================

STRATEGIES_V3_PART2 = [
    # Group 4 continued: Channel / Breakout
    ("V3_DONCH_VOL",  signals_v3_donch_vol,  40),
    ("V3_KELT_ADX",   signals_v3_kelt_adx,   40),
    ("V3_KELT_WALK",  signals_v3_kelt_walk,   40),
    # Group 5: Price Action Patterns
    ("V3_HIKKAKE",    signals_v3_hikkake,     40),
    ("V3_MSTAR",      signals_v3_mstar,       40),
    ("V3_NR7",        signals_v3_nr7,         40),
    ("V3_BRICK",      signals_v3_brick,       40),
    # Group 6: Statistical / Correlation
    ("V3_ZSCORE",     signals_v3_zscore,      40),
    ("V3_PAIR_Z",     signals_v3_pair_z,      40),
    ("V3_HURST",      signals_v3_hurst,       40),
    # Group 7: Session / Volume
    ("V3_ORB",        signals_v3_orb,         60),
    ("V3_TKY_LDN",    signals_v3_tky_ldn,     60),
    ("V3_TVOL",       signals_v3_tvol,        40),
    # Group 8: Heikin-Ashi / Trend
    ("V3_HA_PURE",    signals_v3_ha_pure,     40),
    ("V3_HA_ADX",     signals_v3_ha_adx,      40),
]
