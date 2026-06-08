"""Refined backtest — tightened filters on the 5 most promising strategies.

S11_RSI: Best PF, tighten RSI thresholds
S15_VWAP: Profitable combos, tighten deviation threshold
S9_FBR: PF 0.83, add volume/body confirmation
S10_WDG: PF 0.86, add stronger divergence requirement
S3_SB: PF 0.78, tighten M-W pattern matching

Also: session filters (London/NY only), stricter cooldowns, pair-specific tuning.
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

ACTIVE_HOURS = set(range(7, 17))  # London + early NY only


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


def resample_m5(df):
    groups = len(df) // 5
    trimmed = df.iloc[:groups * 5].copy()
    trimmed["g"] = np.repeat(range(groups), 5)
    m5 = trimmed.groupby("g").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "timestamp": "last",
    }).reset_index(drop=True)
    return m5


def add_indicators(df, pair):
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    n = len(df)
    pip = 0.01 if "JPY" in pair else 0.0001

    df["ema20"] = pd.Series(c).ewm(span=20, adjust=False).mean().values

    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    df["atr14"] = pd.Series(tr).rolling(14).mean().values

    sma20 = pd.Series(c).rolling(20).mean().values
    std20 = pd.Series(c).rolling(20).std().values
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"] = sma20

    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).values

    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    df["pip"] = pip

    slope = np.zeros(n)
    slope[10:] = df["ema20"].values[10:] - df["ema20"].values[:-10]
    df["ema20_slope"] = slope

    return df


def simulate_trades(highs, lows, closes, indices, dirs, entries, sls, tps, max_bars, pip):
    n_data = len(highs)
    results = []
    for k in range(len(indices)):
        idx = indices[k]; d = dirs[k]; entry = entries[k]; sl = sls[k]; tp = tps[k]
        end = min(idx + max_bars, n_data)
        exit_r = "time"; pnl = 0.0; bars = end - idx

        for j in range(idx + 1, end):
            if d == 1:
                if lows[j] <= sl:
                    exit_r = "sl"; pnl = sl - entry; bars = j - idx; break
                if highs[j] >= tp:
                    exit_r = "tp"; pnl = tp - entry; bars = j - idx; break
            else:
                if highs[j] >= sl:
                    exit_r = "sl"; pnl = entry - sl; bars = j - idx; break
                if lows[j] <= tp:
                    exit_r = "tp"; pnl = entry - tp; bars = j - idx; break
        else:
            c_exit = closes[min(end - 1, n_data - 1)]
            pnl = (c_exit - entry) if d == 1 else (entry - c_exit)

        results.append({"exit": exit_r, "pnl_pips": round(pnl / pip, 2), "bars": bars})
    return results


# ---------------------------------------------------------------------------
#  REFINED STRATEGIES
# ---------------------------------------------------------------------------

def s11_rsi_v2(df, pair, pip, spread):
    """RSI/BB Mean Reversion — tighter thresholds, session filter, confirmation bar."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    rsi = df["rsi14"].values
    bb_u = df["bb_upper"].values; bb_l = df["bb_lower"].values; bb_m = df["bb_mid"].values
    slope = df["ema20_slope"].values
    hours = df["hour"].values; dow = df["dow"].values
    n = len(df)

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -30

    for i in range(30, n - 12):
        if i - last < 30:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(rsi[i]) or np.isnan(bb_u[i]):
            continue
        if hours[i] not in ACTIVE_HOURS or dow[i] >= 5:
            continue

        br = h[i] - l[i]
        if br <= 0:
            continue
        body = abs(c[i] - o[i])

        # Stronger extremes
        if rsi[i] < 20 and c[i] < bb_l[i]:
            if abs(slope[i]) > atr[i] * 0.15 and slope[i] < 0:
                continue
            # Require reversal bar (close in upper half)
            if c[i] < (l[i] + h[i]) / 2:
                continue
            e = c[i] + spread; s = l[i] - atr[i] * 0.5; t = float(bb_m[i])
            if t <= e:
                continue
            d = 1
        elif rsi[i] > 80 and c[i] > bb_u[i]:
            if abs(slope[i]) > atr[i] * 0.15 and slope[i] > 0:
                continue
            if c[i] > (l[i] + h[i]) / 2:
                continue
            e = c[i] - spread; s = h[i] + atr[i] * 0.5; t = float(bb_m[i])
            if t >= e:
                continue
            d = -1
        else:
            continue

        sl_d = abs(e - s)
        tp_d = abs(e - t)
        if sl_d > 20 * pip or sl_d < 1 * pip or tp_d < sl_d * 1.2:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def s15_vwap_v2(df_m1, pair, pip, spread):
    """VWAP Reversion — tighter deviation, session-only, better R:R."""
    m5 = resample_m5(df_m1)
    m5 = add_indicators(m5, pair)
    c = m5["close"].values; h = m5["high"].values; l = m5["low"].values
    vol = m5["volume"].values if "volume" in m5.columns else np.zeros(len(m5))
    atr = m5["atr14"].values
    hours = m5["hour"].values; dates = m5["date"].values; ts = m5["timestamp"].values
    dow = m5["dow"].values
    n = len(m5)

    idx, dirs, ent, sls, tps = [], [], [], [], []
    hrs_out, ts_out = [], []
    last = -15; cur_date = None; sess_start = 0

    for i in range(20, n - 60):
        if i - last < 15:
            continue
        hr = hours[i]
        if hr not in ACTIVE_HOURS or dow[i] >= 5:
            cur_date = None; continue
        d = dates[i]
        if d != cur_date:
            sess_start = i; cur_date = d
        if i - sess_start < 12:
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
        if abs(dev / a) < 3.0:  # tighter: was 2.5
            continue

        sl_dist = a * 0.6  # slightly wider stop
        if dev > 0:
            e = c[i] - spread / 2; s = e + sl_dist; t = float(vwap); direction = -1
        else:
            e = c[i] + spread / 2; s = e - sl_dist; t = float(vwap); direction = 1

        if direction == 1 and t <= e:
            continue
        if direction == -1 and t >= e:
            continue

        tp_d = abs(e - t)
        if tp_d < sl_dist * 1.5:
            continue

        idx.append(i); dirs.append(direction); ent.append(e); sls.append(s); tps.append(t)
        hrs_out.append(hr); ts_out.append(ts[i])
        last = i

    return idx, dirs, ent, sls, tps, h, l, c, hrs_out, ts_out, n


def s9_fbr_v2(df, pair, pip, spread):
    """Failed Breakout Reversal — session filter, stronger confirmation, wider lookback."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    hours = df["hour"].values; dow = df["dow"].values
    n = len(df)

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -40

    for i in range(50, n - 12):
        if i - last < 40:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if hours[i] not in ACTIVE_HOURS or dow[i] >= 5:
            continue

        prev_h = h[i-50:i-3]; prev_l = l[i-50:i-3]
        if len(prev_h) < 20:
            continue
        sw_hi = np.max(prev_h); sw_lo = np.min(prev_l)

        broke_hi = h[i-1] > sw_hi and h[i-1] - sw_hi < 3 * pip
        broke_lo = l[i-1] < sw_lo and sw_lo - l[i-1] < 3 * pip

        if not (broke_hi or broke_lo):
            continue

        br = h[i] - l[i]
        if br <= 0:
            continue

        # Require strong reversal bar
        if broke_hi:
            if not (c[i] < sw_hi and c[i] < o[i]):
                continue
            body_pct = (o[i] - c[i]) / br
            if body_pct < 0.5:
                continue
            # Close below midpoint of breakout bar
            if c[i] > (h[i-1] + l[i-1]) / 2:
                continue
            e = c[i] - spread; s = h[i-1] + 2*pip; t = e - 2*abs(e-s); d = -1
        else:
            if not (c[i] > sw_lo and c[i] > o[i]):
                continue
            body_pct = (c[i] - o[i]) / br
            if body_pct < 0.5:
                continue
            if c[i] < (h[i-1] + l[i-1]) / 2:
                continue
            e = c[i] + spread; s = l[i-1] - 2*pip; t = e + 2*abs(e-s); d = 1

        sl_d = abs(e - s)
        if sl_d > 12 * pip or sl_d < 2 * pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


def s10_wedge_v2(df, pair, pip, spread):
    """Wedge/Three-Push — session filter, stronger divergence, stricter shape."""
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    atr = df["atr14"].values; rsi = df["rsi14"].values
    hours = df["hour"].values; dow = df["dow"].values
    n = len(df)

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -50
    w = 30

    for i in range(w + 5, n - 12):
        if i - last < 50:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(rsi[i]):
            continue
        if hours[i] not in ACTIVE_HOURS or dow[i] >= 5:
            continue

        seg_h = h[i-w:i+1]; seg_l = l[i-w:i+1]; seg_rsi = rsi[i-w:i+1]

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
                p1 = sh[1][1] - sh[0][1]; p2 = sh[2][1] - sh[1][1]
                # Stricter: diminishing pushes AND RSI divergence >= 5 points
                if 0 < p2 < p1 * 0.8 and sh[2][2] < sh[1][2] - 5:
                    if c[i] < o[i] and (o[i]-c[i]) > 0.5*(h[i]-l[i]):
                        e = c[i] - spread
                        s = h[i-w+sh[2][0]] + 2*pip
                        t = e - 2.5*abs(e-s)
                        sl_d = abs(e - s)
                        if 2*pip < sl_d < 12*pip:
                            idx.append(i); dirs.append(-1)
                            ent.append(e); sls.append(s); tps.append(t)
                            last = i; found = True

        if not found and len(swing_lows) >= 3:
            sl_pts = swing_lows[-3:]
            if sl_pts[0][1] > sl_pts[1][1] > sl_pts[2][1]:
                p1 = sl_pts[0][1] - sl_pts[1][1]; p2 = sl_pts[1][1] - sl_pts[2][1]
                if 0 < p2 < p1 * 0.8 and sl_pts[2][2] > sl_pts[1][2] + 5:
                    if c[i] > o[i] and (c[i]-o[i]) > 0.5*(h[i]-l[i]):
                        e = c[i] + spread
                        s = l[i-w+sl_pts[2][0]] - 2*pip
                        t = e + 2.5*abs(e-s)
                        sl_d = abs(e - s)
                        if 2*pip < sl_d < 12*pip:
                            idx.append(i); dirs.append(1)
                            ent.append(e); sls.append(s); tps.append(t)
                            last = i

    return idx, dirs, ent, sls, tps


def s3_sb_v2(df, pair, pip, spread):
    """Second Break — session filter, tighter pattern, bigger body confirmation."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values
    ema20 = df["ema20"].values; atr = df["atr14"].values
    slope = df["ema20_slope"].values
    hours = df["hour"].values; dow = df["dow"].values
    n = len(df)

    idx, dirs, ent, sls, tps = [], [], [], [], []
    last = -40

    for i in range(40, n - 12):
        if i - last < 40:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if hours[i] not in ACTIVE_HOURS or dow[i] >= 5:
            continue
        if abs(slope[i]) < atr[i] * 0.08:
            continue

        bull = slope[i] > 0
        w = 20

        if bull:
            # Require confirmation candle
            if not (c[i] > o[i] and (c[i]-o[i]) > 0.4*(h[i]-l[i])):
                continue
            rl = l[i-w:i+1]
            idx_s = np.argsort(rl)
            l1p, l2p = idx_s[0], idx_s[1]
            if abs(l1p - l2p) < 4:
                continue
            lv1, lv2 = rl[l1p], rl[l2p]
            if abs(lv1 - lv2) > 2 * pip:
                continue
            mid_seg = h[i-w+min(l1p,l2p):i-w+max(l1p,l2p)+1]
            if len(mid_seg) == 0 or np.max(mid_seg) - max(lv1,lv2) < 4 * pip:
                continue
            if abs(c[i] - ema20[i]) > atr[i]:
                continue
            e = c[i] + spread; s = min(lv1,lv2) - pip; t = e + 10*pip; d = 1
        else:
            if not (c[i] < o[i] and (o[i]-c[i]) > 0.4*(h[i]-l[i])):
                continue
            rh = h[i-w:i+1]
            idx_s = np.argsort(-rh)
            h1p, h2p = idx_s[0], idx_s[1]
            if abs(h1p - h2p) < 4:
                continue
            hv1, hv2 = rh[h1p], rh[h2p]
            if abs(hv1 - hv2) > 2 * pip:
                continue
            mid_seg = l[i-w+min(h1p,h2p):i-w+max(h1p,h2p)+1]
            if len(mid_seg) == 0 or min(hv1,hv2) - np.min(mid_seg) < 4 * pip:
                continue
            if abs(c[i] - ema20[i]) > atr[i]:
                continue
            e = c[i] - spread; s = max(hv1,hv2) + pip; t = e - 10*pip; d = -1

        sl_d = abs(e - s)
        if sl_d > 10 * pip or sl_d < 2 * pip:
            continue

        idx.append(i); dirs.append(d); ent.append(e); sls.append(s); tps.append(t)
        last = i

    return idx, dirs, ent, sls, tps


STRATEGIES = [
    ("S11v2_RSI",  s11_rsi_v2,    12, False),
    ("S15v2_VWAP", s15_vwap_v2,   60, True),
    ("S9v2_FBR",   s9_fbr_v2,     12, False),
    ("S10v2_WDG",  s10_wedge_v2,  12, False),
    ("S3v2_SB",    s3_sb_v2,      12, False),
]


def run_strat(name, func, df, df_m1, pair, pip, spread, mb, uses_m1):
    if uses_m1:
        result = func(df_m1, pair, pip, spread)
        idx, dirs, ent, sls, tps, h_a, l_a, c_a, hrs, tss, n_d = result
        if not idx:
            return []
        sim = simulate_trades(h_a, l_a, c_a, idx, dirs, ent, sls, tps, mb, pip)
        trades = []
        for k in range(len(idx)):
            trades.append({
                "strategy": name, "pair": pair,
                "direction": "long" if dirs[k] == 1 else "short",
                "hour": int(hrs[k]),
                "pnl_pips": sim[k]["pnl_pips"], "exit_reason": sim[k]["exit"],
                "bars_held": sim[k]["bars"],
                "sl_pips": round(abs(ent[k]-sls[k])/pip, 1),
                "tp_pips": round(abs(ent[k]-tps[k])/pip, 1),
                "timestamp": str(tss[k]),
            })
        return trades

    idx, dirs, ent, sls, tps = func(df, pair, pip, spread)
    if not idx:
        return []
    h_a = df["high"].values; l_a = df["low"].values; c_a = df["close"].values
    hours = df["hour"].values; ts = df["timestamp"].values
    sim = simulate_trades(h_a, l_a, c_a, idx, dirs, ent, sls, tps, mb, pip)
    trades = []
    for k in range(len(idx)):
        trades.append({
            "strategy": name, "pair": pair,
            "direction": "long" if dirs[k] == 1 else "short",
            "hour": int(hours[idx[k]]),
            "pnl_pips": sim[k]["pnl_pips"], "exit_reason": sim[k]["exit"],
            "bars_held": sim[k]["bars"],
            "sl_pips": round(abs(ent[k]-sls[k])/pip, 1),
            "tp_pips": round(abs(ent[k]-tps[k])/pip, 1),
            "timestamp": str(ts[idx[k]]),
        })
    return trades


def print_report(all_trades):
    if not all_trades:
        print("  No trades."); return

    print(f"\n{'='*90}")
    print(f"  REFINED BACKTEST — {len(all_trades):,} trades")
    print(f"{'='*90}")

    total = sum(t["pnl_pips"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["pnl_pips"] > 0)
    wr = wins / len(all_trades) * 100
    gp = sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] > 0)
    gl = abs(sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] <= 0))
    pf = gp / gl if gl else 999

    print(f"\n  Overall: {wins:,}W / {len(all_trades)-wins:,}L | WR {wr:.1f}% | PnL {total:+,.0f} pips | PF {pf:.2f}")

    print(f"\n  {'Strategy':14s} {'Trades':>8s} {'WR':>7s} {'PnL':>12s} {'Avg':>8s} {'PF':>7s} {'MaxDD':>8s}")
    print(f"  {'-'*72}")
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
        avg = p/len(st)
        # Max drawdown
        eq = np.cumsum([t["pnl_pips"] for t in st])
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        mdd = np.max(dd) if len(dd) > 0 else 0
        print(f"  {s:14s} {len(st):8,d} {w/len(st)*100:6.1f}% {p:+12,.0f} {avg:+8.2f} {pf_v:7.2f} {mdd:8,.0f}")

    print(f"\n  TOP 30 COMBOS:")
    print(f"  {'Combo':38s} {'N':>6s} {'WR':>7s} {'PnL':>10s} {'Avg':>7s} {'PF':>6s}")
    combos = defaultdict(list)
    for t in all_trades:
        combos[f"{t['strategy']} {t['pair']} {t['direction']}"].append(t)
    ranked = []
    for k, ct in combos.items():
        if len(ct) < 10:
            continue
        w = sum(1 for t in ct if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in ct)
        g = sum(t["pnl_pips"] for t in ct if t["pnl_pips"] > 0)
        lo = abs(sum(t["pnl_pips"] for t in ct if t["pnl_pips"] <= 0))
        ranked.append((k, len(ct), w/len(ct)*100, p, p/len(ct), g/lo if lo else 999))

    print("  --- PROFITABLE ---")
    for k, n, w, p, a, pf_v in sorted(ranked, key=lambda x: -x[3])[:30]:
        tag = " ***" if pf_v > 1.2 and n > 50 else ""
        print(f"  {k:38s} {n:6d} {w:6.1f}% {p:+10,.0f} {a:+7.2f} {pf_v:6.2f}{tag}")
    print("  --- UNPROFITABLE ---")
    for k, n, w, p, a, pf_v in sorted(ranked, key=lambda x: x[3])[:15]:
        print(f"  {k:38s} {n:6d} {w:6.1f}% {p:+10,.0f} {a:+7.2f} {pf_v:6.2f}")

    print(f"\n  BY YEAR (all strategies):")
    yrs = defaultdict(list)
    for t in all_trades:
        yrs[t["timestamp"][:4]].append(t)
    print(f"  {'Year':>6s} {'N':>7s} {'WR':>7s} {'PnL':>10s} {'PF':>6s}")
    for y in sorted(yrs):
        yt = yrs[y]; w = sum(1 for t in yt if t["pnl_pips"]>0)
        p = sum(t["pnl_pips"] for t in yt)
        g = sum(t["pnl_pips"] for t in yt if t["pnl_pips"]>0)
        lo = abs(sum(t["pnl_pips"] for t in yt if t["pnl_pips"]<=0))
        print(f"  {y:>6s} {len(yt):7,d} {w/len(yt)*100:6.1f}% {p:+10,.0f} {g/lo if lo else 999:6.2f}")


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
    with open(OUTPUT_DIR/"refined_combo_stats.json","w") as f:
        json.dump(dict(combo),f,indent=2,sort_keys=True)

    lessons = []
    for key, s in sorted(combo.items(), key=lambda x:-x[1]["pnl"]):
        if s["trades"]<30: continue
        if s["win_rate"]>=50 and s["avg_pips"]>0.3:
            lessons.append({"theme":f"REFINED-EDGE: {key} — {s['win_rate']}% WR, {s['trades']} trades, avg {s['avg_pips']:+.2f}p",
                            "confidence":min(0.95,0.5+s["win_rate"]/200),"source":"backtest_refined","pinned":True})
        elif s["win_rate"]<42 and s["avg_pips"]<-0.5:
            lessons.append({"theme":f"REFINED-AVOID: {key} — {s['win_rate']}% WR, {s['trades']} trades, avg {s['avg_pips']:+.2f}p",
                            "confidence":min(0.95,0.5+(100-s["win_rate"])/200),"source":"backtest_refined","pinned":True})
    with open(OUTPUT_DIR/"refined_lessons.jsonl","w") as f:
        for le in lessons:
            f.write(json.dumps(le)+"\n")

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
    with open(OUTPUT_DIR/"refined_summary.json","w") as f:
        json.dump(strat_sum,f,indent=2,sort_keys=True)

    print(f"\n  Saved: refined_combo_stats.json, refined_summary.json, refined_lessons.jsonl ({len(lessons)} lessons)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=0)
    args = parser.parse_args()

    print("="*90)
    print(f"  REFINED 5-STRATEGY BACKTEST" + (f" (last {args.years}yr)" if args.years else " (full history)"))
    print("="*90)

    all_trades = []
    t0 = time_mod.time()

    for fb, pair in PAIR_MAP.items():
        fn = f"{fb}_1min.csv"
        print(f"\n  {pair} loading...", end=" ", flush=True)
        df_m1 = load_pair(fn, pair, args.years)
        if df_m1.empty:
            print("SKIP"); continue
        print(f"{len(df_m1):,} M1", end="", flush=True)

        m5 = resample_m5(df_m1)
        m5 = add_indicators(m5, pair)
        pip = 0.01 if "JPY" in pair else 0.0001
        spread = SPREAD_PIPS.get(pair, 0.5) * pip
        print(f" → {len(m5):,} M5", end="", flush=True)

        for sname, sfunc, mb, uses_m1 in STRATEGIES:
            print(f" {sname}", end="", flush=True)
            try:
                trades = run_strat(sname, sfunc, m5, df_m1, pair, pip, spread, mb, uses_m1)
                print(f":{len(trades)}", end="", flush=True)
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
