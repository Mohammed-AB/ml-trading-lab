"""Comprehensive backtest on 1-MINUTE data — Models C, D, E.

Reads from ~/Downloads/FX-1-Minute-Data-master/forex_data/1min/
Model C uses native M1 bars (its designed timeframe).
Models D and E resample to M5 internally for ATR/VWAP.

Usage:
    python3 backtest_1m.py
    python3 backtest_1m.py --years 5
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


def load_pair(filename: str, pair: str, years: int = 0) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
    if years > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=years * 365)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


def add_m1_indicators(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    c = df["close"]
    h, l = df["high"], df["low"]

    df["ema5"] = c.ewm(span=5, adjust=False).mean()
    df["ema40"] = c.ewm(span=40, adjust=False).mean()

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    pip_size = 0.01 if "JPY" in pair else 0.0001
    df["atr_pips"] = df["atr14"] / pip_size
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    return df


def resample_m5(df: pd.DataFrame) -> pd.DataFrame:
    """Resample M1 to M5 for models that need it."""
    groups = len(df) // 5
    trimmed = df.iloc[:groups * 5].copy()
    trimmed["g"] = np.repeat(range(groups), 5)

    m5 = trimmed.groupby("g").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
        "timestamp": "last", "hour": "last", "dow": "last", "date": "last",
    }).reset_index(drop=True)

    c, h, l = m5["close"], m5["high"], m5["low"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    m5["atr14"] = tr.rolling(14).mean()
    return m5


def simulate_trade(highs, lows, closes, entry_idx: int, n: int,
                   direction: str, entry: float, sl: float, tp: float,
                   max_bars: int) -> dict:
    end = min(entry_idx + max_bars, n)
    for j in range(entry_idx + 1, end):
        if direction == "long":
            if lows[j] <= sl:
                return {"exit_reason": "sl_hit", "pnl": sl - entry, "bars": j - entry_idx}
            if highs[j] >= tp:
                return {"exit_reason": "tp_hit", "pnl": tp - entry, "bars": j - entry_idx}
        else:
            if highs[j] >= sl:
                return {"exit_reason": "sl_hit", "pnl": entry - sl, "bars": j - entry_idx}
            if lows[j] <= tp:
                return {"exit_reason": "tp_hit", "pnl": entry - tp, "bars": j - entry_idx}

    c_exit = closes[min(end - 1, n - 1)]
    pnl = (c_exit - entry) if direction == "long" else (entry - c_exit)
    return {"exit_reason": "time_stop", "pnl": pnl, "bars": end - entry_idx}


def backtest_model_c_m1(df: pd.DataFrame, pair: str) -> list:
    """Model C on its native M1 timeframe."""
    pip_size = 0.01 if "JPY" in pair else 0.0001
    spread = SPREAD_PIPS.get(pair, 0.5) * pip_size
    atr_mult = 1.3549
    rr = 0.5009

    ema5 = df["ema5"].values
    ema40 = df["ema40"].values
    atr = df["atr14"].values
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    hours = df["hour"].values
    n = len(df)

    cross_up = np.zeros(n, dtype=bool)
    cross_down = np.zeros(n, dtype=bool)
    cross_up[1:] = (ema5[:-1] <= ema40[:-1]) & (ema5[1:] > ema40[1:])
    cross_down[1:] = (ema5[:-1] >= ema40[:-1]) & (ema5[1:] < ema40[1:])

    signal_indices = np.where(cross_up | cross_down)[0]

    trades = []
    last_idx = -10
    ts_arr = df["timestamp"].values

    for idx in signal_indices:
        if idx - last_idx < 10:
            continue
        if idx >= n - 120:
            continue
        if np.isnan(atr[idx]) or atr[idx] <= 0:
            continue

        direction = "long" if cross_up[idx] else "short"
        sl_dist = atr[idx] * atr_mult
        tp_dist = sl_dist * rr

        if direction == "long":
            entry = close_arr[idx] + spread / 2
            sl_p = entry - sl_dist
            tp_p = entry + tp_dist
        else:
            entry = close_arr[idx] - spread / 2
            sl_p = entry + sl_dist
            tp_p = entry - tp_dist

        result = simulate_trade(high_arr, low_arr, close_arr, idx, n,
                                direction, entry, sl_p, tp_p, max_bars=120)

        trades.append({
            "model": "C", "pair": pair, "direction": direction,
            "hour": int(hours[idx]),
            "pnl_pips": round(result["pnl"] / pip_size, 2),
            "exit_reason": result["exit_reason"],
            "bars_held": result["bars"],
            "sl_pips": round(sl_dist / pip_size, 1),
            "tp_pips": round(tp_dist / pip_size, 1),
            "timestamp": str(ts_arr[idx]),
        })
        last_idx = idx

    return trades


def backtest_model_d_m1(df: pd.DataFrame, pair: str) -> list:
    """Model D on M1 — uses M1 closes for 5-bar move, M5 ATR for gating."""
    supported = {"EUR_USD", "GBP_USD", "AUD_USD", "EUR_GBP",
                 "USD_CHF", "USD_CAD", "NZD_USD"}
    if pair not in supported:
        return []

    pip_size = 0.01 if "JPY" in pair else 0.0001
    spread = SPREAD_PIPS.get(pair, 0.5) * pip_size
    tp_pips, sl_pips = 4.0, 5.0
    trigger_pips = 10.0

    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    hours = df["hour"].values
    dows = df["dow"].values
    atr_p = df["atr_pips"].values
    n = len(df)
    ts_arr = df["timestamp"].values

    prev5_move = np.zeros(n)
    prev5_move[5:] = (close_arr[5:] - close_arr[:-5]) / pip_size

    trades = []
    last_idx = -15

    for i in range(6, n - 30):
        if i - last_idx < 15:
            continue
        if dows[i] >= 5:
            continue
        if np.isnan(atr_p[i]) or atr_p[i] < 3.0:
            continue

        h = hours[i]
        mv = prev5_move[i]
        direction = None

        if h in (0, 7, 20) and mv <= -trigger_pips:
            direction = "long"
            entry = close_arr[i] + spread / 2
            sl_p = entry - sl_pips * pip_size
            tp_p = entry + tp_pips * pip_size
        elif h in (19, 22) and mv >= trigger_pips:
            direction = "short"
            entry = close_arr[i] - spread / 2
            sl_p = entry + sl_pips * pip_size
            tp_p = entry - tp_pips * pip_size

        if direction is None:
            continue

        result = simulate_trade(high_arr, low_arr, close_arr, i, n,
                                direction, entry, sl_p, tp_p, max_bars=75)

        trades.append({
            "model": "D", "pair": pair, "direction": direction,
            "hour": int(h),
            "pnl_pips": round(result["pnl"] / pip_size, 2),
            "exit_reason": result["exit_reason"],
            "bars_held": result["bars"],
            "sl_pips": sl_pips, "tp_pips": tp_pips,
            "timestamp": str(ts_arr[i]),
        })
        last_idx = i

    return trades


def backtest_model_e_m5(m5: pd.DataFrame, pair: str) -> list:
    """Model E on resampled M5 — VWAP reversion during London/NY."""
    pip_size = 0.01 if "JPY" in pair else 0.0001
    spread = SPREAD_PIPS.get(pair, 0.5) * pip_size
    ext_atr = 2.5
    sl_atr_mult = 0.5
    min_bars = 10
    active = set(range(7, 16))

    close_arr = m5["close"].values
    high_arr = m5["high"].values
    low_arr = m5["low"].values
    atr = m5["atr14"].values
    vol = m5["volume"].values if "volume" in m5.columns else np.zeros(len(m5))
    hours = m5["hour"].values
    dates = m5["date"].values
    n = len(m5)
    ts_arr = m5["timestamp"].values

    trades = []
    last_idx = -13
    current_date = None
    session_start = 0

    for i in range(20, n - 60):
        if i - last_idx < 12:
            continue
        h = hours[i]
        if h not in active:
            current_date = None
            continue

        d = dates[i]
        if d != current_date:
            session_start = i
            current_date = d

        if i - session_start < min_bars:
            continue

        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        sess_c = close_arr[session_start:i + 1]
        sess_h = high_arr[session_start:i + 1]
        sess_l = low_arr[session_start:i + 1]
        sess_v = vol[session_start:i + 1]

        if sess_v.sum() > 0:
            typical = (sess_h + sess_l + sess_c) / 3.0
            vwap = (typical * sess_v).sum() / sess_v.sum()
        else:
            vwap = sess_c.mean()

        dev = close_arr[i] - vwap
        if abs(dev / a) < ext_atr:
            continue

        sl_dist = a * sl_atr_mult
        if dev > 0:
            direction = "short"
            entry = close_arr[i] - spread / 2
            sl_p = entry + sl_dist
            tp_p = float(vwap)
        else:
            direction = "long"
            entry = close_arr[i] + spread / 2
            sl_p = entry - sl_dist
            tp_p = float(vwap)

        if direction == "long" and tp_p <= entry:
            continue
        if direction == "short" and tp_p >= entry:
            continue

        result = simulate_trade(high_arr, low_arr, close_arr, i, n,
                                direction, entry, sl_p, tp_p, max_bars=60)

        trades.append({
            "model": "E", "pair": pair, "direction": direction,
            "hour": int(h),
            "pnl_pips": round(result["pnl"] / pip_size, 2),
            "exit_reason": result["exit_reason"],
            "bars_held": result["bars"],
            "sl_pips": round(sl_dist / pip_size, 1),
            "tp_pips": round(abs(tp_p - entry) / pip_size, 1),
            "timestamp": str(ts_arr[i]),
        })
        last_idx = i

    return trades


def print_report(all_trades: list):
    if not all_trades:
        print("  No trades.")
        return

    print(f"\n{'='*80}")
    print(f"  1-MINUTE BACKTEST RESULTS — {len(all_trades)} trades")
    print(f"{'='*80}")

    total_pips = sum(t["pnl_pips"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["pnl_pips"] > 0)
    wr = wins / len(all_trades) * 100
    gp = sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] > 0)
    gl = abs(sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] <= 0))
    pf = gp / gl if gl > 0 else 999

    print(f"\n  Overall: {wins}W / {len(all_trades)-wins}L | WR: {wr:.1f}% | PnL: {total_pips:+.0f} pips | PF: {pf:.2f}")

    print(f"\n  {'Model':6s} {'Trades':>8s} {'WR':>7s} {'PnL':>12s} {'Avg':>8s} {'PF':>6s}")
    for m in sorted(set(t["model"] for t in all_trades)):
        mt = [t for t in all_trades if t["model"] == m]
        w = sum(1 for t in mt if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in mt)
        g = sum(t["pnl_pips"] for t in mt if t["pnl_pips"] > 0)
        lo = abs(sum(t["pnl_pips"] for t in mt if t["pnl_pips"] <= 0))
        print(f"  {m:6s} {len(mt):8d} {w/len(mt)*100:6.1f}% {p:+12.0f} {p/len(mt):+8.2f} {g/lo if lo else 999:6.2f}")

    print(f"\n  {'Combo':32s} {'N':>6s} {'WR':>7s} {'PnL':>10s} {'Avg':>7s} {'PF':>6s}")
    combos = defaultdict(list)
    for t in all_trades:
        combos[f"{t['model']} {t['pair']} {t['direction']}"].append(t)

    ranked = []
    for k, ct in combos.items():
        if len(ct) < 20:
            continue
        w = sum(1 for t in ct if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in ct)
        g = sum(t["pnl_pips"] for t in ct if t["pnl_pips"] > 0)
        lo = abs(sum(t["pnl_pips"] for t in ct if t["pnl_pips"] <= 0))
        pf_val = g / lo if lo > 0 else 999
        ranked.append((k, len(ct), w / len(ct) * 100, p, p / len(ct), pf_val))

    print("  --- BEST ---")
    for k, n, w, p, a, pf_val in sorted(ranked, key=lambda x: -x[3])[:20]:
        print(f"  {k:32s} {n:6d} {w:6.1f}% {p:+10.0f} {a:+7.2f} {pf_val:6.2f}")
    print("  --- WORST ---")
    for k, n, w, p, a, pf_val in sorted(ranked, key=lambda x: x[3])[:15]:
        print(f"  {k:32s} {n:6d} {w:6.1f}% {p:+10.0f} {a:+7.2f} {pf_val:6.2f}")

    print(f"\n  {'Hour':>4s} {'N':>6s} {'WR':>7s} {'PnL':>10s} {'Avg':>7s}")
    hs = defaultdict(list)
    for t in all_trades:
        hs[t["hour"]].append(t)
    for h in range(24):
        ht = hs.get(h, [])
        if not ht:
            continue
        w = sum(1 for t in ht if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in ht)
        print(f"  {h:4d} {len(ht):6d} {w/len(ht)*100:6.1f}% {p:+10.0f} {p/len(ht):+7.2f}")

    print(f"\n  {'Year':>6s} {'N':>6s} {'WR':>7s} {'PnL':>10s}")
    yrs = defaultdict(list)
    for t in all_trades:
        yrs[t["timestamp"][:4]].append(t)
    for y in sorted(yrs):
        yt = yrs[y]
        w = sum(1 for t in yt if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in yt)
        print(f"  {y:>6s} {len(yt):6d} {w/len(yt)*100:6.1f}% {p:+10.0f}")

    # Model C specific: exit reason breakdown
    mc = [t for t in all_trades if t["model"] == "C"]
    if mc:
        print(f"\n  Model C exit reasons:")
        reasons = defaultdict(int)
        for t in mc:
            reasons[t["exit_reason"]] += 1
        for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {r:15s}: {c:7d} ({c/len(mc)*100:.1f}%)")

        print(f"\n  Model C avg SL/TP pips:")
        avg_sl = np.mean([t["sl_pips"] for t in mc])
        avg_tp = np.mean([t["tp_pips"] for t in mc])
        print(f"    Avg SL: {avg_sl:.1f} pips | Avg TP: {avg_tp:.1f} pips | R:R = 1:{avg_tp/avg_sl:.2f}")


def save_results(all_trades: list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combo_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0, "trades": 0})
    for t in all_trades:
        key = f"{t['model']}_{t['pair']}_{t['direction']}"
        combo_stats[key]["trades"] += 1
        combo_stats[key]["pnl"] += t["pnl_pips"]
        if t["pnl_pips"] > 0:
            combo_stats[key]["wins"] += 1
        else:
            combo_stats[key]["losses"] += 1
    for s in combo_stats.values():
        s["win_rate"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["avg_pips"] = round(s["pnl"] / s["trades"], 2) if s["trades"] else 0

    with open(OUTPUT_DIR / "model_combo_stats_1m.json", "w") as f:
        json.dump(dict(combo_stats), f, indent=2, sort_keys=True)

    hour_edge = {}
    for t in all_trades:
        h = str(t["hour"])
        if h not in hour_edge:
            hour_edge[h] = {"trades": 0, "wins": 0, "pnl": 0,
                            "long_wins": 0, "long_n": 0, "short_wins": 0, "short_n": 0}
        s = hour_edge[h]
        s["trades"] += 1; s["pnl"] += t["pnl_pips"]
        if t["pnl_pips"] > 0: s["wins"] += 1
        if t["direction"] == "long":
            s["long_n"] += 1
            if t["pnl_pips"] > 0: s["long_wins"] += 1
        else:
            s["short_n"] += 1
            if t["pnl_pips"] > 0: s["short_wins"] += 1
    for s in hour_edge.values():
        s["wr"] = round(s["wins"] / s["trades"] * 100, 1) if s["trades"] else 0
        s["long_wr"] = round(s["long_wins"] / s["long_n"] * 100, 1) if s["long_n"] else 0
        s["short_wr"] = round(s["short_wins"] / s["short_n"] * 100, 1) if s["short_n"] else 0

    with open(OUTPUT_DIR / "hour_edge_1m.json", "w") as f:
        json.dump(hour_edge, f, indent=2, sort_keys=True)

    lessons = []
    for key, s in sorted(combo_stats.items(), key=lambda x: -x[1]["pnl"]):
        if s["trades"] < 100:
            continue
        if s["win_rate"] >= 55 and s["avg_pips"] > 0:
            lessons.append({
                "theme": f"1M-EDGE: {key} — {s['win_rate']}% WR, {s['trades']} trades, avg {s['avg_pips']:+.2f}p",
                "confidence": min(0.95, 0.5 + s["win_rate"] / 200),
                "source": "backtest_1m", "pinned": True,
            })
        elif s["win_rate"] < 45 and s["avg_pips"] < -0.5:
            lessons.append({
                "theme": f"1M-AVOID: {key} — {s['win_rate']}% WR, {s['trades']} trades, avg {s['avg_pips']:+.2f}p",
                "confidence": min(0.95, 0.5 + (100 - s["win_rate"]) / 200),
                "source": "backtest_1m", "pinned": True,
            })

    with open(OUTPUT_DIR / "backtest_1m_lessons.jsonl", "w") as f:
        for l in lessons:
            f.write(json.dumps(l) + "\n")

    print(f"\n  Saved: model_combo_stats_1m.json, hour_edge_1m.json, backtest_1m_lessons.jsonl ({len(lessons)} lessons)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print(f"  1-MINUTE BACKTEST — Models C, D, E" + (f" (last {args.years}yr)" if args.years else " (full history)"))
    print("=" * 80)

    all_trades = []
    t0 = time_mod.time()

    for fb, pair in PAIR_MAP.items():
        fn = f"{fb}_1min.csv"
        print(f"\n  {pair} loading...", end=" ", flush=True)
        df = load_pair(fn, pair, args.years)
        if df.empty:
            print("SKIP")
            continue
        print(f"{len(df):,} bars", end="", flush=True)

        print(" | indicators...", end="", flush=True)
        df = add_m1_indicators(df, pair)

        print(" C", end="", flush=True)
        trades_c = backtest_model_c_m1(df, pair)
        print(f":{len(trades_c)}", end="", flush=True)
        all_trades.extend(trades_c)

        print(" D", end="", flush=True)
        trades_d = backtest_model_d_m1(df, pair)
        print(f":{len(trades_d)}", end="", flush=True)
        all_trades.extend(trades_d)

        print(" E(m5)", end="", flush=True)
        m5 = resample_m5(df)
        trades_e = backtest_model_e_m5(m5, pair)
        print(f":{len(trades_e)}", end="", flush=True)
        all_trades.extend(trades_e)

        elapsed = time_mod.time() - t0
        print(f" | {elapsed:.0f}s")

    total_time = time_mod.time() - t0
    print(f"\n  Total: {len(all_trades):,} trades in {total_time:.0f}s ({total_time/60:.1f}min)")

    print_report(all_trades)
    save_results(all_trades)
    print("\n  Done.")


if __name__ == "__main__":
    main()
