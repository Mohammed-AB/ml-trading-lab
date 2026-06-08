"""Comprehensive vectorized backtest of Models C, D, E on 5-minute data.

Usage:
    python3 backtest_all_models.py
    python3 backtest_all_models.py --years 5
"""

import argparse
import json
import sys
import time as time_mod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR_5M = Path.home() / "Downloads" / "FX-1-Minute-Data-master" / "forex_data" / "5min"
DATA_DIR_1M = Path.home() / "Downloads" / "FX-1-Minute-Data-master" / "forex_data" / "1min"
DATA_DIR = DATA_DIR_1M  # default to 1-minute
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


def load_and_prepare(filename: str, pair: str, years: int = 0) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)

    if years > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=years * 365)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    c, h, l = df["close"], df["high"], df["low"]

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


def vectorized_trade_sim(df: pd.DataFrame, signals: pd.DataFrame,
                         max_bars: int = 60) -> list:
    """Simulate trades using forward-looking high/low arrays."""
    if signals.empty:
        return []

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    trades = []

    for _, sig in signals.iterrows():
        idx = int(sig["idx"])
        direction = sig["direction"]
        entry = sig["entry"]
        sl = sig["sl"]
        tp = sig["tp"]
        sl_pips = sig["sl_pips"]
        tp_pips = sig["tp_pips"]

        end = min(idx + max_bars, n)
        pnl_price = 0.0
        exit_reason = "time_stop"
        bars_held = end - idx - 1

        for j in range(idx + 1, end):
            if direction == "long":
                if lows[j] <= sl:
                    pnl_price = sl - entry
                    exit_reason = "sl_hit"
                    bars_held = j - idx
                    break
                if highs[j] >= tp:
                    pnl_price = tp - entry
                    exit_reason = "tp_hit"
                    bars_held = j - idx
                    break
            else:
                if highs[j] >= sl:
                    pnl_price = entry - sl
                    exit_reason = "sl_hit"
                    bars_held = j - idx
                    break
                if lows[j] <= tp:
                    pnl_price = entry - tp
                    exit_reason = "tp_hit"
                    bars_held = j - idx
                    break
        else:
            if end > idx + 1:
                c_exit = closes[min(end - 1, n - 1)]
                pnl_price = (c_exit - entry) if direction == "long" else (entry - c_exit)

        pip_size = sig["pip_size"]
        pnl_pips = round(pnl_price / pip_size, 2)

        trades.append({
            "model": sig["model"], "pair": sig["pair"], "direction": direction,
            "hour": int(sig["hour"]), "pnl_pips": pnl_pips,
            "exit_reason": exit_reason, "bars_held": bars_held,
            "sl_pips": round(sl_pips, 1), "tp_pips": round(tp_pips, 1),
            "timestamp": str(sig["timestamp"]),
        })

    return trades


def find_model_c_signals(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Vectorized EMA crossover detection."""
    pip_size = 0.01 if "JPY" in pair else 0.0001
    spread = SPREAD_PIPS.get(pair, 0.5) * pip_size
    atr_mult = 1.3549
    rr = 0.5009

    ema5 = df["ema5"].values
    ema40 = df["ema40"].values
    atr = df["atr14"].values
    close = df["close"].values

    cross_up = (ema5[:-1] <= ema40[:-1]) & (ema5[1:] > ema40[1:])
    cross_down = (ema5[:-1] >= ema40[:-1]) & (ema5[1:] < ema40[1:])

    signals = []
    last_signal_idx = -7

    for i in np.where(cross_up | cross_down)[0]:
        idx = i + 1
        if idx - last_signal_idx < 6:
            continue
        if idx >= len(df) - 60:
            continue
        if np.isnan(atr[idx]) or atr[idx] <= 0:
            continue

        direction = "long" if cross_up[i] else "short"
        sl_dist = atr[idx] * atr_mult
        tp_dist = sl_dist * rr

        if direction == "long":
            entry = close[idx] + spread / 2
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:
            entry = close[idx] - spread / 2
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        signals.append({
            "idx": idx, "model": "C", "pair": pair, "direction": direction,
            "entry": entry, "sl": sl_price, "tp": tp_price,
            "sl_pips": sl_dist / pip_size, "tp_pips": tp_dist / pip_size,
            "pip_size": pip_size, "hour": df.iloc[idx]["hour"],
            "timestamp": df.iloc[idx]["timestamp"],
        })
        last_signal_idx = idx

    return pd.DataFrame(signals) if signals else pd.DataFrame()


def find_model_d_signals(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Vectorized hour-conditional mean reversion."""
    supported = {"EUR_USD", "GBP_USD", "AUD_USD", "EUR_GBP",
                 "USD_CHF", "USD_CAD", "NZD_USD"}
    if pair not in supported:
        return pd.DataFrame()

    pip_size = 0.01 if "JPY" in pair else 0.0001
    spread = SPREAD_PIPS.get(pair, 0.5) * pip_size
    tp_pips, sl_pips = 4.0, 5.0
    trigger_pips = 10.0

    close = df["close"].values
    hours = df["hour"].values
    dows = df["dow"].values
    atr_p = df["atr_pips"].values

    prev5_move = np.zeros(len(df))
    prev5_move[5:] = (close[5:] - close[:-5]) / pip_size

    signals = []
    last_signal_idx = -4

    for i in range(6, len(df) - 30):
        if i - last_signal_idx < 3:
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
            entry = close[i] + spread / 2
            sl_price = entry - sl_pips * pip_size
            tp_price = entry + tp_pips * pip_size
        elif h in (19, 22) and mv >= trigger_pips:
            direction = "short"
            entry = close[i] - spread / 2
            sl_price = entry + sl_pips * pip_size
            tp_price = entry - tp_pips * pip_size

        if direction is None:
            continue

        signals.append({
            "idx": i, "model": "D", "pair": pair, "direction": direction,
            "entry": entry, "sl": sl_price, "tp": tp_price,
            "sl_pips": sl_pips, "tp_pips": tp_pips,
            "pip_size": pip_size, "hour": h,
            "timestamp": df.iloc[i]["timestamp"],
        })
        last_signal_idx = i

    return pd.DataFrame(signals) if signals else pd.DataFrame()


def find_model_e_signals(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Session VWAP reversion signals."""
    pip_size = 0.01 if "JPY" in pair else 0.0001
    spread = SPREAD_PIPS.get(pair, 0.5) * pip_size
    ext_atr = 2.5
    sl_atr_mult = 0.5
    min_bars = 10
    active = set(range(7, 16))

    signals = []
    last_signal_idx = -13

    dates = df["date"].values
    hours = df["hour"].values
    close = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    atr = df["atr14"].values
    vol = df["volume"].values if "volume" in df.columns else np.zeros(len(df))

    current_date = None
    session_start = 0

    for i in range(20, len(df) - 60):
        if i - last_signal_idx < 12:
            continue

        h = hours[i]
        if h not in active:
            current_date = None
            continue

        d = dates[i]
        if d != current_date:
            session_start = i
            current_date = d

        bars_in = i - session_start
        if bars_in < min_bars:
            continue

        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        sl_idx = slice(session_start, i + 1)
        sess_close = close[sl_idx]
        sess_high = high_arr[sl_idx]
        sess_low = low_arr[sl_idx]
        sess_vol = vol[sl_idx]

        if sess_vol.sum() > 0:
            typical = (sess_high + sess_low + sess_close) / 3.0
            vwap = (typical * sess_vol).sum() / sess_vol.sum()
        else:
            vwap = sess_close.mean()

        dev = close[i] - vwap
        dev_atr = dev / a

        if abs(dev_atr) < ext_atr:
            continue

        sl_dist = a * sl_atr_mult

        if dev > 0:
            direction = "short"
            entry = close[i] - spread / 2
            sl_price = entry + sl_dist
            tp_price = float(vwap)
        else:
            direction = "long"
            entry = close[i] + spread / 2
            sl_price = entry - sl_dist
            tp_price = float(vwap)

        if direction == "long" and tp_price <= entry:
            continue
        if direction == "short" and tp_price >= entry:
            continue

        signals.append({
            "idx": i, "model": "E", "pair": pair, "direction": direction,
            "entry": entry, "sl": sl_price, "tp": tp_price,
            "sl_pips": sl_dist / pip_size,
            "tp_pips": abs(tp_price - entry) / pip_size,
            "pip_size": pip_size, "hour": h,
            "timestamp": df.iloc[i]["timestamp"],
        })
        last_signal_idx = i

    return pd.DataFrame(signals) if signals else pd.DataFrame()


def print_report(all_trades: list):
    if not all_trades:
        print("  No trades.")
        return

    print(f"\n{'='*80}")
    print(f"  RESULTS — {len(all_trades)} trades")
    print(f"{'='*80}")

    total_pips = sum(t["pnl_pips"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["pnl_pips"] > 0)
    wr = wins / len(all_trades) * 100
    gp = sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] > 0)
    gl = abs(sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] <= 0))
    pf = gp / gl if gl > 0 else 999

    print(f"\n  Overall: {wins}W / {len(all_trades)-wins}L | WR: {wr:.1f}% | PnL: {total_pips:+.0f} pips | PF: {pf:.2f}")

    print(f"\n  {'Model':6s} {'Trades':>8s} {'WR':>7s} {'PnL':>10s} {'Avg':>8s} {'PF':>6s}")
    for m in sorted(set(t["model"] for t in all_trades)):
        mt = [t for t in all_trades if t["model"] == m]
        w = sum(1 for t in mt if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in mt)
        g = sum(t["pnl_pips"] for t in mt if t["pnl_pips"] > 0)
        lo = abs(sum(t["pnl_pips"] for t in mt if t["pnl_pips"] <= 0))
        print(f"  {m:6s} {len(mt):8d} {w/len(mt)*100:6.1f}% {p:+10.0f} {p/len(mt):+8.2f} {g/lo if lo else 999:6.2f}")

    print(f"\n  {'Combo':32s} {'N':>6s} {'WR':>7s} {'PnL':>9s} {'Avg':>7s}")
    combos = defaultdict(list)
    for t in all_trades:
        combos[f"{t['model']} {t['pair']} {t['direction']}"].append(t)
    ranked = []
    for k, ct in combos.items():
        if len(ct) < 20:
            continue
        w = sum(1 for t in ct if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in ct)
        ranked.append((k, len(ct), w / len(ct) * 100, p, p / len(ct)))
    for k, n, w, p, a in sorted(ranked, key=lambda x: -x[3])[:15]:
        print(f"  {k:32s} {n:6d} {w:6.1f}% {p:+9.0f} {a:+7.2f}")
    print("  ...")
    for k, n, w, p, a in sorted(ranked, key=lambda x: x[3])[:10]:
        print(f"  {k:32s} {n:6d} {w:6.1f}% {p:+9.0f} {a:+7.2f}")

    print(f"\n  {'Hour':>4s} {'N':>6s} {'WR':>7s} {'PnL':>9s} {'Avg':>7s}")
    hs = defaultdict(list)
    for t in all_trades:
        hs[t["hour"]].append(t)
    for h in range(24):
        ht = hs.get(h, [])
        if not ht:
            continue
        w = sum(1 for t in ht if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in ht)
        print(f"  {h:4d} {len(ht):6d} {w/len(ht)*100:6.1f}% {p:+9.0f} {p/len(ht):+7.2f}")

    print(f"\n  {'Year':>6s} {'N':>6s} {'WR':>7s} {'PnL':>9s}")
    yrs = defaultdict(list)
    for t in all_trades:
        yrs[t["timestamp"][:4]].append(t)
    for y in sorted(yrs):
        yt = yrs[y]
        w = sum(1 for t in yt if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in yt)
        print(f"  {y:>6s} {len(yt):6d} {w/len(yt)*100:6.1f}% {p:+9.0f}")


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

    with open(OUTPUT_DIR / "model_combo_stats.json", "w") as f:
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

    with open(OUTPUT_DIR / "hour_edge_backtest.json", "w") as f:
        json.dump(hour_edge, f, indent=2, sort_keys=True)

    lessons = []
    for key, s in sorted(combo_stats.items(), key=lambda x: -x[1]["pnl"]):
        if s["trades"] < 50:
            continue
        if s["win_rate"] >= 55 and s["avg_pips"] > 0:
            lessons.append({
                "theme": f"EDGE: {key} — {s['win_rate']}% WR, {s['trades']} trades, avg {s['avg_pips']:+.2f}p (backtest)",
                "confidence": min(0.95, 0.5 + s["win_rate"] / 200),
                "source": "backtest", "pinned": True,
            })
        elif s["win_rate"] < 45 and s["avg_pips"] < -0.5:
            lessons.append({
                "theme": f"AVOID: {key} — {s['win_rate']}% WR, {s['trades']} trades, avg {s['avg_pips']:+.2f}p (backtest)",
                "confidence": min(0.95, 0.5 + (100 - s["win_rate"]) / 200),
                "source": "backtest", "pinned": True,
            })

    with open(OUTPUT_DIR / "backtest_lessons.jsonl", "w") as f:
        for l in lessons:
            f.write(json.dumps(l) + "\n")

    print(f"\n  Saved: model_combo_stats.json, hour_edge_backtest.json, backtest_lessons.jsonl ({len(lessons)} lessons)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print(f"  BACKTEST — Models C, D, E on 5min data" + (f" (last {args.years}yr)" if args.years else ""))
    print("=" * 80)

    all_trades = []
    t0 = time_mod.time()

    for fb, pair in PAIR_MAP.items():
        fn = f"{fb}_5min.csv"
        print(f"\n  {pair} ...", end=" ", flush=True)
        df = load_and_prepare(fn, pair, args.years)
        if df.empty:
            print("SKIP (not found)")
            continue
        print(f"{len(df)} bars", end="", flush=True)

        sigs_c = find_model_c_signals(df, pair)
        print(f" | C:{len(sigs_c)}", end="", flush=True)
        if not sigs_c.empty:
            all_trades.extend(vectorized_trade_sim(df, sigs_c, max_bars=60))

        sigs_d = find_model_d_signals(df, pair)
        print(f" D:{len(sigs_d)}", end="", flush=True)
        if not sigs_d.empty:
            all_trades.extend(vectorized_trade_sim(df, sigs_d, max_bars=15))

        sigs_e = find_model_e_signals(df, pair)
        print(f" E:{len(sigs_e)}", end="", flush=True)
        if not sigs_e.empty:
            all_trades.extend(vectorized_trade_sim(df, sigs_e, max_bars=60))

        print(f" | {time_mod.time()-t0:.0f}s")

    print(f"\n  Total: {len(all_trades)} trades in {time_mod.time()-t0:.0f}s")

    print_report(all_trades)
    save_results(all_trades)
    print("\n  Done.")


if __name__ == "__main__":
    main()
