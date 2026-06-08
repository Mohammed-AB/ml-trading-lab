#!/usr/bin/env python3
"""Phase 3: ML test-set backtest with probability threshold sweep.

Memory-safe: loads one pair at a time, predicts, simulates, frees.
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scalp_mode.ml.bar_features import (  # noqa: E402
    FEATURE_COLUMNS,
    SPREAD_PIPS_DEFAULT,
    spread_half_price,
    pip_for_pair,
)

from ml_labels import N_FUTURE  # noqa: E402

ML_DIR = ROOT / "data" / "ml"
# Hold-out backtest: after validation window (Feb 2026) for V2 splits.
TEST_START = "2026-03-01"
MAX_BARS = N_FUTURE
TP_PIPS = 15.0
SL_PIPS = 10.0

PAIRS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CAD",
    "AUD_USD",
    "NZD_USD",
]


def sim_trade(i, direction, high, low, close, pip, spread_half, max_bars=MAX_BARS):
    exit_half_pips = spread_half / pip
    n = len(close)
    end = min(i + max_bars + 1, n)
    if direction == "long":
        entry = close[i] + spread_half
        tp = entry + TP_PIPS * pip
        sl = entry - SL_PIPS * pip
        for j in range(i + 1, end):
            if low[j] <= sl and high[j] >= tp:
                return (-SL_PIPS - exit_half_pips, j - i, "sl_ambiguous")
            if low[j] <= sl:
                return (-SL_PIPS - exit_half_pips, j - i, "sl")
            if high[j] >= tp:
                return (TP_PIPS - exit_half_pips, j - i, "tp")
        j = end - 1
        return ((close[j] - spread_half - entry) / pip, end - 1 - i, "time")
    else:
        entry = close[i] - spread_half
        tp = entry - TP_PIPS * pip
        sl = entry + SL_PIPS * pip
        for j in range(i + 1, end):
            if high[j] >= sl and low[j] <= tp:
                return (-SL_PIPS - exit_half_pips, j - i, "sl_ambiguous")
            if high[j] >= sl:
                return (-SL_PIPS - exit_half_pips, j - i, "sl")
            if low[j] <= tp:
                return (TP_PIPS - exit_half_pips, j - i, "tp")
        j = end - 1
        return ((entry - (close[j] + spread_half)) / pip, end - 1 - i, "time")


def backtest_pair(pair, h, l, c, ts, p_long, p_short, thresh):
    trades = []
    n = len(c)
    i = 0
    while i < n - MAX_BARS - 2:
        pl = float(p_long[i])
        ps = float(p_short[i])
        if pl < thresh and ps < thresh:
            i += 1
            continue
        if pl >= thresh and pl >= ps:
            direction = "long"
        elif ps >= thresh:
            direction = "short"
        else:
            i += 1
            continue
        pip = pip_for_pair(pair)
        sh = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 0.6))
        pnl, bars, reason = sim_trade(i, direction, h, l, c, pip, sh)
        trades.append({
            "pair": pair, "direction": direction,
            "pnl_pips": pnl, "bars": bars, "exit": reason,
            "hour": int(ts[i].hour), "year": int(ts[i].year),
        })
        i = i + max(1, bars)
    return trades


def profit_factor(pnls):
    wins = sum(x for x in pnls if x > 0)
    losses = -sum(x for x in pnls if x < 0)
    if losses < 1e-9:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def max_dd_pips(equity):
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)) if len(dd) else 0.0


def summarize(trades, n_days):
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "total_pips": 0.0,
                "avg_pips": 0.0, "tpd": 0.0, "max_dd": 0.0}
    pnls = [t["pnl_pips"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    eq = np.cumsum(np.array(pnls, dtype=np.float64))
    return {
        "n": len(trades), "wr": wins / len(pnls),
        "pf": profit_factor(pnls),
        "total_pips": float(sum(pnls)),
        "avg_pips": float(np.mean(pnls)),
        "tpd": len(trades) / max(n_days, 1e-6),
        "max_dd": max_dd_pips(eq),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-dir", type=Path, default=ML_DIR)
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--min-tpd", type=float, default=3.0)
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=PAIRS,
        help="Pairs to include (default: V2 six-pair set)",
    )
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.45, 0.50, 0.55, 0.60],
        help="Probability grid for sweep (default: 0.45 … 0.60)",
    )
    args = ap.parse_args()

    import lightgbm as lgb

    long_m = lgb.Booster(model_file=str(args.ml_dir / "model_long.txt"))
    short_m = lgb.Booster(model_file=str(args.ml_dir / "model_short.txt"))

    test_start_ts = pd.Timestamp(args.test_start, tz="UTC")

    # Pre-compute predictions per pair (one at a time to limit RAM)
    pair_data: dict[str, dict] = {}
    for p in args.pairs:
        path = args.ml_dir / f"features_{p}.parquet"
        if not path.exists():
            print(f"WARN missing {path}", flush=True)
            continue
        cols = ["timestamp", "pair", "high", "low", "close"] + FEATURE_COLUMNS
        df = pd.read_parquet(path, columns=cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[df["timestamp"] >= test_start_ts].reset_index(drop=True)
        if df.empty:
            del df; gc.collect()
            continue
        X = np.nan_to_num(
            df[FEATURE_COLUMNS].values.astype(np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        pl = long_m.predict(X).astype(np.float32)
        ps = short_m.predict(X).astype(np.float32)
        h = df["high"].values.astype(np.float64)
        l = df["low"].values.astype(np.float64)
        c = df["close"].values.astype(np.float64)
        ts = pd.to_datetime(df["timestamp"]).values
        ts_py = pd.DatetimeIndex(ts)
        pair_data[p] = {"h": h, "l": l, "c": c, "ts": ts_py, "pl": pl, "ps": ps}
        print(f"  {p}: {len(df):,} test bars", flush=True)
        del df, X; gc.collect()

    if not pair_data:
        raise SystemExit("No test data. Run ml_features.py and ml_train.py first.")

    # Date range
    all_ts = [pd.Timestamp(d["ts"].min()) for d in pair_data.values()]
    all_te = [pd.Timestamp(d["ts"].max()) for d in pair_data.values()]
    t0 = min(all_ts)
    t1 = max(all_te)
    n_days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
    print(f"\nTest window {t0} .. {t1} (~{n_days:.0f} days)\n", flush=True)

    thresholds = np.array(sorted(set(args.thresholds)), dtype=np.float64)
    best = None

    for th in thresholds:
        all_trades: list[dict] = []
        for p, d in pair_data.items():
            all_trades.extend(
                backtest_pair(p, d["h"], d["l"], d["c"], d["ts"],
                              d["pl"], d["ps"], float(th))
            )
        s = summarize(all_trades, n_days)
        s["thresh"] = float(th)
        print(
            f"th={th:.3f} n={s['n']:5d} WR={s['wr']:.1%} PF={s['pf']:.2f} "
            f"pips={s['total_pips']:+.0f} avg={s['avg_pips']:+.2f} "
            f"tpd={s['tpd']:.2f} maxDD={s['max_dd']:.0f}",
            flush=True,
        )
        if s["n"] == 0:
            continue
        if s["tpd"] >= args.min_tpd:
            if best is None or s["pf"] > best["pf"]:
                best = s

    if best:
        thb = float(best["thresh"])
        print(
            f"\nSweet spot (max PF with tpd>={args.min_tpd}): "
            f"thresh={thb:.3f} PF={best['pf']:.2f} "
            f"WR={best['wr']:.1%} tpd={best['tpd']:.2f} trades={best['n']}",
            flush=True,
        )
        all_trades = []
        for p, d in pair_data.items():
            all_trades.extend(
                backtest_pair(p, d["h"], d["l"], d["c"], d["ts"],
                              d["pl"], d["ps"], thb)
            )
        by_pair: dict[str, list] = defaultdict(list)
        by_dir: dict[str, list] = defaultdict(list)
        by_year: dict[int, list] = defaultdict(list)
        for t in all_trades:
            by_pair[t["pair"]].append(t)
            by_dir[t["direction"]].append(t)
            by_year[t["year"]].append(t)
        print("\nBreakdown @ sweet threshold:", flush=True)
        for pk in sorted(by_pair):
            s = summarize(by_pair[pk], n_days)
            print(f"  {pk}: n={s['n']} WR={s['wr']:.1%} PF={s['pf']:.2f} pips={s['total_pips']:+.0f}")
        for dk in sorted(by_dir):
            s = summarize(by_dir[dk], n_days)
            print(f"  dir {dk}: n={s['n']} WR={s['wr']:.1%} PF={s['pf']:.2f}")
        for yk in sorted(by_year):
            s = summarize(by_year[yk], n_days)
            print(f"  year {yk}: n={s['n']} WR={s['wr']:.1%} PF={s['pf']:.2f}")
    else:
        print("\nNo threshold met min trades/day -- relax --min-tpd or thresholds")


if __name__ == "__main__":
    main()
