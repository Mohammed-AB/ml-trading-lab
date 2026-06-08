#!/usr/bin/env python3
"""Exhaustive 5-year sweep: ALL strategies x parameter grid on local M1 data.

Tests every strategy with multiple TP/SL, exit profiles, filters, and timeframes
against 5 years of data. Only keeps strategies with PF > 1.1 AND N > 200.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_strategies import (  # noqa: E402
    STRATEGIES, add_indicators, load_pair, resample_generic,
    simulate_trades_v3,
)
from indicators_extended import add_indicators_extended  # noqa: E402
from signal_filters import apply_filters  # noqa: E402
from strategy_arena.runner import RESEARCH_ALL  # noqa: E402
from strategies_v3 import STRATEGIES_V3  # noqa: E402
from strategy_lab import summarize_trades  # noqa: E402
from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402

PAIRS = {
    "EUR_USD": "EURUSD_1min.csv",
    "USD_JPY": "USDJPY_1min.csv",
    "USD_CAD": "USDCAD_1min.csv",
    "AUD_USD": "AUDUSD_1min.csv",
    "NZD_USD": "NZDUSD_1min.csv",
}

FX_DATA_DIR = Path.home() / "Downloads" / "FX-1-Minute-Data-master" / "forex_data" / "1min"

ALL_STRATS = list(STRATEGIES) + list(RESEARCH_ALL) + list(STRATEGIES_V3)

TP_MULTS = [1.0, 1.5, 2.0, 3.0]
EXITS = ["none", "be_trail", "chandelier_2", "atr_trail_1.5"]
FILTERS = ["none", "adx_mtf", "session"]
TFS = [5, 15, 60]


def prepare_tf(df_m1, pair, n):
    df = resample_generic(df_m1, n) if n > 1 else df_m1.copy()
    df = add_indicators(df, pair)
    df = add_indicators_extended(df, pair)
    return df


def scale_tp(entries, sls, tps, directions, pip, tp_mult):
    new_tp, new_sl = [], []
    for i in range(len(entries)):
        risk = abs(entries[i] - sls[i])
        if risk < pip * 0.25:
            risk = pip * 0.25
        d = directions[i]
        new_sl.append(sls[i])
        new_tp.append(entries[i] + d * tp_mult * risk)
    return new_tp, new_sl


def run_one_strat_pair(sname, sig_fn, mb, df, pair, pip, spread, hsp):
    """Run all parameter combos for one strategy on one pair. Return list of result dicts."""
    try:
        raw = sig_fn(df, pair, pip, spread)
    except Exception:
        return []
    if not raw:
        return []
    if sname == "S15_VWAP":
        if not raw[0]:
            return []
        idx_r, dir_r, ent_r, sl_r, tp_r = raw[0], raw[1], raw[2], raw[3], raw[4]
    else:
        idx_r, dir_r, ent_r, sl_r, tp_r = raw
    if not idx_r:
        return []

    atr = df["atr14"].values
    tss = df["timestamp"].values
    results = []

    for tp_m in TP_MULTS:
        scaled_tp, scaled_sl = scale_tp(ent_r, sl_r, tp_r, dir_r, pip, tp_m)
        for filt in FILTERS:
            f_idx, f_dir, f_ent, f_sl, f_tp = apply_filters(
                df, idx_r, dir_r, ent_r, scaled_sl, scaled_tp, preset=filt
            )
            if not f_idx or len(f_idx) < 5:
                continue
            for ex in EXITS:
                sim = simulate_trades_v3(
                    df["high"].values, df["low"].values, df["close"].values,
                    np.array(f_idx, dtype=np.int64),
                    np.array(f_dir, dtype=np.int64),
                    np.array(f_ent, dtype=np.float64),
                    np.array(f_sl, dtype=np.float64),
                    np.array(f_tp, dtype=np.float64),
                    mb, pip, half_spread=hsp, atr=atr, exit_mode=ex,
                )
                trades = []
                for k in range(len(sim)):
                    t = sim[k].copy()
                    t["timestamp"] = str(tss[f_idx[k]])
                    trades.append(t)

                if len(trades) < 10:
                    continue
                m = summarize_trades(trades, 30.0)
                results.append({
                    "sname": sname,
                    "pair": pair,
                    "tp": tp_m,
                    "exit": ex,
                    "filter": filt,
                    "n": m["n"],
                    "pf": m["pf"],
                    "wr": m["wr"],
                    "pips": m["total_pips"],
                })
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx-dir", type=Path, default=FX_DATA_DIR)
    ap.add_argument("--years", type=int, default=5)
    cli = ap.parse_args()
    fx_dir = cli.fx_dir
    years = cli.years
    print(f"EXHAUSTIVE {years}-YEAR SWEEP: {len(ALL_STRATS)} strategies x {len(PAIRS)} pairs x {len(TFS)} TFs")
    print(f"Data dir: {fx_dir}")
    print(f"Grid: {len(TP_MULTS)} TP x {len(EXITS)} exits x {len(FILTERS)} filters = {len(TP_MULTS)*len(EXITS)*len(FILTERS)} combos/strat")
    print()

    all_results: list[dict] = []
    strat_count = 0

    for pair, filename in PAIRS.items():
        print(f"\n{'='*50}")
        print(f"PAIR: {pair}")
        print(f"{'='*50}")
        fpath = fx_dir / filename
        if not fpath.exists():
            print(f"  SKIP (not found: {fpath})")
            continue
        df = pd.read_csv(fpath, parse_dates=["datetime"])
        df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
        if years > 0:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=years * 365)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        df_m1 = df
        del df
        if df_m1.empty:
            print("  SKIP")
            continue
        print(f"  {len(df_m1):,} M1 bars")

        pip = pip_for_pair(pair)
        spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip
        hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))

        for tf_n in TFS:
            print(f"\n  TF: M{tf_n}")
            df = prepare_tf(df_m1, pair, tf_n)
            print(f"  {len(df):,} bars")

            for sname, sig_fn, mb in ALL_STRATS:
                results = run_one_strat_pair(sname, sig_fn, mb, df, pair, pip, spread, hsp)
                if results:
                    for r in results:
                        r["tf"] = tf_n
                    all_results.extend(results)
                    best = max(results, key=lambda x: x["pf"] * np.log1p(x["n"]))
                    if best["pf"] > 1.1 and best["n"] > 50:
                        print(f"    {sname:15s} tp={best['tp']} {best['exit']:15s} {best['filter']:10s} N={best['n']:5d} PF={best['pf']:5.2f} pips={best['pips']:+,.0f}")

            del df
            gc.collect()

        del df_m1
        gc.collect()

    # Aggregate across pairs
    print(f"\n\n{'='*70}")
    print(f"AGGREGATION: {len(all_results)} raw results")
    print(f"{'='*70}")

    buckets: dict[str, list[dict]] = {}
    for r in all_results:
        key = f"{r['sname']}|M{r['tf']}|tp{r['tp']}|{r['exit']}|{r['filter']}"
        buckets.setdefault(key, []).append(r)

    merged = []
    for key, parts in buckets.items():
        total_n = sum(p["n"] for p in parts)
        if total_n < 200:
            continue
        total_pips = sum(p["pips"] for p in parts)
        pf_w = sum(p["pf"] * p["n"] for p in parts) / max(total_n, 1)
        wr_w = sum(p["wr"] * p["n"] for p in parts) / max(total_n, 1)
        merged.append({
            "variant": key,
            "pairs": len(parts),
            "n": total_n,
            "pf": round(pf_w, 3),
            "wr": round(wr_w, 3),
            "pips": round(total_pips, 1),
        })

    merged.sort(key=lambda x: -x["pf"] * np.log1p(x["n"]))

    profitable = [m for m in merged if m["pf"] >= 1.1]
    strong = [m for m in merged if m["pf"] >= 1.2 and m["n"] >= 500]

    print(f"\nTotal merged variants (N>=200): {len(merged)}")
    print(f"PF >= 1.1: {len(profitable)}")
    print(f"PF >= 1.2 AND N >= 500: {len(strong)}")

    print(f"\n--- TOP 30 BY PF*log(N) ---")
    for m in merged[:30]:
        print(f"  {m['variant']:60s}  pairs={m['pairs']}  N={m['n']:5d}  PF={m['pf']:5.2f}  WR={m['wr']*100:.0f}%  pips={m['pips']:+,.1f}")

    print(f"\n--- STRONG (PF>=1.2, N>=500) ---")
    for m in strong[:20]:
        print(f"  {m['variant']:60s}  pairs={m['pairs']}  N={m['n']:5d}  PF={m['pf']:5.2f}  pips={m['pips']:+,.1f}")

    print(f"\n--- PROFITABLE (PF>=1.1, top 20 by pips) ---")
    for m in sorted(profitable, key=lambda x: -x["pips"])[:20]:
        print(f"  {m['variant']:60s}  pairs={m['pairs']}  N={m['n']:5d}  PF={m['pf']:5.2f}  pips={m['pips']:+,.1f}")

    report = {
        "years": years,
        "total_strategies": len(ALL_STRATS),
        "total_raw": len(all_results),
        "total_merged": len(merged),
        "profitable_count": len(profitable),
        "strong_count": len(strong),
        "top_30": merged[:30],
        "strong": strong[:20],
        "profitable_by_pips": sorted(profitable, key=lambda x: -x["pips"])[:30],
    }

    out = ROOT / "data" / "sweep_5yr_full.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
