#!/usr/bin/env python3
"""Ultimate strategy sweep V3: 80 strategies x multiple TFs x 10 exit profiles x 8 filters.

Progressive filtering: Round 1 coarse grid, Round 2 fine-grid winners,
Round 3 validation (3-way OOS, 2x spread, Monte Carlo).

  python profit_lab_v3.py --data-dir data/raw --quick
  python -m strategy_arena --sweep-v3
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_strategies import (  # noqa: E402
    STRATEGIES,
    EXIT_PROFILES_V3,
    add_indicators,
    resample_generic,
    simulate_trades_v3,
    simulate_trades_vec,
)
from indicators_extended import add_indicators_extended  # noqa: E402
from signal_filters import FILTER_PRESETS, apply_filters  # noqa: E402
from strategy_arena.config import DEFAULT_DATA_RAW, OOS_START, V2_PAIRS  # noqa: E402
from strategy_arena.loader import list_available_pairs, load_oanda_m1  # noqa: E402
from strategy_arena.runner import RESEARCH_ALL  # noqa: E402
from strategies_v3 import STRATEGIES_V3  # noqa: E402

from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402

from strategy_lab import filter_oos, summarize_trades  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_STRATEGIES = list(STRATEGIES) + list(RESEARCH_ALL) + list(STRATEGIES_V3)

TF_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240,
}

STRATEGY_TFS: dict[str, list[str]] = {}
for sn, _, _ in STRATEGIES:
    STRATEGY_TFS[sn] = ["M5"]
for sn, _, _ in RESEARCH_ALL:
    STRATEGY_TFS[sn] = ["M5"]
for sn, _, _ in STRATEGIES_V3:
    if sn.startswith("V3_KUMO") or sn.startswith("V3_KIJUN") or sn.startswith("V3_CHIKOU"):
        STRATEGY_TFS[sn] = ["M15", "H1", "H4"]
    elif sn.startswith("V3_ORB") or sn.startswith("V3_TKY"):
        STRATEGY_TFS[sn] = ["M5", "M15"]
    elif sn.startswith("V3_HA"):
        STRATEGY_TFS[sn] = ["M15", "H1", "H4"]
    elif sn.startswith("V3_DONCH") or sn.startswith("V3_KELT"):
        STRATEGY_TFS[sn] = ["M5", "M15", "H1"]
    elif sn.startswith("V3_ZSCORE") or sn.startswith("V3_PAIR") or sn.startswith("V3_HURST"):
        STRATEGY_TFS[sn] = ["M15", "H1"]
    elif sn.startswith("V3_BRICK"):
        STRATEGY_TFS[sn] = ["M5", "M15"]
    else:
        STRATEGY_TFS[sn] = ["M5", "M15", "H1"]


def prepare_tf(df_m1: pd.DataFrame, pair: str, tf: str) -> pd.DataFrame:
    n = TF_MAP.get(tf, 5)
    if n == 1:
        df = df_m1.copy()
    else:
        df = resample_generic(df_m1, n)
    df = add_indicators(df, pair)
    df = add_indicators_extended(df, pair)
    return df


def spread_price_full(pair: str) -> float:
    return SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip_for_pair(pair)


def scale_tp_sl(entries, sls, tps, directions, pip, tp_mult, sl_mult=1.0):
    """Scale TP to tp_mult * risk, optionally scale SL too."""
    new_tp = []
    new_sl = []
    for i in range(len(entries)):
        risk = abs(entries[i] - sls[i])
        if risk < pip * 0.25:
            risk = pip * 0.25
        d = directions[i]
        new_sl.append(entries[i] - d * sl_mult * risk)
        new_tp.append(entries[i] + d * tp_mult * risk)
    return new_tp, new_sl


# ---------------------------------------------------------------------------
# Worker: process one (pair, TF) batch
# ---------------------------------------------------------------------------

def _worker_pair_tf(args_tuple: tuple) -> list[dict]:
    (pair, data_dir_s, tf, oos_start_s, quick) = args_tuple
    data_dir = Path(data_dir_s)
    oos_start = pd.Timestamp(oos_start_s)
    pip = pip_for_pair(pair)
    spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip
    hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))

    df_m1 = load_oanda_m1(data_dir, pair)
    if df_m1.empty or len(df_m1) < 3000:
        return []
    df = prepare_tf(df_m1, pair, tf)
    del df_m1
    gc.collect()
    if len(df) < 200:
        return []

    atr = df["atr14"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    tp_mults = (1.5, 2.0, 3.0) if not quick else (2.0,)
    exits = ("none", "be_trail", "chandelier_2", "atr_trail_1.5") if not quick else ("none", "be_trail")
    filters = ("none", "adx_mtf", "session") if not quick else ("none",)

    out: list[dict] = []

    for sname, sig, mb in ALL_STRATEGIES:
        allowed_tfs = STRATEGY_TFS.get(sname, ["M5"])
        if tf not in allowed_tfs:
            continue

        try:
            raw = sig(df, pair, pip, spread)
        except Exception:
            continue
        if not raw:
            continue

        if sname == "S15_VWAP":
            if not raw[0]:
                continue
            idx_r, dir_r, ent_r, sl_r, tp_r = raw[0], raw[1], raw[2], raw[3], raw[4]
        else:
            idx_r, dir_r, ent_r, sl_r, tp_r = raw
        if not idx_r:
            continue

        for tp_m in tp_mults:
            scaled_tp, scaled_sl = scale_tp_sl(ent_r, sl_r, tp_r, dir_r, pip, tp_m)
            for filt in filters:
                f_idx, f_dir, f_ent, f_sl, f_tp = apply_filters(
                    df, idx_r, dir_r, ent_r, scaled_sl, scaled_tp, preset=filt
                )
                if not f_idx:
                    continue
                for ex in exits:
                    tag = f"{sname}|{tf}|tp{tp_m}|{ex}|{filt}"
                    sim = simulate_trades_v3(
                        highs, lows, closes,
                        np.array(f_idx, dtype=np.int64),
                        np.array(f_dir, dtype=np.int64),
                        np.array(f_ent, dtype=np.float64),
                        np.array(f_sl, dtype=np.float64),
                        np.array(f_tp, dtype=np.float64),
                        mb, pip,
                        half_spread=hsp,
                        atr=atr,
                        exit_mode=ex,
                    )
                    tss = df["timestamp"].values
                    tr = []
                    for ki in range(len(sim)):
                        t = sim[ki].copy()
                        t["timestamp"] = str(tss[f_idx[ki]])
                        tr.append(t)
                    oos_tr = filter_oos(tr, oos_start)
                    m = summarize_trades(oos_tr, 30.0)
                    if m["n"] >= 3:
                        out.append({
                            "variant": tag,
                            "pair": pair,
                            "tf": tf,
                            "n": m["n"],
                            "pf": m["pf"],
                            "wr": m["wr"],
                            "pips": m["total_pips"],
                        })
    del df
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(rows: list[dict]) -> list[dict]:
    """Merge same variant across pairs: sum N, weighted PF."""
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r["variant"], []).append(r)
    merged = []
    for v, parts in buckets.items():
        total_n = sum(p["n"] for p in parts)
        if total_n == 0:
            continue
        pf_w = sum(p["pf"] * p["n"] for p in parts) / max(total_n, 1)
        total_pips = sum(p["pips"] for p in parts)
        merged.append({
            "variant": v,
            "pairs": len(parts),
            "n_total": total_n,
            "pf": round(pf_w, 3),
            "pips": round(total_pips, 1),
        })
    merged.sort(key=lambda x: -x["pf"] * np.log1p(max(x["n_total"], 1)))
    return merged


def _monte_carlo_pf(pnls: list[float], n_iter: int = 1000) -> dict:
    arr = np.array(pnls, dtype=np.float64)
    rng = np.random.default_rng(42)
    pfs = []
    for _ in range(n_iter):
        shuffled = rng.permutation(arr)
        w = float(shuffled[shuffled > 0].sum())
        lo = float(-shuffled[shuffled < 0].sum())
        pfs.append(w / lo if lo > 1e-9 else (99.0 if w > 0 else 0.0))
    pfs_arr = np.array(pfs)
    return {
        "median_pf": round(float(np.median(pfs_arr)), 3),
        "p5_pf": round(float(np.percentile(pfs_arr, 5)), 3),
        "p95_pf": round(float(np.percentile(pfs_arr, 95)), 3),
        "mean_pf": round(float(np.mean(pfs_arr)), 3),
    }


def _oos_thirds(trades: list[dict]) -> tuple[list, list, list]:
    if not trades:
        return [], [], []
    n = len(trades)
    t = n // 3
    return trades[:t], trades[t:2*t], trades[2*t:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ultimate strategy sweep V3")
    ap.add_argument("--data-dir", type=Path, default=ROOT / DEFAULT_DATA_RAW)
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "sweep_v3_results.json")
    ap.add_argument("--quick", action="store_true", help="Small grid for smoke test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--round", type=int, default=0, help="0=all, 1=coarse, 2=fine, 3=validate")
    args = ap.parse_args(argv)

    pairs = [p for p in V2_PAIRS if p in list_available_pairs(args.data_dir)]
    if not pairs:
        print(f"No pairs in {args.data_dir}", file=sys.stderr)
        return 1

    tfs = ["M5", "M15", "H1"] if not args.quick else ["M5"]
    oos_start = pd.Timestamp(OOS_START)

    report: dict[str, Any] = {
        "oos_start": OOS_START,
        "pairs": pairs,
        "tfs": tfs,
        "n_strategies": len(ALL_STRATEGIES),
    }

    # ---- Round 1: Coarse grid (parallel by pair x TF) ----
    if args.round in (0, 1):
        print(f"Round 1: coarse sweep — {len(ALL_STRATEGIES)} strategies x {len(tfs)} TFs x {len(pairs)} pairs …")
        work = []
        for p in pairs:
            for tf in tfs:
                work.append((p, str(args.data_dir), tf, str(oos_start), args.quick))

        all_rows: list[dict] = []
        n_workers = min(args.workers, len(work))
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futs = {ex.submit(_worker_pair_tf, w): w for w in work}
                done = 0
                for fu in as_completed(futs):
                    done += 1
                    rows = fu.result()
                    all_rows.extend(rows)
                    w = futs[fu]
                    print(f"  [{done}/{len(work)}] {w[0]}|{w[2]}: {len(rows)} variants with N>=3")
        else:
            for i, w in enumerate(work):
                rows = _worker_pair_tf(w)
                all_rows.extend(rows)
                print(f"  [{i+1}/{len(work)}] {w[0]}|{w[2]}: {len(rows)} variants with N>=3")

        merged = _aggregate(all_rows)
        report["round1"] = {
            "total_raw": len(all_rows),
            "total_merged": len(merged),
            "top_100": merged[:100],
        }

        pf_pass = [m for m in merged if m["pf"] >= 1.0 and m["n_total"] >= 20]
        report["round1"]["pf_ge_1_n_ge_20"] = len(pf_pass)
        report["round1"]["pf_ge_1_top20"] = pf_pass[:20]

        strong = [m for m in merged if m["pf"] >= 1.2 and m["n_total"] >= 40]
        report["round1"]["strong_pf12_n40"] = strong[:30]

        print(f"Round 1 done: {len(all_rows)} raw, {len(merged)} merged, {len(pf_pass)} PF>=1.0 N>=20, {len(strong)} PF>=1.2 N>=40")

    # ---- Round 2: Fine-grid top 50 variants ----
    if args.round in (0, 2):
        print("Round 2: fine-grid top variants …")
        top_from_r1 = report.get("round1", {}).get("pf_ge_1_top20", [])
        if not top_from_r1:
            top_from_r1 = report.get("round1", {}).get("top_100", [])[:20]

        fine_tp = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0] if not args.quick else [1.5, 2.0]
        fine_exits = list(EXIT_PROFILES_V3) if not args.quick else ["none", "be", "be_trail", "chandelier_2"]
        fine_filters = list(FILTER_PRESETS.keys()) if not args.quick else ["none", "adx_mtf"]

        fine_rows: list[dict] = []
        for p in pairs:
            df_m1 = load_oanda_m1(args.data_dir, p)
            if df_m1.empty:
                continue
            pip = pip_for_pair(p)
            spread = SPREAD_PIPS_DEFAULT.get(p, 1.5) * pip
            hsp = spread_half_price(p, SPREAD_PIPS_DEFAULT.get(p, 1.5))

            dfs_by_tf: dict[str, pd.DataFrame] = {}
            for tf in tfs:
                dfs_by_tf[tf] = prepare_tf(df_m1, p, tf)
            del df_m1
            gc.collect()

            for variant_info in top_from_r1:
                v = variant_info["variant"]
                parts = v.split("|")
                sname = parts[0]
                vtf = parts[1] if len(parts) > 1 else "M5"
                if vtf not in dfs_by_tf:
                    continue
                df = dfs_by_tf[vtf]
                atr = df["atr14"].values

                sig_fn = None
                mb = 40
                for n, fn, m in ALL_STRATEGIES:
                    if n == sname:
                        sig_fn = fn
                        mb = m
                        break
                if sig_fn is None:
                    continue

                try:
                    raw = sig_fn(df, p, pip, spread)
                except Exception:
                    continue
                if not raw:
                    continue
                if sname == "S15_VWAP":
                    if not raw[0]:
                        continue
                    idx_r, dir_r, ent_r, sl_r, tp_r = raw[0], raw[1], raw[2], raw[3], raw[4]
                else:
                    idx_r, dir_r, ent_r, sl_r, tp_r = raw
                if not idx_r:
                    continue

                for tp_m in fine_tp:
                    scaled_tp, scaled_sl = scale_tp_sl(ent_r, sl_r, tp_r, dir_r, pip, tp_m)
                    for filt in fine_filters:
                        f_idx, f_dir, f_ent, f_sl, f_tp = apply_filters(
                            df, idx_r, dir_r, ent_r, scaled_sl, scaled_tp, preset=filt
                        )
                        if not f_idx:
                            continue
                        for ex in fine_exits:
                            tag = f"R2:{sname}|{vtf}|tp{tp_m}|{ex}|{filt}"
                            sim = simulate_trades_v3(
                                df["high"].values, df["low"].values, df["close"].values,
                                np.array(f_idx, dtype=np.int64),
                                np.array(f_dir, dtype=np.int64),
                                np.array(f_ent, dtype=np.float64),
                                np.array(f_sl, dtype=np.float64),
                                np.array(f_tp, dtype=np.float64),
                                mb, pip, half_spread=hsp, atr=atr, exit_mode=ex,
                            )
                            tss = df["timestamp"].values
                            tr = []
                            for ki in range(len(sim)):
                                t = sim[ki].copy()
                                t["timestamp"] = str(tss[f_idx[ki]])
                                tr.append(t)
                            oos_tr = filter_oos(tr, oos_start)
                            m = summarize_trades(oos_tr, 30.0)
                            if m["n"] >= 3:
                                fine_rows.append({
                                    "variant": tag,
                                    "pair": p,
                                    "tf": vtf,
                                    "n": m["n"],
                                    "pf": m["pf"],
                                    "wr": m["wr"],
                                    "pips": m["total_pips"],
                                    "trades_oos": oos_tr,
                                })
            for df in dfs_by_tf.values():
                del df
            gc.collect()

        merged_r2 = _aggregate([{k: v for k, v in r.items() if k != "trades_oos"} for r in fine_rows])
        report["round2"] = {
            "total": len(fine_rows),
            "merged": len(merged_r2),
            "top_50": merged_r2[:50],
        }
        strong_r2 = [m for m in merged_r2 if m["pf"] >= 1.2 and m["n_total"] >= 30]
        report["round2"]["strong"] = strong_r2[:30]
        print(f"Round 2 done: {len(fine_rows)} raw, {len(merged_r2)} merged, {len(strong_r2)} strong")

    # ---- Round 3: Validation ----
    if args.round in (0, 3):
        print("Round 3: validation (MC, 2x spread, 3-way OOS) …")
        candidates = report.get("round2", {}).get("strong", [])
        if not candidates:
            candidates = report.get("round2", {}).get("top_50", [])[:10]
        if not candidates:
            candidates = report.get("round1", {}).get("pf_ge_1_top20", [])

        mc_results = []
        for cand in candidates[:20]:
            matching = [r for r in fine_rows if r["variant"] == cand["variant"]] if args.round == 0 else []
            all_pnls = []
            for r in matching:
                all_pnls.extend([float(t["pnl_pips"]) for t in r.get("trades_oos", [])])
            if len(all_pnls) >= 10:
                mc = _monte_carlo_pf(all_pnls, 50 if args.quick else 1000)
                mc_results.append({"variant": cand["variant"], "mc": mc})

                a, b, c = _oos_thirds([{"pnl_pips": p} for p in all_pnls])
                mc_results[-1]["three_way"] = {
                    "a": summarize_trades(a, 10.0),
                    "b": summarize_trades(b, 10.0),
                    "c": summarize_trades(c, 10.0),
                }

        report["round3"] = {
            "monte_carlo": mc_results,
            "n_validated": len(mc_results),
        }

        survivors = [
            r for r in mc_results
            if r["mc"]["p5_pf"] >= 0.8 and r["mc"]["median_pf"] >= 1.0
        ]
        report["round3"]["survivors"] = survivors
        print(f"Round 3 done: {len(mc_results)} validated, {len(survivors)} survive MC")

    # ---- Summary ----
    best = None
    for pool in [
        report.get("round3", {}).get("survivors", []),
        report.get("round2", {}).get("strong", []),
        report.get("round1", {}).get("strong_pf12_n40", []),
        report.get("round1", {}).get("pf_ge_1_top20", []),
    ]:
        if pool:
            best = pool[0]
            break

    report["best"] = best
    report["success"] = best is not None and best.get("pf", best.get("mc", {}).get("median_pf", 0)) >= 1.2

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    out_data = json.loads(json.dumps(report, default=str))
    for r in out_data.get("round2", {}).get("top_50", []):
        r.pop("trades_oos", None)
    args.out_json.write_text(json.dumps(out_data, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out_json}")
    print(f"Best: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
