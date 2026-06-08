#!/usr/bin/env python3
"""Deep validation of top V3 strategies: per-pair, per-month, drawdown, 2x spread, ensemble."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_strategies import add_indicators, resample_generic, simulate_trades_v3  # noqa: E402
from indicators_extended import add_indicators_extended  # noqa: E402
from signal_filters import apply_filters  # noqa: E402
from strategies_v3 import STRATEGIES_V3  # noqa: E402
from strategy_arena.config import DEFAULT_DATA_RAW, OOS_START, V2_PAIRS  # noqa: E402
from strategy_arena.loader import list_available_pairs, load_oanda_m1  # noqa: E402
from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402
from strategy_lab import filter_oos, summarize_trades  # noqa: E402


TARGETS = ["V3_TKY_LDN", "V3_HA_ADX", "V3_NR7"]

BEST_CONFIGS = {
    "V3_TKY_LDN": {"tf": 15, "tp_mult": 2.0, "exit": "none", "filter": "adx_mtf", "mb": 60},
    "V3_HA_ADX":  {"tf": 60, "tp_mult": 1.5, "exit": "none", "filter": "session", "mb": 40},
    "V3_NR7":     {"tf": 60, "tp_mult": 3.0, "exit": "none", "filter": "adx_mtf", "mb": 40},
}


def prepare_tf(df_m1, pair, n):
    if n == 1:
        df = df_m1.copy()
    else:
        df = resample_generic(df_m1, n)
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


def run_strategy(df, pair, sname, cfg, spread_mult=1.0):
    """Run one strategy on one pair, return list of trade dicts with timestamps."""
    pip = pip_for_pair(pair)
    spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip * spread_mult
    hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5) * spread_mult)

    sig_fn = None
    for n, fn, mb in STRATEGIES_V3:
        if n == sname:
            sig_fn = fn
            break
    if sig_fn is None:
        return []

    try:
        raw = sig_fn(df, pair, pip, spread)
    except Exception:
        return []
    if not raw:
        return []
    idx_r, dir_r, ent_r, sl_r, tp_r = raw
    if not idx_r:
        return []

    tp_mult = cfg["tp_mult"]
    scaled_tp, scaled_sl = scale_tp(ent_r, sl_r, tp_r, dir_r, pip, tp_mult)

    filt = cfg["filter"]
    f_idx, f_dir, f_ent, f_sl, f_tp = apply_filters(
        df, idx_r, dir_r, ent_r, scaled_sl, scaled_tp, preset=filt
    )
    if not f_idx:
        return []

    atr = df["atr14"].values
    sim = simulate_trades_v3(
        df["high"].values, df["low"].values, df["close"].values,
        np.array(f_idx, dtype=np.int64),
        np.array(f_dir, dtype=np.int64),
        np.array(f_ent, dtype=np.float64),
        np.array(f_sl, dtype=np.float64),
        np.array(f_tp, dtype=np.float64),
        cfg["mb"], pip, half_spread=hsp, atr=atr, exit_mode=cfg["exit"],
    )

    tss = df["timestamp"].values
    hours = df["hour"].values if "hour" in df.columns else np.zeros(len(df), dtype=int)
    trades = []
    for k in range(len(sim)):
        t = sim[k].copy()
        t["timestamp"] = str(tss[f_idx[k]])
        t["hour"] = int(hours[f_idx[k]])
        t["direction"] = int(f_dir[k])
        t["pair"] = pair
        trades.append(t)
    return trades


def max_drawdown_pips(trades):
    if not trades:
        return 0.0
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    for t in trades:
        equity += t["pnl_pips"]
        peak = max(peak, equity)
        dd = peak - equity
        mdd = max(mdd, dd)
    return round(mdd, 2)


def per_month_breakdown(trades):
    by_month: dict[str, list] = {}
    for t in trades:
        ts = pd.Timestamp(t["timestamp"])
        key = ts.strftime("%Y-%m")
        by_month.setdefault(key, []).append(t)
    result = {}
    for m, tl in sorted(by_month.items()):
        s = summarize_trades(tl, 30.0)
        result[m] = {"n": s["n"], "pf": s["pf"], "wr": round(s["wr"] * 100, 1), "pips": s["total_pips"]}
    return result


def per_hour_breakdown(trades):
    by_hour: dict[int, list] = {}
    for t in trades:
        by_hour.setdefault(t.get("hour", 0), []).append(t)
    result = {}
    for h in sorted(by_hour):
        tl = by_hour[h]
        s = summarize_trades(tl, 30.0)
        result[h] = {"n": s["n"], "pf": s["pf"], "pips": s["total_pips"]}
    return result


def per_dow_breakdown(trades):
    by_dow: dict[int, list] = {}
    for t in trades:
        ts = pd.Timestamp(t["timestamp"])
        by_dow.setdefault(ts.dayofweek, []).append(t)
    names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    result = {}
    for d in sorted(by_dow):
        tl = by_dow[d]
        s = summarize_trades(tl, 30.0)
        result[names.get(d, str(d))] = {"n": s["n"], "pf": s["pf"], "pips": s["total_pips"]}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Deep validation of top strategies")
    ap.add_argument("--data-dir", type=Path, default=ROOT / DEFAULT_DATA_RAW)
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "deep_validation.json")
    args = ap.parse_args(argv)

    pairs = [p for p in V2_PAIRS if p in list_available_pairs(args.data_dir)]
    oos_start = pd.Timestamp(OOS_START)
    report: dict[str, Any] = {"oos_start": OOS_START, "pairs": pairs}

    for sname in TARGETS:
        cfg = BEST_CONFIGS[sname]
        print(f"\n{'='*60}")
        print(f"Validating {sname} (TF=M{cfg['tf']}, TP={cfg['tp_mult']}x, exit={cfg['exit']}, filter={cfg['filter']})")
        print(f"{'='*60}")

        all_trades_1x: list[dict] = []
        all_trades_2x: list[dict] = []
        per_pair: dict[str, dict] = {}

        for pair in pairs:
            df_m1 = load_oanda_m1(args.data_dir, pair)
            if df_m1.empty:
                continue
            df = prepare_tf(df_m1, pair, cfg["tf"])
            del df_m1
            gc.collect()

            trades_1x = run_strategy(df, pair, sname, cfg, spread_mult=1.0)
            oos_1x = filter_oos(trades_1x, oos_start)

            trades_2x = run_strategy(df, pair, sname, cfg, spread_mult=2.0)
            oos_2x = filter_oos(trades_2x, oos_start)

            m1 = summarize_trades(oos_1x, 30.0)
            m2 = summarize_trades(oos_2x, 30.0)
            mdd = max_drawdown_pips(oos_1x)

            per_pair[pair] = {
                "n": m1["n"], "pf": m1["pf"], "wr": round(m1["wr"] * 100, 1),
                "pips": m1["total_pips"], "mdd": mdd,
                "stress_2x": {"n": m2["n"], "pf": m2["pf"], "pips": m2["total_pips"]},
            }
            print(f"  {pair}: N={m1['n']:4d}  PF={m1['pf']:5.2f}  WR={m1['wr']*100:.0f}%  pips={m1['total_pips']:+.1f}  MDD={mdd:.1f}  |  2x: PF={m2['pf']:.2f} pips={m2['total_pips']:+.1f}")

            all_trades_1x.extend(oos_1x)
            all_trades_2x.extend(oos_2x)
            del df
            gc.collect()

        all_trades_1x.sort(key=lambda t: t["timestamp"])
        m_all = summarize_trades(all_trades_1x, 30.0)
        m_2x = summarize_trades(all_trades_2x, 30.0)
        mdd_all = max_drawdown_pips(all_trades_1x)

        print(f"\n  TOTAL:  N={m_all['n']}  PF={m_all['pf']:.2f}  WR={m_all['wr']*100:.0f}%  pips={m_all['total_pips']:+.1f}  MDD={mdd_all:.1f}")
        print(f"  2xSprd: N={m_2x['n']}  PF={m_2x['pf']:.2f}  pips={m_2x['total_pips']:+.1f}")

        months = per_month_breakdown(all_trades_1x)
        hours = per_hour_breakdown(all_trades_1x)
        dows = per_dow_breakdown(all_trades_1x)

        print(f"\n  Per month:")
        for m, s in months.items():
            print(f"    {m}: N={s['n']:3d}  PF={s['pf']:5.2f}  WR={s['wr']:.0f}%  pips={s['pips']:+.1f}")

        print(f"\n  Per DOW:")
        for d, s in dows.items():
            print(f"    {d}: N={s['n']:3d}  PF={s['pf']:5.2f}  pips={s['pips']:+.1f}")

        print(f"\n  Per hour (top 5 by pips):")
        for h, s in sorted(hours.items(), key=lambda x: -x[1]["pips"])[:5]:
            print(f"    H{h:02d}: N={s['n']:3d}  PF={s['pf']:5.2f}  pips={s['pips']:+.1f}")

        report[sname] = {
            "config": cfg,
            "total": {"n": m_all["n"], "pf": m_all["pf"], "wr": round(m_all["wr"]*100, 1),
                       "pips": m_all["total_pips"], "mdd": mdd_all},
            "stress_2x": {"n": m_2x["n"], "pf": m_2x["pf"], "pips": m_2x["total_pips"]},
            "per_pair": per_pair,
            "per_month": months,
            "per_hour": hours,
            "per_dow": dows,
            "trades_oos": all_trades_1x,
        }

    # ---- Ensemble: 2-of-3 voting ----
    print(f"\n{'='*60}")
    print("ENSEMBLE: 2-of-3 voting (TKY_LDN + HA_ADX + NR7)")
    print(f"{'='*60}")

    ensemble_trades: list[dict] = []
    for pair in pairs:
        strat_trades: dict[str, list[dict]] = {}
        for sname in TARGETS:
            strat_trades[sname] = [t for t in report[sname]["trades_oos"] if t["pair"] == pair]

        by_date_dir: dict[tuple, list[str]] = {}
        for sname in TARGETS:
            for t in strat_trades[sname]:
                ts = pd.Timestamp(t["timestamp"])
                date_key = ts.date()
                d = t["direction"]
                key = (date_key, d)
                by_date_dir.setdefault(key, []).append(sname)

        for (date_key, direction), strats in by_date_dir.items():
            if len(set(strats)) < 2:
                continue
            best_trade = None
            best_pips = -999
            for sname in set(strats):
                for t in strat_trades[sname]:
                    ts = pd.Timestamp(t["timestamp"])
                    if ts.date() == date_key and t["direction"] == direction:
                        if t["pnl_pips"] > best_pips or best_trade is None:
                            best_trade = t.copy()
                            best_trade["ensemble_strats"] = list(set(strats))
                            best_pips = t["pnl_pips"]
            if best_trade:
                ensemble_trades.append(best_trade)

    ensemble_trades.sort(key=lambda t: t["timestamp"])
    m_ens = summarize_trades(ensemble_trades, 30.0)
    mdd_ens = max_drawdown_pips(ensemble_trades)

    print(f"  N={m_ens['n']}  PF={m_ens['pf']:.2f}  WR={m_ens['wr']*100:.0f}%  pips={m_ens['total_pips']:+.1f}  MDD={mdd_ens:.1f}")

    ens_months = per_month_breakdown(ensemble_trades)
    print(f"\n  Per month:")
    for m, s in ens_months.items():
        print(f"    {m}: N={s['n']:3d}  PF={s['pf']:5.2f}  pips={s['pips']:+.1f}")

    report["ensemble"] = {
        "method": "2-of-3 voting, best trade per day/direction",
        "total": {"n": m_ens["n"], "pf": m_ens["pf"], "wr": round(m_ens["wr"]*100, 1),
                   "pips": m_ens["total_pips"], "mdd": mdd_ens},
        "per_month": ens_months,
    }

    out = {k: v for k, v in report.items()}
    for sname in TARGETS:
        if sname in out:
            out[sname] = {k: v for k, v in out[sname].items() if k != "trades_oos"}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
