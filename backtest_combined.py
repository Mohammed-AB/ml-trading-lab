#!/usr/bin/env python3
"""Combined backtest: TKY_LDN + HA_ADX + NR7 with tiered risk sizing.

Simulates a $500 OANDA account with daily compounding, 2% base risk,
conviction boosts, and daily loss cap. Uses FULL 12-month data (IS+OOS).
"""

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
from strategies_v3_part2 import (  # noqa: E402
    signals_v3_tky_ldn, signals_v3_ha_adx, signals_v3_nr7,
)
from strategy_arena.config import DEFAULT_DATA_RAW, OOS_START, V2_PAIRS  # noqa: E402
from strategy_arena.loader import list_available_pairs, load_oanda_m1  # noqa: E402
from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402
from strategy_lab import summarize_trades  # noqa: E402


STRAT_CONFIGS = {
    "TKY_LDN": {
        "fn": signals_v3_tky_ldn, "tf": 15, "tp_mult": 2.0, "exit": "none",
        "filter": "adx_mtf", "mb": 60,
        "skip_pairs": {"USD_CAD"}, "skip_dow": {3},
    },
    "HA_ADX": {
        "fn": signals_v3_ha_adx, "tf": 60, "tp_mult": 1.5, "exit": "none",
        "filter": "session", "mb": 40,
        "skip_pairs": set(), "skip_dow": {5, 6},
    },
    "NR7": {
        "fn": signals_v3_nr7, "tf": 60, "tp_mult": 3.0, "exit": "none",
        "filter": "adx_mtf", "mb": 40,
        "skip_pairs": {"GBP_USD"}, "skip_dow": set(),
    },
}

CONVICTION_PAIRS = {"NZD_USD", "AUD_USD", "USD_JPY", "EUR_USD"}


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


def get_risk_pct(pair, adx, dow, daily_pnl_pct):
    if daily_pnl_pct <= -6.0:
        return 0.0
    if dow == 4:
        return 1.0
    if daily_pnl_pct < -2.0:
        return 1.0
    if adx >= 35 and pair in CONVICTION_PAIRS:
        return 3.0
    return 2.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Combined 3-strategy backtest with sizing")
    ap.add_argument("--data-dir", type=Path, default=ROOT / DEFAULT_DATA_RAW)
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "combined_backtest.json")
    ap.add_argument("--account", type=float, default=500.0)
    ap.add_argument("--full-range", action="store_true", help="Use full 12 months, not just OOS")
    args = ap.parse_args(argv)

    pairs = [p for p in V2_PAIRS if p in list_available_pairs(args.data_dir)]
    oos_start = pd.Timestamp(OOS_START)
    if oos_start.tzinfo is not None:
        oos_start = oos_start.tz_localize(None)
    starting_balance = args.account

    print(f"Combined backtest: {len(STRAT_CONFIGS)} strategies x {len(pairs)} pairs")
    print(f"Account: ${starting_balance:.0f}, OOS start: {OOS_START}")
    if args.full_range:
        print("Using FULL 12-month range (IS + OOS)")

    all_trades: list[dict] = []

    for pair in pairs:
        df_m1 = load_oanda_m1(args.data_dir, pair)
        if df_m1.empty:
            continue
        pip = pip_for_pair(pair)
        spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip
        hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))

        dfs: dict[int, pd.DataFrame] = {}
        for sname, cfg in STRAT_CONFIGS.items():
            tf_n = cfg["tf"]
            if tf_n not in dfs:
                dfs[tf_n] = prepare_tf(df_m1, pair, tf_n)

        del df_m1
        gc.collect()

        for sname, cfg in STRAT_CONFIGS.items():
            if pair in cfg["skip_pairs"]:
                continue

            df = dfs[cfg["tf"]]
            try:
                raw = cfg["fn"](df, pair, pip, spread)
            except Exception:
                continue
            if not raw or not raw[0]:
                continue
            idx_r, dir_r, ent_r, sl_r, tp_r = raw

            scaled_tp, scaled_sl = scale_tp(ent_r, sl_r, tp_r, dir_r, pip, cfg["tp_mult"])
            f_idx, f_dir, f_ent, f_sl, f_tp = apply_filters(
                df, idx_r, dir_r, ent_r, scaled_sl, scaled_tp, preset=cfg["filter"]
            )
            if not f_idx:
                continue

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
            adx_arr = df["adx"].values if "adx" in df.columns else np.zeros(len(df))

            for k in range(len(sim)):
                ts = pd.Timestamp(str(tss[f_idx[k]]))
                dow = ts.dayofweek
                if dow in cfg["skip_dow"]:
                    continue

                if not args.full_range and ts < oos_start:
                    continue

                t = sim[k].copy()
                t["timestamp"] = str(tss[f_idx[k]])
                t["pair"] = pair
                t["strategy"] = sname
                t["hour"] = int(hours[f_idx[k]])
                t["dow"] = dow
                t["adx"] = float(adx_arr[f_idx[k]]) if f_idx[k] < len(adx_arr) else 0
                t["risk_pips"] = abs(f_ent[k] - f_sl[k]) / pip
                all_trades.append(t)

        for df in dfs.values():
            del df
        gc.collect()

    all_trades.sort(key=lambda t: t["timestamp"])
    print(f"\nTotal trades: {len(all_trades)}")

    equity = starting_balance
    peak = starting_balance
    max_dd_pct = 0.0
    daily_pnl: dict[str, float] = {}
    equity_curve: list[dict] = []
    monthly_returns: dict[str, float] = {}

    for t in all_trades:
        ts = pd.Timestamp(t["timestamp"])
        day_key = ts.strftime("%Y-%m-%d")
        month_key = ts.strftime("%Y-%m")

        day_pnl_so_far = daily_pnl.get(day_key, 0.0)
        adx = t.get("adx", 0)
        dow = t.get("dow", 0)
        risk_pct = get_risk_pct(t["pair"], adx, dow, day_pnl_so_far / max(equity, 1) * 100)

        if risk_pct <= 0:
            continue

        risk_usd = equity * risk_pct / 100
        risk_pips = t.get("risk_pips", 40)
        if risk_pips < 1:
            risk_pips = 40

        pip_value = risk_usd / risk_pips
        pnl_usd = t["pnl_pips"] * pip_value

        equity += pnl_usd
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + pnl_usd

        peak = max(peak, equity)
        dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd_pct)

        monthly_returns.setdefault(month_key, starting_balance if month_key not in monthly_returns else monthly_returns[month_key])

        equity_curve.append({
            "ts": str(ts),
            "equity": round(equity, 2),
            "pnl_usd": round(pnl_usd, 2),
            "risk_pct": risk_pct,
            "strategy": t["strategy"],
            "pair": t["pair"],
        })

    final_equity = equity
    total_return = (final_equity - starting_balance) / starting_balance * 100

    by_strat: dict[str, list] = {}
    for t in all_trades:
        by_strat.setdefault(t["strategy"], []).append(t)

    print(f"\n{'='*60}")
    print(f"COMBINED RESULTS (${starting_balance:.0f} account)")
    print(f"{'='*60}")
    print(f"Final equity: ${final_equity:.2f}")
    print(f"Total return: {total_return:+.1f}%")
    print(f"Max drawdown: {max_dd_pct:.1f}%")
    print(f"Total trades: {len(all_trades)}")

    n_days = len(daily_pnl)
    if n_days > 0:
        months = n_days / 30.0
        monthly_pct = total_return / months if months > 0 else 0
        print(f"Days traded: {n_days}")
        print(f"Avg monthly return: {monthly_pct:.1f}%")

    print(f"\nPer strategy:")
    for sname, trades in sorted(by_strat.items()):
        m = summarize_trades(trades, 30.0)
        print(f"  {sname:10s}: N={m['n']:4d}  PF={m['pf']:5.2f}  WR={m['wr']*100:.0f}%  pips={m['total_pips']:+.1f}")

    by_month_equity: dict[str, list] = {}
    for e in equity_curve:
        mk = e["ts"][:7]
        by_month_equity.setdefault(mk, []).append(e)

    print(f"\nMonthly equity progression:")
    for mk in sorted(by_month_equity):
        entries = by_month_equity[mk]
        month_pnl = sum(e["pnl_usd"] for e in entries)
        end_eq = entries[-1]["equity"]
        start_eq = end_eq - month_pnl
        pct = month_pnl / start_eq * 100 if start_eq > 0 else 0
        print(f"  {mk}: {len(entries)} trades, ${month_pnl:+.2f} ({pct:+.1f}%), equity=${end_eq:.2f}")

    report = {
        "starting_balance": starting_balance,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "total_trades": len(all_trades),
        "days_traded": n_days,
        "per_strategy": {
            sname: summarize_trades(trades, 30.0) for sname, trades in by_strat.items()
        },
        "equity_curve_sample": equity_curve[::max(1, len(equity_curve)//50)],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
