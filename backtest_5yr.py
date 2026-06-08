#!/usr/bin/env python3
"""5-year backtest of TKY_LDN + HA_ADX + NR7 on local M1 data."""

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

from backtest_strategies import load_pair, add_indicators, resample_generic, simulate_trades_v3  # noqa: E402
from indicators_extended import add_indicators_extended  # noqa: E402
from signal_filters import apply_filters  # noqa: E402
from strategies_v3_part2 import signals_v3_tky_ldn, signals_v3_ha_adx, signals_v3_nr7  # noqa: E402
from strategy_lab import summarize_trades  # noqa: E402

PAIRS = {
    "EUR_USD": "EURUSD_1min.csv",
    "USD_JPY": "USDJPY_1min.csv",
    "USD_CAD": "USDCAD_1min.csv",
    "AUD_USD": "AUDUSD_1min.csv",
    "NZD_USD": "NZDUSD_1min.csv",
}

STRATS = {
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
        "skip_pairs": set(), "skip_dow": set(),
    },
}


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


from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402


def main():
    years = 5
    account = 500.0
    risk_pct = 2.0

    print(f"5-year backtest: {len(STRATS)} strategies x {len(PAIRS)} pairs x {years} years")
    print(f"Account: ${account:.0f}, Risk: {risk_pct}%\n")

    all_trades: list[dict] = []

    for pair, filename in PAIRS.items():
        print(f"Loading {pair} ({years}yr)...", end=" ", flush=True)
        df_m1 = load_pair(filename, pair, years=years)
        if df_m1.empty:
            print("SKIP (no data)")
            continue
        print(f"{len(df_m1):,} M1 bars")

        pip = pip_for_pair(pair)
        spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip
        hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))

        dfs: dict[int, pd.DataFrame] = {}
        for sname, cfg in STRATS.items():
            tf_n = cfg["tf"]
            if tf_n not in dfs:
                print(f"  Resampling to M{tf_n}...", end=" ", flush=True)
                dfs[tf_n] = prepare_tf(df_m1, pair, tf_n)
                print(f"{len(dfs[tf_n]):,} bars")

        del df_m1
        gc.collect()

        for sname, cfg in STRATS.items():
            if pair in cfg["skip_pairs"]:
                continue
            df = dfs[cfg["tf"]]
            try:
                raw = cfg["fn"](df, pair, pip, spread)
            except Exception as e:
                print(f"  {sname} error: {e}")
                continue
            if not raw or not raw[0]:
                print(f"  {sname}: 0 signals")
                continue

            idx_r, dir_r, ent_r, sl_r, tp_r = raw
            scaled_tp, scaled_sl = scale_tp(ent_r, sl_r, tp_r, dir_r, pip, cfg["tp_mult"])
            f_idx, f_dir, f_ent, f_sl, f_tp = apply_filters(
                df, idx_r, dir_r, ent_r, scaled_sl, scaled_tp, preset=cfg["filter"]
            )
            if not f_idx:
                print(f"  {sname}: 0 after filter")
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
            count = 0
            for k in range(len(sim)):
                ts = pd.Timestamp(str(tss[f_idx[k]]))
                dow = ts.dayofweek
                if dow in cfg["skip_dow"]:
                    continue
                t = sim[k].copy()
                t["timestamp"] = str(tss[f_idx[k]])
                t["pair"] = pair
                t["strategy"] = sname
                t["risk_pips"] = abs(f_ent[k] - f_sl[k]) / pip
                count += 1
                all_trades.append(t)
            print(f"  {sname}: {count} trades")

        for df in dfs.values():
            del df
        gc.collect()

    all_trades.sort(key=lambda t: t["timestamp"])
    print(f"\nTotal trades: {len(all_trades)}")

    # Per-strategy stats
    by_strat: dict[str, list] = {}
    for t in all_trades:
        by_strat.setdefault(t["strategy"], []).append(t)

    print(f"\n{'='*60}")
    print("RAW PIP PERFORMANCE (no compounding)")
    print(f"{'='*60}")
    for sname, trades in sorted(by_strat.items()):
        m = summarize_trades(trades, 30.0)
        print(f"  {sname:10s}: N={m['n']:5d}  PF={m['pf']:5.2f}  WR={m['wr']*100:.0f}%  pips={m['total_pips']:+,.1f}")

    m_all = summarize_trades(all_trades, 30.0)
    print(f"  {'COMBINED':10s}: N={m_all['n']:5d}  PF={m_all['pf']:5.2f}  WR={m_all['wr']*100:.0f}%  pips={m_all['total_pips']:+,.1f}")

    # Per-year breakdown
    by_year: dict[str, list] = {}
    for t in all_trades:
        yr = t["timestamp"][:4]
        by_year.setdefault(yr, []).append(t)

    print(f"\nPer year:")
    for yr in sorted(by_year):
        m = summarize_trades(by_year[yr], 30.0)
        print(f"  {yr}: N={m['n']:5d}  PF={m['pf']:5.2f}  WR={m['wr']*100:.0f}%  pips={m['total_pips']:+,.1f}")

    # Equity simulation with compounding
    print(f"\n{'='*60}")
    print(f"EQUITY SIM: ${account:.0f} start, {risk_pct}% risk, daily compounding")
    print(f"{'='*60}")

    equity = account
    peak = account
    max_dd_pct = 0.0
    by_month_eq: dict[str, list] = {}

    for t in all_trades:
        risk_usd = equity * risk_pct / 100
        risk_p = t.get("risk_pips", 40)
        if risk_p < 1:
            risk_p = 40
        pip_val = risk_usd / risk_p
        pnl_usd = t["pnl_pips"] * pip_val
        equity += pnl_usd
        if equity < 0:
            equity = 0
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd)

        mk = t["timestamp"][:7]
        by_month_eq.setdefault(mk, []).append(pnl_usd)

    print(f"Final equity: ${equity:,.2f}")
    print(f"Total return: {(equity - account) / account * 100:+,.1f}%")
    print(f"Max drawdown: {max_dd_pct:.1f}%")

    print(f"\nMonthly equity (sampled):")
    running = account
    months = sorted(by_month_eq.keys())
    for mk in months:
        month_pnl = sum(by_month_eq[mk])
        start = running
        running += month_pnl
        pct = month_pnl / start * 100 if start > 0 else 0
        if mk[-2:] in ("01", "04", "07", "10") or mk >= "2026":
            print(f"  {mk}: ${running:>10,.2f} ({pct:+6.1f}%)")

    report = {
        "years": years,
        "account": account,
        "risk_pct": risk_pct,
        "total_trades": len(all_trades),
        "final_equity": round(equity, 2),
        "total_return_pct": round((equity - account) / account * 100, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "per_strategy": {s: summarize_trades(t, 30.0) for s, t in by_strat.items()},
        "per_year": {y: summarize_trades(t, 30.0) for y, t in by_year.items()},
        "combined": summarize_trades(all_trades, 30.0),
    }

    out = ROOT / "data" / "backtest_5yr.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
