#!/usr/bin/env python3
"""Final validation: V3_BOS + V3_NR7 on 5yr data + S15_VWAP verification.

Per-year, per-pair, equity simulation with tiered sizing, drawdown analysis.
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

from backtest_strategies import add_indicators, resample_generic, simulate_trades_v3  # noqa: E402
from backtest_strategies import STRATEGIES as BOOK_STRATEGIES  # noqa: E402
from indicators_extended import add_indicators_extended  # noqa: E402
from signal_filters import apply_filters  # noqa: E402
from strategies_v3_part2 import signals_v3_nr7  # noqa: E402
from strategies_v3 import STRATEGIES_V3  # noqa: E402
from strategy_lab import summarize_trades  # noqa: E402
from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402

import argparse


PAIRS = {
    "EUR_USD": "EURUSD_1min.csv",
    "USD_JPY": "USDJPY_1min.csv",
    "USD_CAD": "USDCAD_1min.csv",
    "AUD_USD": "AUDUSD_1min.csv",
    "NZD_USD": "NZDUSD_1min.csv",
}

# Get V3_BOS signal function
signals_v3_bos = None
for n, fn, mb in STRATEGIES_V3:
    if n == "V3_BOS":
        signals_v3_bos = fn
        break

# Get S15_VWAP signal function
signals_s15_vwap = None
for n, fn, mb in BOOK_STRATEGIES:
    if n == "S15_VWAP":
        signals_s15_vwap = fn
        break

STRATS = {
    "BOS": {"fn": signals_v3_bos, "tf": 60, "tp_mult": 2.0, "exit": "none", "filter": "none", "mb": 40},
    "NR7": {"fn": signals_v3_nr7, "tf": 60, "tp_mult": 3.0, "exit": "none", "filter": "none", "mb": 40},
}

if signals_s15_vwap:
    STRATS["S15_VWAP"] = {"fn": signals_s15_vwap, "tf": 5, "tp_mult": 3.0, "exit": "atr_trail_1.5", "filter": "none", "mb": 60}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx-dir", type=Path, required=True)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--account", type=float, default=500.0)
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "final_validation.json")
    args = ap.parse_args()

    print(f"FINAL VALIDATION: {len(STRATS)} strategies x {len(PAIRS)} pairs x {args.years} years")
    print(f"Account: ${args.account}, data: {args.fx_dir}\n")

    all_trades: dict[str, list[dict]] = {s: [] for s in STRATS}

    for pair, filename in PAIRS.items():
        fpath = args.fx_dir / filename
        if not fpath.exists():
            print(f"  {pair}: SKIP (no file)")
            continue
        print(f"Loading {pair}...", end=" ", flush=True)
        df = pd.read_csv(fpath, parse_dates=["datetime"])
        df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
        if args.years > 0:
            cutoff = df["timestamp"].max() - pd.Timedelta(days=args.years * 365)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        df_m1 = df
        del df
        print(f"{len(df_m1):,} M1 bars")

        pip = pip_for_pair(pair)
        spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip
        hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))

        dfs: dict[int, pd.DataFrame] = {}
        for sname, cfg in STRATS.items():
            tf_n = cfg["tf"]
            if tf_n not in dfs:
                dfs[tf_n] = prepare_tf(df_m1, pair, tf_n)

        del df_m1
        gc.collect()

        for sname, cfg in STRATS.items():
            df = dfs[cfg["tf"]]
            try:
                raw = cfg["fn"](df, pair, pip, spread)
            except Exception as e:
                print(f"  {sname} error: {e}")
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
            count = 0
            for k in range(len(sim)):
                t = sim[k].copy()
                t["timestamp"] = str(tss[f_idx[k]])
                t["pair"] = pair
                t["strategy"] = sname
                t["risk_pips"] = abs(f_ent[k] - f_sl[k]) / pip
                all_trades[sname].append(t)
                count += 1
            print(f"  {sname:10s} {pair}: {count} trades")

        for df in dfs.values():
            del df
        gc.collect()

    report: dict = {"account": args.account, "years": args.years}

    # Per-strategy analysis
    for sname, trades in all_trades.items():
        trades.sort(key=lambda t: t["timestamp"])
        m = summarize_trades(trades, 30.0)
        print(f"\n{'='*60}")
        print(f"{sname}: N={m['n']}  PF={m['pf']:.2f}  WR={m['wr']*100:.0f}%  pips={m['total_pips']:+,.1f}")

        # Per year
        by_year: dict[str, list] = {}
        for t in trades:
            by_year.setdefault(t["timestamp"][:4], []).append(t)
        print(f"\n  Per year:")
        yr_data = {}
        for yr in sorted(by_year):
            ym = summarize_trades(by_year[yr], 30.0)
            print(f"    {yr}: N={ym['n']:5d}  PF={ym['pf']:5.2f}  WR={ym['wr']*100:.0f}%  pips={ym['total_pips']:+,.1f}")
            yr_data[yr] = {"n": ym["n"], "pf": ym["pf"], "pips": ym["total_pips"]}

        # Per pair
        by_pair: dict[str, list] = {}
        for t in trades:
            by_pair.setdefault(t["pair"], []).append(t)
        print(f"\n  Per pair:")
        pair_data = {}
        for p in sorted(by_pair):
            pm = summarize_trades(by_pair[p], 30.0)
            print(f"    {p}: N={pm['n']:5d}  PF={pm['pf']:5.2f}  WR={pm['wr']*100:.0f}%  pips={pm['total_pips']:+,.1f}")
            pair_data[p] = {"n": pm["n"], "pf": pm["pf"], "pips": pm["total_pips"]}

        # Win/loss streaks
        max_win_streak = 0
        max_loss_streak = 0
        cur_w, cur_l = 0, 0
        for t in trades:
            if t["pnl_pips"] > 0:
                cur_w += 1
                cur_l = 0
                max_win_streak = max(max_win_streak, cur_w)
            else:
                cur_l += 1
                cur_w = 0
                max_loss_streak = max(max_loss_streak, cur_l)
        print(f"\n  Max win streak: {max_win_streak}, Max loss streak: {max_loss_streak}")

        report[sname] = {
            "total": {"n": m["n"], "pf": m["pf"], "wr": round(m["wr"]*100, 1), "pips": m["total_pips"]},
            "per_year": yr_data,
            "per_pair": pair_data,
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        }

    # Combined BOS+NR7 equity simulation
    print(f"\n{'='*60}")
    print(f"COMBINED BOS+NR7 EQUITY SIM: ${args.account} start, 2% risk")
    print(f"{'='*60}")

    combined = sorted(all_trades["BOS"] + all_trades["NR7"], key=lambda t: t["timestamp"])
    equity = args.account
    peak = equity
    max_dd_pct = 0.0
    by_month: dict[str, float] = {}
    worst_month = ("", 0.0)
    best_month = ("", 0.0)

    for t in combined:
        risk_pct = 2.0
        risk_usd = equity * risk_pct / 100
        risk_p = t.get("risk_pips", 40)
        if risk_p < 1:
            risk_p = 40
        pip_val = risk_usd / risk_p
        pnl_usd = t["pnl_pips"] * pip_val
        equity += pnl_usd
        if equity < 10:
            equity = 10
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd_pct = max(max_dd_pct, dd)

        mk = t["timestamp"][:7]
        by_month[mk] = by_month.get(mk, 0) + pnl_usd

    print(f"Final equity: ${equity:,.2f}")
    print(f"Total return: {(equity - args.account) / args.account * 100:+,.1f}%")
    print(f"Max drawdown: {max_dd_pct:.1f}%")
    print(f"Total trades: {len(combined)}")

    # Monthly returns as % of running equity
    running = args.account
    print(f"\nMonthly returns:")
    pos_months = 0
    neg_months = 0
    for mk in sorted(by_month):
        pnl = by_month[mk]
        pct = pnl / running * 100 if running > 0 else 0
        running += pnl
        if pct > 0:
            pos_months += 1
        else:
            neg_months += 1
        if pct < worst_month[1]:
            worst_month = (mk, pct)
        if pct > best_month[1]:
            best_month = (mk, pct)
        if mk[-2:] in ("01", "04", "07", "10") or mk >= "2026":
            print(f"  {mk}: ${running:>10,.2f} ({pct:+6.1f}%)")

    print(f"\nPositive months: {pos_months}, Negative months: {neg_months}")
    print(f"Best month: {best_month[0]} ({best_month[1]:+.1f}%)")
    print(f"Worst month: {worst_month[0]} ({worst_month[1]:+.1f}%)")

    report["combined_bos_nr7"] = {
        "final_equity": round(equity, 2),
        "total_return_pct": round((equity - args.account) / args.account * 100, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "trades": len(combined),
        "positive_months": pos_months,
        "negative_months": neg_months,
        "best_month": best_month,
        "worst_month": worst_month,
    }

    # S15_VWAP verification - check for look-ahead or bugs
    if "S15_VWAP" in all_trades and all_trades["S15_VWAP"]:
        vwap_trades = all_trades["S15_VWAP"]
        print(f"\n{'='*60}")
        print(f"S15_VWAP VERIFICATION")
        print(f"{'='*60}")
        # Check win rate distribution - if too high, likely a bug
        wins = sum(1 for t in vwap_trades if t["pnl_pips"] > 0)
        losses = sum(1 for t in vwap_trades if t["pnl_pips"] <= 0)
        avg_win = np.mean([t["pnl_pips"] for t in vwap_trades if t["pnl_pips"] > 0]) if wins > 0 else 0
        avg_loss = np.mean([abs(t["pnl_pips"]) for t in vwap_trades if t["pnl_pips"] < 0]) if losses > 0 else 0
        print(f"  Wins: {wins}, Losses: {losses}, WR: {wins/(wins+losses)*100:.1f}%")
        print(f"  Avg win: {avg_win:+.1f} pips, Avg loss: {avg_loss:.1f} pips")
        print(f"  Reward:Risk ratio: {avg_win/avg_loss:.1f}:1" if avg_loss > 0 else "  No losses!")

        # Check exit reasons
        exits = {}
        for t in vwap_trades:
            exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1
        print(f"  Exit reasons: {exits}")

        # Sample trades
        print(f"\n  First 5 trades:")
        for t in vwap_trades[:5]:
            print(f"    {t['timestamp']} {t['pair']} pnl={t['pnl_pips']:+.1f} exit={t['exit_reason']} risk={t.get('risk_pips',0):.0f}p")
        print(f"  Last 5 trades:")
        for t in vwap_trades[-5:]:
            print(f"    {t['timestamp']} {t['pair']} pnl={t['pnl_pips']:+.1f} exit={t['exit_reason']} risk={t.get('risk_pips',0):.0f}p")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
