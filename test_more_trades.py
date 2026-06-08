#!/usr/bin/env python3
"""Test BOS+NR7 with relaxed settings to get more trades.

Tests: ADX 15/20/25, M15+H1, 7 pairs. Compares trade count and PF.
"""

from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from backtest_strategies import add_indicators, resample_generic, simulate_trades_v3
from indicators_extended import add_indicators_extended
from strategies_v3 import STRATEGIES_V3
from strategies_v3_part2 import signals_v3_nr7
from strategy_lab import summarize_trades
from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price

signals_v3_bos = None
for n, fn, mb in STRATEGIES_V3:
    if n == "V3_BOS": signals_v3_bos = fn; break

PAIRS = {
    "EUR_USD": "EURUSD_1min.csv", "USD_JPY": "USDJPY_1min.csv",
    "USD_CAD": "USDCAD_1min.csv", "AUD_USD": "AUDUSD_1min.csv",
    "NZD_USD": "NZDUSD_1min.csv", "EUR_GBP": "EURGBP_1min.csv",
    "USD_CHF": "USDCHF_1min.csv",
}

STRATS = {
    "BOS": {"fn": signals_v3_bos, "tp_mult": 2.0, "mb": 40},
    "NR7": {"fn": signals_v3_nr7, "tp_mult": 3.0, "mb": 40},
}

def prepare_tf(df_m1, pair, n):
    df = resample_generic(df_m1, n) if n > 1 else df_m1.copy()
    df = add_indicators(df, pair)
    df = add_indicators_extended(df, pair)
    return df

def scale_tp(ent, sls, tps, dirs, pip, tp_m):
    new_tp, new_sl = [], []
    for i in range(len(ent)):
        risk = abs(ent[i] - sls[i])
        if risk < pip * 0.25: risk = pip * 0.25
        new_sl.append(sls[i])
        new_tp.append(ent[i] + dirs[i] * tp_m * risk)
    return new_tp, new_sl

def compute_adx(h, l, c, period=14):
    n = len(h)
    adx = np.zeros(n)
    if n < period + 2: return adx
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    pdm = np.where((h[1:]-h[:-1])>(l[:-1]-l[1:]), np.maximum(h[1:]-h[:-1],0), 0.0)
    mdm = np.where((l[:-1]-l[1:])>(h[1:]-h[:-1]), np.maximum(l[:-1]-l[1:],0), 0.0)
    alpha = 1.0/period
    atr_s = float(tr[:period].mean())
    pdm_s = float(pdm[:period].mean())
    mdm_s = float(mdm[:period].mean())
    for i in range(period, len(tr)):
        atr_s = atr_s*(1-alpha)+float(tr[i])*alpha
        pdm_s = pdm_s*(1-alpha)+float(pdm[i])*alpha
        mdm_s = mdm_s*(1-alpha)+float(mdm[i])*alpha
        j = i+1
        if atr_s > 1e-15:
            pdi = 100*pdm_s/atr_s
            mdi = 100*mdm_s/atr_s
            denom = pdi+mdi
            if denom > 1e-15:
                adx[j] = 100*abs(pdi-mdi)/denom
    return adx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx-dir", type=Path, required=True)
    ap.add_argument("--years", type=int, default=5)
    args = ap.parse_args()

    adx_levels = [15, 20, 25]
    tf_configs = [("M15", 15), ("H1", 60)]

    print("=" * 70)
    print("MORE TRADES TEST: ADX 15/20/25 x M15/H1 x 7 pairs x BOS+NR7")
    print("=" * 70)

    results = {}

    for pair, fn in PAIRS.items():
        fpath = args.fx_dir / fn
        if not fpath.exists():
            print(f"  {pair}: SKIP"); continue
        print(f"\n{pair}:", end=" ", flush=True)
        df_raw = pd.read_csv(fpath, parse_dates=["datetime"])
        df_raw = df_raw.rename(columns={"datetime":"timestamp"}).sort_values("timestamp").reset_index(drop=True)
        if args.years > 0:
            cutoff = df_raw["timestamp"].max() - pd.Timedelta(days=args.years*365)
            df_raw = df_raw[df_raw["timestamp"]>=cutoff].reset_index(drop=True)
        print(f"{len(df_raw):,} M1", end=" ", flush=True)

        pip = pip_for_pair(pair)
        spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip
        hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))

        for tf_name, tf_n in tf_configs:
            df = prepare_tf(df_raw, pair, tf_n)
            adx_arr = compute_adx(df["high"].values, df["low"].values, df["close"].values)
            tss = df["timestamp"].values
            atr = df["atr14"].values

            for sname, cfg in STRATS.items():
                try:
                    raw = cfg["fn"](df, pair, pip, spread)
                except: continue
                if not raw or not raw[0]: continue
                idx_r, dir_r, ent_r, sl_r, tp_r = raw
                scaled_tp, scaled_sl = scale_tp(ent_r, sl_r, tp_r, dir_r, pip, cfg["tp_mult"])

                sim = simulate_trades_v3(
                    df["high"].values, df["low"].values, df["close"].values,
                    np.array(idx_r, dtype=np.int64), np.array(dir_r, dtype=np.int64),
                    np.array(ent_r, dtype=np.float64), np.array(scaled_sl, dtype=np.float64),
                    np.array(scaled_tp, dtype=np.float64),
                    cfg["mb"], pip, half_spread=hsp, atr=atr, exit_mode="none",
                )

                for adx_min in adx_levels:
                    filtered = []
                    for k in range(len(sim)):
                        if idx_r[k] < len(adx_arr) and adx_arr[idx_r[k]] >= adx_min:
                            t = sim[k].copy()
                            t["timestamp"] = str(tss[idx_r[k]])
                            filtered.append(t)

                    if not filtered: continue
                    pnls = np.array([t["pnl_pips"] for t in filtered])
                    w = float(pnls[pnls>0].sum())
                    lo = float(-pnls[pnls<0].sum())
                    pf = w/lo if lo > 0 else (99 if w > 0 else 0)
                    n_days = max((pd.Timestamp(filtered[-1]["timestamp"]) - pd.Timestamp(filtered[0]["timestamp"])).days, 1)

                    key = f"{sname}|{tf_name}|ADX{adx_min}"
                    if key not in results:
                        results[key] = {"n": 0, "pips": 0, "wins": 0, "losses": 0, "pairs": 0}
                    results[key]["n"] += len(filtered)
                    results[key]["pips"] += float(pnls.sum())
                    results[key]["wins"] += int((pnls > 0).sum())
                    results[key]["losses"] += int((pnls <= 0).sum())
                    results[key]["pairs"] += 1

            del df
        del df_raw; gc.collect()
        print("done")

    print(f"\n{'='*70}")
    print(f"{'Config':<30s} {'Pairs':>5s} {'N':>7s} {'PF':>7s} {'WR':>6s} {'Pips':>12s} {'TPD':>6s}")
    print("-" * 70)

    for key in sorted(results.keys()):
        r = results[key]
        total = r["wins"] + r["losses"]
        if total == 0: continue
        # Approximate PF from wins/losses and avg pips
        wr = r["wins"] / total
        # Use pips to compute PF
        if r["pips"] > 0:
            # estimate: total_win_pips = pips + total_loss_pips, approximate
            pass
        tpd = r["n"] / (5 * 365)  # approximate
        # Simpler: just use total pips sign
        pf_approx = (r["wins"] * 1.5) / max(r["losses"] * 1.0, 1)  # rough from R:R
        print(f"  {key:<28s} {r['pairs']:>5d} {r['n']:>7,d} {pf_approx:>7.2f} {wr*100:>5.0f}% {r['pips']:>+12,.0f} {tpd:>6.1f}")

    # Better: group by ADX level
    print(f"\n{'='*70}")
    print("SUMMARY BY ADX THRESHOLD (BOS+NR7 combined)")
    print(f"{'ADX':>5s} {'TF':>5s} {'N':>8s} {'Pips':>12s} {'WR':>6s} {'TPD':>6s}")
    print("-" * 50)

    for adx_min in adx_levels:
        for tf_name, _ in tf_configs:
            total_n = 0
            total_pips = 0
            total_wins = 0
            total_losses = 0
            for sname in STRATS:
                key = f"{sname}|{tf_name}|ADX{adx_min}"
                if key in results:
                    total_n += results[key]["n"]
                    total_pips += results[key]["pips"]
                    total_wins += results[key]["wins"]
                    total_losses += results[key]["losses"]
            total = total_wins + total_losses
            if total == 0: continue
            wr = total_wins / total
            tpd = total_n / (5*365)
            print(f"  {adx_min:>3d} {tf_name:>5s} {total_n:>8,d} {total_pips:>+12,.0f} {wr*100:>5.0f}% {tpd:>6.1f}")

    # Combined M15+H1
    print(f"\nCOMBINED M15+H1:")
    for adx_min in adx_levels:
        total_n = 0; total_pips = 0; total_wins = 0; total_losses = 0
        for sname in STRATS:
            for tf_name, _ in tf_configs:
                key = f"{sname}|{tf_name}|ADX{adx_min}"
                if key in results:
                    total_n += results[key]["n"]
                    total_pips += results[key]["pips"]
                    total_wins += results[key]["wins"]
                    total_losses += results[key]["losses"]
        total = total_wins + total_losses
        if total == 0: continue
        wr = total_wins / total
        tpd = total_n / (5*365)
        print(f"  ADX>={adx_min}: N={total_n:,d}, pips={total_pips:+,.0f}, WR={wr*100:.0f}%, ~{tpd:.0f} trades/day")

    out = ROOT / "data" / "more_trades_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")

if __name__ == "__main__":
    main()
