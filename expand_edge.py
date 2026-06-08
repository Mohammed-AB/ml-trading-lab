#!/usr/bin/env python3
"""Expand the edge: extra pairs, S15_VWAP verify, mean-reversion complement, ML filter all.

1. Test BOS+NR7 on EUR_GBP and USD_CHF (new pairs)
2. Verify S15_VWAP with detailed trade inspection
3. Add V3_ZSCORE (mean reversion) as range complement
4. ML Bouncer walk-forward on the expanded set
5. Correlation check: how often do signals cluster
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
    STRATEGIES as BOOK_STRATEGIES,
    add_indicators, resample_generic, simulate_trades_v3,
)
from indicators_extended import add_indicators_extended  # noqa: E402
from signal_filters import apply_filters  # noqa: E402
from strategies_v3 import STRATEGIES_V3  # noqa: E402
from strategies_v3_part2 import signals_v3_nr7, signals_v3_zscore  # noqa: E402
from strategy_lab import summarize_trades  # noqa: E402
from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402

import argparse

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Strategy functions
signals_v3_bos = None
for n, fn, mb in STRATEGIES_V3:
    if n == "V3_BOS":
        signals_v3_bos = fn
        break
signals_s15_vwap = None
for n, fn, mb in BOOK_STRATEGIES:
    if n == "S15_VWAP":
        signals_s15_vwap = fn
        break

ALL_PAIRS = {
    "EUR_USD": "EURUSD_1min.csv",
    "USD_JPY": "USDJPY_1min.csv",
    "USD_CAD": "USDCAD_1min.csv",
    "AUD_USD": "AUDUSD_1min.csv",
    "NZD_USD": "NZDUSD_1min.csv",
    "EUR_GBP": "EURGBP_1min.csv",
    "USD_CHF": "USDCHF_1min.csv",
}

STRATS = {
    "BOS": {"fn": signals_v3_bos, "tf": 60, "tp_mult": 2.0, "exit": "none", "mb": 40},
    "NR7": {"fn": signals_v3_nr7, "tf": 60, "tp_mult": 3.0, "exit": "none", "mb": 40},
    "ZSCORE": {"fn": signals_v3_zscore, "tf": 60, "tp_mult": 1.5, "exit": "none", "mb": 40},
}

FEATURE_COLS = [
    "atr_ratio", "atr_rank", "bb_pos", "dist_bb_upper", "dist_bb_lower",
    "dist_pivot_pp", "dist_cam_r3", "dist_cam_s3",
    "rsi14", "rsi_slope3", "adx", "adx_slope3", "plus_di", "minus_di",
    "di_diff", "ema20_slope_norm", "macd_hist", "macd_hist_slope3",
    "stoch_k", "stoch_d", "williams_r",
    "bar_range_atr", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "volume_ratio", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "hurst", "zscore_30", "cmf", "supertrend_dir", "direction",
]


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


def extract_features(df, idx, direction):
    """Micro-structure features at signal bar."""
    i = idx
    c = float(df["close"].iat[i])
    h = float(df["high"].iat[i])
    l = float(df["low"].iat[i])
    o = float(df["open"].iat[i])
    atr = float(df["atr14"].iat[i]) if not np.isnan(df["atr14"].iat[i]) else 1e-8
    if atr < 1e-8:
        atr = 1e-8

    atr_sma20 = float(df["atr14"].iloc[max(0, i-20):i+1].mean())
    atr_ratio = atr / atr_sma20 if atr_sma20 > 1e-8 else 1.0
    atr_vals = df["atr14"].iloc[max(0, i-100):i+1].dropna()
    atr_rank = float((atr_vals < atr).sum()) / max(len(atr_vals), 1)

    bb_u = float(df["bb_upper"].iat[i]) if "bb_upper" in df else c
    bb_l = float(df["bb_lower"].iat[i]) if "bb_lower" in df else c
    bb_r = bb_u - bb_l if bb_u > bb_l else 1e-8
    rsi = float(df["rsi14"].iat[i]) if "rsi14" in df else 50.0
    rsi_s3 = (rsi - float(df["rsi14"].iat[max(0, i-3)])) if i >= 3 and "rsi14" in df else 0.0
    adx_v = float(df["adx"].iat[i]) if "adx" in df else 0.0
    adx_s3 = (adx_v - float(df["adx"].iat[max(0, i-3)])) if i >= 3 and "adx" in df else 0.0
    pdi = float(df["plus_di"].iat[i]) if "plus_di" in df else 0.0
    mdi = float(df["minus_di"].iat[i]) if "minus_di" in df else 0.0
    mh = float(df["macd"].iat[i]) - float(df["macd_signal"].iat[i]) if "macd" in df else 0.0
    mh_prev = (float(df["macd"].iat[max(0, i-3)]) - float(df["macd_signal"].iat[max(0, i-3)])) if i >= 3 else 0.0
    es = float(df["ema20_slope"].iat[i]) if "ema20_slope" in df else 0.0
    br = h - l
    vol = float(df["volume"].iat[i]) if "volume" in df else 0.0
    vol_sma = float(df["volume"].iloc[max(0, i-20):i+1].mean()) if "volume" in df else 1.0
    hr = int(df["hour"].iat[i]) if "hour" in df else 0
    dw = int(df["dow"].iat[i]) if "dow" in df else 0
    pp = float(df["pivot_pp"].iat[i]) if "pivot_pp" in df else c
    cr3 = float(df["cam_r3"].iat[i]) if "cam_r3" in df else c
    cs3 = float(df["cam_s3"].iat[i]) if "cam_s3" in df else c

    return {
        "atr_ratio": atr_ratio, "atr_rank": atr_rank,
        "bb_pos": (c - bb_l) / bb_r,
        "dist_bb_upper": (bb_u - c) / atr, "dist_bb_lower": (c - bb_l) / atr,
        "dist_pivot_pp": (c - pp) / atr, "dist_cam_r3": (cr3 - c) / atr, "dist_cam_s3": (c - cs3) / atr,
        "rsi14": rsi, "rsi_slope3": rsi_s3, "adx": adx_v, "adx_slope3": adx_s3,
        "plus_di": pdi, "minus_di": mdi, "di_diff": pdi - mdi,
        "ema20_slope_norm": es / atr,
        "macd_hist": mh / atr, "macd_hist_slope3": (mh - mh_prev) / atr,
        "stoch_k": float(df["stoch_k"].iat[i]) if "stoch_k" in df else 50,
        "stoch_d": float(df["stoch_d"].iat[i]) if "stoch_d" in df else 50,
        "williams_r": float(df["williams_r"].iat[i]) if "williams_r" in df else -50,
        "bar_range_atr": br / atr, "body_pct": abs(c - o) / br if br > 1e-8 else 0,
        "upper_wick_pct": (h - max(c, o)) / br if br > 1e-8 else 0,
        "lower_wick_pct": (min(c, o) - l) / br if br > 1e-8 else 0,
        "volume_ratio": vol / vol_sma if vol_sma > 0 else 1,
        "hour_sin": np.sin(2 * np.pi * hr / 24), "hour_cos": np.cos(2 * np.pi * hr / 24),
        "dow_sin": np.sin(2 * np.pi * dw / 5), "dow_cos": np.cos(2 * np.pi * dw / 5),
        "hurst": float(df["hurst"].iat[i]) if "hurst" in df else 0.5,
        "zscore_30": float(df["zscore_30"].iat[i]) if "zscore_30" in df else 0,
        "cmf": float(df["cmf"].iat[i]) if "cmf" in df else 0,
        "supertrend_dir": float(df["supertrend_dir"].iat[i]) if "supertrend_dir" in df else 0,
        "direction": float(direction),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx-dir", type=Path, required=True)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "expanded_edge.json")
    args = ap.parse_args()

    print("=" * 60)
    print("EXPANDED EDGE: 7 pairs, BOS+NR7+ZSCORE, ML filter, correlation check")
    print("=" * 60)

    # ---- Part 1: Raw strategy performance per pair ----
    print("\n1. Raw strategy performance across all 7 pairs...")
    all_rows: list[dict] = []
    pair_signals: dict[str, list[dict]] = {}

    for pair, filename in ALL_PAIRS.items():
        fpath = args.fx_dir / filename
        if not fpath.exists():
            print(f"  {pair}: SKIP")
            continue
        print(f"  {pair}:", end=" ", flush=True)
        df_raw = pd.read_csv(fpath, parse_dates=["datetime"])
        df_raw = df_raw.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
        if args.years > 0:
            cutoff = df_raw["timestamp"].max() - pd.Timedelta(days=args.years * 365)
            df_raw = df_raw[df_raw["timestamp"] >= cutoff].reset_index(drop=True)

        pip = pip_for_pair(pair)
        spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip
        hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))
        df = prepare_tf(df_raw, pair, 60)
        del df_raw
        gc.collect()

        atr = df["atr14"].values
        tss = df["timestamp"].values

        for sname, cfg in STRATS.items():
            try:
                raw = cfg["fn"](df, pair, pip, spread)
            except Exception:
                continue
            if not raw or not raw[0]:
                continue
            idx_r, dir_r, ent_r, sl_r, tp_r = raw
            scaled_tp, scaled_sl = scale_tp(ent_r, sl_r, tp_r, dir_r, pip, cfg["tp_mult"])

            sim = simulate_trades_v3(
                df["high"].values, df["low"].values, df["close"].values,
                np.array(idx_r, dtype=np.int64), np.array(dir_r, dtype=np.int64),
                np.array(ent_r, dtype=np.float64), np.array(scaled_sl, dtype=np.float64),
                np.array(scaled_tp, dtype=np.float64),
                cfg["mb"], pip, half_spread=hsp, atr=atr, exit_mode=cfg["exit"],
            )

            for k in range(len(sim)):
                feat = extract_features(df, idx_r[k], dir_r[k])
                feat["label"] = 1 if sim[k]["exit_reason"] == "tp" else 0
                feat["pnl_pips"] = sim[k]["pnl_pips"]
                feat["timestamp"] = str(tss[idx_r[k]])
                feat["pair"] = pair
                feat["strategy"] = sname
                all_rows.append(feat)

                pair_signals.setdefault(pair, []).append({
                    "ts": str(tss[idx_r[k]]), "strategy": sname, "dir": dir_r[k],
                })

            m = summarize_trades(sim, 30.0)
            print(f"{sname}={m['n']}t/PF{m['pf']:.2f} ", end="")

        print()
        del df
        gc.collect()

    dataset = pd.DataFrame(all_rows)
    print(f"\nTotal dataset: {len(dataset)} signals")

    # Per-strategy per-pair summary
    print("\n  Per strategy x pair:")
    for sname in STRATS:
        sub = dataset[dataset["strategy"] == sname]
        for pair in sorted(ALL_PAIRS):
            ps = sub[sub["pair"] == pair]
            if len(ps) == 0:
                continue
            pnls = ps["pnl_pips"].values
            w = float(pnls[pnls > 0].sum())
            lo = float(-pnls[pnls < 0].sum())
            pf = w / lo if lo > 0 else (99 if w > 0 else 0)
            print(f"    {sname:8s} {pair:8s}: N={len(ps):5d}  PF={pf:5.2f}  pips={pnls.sum():+,.0f}")

    # ---- Part 2: ML Bouncer walk-forward ----
    if HAS_LGB:
        print("\n2. ML Bouncer walk-forward (6m train, 1m test)...")
        dataset["ts"] = pd.to_datetime(dataset["timestamp"])
        dataset = dataset.sort_values("ts").reset_index(drop=True)
        min_d = dataset["ts"].min()
        max_d = dataset["ts"].max()

        wf_results = []
        ws = min_d
        while ws + pd.DateOffset(months=7) <= max_d:
            te = ws + pd.DateOffset(months=6)
            test_e = te + pd.DateOffset(months=1)
            train = dataset[(dataset["ts"] >= ws) & (dataset["ts"] < te)]
            test = dataset[(dataset["ts"] >= te) & (dataset["ts"] < test_e)]
            if len(train) < 80 or len(test) < 10:
                ws += pd.DateOffset(months=1)
                continue

            X_tr = np.nan_to_num(train[FEATURE_COLS].values.astype(np.float32))
            y_tr = train["label"].values.astype(np.int32)
            X_te = np.nan_to_num(test[FEATURE_COLS].values.astype(np.float32))

            params = {"objective": "binary", "metric": "auc", "verbosity": -1,
                      "max_depth": 5, "num_leaves": 31, "learning_rate": 0.05,
                      "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8}

            if HAS_OPTUNA and len(train) > 150:
                split = int(len(X_tr) * 0.8)
                def obj(trial):
                    p = {**params,
                         "max_depth": trial.suggest_int("md", 3, 7),
                         "num_leaves": trial.suggest_int("nl", 8, 48),
                         "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                         "min_child_samples": trial.suggest_int("mcs", 5, 30)}
                    ds = lgb.Dataset(X_tr[:split], y_tr[:split])
                    dv = lgb.Dataset(X_tr[split:], y_tr[split:], reference=ds)
                    m = lgb.train(p, ds, 150, valid_sets=[dv], callbacks=[lgb.early_stopping(15, verbose=False)])
                    from sklearn.metrics import roc_auc_score
                    return roc_auc_score(y_tr[split:], m.predict(X_tr[split:]))
                study = optuna.create_study(direction="maximize")
                study.optimize(obj, n_trials=15, show_progress_bar=False)
                params.update({k: v for k, v in study.best_params.items()
                               if k in ("md", "nl", "lr", "mcs")})
                params["max_depth"] = params.pop("md", 5)
                params["num_leaves"] = params.pop("nl", 31)
                params["learning_rate"] = params.pop("lr", 0.05)
                params["min_child_samples"] = params.pop("mcs", 10)

            mdl = lgb.train(params, lgb.Dataset(X_tr, y_tr), 150)
            probs = mdl.predict(X_te)

            for thr in [0.45, 0.50, 0.55]:
                mask = probs >= thr
                if mask.sum() < 3:
                    continue
                ft = test.iloc[mask.nonzero()[0]]
                pnls = ft["pnl_pips"].values
                w = float(pnls[pnls > 0].sum())
                lo = float(-pnls[pnls < 0].sum())
                pf = w / lo if lo > 0 else (99 if w > 0 else 0)
                uf_pnls = test["pnl_pips"].values
                uf_w = float(uf_pnls[uf_pnls > 0].sum())
                uf_l = float(-uf_pnls[uf_pnls < 0].sum())
                uf_pf = uf_w / uf_l if uf_l > 0 else 0
                wf_results.append({
                    "window": f"{ws.strftime('%Y-%m')}→{test_e.strftime('%Y-%m')}",
                    "thr": thr, "n_filt": int(mask.sum()), "pf_filt": round(pf, 2),
                    "pips_filt": round(float(pnls.sum()), 1),
                    "pf_raw": round(uf_pf, 2), "n_raw": len(test),
                })
            ws += pd.DateOffset(months=1)

        print(f"  {len(wf_results)} test windows")
        for thr in [0.45, 0.50, 0.55]:
            entries = [r for r in wf_results if r["thr"] == thr]
            if not entries:
                continue
            avg_pf = np.mean([r["pf_filt"] for r in entries])
            tot_pips = sum(r["pips_filt"] for r in entries)
            tot_n = sum(r["n_filt"] for r in entries)
            better = sum(1 for r in entries if r["pf_filt"] > r["pf_raw"])
            print(f"  thr={thr:.2f}: avg_PF={avg_pf:.2f}, pips={tot_pips:+,.0f}, N={tot_n}, better={better}/{len(entries)}")
    else:
        wf_results = []

    # ---- Part 3: Correlation check ----
    print("\n3. Signal clustering check...")
    from collections import Counter
    date_counts = Counter()
    for pair, sigs in pair_signals.items():
        for s in sigs:
            date_counts[s["ts"][:10]] += 1

    max_cluster = max(date_counts.values()) if date_counts else 0
    avg_per_day = np.mean(list(date_counts.values())) if date_counts else 0
    days_over_5 = sum(1 for v in date_counts.values() if v > 5)
    print(f"  Max signals on one day: {max_cluster}")
    print(f"  Avg signals/day: {avg_per_day:.1f}")
    print(f"  Days with >5 signals: {days_over_5} (risk of correlation)")

    # ---- Report ----
    report = {
        "pairs": list(ALL_PAIRS.keys()),
        "strategies": list(STRATS.keys()),
        "dataset_size": len(dataset),
        "wf_results_summary": {
            str(thr): {
                "avg_pf": round(np.mean([r["pf_filt"] for r in entries]), 2),
                "total_pips": round(sum(r["pips_filt"] for r in entries), 1),
                "total_n": sum(r["n_filt"] for r in entries),
            }
            for thr in [0.45, 0.50, 0.55]
            if (entries := [r for r in wf_results if r["thr"] == thr])
        },
        "correlation": {
            "max_signals_day": max_cluster,
            "avg_signals_day": round(avg_per_day, 1),
            "days_over_5": days_over_5,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
