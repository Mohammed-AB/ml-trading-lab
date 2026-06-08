#!/usr/bin/env python3
"""ML Bouncer: walk-forward LightGBM filter for BOS + NR7 signals.

Labels: 1 = trade hit TP, 0 = trade hit SL or time exit.
Features: micro-structure snapshot at signal bar.
Training: 6-month train, 1-month test, slide forward.
Optuna: tunes max_depth, learning_rate, num_leaves per window.

Usage:
  python ml_filter.py --fx-dir data/fx_5yr --years 5
"""

from __future__ import annotations

import argparse
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
from indicators_extended import add_indicators_extended  # noqa: E402
from strategies_v3 import STRATEGIES_V3  # noqa: E402
from strategies_v3_part2 import signals_v3_nr7  # noqa: E402
from strategy_lab import summarize_trades  # noqa: E402
from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402

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

PAIRS = {
    "EUR_USD": "EURUSD_1min.csv",
    "USD_JPY": "USDJPY_1min.csv",
    "USD_CAD": "USDCAD_1min.csv",
    "AUD_USD": "AUDUSD_1min.csv",
    "NZD_USD": "NZDUSD_1min.csv",
}

signals_v3_bos = None
for n, fn, mb in STRATEGIES_V3:
    if n == "V3_BOS":
        signals_v3_bos = fn
        break

STRATS = {
    "BOS": {"fn": signals_v3_bos, "tf": 60, "tp_mult": 2.0, "exit": "none", "mb": 40},
    "NR7": {"fn": signals_v3_nr7, "tf": 60, "tp_mult": 3.0, "exit": "none", "mb": 40},
}

FEATURE_COLS = [
    "atr_ratio", "atr_rank", "bb_pos", "dist_bb_upper", "dist_bb_lower",
    "dist_pivot_pp", "dist_cam_r3", "dist_cam_s3",
    "rsi14", "rsi_slope3", "adx", "adx_slope3", "plus_di", "minus_di",
    "di_diff", "ema20_slope_norm", "macd_hist", "macd_hist_slope3",
    "stoch_k", "stoch_d", "williams_r",
    "bar_range_atr", "body_pct", "upper_wick_pct", "lower_wick_pct",
    "volume_ratio", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "hurst", "zscore_30", "cmf", "supertrend_dir",
    "direction",
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
    """Extract micro-structure features at signal bar."""
    i = idx
    c = float(df["close"].iat[i])
    h = float(df["high"].iat[i])
    l = float(df["low"].iat[i])
    o = float(df["open"].iat[i])
    atr = float(df["atr14"].iat[i]) if not np.isnan(df["atr14"].iat[i]) else 1e-8

    atr_sma20 = float(df["atr14"].iloc[max(0, i-20):i+1].mean())
    atr_ratio = atr / atr_sma20 if atr_sma20 > 1e-8 else 1.0
    atr_vals = df["atr14"].iloc[max(0, i-100):i+1].dropna()
    atr_rank = float((atr_vals < atr).sum()) / max(len(atr_vals), 1)

    bb_upper = float(df["bb_upper"].iat[i]) if "bb_upper" in df else c
    bb_lower = float(df["bb_lower"].iat[i]) if "bb_lower" in df else c
    bb_range = bb_upper - bb_lower if bb_upper > bb_lower else 1e-8
    bb_pos = (c - bb_lower) / bb_range

    rsi = float(df["rsi14"].iat[i]) if "rsi14" in df else 50.0
    rsi_slope3 = (rsi - float(df["rsi14"].iat[max(0, i-3)])) if i >= 3 and "rsi14" in df else 0.0

    adx_v = float(df["adx"].iat[i]) if "adx" in df else 0.0
    adx_slope3 = (adx_v - float(df["adx"].iat[max(0, i-3)])) if i >= 3 and "adx" in df else 0.0
    pdi = float(df["plus_di"].iat[i]) if "plus_di" in df else 0.0
    mdi = float(df["minus_di"].iat[i]) if "minus_di" in df else 0.0

    macd_hist = float(df["macd"].iat[i]) - float(df["macd_signal"].iat[i]) if "macd" in df else 0.0
    prev_hist = (float(df["macd"].iat[max(0, i-3)]) - float(df["macd_signal"].iat[max(0, i-3)])) if i >= 3 else 0.0
    macd_hist_slope3 = macd_hist - prev_hist

    ema_slope = float(df["ema20_slope"].iat[i]) if "ema20_slope" in df else 0.0
    ema_slope_norm = ema_slope / atr if atr > 1e-8 else 0.0

    bar_range = h - l
    bar_range_atr = bar_range / atr if atr > 1e-8 else 0.0
    body_pct = abs(c - o) / bar_range if bar_range > 1e-8 else 0.0
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    upper_wick_pct = upper_wick / bar_range if bar_range > 1e-8 else 0.0
    lower_wick_pct = lower_wick / bar_range if bar_range > 1e-8 else 0.0

    vol = float(df["volume"].iat[i]) if "volume" in df else 0.0
    vol_sma20 = float(df["volume"].iloc[max(0, i-20):i+1].mean()) if "volume" in df else 1.0
    vol_ratio = vol / vol_sma20 if vol_sma20 > 0 else 1.0

    hour = int(df["hour"].iat[i]) if "hour" in df else 0
    dow = int(df["dow"].iat[i]) if "dow" in df else 0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 5)
    dow_cos = np.cos(2 * np.pi * dow / 5)

    pivot_pp = float(df["pivot_pp"].iat[i]) if "pivot_pp" in df else c
    cam_r3 = float(df["cam_r3"].iat[i]) if "cam_r3" in df else c
    cam_s3 = float(df["cam_s3"].iat[i]) if "cam_s3" in df else c

    return {
        "atr_ratio": atr_ratio, "atr_rank": atr_rank,
        "bb_pos": bb_pos,
        "dist_bb_upper": (bb_upper - c) / atr if atr > 1e-8 else 0,
        "dist_bb_lower": (c - bb_lower) / atr if atr > 1e-8 else 0,
        "dist_pivot_pp": (c - pivot_pp) / atr if atr > 1e-8 else 0,
        "dist_cam_r3": (cam_r3 - c) / atr if atr > 1e-8 else 0,
        "dist_cam_s3": (c - cam_s3) / atr if atr > 1e-8 else 0,
        "rsi14": rsi, "rsi_slope3": rsi_slope3,
        "adx": adx_v, "adx_slope3": adx_slope3,
        "plus_di": pdi, "minus_di": mdi, "di_diff": pdi - mdi,
        "ema20_slope_norm": ema_slope_norm,
        "macd_hist": macd_hist / atr if atr > 1e-8 else 0,
        "macd_hist_slope3": macd_hist_slope3 / atr if atr > 1e-8 else 0,
        "stoch_k": float(df["stoch_k"].iat[i]) if "stoch_k" in df else 50,
        "stoch_d": float(df["stoch_d"].iat[i]) if "stoch_d" in df else 50,
        "williams_r": float(df["williams_r"].iat[i]) if "williams_r" in df else -50,
        "bar_range_atr": bar_range_atr, "body_pct": body_pct,
        "upper_wick_pct": upper_wick_pct, "lower_wick_pct": lower_wick_pct,
        "volume_ratio": vol_ratio,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "hurst": float(df["hurst"].iat[i]) if "hurst" in df else 0.5,
        "zscore_30": float(df["zscore_30"].iat[i]) if "zscore_30" in df else 0,
        "cmf": float(df["cmf"].iat[i]) if "cmf" in df else 0,
        "supertrend_dir": float(df["supertrend_dir"].iat[i]) if "supertrend_dir" in df else 0,
        "direction": float(direction),
    }


def build_dataset(fx_dir, years):
    """Generate labeled feature dataset from all pairs."""
    rows = []
    for pair, filename in PAIRS.items():
        fpath = fx_dir / filename
        if not fpath.exists():
            continue
        print(f"  Building dataset for {pair}...", end=" ", flush=True)
        df_raw = pd.read_csv(fpath, parse_dates=["datetime"])
        df_raw = df_raw.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
        if years > 0:
            cutoff = df_raw["timestamp"].max() - pd.Timedelta(days=years * 365)
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
                np.array(idx_r, dtype=np.int64),
                np.array(dir_r, dtype=np.int64),
                np.array(ent_r, dtype=np.float64),
                np.array(scaled_sl, dtype=np.float64),
                np.array(scaled_tp, dtype=np.float64),
                cfg["mb"], pip, half_spread=hsp, atr=atr, exit_mode=cfg["exit"],
            )

            count = 0
            for k in range(len(sim)):
                feat = extract_features(df, idx_r[k], dir_r[k])
                label = 1 if sim[k]["exit_reason"] == "tp" else 0
                feat["label"] = label
                feat["pnl_pips"] = sim[k]["pnl_pips"]
                feat["timestamp"] = str(tss[idx_r[k]])
                feat["pair"] = pair
                feat["strategy"] = sname
                rows.append(feat)
                count += 1

        print(f"{count} signals")
        del df
        gc.collect()

    return pd.DataFrame(rows)


def optuna_lgb(X_train, y_train, X_val, y_val, n_trials=30):
    """Use Optuna to find best LightGBM hyperparams."""
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        ds_train = lgb.Dataset(X_train, label=y_train)
        ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_train)
        model = lgb.train(
            params, ds_train, num_boost_round=200,
            valid_sets=[ds_val], callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        preds = model.predict(X_val)
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def walk_forward(dataset, train_months=6, test_months=1):
    """Walk-forward: train on 6 months, test on 1, slide forward."""
    dataset["ts"] = pd.to_datetime(dataset["timestamp"])
    dataset = dataset.sort_values("ts").reset_index(drop=True)

    min_date = dataset["ts"].min()
    max_date = dataset["ts"].max()

    results = []
    window_start = min_date

    while window_start + pd.DateOffset(months=train_months + test_months) <= max_date:
        train_end = window_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        train_mask = (dataset["ts"] >= window_start) & (dataset["ts"] < train_end)
        test_mask = (dataset["ts"] >= train_end) & (dataset["ts"] < test_end)

        train_df = dataset[train_mask]
        test_df = dataset[test_mask]

        if len(train_df) < 100 or len(test_df) < 20:
            window_start += pd.DateOffset(months=test_months)
            continue

        X_train = train_df[FEATURE_COLS].values.astype(np.float32)
        y_train = train_df["label"].values.astype(np.int32)
        X_test = test_df[FEATURE_COLS].values.astype(np.float32)
        y_test = test_df["label"].values.astype(np.int32)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Optuna tuning
        if HAS_OPTUNA and len(train_df) > 200:
            split = int(len(X_train) * 0.8)
            best_params = optuna_lgb(
                X_train[:split], y_train[:split],
                X_train[split:], y_train[split:],
                n_trials=20,
            )
        else:
            best_params = {"max_depth": 5, "num_leaves": 31, "learning_rate": 0.05}

        params = {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            **best_params,
        }

        ds_train = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, ds_train, num_boost_round=150)

        probs = model.predict(X_test)

        for threshold in [0.45, 0.50, 0.55, 0.60]:
            mask = probs >= threshold
            if mask.sum() < 5:
                continue
            filtered_test = test_df.iloc[mask.nonzero()[0]]
            pnls = filtered_test["pnl_pips"].values
            wins = (pnls > 0).sum()
            losses = (pnls <= 0).sum()
            total_pips = float(pnls.sum())
            w_sum = float(pnls[pnls > 0].sum())
            l_sum = float(-pnls[pnls < 0].sum())
            pf = w_sum / l_sum if l_sum > 0 else (99.0 if w_sum > 0 else 0.0)

            unfiltered_pnls = test_df["pnl_pips"].values
            uf_w = float(unfiltered_pnls[unfiltered_pnls > 0].sum())
            uf_l = float(-unfiltered_pnls[unfiltered_pnls < 0].sum())
            uf_pf = uf_w / uf_l if uf_l > 0 else 0

            results.append({
                "window": f"{window_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}",
                "train_n": len(train_df),
                "test_n": len(test_df),
                "threshold": threshold,
                "filtered_n": int(mask.sum()),
                "filtered_pf": round(pf, 3),
                "filtered_pips": round(total_pips, 1),
                "filtered_wr": round(wins / max(wins + losses, 1), 3),
                "unfiltered_pf": round(uf_pf, 3),
                "unfiltered_n": len(test_df),
                "improvement": round(pf - uf_pf, 3) if uf_pf < 90 else 0,
            })

        window_start += pd.DateOffset(months=test_months)

    return results


def main():
    ap = argparse.ArgumentParser(description="ML Bouncer filter for BOS+NR7")
    ap.add_argument("--fx-dir", type=Path, required=True)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "ml_filter_results.json")
    args = ap.parse_args()

    if not HAS_LGB:
        print("ERROR: lightgbm not installed. pip install lightgbm", file=sys.stderr)
        return 1

    print("="*60)
    print("ML BOUNCER: Walk-Forward Filter for BOS + NR7")
    print("="*60)

    print("\n1. Building labeled dataset...")
    dataset = build_dataset(args.fx_dir, args.years)
    print(f"   Total samples: {len(dataset)}")
    print(f"   Label distribution: {dataset['label'].value_counts().to_dict()}")
    print(f"   Win rate (unfiltered): {dataset['label'].mean()*100:.1f}%")

    print(f"\n2. Walk-forward training (6-month train, 1-month test)...")
    wf_results = walk_forward(dataset)

    print(f"\n3. Results across {len(wf_results)} test windows:")
    print(f"{'Window':>25s} {'Thr':>5s} {'N_filt':>7s} {'PF_filt':>8s} {'PF_raw':>7s} {'Improve':>8s} {'Pips':>8s}")
    print("-" * 75)

    total_filtered_pips = 0
    total_unfiltered_pips = 0
    total_filtered_n = 0
    total_unfiltered_n = 0

    best_threshold = 0.50
    best_by_thr: dict[float, list] = {}

    for r in wf_results:
        best_by_thr.setdefault(r["threshold"], []).append(r)
        if r["threshold"] == 0.50:
            print(f"  {r['window']:>25s} {r['threshold']:>5.2f} {r['filtered_n']:>7d} {r['filtered_pf']:>8.2f} {r['unfiltered_pf']:>7.2f} {r['improvement']:>+8.3f} {r['filtered_pips']:>+8.1f}")
            total_filtered_pips += r["filtered_pips"]
            total_filtered_n += r["filtered_n"]

    print(f"\n4. Summary by threshold:")
    for thr in sorted(best_by_thr):
        entries = best_by_thr[thr]
        avg_pf = np.mean([r["filtered_pf"] for r in entries])
        total_pips = sum(r["filtered_pips"] for r in entries)
        total_n = sum(r["filtered_n"] for r in entries)
        avg_improve = np.mean([r["improvement"] for r in entries])
        wins = sum(1 for r in entries if r["filtered_pf"] > r["unfiltered_pf"])
        print(f"  thr={thr:.2f}: avg_PF={avg_pf:.2f}, total_pips={total_pips:+,.0f}, N={total_n}, avg_improve={avg_improve:+.3f}, better_in={wins}/{len(entries)} windows")

    report = {
        "dataset_size": len(dataset),
        "label_distribution": dataset["label"].value_counts().to_dict(),
        "unfiltered_wr": round(dataset["label"].mean(), 4),
        "wf_results": wf_results,
        "summary_by_threshold": {
            str(thr): {
                "avg_pf": round(np.mean([r["filtered_pf"] for r in entries]), 3),
                "total_pips": round(sum(r["filtered_pips"] for r in entries), 1),
                "total_n": sum(r["filtered_n"] for r in entries),
                "windows_improved": sum(1 for r in entries if r["filtered_pf"] > r["unfiltered_pf"]),
                "total_windows": len(entries),
            }
            for thr, entries in best_by_thr.items()
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
