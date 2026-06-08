#!/usr/bin/env python3
"""Phase 2: Train LightGBM long + short classifiers on parquet features.

Memory-optimised for 16 GB: loads one pair at a time, samples, converts to
float32 numpy, frees the DataFrame, then moves to the next pair.
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
sys.path.insert(0, str(ROOT / "src"))

from scalp_mode.ml.bar_features import FEATURE_COLUMNS  # noqa: E402

ML_DIR = ROOT / "data" / "ml"
READ_COLS = ["timestamp", "label_long", "label_short"] + FEATURE_COLUMNS
N_FEAT = len(FEATURE_COLUMNS)


def extract_pair(path: Path, train_end_ts, val_end_ts, max_tv: int, max_te: int):
    """Load one parquet, split+sample, return numpy arrays, free DataFrame."""
    df = pd.read_parquet(path, columns=READ_COLS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    tv = df[df["timestamp"] <= val_end_ts]
    te = df[df["timestamp"] > val_end_ts]
    del df
    gc.collect()

    if len(tv) > max_tv:
        tv = tv.sample(n=max_tv, random_state=42).sort_values("timestamp")
    if len(te) > max_te:
        te = te.sample(n=max_te, random_state=43).sort_values("timestamp")

    is_train = (tv["timestamp"] <= train_end_ts).values  # bool mask

    X_tv = tv[FEATURE_COLUMNS].values.astype(np.float32)
    yl_tv = tv["label_long"].values.astype(np.int8)
    ys_tv = tv["label_short"].values.astype(np.int8)
    del tv
    gc.collect()

    X_te = te[FEATURE_COLUMNS].values.astype(np.float32)
    yl_te = te["label_long"].values.astype(np.int8)
    ys_te = te["label_short"].values.astype(np.int8)
    del te
    gc.collect()

    return is_train, X_tv, yl_tv, ys_tv, X_te, yl_te, ys_te


def metrics(y_true, y_prob, thresh=0.5):
    from sklearn.metrics import roc_auc_score
    y_pred = (y_prob >= thresh).astype(int)
    acc = float((y_pred == y_true).mean())
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    return acc, auc


def predict_chunks(booster, X, chunk=300_000):
    out = np.empty(len(X), dtype=np.float64)
    for i in range(0, len(X), chunk):
        out[i : i + chunk] = booster.predict(X[i : i + chunk])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml-dir", type=Path, default=ML_DIR)
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=[
            "EUR_USD",
            "GBP_USD",
            "USD_JPY",
            "USD_CAD",
            "AUD_USD",
            "NZD_USD",
        ],
    )
    ap.add_argument("--train-end", default="2025-11-30 23:59:59")
    ap.add_argument("--val-end", default="2026-02-28 23:59:59")
    ap.add_argument("--max-tv-per-pair", type=int, default=500_000)
    ap.add_argument("--max-te-per-pair", type=int, default=250_000)
    ap.add_argument("--only", choices=["long", "short"], default=None,
                    help="Train only one model (skip the other)")
    args = ap.parse_args()

    import lightgbm as lgb

    val_end_ts = pd.Timestamp(args.val_end, tz="UTC")
    train_end_ts = pd.Timestamp(args.train_end, tz="UTC")

    # Collect numpy arrays one pair at a time to cap peak RAM
    all_is_train, all_X_tv, all_yl_tv, all_ys_tv = [], [], [], []
    all_X_te, all_yl_te, all_ys_te = [], [], []

    for p in args.pairs:
        path = args.ml_dir / f"features_{p}.parquet"
        if not path.exists():
            print(f"  WARN: missing {path}", flush=True)
            continue
        itr, xtv, yltv, ystv, xte, ylte, yste = extract_pair(
            path, train_end_ts, val_end_ts,
            args.max_tv_per_pair, args.max_te_per_pair,
        )
        print(f"  {p}: tv={len(xtv):,} (train={itr.sum():,} val={len(xtv)-itr.sum():,}) "
              f"te={len(xte):,}", flush=True)
        all_is_train.append(itr)
        all_X_tv.append(xtv)
        all_yl_tv.append(yltv)
        all_ys_tv.append(ystv)
        all_X_te.append(xte)
        all_yl_te.append(ylte)
        all_ys_te.append(yste)

    print("Concatenating...", flush=True)
    tr_mask = np.concatenate(all_is_train)
    va_mask = ~tr_mask
    X_tv = np.nan_to_num(np.vstack(all_X_tv), nan=0.0, posinf=0.0, neginf=0.0)
    y_long_tv = np.concatenate(all_yl_tv)
    y_short_tv = np.concatenate(all_ys_tv)
    X_te = np.nan_to_num(np.vstack(all_X_te), nan=0.0, posinf=0.0, neginf=0.0)
    y_long_te = np.concatenate(all_yl_te)
    y_short_te = np.concatenate(all_ys_te)
    del all_is_train, all_X_tv, all_yl_tv, all_ys_tv
    del all_X_te, all_yl_te, all_ys_te
    gc.collect()

    mem_gb = (X_tv.nbytes + X_te.nbytes) / 1e9
    print(f"Train: {tr_mask.sum():,}  Val: {va_mask.sum():,}  "
          f"Test: {len(X_te):,}  RAM(features): {mem_gb:.2f} GB", flush=True)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }

    report = {
        "pairs": args.pairs,
        "n_train": int(tr_mask.sum()),
        "n_val": int(va_mask.sum()),
        "n_test": int(len(X_te)),
        "features": FEATURE_COLUMNS,
    }

    models_to_train = [
        ("long", y_long_tv, y_long_te),
        ("short", y_short_tv, y_short_te),
    ]
    if args.only:
        models_to_train = [(n, ytv, yte) for n, ytv, yte in models_to_train if n == args.only]

    for name, y_tv, y_te in models_to_train:
        print(f"\n--- Training {name.upper()} model ---", flush=True)
        train_ds = lgb.Dataset(X_tv[tr_mask], y_tv[tr_mask],
                               feature_name=FEATURE_COLUMNS, free_raw_data=True)
        val_ds = lgb.Dataset(X_tv[va_mask], y_tv[va_mask], reference=train_ds,
                             feature_name=FEATURE_COLUMNS, free_raw_data=True)

        checkpoint_path = args.ml_dir / f"model_{name}_checkpoint.txt"
        args.ml_dir.mkdir(parents=True, exist_ok=True)

        def _checkpoint(env):
            if (env.iteration + 1) % 200 == 0:
                env.model.save_model(str(checkpoint_path))
                print(f"  [checkpoint saved at round {env.iteration + 1}]", flush=True)

        booster = lgb.train(
            params, train_ds, num_boost_round=1500,
            valid_sets=[train_ds, val_ds], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50), _checkpoint],
        )
        del train_ds, val_ds
        gc.collect()

        out_model = args.ml_dir / f"model_{name}.txt"
        args.ml_dir.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(out_model))
        print(f"Saved {out_model} (before eval)", flush=True)

        p_tr = predict_chunks(booster, X_tv[tr_mask])
        p_va = predict_chunks(booster, X_tv[va_mask])
        p_te = predict_chunks(booster, X_te)

        acc_tr, auc_tr = metrics(y_tv[tr_mask], p_tr)
        acc_va, auc_va = metrics(y_tv[va_mask], p_va)
        acc_te, auc_te = metrics(y_te, p_te)

        imp = sorted(
            zip(FEATURE_COLUMNS, booster.feature_importance(importance_type="gain")),
            key=lambda x: -x[1],
        )[:25]

        del booster
        gc.collect()

        report[name] = {
            "train_acc": acc_tr, "train_auc": auc_tr,
            "val_acc": acc_va, "val_auc": auc_va,
            "test_acc": acc_te, "test_auc": auc_te,
            "top_features": [{"name": a, "gain": float(b)} for a, b in imp],
        }

        print(f"\n=== {name.upper()} ===", flush=True)
        print(f"train acc={acc_tr:.4f} auc={auc_tr:.4f} | "
              f"val acc={acc_va:.4f} auc={auc_va:.4f} | "
              f"test acc={acc_te:.4f} auc={auc_te:.4f}", flush=True)
        print("Top features:", flush=True)
        for a, b in imp[:20]:
            print(f"  {a}: {b:.1f}", flush=True)
        print(f"Saved {out_model}", flush=True)

    rep_path = args.ml_dir / "train_report.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {rep_path}", flush=True)


if __name__ == "__main__":
    main()
