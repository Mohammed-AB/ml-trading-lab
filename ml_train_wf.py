#!/usr/bin/env python3
"""Walk-forward LightGBM training (expanding window, 1-day purge before val).

Each fold: train on all bars with timestamp <= train_end - 1 day (purge),
validate on calendar month (k+1), optional early stopping, save
``wf/model_{long|short}_fold{f}.txt``. Writes ``wf_manifest.json`` with test
windows for :func:`strategy_arena.ml_sweep.run_ml_v2_sweep`.
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
PURGE = pd.Timedelta(days=1)


def _read_cols(label_long: str, label_short: str, use_v2_features: bool) -> list[str]:
    base = ["timestamp", label_long, label_short] + FEATURE_COLUMNS
    if use_v2_features:
        from ml_rule_features import RULE_M5_COLUMN_NAMES  # noqa: WPS433

        base.extend(RULE_M5_COLUMN_NAMES)
    return base


def main() -> None:
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
    ap.add_argument(
        "--label-long",
        default="label_long",
        help="Target column for long model (e.g. label_long_atr)",
    )
    ap.add_argument(
        "--label-short",
        default="label_short",
        help="Target column for short model",
    )
    ap.add_argument(
        "--features-v2",
        action="store_true",
        help="Include rule_* columns (requires ml_features.py --with-rules)",
    )
    ap.add_argument("--min-train-rows", type=int, default=50_000)
    ap.add_argument(
        "--tp2",
        action="store_true",
        help="Train on wider ATR labels (label_long_atr2 / label_short_atr2 from ml_features)",
    )
    args = ap.parse_args()

    if args.tp2:
        args.label_long = "label_long_atr2"
        args.label_short = "label_short_atr2"

    import lightgbm as lgb  # noqa: WPS433

    cols = _read_cols(args.label_long, args.label_short, args.features_v2)

    first = next(
        (args.ml_dir / f"features_{p}.parquet" for p in args.pairs if (args.ml_dir / f"features_{p}.parquet").exists()),
        None,
    )
    if first is None:
        raise SystemExit(f"No features_*.parquet under {args.ml_dir}")
    available = set(pd.read_parquet(first).columns)
    use_cols = [c for c in cols if c in available]
    miss = set(cols) - available
    if miss:
        raise SystemExit(f"Parquet missing columns {sorted(miss)} — run ml_features.py (with --with-rules if --features-v2)")

    feat_names = [c for c in use_cols if c not in ("timestamp", args.label_long, args.label_short)]

    dfs = []
    for p in args.pairs:
        path = args.ml_dir / f"features_{p}.parquet"
        if not path.exists():
            print(f"WARN skip missing {path}", flush=True)
            continue
        d = pd.read_parquet(path, columns=use_cols)
        d["pair"] = p
        dfs.append(d)
    if not dfs:
        raise SystemExit("No parquet inputs found")

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], utc=True)
    all_df = all_df.sort_values("timestamp").reset_index(drop=True)

    months = sorted(all_df["timestamp"].dt.to_period("M").unique())
    if len(months) < 3:
        raise SystemExit("Need at least 3 calendar months in combined data")

    def _tz_ts(period, how="start"):
        """Period → tz-aware UTC Timestamp."""
        t = period.to_timestamp(how=how)
        return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")

    wf_dir = args.ml_dir / "wf"
    wf_dir.mkdir(parents=True, exist_ok=True)

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

    folds_meta: list[dict] = []

    for k in range(len(months) - 2):
        m_train = months[k]
        m_val = months[k + 1]
        m_test = months[k + 2]
        val_start = _tz_ts(m_val, "start")
        val_end = _tz_ts(m_val, "end")
        test_start = _tz_ts(m_test, "start")
        test_end = _tz_ts(m_test, "end")
        train_cutoff = val_start - PURGE

        tr = all_df[all_df["timestamp"] <= train_cutoff]
        va = all_df[(all_df["timestamp"] >= val_start) & (all_df["timestamp"] <= val_end)]
        if len(tr) < args.min_train_rows or len(va) < 5000:
            print(f"  fold{k}: skip (train={len(tr):,} val={len(va):,})", flush=True)
            continue

        X_tr = np.nan_to_num(tr[feat_names].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        X_va = np.nan_to_num(va[feat_names].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        yl_tr = tr[args.label_long].values.astype(np.int8)
        yl_va = va[args.label_long].values.astype(np.int8)
        ys_tr = tr[args.label_short].values.astype(np.int8)
        ys_va = va[args.label_short].values.astype(np.int8)

        for side, y_tr, y_va, name in (
            ("long", yl_tr, yl_va, "long"),
            ("short", ys_tr, ys_va, "short"),
        ):
            train_ds = lgb.Dataset(X_tr, y_tr, feature_name=feat_names, free_raw_data=True)
            val_ds = lgb.Dataset(X_va, y_va, reference=train_ds, feature_name=feat_names, free_raw_data=True)
            booster = lgb.train(
                params,
                train_ds,
                num_boost_round=800,
                valid_sets=[train_ds, val_ds],
                valid_names=["train", "val"],
                callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(0)],
            )
            out_path = wf_dir / f"model_{name}_fold{k}.txt"
            booster.save_model(str(out_path))
            del train_ds, val_ds, booster
            gc.collect()

        folds_meta.append(
            {
                "fold_id": k,
                "train_last_month": str(m_train),
                "val_month": str(m_val),
                "test_month": str(m_test),
                "test_start_utc": test_start.isoformat(),
                "test_end_utc": test_end.isoformat(),
                "model_long": str(wf_dir / f"model_long_fold{k}.txt"),
                "model_short": str(wf_dir / f"model_short_fold{k}.txt"),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
            }
        )
        print(f"  fold{k}: trained train<={train_cutoff.date()} val={m_val} test={m_test}", flush=True)
        del tr, va, X_tr, X_va, yl_tr, yl_va, ys_tr, ys_va
        gc.collect()

    manifest = {
        "feature_names": feat_names,
        "label_long": args.label_long,
        "label_short": args.label_short,
        "folds": folds_meta,
    }
    man_path = args.ml_dir / "wf_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {man_path} ({len(folds_meta)} folds)", flush=True)


if __name__ == "__main__":
    main()
