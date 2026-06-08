#!/usr/bin/env python3
"""Phase 1: Build ML feature parquet per FX pair from M1 CSVs.

Reads OANDA CSVs from ``data/raw`` (from ``scripts/fetch_historical.py --v2``) or
legacy FX-1-Minute dumps if present.

Writes: data/ml/features_<PAIR>.parquet

Usage:
  python3 ml_features.py
  python3 ml_features.py --pairs EUR_USD --years 2
  python3 ml_features.py --data-dir data/raw --fetch-months 12 \\
      --pairs EUR_USD GBP_USD USD_JPY USD_CAD AUD_USD NZD_USD
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scalp_mode.ml.bar_features import (  # noqa: E402
    FEATURE_COLUMNS,
    SPREAD_PIPS_DEFAULT,
    add_ml_features,
    pip_for_pair,
    spread_half_price,
)
from ml_labels import (  # noqa: E402
    N_FUTURE,
    compute_labels,
    compute_labels_atr,
    compute_labels_atr_tp2,
)

# ML V2 default universe (matches training / live deployment).
V2_PAIRS = [
    "EUR_USD",
    "GBP_USD",
    "USD_JPY",
    "USD_CAD",
    "AUD_USD",
    "NZD_USD",
]

# Legacy filenames under ~/Downloads/FX-1-Minute-Data-master/forex_data/1min/
LEGACY_PAIR_FILES = {
    "EUR_USD": ("EURUSD_1min.csv", "EURUSD.csv"),
    "GBP_USD": ("GBPUSD_1min.csv", "GBPUSD.csv"),
    "USD_JPY": ("USDJPY_1min.csv", "USDJPY.csv"),
    "USD_CAD": ("USDCAD_1min.csv", "USDCAD.csv"),
    "AUD_USD": ("AUDUSD_1min.csv", "AUDUSD.csv"),
    "NZD_USD": ("NZDUSD_1min.csv", "NZDUSD.csv"),
}


def _resolve_csv_path(pair: str, data_dir: Path, fetch_months: int) -> Path | None:
    """Prefer OANDA ``{PAIR}_M1_{N}m.csv``; fall back to legacy FX-1-Minute names."""
    oanda = data_dir / f"{pair}_M1_{fetch_months}m.csv"
    if oanda.exists():
        return oanda
    # Try alternate month counts (e.g. user fetched 11 or 13 months).
    for p in sorted(data_dir.glob(f"{pair}_M1_*m.csv")):
        return p
    names = LEGACY_PAIR_FILES.get(pair)
    if not names:
        return None
    for fn in names:
        cand = data_dir / fn
        if cand.exists():
            return cand
    return data_dir / names[0]


def load_csv(path: Path, years: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else "datetime"
    if ts_col not in df.columns:
        raise ValueError(f"CSV {path} must have 'timestamp' or 'datetime'")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
    if ts_col != "timestamp":
        df = df.drop(columns=[ts_col], errors="ignore")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if years and years > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=int(years * 365))
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"CSV {path} missing required column {col}")
    return df


def process_pair(
    pair: str,
    years: float,
    dry_run: bool,
    data_dir: Path,
    out_dir: Path,
    fetch_months: int,
    with_rules: bool = False,
) -> None:
    path = _resolve_csv_path(pair, data_dir, fetch_months)
    if path is None:
        print(f"SKIP {pair}: could not resolve CSV under {data_dir}")
        return

    t0 = time.time()
    df = load_csv(path, years)
    if df.empty:
        print(f"SKIP {pair}: missing or empty {path}")
        return

    if with_rules:
        g = len(df) // 5
        if g > 0:
            df = df.iloc[: g * 5].copy().reset_index(drop=True)

    pip = pip_for_pair(pair)
    sp = SPREAD_PIPS_DEFAULT.get(pair, 2.0)
    sh = spread_half_price(pair, sp)

    print(f"{pair}: {len(df):,} rows from {path.name} load {(time.time()-t0):.1f}s")
    t1 = time.time()
    feat = add_ml_features(df, pair)
    print(f"  features {(time.time()-t1):.1f}s")

    if with_rules:
        t1b = time.time()
        from ml_rule_features import build_rule_signal_features_for_m1  # noqa: WPS433

        rdf = build_rule_signal_features_for_m1(df, pair)
        for c in rdf.columns:
            feat[c] = rdf[c].values
        print(f"  rule features {(time.time()-t1b):.1f}s")

    t2 = time.time()
    h = feat["high"].values.astype(np.float64)
    l = feat["low"].values.astype(np.float64)
    c = feat["close"].values.astype(np.float64)
    atr = feat["atr14"].values.astype(np.float64)
    long_l, short_l = compute_labels(h, l, c, sh, pip, N_FUTURE)
    feat["label_long"] = long_l.astype(np.int8)
    feat["label_short"] = short_l.astype(np.int8)
    la, sa = compute_labels_atr(h, l, c, atr, sh, pip, N_FUTURE)
    feat["label_long_atr"] = la.astype(np.int8)
    feat["label_short_atr"] = sa.astype(np.int8)
    la2, sa2 = compute_labels_atr_tp2(h, l, c, atr, sh, pip, N_FUTURE)
    feat["label_long_atr2"] = la2.astype(np.int8)
    feat["label_short_atr2"] = sa2.astype(np.int8)
    print(
        f"  labels {(time.time()-t2):.1f}s  "
        f"pos L={long_l.mean():.3f} S={short_l.mean():.3f} | "
        f"atr L={la.mean():.3f} S={sa.mean():.3f}"
    )

    keep = (
        ["timestamp", "pair", "open", "high", "low", "close", "atr14"]
        + FEATURE_COLUMNS
        + [
            "label_long",
            "label_short",
            "label_long_atr",
            "label_short_atr",
            "label_long_atr2",
            "label_short_atr2",
        ]
    )
    if with_rules:
        from ml_rule_features import RULE_M5_COLUMN_NAMES  # noqa: WPS433

        keep = list(keep) + list(RULE_M5_COLUMN_NAMES)
    feat["pair"] = pair
    out = feat[keep].copy()
    warm = 120
    end = max(warm, len(out) - N_FUTURE - 2)
    out = out.iloc[warm:end].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"features_{pair}.parquet"
    if dry_run:
        print(f"  DRY RUN would write {out_path} rows={len(out):,}")
        return

    t3 = time.time()
    out.to_parquet(out_path, index=False, engine="pyarrow", compression="zstd")
    print(f"  wrote {out_path} ({len(out):,} rows) in {(time.time()-t3):.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Directory with OANDA M1 CSVs or legacy FX-1-Minute CSVs",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "ml",
        help="Output directory for parquet",
    )
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=V2_PAIRS,
        help="Pairs to process (default: V2 six-pair set)",
    )
    ap.add_argument(
        "--fetch-months",
        type=int,
        default=12,
        help="Expected OANDA filename suffix {PAIR}_M1_{N}m.csv (default 12)",
    )
    ap.add_argument(
        "--years",
        type=float,
        default=0,
        help="If >0, only last N years of the CSV (0 = use all rows)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--with-rules",
        action="store_true",
        help="Append rule_* columns (trim rows to multiple of 5 M1 bars)",
    )
    args = ap.parse_args()

    for p in args.pairs:
        p = p.upper().replace("/", "_")
        process_pair(
            p,
            args.years,
            args.dry_run,
            args.data_dir,
            args.out_dir,
            args.fetch_months,
            with_rules=args.with_rules,
        )


if __name__ == "__main__":
    main()
