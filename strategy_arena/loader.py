"""Load M1 OANDA CSVs and prepare DataFrames for book/research strategies."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .config import FETCH_MONTHS, V2_PAIRS  # noqa: E402

# Reuse indicator pipeline from existing backtest
from backtest_strategies import add_indicators, resample_m5  # noqa: E402


def oanda_m1_path(data_dir: Path, pair: str, months: int = FETCH_MONTHS) -> Path:
    return data_dir / f"{pair}_M1_{months}m.csv"


def load_oanda_m1(
    data_dir: Path,
    pair: str,
    months: int = FETCH_MONTHS,
) -> pd.DataFrame:
    path = oanda_m1_path(data_dir, pair, months)
    if not path.exists():
        matches = sorted(data_dir.glob(f"{pair}_M1_*m.csv"))
        if not matches:
            return pd.DataFrame()
        path = matches[0]
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else "datetime"
    if ts_col not in df.columns:
        raise ValueError(f"{path} needs timestamp or datetime")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
    if ts_col != "timestamp":
        df = df.drop(columns=[ts_col], errors="ignore")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 1.0
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"{path} missing {col}")
    return df


def prepare_m5(df_m1: pd.DataFrame, pair: str) -> pd.DataFrame:
    """M5 bars with indicators (used by most book strategies)."""
    df = resample_m5(df_m1)
    return add_indicators(df, pair)


def list_available_pairs(data_dir: Path) -> list[str]:
    out: list[str] = []
    for p in V2_PAIRS:
        if oanda_m1_path(data_dir, p).exists():
            out.append(p)
            continue
        if list(data_dir.glob(f"{p}_M1_*m.csv")):
            out.append(p)
    return out
