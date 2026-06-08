"""ML gate: probability threshold + SL/TP grid on hold-out (same as ml_backtest, multi-R:R).

``run_ml_v2_sweep`` uses ``wf_manifest.json`` + per-fold models and ATR-scaled
SL/TP (see :mod:`ml_train_wf`).
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Project-specific imports are lazy: keep --ml fast when features_*.parquet are absent.


def _sim_trade(
    i: int,
    direction: str,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pip: float,
    spread_half: float,
    sl_pips: float,
    tp_pips: float,
    max_bars: int,
) -> tuple[float, int, str]:
    """Exit at bid/ask: subtract one half-spread (in pips) from favourable exits."""
    exit_half_pips = spread_half / pip
    n = len(close)
    end = min(i + max_bars + 1, n)
    if direction == "long":
        entry = close[i] + spread_half
        tp = entry + tp_pips * pip
        sl = entry - sl_pips * pip
        for j in range(i + 1, end):
            if low[j] <= sl and high[j] >= tp:
                return -sl_pips - exit_half_pips, j - i, "sl_ambiguous"
            if low[j] <= sl:
                return -sl_pips - exit_half_pips, j - i, "sl"
            if high[j] >= tp:
                return tp_pips - exit_half_pips, j - i, "tp"
        j = end - 1
        return (close[j] - spread_half - entry) / pip, end - 1 - i, "time"
    entry = close[i] - spread_half
    tp = entry - tp_pips * pip
    sl = entry + sl_pips * pip
    for j in range(i + 1, end):
        if high[j] >= sl and low[j] <= tp:
            return -sl_pips - exit_half_pips, j - i, "sl_ambiguous"
        if high[j] >= sl:
            return -sl_pips - exit_half_pips, j - i, "sl"
        if low[j] <= tp:
            return tp_pips - exit_half_pips, j - i, "tp"
    j = end - 1
    return (entry - (close[j] + spread_half)) / pip, end - 1 - i, "time"


def _sim_trade_atr(
    i: int,
    direction: str,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    pip: float,
    spread_half: float,
    max_bars: int,
    sl_mult: float = 1.0,
    tp_mult: float = 1.5,
) -> tuple[float, int, str]:
    """SL/TP distances from ATR at bar *i* (price), with exit half-spread like :func:`_sim_trade`."""
    a = float(atr[i]) if np.isfinite(atr[i]) else pip * 0.05
    a = max(a, pip * 0.05)
    sl_pips = (sl_mult * a) / pip
    tp_pips = (tp_mult * a) / pip
    return _sim_trade(
        i, direction, high, low, close, pip, spread_half, sl_pips, tp_pips, max_bars
    )


def _backtest_pair_sltp(
    pair: str,
    h, l, c, ts, pl, ps, thresh, sl_p, tp_p, max_bars: int
) -> list[dict]:
    from scalp_mode.ml.bar_features import (  # noqa: WPS433
        SPREAD_PIPS_DEFAULT,
        pip_for_pair,
        spread_half_price,
    )

    trades: list[dict] = []
    n = len(c)
    i = 0
    pip = pip_for_pair(pair)
    sh = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))
    while i < n - max_bars - 2:
        p_lo = float(pl[i])
        p_sh = float(ps[i])
        if p_lo < thresh and p_sh < thresh:
            i += 1
            continue
        if p_lo >= thresh and p_lo >= p_sh:
            direction = "long"
        elif p_sh >= thresh:
            direction = "short"
        else:
            i += 1
            continue
        pnl, bars, _ = _sim_trade(
            i, direction, h, l, c, pip, sh, sl_p, tp_p, max_bars
        )
        trades.append(
            {
                "pair": pair,
                "direction": direction,
                "pnl_pips": pnl,
                "bars": bars,
                "hour": int(ts[i].hour) if hasattr(ts[i], "hour") else 0,
            }
        )
        i = i + max(1, bars)
    return trades


def run_ml_sltp_sweep(
    ml_dir: Path,
    test_start: str = "2026-03-01",
    thresh: float = 0.45,
    pairs: list[str] | None = None,
) -> list[dict]:
    """Sweeps fixed R:R (SL, TP) pips; same LightGBM models; hold-out from test_start."""
    ml_dir = Path(ml_dir)
    if not list(ml_dir.glob("features_*.parquet")):
        return []
    if not (ml_dir / "model_long.txt").is_file() or not (ml_dir / "model_short.txt").is_file():
        return []

    from ml_backtest import PAIRS, summarize  # noqa: WPS433
    from ml_labels import N_FUTURE  # noqa: WPS433
    from scalp_mode.ml.bar_features import FEATURE_COLUMNS  # noqa: WPS433

    import lightgbm as lgb

    if pairs is None:
        pairs = list(PAIRS)
    long_m = lgb.Booster(model_file=str(ml_dir / "model_long.txt"))
    short_m = lgb.Booster(model_file=str(ml_dir / "model_short.txt"))
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    grids = [
        (5.0, 8.0),
        (8.0, 12.0),
        (10.0, 15.0),
        (10.0, 20.0),
        (15.0, 20.0),
        (15.0, 25.0),
    ]
    out: list[dict] = []
    for sl_p, tp_p in grids:
        all_tr: list[dict] = []
        t0, t1 = None, None
        for p in pairs:
            path = ml_dir / f"features_{p}.parquet"
            if not path.exists():
                continue
            cols = ["timestamp", "pair", "high", "low", "close"] + FEATURE_COLUMNS
            df = pd.read_parquet(path, columns=cols)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df[df["timestamp"] >= test_start_ts].reset_index(drop=True)
            if df.empty:
                del df
                gc.collect()
                continue
            X = np.nan_to_num(
                df[FEATURE_COLUMNS].values.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            p_lo = long_m.predict(X).astype(np.float32)
            p_sh = short_m.predict(X).astype(np.float32)
            h = df["high"].values.astype(np.float64)
            lo = df["low"].values.astype(np.float64)
            cl = df["close"].values.astype(np.float64)
            tsv = pd.DatetimeIndex(df["timestamp"])
            if t0 is None:
                t0 = tsv.min()
            t1 = tsv.max()
            all_tr.extend(
                _backtest_pair_sltp(
                    p, h, lo, cl, tsv, p_lo, p_sh, thresh, sl_p, tp_p, N_FUTURE
                )
            )
            del df, X
            gc.collect()
        n_days = 1.0
        if t0 is not None and t1 is not None:
            n_days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
        s = summarize(all_tr, n_days) if all_tr else summarize([], n_days)
        be = 100.0 * sl_p / (sl_p + tp_p) if (sl_p + tp_p) else 0.0
        out.append(
            {
                "sl_pips": sl_p,
                "tp_pips": tp_p,
                "breakeven_wr_pct": round(be, 2),
                "threshold": thresh,
                **s,
            }
        )
    return out


def _backtest_pair_atr(
    pair: str,
    h,
    l,
    c,
    atr,
    ts,
    pl,
    ps,
    thresh: float,
    max_bars: int,
    tp_mult: float,
) -> list[dict]:
    from scalp_mode.ml.bar_features import (  # noqa: WPS433
        SPREAD_PIPS_DEFAULT,
        pip_for_pair,
        spread_half_price,
    )

    trades: list[dict] = []
    n = len(c)
    i = 0
    pip = pip_for_pair(pair)
    sh = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))
    while i < n - max_bars - 2:
        p_lo = float(pl[i])
        p_sh = float(ps[i])
        if p_lo < thresh and p_sh < thresh:
            i += 1
            continue
        if p_lo >= thresh and p_lo >= p_sh:
            direction = "long"
        elif p_sh >= thresh:
            direction = "short"
        else:
            i += 1
            continue
        pnl, bars, _ = _sim_trade_atr(
            i, direction, h, l, c, atr, pip, sh, max_bars, 1.0, tp_mult
        )
        trades.append(
            {
                "pair": pair,
                "direction": direction,
                "pnl_pips": pnl,
                "bars": bars,
                "hour": int(ts[i].hour) if hasattr(ts[i], "hour") else 0,
            }
        )
        i = i + max(1, bars)
    return trades


def run_ml_v2_sweep(
    ml_dir: Path,
    global_test_start: str = "2026-03-01",
    thresholds: list[float] | None = None,
    tp_mult: float = 1.5,
    pairs: list[str] | None = None,
) -> list[dict]:
    """Walk-forward test windows from ``wf_manifest.json``, ATR SL/TP, threshold grid."""
    ml_dir = Path(ml_dir)
    man_path = ml_dir / "wf_manifest.json"
    if not man_path.is_file():
        return []
    if not list(ml_dir.glob("features_*.parquet")):
        return []

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    feat_names: list[str] = manifest["feature_names"]
    folds: list[dict] = manifest.get("folds") or []
    if not folds:
        return []

    from ml_backtest import PAIRS, summarize  # noqa: WPS433
    from ml_labels import N_FUTURE  # noqa: WPS433

    import lightgbm as lgb  # noqa: WPS433

    if pairs is None:
        pairs = list(PAIRS)
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65]

    gts = pd.Timestamp(global_test_start, tz="UTC")
    out: list[dict] = []

    for thresh in thresholds:
        all_tr: list[dict] = []
        t0, t1 = None, None
        for fold in folds:
            long_m = lgb.Booster(model_file=fold["model_long"])
            short_m = lgb.Booster(model_file=fold["model_short"])
            f_start = pd.Timestamp(fold["test_start_utc"])
            f_end = pd.Timestamp(fold["test_end_utc"])
            for p in pairs:
                path = ml_dir / f"features_{p}.parquet"
                if not path.exists():
                    continue
                full = pd.read_parquet(path)
                need = ["timestamp", "high", "low", "close", "atr14"] + feat_names
                for c in need:
                    if c not in full.columns:
                        raise ValueError(f"{path} missing {c} for ML V2 sweep")
                df = full[need].copy()
                del full
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                m = (df["timestamp"] >= max(f_start, gts)) & (df["timestamp"] <= f_end)
                df = df.loc[m].reset_index(drop=True)
                if df.empty:
                    del df
                    gc.collect()
                    continue
                X = np.nan_to_num(
                    df[feat_names].values.astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                p_lo = long_m.predict(X).astype(np.float32)
                p_sh = short_m.predict(X).astype(np.float32)
                h = df["high"].values.astype(np.float64)
                lo = df["low"].values.astype(np.float64)
                cl = df["close"].values.astype(np.float64)
                atr = df["atr14"].values.astype(np.float64)
                tsv = pd.DatetimeIndex(df["timestamp"])
                if t0 is None or tsv.min() < t0:
                    t0 = tsv.min()
                if t1 is None or tsv.max() > t1:
                    t1 = tsv.max()
                all_tr.extend(
                    _backtest_pair_atr(
                        p, h, lo, cl, atr, tsv, p_lo, p_sh, thresh, N_FUTURE, tp_mult
                    )
                )
                del df, X
                gc.collect()
            del long_m, short_m
            gc.collect()

        n_days = 1.0
        if t0 is not None and t1 is not None:
            n_days = max((t1 - t0).total_seconds() / 86400.0, 1.0)
        s = summarize(all_tr, n_days) if all_tr else summarize([], n_days)
        out.append({"threshold": thresh, "tp_atr_mult": tp_mult, **s})
    return out
