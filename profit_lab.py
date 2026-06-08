#!/usr/bin/env python3
"""Profit-first strategy lab V2: optimize PF (not WR), wide R:R, BE/trail sim.

Families A–E and rounds 1–4 per plan. Run on VM with full grid; use ``--quick`` locally.

  python profit_lab.py --data-dir data/raw
  python -m strategy_arena --profit-lab
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_strategies import (  # noqa: E402
    STRATEGIES,
    simulate_trades_advanced,
    simulate_trades_vec,
)
from strategy_arena.config import DEFAULT_DATA_RAW, OOS_START, V2_PAIRS  # noqa: E402
from strategy_arena.loader import list_available_pairs, load_oanda_m1, prepare_m5  # noqa: E402
from strategy_arena.runner import RESEARCH_ALL  # noqa: E402

from scalp_mode.engine.hour_edge_filter import LONG_EDGE_HOURS, SHORT_EDGE_HOURS  # noqa: E402
from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402

from strategy_lab import (  # noqa: E402
    build_timeframes,
    filter_oos,
    spread_price_full,
    summarize_trades,
    tf_ix_from_m5,
)


# ---------------------------------------------------------------------------
# TP scaling (reward = tp_rr_mult × risk)
# ---------------------------------------------------------------------------


def scale_tps_to_rr(
    entries: list[float],
    sls: list[float],
    tps: list[float],
    dirs: list[int],
    pip: float,
    tp_rr_mult: float,
) -> list[float]:
    out: list[float] = []
    for k in range(len(entries)):
        e = float(entries[k])
        sl = float(sls[k])
        tp0 = float(tps[k])
        d = int(dirs[k])
        risk = abs(e - sl)
        if risk < pip * 0.25:
            risk = pip * 0.25
        if d == 1:
            out.append(e + tp_rr_mult * risk)
        else:
            out.append(e - tp_rr_mult * risk)
    return out


def exit_profile_kwargs(profile: str) -> dict[str, Any]:
    """Map profile name to simulate_trades_advanced kwargs."""
    if profile == "none":
        return {}
    if profile == "be":
        return {"be_trigger_pct": 0.6}
    if profile == "trail":
        return {"trail_start_R": 2.0, "trail_lock_R": 1.0}
    if profile == "be_trail":
        return {
            "be_trigger_pct": 0.6,
            "trail_start_R": 2.0,
            "trail_lock_R": 1.0,
        }
    if profile == "decay":
        return {"decay_bars": 24, "decay_progress_frac": 0.12}
    return {}


def _simulate_book(
    name: str,
    result: Any,
    df: pd.DataFrame,
    pair: str,
    pip: float,
    max_bars: int,
    scaled_tps: list[float],
    atr: np.ndarray | None,
    profile: str,
) -> list[dict]:
    hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))
    kw = exit_profile_kwargs(profile)

    if name == "S15_VWAP":
        idx, dirs, ent, sls, _tps, h_arr, l_arr, c_arr, hrs, tss, _n = result
        if not idx:
            return []
        tps_arr = np.array(scaled_tps, dtype=np.float64)
        if kw:
            sim = simulate_trades_advanced(
                h_arr,
                l_arr,
                c_arr,
                np.array(idx, dtype=np.int64),
                np.array(dirs, dtype=np.int64),
                np.array(ent, dtype=np.float64),
                np.array(sls, dtype=np.float64),
                tps_arr,
                60,
                pip,
                half_spread=hsp,
                atr=None,
                **kw,
            )
        else:
            sim = simulate_trades_vec(
                h_arr,
                l_arr,
                c_arr,
                np.array(idx, dtype=np.int64),
                np.array(dirs, dtype=np.int64),
                np.array(ent, dtype=np.float64),
                np.array(sls, dtype=np.float64),
                tps_arr,
                60,
                pip,
                half_spread=hsp,
            )
        trades = []
        for k in range(len(idx)):
            trades.append(
                {
                    "variant": "",
                    "strategy": name,
                    "pair": pair,
                    "direction": "long" if dirs[k] == 1 else "short",
                    "hour": int(hrs[k]),
                    "pnl_pips": float(sim[k]["pnl_pips"]),
                    "exit_reason": str(sim[k]["exit_reason"]),
                    "bars_held": int(sim[k]["bars_held"]),
                    "timestamp": str(tss[k]),
                }
            )
        return trades

    idx, dirs, ent, sls, _tps = result
    if not idx:
        return []
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    hours = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    tss = df["timestamp"].values
    atr_use = atr if atr is not None else df["atr14"].values
    tps_arr = np.array(scaled_tps, dtype=np.float64)
    if kw:
        sim = simulate_trades_advanced(
            h_arr,
            l_arr,
            c_arr,
            np.array(idx, dtype=np.int64),
            np.array(dirs, dtype=np.int64),
            np.array(ent, dtype=np.float64),
            np.array(sls, dtype=np.float64),
            tps_arr,
            max_bars,
            pip,
            half_spread=hsp,
            atr=atr_use,
            **kw,
        )
    else:
        sim = simulate_trades_vec(
            h_arr,
            l_arr,
            c_arr,
            np.array(idx, dtype=np.int64),
            np.array(dirs, dtype=np.int64),
            np.array(ent, dtype=np.float64),
            np.array(sls, dtype=np.float64),
            tps_arr,
            max_bars,
            pip,
            half_spread=hsp,
        )
    trades = []
    for k in range(len(idx)):
        trades.append(
            {
                "variant": "",
                "strategy": name,
                "pair": pair,
                "direction": "long" if dirs[k] == 1 else "short",
                "hour": int(hours[idx[k]]),
                "pnl_pips": float(sim[k]["pnl_pips"]),
                "exit_reason": str(sim[k]["exit_reason"]),
                "bars_held": int(sim[k]["bars_held"]),
                "timestamp": str(tss[idx[k]]),
            }
        )
    return trades


def _simulate_research(
    name: str,
    result: tuple,
    df: pd.DataFrame,
    pair: str,
    pip: float,
    max_bars: int,
    scaled_tps: list[float],
    profile: str,
) -> list[dict]:
    _ = name
    idx, dirs, ent, sls, _tps = result
    if not idx:
        return []
    hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))
    kw = exit_profile_kwargs(profile)
    atr = df["atr14"].values
    tps_arr = np.array(scaled_tps, dtype=np.float64)
    if kw:
        sim = simulate_trades_advanced(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            np.array(idx, dtype=np.int64),
            np.array(dirs, dtype=np.int64),
            np.array(ent, dtype=np.float64),
            np.array(sls, dtype=np.float64),
            tps_arr,
            max_bars,
            pip,
            half_spread=hsp,
            atr=atr,
            **kw,
        )
    else:
        sim = simulate_trades_vec(
            df["high"].values,
            df["low"].values,
            df["close"].values,
            np.array(idx, dtype=np.int64),
            np.array(dirs, dtype=np.int64),
            np.array(ent, dtype=np.float64),
            np.array(sls, dtype=np.float64),
            tps_arr,
            max_bars,
            pip,
            half_spread=hsp,
        )
    hours = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    tss = df["timestamp"].values
    trades = []
    for k in range(len(idx)):
        trades.append(
            {
                "variant": "",
                "strategy": name,
                "pair": pair,
                "direction": "long" if dirs[k] == 1 else "short",
                "hour": int(hours[idx[k]]),
                "pnl_pips": float(sim[k]["pnl_pips"]),
                "exit_reason": str(sim[k]["exit_reason"]),
                "bars_held": int(sim[k]["bars_held"]),
                "timestamp": str(tss[idx[k]]),
            }
        )
    return trades


# ---------------------------------------------------------------------------
# Family C — hour edge, looser filters, wide R:R
# ---------------------------------------------------------------------------


def signals_c_hour_loose(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    _d1: pd.DataFrame,
    pair: str,
    spread: float,
    tp_atr: float,
    sl_atr: float,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    _ = pair, _d1
    c = m5["close"].values
    l = m5["low"].values
    h = m5["high"].values
    atr = m5["atr14"].values
    rsi = m5["rsi14"].values
    e9 = m5["ema9"].values
    e20 = m5["ema20"].values
    hour = m5["hour"].values.astype(np.int64)
    n = len(m5)
    nh1, nh4 = len(h1), len(h4)
    h1_e9, h1_e20 = h1["ema9"].values, h1["ema20"].values
    h4_e20, h4_e40 = h4["ema20"].values, h4["ema40"].values
    long_h = set(LONG_EDGE_HOURS.keys())
    short_h = set(SHORT_EDGE_HOURS.keys())

    idx, dirs, ents, sls, tps = [], [], [], [], []
    last = -8
    for i in range(40, n - 2):
        if i - last < 6:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        hi = hour[i]
        h1_i = min(tf_ix_from_m5(i, 60), nh1 - 1)
        h4_i = min(tf_ix_from_m5(i, 240), nh4 - 1)
        if hi in long_h:
            if not (h1_e9[h1_i] >= h1_e20[h1_i] * 0.998 and e9[i] >= e20[i] * 0.998):
                continue
            if rsi[i] <= 35:
                continue
            if h4_e20[h4_i] < h4_e40[h4_i] * 0.995:
                continue
            c_i = c[i]
            entry = c_i + spread
            sl = entry - sl_atr * a
            tp = entry + tp_atr * a
            idx.append(i)
            dirs.append(1)
            ents.append(entry)
            sls.append(sl)
            tps.append(tp)
            last = i
        elif hi in short_h:
            if not (h1_e9[h1_i] <= h1_e20[h1_i] * 1.002 and e9[i] <= e20[i] * 1.002):
                continue
            if rsi[i] >= 65:
                continue
            if h4_e20[h4_i] > h4_e40[h4_i] * 1.005:
                continue
            c_i = c[i]
            entry = c_i - spread
            sl = entry + sl_atr * a
            tp = entry - tp_atr * a
            idx.append(i)
            dirs.append(-1)
            ents.append(entry)
            sls.append(sl)
            tps.append(tp)
            last = i
    return idx, dirs, ents, sls, tps


# ---------------------------------------------------------------------------
# Family D — wider sessions, higher TP vs range
# ---------------------------------------------------------------------------


def signals_d_compress_wide(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    pair: str,
    spread: float,
    squeeze_pct: float,
    tp_range_mult: float,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    c = m5["close"].values
    h = m5["high"].values
    l = m5["low"].values
    atr = m5["atr14"].values
    sq = m5["bb_squeeze_pct"].values
    e20 = m5["ema20"].values
    hour = m5["hour"].values.astype(np.int64)
    br = m5["bar_range"].values
    n = len(m5)
    nm15 = len(m15)
    m15_e9 = m15["ema9"].values
    m15_e20 = m15["ema20"].values

    def sess_ok(hi: int) -> bool:
        return (6 <= hi <= 10) or (12 <= hi <= 16)

    idx, dirs, ents, sls, tps = [], [], [], [], []
    last = -8
    for i in range(25, n - 3):
        if i - last < 6:
            continue
        if not sess_ok(int(hour[i])):
            continue
        if not (sq[i] < squeeze_pct and sq[i - 1] < squeeze_pct):
            continue
        window = br[i - 19 : i + 1]
        if len(window) < 20 or br[i] > np.min(window) * 1.2:
            continue
        m15_i = min(tf_ix_from_m5(i, 15), nm15 - 1)
        bull_m15 = m15_e9[m15_i] > m15_e20[m15_i]
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        rng = max(h[i] - l[i], a * 0.5)
        prev_hi = max(h[i - 3 : i])
        prev_lo = min(l[i - 3 : i])
        if bull_m15 and c[i] > e20[i] and h[i] > prev_hi:
            c_i = c[i]
            entry = c_i + spread
            sl = entry - 1.0 * rng
            tp = entry + tp_range_mult * rng
            idx.append(i)
            dirs.append(1)
            ents.append(entry)
            sls.append(sl)
            tps.append(tp)
            last = i
        elif (not bull_m15) and c[i] < e20[i] and l[i] < prev_lo:
            c_i = c[i]
            entry = c_i - spread
            sl = entry + 1.0 * rng
            tp = entry - tp_range_mult * rng
            idx.append(i)
            dirs.append(-1)
            ents.append(entry)
            sls.append(sl)
            tps.append(tp)
            last = i
    return idx, dirs, ents, sls, tps


# ---------------------------------------------------------------------------
# Family B + E — ML
# ---------------------------------------------------------------------------


def run_family_b_ml(
    ml_dir: Path,
    pairs: list[str],
    oos_start: pd.Timestamp,
    thresholds: tuple[float, ...],
    tp_mults: tuple[float, ...],
    profiles: tuple[str, ...],
    spread_mult: float = 1.0,
) -> list[dict]:
    """Return list of {variant, trades} for ML + ATR exits (OOS window per fold)."""
    ml_dir = Path(ml_dir)
    man = ml_dir / "wf_manifest.json"
    if not man.is_file():
        return []
    import json as json_mod  # noqa: WPS433

    manifest = json_mod.loads(man.read_text(encoding="utf-8"))
    feat_names: list[str] = manifest["feature_names"]
    folds: list[dict] = manifest.get("folds") or []
    if not folds:
        return []

    try:
        import lightgbm as lgb  # noqa: WPS433
    except ImportError:
        return []

    from ml_labels import N_FUTURE  # noqa: WPS433
    from strategy_arena.ml_sweep import _sim_trade_atr  # noqa: WPS433

    gts = pd.Timestamp(oos_start)
    if gts.tz is None:
        gts = gts.tz_localize("UTC")
    else:
        gts = gts.tz_convert("UTC")

    rows_out: list[dict] = []
    for thresh in thresholds:
        for tp_m in tp_mults:
            for prof in profiles:
                variant = f"B_ML_thr{thresh:.2f}_tp{tp_m}_{prof}"
                bucket: list[dict] = []
                kw = exit_profile_kwargs(prof)
                for fold in folds:
                    long_m = lgb.Booster(model_file=fold["model_long"])
                    short_m = lgb.Booster(model_file=fold["model_short"])
                    f_start = pd.Timestamp(fold["test_start_utc"])
                    f_end = pd.Timestamp(fold["test_end_utc"])
                    for p in pairs:
                        path = ml_dir / f"features_{p}.parquet"
                        if not path.exists():
                            continue
                        need = ["timestamp", "high", "low", "close", "atr14"] + feat_names
                        raw = pd.read_parquet(path)
                        if not all(c in raw.columns for c in need):
                            del raw
                            gc.collect()
                            continue
                        df = raw[need].copy()
                        del raw
                        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                        m = (df["timestamp"] >= max(f_start, gts)) & (df["timestamp"] <= f_end)
                        df = df.loc[m].reset_index(drop=True)
                        if df.empty:
                            del df
                            gc.collect()
                            continue
                        h = df["high"].values.astype(np.float64)
                        lo = df["low"].values.astype(np.float64)
                        cl = df["close"].values.astype(np.float64)
                        atr = df["atr14"].values.astype(np.float64)
                        tsv = pd.DatetimeIndex(df["timestamp"])
                        X = np.nan_to_num(
                            df[feat_names].values.astype(np.float32),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        p_lo = long_m.predict(X).astype(np.float32)
                        p_sh = short_m.predict(X).astype(np.float32)
                        pip = pip_for_pair(p)
                        sp = SPREAD_PIPS_DEFAULT.get(p, 1.5) * spread_mult
                        sh = spread_half_price(p, sp)
                        n = len(cl)
                        ii = 0
                        while ii < n - N_FUTURE - 2:
                            plv = float(p_lo[ii])
                            psv = float(p_sh[ii])
                            if plv < thresh and psv < thresh:
                                ii += 1
                                continue
                            if plv >= thresh and plv >= psv:
                                direction = "long"
                            elif psv >= thresh:
                                direction = "short"
                            else:
                                ii += 1
                                continue
                            if not kw:
                                pnl, bars, _ = _sim_trade_atr(
                                    ii, direction, h, lo, cl, atr, pip, sh, N_FUTURE, 1.0, tp_m
                                )
                            else:
                                a = float(atr[ii]) if np.isfinite(atr[ii]) else pip * 0.05
                                a = max(a, pip * 0.05)
                                if direction == "long":
                                    entry = cl[ii] + sh
                                    sl = entry - 1.0 * a
                                    tp = entry + tp_m * a
                                    d = 1
                                else:
                                    entry = cl[ii] - sh
                                    sl = entry + 1.0 * a
                                    tp = entry - tp_m * a
                                    d = -1
                                sim = simulate_trades_advanced(
                                    h,
                                    lo,
                                    cl,
                                    np.array([ii], dtype=np.int64),
                                    np.array([d], dtype=np.int64),
                                    np.array([entry], dtype=np.float64),
                                    np.array([sl], dtype=np.float64),
                                    np.array([tp], dtype=np.float64),
                                    N_FUTURE,
                                    pip,
                                    half_spread=sh,
                                    atr=atr,
                                    **kw,
                                )
                                pnl = sim[0]["pnl_pips"]
                                bars = sim[0]["bars_held"]
                            bucket.append(
                                {
                                    "variant": variant,
                                    "strategy": "F_B_ML",
                                    "pair": p,
                                    "direction": direction,
                                    "pnl_pips": pnl,
                                    "bars_held": bars,
                                    "timestamp": str(tsv[ii]),
                                    "hour": int(tsv[ii].hour),
                                }
                            )
                            ii += max(1, int(bars))
                        del df, X
                        gc.collect()
                    del long_m, short_m
                    gc.collect()
                if bucket:
                    rows_out.append({"variant": variant, "trades": bucket})
    return rows_out


def run_family_e_gated(
    ml_dir: Path,
    pairs: list[str],
    oos_start: pd.Timestamp,
    base_strategies: list[str],
    thresholds: tuple[float, ...],
    data_dir: Path,
    spread_mult: float = 1.0,
) -> list[dict]:
    """Gate base strategy signals with ML direction agreement (M1 features row)."""
    ml_dir = Path(ml_dir)
    if not (ml_dir / "wf_manifest.json").is_file():
        return []
    # Use last fold models only for speed
    import json as json_mod  # noqa: WPS433

    man = json_mod.loads((ml_dir / "wf_manifest.json").read_text(encoding="utf-8"))
    folds = man.get("folds") or []
    if not folds:
        return []
    fold = folds[-1]
    feat_names: list[str] = man["feature_names"]
    try:
        import lightgbm as lgb  # noqa: WPS433
    except ImportError:
        return []
    long_m = lgb.Booster(model_file=fold["model_long"])
    short_m = lgb.Booster(model_file=fold["model_short"])

    name_to_sig_book = {n: fn for n, fn, _ in STRATEGIES}
    name_to_sig_r = {n: fn for n, fn, _ in RESEARCH_ALL}
    rows: list[dict] = []

    for base in base_strategies:
        if base not in name_to_sig_book and base not in name_to_sig_r:
            continue
        for thr in thresholds:
            variant = f"E_{base}_mlgate{thr}"
            bucket: list[dict] = []
            for p in pairs:
                df_m1 = load_oanda_m1(data_dir, p)
                if df_m1.empty:
                    continue
                path = ml_dir / f"features_{p}.parquet"
                if not path.exists():
                    continue
                feat_df = pd.read_parquet(path)
                feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"], utc=True)
                feat_df = feat_df.sort_values("timestamp").reset_index(drop=True)
                df_m5 = prepare_m5(df_m1, p)
                pip = pip_for_pair(p)
                spread = SPREAD_PIPS_DEFAULT.get(p, 1.5) * pip_for_pair(p) * spread_mult
                if base in name_to_sig_book:
                    sig = name_to_sig_book[base]
                    mb = next(mb for n, _, mb in STRATEGIES if n == base)
                    raw = sig(df_m5, p, pip, spread)
                else:
                    sig = name_to_sig_r[base]
                    mb = next(mb for n, _, mb in RESEARCH_ALL if n == base)
                    raw = sig(df_m5, p, pip, spread)
                if base == "S15_VWAP":
                    continue
                idx, dirs, ent, sls, tps = raw
                if not idx:
                    continue
                tps2 = scale_tps_to_rr(ent, sls, tps, dirs, pip, 2.0)
                need = ["timestamp", "high", "low", "close", "atr14"] + feat_names
                if not all(c in feat_df.columns for c in need):
                    continue
                fts = feat_df["timestamp"].values.astype("datetime64[ns]")
                X_all = np.nan_to_num(
                    feat_df[feat_names].values.astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                p_lo = long_m.predict(X_all).astype(np.float32)
                p_sh = short_m.predict(X_all).astype(np.float32)
                hsp = spread_half_price(p, SPREAD_PIPS_DEFAULT.get(p, 1.5) * spread_mult)
                atr_m5 = df_m5["atr14"].values
                filt_idx, filt_d, filt_e, filt_s, filt_t = [], [], [], [], []
                for k in range(len(idx)):
                    raw_ts = df_m5["timestamp"].iloc[idx[k]]
                    bar_ts = pd.Timestamp(raw_ts)
                    if bar_ts.tzinfo is None:
                        bar_ts = bar_ts.tz_localize("UTC")
                    b64 = bar_ts.to_datetime64()
                    j = int(np.searchsorted(fts, b64, side="right") - 1)
                    if j < 0 or j >= len(feat_df):
                        continue
                    d = dirs[k]
                    if d == 1 and float(p_lo[j]) < thr:
                        continue
                    if d == -1 and float(p_sh[j]) < thr:
                        continue
                    filt_idx.append(idx[k])
                    filt_d.append(d)
                    filt_e.append(ent[k])
                    filt_s.append(sls[k])
                    filt_t.append(tps2[k])
                if not filt_idx:
                    continue
                sim = simulate_trades_advanced(
                    df_m5["high"].values,
                    df_m5["low"].values,
                    df_m5["close"].values,
                    np.array(filt_idx, dtype=np.int64),
                    np.array(filt_d, dtype=np.int64),
                    np.array(filt_e, dtype=np.float64),
                    np.array(filt_s, dtype=np.float64),
                    np.array(filt_t, dtype=np.float64),
                    mb,
                    pip,
                    half_spread=hsp,
                    atr=atr_m5,
                    be_trigger_pct=0.6,
                    trail_start_R=2.0,
                    trail_lock_R=1.0,
                )
                hours = df_m5["hour"].values
                tss = df_m5["timestamp"].values
                for k in range(len(filt_idx)):
                    bucket.append(
                        {
                            "variant": variant,
                            "strategy": base,
                            "pair": p,
                            "direction": "long" if filt_d[k] == 1 else "short",
                            "hour": int(hours[filt_idx[k]]),
                            "pnl_pips": float(sim[k]["pnl_pips"]),
                            "exit_reason": str(sim[k]["exit_reason"]),
                            "bars_held": int(sim[k]["bars_held"]),
                            "timestamp": str(tss[filt_idx[k]]),
                        }
                    )
                del df_m1, df_m5, feat_df
                gc.collect()
            if bucket:
                rows.append({"variant": variant, "trades": bucket})
    del long_m, short_m
    gc.collect()
    return rows


# ---------------------------------------------------------------------------
# Round 1 worker (Family A per pair)
# ---------------------------------------------------------------------------


def _worker_family_a_pair(args: tuple) -> list[dict]:
    (
        pair,
        data_dir,
        oos_start_iso,
        quick,
    ) = args
    oos_start = pd.Timestamp(oos_start_iso)
    data_dir = Path(data_dir)
    tp_rr_list = (2.0, 2.5) if quick else (1.5, 2.0, 2.5, 3.0)
    profiles = ("none", "be", "trail", "be_trail") if not quick else ("none", "be_trail")

    df_m1 = load_oanda_m1(data_dir, pair)
    if df_m1.empty or len(df_m1) < 3000:
        return []
    df = prepare_m5(df_m1, pair)
    del df_m1
    gc.collect()
    pip = pip_for_pair(pair)
    spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip
    atr = df["atr14"].values

    out: list[dict] = []

    for sname, sig, mb in STRATEGIES:
        raw = sig(df, pair, pip, spread)
        if sname == "S15_VWAP":
            if not raw or not raw[0]:
                continue
        else:
            if not raw:
                continue
        if sname == "S15_VWAP":
            idx, dirs, ent, sls, tps = raw[0], raw[1], raw[2], raw[3], raw[4]
            base_tps = list(tps)
        else:
            idx, dirs, ent, sls, tps = raw
            base_tps = list(tps)
        if not idx:
            continue
        for tp_rr in tp_rr_list:
            scaled = scale_tps_to_rr(ent, sls, base_tps, dirs, pip, tp_rr)
            for prof in profiles:
                tag = f"A:{sname}|tp{tp_rr}|{prof}"
                if sname == "S15_VWAP":
                    tr = _simulate_book(
                        sname, raw, df, pair, pip, mb, scaled, None, prof
                    )
                else:
                    tr = _simulate_book(
                        sname, (idx, dirs, ent, sls, base_tps), df, pair, pip, mb, scaled, atr, prof
                    )
                for t in tr:
                    t["variant"] = tag
                oos = filter_oos(tr, oos_start)
                m = summarize_trades(oos, 30.0)
                out.append(
                    {
                        "family": "A",
                        "variant": tag,
                        "pair": pair,
                        "oos": m,
                        "n_oos": m["n"],
                        "pf_oos": m["pf"],
                    }
                )

    for sname, sig, mb in RESEARCH_ALL:
        raw = sig(df, pair, pip, spread)
        if not raw or not raw[0]:
            continue
        idx, dirs, ent, sls, tps = raw
        base_tps = list(tps)
        for tp_rr in tp_rr_list:
            scaled = scale_tps_to_rr(ent, sls, base_tps, dirs, pip, tp_rr)
            for prof in profiles:
                tag = f"A:{sname}|tp{tp_rr}|{prof}"
                tr = _simulate_research(
                    sname, (idx, dirs, ent, sls, base_tps), df, pair, pip, mb, scaled, prof
                )
                for t in tr:
                    t["variant"] = tag
                oos = filter_oos(tr, oos_start)
                m = summarize_trades(oos, 30.0)
                out.append(
                    {
                        "family": "A",
                        "variant": tag,
                        "pair": pair,
                        "oos": m,
                        "n_oos": m["n"],
                        "pf_oos": m["pf"],
                    }
                )
    del df
    gc.collect()
    return out


def _aggregate_family_a(rows: list[dict]) -> list[dict]:
    """Merge same variant across pairs by summing metrics from trades — use stored oos only."""
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r["variant"], []).append(r)
    merged = []
    for v, parts in buckets.items():
        # approximate: weight by n
        total_n = sum(p["n_oos"] for p in parts)
        if total_n == 0:
            continue
        # recompute PF from weighted avg pips — wrong without raw trades
        # Store list of per-pair pf and n for report
        pf_w = sum(p["pf_oos"] * p["n_oos"] for p in parts) / max(total_n, 1)
        merged.append(
            {
                "variant": v,
                "family": "A",
                "pairs": len(parts),
                "n_oos_total": total_n,
                "pf_proxy": pf_w,
                "parts": parts,
            }
        )
    merged.sort(key=lambda x: -x["pf_proxy"] * np.log1p(x["n_oos_total"]))
    return merged


def run_family_cd_pair(pair: str, data_dir: Path, oos_start: pd.Timestamp, quick: bool) -> list[dict]:
    df_m1 = load_oanda_m1(data_dir, pair)
    if df_m1.empty:
        return []
    tf = build_timeframes(df_m1, pair)
    m5, m15, h1, h4, d1 = tf["m5"], tf["m15"], tf["h1"], tf["h4"], tf["d1"]
    spr = spread_price_full(pair)
    pip = pip_for_pair(pair)
    out: list[dict] = []
    tp_sl = [(2.0, 1.0), (2.5, 1.0), (3.0, 1.0)] if not quick else [(2.5, 1.0)]
    profiles = ("none", "be_trail") if not quick else ("be_trail",)
    for tp_atr, sl_atr in tp_sl:
        for prof in profiles:
            tag = f"C:hour|tp{tp_atr}_sl{sl_atr}|{prof}"
            ix, di, en, sl, tp = signals_c_hour_loose(m5, h1, h4, d1, pair, spr, tp_atr, sl_atr)
            kw = exit_profile_kwargs(prof)
            hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))
            if not ix:
                m = summarize_trades([], 30.0)
            else:
                if kw:
                    sim = simulate_trades_advanced(
                        m5["high"].values,
                        m5["low"].values,
                        m5["close"].values,
                        np.array(ix, dtype=np.int64),
                        np.array(di, dtype=np.int64),
                        np.array(en, dtype=np.float64),
                        np.array(sl, dtype=np.float64),
                        np.array(tp, dtype=np.float64),
                        40,
                        pip,
                        half_spread=hsp,
                        atr=m5["atr14"].values,
                        **kw,
                    )
                else:
                    sim = simulate_trades_vec(
                        m5["high"].values,
                        m5["low"].values,
                        m5["close"].values,
                        np.array(ix, dtype=np.int64),
                        np.array(di, dtype=np.int64),
                        np.array(en, dtype=np.float64),
                        np.array(sl, dtype=np.float64),
                        np.array(tp, dtype=np.float64),
                        40,
                        pip,
                        half_spread=hsp,
                    )
                tr = []
                hrs = m5["hour"].values
                tss = m5["timestamp"].values
                for k in range(len(ix)):
                    tr.append(
                        {
                            "variant": tag,
                            "strategy": "C",
                            "pair": pair,
                            "direction": "long" if di[k] == 1 else "short",
                            "hour": int(hrs[ix[k]]),
                            "pnl_pips": float(sim[k]["pnl_pips"]),
                            "exit_reason": str(sim[k]["exit_reason"]),
                            "bars_held": int(sim[k]["bars_held"]),
                            "timestamp": str(tss[ix[k]]),
                        }
                    )
                oos = filter_oos(tr, oos_start)
                m = summarize_trades(oos, 30.0)
            out.append({"variant": tag, "family": "C", "pair": pair, "oos": m, "n_oos": m["n"], "pf_oos": m["pf"]})

    sqs = (0.25, 0.35, 0.4) if not quick else (0.35,)
    tp_rm = (2.0, 2.5) if not quick else (2.0,)
    for sq in sqs:
        for tpm in tp_rm:
            for prof in profiles:
                tag = f"D:compress|sq{sq}_tp{tpm}|{prof}"
                ix, di, en, sl, tp = signals_d_compress_wide(m5, m15, pair, spr, sq, tpm)
                kw = exit_profile_kwargs(prof)
                hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))
                if not ix:
                    m = summarize_trades([], 30.0)
                else:
                    if kw:
                        sim = simulate_trades_advanced(
                            m5["high"].values,
                            m5["low"].values,
                            m5["close"].values,
                            np.array(ix, dtype=np.int64),
                            np.array(di, dtype=np.int64),
                            np.array(en, dtype=np.float64),
                            np.array(sl, dtype=np.float64),
                            np.array(tp, dtype=np.float64),
                            50,
                            pip,
                            half_spread=hsp,
                            atr=m5["atr14"].values,
                            **kw,
                        )
                    else:
                        sim = simulate_trades_vec(
                            m5["high"].values,
                            m5["low"].values,
                            m5["close"].values,
                            np.array(ix, dtype=np.int64),
                            np.array(di, dtype=np.int64),
                            np.array(en, dtype=np.float64),
                            np.array(sl, dtype=np.float64),
                            np.array(tp, dtype=np.float64),
                            50,
                            pip,
                            half_spread=hsp,
                        )
                    tr = []
                    hrs = m5["hour"].values
                    tss = m5["timestamp"].values
                    for k in range(len(ix)):
                        tr.append(
                            {
                                "variant": tag,
                                "strategy": "D",
                                "pair": pair,
                                "direction": "long" if di[k] == 1 else "short",
                                "hour": int(hrs[ix[k]]),
                                "pnl_pips": float(sim[k]["pnl_pips"]),
                                "exit_reason": str(sim[k]["exit_reason"]),
                                "bars_held": int(sim[k]["bars_held"]),
                                "timestamp": str(tss[ix[k]]),
                            }
                        )
                    oos = filter_oos(tr, oos_start)
                    m = summarize_trades(oos, 30.0)
                out.append({"variant": tag, "family": "D", "pair": pair, "oos": m, "n_oos": m["n"], "pf_oos": m["pf"]})

    del tf, m5, m15, h1, h4, d1
    gc.collect()
    return out


def _oos_thirds(trades: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    if len(trades) < 9:
        return trades, [], []
    tss = sorted(pd.Timestamp(t["timestamp"]) for t in trades)
    t1, t2 = tss[len(tss) // 3], tss[2 * len(tss) // 3]
    a, b, c = [], [], []
    for t in trades:
        ts = pd.Timestamp(t["timestamp"])
        if ts < t1:
            a.append(t)
        elif ts < t2:
            b.append(t)
        else:
            c.append(t)
    return a, b, c


def _monte_carlo_pf(pnls: list[float], n_shuffles: int = 1000, seed: int = 42) -> dict[str, float]:
    if len(pnls) < 5:
        return {"p5_pf": 0.0, "median_pf": 0.0}
    rng = random.Random(seed)
    pfs = []
    for _ in range(n_shuffles):
        idx = rng.choices(range(len(pnls)), k=len(pnls))
        sample = [pnls[i] for i in idx]
        gp = sum(x for x in sample if x > 0)
        gl = -sum(x for x in sample if x < 0)
        pfs.append(gp / gl if gl > 1e-9 else (99.0 if gp > 0 else 0.0))
    pfs.sort()
    return {
        "p5_pf": float(pfs[int(0.05 * len(pfs))]),
        "median_pf": float(pfs[len(pfs) // 2]),
    }


def meets_profit_gate(m: dict[str, Any], min_n: int = 50, min_pf: float = 1.3) -> bool:
    return m["n"] >= min_n and m["pf"] >= min_pf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profit-first strategy lab V2")
    ap.add_argument("--data-dir", type=Path, default=ROOT / DEFAULT_DATA_RAW)
    ap.add_argument("--ml-dir", type=Path, default=ROOT / "data" / "ml")
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "profit_lab_results.json")
    ap.add_argument("--quick", action="store_true", help="Smaller grids for local smoke test")
    ap.add_argument("--workers", type=int, default=8, help="Process pool size for Family A")
    ap.add_argument("--round", type=int, default=0, help="0=all rounds, 1–4 single phase")
    args = ap.parse_args(argv)

    oos_start = pd.Timestamp(OOS_START)
    pairs = [p for p in V2_PAIRS if p in list_available_pairs(args.data_dir)]
    if len(pairs) < 1:
        print(f"No pairs in {args.data_dir}", file=sys.stderr)
        return 1

    report: dict[str, Any] = {
        "oos_start": OOS_START,
        "pairs": pairs,
        "round1": {},
        "round2": {},
        "round3": {},
        "round4": {},
    }

    # ----- Round 1: Family A (parallel), C+D per pair, B ML -----
    if args.round in (0, 1):
        print("Round 1: Family A (book+research wide R:R) …")
        worker_args = [(p, str(args.data_dir), str(oos_start), args.quick) for p in pairs]
        all_a: list[dict] = []
        if args.workers > 1 and len(pairs) > 1:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pairs))) as ex:
                futs = [ex.submit(_worker_family_a_pair, wa) for wa in worker_args]
                for fu in as_completed(futs):
                    all_a.extend(fu.result())
        else:
            for wa in worker_args:
                all_a.extend(_worker_family_a_pair(wa))
        report["round1"]["family_a_parts"] = len(all_a)
        merged_a = _aggregate_family_a(all_a)
        report["round1"]["family_a_top"] = merged_a[:40]

        print("Round 1: Family C+D …")
        cd_rows: list[dict] = []
        for p in pairs:
            cd_rows.extend(run_family_cd_pair(p, args.data_dir, oos_start, args.quick))
        report["round1"]["family_cd"] = cd_rows

        print("Round 1: Family B (ML) …")
        th = (0.45, 0.50, 0.55) if args.quick else (0.45, 0.50, 0.55, 0.60, 0.65)
        tpm = (1.5, 2.0) if args.quick else (1.5, 2.0, 2.5)
        prof_b = ("none", "be_trail") if args.quick else ("none", "be", "trail", "be_trail")
        b_rows = run_family_b_ml(args.ml_dir, pairs, oos_start, th, tpm, prof_b)
        report["round1"]["family_b"] = [
            {"variant": r["variant"], "oos": summarize_trades(filter_oos(r["trades"], oos_start), 30.0)}
            for r in b_rows
        ]

    # Pick top strategies for round 2–4 from merged A + CD + B
    candidates: list[tuple[str, float, int]] = []
    for m in report.get("round1", {}).get("family_a_top", []):
        candidates.append((m["variant"], float(m["pf_proxy"]), int(m["n_oos_total"])))
    for r in report.get("round1", {}).get("family_cd", []):
        candidates.append((r["variant"], float(r["pf_oos"]), int(r["n_oos"])))
    for r in report.get("round1", {}).get("family_b", []):
        o = r["oos"]
        candidates.append((r["variant"], float(o["pf"]), int(o["n"])))
    candidates.sort(key=lambda x: -x[1] * np.log1p(max(x[2], 1)))
    top_variants = [c[0] for c in candidates[:25]]

    top_names: list[str] = []
    # ----- Round 2: fine grid on top 5 variant stems (re-parse strategy names from A:*) -----
    if args.round in (0, 2):
        print("Round 2: fine TP grid on top book/research names …")
        for v in top_variants[:20]:
            if v.startswith("A:"):
                name = v.split("|")[0].replace("A:", "")
                if name and name not in top_names:
                    top_names.append(name)
        top_names = top_names[:5]
        if not top_names:
            top_names = ["R2_EXH", "S4_EMA", "S9_FBR", "S16_HL2", "R8_THREE"]
        fine_rr = [1.7, 2.0, 2.3] if not args.quick else [2.0]
        r2_rows: list[dict] = []
        for p in pairs[:3] if args.quick else pairs:
            df_m1 = load_oanda_m1(args.data_dir, p)
            if df_m1.empty:
                continue
            df = prepare_m5(df_m1, p)
            pip = pip_for_pair(p)
            spread = SPREAD_PIPS_DEFAULT.get(p, 1.5) * pip
            atr = df["atr14"].values
            for sname, sig, mb in STRATEGIES:
                if sname not in top_names or sname == "S15_VWAP":
                    continue
                raw = sig(df, p, pip, spread)
                if not raw:
                    continue
                idx, dirs, ent, sls, tps = raw
                for tp_rr in fine_rr:
                    scaled = scale_tps_to_rr(ent, sls, list(tps), dirs, pip, tp_rr)
                    tag = f"A2:{sname}|tp{tp_rr}|be_trail"
                    tr = _simulate_book(
                        sname, (idx, dirs, ent, sls, list(tps)), df, p, pip, mb, scaled, atr, "be_trail"
                    )
                    for t in tr:
                        t["variant"] = tag
                    oos = filter_oos(tr, oos_start)
                    m = summarize_trades(oos, 30.0)
                    r2_rows.append({"variant": tag, "pair": p, "oos": m})
            del df_m1, df
            gc.collect()
        report["round2"] = {"rows": r2_rows, "oos_thirds_top": []}
        # 3-way OOS on best round2 row
        if r2_rows:
            best_r2 = max(r2_rows, key=lambda r: r["oos"]["pf"] * np.log1p(max(r["oos"]["n"], 1)))
            report["round2"]["best"] = best_r2

    # ----- Round 3: Family E + 2x spread stress on best OOS trades -----
    if args.round in (0, 3):
        print("Round 3: Family E (ML gate) + spread stress …")
        if not top_names:
            top_names.extend(["R2_EXH", "S4_EMA", "S9_FBR", "S16_HL2", "R8_THREE"])
        base_list = top_names[:5]
        e_rows = run_family_e_gated(
            args.ml_dir,
            pairs,
            oos_start,
            base_list[:5],
            (0.45, 0.50, 0.55),
            args.data_dir,
            spread_mult=1.0,
        )
        e_detail = []
        for r in e_rows:
            to = filter_oos(r["trades"], oos_start)
            e_detail.append(
                {
                    "variant": r["variant"],
                    "oos": summarize_trades(to, 30.0),
                    "trades_oos": to,
                }
            )
        report["round3"]["family_e"] = [{"variant": x["variant"], "oos": x["oos"]} for x in e_detail]
        report["round3"]["family_e_trades"] = e_detail
        e_stress = run_family_e_gated(
            args.ml_dir,
            pairs,
            oos_start,
            base_list[:3],
            (0.50,),
            args.data_dir,
            spread_mult=2.0,
        )
        report["round3"]["family_e_2x_spread"] = [
            {"variant": r["variant"], "oos": summarize_trades(filter_oos(r["trades"], oos_start), 30.0)}
            for r in e_stress
        ]

    # ----- Round 4: IS hold + Monte Carlo on best OOS pool -----
    if args.round in (0, 4):
        print("Round 4: in-sample check + bootstrap PF …")
        mc_out = []
        n_mc = 50 if args.quick else 1000
        for block in report.get("round3", {}).get("family_e_trades", []):
            if block["oos"]["n"] < 5:
                continue
            pnls = [float(t["pnl_pips"]) for t in block["trades_oos"]]
            mc_out.append({"variant": block["variant"], "mc": _monte_carlo_pf(pnls, n_mc)})
        report["round4"] = {"monte_carlo": mc_out}
        fe = report.get("round3", {}).get("family_e_trades", [])
        if fe and fe[0].get("trades_oos"):
            a, b, c = _oos_thirds(fe[0]["trades_oos"])
            report["round4"]["oos_three_way_first_e"] = {
                "variant": fe[0]["variant"],
                "a": summarize_trades(a, 30.0),
                "b": summarize_trades(b, 30.0),
                "c": summarize_trades(c, 30.0),
            }

    # Success: any variant PF>=1.3, N>=50 (family A proxy, B, C/D)
    best_any = None
    for r in report.get("round1", {}).get("family_b", []):
        if meets_profit_gate(r["oos"], min_n=30 if args.quick else 50):
            best_any = r
            break
    if best_any is None:
        for m in report.get("round1", {}).get("family_a_top", [])[:30]:
            if m.get("pf_proxy", 0) >= 1.3 and m.get("n_oos_total", 0) >= (30 if args.quick else 50):
                best_any = m
                break
    report["plan_success_pf_13"] = best_any is not None
    report["best_hint"] = best_any

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
