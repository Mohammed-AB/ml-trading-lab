#!/usr/bin/env python3
"""High win-rate strategy lab: multi-TF families, OOS scoring, rounds 1–3.

Implements the plan families:
  F1 — session mean-reversion (Asian / late NY, RSI/BB, ATR regime, H1 flat)
  F2 — hour-edge directional (UTC hours from hour_edge_filter + H1/D1 alignment)
  F3 — compression / squeeze breakout (London / NY open windows)
  F4 — ML ensemble with high threshold + optional hour/vol filters (WF models)

Run: ``python strategy_lab.py`` or ``python -m strategy_arena --lab``
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_strategies import add_indicators, resample_generic, simulate_trades_vec  # noqa: E402

from strategy_arena.config import DEFAULT_DATA_RAW, OOS_START, V2_PAIRS  # noqa: E402
from strategy_arena.loader import list_available_pairs, load_oanda_m1  # noqa: E402

from scalp_mode.engine.hour_edge_filter import LONG_EDGE_HOURS, SHORT_EDGE_HOURS  # noqa: E402
from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair, spread_half_price  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def spread_price_full(pair: str) -> float:
    return SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip_for_pair(pair)


def m1_end_ix_from_m5(i: int) -> int:
    return (i + 1) * 5 - 1


def tf_ix_from_m5(i: int, m1_bars: int) -> int:
    """M5 bar index ``i`` → index into a series built from M1 with ``m1_bars`` per bar."""
    return m1_end_ix_from_m5(i) // m1_bars


def profit_factor(pnls: list[float]) -> float:
    wins = sum(x for x in pnls if x > 0)
    losses = -sum(x for x in pnls if x < 0)
    if losses < 1e-9:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def summarize_trades(trades: list[dict], n_days: float = 1.0) -> dict[str, Any]:
    if not trades:
        return {
            "n": 0,
            "wr": 0.0,
            "wr_pct": 0.0,
            "pf": 0.0,
            "total_pips": 0.0,
            "avg_pips": 0.0,
            "tpd": 0.0,
        }
    pnls = [float(t["pnl_pips"]) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    n = len(pnls)
    pf_raw = float(profit_factor(pnls))
    if not np.isfinite(pf_raw):
        pf_raw = 99.0 if pf_raw > 0 else 0.0
    return {
        "n": n,
        "wr": wins / n,
        "wr_pct": 100.0 * wins / n,
        "pf": pf_raw,
        "total_pips": float(sum(pnls)),
        "avg_pips": float(np.mean(pnls)),
        "tpd": n / max(n_days, 1e-6),
    }


def filter_oos(trades: list[dict], oos_start: pd.Timestamp) -> list[dict]:
    oos = pd.Timestamp(oos_start)
    if oos.tz is None:
        oos = oos.tz_localize("UTC")
    else:
        oos = oos.tz_convert("UTC")
    out: list[dict] = []
    for t in trades:
        ts = pd.Timestamp(t["timestamp"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if ts >= oos:
            out.append(t)
    return out


def split_oos_halves(trades: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(trades) < 4:
        return trades, []
    tss = sorted(pd.Timestamp(t["timestamp"]) for t in trades)
    mid = tss[len(tss) // 2]
    a, b = [], []
    for t in trades:
        ts = pd.Timestamp(t["timestamp"])
        if ts < mid:
            a.append(t)
        else:
            b.append(t)
    return a, b


def sim_from_signals(
    df: pd.DataFrame,
    pair: str,
    pip: float,
    spread: float,
    idx: list[int],
    dirs: list[int],
    ent: list[float],
    sls: list[float],
    tps: list[float],
    max_bars: int,
) -> list[dict]:
    if not idx:
        return []
    hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))
    sim = simulate_trades_vec(
        df["high"].values,
        df["low"].values,
        df["close"].values,
        np.array(idx, dtype=np.int64),
        np.array(dirs, dtype=np.int64),
        np.array(ent, dtype=np.float64),
        np.array(sls, dtype=np.float64),
        np.array(tps, dtype=np.float64),
        max_bars,
        pip,
        half_spread=hsp,
    )
    hours = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    tss = df["timestamp"].values
    trades: list[dict] = []
    for k in range(len(idx)):
        trades.append(
            {
                "family": "",
                "variant": "",
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


def build_timeframes(df_m1: pd.DataFrame, pair: str) -> dict[str, pd.DataFrame]:
    m5 = add_indicators(resample_generic(df_m1, 5), pair)
    m15 = add_indicators(resample_generic(df_m1, 15), pair)
    h1 = add_indicators(resample_generic(df_m1, 60), pair)
    h4 = add_indicators(resample_generic(df_m1, 240), pair)
    d1 = add_indicators(resample_generic(df_m1, 1440), pair)
    return {"m5": m5, "m15": m15, "h1": h1, "h4": h4, "d1": d1}


# ---------------------------------------------------------------------------
# Family 1 — session mean-reversion
# ---------------------------------------------------------------------------


def _session_asian(hour: int) -> bool:
    return hour >= 22 or hour <= 5


def _session_late_ny(hour: int) -> bool:
    return hour in (20, 21)


def signals_f1_mr(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    pair: str,
    spread: float,
    *,
    session: str,
    tp_atr: float,
    sl_atr: float,
    rsi_lo: int,
    rsi_hi: int,
    flat_max: float,
    max_bars: int = 20,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    pip = float(m5["pip"].iloc[0])
    h = m5["high"].values
    l = m5["low"].values
    c = m5["close"].values
    atr = m5["atr14"].values
    rsi = m5["rsi14"].values
    bu = m5["bb_upper"].values
    bl = m5["bb_lower"].values
    hour = m5["hour"].values.astype(np.int64)
    n = len(m5)
    nh1 = len(h1)
    h1_e20 = h1["ema20"].values
    h1_e40 = h1["ema40"].values
    h1_atr = h1["atr14"].values

    idx, dirs, ents, sls, tps = [], [], [], [], []
    last = -max_bars

    for i in range(120, n - 2):
        if i - last < 8:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        win = atr[i - 119 : i + 1]
        win = win[np.isfinite(win)]
        if len(win) < 60:
            continue
        if a > float(np.percentile(win, 60)):
            continue
        hi = hour[i]
        if session == "asian" and not _session_asian(hi):
            continue
        if session == "late_ny" and not _session_late_ny(hi):
            continue
        if session == "both" and not (_session_asian(hi) or _session_late_ny(hi)):
            continue

        h1_i = min(tf_ix_from_m5(i, 60), nh1 - 1)
        ha = h1_atr[h1_i]
        if not np.isfinite(ha) or ha <= 0:
            continue
        if abs(h1_e20[h1_i] - h1_e40[h1_i]) / ha > flat_max:
            continue

        rsi_i = rsi[i]
        touch_bb_lo = l[i] <= bl[i] + 0.05 * a
        touch_bb_hi = h[i] >= bu[i] - 0.05 * a

        # Long MR
        if rsi_i < rsi_lo and touch_bb_lo:
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
            continue

        # Short MR
        if rsi_i > rsi_hi and touch_bb_hi:
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
# Family 2 — hour-edge directional
# ---------------------------------------------------------------------------


def signals_f2_hour_edge(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
    h4: pd.DataFrame,
    d1: pd.DataFrame,
    pair: str,
    spread: float,
    *,
    tp_atr: float,
    sl_atr: float,
    max_bars: int = 30,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    _ = pair, max_bars
    c = m5["close"].values
    l = m5["low"].values
    h = m5["high"].values
    atr = m5["atr14"].values
    rsi = m5["rsi14"].values
    e9 = m5["ema9"].values
    e20 = m5["ema20"].values
    hour = m5["hour"].values.astype(np.int64)
    n = len(m5)
    nh1, nh4, nd1 = len(h1), len(h4), len(d1)
    h1_e9, h1_e20 = h1["ema9"].values, h1["ema20"].values
    h4_e20, h4_e40 = h4["ema20"].values, h4["ema40"].values
    d1_e20, d1_e40 = d1["ema20"].values, d1["ema40"].values

    long_hours = set(LONG_EDGE_HOURS.keys())
    short_hours = set(SHORT_EDGE_HOURS.keys())

    idx, dirs, ents, sls, tps = [], [], [], [], []
    last = -15

    for i in range(50, n - 2):
        if i - last < 10:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        hi = hour[i]
        h1_i = min(tf_ix_from_m5(i, 60), nh1 - 1)
        h4_i = min(tf_ix_from_m5(i, 240), nh4 - 1)
        d1_i = min(tf_ix_from_m5(i, 1440), nd1 - 1)

        if hi in long_hours:
            if not (h1_e9[h1_i] > h1_e20[h1_i] and e9[i] > e20[i]):
                continue
            if rsi[i] <= 40:
                continue
            if not (l[i] <= e20[i] + 0.2 * a and c[i] > e20[i]):
                continue
            if h4_e20[h4_i] < h4_e40[h4_i] * 0.999:
                continue
            if d1_e20[d1_i] < d1_e40[d1_i] * 0.998:
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
        elif hi in short_hours:
            if not (h1_e9[h1_i] < h1_e20[h1_i] and e9[i] < e20[i]):
                continue
            if rsi[i] >= 60:
                continue
            if not (h[i] >= e20[i] - 0.2 * a and c[i] < e20[i]):
                continue
            if h4_e20[h4_i] > h4_e40[h4_i] * 1.001:
                continue
            if d1_e20[d1_i] > d1_e40[d1_i] * 1.002:
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
# Family 3 — squeeze / compression breakout (session windows)
# ---------------------------------------------------------------------------


def signals_f3_compress(
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    pair: str,
    spread: float,
    *,
    session: str,
    squeeze_pct: float,
    max_bars: int = 40,
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    c = m5["close"].values
    h = m5["high"].values
    l = m5["low"].values
    atr = m5["atr14"].values
    sq = m5["bb_squeeze_pct"].values
    e9, e20 = m5["ema9"].values, m5["ema20"].values
    hour = m5["hour"].values.astype(np.int64)
    br = m5["bar_range"].values
    n = len(m5)
    nm15 = len(m15)
    m15_e9 = m15["ema9"].values
    m15_e20 = m15["ema20"].values

    def sess_ok(hi: int) -> bool:
        if session == "london":
            return 7 <= hi <= 9
        if session == "ny":
            return 13 <= hi <= 15
        return (7 <= hi <= 9) or (13 <= hi <= 15)

    idx, dirs, ents, sls, tps = [], [], [], [], []
    last = -12

    for i in range(25, n - 3):
        if i - last < 8:
            continue
        if not sess_ok(int(hour[i])):
            continue
        if not (sq[i] < squeeze_pct and sq[i - 1] < squeeze_pct):
            continue
        window = br[i - 19 : i + 1]
        if len(window) < 20 or br[i] > np.min(window) * 1.15:
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
            tp = entry + 1.5 * rng
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
            tp = entry - 1.5 * rng
            idx.append(i)
            dirs.append(-1)
            ents.append(entry)
            sls.append(sl)
            tps.append(tp)
            last = i

    return idx, dirs, ents, sls, tps


# ---------------------------------------------------------------------------
# Family 4 — ML selective (WF + high threshold)
# ---------------------------------------------------------------------------


def run_family4_ml(
    ml_dir: Path,
    pairs: list[str],
    oos_start: pd.Timestamp,
    *,
    thresholds: Iterable[float],
    tp_mult: float,
    sl_mult: float,
    hour_filter: bool,
    vol_cap_pct: float = 80.0,
) -> list[dict]:
    """Returns flat list of trade dicts tagged family F4 (OOS only)."""
    ml_dir = Path(ml_dir)
    man_path = ml_dir / "wf_manifest.json"
    if not man_path.is_file():
        return []
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    feat_names: list[str] = manifest["feature_names"]
    folds: list[dict] = manifest.get("folds") or []
    if not folds:
        return []

    try:
        import lightgbm as lgb  # noqa: WPS433
    except ImportError:
        return []

    from ml_labels import N_FUTURE  # noqa: WPS433

    edge_hours = set(LONG_EDGE_HOURS.keys()) | set(SHORT_EDGE_HOURS.keys())
    gts = pd.Timestamp(oos_start)
    if gts.tz is None:
        gts = gts.tz_localize("UTC")
    else:
        gts = gts.tz_convert("UTC")

    all_trades: list[dict] = []

    for thresh in thresholds:
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
                miss = [c for c in need if c not in raw.columns]
                if miss:
                    del raw
                    gc.collect()
                    continue
                df = raw[need].copy()
                del raw
                gc.collect()
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
                # vol filter: ATR not above rolling vol_cap_pct percentile (favorable regime)
                atr_s = pd.Series(atr)
                q = atr_s.rolling(100, min_periods=30).quantile(vol_cap_pct / 100.0)
                atr_pct = (atr_s <= q.fillna(atr_s)).values.astype(bool)
                X = np.nan_to_num(
                    df[feat_names].values.astype(np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                p_lo = long_m.predict(X).astype(np.float32)
                p_sh = short_m.predict(X).astype(np.float32)
                if hour_filter:
                    hours = tsv.hour.values.astype(np.int64)
                    mask_h = np.isin(hours, list(edge_hours))
                else:
                    mask_h = np.ones(len(df), dtype=bool)
                # apply threshold + hour + vol by masking probs
                p_lo = np.where(atr_pct & mask_h, p_lo, 0.0)
                p_sh = np.where(atr_pct & mask_h, p_sh, 0.0)
                # custom backtest with threshold
                from scalp_mode.ml.bar_features import pip_for_pair as _pip  # noqa
                from scalp_mode.ml.bar_features import spread_half_price as _shp  # noqa

                pip = _pip(p)
                sh = _shp(p, SPREAD_PIPS_DEFAULT.get(p, 1.5))
                trades_local: list[dict] = []
                ii = 0
                n = len(cl)
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
                    from strategy_arena.ml_sweep import _sim_trade_atr  # noqa

                    pnl, bars, _ = _sim_trade_atr(
                        ii, direction, h, lo, cl, atr, pip, sh, N_FUTURE, sl_mult, tp_mult
                    )
                    trades_local.append(
                        {
                            "family": "F4_ML",
                            "variant": f"thr{thresh:.2f}_h{int(hour_filter)}_tp{tp_mult}_sl{sl_mult}",
                            "pair": p,
                            "direction": direction,
                            "pnl_pips": pnl,
                            "bars_held": bars,
                            "timestamp": str(tsv[ii]),
                            "hour": int(tsv[ii].hour),
                        }
                    )
                    ii += max(1, bars)
                all_trades.extend(trades_local)
                del df, X
                gc.collect()
            del long_m, short_m
            gc.collect()

    return all_trades


# ---------------------------------------------------------------------------
# Lab driver
# ---------------------------------------------------------------------------


@dataclass
class VariantResult:
    family: str
    variant: str
    trades_oos: list[dict]
    metrics: dict[str, Any]
    pairs_hit: int


def meets_soft_gate(m: dict[str, Any], min_n: int = 30) -> bool:
    return m["n"] >= min_n and m["wr_pct"] >= 65.0 and m["pf"] >= 1.2


def meets_hard_plan(m: dict[str, Any], min_n: int = 30) -> bool:
    return m["n"] >= min_n and m["wr_pct"] >= 70.0 and m["pf"] >= 1.3


def score_variant(m: dict[str, Any]) -> float:
    """Higher is better; zero if too few trades."""
    if m["n"] < 10:
        return 0.0
    return m["wr_pct"] * min(m["pf"], 3.0) * np.log1p(m["n"])


def run_round1(
    data_dir: Path,
    pairs: list[str],
    oos_start: pd.Timestamp,
    ml_dir: Path | None = None,
) -> list[VariantResult]:
    results: list[VariantResult] = []
    n_days_oos = 30.0

    for pair in pairs:
        df_m1 = load_oanda_m1(data_dir, pair)
        if df_m1.empty or len(df_m1) < 5000:
            continue
        tf = build_timeframes(df_m1, pair)
        m5, m15, h1, h4, d1 = tf["m5"], tf["m15"], tf["h1"], tf["h4"], tf["d1"]
        spread = spread_price_full(pair)
        pip = pip_for_pair(pair)

        # F1 grid
        for session in ("asian", "late_ny", "both"):
            for tp_atr in (0.5, 0.65, 0.8):
                for sl_atr in (1.2, 1.5):
                    for rsi_lo, rsi_hi in ((25, 75), (30, 70)):
                        for flat_max in (0.35, 0.5):
                            tag = f"F1_{session}_tp{tp_atr}_sl{sl_atr}_rsi{rsi_lo}_flat{flat_max}"
                            ix, di, en, sl, tp = signals_f1_mr(
                                m5,
                                h1,
                                pair,
                                spread,
                                session=session,
                                tp_atr=tp_atr,
                                sl_atr=sl_atr,
                                rsi_lo=rsi_lo,
                                rsi_hi=rsi_hi,
                                flat_max=flat_max,
                            )
                            tr = sim_from_signals(m5, pair, pip, spread, ix, di, en, sl, tp, 22)
                            for t in tr:
                                t["family"] = "F1_MR"
                                t["variant"] = tag
                            oos = filter_oos(tr, oos_start)
                            m = summarize_trades(oos, n_days_oos)
                            results.append(
                                VariantResult("F1_MR", tag, oos, m, 1 if m["n"] >= 5 else 0)
                            )

        # F2 grid
        for tp_atr, sl_atr in ((1.0, 1.0), (0.9, 1.1), (1.1, 0.9)):
            tag = f"F2_hour_tp{tp_atr}_sl{sl_atr}"
            ix, di, en, sl, tp = signals_f2_hour_edge(
                m5, h1, h4, d1, pair, spread, tp_atr=tp_atr, sl_atr=sl_atr
            )
            tr = sim_from_signals(m5, pair, pip, spread, ix, di, en, sl, tp, 32)
            for t in tr:
                t["family"] = "F2_HOUR"
                t["variant"] = tag
            oos = filter_oos(tr, oos_start)
            m = summarize_trades(oos, n_days_oos)
            results.append(VariantResult("F2_HOUR", tag, oos, m, 1 if m["n"] >= 5 else 0))

        # F3 grid
        for session in ("london", "ny", "both"):
            for sq in (0.2, 0.3):
                tag = f"F3_{session}_sq{sq}"
                ix, di, en, sl, tp = signals_f3_compress(
                    m5, m15, pair, spread, session=session, squeeze_pct=sq
                )
                tr = sim_from_signals(m5, pair, pip, spread, ix, di, en, sl, tp, 45)
                for t in tr:
                    t["family"] = "F3_COMP"
                    t["variant"] = tag
                oos = filter_oos(tr, oos_start)
                m = summarize_trades(oos, n_days_oos)
                results.append(VariantResult("F3_COMP", tag, oos, m, 1 if m["n"] >= 5 else 0))

        del tf, m5, m15, h1, h4, d1, df_m1
        gc.collect()

    # Aggregate F1/F2/F3 by variant name across pairs
    merged = _aggregate_by_variant(results)
    # F4 once (all pairs inside): hour-gated + unrestricted thresholds
    ml_root = Path(ml_dir) if ml_dir is not None else ROOT / "data" / "ml"
    for hour_f, th_list in (
        (True, (0.68, 0.72, 0.76)),
        (False, (0.65, 0.70, 0.75)),
    ):
        f4_tr = run_family4_ml(
            ml_root,
            pairs,
            oos_start,
            thresholds=th_list,
            tp_mult=1.5,
            sl_mult=1.0,
            hour_filter=hour_f,
        )
        by_v: dict[str, list[dict]] = {}
        for t in f4_tr:
            v = t.get("variant", "F4")
            by_v.setdefault(v, []).append(t)
        for v, trs in by_v.items():
            m = summarize_trades(trs, n_days_oos)
            merged.append(VariantResult("F4_ML", v, trs, m, len({x["pair"] for x in trs})))

    return merged


def _aggregate_by_variant(per_pair_results: list[VariantResult]) -> list[VariantResult]:
    buckets: dict[str, list[dict]] = {}
    fam: dict[str, str] = {}
    for r in per_pair_results:
        buckets.setdefault(r.variant, []).extend(r.trades_oos)
        fam[r.variant] = r.family
    out: list[VariantResult] = []
    for v, trs in buckets.items():
        m = summarize_trades(trs, 30.0)
        pairs_hit = len({t["pair"] for t in trs})
        out.append(VariantResult(fam[v], v, trs, m, pairs_hit))
    return out


def run_round2(
    data_dir: Path,
    pairs: list[str],
    oos_start: pd.Timestamp,
    seeds: list[VariantResult],
) -> list[VariantResult]:
    """Expand SL/TP grid when F1 or F2 appears in top seeds (no variant-string parsing)."""
    seeds_sorted = sorted(seeds, key=lambda r: score_variant(r.metrics), reverse=True)
    topn = seeds_sorted[:12]
    has_f1 = any(s.family == "F1_MR" for s in topn)
    has_f2 = any(s.family == "F2_HOUR" for s in topn)
    extra: list[VariantResult] = []
    n_days_oos = 30.0

    if has_f1:
        for session in ("asian", "late_ny", "both"):
            for tp2 in (0.45, 0.55, 0.65, 0.75):
                for sl2 in (1.15, 1.35, 1.55):
                    tag = f"F1R2_{session}_tp{tp2:.2f}_sl{sl2:.2f}"
                    bucket: list[dict] = []
                    for pair in pairs:
                        df_m1 = load_oanda_m1(data_dir, pair)
                        if df_m1.empty:
                            continue
                        tf = build_timeframes(df_m1, pair)
                        m5, h1 = tf["m5"], tf["h1"]
                        spread = spread_price_full(pair)
                        pip = pip_for_pair(pair)
                        ix, di, en, sl, tp = signals_f1_mr(
                            m5,
                            h1,
                            pair,
                            spread,
                            session=session,
                            tp_atr=tp2,
                            sl_atr=sl2,
                            rsi_lo=28,
                            rsi_hi=72,
                            flat_max=0.4,
                        )
                        tr = sim_from_signals(m5, pair, pip, spread, ix, di, en, sl, tp, 22)
                        for t in tr:
                            t["family"] = "F1_MR_R2"
                            t["variant"] = tag
                        bucket.extend(filter_oos(tr, oos_start))
                        del tf, df_m1
                        gc.collect()
                    m = summarize_trades(bucket, n_days_oos)
                    extra.append(
                        VariantResult(
                            "F1_MR_R2",
                            tag,
                            bucket,
                            m,
                            len({t["pair"] for t in bucket}),
                        )
                    )

    if has_f2:
        for tp_atr, sl_atr in ((1.0, 1.0), (1.2, 0.85), (0.85, 1.15), (1.0, 1.3)):
            tag = f"F2R2_tp{tp_atr}_sl{sl_atr}"
            bucket: list[dict] = []
            for pair in pairs:
                df_m1 = load_oanda_m1(data_dir, pair)
                if df_m1.empty:
                    continue
                tf = build_timeframes(df_m1, pair)
                m5, h1, h4, d1 = tf["m5"], tf["h1"], tf["h4"], tf["d1"]
                spread = spread_price_full(pair)
                pip = pip_for_pair(pair)
                ix, di, en, sl, tp = signals_f2_hour_edge(
                    m5, h1, h4, d1, pair, spread, tp_atr=tp_atr, sl_atr=sl_atr
                )
                tr = sim_from_signals(m5, pair, pip, spread, ix, di, en, sl, tp, 35)
                for t in tr:
                    t["family"] = "F2_HOUR_R2"
                    t["variant"] = tag
                bucket.extend(filter_oos(tr, oos_start))
                del tf, df_m1
                gc.collect()
            m = summarize_trades(bucket, n_days_oos)
            extra.append(
                VariantResult(
                    "F2_HOUR_R2",
                    tag,
                    bucket,
                    m,
                    len({t["pair"] for t in bucket}),
                )
            )

    return extra


def run_round3_mtf_wf(
    data_dir: Path,
    pairs: list[str],
    oos_start: pd.Timestamp,
    candidates: list[VariantResult],
) -> list[dict]:
    """Walk-forward style: OOS split halves + require H4 alignment on F2-style configs."""
    report: list[dict] = []
    top = sorted(candidates, key=lambda r: score_variant(r.metrics), reverse=True)[:6]
    n_days_oos = 30.0

    for cand in top:
        a, b = split_oos_halves(cand.trades_oos)
        ma, mb = summarize_trades(a, n_days_oos), summarize_trades(b, n_days_oos)
        report.append(
            {
                "variant": cand.variant,
                "family": cand.family,
                "combined": cand.metrics,
                "oos_half_a": ma,
                "oos_half_b": mb,
                "passes_split": ma["n"] >= 5 and mb["n"] >= 5 and ma["wr_pct"] >= 50 and mb["wr_pct"] >= 50,
            }
        )

    # MTF stress: re-run best F2 with extra H1 slope filter
    f2_seeds = [c for c in candidates if c.family.startswith("F2") and c.metrics["n"] >= 15]
    f2_seeds.sort(key=lambda r: score_variant(r.metrics), reverse=True)
    for seed in f2_seeds[:2]:
        tag = seed.variant + "_h4strict"
        bucket: list[dict] = []
        for pair in pairs:
            df_m1 = load_oanda_m1(data_dir, pair)
            if df_m1.empty:
                continue
            tf = build_timeframes(df_m1, pair)
            m5, h1, h4, d1 = tf["m5"], tf["h1"], tf["h4"], tf["d1"]
            spread = spread_price_full(pair)
            pip = pip_for_pair(pair)
            ix, di, en, sl, tp = signals_f2_hour_edge(
                m5, h1, h4, d1, pair, spread, tp_atr=1.0, sl_atr=1.0
            )
            # filter indices where H4 trend strongly agrees
            h4_e20 = h4["ema20"].values
            h4_e40 = h4["ema40"].values
            nh4 = len(h4)
            ix2, d2, e2, s2, t2 = [], [], [], [], []
            for k in range(len(ix)):
                i = ix[k]
                h4_i = min(tf_ix_from_m5(i, 240), nh4 - 1)
                if di[k] == 1 and h4_e20[h4_i] <= h4_e40[h4_i] * 1.002:
                    continue
                if di[k] == -1 and h4_e20[h4_i] >= h4_e40[h4_i] * 0.998:
                    continue
                ix2.append(i)
                d2.append(di[k])
                e2.append(en[k])
                s2.append(sl[k])
                t2.append(tp[k])
            tr = sim_from_signals(m5, pair, pip, spread, ix2, d2, e2, s2, t2, 35)
            for t in tr:
                t["family"] = "F2_MTF_R3"
                t["variant"] = tag
            bucket.extend(filter_oos(tr, oos_start))
            del tf, df_m1
            gc.collect()
        m = summarize_trades(bucket, n_days_oos)
        report.append(
            {
                "variant": tag,
                "family": "F2_MTF_R3",
                "combined": m,
                "note": "H4 strict confluence rerun",
            }
        )

    return report


def print_leaderboard(rows: list[VariantResult], title: str, limit: int = 25) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Variant':48s} {'N':>6} {'WR%':>7} {'PF':>6} {'Pips':>10} {'Pairs':>5}")
    print("-" * 88)
    for r in sorted(rows, key=lambda x: score_variant(x.metrics), reverse=True)[:limit]:
        m = r.metrics
        print(
            f"{r.variant[:48]:48s} {m['n']:6d} {m['wr_pct']:6.1f} {m['pf']:6.2f} "
            f"{m['total_pips']:+10.0f} {r.pairs_hit:5d}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="High WR strategy lab (plan implementation).")
    ap.add_argument("--data-dir", type=Path, default=ROOT / DEFAULT_DATA_RAW)
    ap.add_argument("--round", type=int, default=0, help="Run only round N (1,2,3); 0=all")
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "strategy_lab_results.json")
    ap.add_argument(
        "--ml-dir",
        type=Path,
        default=None,
        help="ML artifacts (wf_manifest.json, features_*.parquet); default data/ml",
    )
    args = ap.parse_args(argv)

    oos_start = pd.Timestamp(OOS_START)
    pairs = [p for p in V2_PAIRS if p in list_available_pairs(args.data_dir)]
    if len(pairs) < 3:
        print(
            f"Need at least 3 V2 pair CSVs under {args.data_dir}. Found: {pairs}",
            file=sys.stderr,
        )
        return 1

    all_rows: list[VariantResult] = []
    wf_report: list[dict] = []

    if args.round in (0, 1):
        print("Round 1: all families (aggregated per variant across pairs)...")
        r1 = run_round1(args.data_dir, pairs, oos_start, ml_dir=args.ml_dir)
        all_rows.extend(r1)
        print_leaderboard(r1, "Round 1 (OOS)")

    if args.round in (0, 2):
        seeds = sorted(
            all_rows or run_round1(args.data_dir, pairs, oos_start, ml_dir=args.ml_dir),
            key=lambda r: score_variant(r.metrics),
            reverse=True,
        )[:12]
        print("\nRound 2: grid on top F1/F2 seeds...")
        r2 = run_round2(args.data_dir, pairs, oos_start, seeds)
        all_rows.extend(r2)
        print_leaderboard(r2, "Round 2 (OOS)")

    if args.round in (0, 3):
        pool_rows = all_rows
        if args.round == 3 and not pool_rows:
            print("Round 3 only: running rounds 1–2 to build candidate pool...")
            r1_only = run_round1(args.data_dir, pairs, oos_start, ml_dir=args.ml_dir)
            seeds_only = sorted(r1_only, key=lambda r: score_variant(r.metrics), reverse=True)[:12]
            r2_only = run_round2(args.data_dir, pairs, oos_start, seeds_only)
            all_rows.extend(r1_only)
            all_rows.extend(r2_only)
            pool_rows = r1_only + r2_only
        pool = sorted(pool_rows, key=lambda r: score_variant(r.metrics), reverse=True)[:24]
        print("\nRound 3: OOS halves + MTF H4 strict...")
        wf_report = run_round3_mtf_wf(args.data_dir, pairs, oos_start, pool)
        print(json.dumps(wf_report, indent=2)[:4000])

    best = max(all_rows, key=lambda r: score_variant(r.metrics)) if all_rows else None
    plan_ok = best is not None and meets_hard_plan(best.metrics)

    out = {
        "oos_start": OOS_START,
        "pairs": pairs,
        "plan_success_70_wr_13_pf": plan_ok,
        "best_variant": best.variant if best else None,
        "best_metrics": best.metrics if best else {},
        "best_meets_soft_gate_65_12": meets_soft_gate(best.metrics) if best else False,
        "walk_forward": wf_report,
        "n_variants_scored": len(all_rows),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote summary {args.out_json}")

    if best:
        print(
            f"\nBest: {best.variant}  WR={best.metrics['wr_pct']:.1f}%  PF={best.metrics['pf']:.2f}  N={best.metrics['n']}"
        )
        if not plan_ok:
            print(
                "Note: 70%+ WR and PF>1.3 on OOS with >=30 trades is aggressive; "
                "review JSON and tighten live gates if promoting a variant.",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
