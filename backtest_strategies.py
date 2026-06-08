"""Backtest 13 book-derived strategies on M5-resampled historical data.

Reads M1 from ~/Downloads/FX-1-Minute-Data-master/forex_data/1min/
Resamples to M5 (the native timeframe the book strategies use).

Usage:
    python3 backtest_strategies.py
    python3 backtest_strategies.py --years 5
"""

import argparse
import json
import sys
import time as time_mod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

try:
    from scalp_mode.ml.bar_features import (  # noqa: E402
        SPREAD_PIPS_DEFAULT as SPREAD_PIPS,
        spread_half_price,
    )
except ImportError:

    def spread_half_price(pair: str, spread_pips: float | None = None) -> float:
        pip = 0.01 if "JPY" in pair.upper() else 0.0001
        sp = spread_pips if spread_pips is not None else 1.5
        return (sp * pip) / 2.0

    SPREAD_PIPS = {
        "EUR_USD": 1.5,
        "GBP_USD": 2.0,
        "USD_JPY": 1.5,
        "USD_CAD": 2.0,
        "AUD_USD": 2.0,
        "NZD_USD": 2.0,
        "USD_CHF": 2.0,
        "EUR_GBP": 2.0,
    }


DATA_DIR = Path.home() / "Downloads" / "FX-1-Minute-Data-master" / "forex_data" / "1min"
OUTPUT_DIR = Path("data/brain/lessons")

PAIR_MAP = {
    "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY",
    "USDCHF": "USD_CHF", "USDCAD": "USD_CAD", "AUDUSD": "AUD_USD",
    "NZDUSD": "NZD_USD", "EURGBP": "EUR_GBP",
}


def load_pair(filename: str, pair: str, years: int = 0) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
    if years > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=years * 365)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df


def resample_generic(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Resample M1 OHLCV to Mn by grouping ``n`` consecutive rows (bar-count, not wall-clock).

    ``n=5`` matches :func:`resample_m5`; ``n=15`` approximates M15, ``n=60`` H1, ``n=240`` H4.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    groups = len(df) // n
    if groups == 0:
        return df.iloc[0:0].copy()
    trimmed = df.iloc[: groups * n].copy()
    trimmed["g"] = np.repeat(range(groups), n)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "timestamp": "last",
    }
    return trimmed.groupby("g").agg(agg).reset_index(drop=True)


def resample_m5(df: pd.DataFrame) -> pd.DataFrame:
    return resample_generic(df, 5)


def add_indicators(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    n = len(df)
    pip = 0.01 if "JPY" in pair else 0.0001

    df["ema9"] = pd.Series(c).ewm(span=9, adjust=False).mean().values
    df["ema20"] = pd.Series(c).ewm(span=20, adjust=False).mean().values
    df["ema40"] = pd.Series(c).ewm(span=40, adjust=False).mean().values

    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    df["atr14"] = pd.Series(tr).rolling(14).mean().values

    sma20 = pd.Series(c).rolling(20).mean().values
    std20 = pd.Series(c).rolling(20).std().values
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"] = sma20
    bw = pd.Series(std20)
    bw_min = bw.rolling(100, min_periods=20).min()
    bw_max = bw.rolling(100, min_periods=20).max()
    denom = bw_max - bw_min
    df["bb_squeeze_pct"] = np.where(denom > 0, (bw - bw_min) / denom, 0.5)

    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).values

    df["bar_range"] = h - l
    df["body"] = np.abs(c - o)
    df["is_bull"] = c > o
    df["is_bear"] = c < o
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    df["pip"] = pip

    ema20_v = df["ema20"].values
    slope = np.zeros(n)
    slope[10:] = ema20_v[10:] - ema20_v[:-10]
    df["ema20_slope"] = slope

    # MACD + Stochastic (used by strategy_arena research_v2 / research_pdf)
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    df["macd"] = ema12 - ema26
    df["macd_signal"] = pd.Series(df["macd"]).ewm(span=9, adjust=False).mean().values
    low14 = pd.Series(l).rolling(14).min().values
    high14 = pd.Series(h).rolling(14).max().values
    denom = high14 - low14
    st_k = np.where(denom > 1e-15, 100.0 * (c - low14) / denom, 50.0)
    df["stoch_k"] = st_k
    df["stoch_d"] = pd.Series(st_k).rolling(3, min_periods=1).mean().values

    return df


# ---------------------------------------------------------------------------
#  VECTORISED TRADE SIMULATOR
# ---------------------------------------------------------------------------

def simulate_trades_vec(
    highs,
    lows,
    closes,
    indices,
    directions,
    entries,
    sls,
    tps,
    max_bars,
    pip,
    half_spread: float = 0.0,
):
    """Batch-simulate many trades at once.

    If ``half_spread`` > 0, exits assume bid/ask: long exits sell at bid
    (TP/SL levels minus ``half_spread``); short exits buy at ask (levels plus
    ``half_spread``). Time exits use the same adjustment on ``closes``.

    When both SL and TP are touched the same bar, the **conservative** outcome
    is used (count as SL / full loss side).
    """
    n_data = len(highs)
    results = []
    for k in range(len(indices)):
        idx = indices[k]
        d = directions[k]
        entry = entries[k]
        sl = sls[k]
        tp = tps[k]
        end = min(idx + max_bars, n_data)
        exit_r = "time"
        pnl = 0.0
        bars = end - idx

        for j in range(idx + 1, end):
            if d == 1:  # long
                hit_sl = lows[j] <= sl
                hit_tp = highs[j] >= tp
                if hit_sl and hit_tp:
                    exit_r = "sl"
                    pnl = (sl - half_spread) - entry
                    bars = j - idx
                    break
                if hit_sl:
                    exit_r = "sl"
                    pnl = (sl - half_spread) - entry
                    bars = j - idx
                    break
                if hit_tp:
                    exit_r = "tp"
                    pnl = (tp - half_spread) - entry
                    bars = j - idx
                    break
            else:  # short
                hit_sl = highs[j] >= sl
                hit_tp = lows[j] <= tp
                if hit_sl and hit_tp:
                    exit_r = "sl"
                    pnl = entry - (sl + half_spread)
                    bars = j - idx
                    break
                if hit_sl:
                    exit_r = "sl"
                    pnl = entry - (sl + half_spread)
                    bars = j - idx
                    break
                if hit_tp:
                    exit_r = "tp"
                    pnl = entry - (tp + half_spread)
                    bars = j - idx
                    break
        else:
            c_exit = closes[min(end - 1, n_data - 1)]
            if d == 1:
                pnl = (c_exit - half_spread) - entry
            else:
                pnl = entry - (c_exit + half_spread)

        results.append({
            "exit_reason": exit_r,
            "pnl_pips": round(pnl / pip, 2),
            "bars_held": bars,
        })
    return results


def simulate_trades_advanced(
    highs,
    lows,
    closes,
    indices,
    directions,
    entries,
    sls,
    tps,
    max_bars,
    pip,
    half_spread: float = 0.0,
    atr: np.ndarray | None = None,
    be_trigger_pct: float | None = None,
    be_buffer_pips: float = 1.0,
    trail_start_R: float | None = None,
    trail_lock_R: float | None = None,
    decay_bars: int | None = None,
    decay_progress_frac: float = 0.15,
):
    """Trade simulation with optional breakeven, **R-based** trail lock, and time-decay exit.

    When ``be_trigger_pct``, ``trail_start_R``, and ``decay_bars`` are all ``None``,
    delegates to :func:`simulate_trades_vec` (identical behaviour).

    **Breakeven:** when favourable excursion reaches ``be_trigger_pct`` of the distance
    from entry to TP (long: ``(wm_high - entry) / (tp - entry)``), move SL to entry
    ± ``be_buffer_pips`` pips (only tightens SL).

    **Trail:** when profit in **R** = ``(wm - entry) / risk`` >= ``trail_start_R`` with
    ``risk = abs(entry - initial_sl)``, set SL to lock ``trail_lock_R`` × risk
    (defaults to ``1.0`` if ``trail_start_R`` is set and ``trail_lock_R`` is ``None``).
    Trade-manager parity: start at 2R, lock 1R → ``trail_start_R=2``, ``trail_lock_R=1``.

    **Decay:** after ``decay_bars`` bars since entry, if progress toward TP is below
    ``decay_progress_frac``, exit at that bar's close (reduces slow losers).

    Same-bar SL+TP → **SL (conservative)**.
    """
    use_adv = (
        be_trigger_pct is not None
        or trail_start_R is not None
        or decay_bars is not None
    )
    if not use_adv:
        return simulate_trades_vec(
            highs,
            lows,
            closes,
            indices,
            directions,
            entries,
            sls,
            tps,
            max_bars,
            pip,
            half_spread=half_spread,
        )

    n_data = len(highs)
    if atr is None:
        atr = np.full(n_data, pip * 10.0, dtype=np.float64)
    else:
        atr = np.asarray(atr, dtype=np.float64)

    trail_lock = 1.0 if trail_lock_R is None else float(trail_lock_R)
    trail_start = float(trail_start_R) if trail_start_R is not None else None
    be_buf = be_buffer_pips * pip

    results: list[dict] = []
    for k in range(len(indices)):
        idx = int(indices[k])
        d = int(directions[k])
        entry = float(entries[k])
        initial_sl = float(sls[k])
        tp = float(tps[k])
        sl = initial_sl
        risk = abs(entry - initial_sl)
        if risk < pip * 0.25:
            risk = pip * 0.25

        end = min(idx + max_bars, n_data)
        exit_r = "time"
        pnl = 0.0
        bars = end - idx

        if d == 1:
            tp_dist = tp - entry
            if tp_dist <= pip * 0.1:
                tp_dist = pip * 0.1
            wm = entry
            for j in range(idx + 1, end):
                wm = max(wm, float(highs[j]))
                # decay: chop exit
                if decay_bars is not None and (j - idx) >= decay_bars:
                    prog = (wm - entry) / tp_dist
                    if prog < decay_progress_frac:
                        c_exit = float(closes[j])
                        exit_r = "decay"
                        pnl = (c_exit - half_spread) - entry
                        bars = j - idx
                        break
                # breakeven
                if be_trigger_pct is not None:
                    progress = (wm - entry) / tp_dist
                    if progress >= be_trigger_pct:
                        be_sl = entry + be_buf
                        if be_sl > sl:
                            sl = be_sl
                # trail lock R
                if trail_start is not None:
                    current_r = (wm - entry) / risk
                    if current_r >= trail_start:
                        trail_sl = entry + trail_lock * risk
                        if trail_sl > sl:
                            sl = trail_sl

                hit_sl = float(lows[j]) <= sl
                hit_tp = float(highs[j]) >= tp
                if hit_sl and hit_tp:
                    exit_r = "sl"
                    pnl = (sl - half_spread) - entry
                    bars = j - idx
                    break
                if hit_sl:
                    exit_r = "sl"
                    pnl = (sl - half_spread) - entry
                    bars = j - idx
                    break
                if hit_tp:
                    exit_r = "tp"
                    pnl = (tp - half_spread) - entry
                    bars = j - idx
                    break
            else:
                c_exit = float(closes[min(end - 1, n_data - 1)])
                pnl = (c_exit - half_spread) - entry
        else:  # short
            tp_dist = entry - tp
            if tp_dist <= pip * 0.1:
                tp_dist = pip * 0.1
            wm = entry
            for j in range(idx + 1, end):
                wm = min(wm, float(lows[j]))
                if decay_bars is not None and (j - idx) >= decay_bars:
                    prog = (entry - wm) / tp_dist
                    if prog < decay_progress_frac:
                        c_exit = float(closes[j])
                        exit_r = "decay"
                        pnl = entry - (c_exit + half_spread)
                        bars = j - idx
                        break
                if be_trigger_pct is not None:
                    progress = (entry - wm) / tp_dist
                    if progress >= be_trigger_pct:
                        be_sl = entry - be_buf
                        if be_sl < sl:
                            sl = be_sl
                if trail_start is not None:
                    current_r = (entry - wm) / risk
                    if current_r >= trail_start:
                        trail_sl = entry - trail_lock * risk
                        if trail_sl < sl:
                            sl = trail_sl

                hit_sl = float(highs[j]) >= sl
                hit_tp = float(lows[j]) <= tp
                if hit_sl and hit_tp:
                    exit_r = "sl"
                    pnl = entry - (sl + half_spread)
                    bars = j - idx
                    break
                if hit_sl:
                    exit_r = "sl"
                    pnl = entry - (sl + half_spread)
                    bars = j - idx
                    break
                if hit_tp:
                    exit_r = "tp"
                    pnl = entry - (tp + half_spread)
                    bars = j - idx
                    break
            else:
                c_exit = float(closes[min(end - 1, n_data - 1)])
                pnl = entry - (c_exit + half_spread)

        results.append({
            "exit_reason": exit_r,
            "pnl_pips": round(pnl / pip, 2),
            "bars_held": bars,
        })
    return results


# ---------------------------------------------------------------------------
#  V3 EXIT PROFILES — single-function interface for the mega sweep
# ---------------------------------------------------------------------------

EXIT_PROFILES_V3 = (
    "none", "be", "trail_1R", "trail_2R",
    "chandelier_2", "chandelier_3",
    "atr_trail_1.5", "atr_trail_2.5",
    "partial_tp", "be_trail",
)


def _exit_profile_to_kwargs_v3(profile: str) -> dict:
    """Convert a V3 exit profile name to kwargs for simulate_trades_advanced."""
    if profile == "none":
        return {}
    if profile == "be":
        return {"be_trigger_pct": 0.6}
    if profile == "trail_1R":
        return {"trail_start_R": 1.5, "trail_lock_R": 0.5}
    if profile == "trail_2R":
        return {"trail_start_R": 2.0, "trail_lock_R": 1.0}
    if profile == "be_trail":
        return {"be_trigger_pct": 0.6, "trail_start_R": 2.0, "trail_lock_R": 1.0}
    if profile == "chandelier_2":
        return {"trail_start_R": 0.5, "trail_lock_R": 0.0, "chandelier_mult": 2.0}
    if profile == "chandelier_3":
        return {"trail_start_R": 0.5, "trail_lock_R": 0.0, "chandelier_mult": 3.0}
    if profile == "atr_trail_1.5":
        return {"trail_start_R": 1.0, "trail_lock_R": 0.0, "atr_trail_mult": 1.5}
    if profile == "atr_trail_2.5":
        return {"trail_start_R": 1.0, "trail_lock_R": 0.0, "atr_trail_mult": 2.5}
    if profile == "partial_tp":
        return {"be_trigger_pct": 0.6, "trail_start_R": 1.0, "trail_lock_R": 0.5}
    return {}


def simulate_trades_v3(
    highs,
    lows,
    closes,
    indices,
    directions,
    entries,
    sls,
    tps,
    max_bars,
    pip,
    half_spread: float = 0.0,
    atr: np.ndarray | None = None,
    exit_mode: str = "none",
):
    """Unified V3 simulator: wraps advanced sim with named exit profiles.

    ``exit_mode`` is one of :data:`EXIT_PROFILES_V3`.  Chandelier and ATR-trail
    modes use a per-bar dynamic SL instead of the fixed R-lock:

    - **chandelier_N**: SL = watermark_high − N×ATR (long) each bar.
    - **atr_trail_M**: SL = watermark_high − M×ATR (long) each bar.
    - **partial_tp**: BE at 60 % of TP, then trail at 1R lock 0.5R.

    Falls back to :func:`simulate_trades_advanced` for BE / trail / decay modes.
    """
    if exit_mode in ("none", "be", "trail_1R", "trail_2R", "be_trail", "partial_tp"):
        kw = _exit_profile_to_kwargs_v3(exit_mode)
        kw.pop("chandelier_mult", None)
        kw.pop("atr_trail_mult", None)
        return simulate_trades_advanced(
            highs, lows, closes, indices, directions, entries, sls, tps,
            max_bars, pip, half_spread=half_spread, atr=atr, **kw,
        )

    chandelier_mult = _exit_profile_to_kwargs_v3(exit_mode).get("chandelier_mult")
    atr_trail_mult = _exit_profile_to_kwargs_v3(exit_mode).get("atr_trail_mult")
    trail_mult = chandelier_mult or atr_trail_mult or 2.0

    n_data = len(highs)
    if atr is None:
        atr_arr = np.full(n_data, pip * 10.0, dtype=np.float64)
    else:
        atr_arr = np.asarray(atr, dtype=np.float64)

    results: list[dict] = []
    for k in range(len(indices)):
        idx = int(indices[k])
        d = int(directions[k])
        entry = float(entries[k])
        initial_sl = float(sls[k])
        tp = float(tps[k])
        sl = initial_sl
        risk = abs(entry - initial_sl)
        if risk < pip * 0.25:
            risk = pip * 0.25

        end = min(idx + max_bars, n_data)
        exit_r = "time"
        pnl = 0.0
        bars = end - idx

        if d == 1:
            wm = entry
            for j in range(idx + 1, end):
                wm = max(wm, float(highs[j]))
                a = float(atr_arr[min(j, n_data - 1)])
                new_sl = wm - trail_mult * a
                if new_sl > sl:
                    sl = new_sl

                hit_sl = float(lows[j]) <= sl
                hit_tp = float(highs[j]) >= tp
                if hit_sl and hit_tp:
                    exit_r = "sl"
                    pnl = (sl - half_spread) - entry
                    bars = j - idx
                    break
                if hit_sl:
                    exit_r = "sl"
                    pnl = (sl - half_spread) - entry
                    bars = j - idx
                    break
                if hit_tp:
                    exit_r = "tp"
                    pnl = (tp - half_spread) - entry
                    bars = j - idx
                    break
            else:
                c_exit = float(closes[min(end - 1, n_data - 1)])
                pnl = (c_exit - half_spread) - entry
        else:
            wm = entry
            for j in range(idx + 1, end):
                wm = min(wm, float(lows[j]))
                a = float(atr_arr[min(j, n_data - 1)])
                new_sl = wm + trail_mult * a
                if new_sl < sl:
                    sl = new_sl

                hit_sl = float(highs[j]) >= sl
                hit_tp = float(lows[j]) <= tp
                if hit_sl and hit_tp:
                    exit_r = "sl"
                    pnl = entry - (sl + half_spread)
                    bars = j - idx
                    break
                if hit_sl:
                    exit_r = "sl"
                    pnl = entry - (sl + half_spread)
                    bars = j - idx
                    break
                if hit_tp:
                    exit_r = "tp"
                    pnl = entry - (tp + half_spread)
                    bars = j - idx
                    break
            else:
                c_exit = float(closes[min(end - 1, n_data - 1)])
                pnl = entry - (c_exit + half_spread)

        results.append({
            "exit_reason": exit_r,
            "pnl_pips": round(pnl / pip, 2),
            "bars_held": bars,
        })
    return results


# ---------------------------------------------------------------------------
#  STRATEGY SIGNAL GENERATORS — return arrays of (index, direction, entry, sl, tp)
# ---------------------------------------------------------------------------

def signals_s1_dd(df, pair, pip, spread):
    """Double Doji Break at 20 EMA."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    ema20 = df["ema20"].values; atr = df["atr14"].values
    slope = df["ema20_slope"].values
    n = len(df)

    doji = df["bar_range"].values < atr * 0.4
    near_ema = np.abs(c - ema20) < atr * 1.0
    trending = np.abs(slope) > atr * 0.05
    valid_atr = ~np.isnan(atr) & (atr > 0)

    candidates = np.where(
        valid_atr & near_ema & trending & doji & np.roll(doji, 1) & (np.arange(n) > 22)
    )[0]

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -20
    for i in candidates:
        if i - last < 20:
            continue
        highs_close = abs(h[i] - h[i-1]) < 2 * pip
        lows_close = abs(l[i] - l[i-1]) < 2 * pip
        if not (highs_close or lows_close):
            continue

        bull = slope[i] > 0
        if bull:
            e = max(h[i], h[i-1]) + spread
            s = min(l[i], l[i-1]) - pip
            t = e + 10 * pip
        else:
            e = min(l[i], l[i-1]) - spread
            s = max(h[i], h[i-1]) + pip
            t = e - 10 * pip

        sl_d = abs(e - s)
        if sl_d > 10 * pip or sl_d < 1 * pip:
            continue

        indices.append(i); dirs.append(1 if bull else -1)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s2_fb(df, pair, pip, spread):
    """First Break after spike."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -20

    for i in range(5, n - 12):
        if i - last < 20:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        move = abs(c[i] - c[i-3]) / pip
        if move < 12:
            continue

        all_bull = all(c[i-k] > o[i-k] for k in range(3))
        all_bear = all(c[i-k] < o[i-k] for k in range(3))
        if not (all_bull or all_bear):
            continue

        if (h[i] - l[i]) > atr[i] * 0.5:
            continue

        if all_bull:
            e = h[i] + spread; s = l[i] - pip; t = e + 10 * pip; d = 1
        else:
            e = l[i] - spread; s = h[i] + pip; t = e - 10 * pip; d = -1

        sl_d = abs(e - s)
        if sl_d > 10 * pip or sl_d < 1 * pip:
            continue

        indices.append(i); dirs.append(d)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s3_sb(df, pair, pip, spread):
    """Second Break / M-W pattern."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    ema20 = df["ema20"].values; atr = df["atr14"].values
    slope = df["ema20_slope"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -30

    for i in range(40, n - 12):
        if i - last < 30:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if abs(slope[i]) < atr[i] * 0.05:
            continue

        bull = slope[i] > 0
        w = 20

        if bull:
            rl = l[i-w:i+1]
            idx_sorted = np.argsort(rl)
            l1p, l2p = idx_sorted[0], idx_sorted[1]
            if abs(l1p - l2p) < 3:
                continue
            lv1, lv2 = rl[l1p], rl[l2p]
            if abs(lv1 - lv2) > 3 * pip:
                continue
            mid_seg = h[i-w+min(l1p,l2p):i-w+max(l1p,l2p)+1]
            if len(mid_seg) == 0 or np.max(mid_seg) - max(lv1,lv2) < 3 * pip:
                continue
            if abs(c[i] - ema20[i]) > atr[i] * 1.5:
                continue
            e = c[i] + spread; s = min(lv1,lv2) - pip; t = e + 10 * pip; d = 1
        else:
            rh = h[i-w:i+1]
            idx_sorted = np.argsort(-rh)
            h1p, h2p = idx_sorted[0], idx_sorted[1]
            if abs(h1p - h2p) < 3:
                continue
            hv1, hv2 = rh[h1p], rh[h2p]
            if abs(hv1 - hv2) > 3 * pip:
                continue
            mid_seg = l[i-w+min(h1p,h2p):i-w+max(h1p,h2p)+1]
            if len(mid_seg) == 0 or min(hv1,hv2) - np.min(mid_seg) < 3 * pip:
                continue
            if abs(c[i] - ema20[i]) > atr[i] * 1.5:
                continue
            e = c[i] - spread; s = max(hv1,hv2) + pip; t = e - 10 * pip; d = -1

        sl_d = abs(e - s)
        if sl_d > 12 * pip or sl_d < 2 * pip:
            continue

        indices.append(i); dirs.append(d)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s4_ema(df, pair, pip, spread):
    """EMA Bounce / MA Trend."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    ema9 = df["ema9"].values; ema20 = df["ema20"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -15

    for i in range(15, n - 12):
        if i - last < 15:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        bull = all(ema9[i-k] > ema20[i-k] for k in range(8))
        bear = all(ema9[i-k] < ema20[i-k] for k in range(8))
        if not (bull or bear):
            continue

        if bull:
            if not (l[i] <= ema9[i] + pip and c[i] > ema20[i]):
                continue
            if not (c[i] > o[i] and (c[i]-o[i]) > 0.4*(h[i]-l[i])):
                continue
            e = h[i] + spread; s = ema20[i] - atr[i]; t = e + 2*abs(e-s); d = 1
        else:
            if not (h[i] >= ema9[i] - pip and c[i] < ema20[i]):
                continue
            if not (c[i] < o[i] and (o[i]-c[i]) > 0.4*(h[i]-l[i])):
                continue
            e = l[i] - spread; s = ema20[i] + atr[i]; t = e - 2*abs(e-s); d = -1

        sl_d = abs(e - s)
        if sl_d > 15 * pip or sl_d < 1 * pip:
            continue

        indices.append(i); dirs.append(d)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s5_bb(df, pair, pip, spread):
    """Block Break / Bollinger Squeeze."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values; bb_sq = df["bb_squeeze_pct"].values
    slope = df["ema20_slope"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -30

    for i in range(110, n - 12):
        if i - last < 30:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(bb_sq[i]):
            continue
        if bb_sq[i] > 0.25:
            continue

        blk_h = h[i-7:i+1]; blk_l = l[i-7:i+1]
        bt = np.max(blk_h); bb = np.min(blk_l)
        bh = bt - bb
        if bh > atr[i] * 0.7 or bh < 2 * pip:
            continue

        tt = np.sum(np.abs(blk_h - bt) < 1.5 * pip)
        bt_t = np.sum(np.abs(blk_l - bb) < 1.5 * pip)
        if tt < 2 or bt_t < 2:
            continue

        bull = slope[i] > 0
        if bull:
            e = bt + spread; s = bb - pip; t = e + max(10*pip, bh); d = 1
        else:
            e = bb - spread; s = bt + pip; t = e - max(10*pip, bh); d = -1

        sl_d = abs(e - s)
        if sl_d > 15 * pip or sl_d < 2 * pip:
            continue

        indices.append(i); dirs.append(d)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s6_rb(df, pair, pip, spread):
    """Range Break with squeeze."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -40

    for i in range(70, n - 24):
        if i - last < 40:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        seg_h = h[i-60:i-5]; seg_l = l[i-60:i-5]
        r_top = np.max(seg_h); r_bot = np.min(seg_l)
        rh = r_top - r_bot
        if rh < 8 * pip or rh > 60 * pip:
            continue

        tt = np.sum(np.abs(seg_h - r_top) < 2*pip)
        bt = np.sum(np.abs(seg_l - r_bot) < 2*pip)
        if tt < 3 or bt < 3:
            continue

        sq_h = h[i-5:i+1]; sq_l = l[i-5:i+1]
        if np.max(sq_h) - np.min(sq_l) > rh * 0.4:
            continue

        if c[i] > r_top:
            e = c[i] + spread; s = np.min(sq_l) - pip; t = r_top + rh; d = 1
        elif c[i] < r_bot:
            e = c[i] - spread; s = np.max(sq_h) + pip; t = r_bot - rh; d = -1
        else:
            continue

        sl_d = abs(e - s)
        if sl_d > 20 * pip or sl_d < 2 * pip:
            continue

        indices.append(i); dirs.append(d)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s9_fbr(df, pair, pip, spread):
    """Failed Breakout Reversal."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -30

    for i in range(50, n - 12):
        if i - last < 30:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        prev_h = h[i-40:i-3]; prev_l = l[i-40:i-3]
        if len(prev_h) < 10:
            continue
        sw_hi = np.max(prev_h); sw_lo = np.min(prev_l)

        broke_hi = h[i-1] > sw_hi and h[i-1] - sw_hi < 5*pip
        broke_lo = l[i-1] < sw_lo and sw_lo - l[i-1] < 5*pip

        if not (broke_hi or broke_lo):
            continue

        br = h[i] - l[i]
        if br <= 0:
            continue

        if broke_hi:
            if not (c[i] < sw_hi and c[i] < o[i]):
                continue
            if (o[i]-c[i]) < 0.3 * br:
                continue
            e = c[i] - spread; s = h[i-1] + 2*pip; t = e - 2*abs(e-s); d = -1
        else:
            if not (c[i] > sw_lo and c[i] > o[i]):
                continue
            if (c[i]-o[i]) < 0.3 * br:
                continue
            e = c[i] + spread; s = l[i-1] - 2*pip; t = e + 2*abs(e-s); d = 1

        sl_d = abs(e - s)
        if sl_d > 15 * pip or sl_d < 1 * pip:
            continue

        indices.append(i); dirs.append(d)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s10_wedge(df, pair, pip, spread):
    """Wedge / Three-Push Reversal."""
    h = df["high"].values; l = df["low"].values
    c = df["close"].values; o = df["open"].values
    atr = df["atr14"].values; rsi = df["rsi14"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -40
    w = 25

    for i in range(w + 5, n - 12):
        if i - last < 40:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(rsi[i]):
            continue

        seg_h = h[i-w:i+1]; seg_l = l[i-w:i+1]; seg_rsi = rsi[i-w:i+1]

        swing_highs = []
        swing_lows = []
        for k in range(2, len(seg_h)-2):
            if seg_h[k] > max(seg_h[k-1], seg_h[k-2], seg_h[k+1], seg_h[k+2]):
                swing_highs.append((k, seg_h[k], seg_rsi[k]))
            if seg_l[k] < min(seg_l[k-1], seg_l[k-2], seg_l[k+1], seg_l[k+2]):
                swing_lows.append((k, seg_l[k], seg_rsi[k]))

        found = False
        if len(swing_highs) >= 3:
            sh = swing_highs[-3:]
            if sh[0][1] < sh[1][1] < sh[2][1]:
                p1 = sh[1][1] - sh[0][1]; p2 = sh[2][1] - sh[1][1]
                if 0 < p2 < p1 and sh[2][2] < sh[1][2]:
                    if c[i] < o[i] and (o[i]-c[i]) > 0.3*(h[i]-l[i]):
                        e = c[i] - spread
                        s = h[i-w+sh[2][0]] + 2*pip
                        t = e - 2*abs(e-s)
                        sl_d = abs(e - s)
                        if 2*pip < sl_d < 15*pip:
                            indices.append(i); dirs.append(-1)
                            entries.append(e); sl_arr.append(s); tp_arr.append(t)
                            last = i; found = True

        if not found and len(swing_lows) >= 3:
            sl_pts = swing_lows[-3:]
            if sl_pts[0][1] > sl_pts[1][1] > sl_pts[2][1]:
                p1 = sl_pts[0][1] - sl_pts[1][1]; p2 = sl_pts[1][1] - sl_pts[2][1]
                if 0 < p2 < p1 and sl_pts[2][2] > sl_pts[1][2]:
                    if c[i] > o[i] and (c[i]-o[i]) > 0.3*(h[i]-l[i]):
                        e = c[i] + spread
                        s = l[i-w+sl_pts[2][0]] - 2*pip
                        t = e + 2*abs(e-s)
                        sl_d = abs(e - s)
                        if 2*pip < sl_d < 15*pip:
                            indices.append(i); dirs.append(1)
                            entries.append(e); sl_arr.append(s); tp_arr.append(t)
                            last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s11_rsi(df, pair, pip, spread):
    """RSI/Bollinger Mean Reversion."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    rsi = df["rsi14"].values
    bb_u = df["bb_upper"].values; bb_l = df["bb_lower"].values; bb_m = df["bb_mid"].values
    slope = df["ema20_slope"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -20

    for i in range(30, n - 12):
        if i - last < 20:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(rsi[i]) or np.isnan(bb_u[i]):
            continue

        br = h[i] - l[i]
        if br <= 0:
            continue
        body = abs(c[i] - o[i])
        indecision = body / br < 0.35

        if rsi[i] < 25 and c[i] < bb_l[i] and indecision:
            if abs(slope[i]) > atr[i]*0.2 and slope[i] < 0:
                continue
            e = c[i] + spread; s = l[i] - atr[i]; t = float(bb_m[i])
            if t <= e:
                continue
            d = 1
        elif rsi[i] > 75 and c[i] > bb_u[i] and indecision:
            if abs(slope[i]) > atr[i]*0.2 and slope[i] > 0:
                continue
            e = c[i] - spread; s = h[i] + atr[i]; t = float(bb_m[i])
            if t >= e:
                continue
            d = -1
        else:
            continue

        sl_d = abs(e - s)
        if sl_d > 20*pip or sl_d < 1*pip:
            continue

        indices.append(i); dirs.append(d)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s13_abcd(df, pair, pip, spread):
    """ABCD Fibonacci Pullback."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -30

    for i in range(40, n - 24):
        if i - last < 30:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        look = 25
        sh = h[i-look:i-4]; sl_seg = l[i-look:i-4]
        if len(sh) < 8:
            continue

        a_hi = int(np.argmax(sh)); a_lo = int(np.argmin(sl_seg))

        found = False
        if a_hi < len(sh) - 4:
            b_lo = int(np.argmin(sl_seg[a_hi+2:])) + a_hi + 2
            ab = sh[a_hi] - sl_seg[b_lo]
            if ab > 1.5 * atr[i]:
                ret = c[i] - sl_seg[b_lo]
                pct = ret / ab if ab != 0 else 0
                if 0.38 <= pct <= 0.62 and c[i] > o[i]:
                    e = c[i] + spread; s = sl_seg[b_lo] - 2*pip
                    t = sh[a_hi] + ab*0.618
                    sl_d = abs(e-s); tp_d = abs(e-t)
                    if 2*pip < sl_d < 25*pip and tp_d > sl_d:
                        indices.append(i); dirs.append(1)
                        entries.append(e); sl_arr.append(s); tp_arr.append(t)
                        last = i; found = True

        if not found and a_lo < len(sl_seg) - 4:
            b_hi = int(np.argmax(sh[a_lo+2:])) + a_lo + 2
            ab = sh[b_hi] - sl_seg[a_lo]
            if ab > 1.5 * atr[i]:
                ret = sh[b_hi] - c[i]
                pct = ret / ab if ab != 0 else 0
                if 0.38 <= pct <= 0.62 and c[i] < o[i]:
                    e = c[i] - spread; s = sh[b_hi] + 2*pip
                    t = sl_seg[a_lo] - ab*0.618
                    sl_d = abs(e-s); tp_d = abs(e-t)
                    if 2*pip < sl_d < 25*pip and tp_d > sl_d:
                        indices.append(i); dirs.append(-1)
                        entries.append(e); sl_arr.append(s); tp_arr.append(t)
                        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s14_bkpb(df, pair, pip, spread):
    """Breakout Pullback Continuation."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; atr = df["atr14"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -30

    for i in range(55, n - 24):
        if i - last < 30:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        seg_h = h[i-45:i-5]; seg_l = l[i-45:i-5]
        if len(seg_h) < 20:
            continue
        rt = np.max(seg_h); rb = np.min(seg_l)
        rh = rt - rb
        if rh < 5*pip or rh > 50*pip:
            continue

        tt = np.sum(np.abs(seg_h - rt) < 2*pip)
        bt = np.sum(np.abs(seg_l - rb) < 2*pip)
        if tt < 2 or bt < 2:
            continue

        bb_h = h[i-5:i-1]; bb_l = l[i-5:i-1]; bb_c = c[i-5:i-1]; bb_o = o[i-5:i-1]
        br = bb_h - bb_l
        br[br == 0] = 1e-10

        bull_bo = np.any(bb_c > rt) and np.any((bb_c - bb_o) / br > 0.7)
        bear_bo = np.any(bb_c < rb) and np.any((bb_o - bb_c) / br > 0.7)

        if bull_bo and not bear_bo:
            if l[i] > rt - rh*0.5 and c[i] > rt:
                e = c[i]+spread; s = rt-2*pip; t = rt+rh; d = 1
            else:
                continue
        elif bear_bo and not bull_bo:
            if h[i] < rb + rh*0.5 and c[i] < rb:
                e = c[i]-spread; s = rb+2*pip; t = rb-rh; d = -1
            else:
                continue
        else:
            continue

        sl_d = abs(e-s); tp_d = abs(e-t)
        if sl_d > 20*pip or sl_d < 1*pip or tp_d < sl_d*0.8:
            continue

        indices.append(i); dirs.append(d)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


def signals_s15_vwap(df_m1, pair, pip, spread):
    """VWAP Session Reversion — uses its own M5 resample."""
    m5 = resample_m5(df_m1)
    m5 = add_indicators(m5, pair)
    c = m5["close"].values; h = m5["high"].values; l = m5["low"].values
    vol = m5["volume"].values if "volume" in m5.columns else np.zeros(len(m5))
    atr = m5["atr14"].values
    hours = m5["hour"].values; dates = m5["date"].values; ts = m5["timestamp"].values
    n = len(m5)
    active = set(range(7, 16))

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    hrs_out, ts_out = [], []
    last = -12; cur_date = None; sess_start = 0

    for i in range(20, n - 60):
        if i - last < 12:
            continue
        hr = hours[i]
        if hr not in active:
            cur_date = None; continue
        d = dates[i]
        if d != cur_date:
            sess_start = i; cur_date = d
        if i - sess_start < 10:
            continue
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue

        sess_c = c[sess_start:i+1]
        sess_h = h[sess_start:i+1]; sess_l = l[sess_start:i+1]
        sess_v = vol[sess_start:i+1]
        if sess_v.sum() > 0:
            typ = (sess_h + sess_l + sess_c) / 3.0
            vwap = (typ * sess_v).sum() / sess_v.sum()
        else:
            vwap = sess_c.mean()

        dev = c[i] - vwap
        if abs(dev / a) < 2.5:
            continue

        sl_dist = a * 0.5
        if dev > 0:
            e = c[i] - spread/2; s = e + sl_dist; t = float(vwap); direction = -1
        else:
            e = c[i] + spread/2; s = e - sl_dist; t = float(vwap); direction = 1

        if direction == 1 and t <= e:
            continue
        if direction == -1 and t >= e:
            continue

        indices.append(i); dirs.append(direction)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        hrs_out.append(hr); ts_out.append(ts[i])
        last = i

    return indices, dirs, entries, sl_arr, tp_arr, h, l, c, hrs_out, ts_out, n


def signals_s16_hl2(df, pair, pip, spread):
    """High 2 / Low 2 Second Entry."""
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    atr = df["atr14"].values; ema20 = df["ema20"].values
    slope = df["ema20_slope"].values
    n = len(df)

    indices, dirs, entries, sl_arr, tp_arr = [], [], [], [], []
    last = -20

    for i in range(20, n - 12):
        if i - last < 20:
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if abs(slope[i]) < atr[i] * 0.05:
            continue

        bull = slope[i] > 0 and c[i] > ema20[i]
        bear = slope[i] < 0 and c[i] < ema20[i]
        if not (bull or bear):
            continue

        if bull:
            h1 = False; h2_ok = False
            for k in range(max(0, i-12), i+1):
                if k < 1:
                    continue
                if h[k] > h[k-1]:
                    if h1:
                        if k == i:
                            h2_ok = True
                        break
                    else:
                        h1 = True
                        went_down = any(l[m] < l[m-1] for m in range(k+1, min(k+5, i+1)) if m >= 1)
                        if not went_down:
                            h1 = False

            if not h2_ok:
                continue
            pb_low = np.min(l[max(0,i-12):i+1])
            e = h[i]+spread; s = pb_low-pip; t = e+2*abs(e-s); d = 1
        else:
            l1 = False; l2_ok = False
            for k in range(max(0, i-12), i+1):
                if k < 1:
                    continue
                if l[k] < l[k-1]:
                    if l1:
                        if k == i:
                            l2_ok = True
                        break
                    else:
                        l1 = True
                        went_up = any(h[m] > h[m-1] for m in range(k+1, min(k+5, i+1)) if m >= 1)
                        if not went_up:
                            l1 = False

            if not l2_ok:
                continue
            pb_hi = np.max(h[max(0,i-12):i+1])
            e = l[i]-spread; s = pb_hi+pip; t = e-2*abs(e-s); d = -1

        sl_d = abs(e-s)
        if sl_d > 15*pip or sl_d < 1*pip:
            continue

        indices.append(i); dirs.append(d)
        entries.append(e); sl_arr.append(s); tp_arr.append(t)
        last = i

    return indices, dirs, entries, sl_arr, tp_arr


# ---------------------------------------------------------------------------
#  DISPATCH — each returns list of trade dicts
# ---------------------------------------------------------------------------

def run_strategy(name, sig_func, df, pair, pip, spread, max_bars=12):
    """Run a signal function and simulate all its trades."""
    result = sig_func(df, pair, pip, spread)
    hsp = spread_half_price(pair, SPREAD_PIPS.get(pair, 1.5))

    # S15 returns extra arrays (it builds its own M5)
    if name == "S15_VWAP":
        idx, dirs, ent, sls, tps, h_arr, l_arr, c_arr, hrs, tss, n_data = result
        if not idx:
            return []
        sim = simulate_trades_vec(
            h_arr, l_arr, c_arr, idx, dirs, ent, sls, tps, 60, pip, half_spread=hsp
        )
        trades = []
        for k in range(len(idx)):
            trades.append({
                "strategy": name, "pair": pair,
                "direction": "long" if dirs[k] == 1 else "short",
                "hour": int(hrs[k]),
                "pnl_pips": float(sim[k]["pnl_pips"]),
                "exit_reason": str(sim[k]["exit_reason"]),
                "bars_held": int(sim[k]["bars_held"]),
                "sl_pips": round(abs(ent[k]-sls[k])/pip, 1),
                "tp_pips": round(abs(ent[k]-tps[k])/pip, 1),
                "timestamp": str(tss[k]),
            })
        return trades

    idx, dirs, ent, sls, tps = result
    if not idx:
        return []

    h_arr = df["high"].values; l_arr = df["low"].values; c_arr = df["close"].values
    hours = df["hour"].values; ts = df["timestamp"].values
    n_data = len(df)

    sim = simulate_trades_vec(
        h_arr, l_arr, c_arr, idx, dirs, ent, sls, tps, max_bars, pip, half_spread=hsp
    )
    trades = []
    for k in range(len(idx)):
        trades.append({
            "strategy": name, "pair": pair,
            "direction": "long" if dirs[k] == 1 else "short",
            "hour": int(hours[idx[k]]),
            "pnl_pips": float(sim[k]["pnl_pips"]),
            "exit_reason": str(sim[k]["exit_reason"]),
            "bars_held": int(sim[k]["bars_held"]),
            "sl_pips": round(abs(ent[k]-sls[k])/pip, 1),
            "tp_pips": round(abs(ent[k]-tps[k])/pip, 1),
            "timestamp": str(ts[idx[k]]),
        })
    return trades


STRATEGIES = [
    ("S1_DD",    signals_s1_dd,   12),
    ("S2_FB",    signals_s2_fb,   12),
    ("S3_SB",    signals_s3_sb,   12),
    ("S4_EMA",   signals_s4_ema,  12),
    ("S5_BB",    signals_s5_bb,   12),
    ("S6_RB",    signals_s6_rb,   24),
    ("S9_FBR",   signals_s9_fbr,  12),
    ("S10_WDG",  signals_s10_wedge, 12),
    ("S11_RSI",  signals_s11_rsi, 12),
    ("S13_ABCD", signals_s13_abcd, 24),
    ("S14_BKPB", signals_s14_bkpb, 24),
    ("S15_VWAP", signals_s15_vwap, 60),
    ("S16_HL2",  signals_s16_hl2, 12),
]


# ---------------------------------------------------------------------------
#  REPORTING
# ---------------------------------------------------------------------------

def print_report(all_trades):
    if not all_trades:
        print("  No trades.")
        return

    print(f"\n{'='*90}")
    print(f"  STRATEGY BACKTEST — {len(all_trades):,} total trades")
    print(f"{'='*90}")

    total = sum(t["pnl_pips"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["pnl_pips"] > 0)
    wr = wins / len(all_trades) * 100
    gp = sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] > 0)
    gl = abs(sum(t["pnl_pips"] for t in all_trades if t["pnl_pips"] <= 0))
    pf = gp / gl if gl else 999

    print(f"\n  Overall: {wins:,}W / {len(all_trades)-wins:,}L | WR {wr:.1f}% | PnL {total:+,.0f} pips | PF {pf:.2f}")

    print(f"\n  {'Strategy':12s} {'Trades':>8s} {'WR':>7s} {'PnL':>12s} {'Avg':>8s} {'PF':>7s} {'Verdict':>10s}")
    print(f"  {'-'*70}")
    strats = defaultdict(list)
    for t in all_trades:
        strats[t["strategy"]].append(t)
    for s in sorted(strats.keys()):
        st = strats[s]
        w = sum(1 for t in st if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in st)
        g = sum(t["pnl_pips"] for t in st if t["pnl_pips"] > 0)
        lo = abs(sum(t["pnl_pips"] for t in st if t["pnl_pips"] <= 0))
        pf_v = g/lo if lo else 999
        avg = p/len(st)
        v = "PROFIT" if pf_v > 1.05 and avg > 0.1 else ("BREAK-EVEN" if pf_v > 0.95 else "LOSS")
        print(f"  {s:12s} {len(st):8,d} {w/len(st)*100:6.1f}% {p:+12,.0f} {avg:+8.2f} {pf_v:7.2f} {v:>10s}")

    print(f"\n  TOP 25 COMBOS (strat + pair + dir):")
    print(f"  {'Combo':36s} {'N':>6s} {'WR':>7s} {'PnL':>10s} {'Avg':>7s} {'PF':>6s}")
    combos = defaultdict(list)
    for t in all_trades:
        combos[f"{t['strategy']} {t['pair']} {t['direction']}"].append(t)
    ranked = []
    for k, ct in combos.items():
        if len(ct) < 30:
            continue
        w = sum(1 for t in ct if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in ct)
        g = sum(t["pnl_pips"] for t in ct if t["pnl_pips"] > 0)
        lo = abs(sum(t["pnl_pips"] for t in ct if t["pnl_pips"] <= 0))
        ranked.append((k, len(ct), w/len(ct)*100, p, p/len(ct), g/lo if lo else 999))
    for k, n, w, p, a, pf_v in sorted(ranked, key=lambda x: -x[3])[:25]:
        print(f"  {k:36s} {n:6d} {w:6.1f}% {p:+10,.0f} {a:+7.2f} {pf_v:6.2f}")

    print(f"\n  WORST 15 COMBOS:")
    for k, n, w, p, a, pf_v in sorted(ranked, key=lambda x: x[3])[:15]:
        print(f"  {k:36s} {n:6d} {w:6.1f}% {p:+10,.0f} {a:+7.2f} {pf_v:6.2f}")

    print(f"\n  BY HOUR:")
    print(f"  {'Hr':>4s} {'N':>7s} {'WR':>7s} {'PnL':>10s} {'Avg':>7s}")
    hs = defaultdict(list)
    for t in all_trades:
        hs[t["hour"]].append(t)
    for hr in range(24):
        ht = hs.get(hr, [])
        if not ht:
            continue
        w = sum(1 for t in ht if t["pnl_pips"] > 0)
        p = sum(t["pnl_pips"] for t in ht)
        print(f"  {hr:4d} {len(ht):7,d} {w/len(ht)*100:6.1f}% {p:+10,.0f} {p/len(ht):+7.2f}")


def save_results(all_trades):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combo = defaultdict(lambda: {"wins":0,"losses":0,"pnl":0,"trades":0})
    for t in all_trades:
        key = f"{t['strategy']}_{t['pair']}_{t['direction']}"
        s = combo[key]; s["trades"]+=1; s["pnl"]+=t["pnl_pips"]
        if t["pnl_pips"]>0: s["wins"]+=1
        else: s["losses"]+=1
    for s in combo.values():
        s["win_rate"]=round(s["wins"]/s["trades"]*100,1)
        s["avg_pips"]=round(s["pnl"]/s["trades"],2)
        s["pnl"]=round(s["pnl"],1)
    with open(OUTPUT_DIR/"strategy_combo_stats.json","w") as f:
        json.dump(dict(combo),f,indent=2,sort_keys=True)

    lessons = []
    for key, s in sorted(combo.items(), key=lambda x:-x[1]["pnl"]):
        if s["trades"]<50: continue
        if s["win_rate"]>=53 and s["avg_pips"]>0.2:
            lessons.append({"theme":f"STRAT-EDGE: {key} — {s['win_rate']}% WR, {s['trades']} trades, avg {s['avg_pips']:+.2f}p",
                            "confidence":min(0.95,0.5+s["win_rate"]/200),"source":"backtest_strategies","pinned":True})
        elif s["win_rate"]<45 and s["avg_pips"]<-0.5:
            lessons.append({"theme":f"STRAT-AVOID: {key} — {s['win_rate']}% WR, {s['trades']} trades, avg {s['avg_pips']:+.2f}p",
                            "confidence":min(0.95,0.5+(100-s["win_rate"])/200),"source":"backtest_strategies","pinned":True})
    with open(OUTPUT_DIR/"strategy_lessons.jsonl","w") as f:
        for le in lessons:
            f.write(json.dumps(le)+"\n")

    strat_sum = {}
    strats = defaultdict(list)
    for t in all_trades:
        strats[t["strategy"]].append(t)
    for s, trades in strats.items():
        w=sum(1 for t in trades if t["pnl_pips"]>0)
        p=sum(t["pnl_pips"] for t in trades)
        g=sum(t["pnl_pips"] for t in trades if t["pnl_pips"]>0)
        lo=abs(sum(t["pnl_pips"] for t in trades if t["pnl_pips"]<=0))
        strat_sum[s]={"trades":len(trades),"wins":w,"win_rate":round(w/len(trades)*100,1),
                      "total_pips":round(p,1),"avg_pips":round(p/len(trades),2),
                      "profit_factor":round(g/lo,3) if lo else 999}
    with open(OUTPUT_DIR/"strategy_summary.json","w") as f:
        json.dump(strat_sum,f,indent=2,sort_keys=True)

    print(f"\n  Saved: strategy_combo_stats.json, strategy_summary.json, strategy_lessons.jsonl ({len(lessons)} lessons)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=0)
    args = parser.parse_args()

    print("="*90)
    print(f"  13-STRATEGY BACKTEST on M5" + (f" (last {args.years}yr)" if args.years else " (full history)"))
    print(f"  Pairs: {', '.join(PAIR_MAP.values())}")
    print("="*90)

    all_trades = []
    t0 = time_mod.time()

    for fb, pair in PAIR_MAP.items():
        fn = f"{fb}_1min.csv"
        print(f"\n  {pair} loading...", end=" ", flush=True)
        df_m1 = load_pair(fn, pair, args.years)
        if df_m1.empty:
            print("SKIP"); continue
        print(f"{len(df_m1):,} M1 bars", end="", flush=True)

        print(" → M5...", end="", flush=True)
        df = resample_m5(df_m1)
        print(f" {len(df):,} M5 bars", end="", flush=True)

        print(" | ind...", end="", flush=True)
        df = add_indicators(df, pair)
        pip = 0.01 if "JPY" in pair else 0.0001
        spread = SPREAD_PIPS.get(pair, 1.5) * pip

        for sname, sfunc, mb in STRATEGIES:
            print(f" {sname}", end="", flush=True)
            try:
                if sname == "S15_VWAP":
                    trades = run_strategy(sname, sfunc, df_m1, pair, pip, spread, mb)
                else:
                    trades = run_strategy(sname, sfunc, df, pair, pip, spread, mb)
                print(f":{len(trades)}", end="", flush=True)
                all_trades.extend(trades)
            except Exception as e:
                print(f":ERR({e})", end="", flush=True)

        elapsed = time_mod.time() - t0
        print(f" | {elapsed:.0f}s")

    total_time = time_mod.time() - t0
    print(f"\n  Total: {len(all_trades):,} trades in {total_time:.0f}s ({total_time/60:.1f}min)")

    print_report(all_trades)
    save_results(all_trades)
    print("\n  Done.")


if __name__ == "__main__":
    main()
