from __future__ import annotations

"""Modular signal filters for forex backtesting.

Each filter takes a DataFrame and a bar index (plus optional params) and returns
True to *keep* the signal or False to *skip* it.  NaN indicator values default
to True (pass) so missing data never silently kills valid signals.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual filters
# ---------------------------------------------------------------------------

def filter_adx(df: pd.DataFrame, idx: int, threshold: float = 25) -> bool:
    """Keep signal when ADX exceeds *threshold* (trending market)."""
    val = df["adx"].iat[idx]
    if np.isnan(val):
        return True
    return val > threshold


def filter_vol_regime(
    df: pd.DataFrame,
    idx: int,
    lo_pct: float = 20,
    hi_pct: float = 80,
) -> bool:
    """Keep signal when current ATR-14 sits between *lo_pct* and *hi_pct*
    percentile of the trailing 100-bar ATR window (not too quiet, not too wild).
    """
    if "atr14" not in df.columns:
        return True
    atr = df["atr14"].iat[idx]
    if np.isnan(atr):
        return True

    window = 100
    start = max(0, idx - window + 1)
    window_vals = df["atr14"].iloc[start : idx + 1].dropna()
    if len(window_vals) < 2:
        return True

    lo = np.percentile(window_vals, lo_pct)
    hi = np.percentile(window_vals, hi_pct)
    return lo <= atr <= hi


def filter_dow(
    df: pd.DataFrame,
    idx: int,
    skip_days: tuple[int, ...] = (0, 4),
) -> bool:
    """Keep signal when the bar's weekday is NOT in *skip_days*.

    Monday = 0 … Sunday = 6.
    """
    ts = df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else df["time"].iat[idx]
    try:
        dow = ts.weekday()
    except AttributeError:
        return True
    return dow not in skip_days


def filter_mtf_trend(
    df: pd.DataFrame,
    idx: int,
    direction: int = 1,
) -> bool:
    """Keep signal when the EMA-20 slope agrees with *direction*.

    direction  1 → long  (needs positive slope)
    direction -1 → short (needs negative slope)
    """
    if "ema20_slope" not in df.columns:
        return True
    slope = df["ema20_slope"].iat[idx]
    if np.isnan(slope):
        return True
    if direction == 1:
        return slope > 0
    return slope < 0


_SESSION_HOURS: dict[str, set[int]] = {
    "london":    set(range(7, 17)),
    "ny":        set(range(12, 22)),
    "overlap":   set(range(12, 17)),
    "london_ny": set(range(7, 22)),
}


def filter_session(
    df: pd.DataFrame,
    idx: int,
    sessions: str = "london_ny",
) -> bool:
    """Keep signal when the bar falls inside the requested session window(s).

    Supported *sessions*: ``"london"``, ``"ny"``, ``"overlap"``, ``"london_ny"``
    (union of London + NY).
    """
    hours = _SESSION_HOURS.get(sessions)
    if hours is None:
        return True

    if "hour" in df.columns:
        hour = int(df["hour"].iat[idx])
    elif isinstance(df.index, pd.DatetimeIndex):
        hour = df.index[idx].hour
    elif "timestamp" in df.columns:
        try:
            hour = pd.Timestamp(df["timestamp"].iat[idx]).hour
        except Exception:
            return True
    else:
        return True
    return hour in hours


def filter_spread(
    df: pd.DataFrame,
    idx: int,
    pair: str = "",
    max_mult: float = 2.0,
) -> bool:
    """Keep signal when the bar range is large enough relative to a minimum
    threshold (proxy: if price barely moved, spread would eat the profit).

    Uses ``high - low`` of the current bar.  A bar whose range is less than
    ``1 / max_mult`` of the rolling-20 median range is considered too thin.
    """
    if "high" not in df.columns or "low" not in df.columns:
        return True
    bar_range = df["high"].iat[idx] - df["low"].iat[idx]
    if np.isnan(bar_range) or bar_range <= 0:
        return True

    window = 20
    start = max(0, idx - window + 1)
    ranges = (df["high"].iloc[start : idx + 1] - df["low"].iloc[start : idx + 1]).dropna()
    if len(ranges) < 2:
        return True

    med = ranges.median()
    if med <= 0:
        return True
    return bar_range >= med / max_mult


# ---------------------------------------------------------------------------
# Filter registry & presets
# ---------------------------------------------------------------------------

_FILTER_FUNCS: dict[str, callable] = {
    "adx":        filter_adx,
    "vol_regime": filter_vol_regime,
    "dow":        filter_dow,
    "mtf_trend":  filter_mtf_trend,
    "session":    filter_session,
    "spread":     filter_spread,
}

FILTER_PRESETS: dict[str, list[tuple[str, dict]]] = {
    "none":    [],
    "adx":     [("adx", {"threshold": 25})],
    "vol":     [("vol_regime", {"lo_pct": 20, "hi_pct": 80})],
    "mtf":     [("mtf_trend", {})],
    "session": [("session", {"sessions": "london_ny"})],
    "adx_mtf": [("adx", {"threshold": 25}), ("mtf_trend", {})],
    "adx_vol": [("adx", {"threshold": 25}), ("vol_regime", {})],
    "full":    [("adx", {}), ("vol_regime", {}), ("mtf_trend", {}), ("session", {})],
}


# ---------------------------------------------------------------------------
# Batch applier
# ---------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    idx_list: list[int],
    dir_list: list[int],
    ent_list: list[float],
    sl_list: list[float],
    tp_list: list[float],
    preset: str = "none",
) -> tuple[list[int], list[int], list[float], list[float], list[float]]:
    """Apply a filter preset to a batch of signals.

    Returns filtered copies of (*idx_list*, *dir_list*, *ent_list*,
    *sl_list*, *tp_list*) containing only signals that pass every filter
    in the preset.
    """
    spec = FILTER_PRESETS.get(preset, [])
    if not spec:
        return idx_list, dir_list, ent_list, sl_list, tp_list

    keep_idx: list[int] = []
    keep_dir: list[int] = []
    keep_ent: list[float] = []
    keep_sl:  list[float] = []
    keep_tp:  list[float] = []

    for i, idx in enumerate(idx_list):
        direction = dir_list[i]
        passed = True
        for name, kwargs in spec:
            fn = _FILTER_FUNCS.get(name)
            if fn is None:
                continue
            kw = dict(kwargs)
            if name == "mtf_trend":
                kw.setdefault("direction", direction)
            if not fn(df, idx, **kw):
                passed = False
                break
        if passed:
            keep_idx.append(idx)
            keep_dir.append(direction)
            keep_ent.append(ent_list[i])
            keep_sl.append(sl_list[i])
            keep_tp.append(tp_list[i])

    return keep_idx, keep_dir, keep_ent, keep_sl, keep_tp
