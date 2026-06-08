from __future__ import annotations

"""Extended technical indicators for the forex backtesting system.

Adds 12 indicator groups to a DataFrame that already contains OHLCV, timestamp,
and the basic indicators produced by ``backtest_strategies.add_indicators``
(EMA9/20/40, ATR14, Bollinger, RSI14, MACD, Stochastic, etc.).

All computations use only **numpy** and **pandas** -- no external indicator libs.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average via pandas (adjust=False)."""
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=1).max().values


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=1).min().values


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=1).sum().values


# ---------------------------------------------------------------------------
# 1. ADX / +DI / -DI  (14-period)
# ---------------------------------------------------------------------------

def _adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14):
    n = len(h)
    prev_h = np.empty(n); prev_h[0] = h[0]; prev_h[1:] = h[:-1]
    prev_l = np.empty(n); prev_l[0] = l[0]; prev_l[1:] = l[:-1]
    prev_c = np.empty(n); prev_c[0] = c[0]; prev_c[1:] = c[:-1]

    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    up_move = h - prev_h
    down_move = prev_l - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s = _ema(tr, period)
    plus_dm_s = _ema(plus_dm, period)
    minus_dm_s = _ema(minus_dm, period)

    plus_di = np.where(atr_s > 0, 100.0 * plus_dm_s / atr_s, 0.0)
    minus_di = np.where(atr_s > 0, 100.0 * minus_dm_s / atr_s, 0.0)
    di_sum = plus_di + minus_di
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)
    adx = _ema(dx, period)
    return adx, plus_di, minus_di


# ---------------------------------------------------------------------------
# 2. Ichimoku (9/26/52)
# ---------------------------------------------------------------------------

def _ichimoku(h: np.ndarray, l: np.ndarray, c: np.ndarray):
    tenkan = (_rolling_max(h, 9) + _rolling_min(l, 9)) / 2.0
    kijun = (_rolling_max(h, 26) + _rolling_min(l, 26)) / 2.0

    senkou_a_raw = (tenkan + kijun) / 2.0
    senkou_a = np.empty_like(senkou_a_raw)
    senkou_a[:] = np.nan
    senkou_a[26:] = senkou_a_raw[:-26]

    senkou_b_raw = (_rolling_max(h, 52) + _rolling_min(l, 52)) / 2.0
    senkou_b = np.empty_like(senkou_b_raw)
    senkou_b[:] = np.nan
    senkou_b[26:] = senkou_b_raw[:-26]

    chikou = np.empty_like(c)
    chikou[:] = np.nan
    if len(c) > 26:
        chikou[:-26] = c[26:]

    return tenkan, kijun, senkou_a, senkou_b, chikou


# ---------------------------------------------------------------------------
# 3. Keltner Channels (20 EMA mid, 1.5x / 2.0x ATR)
# ---------------------------------------------------------------------------

def _keltner(ema20: np.ndarray, atr14: np.ndarray):
    return (
        ema20 + 1.5 * atr14,
        ema20 - 1.5 * atr14,
        ema20 + 2.0 * atr14,
        ema20 - 2.0 * atr14,
    )


# ---------------------------------------------------------------------------
# 4. Donchian Channels (20-bar and 10-bar)
# ---------------------------------------------------------------------------

def _donchian(h: np.ndarray, l: np.ndarray):
    return (
        _rolling_max(h, 20),
        _rolling_min(l, 20),
        _rolling_max(h, 10),
        _rolling_min(l, 10),
    )


# ---------------------------------------------------------------------------
# 5. Parabolic SAR
# ---------------------------------------------------------------------------

def _parabolic_sar(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> np.ndarray:
    n = len(h)
    psar = np.empty(n)
    psar[0] = l[0]

    bull = True
    af = af_start
    ep = h[0]
    sar = l[0]

    for i in range(1, n):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)

        if bull:
            sar = min(sar, l[i - 1])
            if i >= 2:
                sar = min(sar, l[i - 2])
            if l[i] < sar:
                bull = False
                sar = ep
                ep = l[i]
                af = af_start
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            sar = max(sar, h[i - 1])
            if i >= 2:
                sar = max(sar, h[i - 2])
            if h[i] > sar:
                bull = True
                sar = ep
                ep = h[i]
                af = af_start
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)

        psar[i] = sar

    return psar


# ---------------------------------------------------------------------------
# 6. Supertrend (period=10, multiplier=3.0)
# ---------------------------------------------------------------------------

def _supertrend(
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
):
    n = len(h)
    prev_c = np.empty(n); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values

    hl2 = (h + l) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = np.empty(n)
    lower_band = np.empty(n)
    supertrend = np.empty(n)
    direction = np.ones(n)

    upper_band[0] = upper_basic[0]
    lower_band[0] = lower_basic[0]
    supertrend[0] = upper_basic[0]
    direction[0] = -1.0

    for i in range(1, n):
        if lower_basic[i] > lower_band[i - 1] or c[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i - 1]

        if upper_basic[i] < upper_band[i - 1] or c[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i - 1]

        if direction[i - 1] == 1.0:
            if c[i] < lower_band[i]:
                direction[i] = -1.0
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1.0
                supertrend[i] = lower_band[i]
        else:
            if c[i] > upper_band[i]:
                direction[i] = 1.0
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1.0
                supertrend[i] = upper_band[i]

    return supertrend, direction


# ---------------------------------------------------------------------------
# 7. Williams %R (14-period)
# ---------------------------------------------------------------------------

def _williams_r(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14):
    high14 = _rolling_max(h, period)
    low14 = _rolling_min(l, period)
    denom = high14 - low14
    return np.where(denom > 1e-15, -100.0 * (high14 - c) / denom, -50.0)


# ---------------------------------------------------------------------------
# 8. CMF (Chaikin Money Flow, 20-period)
# ---------------------------------------------------------------------------

def _cmf(h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray, period: int = 20):
    denom = h - l
    mfm = np.where(denom > 1e-15, ((c - l) - (h - c)) / denom, 0.0)
    mfv = mfm * v
    sum_mfv = _rolling_sum(mfv, period)
    sum_vol = _rolling_sum(v, period)
    return np.where(sum_vol > 0, sum_mfv / sum_vol, 0.0)


# ---------------------------------------------------------------------------
# 9. Daily pivot levels (Floor + Camarilla) from PREVIOUS day
# ---------------------------------------------------------------------------

def _daily_pivots(df: pd.DataFrame):
    dates = df["timestamp"].dt.date.values
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    n = len(df)

    pp = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    cam_r3 = np.full(n, np.nan)
    cam_r4 = np.full(n, np.nan)
    cam_s3 = np.full(n, np.nan)
    cam_s4 = np.full(n, np.nan)

    prev_day_h = np.nan
    prev_day_l = np.nan
    prev_day_c = np.nan
    cur_day_h = h[0]
    cur_day_l = l[0]
    cur_date = dates[0]

    for i in range(n):
        d = dates[i]
        if d != cur_date:
            prev_day_h = cur_day_h
            prev_day_l = cur_day_l
            prev_day_c = c[i - 1]
            cur_day_h = h[i]
            cur_day_l = l[i]
            cur_date = d
        else:
            if h[i] > cur_day_h:
                cur_day_h = h[i]
            if l[i] < cur_day_l:
                cur_day_l = l[i]

        if not np.isnan(prev_day_h):
            p = (prev_day_h + prev_day_l + prev_day_c) / 3.0
            pp[i] = p
            r1[i] = 2.0 * p - prev_day_l
            s1[i] = 2.0 * p - prev_day_h
            hl_range = prev_day_h - prev_day_l
            cam_r3[i] = prev_day_c + 1.1 * hl_range / 4.0
            cam_r4[i] = prev_day_c + 1.1 * hl_range / 2.0
            cam_s3[i] = prev_day_c - 1.1 * hl_range / 4.0
            cam_s4[i] = prev_day_c - 1.1 * hl_range / 2.0

    return pp, r1, s1, cam_r3, cam_r4, cam_s3, cam_s4


# ---------------------------------------------------------------------------
# 10. Heikin-Ashi OHLC
# ---------------------------------------------------------------------------

def _heikin_ashi(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
    n = len(c)
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(n)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(l, np.minimum(ha_open, ha_close))
    return ha_open, ha_high, ha_low, ha_close


# ---------------------------------------------------------------------------
# 11. Z-score of close vs 30-bar SMA
# ---------------------------------------------------------------------------

def _zscore(c: np.ndarray, window: int = 30):
    s = pd.Series(c)
    sma = s.rolling(window, min_periods=1).mean().values
    std = s.rolling(window, min_periods=1).std(ddof=0).values
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(std > 1e-15, (c - sma) / std, 0.0)


# ---------------------------------------------------------------------------
# 12. Hurst exponent (rolling 100-bar R/S method)
# ---------------------------------------------------------------------------

def _hurst(c: np.ndarray, window: int = 100) -> np.ndarray:
    """Simplified rolling Hurst exponent via rescaled-range (R/S) analysis.

    For each window the log-returns are split into two halves.  R/S is computed
    for the full window and each half, then H = log(RS_full / mean(RS_halves))
    / log(2).  This gives a rough but fast estimate suitable for bar-by-bar use.
    """
    n = len(c)
    hurst = np.full(n, 0.5)
    log_ret = np.empty(n)
    log_ret[0] = 0.0
    log_ret[1:] = np.log(np.where(c[:-1] > 0, c[1:] / c[:-1], 1.0))

    half = window // 2

    def _rs(arr: np.ndarray) -> float:
        m = arr.mean()
        dev = arr - m
        cumdev = np.cumsum(dev)
        r = cumdev.max() - cumdev.min()
        s = arr.std(ddof=0)
        if s < 1e-15:
            return 0.0
        return r / s

    for i in range(window, n):
        seg = log_ret[i - window + 1 : i + 1]
        rs_full = _rs(seg)
        if rs_full <= 0:
            hurst[i] = 0.5
            continue
        rs_h1 = _rs(seg[:half])
        rs_h2 = _rs(seg[half:])
        rs_avg = (rs_h1 + rs_h2) / 2.0
        if rs_avg <= 0:
            hurst[i] = 0.5
            continue
        h_val = np.log(rs_full / rs_avg) / np.log(2.0)
        hurst[i] = np.clip(h_val, 0.0, 1.0)

    return hurst


# ===================================================================
# PUBLIC API
# ===================================================================

def add_indicators_extended(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Add extended technical indicators to *df* in-place and return it.

    Parameters
    ----------
    df : pd.DataFrame
        Must already contain OHLCV columns (``open``, ``high``, ``low``,
        ``close``, ``volume``), ``timestamp``, and the basic indicators from
        ``backtest_strategies.add_indicators`` (at minimum ``atr14`` and
        ``ema20``).
    pair : str
        Currency pair name, e.g. ``"EUR_USD"``. Currently unused but
        reserved for pair-specific adjustments.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with ~35 new columns appended.
    """
    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    ema20 = df["ema20"].values.astype(np.float64)
    atr14 = df["atr14"].values.astype(np.float64)

    # 1. ADX
    adx, plus_di, minus_di = _adx(h, l, c)
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # 2. Ichimoku
    tenkan, kijun, senkou_a, senkou_b, chikou = _ichimoku(h, l, c)
    df["ichi_tenkan"] = tenkan
    df["ichi_kijun"] = kijun
    df["ichi_senkou_a"] = senkou_a
    df["ichi_senkou_b"] = senkou_b
    df["ichi_chikou"] = chikou

    # 3. Keltner
    ku15, kl15, ku20, kl20 = _keltner(ema20, atr14)
    df["kelt_upper_15"] = ku15
    df["kelt_lower_15"] = kl15
    df["kelt_upper_20"] = ku20
    df["kelt_lower_20"] = kl20

    # 4. Donchian
    dh20, dl20, dh10, dl10 = _donchian(h, l)
    df["donch_high_20"] = dh20
    df["donch_low_20"] = dl20
    df["donch_high_10"] = dh10
    df["donch_low_10"] = dl10

    # 5. Parabolic SAR
    df["psar"] = _parabolic_sar(h, l, c)

    # 6. Supertrend
    st_val, st_dir = _supertrend(h, l, c)
    df["supertrend"] = st_val
    df["supertrend_dir"] = st_dir

    # 7. Williams %R
    df["williams_r"] = _williams_r(h, l, c)

    # 8. CMF
    df["cmf"] = _cmf(h, l, c, v)

    # 9. Daily pivots
    pp, r1, s1, cr3, cr4, cs3, cs4 = _daily_pivots(df)
    df["pivot_pp"] = pp
    df["pivot_r1"] = r1
    df["pivot_s1"] = s1
    df["cam_r3"] = cr3
    df["cam_r4"] = cr4
    df["cam_s3"] = cs3
    df["cam_s4"] = cs4

    # 10. Heikin-Ashi
    ha_o, ha_h, ha_l, ha_c = _heikin_ashi(o, h, l, c)
    df["ha_open"] = ha_o
    df["ha_high"] = ha_h
    df["ha_low"] = ha_l
    df["ha_close"] = ha_c

    # 11. Z-score
    df["zscore_30"] = _zscore(c)

    # 12. Hurst exponent
    df["hurst"] = _hurst(c)

    # Fill remaining NaNs with sensible defaults
    fill_map = {
        "adx": 25.0, "plus_di": 25.0, "minus_di": 25.0,
        "ichi_tenkan": c[0], "ichi_kijun": c[0],
        "ichi_senkou_a": c[0], "ichi_senkou_b": c[0], "ichi_chikou": c[0],
        "pivot_pp": c[0], "pivot_r1": c[0], "pivot_s1": c[0],
        "cam_r3": c[0], "cam_r4": c[0], "cam_s3": c[0], "cam_s4": c[0],
        "williams_r": -50.0,
        "cmf": 0.0,
        "zscore_30": 0.0,
        "hurst": 0.5,
    }
    for col, default in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    return df
