"""Vectorized M1 bar features for LightGBM (training + live inference).

Must match training pipeline: column order in FEATURE_COLUMNS is the model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# OANDA-style pair keys (underscore)
# Realistic retail spread assumptions for label + feature alignment (V2).
SPREAD_PIPS_DEFAULT = {
    "EUR_USD": 1.5,
    "GBP_USD": 2.0,
    "USD_JPY": 1.5,
    "USD_CAD": 2.0,
    "AUD_USD": 2.0,
    "NZD_USD": 2.0,
    # Legacy pairs (not in V2 training list) — kept for live inference fallbacks.
    "USD_CHF": 2.0,
    "EUR_GBP": 2.0,
}


def pip_for_pair(pair: str) -> float:
    return 0.01 if "JPY" in pair.upper() else 0.0001


def spread_half_price(pair: str, spread_pips: float | None = None) -> float:
    pip = pip_for_pair(pair)
    sp = spread_pips if spread_pips is not None else SPREAD_PIPS_DEFAULT.get(
        pair.upper().replace("/", "_"), 0.6
    )
    return (sp * pip) / 2.0


def _ema(series: pd.Series, span: int) -> np.ndarray:
    return series.ewm(span=span, adjust=False).mean().values.astype(np.float64)


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    s = pd.Series(close)
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).values.astype(np.float64)
    return rsi


def _macd(close: np.ndarray):
    s = pd.Series(close)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return (
        line.values.astype(np.float64),
        signal.values.astype(np.float64),
        hist.values.astype(np.float64),
    )


def _stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = pd.Series(low).rolling(k_period).min()
    highest_high = pd.Series(high).rolling(k_period).max()
    rng = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (pd.Series(close) - lowest_low) / rng
    k = k.fillna(50).values.astype(np.float64)
    d = pd.Series(k).rolling(d_period).mean().values.astype(np.float64)
    return k, d


def _session_bucket(hour: int) -> int:
    """0=Asian, 1=London, 2=NY, 3=London/NY overlap (UTC)."""
    london = 7 <= hour < 16
    ny = 13 <= hour < 22
    overlap = 13 <= hour < 16
    if overlap:
        return 3
    if london and not ny:
        return 1
    if ny and not london:
        return 2
    if london and ny:
        return 3
    return 0


def _minutes_since_london_open(ts: pd.Series) -> np.ndarray:
    """Minutes since 07:00 UTC same calendar day."""
    mins = ts.dt.hour.values * 60 + ts.dt.minute.values
    open_m = 7 * 60
    out = mins - open_m
    out = np.where(out < 0, out + 24 * 60, out)
    return out.astype(np.float64)


def _news_window_flag(ts: pd.Series) -> np.ndarray:
    """Rough high-impact US window (UTC)."""
    dow = ts.dt.dayofweek.values
    h = ts.dt.hour.values
    m = ts.dt.minute.values
    wd = dow < 5
    # ~12:30–14:00 UTC band
    tmin = h * 60 + m
    band = (tmin >= 12 * 60 + 25) & (tmin <= 14 * 60 + 5)
    return (wd & band).astype(np.float64)


def add_ml_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Add all ML feature columns to M1 OHLCV frame. Expects columns:
    timestamp, open, high, low, close, volume (volume optional -> 1).
    """
    out = df.copy()
    if "timestamp" not in out.columns:
        raise ValueError("df must have 'timestamp'")
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp"] = ts

    c = out["close"].values.astype(np.float64)
    h = out["high"].values.astype(np.float64)
    l = out["low"].values.astype(np.float64)
    o = out["open"].values.astype(np.float64)
    n = len(out)
    pip = pip_for_pair(pair)
    atr_eps = pip * 0.05

    if "volume" in out.columns:
        vol = out["volume"].values.astype(np.float64)
        vol = np.where(np.isfinite(vol) & (vol >= 0), vol, 0.0)
    else:
        vol = np.ones(n, dtype=np.float64)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values.astype(np.float64)
    atr14 = np.where(atr14 < atr_eps, atr_eps, atr14)

    bar_range = h - l
    body = np.abs(c - o)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l

    out["pa_range_atr_ratio"] = bar_range / atr14
    safe_range = np.where(bar_range > 0, bar_range, 1.0)
    out["pa_body_pct"] = np.where(bar_range > 0, body / safe_range, 0.0)
    out["pa_upper_wick_pct"] = np.where(bar_range > 0, upper_wick / safe_range, 0.0)
    out["pa_lower_wick_pct"] = np.where(bar_range > 0, lower_wick / safe_range, 0.0)

    bull = (c > o).astype(np.float64)
    bear = (c < o).astype(np.float64)
    out["pa_bull_count_5"] = pd.Series(bull).rolling(5, min_periods=1).sum().values
    out["pa_bear_count_5"] = pd.Series(bear).rolling(5, min_periods=1).sum().values

    gap = o - prev_c
    out["pa_gap_atr"] = gap / atr14

    br_rank = pd.Series(bar_range).rolling(20, min_periods=5).rank(pct=True)
    out["pa_range_pct_20"] = br_rank.values.astype(np.float64)

    rng_safe = np.where(bar_range > 1e-12, bar_range, np.nan)
    out["pa_close_pos_in_bar"] = np.nan_to_num((c - l) / rng_safe, nan=0.5)

    doji = (body < bar_range * 0.1) & (bar_range > 0)
    out["pa_doji_count_5"] = (
        pd.Series(doji.astype(float)).rolling(5, min_periods=1).sum().values
    )

    # Bull vs bear pressure last 5 bars (vectorized; ~[-5, 5])
    out["pa_bull_streak"] = (
        pd.Series(bull - bear).rolling(5, min_periods=1).sum().values
    )
    out["pa_body_atr"] = body / atr14
    out["pa_wick_imbalance_atr"] = (upper_wick - lower_wick) / atr14

    ema9 = _ema(out["close"], 9)
    ema20 = _ema(out["close"], 20)
    ema40 = _ema(out["close"], 40)

    out["ma_ema9_atr_dist"] = (c - ema9) / atr14
    out["ma_ema20_atr_dist"] = (c - ema20) / atr14
    out["ma_ema40_atr_dist"] = (c - ema40) / atr14

    def slope(arr, lag=10):
        s = np.zeros_like(arr)
        s[lag:] = arr[lag:] - arr[:-lag]
        return s

    out["ma_ema9_slope"] = slope(ema9, 10) / atr14
    out["ma_ema20_slope"] = slope(ema20, 10) / atr14
    out["ma_ema40_slope"] = slope(ema40, 10) / atr14
    out["ma_ema9_20_spread_atr"] = (ema9 - ema20) / atr14
    out["ma_ema9_40_spread_atr"] = (ema9 - ema40) / atr14

    # bars since 9/20 cross
    sign = np.sign(ema9 - ema20)
    sign = np.where(sign == 0, np.roll(sign, 1), sign)
    chg = (np.roll(sign, 1) != sign) & (np.arange(n) > 1)
    chg[0] = True
    last_ev = np.zeros(n, dtype=np.int32)
    last = 0
    for i in range(n):
        if chg[i]:
            last = i
        last_ev[i] = last
    out["ma_bars_since_ema9_20_cross"] = (np.arange(n) - last_ev).astype(np.float64)

    sign2 = np.sign(ema20 - ema40)
    sign2 = np.where(sign2 == 0, np.roll(sign2, 1), sign2)
    chg2 = (np.roll(sign2, 1) != sign2) & (np.arange(n) > 1)
    chg2[0] = True
    last_ev2 = np.zeros(n, dtype=np.int32)
    last2 = 0
    for i in range(n):
        if chg2[i]:
            last2 = i
        last_ev2[i] = last2
    out["ma_bars_since_ema20_40_cross"] = (np.arange(n) - last_ev2).astype(np.float64)

    # Volatility
    atr_pct = pd.Series(atr14).rolling(100, min_periods=20).rank(pct=True).values
    out["vol_atr14_pct_100"] = np.nan_to_num(atr_pct, nan=0.5)

    sma20 = pd.Series(c).rolling(20).mean().values
    std20 = pd.Series(c).rolling(20).std().values
    bb_bw = np.where(np.abs(sma20) > 1e-12, 4 * std20 / sma20, 0.0)
    bw_pct = pd.Series(bb_bw).rolling(100, min_periods=20).rank(pct=True).values
    out["vol_bb_bw_pct_100"] = np.nan_to_num(bw_pct, nan=0.5)
    bw_low = (
        pd.Series(bb_bw).rolling(100, min_periods=20).quantile(0.2).values
    )
    out["vol_bb_squeeze"] = (bb_bw <= bw_low).astype(np.float64)
    out["vol_bar_range_atr"] = bar_range / atr14
    range20 = pd.Series(h).rolling(20).max().values - pd.Series(l).rolling(20).min().values
    out["vol_range20_atr"] = range20 / atr14
    atr5 = pd.Series(tr).rolling(5, min_periods=1).mean().values.astype(np.float64)
    atr5 = np.where(atr5 < atr_eps, atr_eps, atr5)
    out["vol_atr5_over_14"] = atr5 / atr14

    rsi14 = _rsi(c, 14)
    out["mom_rsi14"] = np.nan_to_num(rsi14, nan=50.0)
    rsi5ago = np.roll(out["mom_rsi14"], 5)
    rsi5ago[:5] = out["mom_rsi14"][:5]
    out["mom_rsi_roc5"] = out["mom_rsi14"] - rsi5ago

    m_line, m_sig, m_hist = _macd(c)
    out["mom_macd_line_atr"] = m_line / atr14
    out["mom_macd_signal_atr"] = m_sig / atr14
    out["mom_macd_hist_atr"] = m_hist / atr14
    c10 = np.roll(c, 10)
    c10[:10] = c[:10]
    out["mom_roc10_atr"] = (c - c10) / atr14
    out["mom_ret1_atr"] = (c - prev_c) / atr14
    c3 = np.roll(c, 3)
    c3[:3] = c[:3]
    out["mom_ret3_atr"] = (c - c3) / atr14

    sk, sd = _stochastic(h, l, c)
    out["mom_stoch_k"] = np.nan_to_num(sk, nan=50.0)
    out["mom_stoch_d"] = np.nan_to_num(sd, nan=50.0)

    roll60_high = pd.Series(h).rolling(60, min_periods=10).max().values
    roll60_low = pd.Series(l).rolling(60, min_periods=10).min().values
    out["struct_dist_60high_atr"] = (roll60_high - c) / atr14
    out["struct_dist_60low_atr"] = (c - roll60_low) / atr14

    zone = 2 * pip
    touch_high = (h >= roll60_high - zone).astype(float)
    touch_low = (l <= roll60_low + zone).astype(float)
    out["struct_touch_high_zone_60"] = (
        pd.Series(touch_high).rolling(60, min_periods=10).sum().values
    )
    out["struct_touch_low_zone_60"] = (
        pd.Series(touch_low).rolling(60, min_periods=10).sum().values
    )

    rh20 = pd.Series(h).rolling(20).max().values
    rl20 = pd.Series(l).rolling(20).min().values
    out["struct_range_compress"] = (rh20 - rl20) / atr14
    rng20_safe = np.where((rh20 - rl20) > 1e-12, rh20 - rl20, np.nan)
    out["struct_close_pct_in_20range"] = np.nan_to_num(
        (c - rl20) / rng20_safe, nan=0.5
    )

    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_rng = bb_upper - bb_lower
    bb_rng = np.where(np.abs(bb_rng) > 1e-12, bb_rng, np.nan)
    out["mom_bb_pct_b"] = np.nan_to_num((c - bb_lower) / bb_rng, nan=0.5)

    dc = np.diff(c)
    dc = np.concatenate([[0], dc])
    sgn = np.sign(dc)
    swing = (np.roll(sgn, 1) != sgn) & (np.arange(n) > 1)
    out["struct_swing_count_40"] = (
        pd.Series(swing.astype(float)).rolling(40, min_periods=10).sum().values
    )

    hour = ts.dt.hour.values
    out["sess_hour"] = hour.astype(np.float64)
    ang = 2 * np.pi * hour.astype(np.float64) / 24.0
    out["sess_hour_sin"] = np.sin(ang)
    out["sess_hour_cos"] = np.cos(ang)
    out["sess_bucket"] = np.array([_session_bucket(int(x)) for x in hour], dtype=np.float64)
    wd = ts.dt.dayofweek.values
    out["sess_dow"] = np.clip(wd, 0, 4).astype(np.float64)
    out["sess_is_friday"] = (wd == 4).astype(np.float64)
    out["sess_min_since_london"] = _minutes_since_london_open(ts)
    out["sess_news_window"] = _news_window_flag(ts)

    # Session VWAP (UTC calendar day; aligns with liquid rollover proxy)
    day = ts.dt.floor("D")
    tp = (h + l + c) / 3.0
    _tmp = pd.DataFrame(
        {"d": day, "tpv": tp * vol, "v": vol},
        index=out.index,
    )
    _tmp["cum_tpv"] = _tmp.groupby("d", sort=False)["tpv"].cumsum()
    _tmp["cum_v"] = _tmp.groupby("d", sort=False)["v"].cumsum()
    vwap = _tmp["cum_tpv"].values / np.maximum(_tmp["cum_v"].values, 1e-12)
    out["vwap_dist_atr"] = (c - vwap) / atr14
    vwap_lag = np.roll(vwap, 10)
    vwap_lag[:10] = vwap[:10]
    out["vwap_slope_atr"] = (vwap - vwap_lag) / atr14

    vol_ma = _tmp.groupby("d", sort=False)["v"].transform(
        lambda x: x.expanding().mean()
    )
    vol_ma = np.maximum(vol_ma.values, 1e-9)
    out["vwap_vol_ratio"] = vol / vol_ma
    vol_rank = _tmp.groupby("d", sort=False)["v"].rank(pct=True).values
    out["vwap_session_vol_pct"] = np.nan_to_num(vol_rank, nan=0.5)

    vol_ma100 = pd.Series(vol).rolling(100, min_periods=10).mean().values
    vol_ma100 = np.where(vol_ma100 > 0, vol_ma100, 1.0)
    out["mom_vol_vs_100"] = vol / vol_ma100

    # Persist ATR for ML V2 labels / sweeps (not in FEATURE_COLUMNS).
    out["atr14"] = atr14.astype(np.float64)

    return out


# Order must be stable for LightGBM / live inference
FEATURE_COLUMNS = [
    "pa_range_atr_ratio",
    "pa_body_pct",
    "pa_upper_wick_pct",
    "pa_lower_wick_pct",
    "pa_bull_count_5",
    "pa_bear_count_5",
    "pa_gap_atr",
    "pa_range_pct_20",
    "pa_close_pos_in_bar",
    "pa_doji_count_5",
    "pa_bull_streak",
    "pa_body_atr",
    "pa_wick_imbalance_atr",
    "ma_ema9_atr_dist",
    "ma_ema20_atr_dist",
    "ma_ema40_atr_dist",
    "ma_ema9_slope",
    "ma_ema20_slope",
    "ma_ema40_slope",
    "ma_ema9_20_spread_atr",
    "ma_ema9_40_spread_atr",
    "ma_bars_since_ema9_20_cross",
    "ma_bars_since_ema20_40_cross",
    "vol_atr14_pct_100",
    "vol_bb_bw_pct_100",
    "vol_bb_squeeze",
    "vol_bar_range_atr",
    "vol_range20_atr",
    "vol_atr5_over_14",
    "mom_rsi14",
    "mom_rsi_roc5",
    "mom_macd_line_atr",
    "mom_macd_signal_atr",
    "mom_macd_hist_atr",
    "mom_roc10_atr",
    "mom_ret1_atr",
    "mom_ret3_atr",
    "mom_stoch_k",
    "mom_stoch_d",
    "mom_bb_pct_b",
    "mom_vol_vs_100",
    "struct_dist_60high_atr",
    "struct_dist_60low_atr",
    "struct_touch_high_zone_60",
    "struct_touch_low_zone_60",
    "struct_range_compress",
    "struct_close_pct_in_20range",
    "struct_swing_count_40",
    "sess_hour",
    "sess_hour_sin",
    "sess_hour_cos",
    "sess_bucket",
    "sess_dow",
    "sess_is_friday",
    "sess_min_since_london",
    "sess_news_window",
    "vwap_dist_atr",
    "vwap_slope_atr",
    "vwap_vol_ratio",
    "vwap_session_vol_pct",
]


def last_bar_feature_dict(df: pd.DataFrame, pair: str) -> dict[str, float]:
    """Return feature name -> value for the last complete row."""
    if df is None or len(df) < 120:
        return {}
    d = add_ml_features(df.tail(500), pair)
    row = d.iloc[-1]
    out = {}
    for col in FEATURE_COLUMNS:
        try:
            v = float(row[col])
            if not np.isfinite(v):
                v = 0.0
        except (KeyError, TypeError, ValueError):
            v = 0.0
        out[col] = v
    return out


def features_matrix(df: pd.DataFrame, pair: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (frame with features, X as float64 shape [n, n_features])."""
    d = add_ml_features(df, pair)
    X = d[FEATURE_COLUMNS].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return d, X
