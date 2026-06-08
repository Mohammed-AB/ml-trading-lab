#!/usr/bin/env python3
"""Fast Momentum ML Scalper: M1 LightGBM for 20+ trades/day.

Walk-forward: 3-month train, 1-month test, Optuna hyperparams.
Labels: ATR-based (1.0x TP / 0.5x SL within 30 bars).
Features: 60 momentum/volatility/structure/time features.
Triggers: only predict when momentum exists (ATR move, volume spike, EMA cross).

Usage:
  python ml_momentum_scalper.py --fx-dir data/fx_5yr --years 5
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair  # noqa: E402

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: pip install lightgbm", file=sys.stderr)
    sys.exit(1)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

PAIRS = {
    "EUR_USD": "EURUSD_1min.csv",
    "USD_JPY": "USDJPY_1min.csv",
    "USD_CAD": "USDCAD_1min.csv",
    "AUD_USD": "AUDUSD_1min.csv",
    "NZD_USD": "NZDUSD_1min.csv",
}


# ---------------------------------------------------------------------------
# INDICATORS (lightweight, M1-oriented)
# ---------------------------------------------------------------------------

def add_m1_indicators(df):
    """Add indicators needed for features + triggers."""
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64) if "volume" in df else np.ones(len(df))
    n = len(df)

    # ATR 14
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.insert(tr, 0, h[0] - l[0])
    df["atr14"] = pd.Series(tr).rolling(14, min_periods=1).mean().values

    # EMAs
    df["ema9"] = pd.Series(c).ewm(span=9, adjust=False).mean().values
    df["ema20"] = pd.Series(c).ewm(span=20, adjust=False).mean().values
    df["ema40"] = pd.Series(c).ewm(span=40, adjust=False).mean().values

    # RSI 14
    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).fillna(50).values

    # MACD
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    df["macd"] = ema12 - ema26
    df["macd_signal"] = pd.Series(df["macd"]).ewm(span=9, adjust=False).mean().values
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Stochastic
    low14 = pd.Series(l).rolling(14, min_periods=1).min().values
    high14 = pd.Series(h).rolling(14, min_periods=1).max().values
    denom = high14 - low14
    df["stoch_k"] = np.where(denom > 1e-15, 100 * (c - low14) / denom, 50.0)
    df["stoch_d"] = pd.Series(df["stoch_k"]).rolling(3, min_periods=1).mean().values

    # Bollinger
    sma20 = pd.Series(c).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(c).rolling(20, min_periods=1).std().values
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"] = sma20

    # ADX (simplified)
    pdm = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]), np.maximum(h[1:] - h[:-1], 0), 0.0)
    mdm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]), np.maximum(l[:-1] - l[1:], 0), 0.0)
    pdm = np.insert(pdm, 0, 0.0)
    mdm = np.insert(mdm, 0, 0.0)
    atr_s = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    pdi = 100 * pd.Series(pdm).ewm(span=14, adjust=False).mean().values / np.maximum(atr_s, 1e-15)
    mdi = 100 * pd.Series(mdm).ewm(span=14, adjust=False).mean().values / np.maximum(atr_s, 1e-15)
    dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-15)
    df["adx"] = pd.Series(dx).ewm(span=14, adjust=False).mean().values

    # Bar metrics
    df["bar_range"] = h - l
    df["body"] = np.abs(c - o)
    df["is_bull"] = (c > o).astype(np.float32)

    # Volume SMA
    df["vol_sma20"] = pd.Series(v).rolling(20, min_periods=1).mean().values

    # Hour / DOW
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["dow"] = ts.dt.dayofweek
        df["minute"] = ts.dt.minute

    return df


# ---------------------------------------------------------------------------
# MOMENTUM TRIGGERS (pre-filter: only predict when something is happening)
# ---------------------------------------------------------------------------

def find_momentum_triggers(df, pip):
    """Return indices where a momentum trigger fires."""
    c = df["close"].values
    atr = df["atr14"].values
    vol = df["volume"].values if "volume" in df else np.ones(len(df))
    vol_sma = df["vol_sma20"].values
    ema9 = df["ema9"].values
    ema20 = df["ema20"].values
    rsi = df["rsi14"].values
    n = len(df)
    triggers = []

    for i in range(40, n):
        if atr[i] < pip * 0.5:
            continue

        fired = False

        # Trigger 1: price moved > 0.5x ATR in last 5 bars
        if not fired:
            move_5 = abs(c[i] - c[max(0, i-5)])
            if move_5 > 0.5 * atr[i]:
                fired = True

        # Trigger 2: volume spike > 2x SMA
        if not fired and vol_sma[i] > 0:
            if vol[i] > 2.0 * vol_sma[i]:
                fired = True

        # Trigger 3: EMA9 crossed EMA20 in last 3 bars
        if not fired:
            for j in range(max(1, i-3), i+1):
                if (ema9[j] > ema20[j]) != (ema9[j-1] > ema20[j-1]):
                    fired = True
                    break

        # Trigger 4: RSI crossed 30 or 70 in last 3 bars
        if not fired:
            for j in range(max(1, i-3), i+1):
                if (rsi[j] > 70 and rsi[j-1] <= 70) or (rsi[j] < 30 and rsi[j-1] >= 30):
                    fired = True
                    break

        if fired:
            triggers.append(i)

    return triggers


# ---------------------------------------------------------------------------
# LABELS: ATR-based momentum (1.0x TP / 0.5x SL within 30 bars)
# ---------------------------------------------------------------------------

def compute_momentum_labels(df, triggers, tp_atr=1.0, sl_atr=0.5, max_bars=30):
    """For each trigger, determine: long_label, short_label, direction."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    atr = df["atr14"].values
    n = len(df)

    labels = []
    for idx in triggers:
        a = atr[idx]
        if a < 1e-10:
            labels.append({"idx": idx, "label": 0, "direction": 0, "pnl": 0})
            continue

        entry = c[idx]
        tp_long = entry + tp_atr * a
        sl_long = entry - sl_atr * a
        tp_short = entry - tp_atr * a
        sl_short = entry + sl_atr * a

        long_result = 0
        short_result = 0
        long_pnl = 0.0
        short_pnl = 0.0

        # Check long
        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if l[j] <= sl_long:
                long_result = 0
                long_pnl = sl_long - entry
                break
            if h[j] >= tp_long:
                long_result = 1
                long_pnl = tp_long - entry
                break
        # Check short
        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if h[j] >= sl_short:
                short_result = 0
                short_pnl = entry - sl_short
                break
            if l[j] <= tp_short:
                short_result = 1
                short_pnl = entry - tp_short
                break

        # Pick best direction
        if long_result == 1 and short_result == 0:
            labels.append({"idx": idx, "label": 1, "direction": 1, "pnl": long_pnl})
        elif short_result == 1 and long_result == 0:
            labels.append({"idx": idx, "label": 1, "direction": -1, "pnl": short_pnl})
        elif long_result == 1 and short_result == 1:
            labels.append({"idx": idx, "label": 1, "direction": 1, "pnl": long_pnl})
        else:
            # Both lose -- pick the one with less loss, label 0
            if abs(long_pnl) < abs(short_pnl):
                labels.append({"idx": idx, "label": 0, "direction": 1, "pnl": long_pnl})
            else:
                labels.append({"idx": idx, "label": 0, "direction": -1, "pnl": short_pnl})

    return labels


# ---------------------------------------------------------------------------
# FEATURES: 60 momentum/vol/structure/time features at trigger bar
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "vel_1", "vel_3", "vel_5", "vel_10", "vel_20",
    "acc_1", "acc_3", "acc_5", "acc_10", "acc_20",
    "consec_bull", "consec_pips", "macd_h_slope1", "macd_h_slope3", "macd_h_slope5",
    "rsi_roc1", "rsi_roc3", "rsi_roc5", "stoch_kd_spread", "stoch_k_dir",
    "atr_ratio", "atr_rank", "bb_pos", "bb_width_ratio", "bar_range_atr",
    "range_5", "range_10", "range_20", "vol_ratio", "spread_proxy",
    "hl_range_5_atr", "hl_range_10_atr", "hl_range_20_atr", "body_pct", "wick_ratio",
    "dist_ema9", "dist_ema20", "dist_ema40", "ema9_20_cross_dir", "bars_since_cross",
    "dist_bb_upper", "dist_bb_lower", "dist_h1_high", "dist_h1_low", "adx_val",
    "adx_slope3", "supertrend_proxy", "rsi_val", "macd_hist_norm", "stoch_k_val",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "min_since_london",
    "min_since_ny", "is_overlap", "is_news_window", "direction_feat", "momentum_score",
]


def extract_features(df, idx, pip):
    """Extract 60 features at bar idx."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    atr = df["atr14"].values
    n = len(df)
    a = max(atr[idx], pip * 0.1)
    cl = c[idx]

    def safe(arr, i):
        return float(arr[max(0, min(i, n-1))])

    # Velocity (price change in ATR units)
    vel = {}
    for w in [1, 3, 5, 10, 20]:
        vel[w] = (cl - safe(c, idx - w)) / a

    # Acceleration
    acc = {}
    for w in [1, 3, 5, 10, 20]:
        v_now = (cl - safe(c, idx - w)) / a
        v_prev = (safe(c, idx - w) - safe(c, idx - 2*w)) / a
        acc[w] = v_now - v_prev

    # Consecutive same-direction bars
    consec = 0
    consec_pips = 0.0
    for j in range(idx, max(idx - 20, 0), -1):
        if c[j] > o[j]:
            consec += 1
            consec_pips += (c[j] - o[j]) / pip
        else:
            break

    # MACD histogram slope
    mh = df["macd_hist"].values
    mh_s1 = (mh[idx] - safe(mh, idx-1)) / a
    mh_s3 = (mh[idx] - safe(mh, idx-3)) / a
    mh_s5 = (mh[idx] - safe(mh, idx-5)) / a

    # RSI rate of change
    rsi = df["rsi14"].values
    rsi_r1 = rsi[idx] - safe(rsi, idx-1)
    rsi_r3 = rsi[idx] - safe(rsi, idx-3)
    rsi_r5 = rsi[idx] - safe(rsi, idx-5)

    # Stochastic
    sk = df["stoch_k"].values[idx]
    sd = df["stoch_d"].values[idx]

    # ATR context
    atr_sma = np.mean(atr[max(0, idx-20):idx+1])
    atr_ratio = a / atr_sma if atr_sma > 1e-10 else 1.0
    atr_vals = atr[max(0, idx-100):idx+1]
    atr_rank = float((atr_vals < a).sum()) / max(len(atr_vals), 1)

    # Bollinger
    bb_u = df["bb_upper"].values[idx]
    bb_l = df["bb_lower"].values[idx]
    bb_r = bb_u - bb_l if bb_u > bb_l else 1e-10
    bb_pos = (cl - bb_l) / bb_r
    bb_w_avg = np.mean(df["bb_upper"].values[max(0,idx-20):idx+1] - df["bb_lower"].values[max(0,idx-20):idx+1])
    bb_width_ratio = bb_r / bb_w_avg if bb_w_avg > 1e-10 else 1.0

    # Bar
    br = h[idx] - l[idx]
    bar_range_atr = br / a
    body_pct = abs(c[idx] - o[idx]) / br if br > 1e-10 else 0
    wick = br - abs(c[idx] - o[idx])
    wick_ratio = wick / br if br > 1e-10 else 0

    # Ranges
    r5 = (max(h[max(0,idx-5):idx+1]) - min(l[max(0,idx-5):idx+1])) / a
    r10 = (max(h[max(0,idx-10):idx+1]) - min(l[max(0,idx-10):idx+1])) / a
    r20 = (max(h[max(0,idx-20):idx+1]) - min(l[max(0,idx-20):idx+1])) / a

    # Volume
    vol_v = df["volume"].values[idx] if "volume" in df else 0
    vol_sma = df["vol_sma20"].values[idx]
    vol_ratio = vol_v / vol_sma if vol_sma > 0 else 1.0
    spread_proxy = br / abs(c[idx] - o[idx]) if abs(c[idx] - o[idx]) > 1e-10 else 5.0

    # EMA distances
    d_ema9 = (cl - df["ema9"].values[idx]) / a
    d_ema20 = (cl - df["ema20"].values[idx]) / a
    d_ema40 = (cl - df["ema40"].values[idx]) / a
    ema9v = df["ema9"].values
    ema20v = df["ema20"].values
    cross_dir = 1.0 if ema9v[idx] > ema20v[idx] else -1.0
    bars_cross = 0
    for j in range(idx, max(idx-50, 0), -1):
        if (ema9v[j] > ema20v[j]) != (ema9v[max(0,j-1)] > ema20v[max(0,j-1)]):
            bars_cross = idx - j
            break

    # ADX
    adx_v = df["adx"].values[idx]
    adx_s3 = adx_v - safe(df["adx"].values, idx-3)

    # H1 high/low (approximate from last 60 bars)
    h1_high = max(h[max(0,idx-60):idx+1])
    h1_low = min(l[max(0,idx-60):idx+1])
    d_h1_high = (h1_high - cl) / a
    d_h1_low = (cl - h1_low) / a

    # Supertrend proxy (EMA20 slope direction)
    st_proxy = 1.0 if ema20v[idx] > safe(ema20v, idx-10) else -1.0

    # Time
    hr = int(df["hour"].values[idx]) if "hour" in df else 0
    dw = int(df["dow"].values[idx]) if "dow" in df else 0
    h_sin = np.sin(2 * np.pi * hr / 24)
    h_cos = np.cos(2 * np.pi * hr / 24)
    d_sin = np.sin(2 * np.pi * dw / 5)
    d_cos = np.cos(2 * np.pi * dw / 5)
    min_london = max(0, (hr - 7) * 60 + (int(df["minute"].values[idx]) if "minute" in df else 0))
    min_ny = max(0, (hr - 12) * 60 + (int(df["minute"].values[idx]) if "minute" in df else 0))
    is_overlap = 1.0 if 12 <= hr <= 16 else 0.0
    is_news = 0.0  # placeholder

    # Direction from momentum
    dir_feat = 1.0 if vel[5] > 0 else -1.0

    # Momentum score (composite)
    mom_score = vel[5] * 0.4 + vel[10] * 0.3 + (rsi[idx] - 50) / 50 * 0.3

    return [
        vel[1], vel[3], vel[5], vel[10], vel[20],
        acc[1], acc[3], acc[5], acc[10], acc[20],
        consec, consec_pips, mh_s1, mh_s3, mh_s5,
        rsi_r1, rsi_r3, rsi_r5, sk - sd, 1.0 if sk > sd else -1.0,
        atr_ratio, atr_rank, bb_pos, bb_width_ratio, bar_range_atr,
        r5, r10, r20, vol_ratio, spread_proxy,
        r5, r10, r20, body_pct, wick_ratio,
        d_ema9, d_ema20, d_ema40, cross_dir, float(bars_cross),
        (bb_u - cl) / a, (cl - bb_l) / a, d_h1_high, d_h1_low, adx_v,
        adx_s3, st_proxy, rsi[idx], mh[idx] / a, sk,
        h_sin, h_cos, d_sin, d_cos, float(min_london),
        float(min_ny), is_overlap, is_news, dir_feat, mom_score,
    ]


# ---------------------------------------------------------------------------
# BUILD DATASET
# ---------------------------------------------------------------------------

def build_pair_dataset(fx_dir, pair, filename, years):
    """Build labeled feature dataset for one pair."""
    fpath = fx_dir / filename
    if not fpath.exists():
        return None
    print(f"  {pair}: loading...", end=" ", flush=True)
    df = pd.read_csv(fpath, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
    if years > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=years * 365)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    print(f"{len(df):,} bars, adding indicators...", end=" ", flush=True)
    df = add_m1_indicators(df)
    pip = pip_for_pair(pair)

    print(f"finding triggers...", end=" ", flush=True)
    triggers = find_momentum_triggers(df, pip)
    print(f"{len(triggers)} triggers, labeling...", end=" ", flush=True)
    labels = compute_momentum_labels(df, triggers)

    print(f"extracting features...", end=" ", flush=True)
    rows = []
    for lab in labels:
        feat = extract_features(df, lab["idx"], pip)
        rows.append({
            **{FEATURE_NAMES[j]: feat[j] for j in range(len(feat))},
            "label": lab["label"],
            "direction": lab["direction"],
            "pnl_pips": lab["pnl"] / pip,
            "timestamp": str(df["timestamp"].iloc[lab["idx"]]),
            "pair": pair,
        })
    print(f"{len(rows)} samples")
    del df
    gc.collect()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# WALK-FORWARD TRAINING + BACKTEST
# ---------------------------------------------------------------------------

def walk_forward_backtest(dataset, n_optuna=20):
    """3-month train, 1-month test, slide forward."""
    dataset["ts"] = pd.to_datetime(dataset["timestamp"])
    dataset = dataset.sort_values("ts").reset_index(drop=True)
    min_d = dataset["ts"].min()
    max_d = dataset["ts"].max()

    results = []
    ws = min_d

    while ws + pd.DateOffset(months=4) <= max_d:
        train_end = ws + pd.DateOffset(months=3)
        test_end = train_end + pd.DateOffset(months=1)

        train = dataset[(dataset["ts"] >= ws) & (dataset["ts"] < train_end)]
        test = dataset[(dataset["ts"] >= train_end) & (dataset["ts"] < test_end)]

        if len(train) < 500 or len(test) < 50:
            ws += pd.DateOffset(months=1)
            continue

        X_tr = np.nan_to_num(train[FEATURE_NAMES].values.astype(np.float32))
        y_tr = train["label"].values.astype(np.int32)
        X_te = np.nan_to_num(test[FEATURE_NAMES].values.astype(np.float32))

        # Optuna
        best_params = {"max_depth": 5, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20}
        if HAS_OPTUNA and len(train) > 1000:
            split = int(len(X_tr) * 0.8)
            def obj(trial):
                p = {
                    "objective": "binary", "metric": "auc", "verbosity": -1,
                    "max_depth": trial.suggest_int("md", 3, 7),
                    "num_leaves": trial.suggest_int("nl", 8, 48),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                    "min_child_samples": trial.suggest_int("mcs", 10, 50),
                    "subsample": 0.8, "colsample_bytree": 0.8,
                }
                ds = lgb.Dataset(X_tr[:split], y_tr[:split])
                dv = lgb.Dataset(X_tr[split:], y_tr[split:], reference=ds)
                m = lgb.train(p, ds, 200, valid_sets=[dv], callbacks=[lgb.early_stopping(20, verbose=False)])
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(y_tr[split:], m.predict(X_tr[split:]))
            study = optuna.create_study(direction="maximize")
            study.optimize(obj, n_trials=n_optuna, show_progress_bar=False)
            bp = study.best_params
            best_params = {
                "max_depth": bp.get("md", 5), "num_leaves": bp.get("nl", 31),
                "learning_rate": bp.get("lr", 0.05), "min_child_samples": bp.get("mcs", 20),
            }

        params = {"objective": "binary", "metric": "auc", "verbosity": -1,
                  "subsample": 0.8, "colsample_bytree": 0.8, **best_params}
        mdl = lgb.train(params, lgb.Dataset(X_tr, y_tr), 200)
        probs = mdl.predict(X_te)

        # Direction: use model for quality, momentum_score for direction
        dir_col = test["direction"].values
        pnl_col = test["pnl_pips"].values

        for thr in [0.50, 0.55, 0.60, 0.65]:
            mask = probs >= thr
            if mask.sum() < 5:
                continue
            filtered = test.iloc[mask.nonzero()[0]]
            pnls = filtered["pnl_pips"].values
            dirs = filtered["direction"].values
            wins = (pnls > 0).sum()
            losses = (pnls <= 0).sum()
            w_sum = float(pnls[pnls > 0].sum())
            l_sum = float(-pnls[pnls < 0].sum())
            pf = w_sum / l_sum if l_sum > 0 else (99.0 if w_sum > 0 else 0.0)
            n_days = (test["ts"].max() - test["ts"].min()).days or 1
            tpd = len(filtered) / n_days

            # Unfiltered baseline
            uf_pnls = test["pnl_pips"].values
            uf_w = float(uf_pnls[uf_pnls > 0].sum())
            uf_l = float(-uf_pnls[uf_pnls < 0].sum())
            uf_pf = uf_w / uf_l if uf_l > 0 else 0

            results.append({
                "window": f"{ws.strftime('%Y-%m')}→{test_end.strftime('%Y-%m')}",
                "thr": thr,
                "n_train": len(train), "n_test": len(test),
                "n_filtered": int(mask.sum()),
                "pf": round(pf, 3),
                "wr": round(wins / max(wins + losses, 1), 3),
                "pips": round(float(pnls.sum()), 1),
                "trades_per_day": round(tpd, 1),
                "pf_unfiltered": round(uf_pf, 3),
                "improvement": round(pf - uf_pf, 3),
            })

        ws += pd.DateOffset(months=1)

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fast Momentum ML Scalper")
    ap.add_argument("--fx-dir", type=Path, required=True)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "ml_momentum_results.json")
    args = ap.parse_args()

    print("=" * 60)
    print("FAST MOMENTUM ML SCALPER")
    print(f"Data: {args.fx_dir}, Years: {args.years}")
    print("=" * 60)

    # Build dataset
    print("\n1. Building labeled dataset from M1 data...")
    all_dfs = []
    for pair, fn in PAIRS.items():
        pdf = build_pair_dataset(args.fx_dir, pair, fn, args.years)
        if pdf is not None and len(pdf) > 0:
            all_dfs.append(pdf)
    if not all_dfs:
        print("No data!")
        return 1

    dataset = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    print(f"\nTotal samples: {len(dataset):,}")
    print(f"Label distribution: {dataset['label'].value_counts().to_dict()}")
    print(f"Win rate (unfiltered): {dataset['label'].mean()*100:.1f}%")
    print(f"Avg pips/trade (unfiltered): {dataset['pnl_pips'].mean():.2f}")

    # Per pair stats
    for pair in PAIRS:
        sub = dataset[dataset["pair"] == pair]
        if len(sub) == 0:
            continue
        pnls = sub["pnl_pips"].values
        w = float(pnls[pnls > 0].sum())
        lo = float(-pnls[pnls < 0].sum())
        pf = w / lo if lo > 0 else 0
        n_days = max((pd.to_datetime(sub["timestamp"]).max() - pd.to_datetime(sub["timestamp"]).min()).days, 1)
        print(f"  {pair}: {len(sub):,} triggers, PF={pf:.2f}, {len(sub)/n_days:.0f}/day, pips={pnls.sum():+,.0f}")

    # Walk-forward
    print(f"\n2. Walk-forward training (3-month train, 1-month test)...")
    t0 = time_mod.time()
    wf_results = walk_forward_backtest(dataset)
    elapsed = time_mod.time() - t0
    print(f"   Done in {elapsed/60:.1f} minutes, {len(wf_results)} test windows")

    # Summary
    print(f"\n3. Results by threshold:")
    print(f"{'Thr':>5s} {'Avg PF':>8s} {'Pips':>10s} {'Trades':>8s} {'TPD':>6s} {'Improve':>8s} {'Better':>8s}")
    print("-" * 60)

    report = {"dataset_size": len(dataset), "wf_results": wf_results, "summary": {}}

    for thr in [0.50, 0.55, 0.60, 0.65]:
        entries = [r for r in wf_results if r["thr"] == thr]
        if not entries:
            continue
        avg_pf = np.mean([r["pf"] for r in entries])
        tot_pips = sum(r["pips"] for r in entries)
        tot_n = sum(r["n_filtered"] for r in entries)
        avg_tpd = np.mean([r["trades_per_day"] for r in entries])
        better = sum(1 for r in entries if r["pf"] > r["pf_unfiltered"])
        avg_imp = np.mean([r["improvement"] for r in entries])
        print(f"{thr:>5.2f} {avg_pf:>8.2f} {tot_pips:>+10,.0f} {tot_n:>8,d} {avg_tpd:>6.1f} {avg_imp:>+8.3f} {better:>3d}/{len(entries)}")
        report["summary"][str(thr)] = {
            "avg_pf": round(avg_pf, 3), "total_pips": round(tot_pips, 1),
            "total_trades": tot_n, "avg_trades_per_day": round(avg_tpd, 1),
            "windows_improved": better, "total_windows": len(entries),
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
