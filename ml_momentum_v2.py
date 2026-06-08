#!/usr/bin/env python3
"""Momentum Scalper V2: Real-world stress test with spread, harder labels, random baseline.

Changes from V1:
  - Spread deducted from every trade
  - Harder labels: 1.5x ATR TP / 1.0x ATR SL (was 1.0/0.5)
  - Stricter triggers: 1.5 ATR move + 3x volume + ADX>20 + London/NY only
  - Random baseline comparison
  - Walk-forward with Optuna

Usage:
  python ml_momentum_v2.py --fx-dir data/fx_5yr --years 5
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

from scalp_mode.ml.bar_features import SPREAD_PIPS_DEFAULT, pip_for_pair

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("pip install lightgbm")

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

FEATURE_NAMES = [
    "vel_1", "vel_3", "vel_5", "vel_10", "vel_20",
    "acc_3", "acc_5", "acc_10",
    "consec_bull", "consec_pips",
    "macd_h_slope3", "rsi_roc3", "rsi_roc5", "stoch_kd_spread",
    "atr_ratio", "atr_rank", "bb_pos", "bb_width_ratio", "bar_range_atr",
    "range_5_atr", "range_10_atr", "vol_ratio", "body_pct", "wick_ratio",
    "dist_ema9", "dist_ema20", "ema_cross_dir", "bars_since_cross",
    "adx_val", "adx_slope3", "rsi_val", "macd_hist_norm", "stoch_k_val",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_overlap", "direction_feat", "momentum_score",
]


def add_m1_indicators(df):
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64) if "volume" in df else np.ones(len(df))

    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    tr = np.insert(tr, 0, h[0] - l[0])
    df["atr14"] = pd.Series(tr).rolling(14, min_periods=1).mean().values
    df["ema9"] = pd.Series(c).ewm(span=9, adjust=False).mean().values
    df["ema20"] = pd.Series(c).ewm(span=20, adjust=False).mean().values
    delta = pd.Series(c).diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)).fillna(50).values
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    df["macd_hist"] = ema12 - ema26 - pd.Series(ema12 - ema26).ewm(span=9, adjust=False).mean().values
    low14 = pd.Series(l).rolling(14, min_periods=1).min().values
    high14 = pd.Series(h).rolling(14, min_periods=1).max().values
    denom = high14 - low14
    df["stoch_k"] = np.where(denom > 1e-15, 100 * (c - low14) / denom, 50.0)
    df["stoch_d"] = pd.Series(df["stoch_k"]).rolling(3, min_periods=1).mean().values
    sma20 = pd.Series(c).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(c).rolling(20, min_periods=1).std().values
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20

    pdm = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]), np.maximum(h[1:] - h[:-1], 0), 0.0)
    mdm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]), np.maximum(l[:-1] - l[1:], 0), 0.0)
    pdm = np.insert(pdm, 0, 0.0)
    mdm = np.insert(mdm, 0, 0.0)
    atr_s = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    pdi = 100 * pd.Series(pdm).ewm(span=14, adjust=False).mean().values / np.maximum(atr_s, 1e-15)
    mdi = 100 * pd.Series(mdm).ewm(span=14, adjust=False).mean().values / np.maximum(atr_s, 1e-15)
    dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-15)
    df["adx"] = pd.Series(dx).ewm(span=14, adjust=False).mean().values

    df["vol_sma20"] = pd.Series(v).rolling(20, min_periods=1).mean().values
    df["bar_range"] = h - l
    df["body"] = np.abs(c - o)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["hour"] = ts.dt.hour
        df["dow"] = ts.dt.dayofweek
    return df


def find_strict_triggers(df, pip):
    """Strict triggers: 1.5 ATR move + 3x volume + ADX>20 + London/NY hours."""
    c = df["close"].values
    atr = df["atr14"].values
    vol = df["volume"].values if "volume" in df else np.ones(len(df))
    vol_sma = df["vol_sma20"].values
    adx = df["adx"].values
    hour = df["hour"].values if "hour" in df else np.zeros(len(df), dtype=int)
    triggers = []

    for i in range(40, len(df)):
        if atr[i] < pip * 0.5:
            continue
        h = int(hour[i])
        if h < 7 or h >= 19:
            continue
        if adx[i] < 20:
            continue
        move_5 = abs(c[i] - c[max(0, i-5)])
        if move_5 < 1.5 * atr[i]:
            continue
        if vol_sma[i] > 0 and vol[i] < 3.0 * vol_sma[i]:
            continue
        triggers.append(i)

    return triggers


def compute_labels_with_spread(df, triggers, pip, spread_pips, tp_atr=1.5, sl_atr=1.0, max_bars=30):
    """Labels with spread deducted. TP harder, SL wider."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    atr = df["atr14"].values
    spread = spread_pips * pip
    labels = []

    for idx in triggers:
        a = atr[idx]
        if a < 1e-10:
            labels.append({"idx": idx, "label": 0, "direction": 0, "pnl": 0})
            continue

        entry = c[idx]
        # Long: TP needs to overcome spread, SL is worsened by spread
        tp_long = entry + tp_atr * a
        sl_long = entry - sl_atr * a

        tp_short = entry - tp_atr * a
        sl_short = entry + sl_atr * a

        n = len(df)
        long_result, short_result = 0, 0
        long_pnl, short_pnl = 0.0, 0.0

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if l[j] <= sl_long:
                long_result = 0
                long_pnl = (sl_long - entry) / pip - spread_pips
                break
            if h[j] >= tp_long:
                long_result = 1
                long_pnl = (tp_long - entry) / pip - spread_pips
                break

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if h[j] >= sl_short:
                short_result = 0
                short_pnl = (entry - sl_short) / pip - spread_pips
                break
            if l[j] <= tp_short:
                short_result = 1
                short_pnl = (entry - tp_short) / pip - spread_pips
                break

        if long_result == 1 and short_result == 0:
            labels.append({"idx": idx, "label": 1, "direction": 1, "pnl": long_pnl})
        elif short_result == 1 and long_result == 0:
            labels.append({"idx": idx, "label": 1, "direction": -1, "pnl": short_pnl})
        elif long_result == 1 and short_result == 1:
            labels.append({"idx": idx, "label": 1, "direction": 1, "pnl": long_pnl})
        else:
            if abs(long_pnl) < abs(short_pnl):
                labels.append({"idx": idx, "label": 0, "direction": 1, "pnl": long_pnl})
            else:
                labels.append({"idx": idx, "label": 0, "direction": -1, "pnl": short_pnl})

    return labels


def extract_features(df, idx, pip):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    atr = df["atr14"].values
    a = max(atr[idx], pip * 0.1)
    cl = c[idx]
    n = len(df)
    def s(arr, i): return float(arr[max(0, min(i, n-1))])

    vel = {w: (cl - s(c, idx-w)) / a for w in [1, 3, 5, 10, 20]}
    acc = {w: vel[w] - (s(c, idx-w) - s(c, idx-2*w)) / a for w in [3, 5, 10]}

    consec, cpips = 0, 0.0
    for j in range(idx, max(idx-20, 0), -1):
        if c[j] > o[j]:
            consec += 1; cpips += (c[j]-o[j])/pip
        else: break

    mh = df["macd_hist"].values
    rsi = df["rsi14"].values
    sk, sd = df["stoch_k"].values[idx], df["stoch_d"].values[idx]
    atr_sma = np.mean(atr[max(0,idx-20):idx+1])
    atr_ratio = a / atr_sma if atr_sma > 1e-10 else 1.0
    atr_vals = atr[max(0,idx-100):idx+1]
    atr_rank = float((atr_vals < a).sum()) / max(len(atr_vals), 1)

    bb_u, bb_l = df["bb_upper"].values[idx], df["bb_lower"].values[idx]
    bb_r = bb_u - bb_l if bb_u > bb_l else 1e-10
    bb_w_avg = np.mean(df["bb_upper"].values[max(0,idx-20):idx+1] - df["bb_lower"].values[max(0,idx-20):idx+1])

    br = h[idx] - l[idx]
    ema9v, ema20v = df["ema9"].values, df["ema20"].values
    cross_dir = 1.0 if ema9v[idx] > ema20v[idx] else -1.0
    bars_cross = 0
    for j in range(idx, max(idx-50,0), -1):
        if (ema9v[j] > ema20v[j]) != (ema9v[max(0,j-1)] > ema20v[max(0,j-1)]):
            bars_cross = idx - j; break

    hr = int(df["hour"].values[idx]) if "hour" in df else 0
    dw = int(df["dow"].values[idx]) if "dow" in df else 0

    return [
        vel[1], vel[3], vel[5], vel[10], vel[20],
        acc[3], acc[5], acc[10],
        consec, cpips,
        (mh[idx] - s(mh, idx-3)) / a, rsi[idx] - s(rsi, idx-3), rsi[idx] - s(rsi, idx-5), sk - sd,
        atr_ratio, atr_rank, (cl - bb_l) / bb_r, bb_r / bb_w_avg if bb_w_avg > 1e-10 else 1.0, br / a,
        (max(h[max(0,idx-5):idx+1]) - min(l[max(0,idx-5):idx+1])) / a,
        (max(h[max(0,idx-10):idx+1]) - min(l[max(0,idx-10):idx+1])) / a,
        (df["volume"].values[idx] / df["vol_sma20"].values[idx]) if "volume" in df and df["vol_sma20"].values[idx] > 0 else 1.0,
        abs(c[idx]-o[idx]) / br if br > 1e-10 else 0,
        (br - abs(c[idx]-o[idx])) / br if br > 1e-10 else 0,
        (cl - ema9v[idx]) / a, (cl - ema20v[idx]) / a,
        cross_dir, float(bars_cross),
        df["adx"].values[idx], df["adx"].values[idx] - s(df["adx"].values, idx-3),
        rsi[idx], mh[idx] / a, sk,
        np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24),
        np.sin(2*np.pi*dw/5), np.cos(2*np.pi*dw/5),
        1.0 if 12 <= hr <= 16 else 0.0,
        1.0 if vel[5] > 0 else -1.0,
        vel[5] * 0.4 + vel[10] * 0.3 + (rsi[idx]-50)/50 * 0.3,
    ]


def build_pair_data(fx_dir, pair, fn, years):
    fpath = fx_dir / fn
    if not fpath.exists(): return None
    print(f"  {pair}: loading...", end=" ", flush=True)
    df = pd.read_csv(fpath, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp"}).sort_values("timestamp").reset_index(drop=True)
    if years > 0:
        cutoff = df["timestamp"].max() - pd.Timedelta(days=years*365)
        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    print(f"{len(df):,} bars...", end=" ", flush=True)
    df = add_m1_indicators(df)
    pip = pip_for_pair(pair)
    spread = SPREAD_PIPS_DEFAULT.get(pair, 1.5)

    print(f"triggers...", end=" ", flush=True)
    triggers = find_strict_triggers(df, pip)
    print(f"{len(triggers)} strict...", end=" ", flush=True)

    labels = compute_labels_with_spread(df, triggers, pip, spread)
    print(f"features...", end=" ", flush=True)

    rows = []
    for lab in labels:
        feat = extract_features(df, lab["idx"], pip)
        rows.append({
            **{FEATURE_NAMES[j]: feat[j] for j in range(len(feat))},
            "label": lab["label"],
            "direction": lab["direction"],
            "pnl_pips": lab["pnl"],
            "timestamp": str(df["timestamp"].iloc[lab["idx"]]),
            "pair": pair,
        })

    # Random baseline: same number of triggers, random positions, same TP/SL/spread
    rng = np.random.default_rng(42)
    valid_range = list(range(40, len(df) - 30))
    n_random = min(len(triggers), len(valid_range))
    random_idx = sorted(rng.choice(valid_range, size=n_random, replace=False))
    random_labels = compute_labels_with_spread(df, random_idx, pip, spread)
    random_pnls = [r["pnl"] for r in random_labels]

    print(f"{len(rows)} samples, {n_random} random baseline")
    del df; gc.collect()
    return pd.DataFrame(rows), random_pnls


def walk_forward(dataset, n_optuna=15):
    dataset["ts"] = pd.to_datetime(dataset["timestamp"])
    dataset = dataset.sort_values("ts").reset_index(drop=True)
    min_d, max_d = dataset["ts"].min(), dataset["ts"].max()
    results = []
    ws = min_d

    while ws + pd.DateOffset(months=4) <= max_d:
        train_end = ws + pd.DateOffset(months=3)
        test_end = train_end + pd.DateOffset(months=1)
        train = dataset[(dataset["ts"] >= ws) & (dataset["ts"] < train_end)]
        test = dataset[(dataset["ts"] >= train_end) & (dataset["ts"] < test_end)]

        if len(train) < 200 or len(test) < 20:
            ws += pd.DateOffset(months=1); continue

        X_tr = np.nan_to_num(train[FEATURE_NAMES].values.astype(np.float32))
        y_tr = train["label"].values.astype(np.int32)
        X_te = np.nan_to_num(test[FEATURE_NAMES].values.astype(np.float32))

        params = {"objective": "binary", "metric": "auc", "verbosity": -1,
                  "max_depth": 5, "num_leaves": 31, "learning_rate": 0.05,
                  "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8}

        if HAS_OPTUNA and len(train) > 500:
            split = int(len(X_tr) * 0.8)
            def obj(trial):
                p = {**params,
                     "max_depth": trial.suggest_int("md", 3, 7),
                     "num_leaves": trial.suggest_int("nl", 8, 48),
                     "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
                     "min_child_samples": trial.suggest_int("mcs", 10, 50)}
                ds = lgb.Dataset(X_tr[:split], y_tr[:split])
                dv = lgb.Dataset(X_tr[split:], y_tr[split:], reference=ds)
                m = lgb.train(p, ds, 200, valid_sets=[dv], callbacks=[lgb.early_stopping(20, verbose=False)])
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(y_tr[split:], m.predict(X_tr[split:]))
            study = optuna.create_study(direction="maximize")
            study.optimize(obj, n_trials=n_optuna, show_progress_bar=False)
            bp = study.best_params
            params.update({"max_depth": bp.get("md",5), "num_leaves": bp.get("nl",31),
                          "learning_rate": bp.get("lr",0.05), "min_child_samples": bp.get("mcs",20)})

        mdl = lgb.train(params, lgb.Dataset(X_tr, y_tr), 200)
        probs = mdl.predict(X_te)

        for thr in [0.50, 0.55, 0.60, 0.65]:
            mask = probs >= thr
            if mask.sum() < 3: continue
            ft = test.iloc[mask.nonzero()[0]]
            pnls = ft["pnl_pips"].values
            w = float(pnls[pnls > 0].sum())
            lo = float(-pnls[pnls < 0].sum())
            pf = w / lo if lo > 0 else (99.0 if w > 0 else 0.0)
            wins = (pnls > 0).sum()
            losses = (pnls <= 0).sum()
            n_days = max((test["ts"].max() - test["ts"].min()).days, 1)

            uf_pnls = test["pnl_pips"].values
            uf_w = float(uf_pnls[uf_pnls > 0].sum())
            uf_l = float(-uf_pnls[uf_pnls < 0].sum())
            uf_pf = uf_w / uf_l if uf_l > 0 else 0

            results.append({
                "window": f"{ws.strftime('%Y-%m')}→{test_end.strftime('%Y-%m')}",
                "thr": thr, "n_filtered": int(mask.sum()),
                "pf_after_spread": round(pf, 3),
                "wr": round(wins / max(wins+losses, 1), 3),
                "pips_after_spread": round(float(pnls.sum()), 1),
                "avg_win": round(float(pnls[pnls>0].mean()), 1) if wins > 0 else 0,
                "avg_loss": round(float(pnls[pnls<=0].mean()), 1) if losses > 0 else 0,
                "tpd": round(int(mask.sum()) / n_days, 1),
                "pf_unfiltered": round(uf_pf, 3),
            })

        ws += pd.DateOffset(months=1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx-dir", type=Path, required=True)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--out-json", type=Path, default=ROOT / "data" / "ml_momentum_v2_results.json")
    args = ap.parse_args()

    print("=" * 60)
    print("MOMENTUM SCALPER V2: STRESS TEST")
    print("Spread deducted | 1.5:1 R:R | Strict triggers | Random baseline")
    print("=" * 60)

    all_dfs = []
    all_random_pnls = []
    for pair, fn in PAIRS.items():
        result = build_pair_data(args.fx_dir, pair, fn, args.years)
        if result:
            df, rpnls = result
            all_dfs.append(df)
            all_random_pnls.extend(rpnls)

    dataset = pd.concat(all_dfs, ignore_index=True)
    del all_dfs; gc.collect()

    print(f"\nTotal samples: {len(dataset):,}")
    print(f"Label distribution: {dataset['label'].value_counts().to_dict()}")
    print(f"Win rate (after spread): {dataset['label'].mean()*100:.1f}%")
    print(f"Avg pips/trade (after spread): {dataset['pnl_pips'].mean():.2f}")

    # Random baseline stats
    rp = np.array(all_random_pnls)
    rw = float(rp[rp > 0].sum())
    rl = float(-rp[rp < 0].sum())
    random_pf = rw / rl if rl > 0 else 0
    random_wr = (rp > 0).sum() / max(len(rp), 1)
    print(f"\nRANDOM BASELINE (same TP/SL/spread, random entries):")
    print(f"  PF={random_pf:.2f}, WR={random_wr*100:.1f}%, avg={rp.mean():.2f} pips, total={rp.sum():+,.0f}")

    # Per pair
    for pair in PAIRS:
        sub = dataset[dataset["pair"] == pair]
        if len(sub) == 0: continue
        pnls = sub["pnl_pips"].values
        w = float(pnls[pnls>0].sum())
        lo = float(-pnls[pnls<0].sum())
        pf = w/lo if lo > 0 else 0
        n_days = max((pd.to_datetime(sub["timestamp"]).max() - pd.to_datetime(sub["timestamp"]).min()).days, 1)
        print(f"  {pair}: {len(sub):,} triggers ({len(sub)/n_days:.0f}/day), PF={pf:.2f}, pips={pnls.sum():+,.0f}")

    print(f"\nWalk-forward training (3m train, 1m test)...")
    t0 = time_mod.time()
    wf = walk_forward(dataset)
    print(f"Done in {(time_mod.time()-t0)/60:.1f} min, {len(wf)} windows")

    print(f"\n{'Thr':>5s} {'PF':>8s} {'Pips':>10s} {'N':>8s} {'TPD':>6s} {'WR':>6s} {'AvgW':>6s} {'AvgL':>6s} {'Improve':>8s} {'Beat':>6s}")
    print("-" * 75)

    report = {"dataset": len(dataset), "random_pf": round(random_pf, 3), "random_wr": round(random_wr, 3), "summary": {}}

    for thr in [0.50, 0.55, 0.60, 0.65]:
        entries = [r for r in wf if r["thr"] == thr]
        if not entries: continue
        avg_pf = np.mean([r["pf_after_spread"] for r in entries])
        tot_pips = sum(r["pips_after_spread"] for r in entries)
        tot_n = sum(r["n_filtered"] for r in entries)
        avg_tpd = np.mean([r["tpd"] for r in entries])
        avg_wr = np.mean([r["wr"] for r in entries])
        avg_w = np.mean([r["avg_win"] for r in entries if r["avg_win"] > 0]) if any(r["avg_win"] > 0 for r in entries) else 0
        avg_l = np.mean([r["avg_loss"] for r in entries if r["avg_loss"] < 0]) if any(r["avg_loss"] < 0 for r in entries) else 0
        better = sum(1 for r in entries if r["pf_after_spread"] > random_pf)
        avg_imp = avg_pf - random_pf

        print(f"{thr:>5.2f} {avg_pf:>8.2f} {tot_pips:>+10,.0f} {tot_n:>8,d} {avg_tpd:>6.1f} {avg_wr*100:>5.0f}% {avg_w:>+5.1f} {avg_l:>+5.1f} {avg_imp:>+8.3f} {better:>3d}/{len(entries)}")

        report["summary"][str(thr)] = {
            "avg_pf": round(avg_pf, 3), "total_pips": round(tot_pips, 1),
            "total_trades": tot_n, "avg_tpd": round(avg_tpd, 1),
            "avg_wr": round(avg_wr, 3), "beats_random": better, "total_windows": len(entries),
            "pf_vs_random": round(avg_imp, 3),
        }

    # Verdict
    best_thr = max(report["summary"].items(), key=lambda x: x[1]["avg_pf"]) if report["summary"] else (None, {})
    if best_thr[1]:
        b = best_thr[1]
        passed = b["avg_pf"] > 1.2 and b["pf_vs_random"] > 0.3 and b["avg_tpd"] >= 5
        print(f"\nVERDICT: {'PASS' if passed else 'FAIL'}")
        print(f"  Best threshold: {best_thr[0]}")
        print(f"  PF after spread: {b['avg_pf']:.2f} (need >1.2)")
        print(f"  vs Random: +{b['pf_vs_random']:.3f} (need >+0.3)")
        print(f"  Trades/day: {b['avg_tpd']:.1f} (need 5-50)")
        report["verdict"] = "PASS" if passed else "FAIL"
    else:
        report["verdict"] = "NO_DATA"

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
