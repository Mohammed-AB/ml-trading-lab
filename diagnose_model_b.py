"""Diagnose Model B — FIXED: exclude current M5 candle from range."""
import pandas as pd
import numpy as np
from src.scalp_mode.engine.feature_engine import FeatureEngine
from src.scalp_mode.engine.regime_engine import RegimeEngine, Regime

df = pd.read_csv("data/EUR_USD_M1_3m.csv")
fe = FeatureEngine()
re_eng = RegimeEngine(
    {"trend": {"ema_slope_thr": 0.15, "rsi_min": 52, "rsi_max": 78},
     "range": {"bb_width_thr": 0.004}})

groups = len(df) // 5
trimmed = df.iloc[:groups * 5].copy()
trimmed["g"] = np.repeat(range(groups), 5)
m5 = trimmed.groupby("g").agg(
    {"open": "first", "high": "max", "low": "min",
     "close": "last", "volume": "sum"}).reset_index(drop=True)

range_bars = 0
poked_top = 0
poked_bottom = 0
wick_ratio_pass_top = 0
wick_ratio_pass_bottom = 0
wick_excess_pass_top = 0
wick_excess_pass_bottom = 0
all_conditions_top = 0
all_conditions_bottom = 0
wick_ratios = []
wick_excesses = []
rsi_values = []

RANGE_WINDOW = 12
WICK_RATIO_MIN = 0.60
WICK_EXCESS_ATR = 0.25
RSI_OB = 65
RSI_REV_DOWN = 60
RSI_OS = 35
RSI_REV_UP = 40

for i in range(max(60, RANGE_WINDOW * 5 + 5), len(df)):
    m1_chunk = df.iloc[max(0, i - 99):i + 1]
    if len(m1_chunk) < 50:
        continue
    ind_m1 = fe.compute(m1_chunk, "M1")

    m5_idx = i // 5
    if m5_idx < RANGE_WINDOW + 21:
        continue
    m5_chunk = m5.iloc[max(0, m5_idx - 49):m5_idx + 1]
    if len(m5_chunk) < 20:
        continue
    ind_m5 = fe.compute(m5_chunk, "M5")
    close_m5 = float(m5_chunk.iloc[-1]["close"])
    regime_result = re_eng.evaluate(ind_m5, close_m5)

    if regime_result.regime != Regime.RANGE:
        continue

    range_bars += 1
    if range_bars % 10000 == 0:
        print(f"  Processing... {range_bars} Range bars checked")

    atr = ind_m1.atr14
    if atr is None or atr <= 0:
        continue

    # FIX: exclude current M5 candle — use previous 12
    if m5_idx < RANGE_WINDOW + 1:
        continue
    range_slice = m5.iloc[m5_idx - RANGE_WINDOW:m5_idx]  # excludes current
    if len(range_slice) < RANGE_WINDOW:
        continue
    range_high = float(range_slice["high"].max())
    range_low = float(range_slice["low"].min())

    bar = df.iloc[i]
    o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
    candle_range = h - l
    if candle_range <= 0:
        continue

    rsi = ind_m1.rsi14
    if rsi is not None:
        rsi_values.append(rsi)

    # --- TOP ---
    upper_wick = h - max(o, c)
    wick_excess_top = h - range_high
    wick_ratio_top = upper_wick / candle_range

    if wick_excess_top > 0:
        poked_top += 1
        wick_excesses.append(wick_excess_top / atr)
        wick_ratios.append(wick_ratio_top)
        if wick_excess_top <= WICK_EXCESS_ATR * atr:
            wick_excess_pass_top += 1
        if wick_ratio_top >= WICK_RATIO_MIN:
            wick_ratio_pass_top += 1
        if c < range_high and wick_excess_top <= WICK_EXCESS_ATR * atr and wick_ratio_top >= WICK_RATIO_MIN:
            if rsi is not None and rsi < RSI_REV_DOWN:
                all_conditions_top += 1

    # --- BOTTOM ---
    lower_wick = min(o, c) - l
    wick_excess_bottom = range_low - l
    wick_ratio_bottom = lower_wick / candle_range

    if wick_excess_bottom > 0:
        poked_bottom += 1
        if wick_excess_bottom <= WICK_EXCESS_ATR * atr:
            wick_excess_pass_bottom += 1
        if wick_ratio_bottom >= WICK_RATIO_MIN:
            wick_ratio_pass_bottom += 1
        if c > range_low and wick_excess_bottom <= WICK_EXCESS_ATR * atr and wick_ratio_bottom >= WICK_RATIO_MIN:
            if rsi is not None and rsi > RSI_REV_UP:
                all_conditions_bottom += 1

print(f"\n=== MODEL B DIAGNOSTIC (EUR_USD) — FIXED RANGE ===")
print(f"  Total Range bars: {range_bars}")
print(f"\n--- TOP (Short) ---")
print(f"  Poked above range_high: {poked_top} ({poked_top/max(range_bars,1)*100:.2f}%)")
print(f"  wick_excess <= 0.25*ATR: {wick_excess_pass_top} ({wick_excess_pass_top/max(poked_top,1)*100:.1f}% of poked)")
print(f"  wick_ratio >= 0.60: {wick_ratio_pass_top} ({wick_ratio_pass_top/max(poked_top,1)*100:.1f}% of poked)")
print(f"  All conditions (top): {all_conditions_top}")
print(f"\n--- BOTTOM (Long) ---")
print(f"  Poked below range_low: {poked_bottom} ({poked_bottom/max(range_bars,1)*100:.2f}%)")
print(f"  wick_excess <= 0.25*ATR: {wick_excess_pass_bottom} ({wick_excess_pass_bottom/max(poked_bottom,1)*100:.1f}% of poked)")
print(f"  wick_ratio >= 0.60: {wick_ratio_pass_bottom} ({wick_ratio_pass_bottom/max(poked_bottom,1)*100:.1f}% of poked)")
print(f"  All conditions (bottom): {all_conditions_bottom}")

if wick_ratios:
    wr = pd.Series(wick_ratios)
    print(f"\n--- Wick Stats ---")
    print(f"  wick_ratio mean: {wr.mean():.3f} median: {wr.median():.3f}")
    print(f"  >= 0.60: {(wr >= 0.60).sum()} ({(wr >= 0.60).mean()*100:.1f}%)")
    print(f"  >= 0.40: {(wr >= 0.40).sum()} ({(wr >= 0.40).mean()*100:.1f}%)")
if wick_excesses:
    we = pd.Series(wick_excesses)
    print(f"  wick_excess/ATR mean: {we.mean():.3f} median: {we.median():.3f}")
    print(f"  <= 0.25: {(we <= 0.25).sum()} ({(we <= 0.25).mean()*100:.1f}%)")
    print(f"  <= 0.50: {(we <= 0.50).sum()} ({(we <= 0.50).mean()*100:.1f}%)")

print(f"\n=== TOTAL ALL-CONDITIONS SIGNALS: {all_conditions_top + all_conditions_bottom} ===")
