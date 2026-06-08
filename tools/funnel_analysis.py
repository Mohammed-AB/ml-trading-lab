"""Pipeline Funnel Analysis — Where exactly do opportunities die?

Tracks every M1 bar through the full pipeline and reports:
1. How many bars survive each step
2. Which step kills the most signals
3. Model A vs Model B breakdown
4. Hourly distribution of opportunities
5. Specific parameter bottlenecks with suggested loosening

This answers: "What must change to double/triple trade count without killing quality?"

Usage:
    python funnel_analysis.py
    python funnel_analysis.py --pair USD_JPY
"""

import sys
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from src.scalp_mode.engine.feature_engine import FeatureEngine
from src.scalp_mode.engine.regime_engine import RegimeEngine, Regime
from src.scalp_mode.engine.model_a import ModelATrigger, TriggerPhase
from src.scalp_mode.engine.model_b import ModelBTrigger


# ============================================================
#  CONFIG (current calibrated values)
# ============================================================
REGIME_CONFIG = {
    "trend": {"ema_slope_thr": 0.15, "rsi_min": 52, "rsi_max": 78},
    "range": {"bb_width_thr": 0.004},
}

MODEL_A_CONFIG = {
    "compression_N": 8, "compression_atr_mult": 2.0,
    "breakout_buffer_atr": 0.10, "retest_timeout": 3,
    "retest_tolerance_atr": 0.15, "body_ratio_min": 0.55,
    "rsi_min_long": 55, "sl_atr": 0.8, "tp_R": 1.7,
    "time_stop_min": 6, "sl_move_threshold_R": 0.8,
    "sl_move_target_R": -0.1, "sl_move_window_min": [2, 4],
}

MODEL_B_CONFIG = {
    "enabled": True, "range_window_M5": 12,
    "wick_ratio_min": 0.40, "wick_excess_atr": 0.25,
    "stop_spread_buffer_mult": 2.0, "stop_atr_buffer": 0.15,
    "rsi_overbought": 65, "rsi_reversal": 60,
    "rsi_oversold": 35, "rsi_reversal_up": 40,
}

DATA_FILES = {
    "EUR_USD": "data/EUR_USD_M1_3m.csv",
    "USD_JPY": "data/USD_JPY_M1_3m.csv",
    "GBP_USD": "data/GBP_USD_M1_3m.csv",
}


def analyze_pair(pair: str):
    """Full funnel analysis for one pair."""
    path = DATA_FILES[pair]
    if not Path(path).exists():
        print(f"  ERROR: {path} not found")
        return

    print(f"\n{'='*70}")
    print(f"  FUNNEL ANALYSIS: {pair}")
    print(f"{'='*70}")

    df = pd.read_csv(path)
    fe = FeatureEngine()
    re_eng = RegimeEngine(REGIME_CONFIG)
    trigger_a = ModelATrigger(MODEL_A_CONFIG)
    trigger_b = ModelBTrigger(MODEL_B_CONFIG)

    # Resample M5
    groups = len(df) // 5
    trimmed = df.iloc[:groups * 5].copy()
    trimmed["g"] = np.repeat(range(groups), 5)
    m5 = trimmed.groupby("g").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).reset_index(drop=True)

    # Counters
    total_bars = 0
    warmup_skip = 0

    # Step 4: Regime
    regime_counts = defaultdict(int)

    # Model A funnel
    a_eligible = 0
    a_phases = defaultdict(int)
    a_valid = 0
    a_near_miss = defaultdict(int)  # almost passed but one condition failed

    # Model A detailed: which sub-condition fails?
    a_no_compression_detail = defaultdict(int)
    a_compression_pass = 0
    a_breakout_pass = 0
    a_momentum_pass = 0

    # Model B funnel
    b_eligible = 0
    b_phases = defaultdict(int)
    b_valid = 0
    b_poked_top = 0
    b_poked_bottom = 0
    b_wick_ratio_pass = 0
    b_wick_excess_pass = 0
    b_rsi_reversal_pass = 0

    # Hourly distribution
    hourly_regime = defaultdict(lambda: defaultdict(int))
    hourly_a_valid = defaultdict(int)
    hourly_b_valid = defaultdict(int)

    # Detailed compression analysis
    compression_ratios = []

    # Detailed wick analysis
    wick_ratios_top = []
    wick_ratios_bottom = []

    print(f"  Analyzing {len(df)} M1 bars...")

    for i in range(60, len(df)):
        total_bars += 1
        if total_bars % 20000 == 0:
            print(f"    ...{total_bars} bars processed")

        # Get hour for distribution
        hour = None
        if "timestamp" in df.columns:
            try:
                ts = pd.Timestamp(df.iloc[i]["timestamp"])
                hour = ts.hour
            except:
                pass

        # M1 indicators
        m1_chunk = df.iloc[max(0, i - 99):i + 1]
        if len(m1_chunk) < 50:
            warmup_skip += 1
            continue
        ind_m1 = fe.compute(m1_chunk, "M1")

        if ind_m1.atr14 is None or ind_m1.atr14 <= 0:
            warmup_skip += 1
            continue

        # M5 indicators
        m5_idx = i // 5
        if m5_idx < 21:
            warmup_skip += 1
            continue
        m5_chunk = m5.iloc[max(0, m5_idx - 49):m5_idx + 1]
        if len(m5_chunk) < 20:
            warmup_skip += 1
            continue
        ind_m5 = fe.compute(m5_chunk, "M5")
        close_m5 = float(m5_chunk.iloc[-1]["close"])

        # Step 4: Regime
        regime_result = re_eng.evaluate(ind_m5, close_m5)
        regime_name = regime_result.regime.value
        regime_counts[regime_name] += 1
        if hour is not None:
            hourly_regime[hour][regime_name] += 1

        # ============ Model A (Trend) ============
        if regime_result.regime in (Regime.TREND_UP, Regime.TREND_DOWN):
            a_eligible += 1
            atr = ind_m1.atr14

            # Detailed compression check
            N = MODEL_A_CONFIG["compression_N"]
            if len(m1_chunk) >= N:
                lookback = m1_chunk.iloc[-N:]
                range_n = float(lookback["high"].max()) - float(lookback["low"].min())
                ratio = range_n / atr if atr > 0 else 999
                compression_ratios.append(ratio)

                comp_limit = MODEL_A_CONFIG["compression_atr_mult"]
                if ratio <= comp_limit:
                    a_compression_pass += 1

                    # Check breakout
                    bar = m1_chunk.iloc[-1]
                    hh = float(lookback["high"].max())
                    buf = MODEL_A_CONFIG["breakout_buffer_atr"] * atr
                    if float(bar["close"]) > hh + buf:
                        a_breakout_pass += 1

            # Full trigger
            sig = trigger_a.evaluate(m1_chunk, ind_m1, regime_result.regime, pair)
            phase = sig.phase.value
            a_phases[phase] += 1
            if sig.phase == TriggerPhase.VALID:
                a_valid += 1
                if hour is not None:
                    hourly_a_valid[hour] += 1

        # ============ Model B (Range) ============
        elif regime_result.regime == Regime.RANGE:
            b_eligible += 1
            atr = ind_m1.atr14

            # Build range (FIXED: exclude current M5)
            rw = MODEL_B_CONFIG["range_window_M5"]
            if m5_idx >= rw + 1:
                range_slice = m5.iloc[m5_idx - rw:m5_idx]
                if len(range_slice) >= rw:
                    range_high = float(range_slice["high"].max())
                    range_low = float(range_slice["low"].min())

                    bar = df.iloc[i]
                    h = float(bar["high"])
                    l = float(bar["low"])
                    o = float(bar["open"])
                    c = float(bar["close"])
                    cr = h - l

                    if cr > 0:
                        # Top poke
                        if h > range_high:
                            b_poked_top += 1
                            upper_wick = h - max(o, c)
                            wr = upper_wick / cr
                            we = (h - range_high) / atr if atr > 0 else 999
                            wick_ratios_top.append(wr)

                            if we <= MODEL_B_CONFIG["wick_excess_atr"]:
                                b_wick_excess_pass += 1
                            if wr >= MODEL_B_CONFIG["wick_ratio_min"]:
                                b_wick_ratio_pass += 1

                        # Bottom poke
                        if l < range_low:
                            b_poked_bottom += 1
                            lower_wick = min(o, c) - l
                            wr = lower_wick / cr
                            wick_ratios_bottom.append(wr)

                            if (range_low - l) / atr <= MODEL_B_CONFIG["wick_excess_atr"]:
                                b_wick_excess_pass += 1
                            if wr >= MODEL_B_CONFIG["wick_ratio_min"]:
                                b_wick_ratio_pass += 1

            # Full trigger
            m5_for_b = m5.iloc[max(0, m5_idx - 49):m5_idx + 1]
            sig = trigger_b.evaluate(
                m1_chunk, m5_for_b, ind_m1, ind_m5,
                regime_result.regime, pair, spread_pips=0.3)
            phase = sig.phase.value
            b_phases[phase] += 1
            if sig.phase == TriggerPhase.VALID:
                b_valid += 1
                if hour is not None:
                    hourly_b_valid[hour] += 1

    # ============================================================
    #  REPORT
    # ============================================================
    analyzed = total_bars - warmup_skip
    print(f"\n{'='*70}")
    print(f"  RESULTS: {pair}")
    print(f"{'='*70}")

    print(f"\n  Total M1 bars: {total_bars}")
    print(f"  Analyzed (post-warmup): {analyzed}")

    # --- Regime ---
    print(f"\n--- Step 4: Regime Distribution ---")
    for k, v in sorted(regime_counts.items()):
        print(f"  {k:12s}: {v:6d} ({v/max(analyzed,1)*100:5.1f}%)")

    # --- Model A Funnel ---
    print(f"\n--- Model A Funnel (Trend bars: {a_eligible}) ---")
    print(f"  Compression pass (ratio <= {MODEL_A_CONFIG['compression_atr_mult']}): "
          f"{a_compression_pass} ({a_compression_pass/max(a_eligible,1)*100:.1f}%)")
    print(f"  Breakout pass: {a_breakout_pass} ({a_breakout_pass/max(a_compression_pass,1)*100:.1f}% of compression pass)")
    print(f"  Trigger phases:")
    for k, v in sorted(a_phases.items(), key=lambda x: -x[1]):
        print(f"    {k:25s}: {v:5d} ({v/max(a_eligible,1)*100:.2f}%)")
    print(f"  VALID signals: {a_valid}")

    if compression_ratios:
        cr = pd.Series(compression_ratios)
        print(f"\n  Compression ratio stats:")
        print(f"    Mean: {cr.mean():.2f}  Median: {cr.median():.2f}")
        for thr in [1.5, 2.0, 2.5, 3.0, 4.0]:
            pct = (cr <= thr).mean() * 100
            count = (cr <= thr).sum()
            print(f"    <= {thr}: {count:5d} ({pct:5.1f}%) → est. {int(count * a_valid / max((cr <= MODEL_A_CONFIG['compression_atr_mult']).sum(), 1))} valid signals")

    # --- Model B Funnel ---
    print(f"\n--- Model B Funnel (Range bars: {b_eligible}) ---")
    total_pokes = b_poked_top + b_poked_bottom
    print(f"  Poked range boundary: {total_pokes} ({total_pokes/max(b_eligible,1)*100:.2f}%)")
    print(f"    Top: {b_poked_top}  Bottom: {b_poked_bottom}")
    print(f"  Wick excess pass: {b_wick_excess_pass} ({b_wick_excess_pass/max(total_pokes,1)*100:.1f}% of poked)")
    print(f"  Wick ratio pass: {b_wick_ratio_pass} ({b_wick_ratio_pass/max(total_pokes,1)*100:.1f}% of poked)")
    print(f"  Trigger phases:")
    for k, v in sorted(b_phases.items(), key=lambda x: -x[1]):
        print(f"    {k:25s}: {v:5d} ({v/max(b_eligible,1)*100:.2f}%)")
    print(f"  VALID signals: {b_valid}")

    if wick_ratios_top:
        wr_top = pd.Series(wick_ratios_top)
        print(f"\n  Wick ratio stats (top pokes):")
        print(f"    Mean: {wr_top.mean():.3f}  Median: {wr_top.median():.3f}")
        for thr in [0.30, 0.40, 0.50, 0.60]:
            pct = (wr_top >= thr).mean() * 100
            print(f"    >= {thr}: {(wr_top >= thr).sum():5d} ({pct:5.1f}%)")

    if wick_ratios_bottom:
        wr_bot = pd.Series(wick_ratios_bottom)
        print(f"\n  Wick ratio stats (bottom pokes):")
        print(f"    Mean: {wr_bot.mean():.3f}  Median: {wr_bot.median():.3f}")
        for thr in [0.30, 0.40, 0.50, 0.60]:
            pct = (wr_bot >= thr).mean() * 100
            print(f"    >= {thr}: {(wr_bot >= thr).sum():5d} ({pct:5.1f}%)")

    # --- Hourly Distribution ---
    if hourly_a_valid or hourly_b_valid:
        print(f"\n--- Hourly Signal Distribution ---")
        print(f"  {'Hour':>4s}  {'A signals':>10s}  {'B signals':>10s}  {'Regime':>20s}")
        for h in range(24):
            a_h = hourly_a_valid.get(h, 0)
            b_h = hourly_b_valid.get(h, 0)
            regime_h = hourly_regime.get(h, {})
            dominant = max(regime_h, key=regime_h.get) if regime_h else "—"
            total_h = sum(regime_h.values())
            print(f"  {h:4d}  {a_h:10d}  {b_h:10d}  {dominant:>12s} ({total_h})")

    # --- Recommendations ---
    print(f"\n{'='*70}")
    print(f"  RECOMMENDATIONS: {pair}")
    print(f"{'='*70}")

    # Model A recommendations
    if a_valid < 100:
        print(f"\n  [Model A] Low signal count ({a_valid}). Options:")
        if compression_ratios:
            cr = pd.Series(compression_ratios)
            current_pass = (cr <= MODEL_A_CONFIG["compression_atr_mult"]).mean() * 100
            next_pass = (cr <= MODEL_A_CONFIG["compression_atr_mult"] + 0.5).mean() * 100
            if next_pass > current_pass * 1.3:
                print(f"    → Raise compression_atr_mult from {MODEL_A_CONFIG['compression_atr_mult']} to "
                      f"{MODEL_A_CONFIG['compression_atr_mult'] + 0.5}: "
                      f"pass rate {current_pass:.0f}% → {next_pass:.0f}%")
            else:
                print(f"    → compression_atr_mult already reasonable ({current_pass:.0f}% pass rate)")

    # Model B recommendations
    if b_valid < 20:
        print(f"\n  [Model B] Very low signal count ({b_valid}). Root causes:")
        if total_pokes == 0:
            print(f"    → CRITICAL: Zero range boundary pokes. Check range_slice excludes current M5 candle.")
        elif total_pokes < b_eligible * 0.02:
            print(f"    → Only {total_pokes/b_eligible*100:.1f}% of Range bars poke boundary. "
                  f"Range is too wide for M1. Consider range_window_M5 = 6-8 instead of {MODEL_B_CONFIG['range_window_M5']}.")
        if b_wick_ratio_pass < total_pokes * 0.2:
            print(f"    → wick_ratio_min={MODEL_B_CONFIG['wick_ratio_min']} too strict. "
                  f"Only {b_wick_ratio_pass}/{total_pokes} pokes pass. "
                  f"Consider 0.30.")
        if b_wick_excess_pass < total_pokes * 0.3:
            print(f"    → wick_excess_atr={MODEL_B_CONFIG['wick_excess_atr']} too strict. "
                  f"Only {b_wick_excess_pass}/{total_pokes} pokes pass. "
                  f"Consider 0.50.")

    # Overall
    total_valid = a_valid + b_valid
    days = max(analyzed // (210 * 5), 1)  # rough estimate
    trades_per_day = total_valid / max(days, 1)
    print(f"\n  [Overall]")
    print(f"    Total valid signals: {total_valid} (A={a_valid}, B={b_valid})")
    print(f"    Est. trades/day: {trades_per_day:.1f}")
    if trades_per_day < 1:
        print(f"    → WARNING: <1 trade/day. For scalping, target 5-10/day minimum.")
        print(f"    → Priority: Fix Model B to exploit {b_eligible/max(analyzed,1)*100:.0f}% Range bars")
    elif trades_per_day < 5:
        print(f"    → MODERATE: {trades_per_day:.1f}/day is selective. Acceptable for V2, not true scalping.")
    else:
        print(f"    → GOOD: {trades_per_day:.1f}/day is reasonable scalping frequency.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Funnel Analysis")
    parser.add_argument("--pair", type=str, default=None, help="Single pair (default: all)")
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else list(DATA_FILES.keys())

    for pair in pairs:
        analyze_pair(pair)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Run test_scenarios.py for parameter optimization after fixing Model B.")
    print(f"  Key insight: Model A works but is inherently low-frequency (~23% of market).")
    print(f"  Model B must work to reach scalping-level frequency.")


if __name__ == "__main__":
    main()
