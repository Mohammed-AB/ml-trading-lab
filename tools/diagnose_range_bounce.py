"""Range Bounce Diagnostic — Test new Model B concept on existing data.

Instead of requiring a single-candle failed breakout (rare on M1),
this concept looks for a multi-bar slowdown + reversal near range boundaries.

The idea:
1. Price approaches range_high or range_low (within 0.3*ATR)
2. Bars show deceleration (shrinking bodies or declining ATR)
3. A reversal candle closes away from the boundary
4. RSI confirms direction change (not extreme — just shifting)

This script tests how many signals this would produce,
and simulates approximate PnL to check if the concept has edge.

Usage:
    python diagnose_range_bounce.py
    python diagnose_range_bounce.py --pair USD_JPY
"""

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

from src.scalp_mode.engine.feature_engine import FeatureEngine
from src.scalp_mode.engine.regime_engine import RegimeEngine, Regime
from src.scalp_mode.utils.pip_utils import price_to_pips, pips_to_price


DATA_FILES = {
    "EUR_USD": "data/EUR_USD_M1_3m.csv",
    "USD_JPY": "data/USD_JPY_M1_3m.csv",
    "GBP_USD": "data/GBP_USD_M1_3m.csv",
}

REGIME_CONFIG = {
    "trend": {"ema_slope_thr": 0.15, "rsi_min": 52, "rsi_max": 78},
    "range": {"bb_width_thr": 0.004},
}

RANGE_WINDOW = 12  # M5 candles for range


def analyze_pair(pair: str, params: dict):
    path = DATA_FILES[pair]
    if not Path(path).exists():
        print(f"  ERROR: {path} not found")
        return

    print(f"\n{'='*70}")
    print(f"  RANGE BOUNCE DIAGNOSTIC: {pair}")
    print(f"  Params: {params}")
    print(f"{'='*70}")

    df = pd.read_csv(path)
    fe = FeatureEngine()
    re_eng = RegimeEngine(REGIME_CONFIG)

    # Resample M5
    groups = len(df) // 5
    trimmed = df.iloc[:groups * 5].copy()
    trimmed["g"] = np.repeat(range(groups), 5)
    m5 = trimmed.groupby("g").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).reset_index(drop=True)

    # Parameters
    approach_atr = params.get("approach_atr", 0.3)
    decel_bars = params.get("decel_bars", 3)
    decel_ratio = params.get("decel_ratio", 0.7)  # body shrinks to 70% or less
    rsi_shift = params.get("rsi_shift", 5)  # RSI moves 5+ points away from boundary
    spread_pips = params.get("spread", 0.3)
    sl_buffer_atr = params.get("sl_buffer_atr", 0.3)
    tp_type = params.get("tp_type", "mid_range")  # mid_range or fixed_R

    # Counters
    range_bars = 0
    near_top = 0
    near_bottom = 0
    decel_top = 0
    decel_bottom = 0
    reversal_top = 0
    reversal_bottom = 0
    valid_top = 0
    valid_bottom = 0

    # Simulated trades
    trades = []
    hourly_signals = defaultdict(int)

    print(f"  Analyzing {len(df)} M1 bars...")

    for i in range(max(100, RANGE_WINDOW * 5 + 10), len(df) - 10):
        if (i - 100) % 20000 == 0:
            print(f"    ...{i - 100} bars processed")

        # M1 indicators
        m1_chunk = df.iloc[max(0, i - 99):i + 1]
        if len(m1_chunk) < 50:
            continue
        ind_m1 = fe.compute(m1_chunk, "M1")
        atr = ind_m1.atr14
        if atr is None or atr <= 0:
            continue

        # M5 regime
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

        # Build range (exclude current M5)
        range_slice = m5.iloc[m5_idx - RANGE_WINDOW:m5_idx]
        if len(range_slice) < RANGE_WINDOW:
            continue
        range_high = float(range_slice["high"].max())
        range_low = float(range_slice["low"].min())
        mid_range = (range_high + range_low) / 2
        range_size = range_high - range_low
        if range_size < atr * 0.5:
            continue

        # Current bar
        bar = df.iloc[i]
        c = float(bar["close"])
        h = float(bar["high"])
        l = float(bar["low"])
        o = float(bar["open"])

        # RSI
        rsi = ind_m1.rsi14
        if rsi is None:
            continue

        # Get hour
        hour = None
        if "timestamp" in df.columns:
            try:
                hour = pd.Timestamp(df.iloc[i]["timestamp"]).hour
            except:
                pass

        # Previous bars for deceleration check
        if len(m1_chunk) < decel_bars + 2:
            continue
        recent = m1_chunk.iloc[-(decel_bars + 1):]
        bodies = [abs(float(r["close"]) - float(r["open"])) for _, r in recent.iterrows()]
        current_body = bodies[-1]
        prev_bodies = bodies[:-1]
        avg_prev_body = sum(prev_bodies) / len(prev_bodies) if prev_bodies else 1

        # Previous RSI (approximate from 2 bars ago)
        if len(m1_chunk) >= 16:
            prev_closes = m1_chunk["close"].astype(float)
            delta = prev_closes.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            rs_prev = avg_gain.iloc[-3] / avg_loss.iloc[-3] if avg_loss.iloc[-3] != 0 else 100
            rsi_prev = float(100 - (100 / (1 + rs_prev)))
        else:
            rsi_prev = rsi

        # ============================================================
        #  CHECK TOP BOUNCE (Short signal)
        # ============================================================
        dist_to_top = range_high - c
        if 0 < dist_to_top <= approach_atr * atr or (h >= range_high and c < range_high):
            near_top += 1

            # Deceleration: current body smaller than average of previous
            is_decel = current_body <= avg_prev_body * decel_ratio
            if is_decel:
                decel_top += 1

                # Reversal: close below open (bearish) + RSI dropping
                is_reversal = c < o and rsi < rsi_prev - rsi_shift
                if is_reversal:
                    reversal_top += 1

                    # Valid signal
                    valid_top += 1
                    if hour is not None:
                        hourly_signals[hour] += 1

                    # Simulate trade
                    entry = c
                    sl = range_high + sl_buffer_atr * atr
                    tp = mid_range
                    risk = abs(sl - entry)
                    reward = abs(entry - tp)

                    if risk > 0 and reward > 0:
                        # Look ahead for outcome
                        pnl = _simulate_trade(
                            df, i, "short", entry, sl, tp, pair, spread_pips)
                        trades.append({
                            "direction": "short",
                            "side": "top",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "risk_pips": price_to_pips(risk, pair),
                            "reward_pips": price_to_pips(reward, pair),
                            "pnl_pips": pnl,
                            "hour": hour,
                            "rsi": rsi,
                        })

        # ============================================================
        #  CHECK BOTTOM BOUNCE (Long signal)
        # ============================================================
        dist_to_bottom = c - range_low
        if 0 < dist_to_bottom <= approach_atr * atr or (l <= range_low and c > range_low):
            near_bottom += 1

            is_decel = current_body <= avg_prev_body * decel_ratio
            if is_decel:
                decel_bottom += 1

                is_reversal = c > o and rsi > rsi_prev + rsi_shift
                if is_reversal:
                    reversal_bottom += 1

                    valid_bottom += 1
                    if hour is not None:
                        hourly_signals[hour] += 1

                    entry = c
                    sl = range_low - sl_buffer_atr * atr
                    tp = mid_range
                    risk = abs(entry - sl)
                    reward = abs(tp - entry)

                    if risk > 0 and reward > 0:
                        pnl = _simulate_trade(
                            df, i, "long", entry, sl, tp, pair, spread_pips)
                        trades.append({
                            "direction": "long",
                            "side": "bottom",
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "risk_pips": price_to_pips(risk, pair),
                            "reward_pips": price_to_pips(reward, pair),
                            "pnl_pips": pnl,
                            "hour": hour,
                            "rsi": rsi,
                        })

    # ============================================================
    #  REPORT
    # ============================================================
    total_valid = valid_top + valid_bottom

    print(f"\n--- Funnel ---")
    print(f"  Range bars:        {range_bars}")
    print(f"  Near top:          {near_top} ({near_top/max(range_bars,1)*100:.2f}%)")
    print(f"  Near bottom:       {near_bottom} ({near_bottom/max(range_bars,1)*100:.2f}%)")
    print(f"  Decel top:         {decel_top} ({decel_top/max(near_top,1)*100:.1f}% of near)")
    print(f"  Decel bottom:      {decel_bottom} ({decel_bottom/max(near_bottom,1)*100:.1f}% of near)")
    print(f"  Reversal top:      {reversal_top}")
    print(f"  Reversal bottom:   {reversal_bottom}")
    print(f"  VALID signals:     {total_valid} (top={valid_top}, bottom={valid_bottom})")

    days = max(range_bars // 210, 1)
    print(f"  Est. signals/day:  {total_valid / max(days, 1):.1f}")

    if trades:
        tdf = pd.DataFrame(trades)
        wins = tdf[tdf["pnl_pips"] > 0]
        losses = tdf[tdf["pnl_pips"] <= 0]
        total_pnl = tdf["pnl_pips"].sum()
        avg_pnl = tdf["pnl_pips"].mean()
        wr = len(wins) / len(tdf) if len(tdf) > 0 else 0
        avg_win = wins["pnl_pips"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl_pips"].mean() if len(losses) > 0 else 0
        gross_profit = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        print(f"\n--- Simulated Performance ---")
        print(f"  Trades:     {len(tdf)}")
        print(f"  Win rate:   {wr:.0%}")
        print(f"  Total PnL:  {total_pnl:+.1f} pips")
        print(f"  Avg PnL:    {avg_pnl:+.2f} pips")
        print(f"  Avg winner: {avg_win:+.2f} pips")
        print(f"  Avg loser:  {avg_loss:+.2f} pips")
        print(f"  PF:         {pf:.2f}")
        print(f"  Avg risk:   {tdf['risk_pips'].mean():.2f} pips")
        print(f"  Avg reward: {tdf['reward_pips'].mean():.2f} pips")

        # By direction
        for side in ["top", "bottom"]:
            sub = tdf[tdf["side"] == side]
            if len(sub) > 0:
                s_wr = (sub["pnl_pips"] > 0).mean()
                s_pnl = sub["pnl_pips"].sum()
                print(f"  {side:8s}: {len(sub):3d} trades | WR={s_wr:.0%} | PnL={s_pnl:+.1f}")

        # Hourly
        if hourly_signals:
            print(f"\n--- Hourly Distribution ---")
            for h in range(24):
                count = hourly_signals.get(h, 0)
                if count > 0:
                    bar = "#" * min(count, 50)
                    print(f"  {h:2d}:00  {count:3d}  {bar}")
    else:
        print(f"\n  No valid signals found.")

    # ============================================================
    #  PARAMETER SENSITIVITY
    # ============================================================
    print(f"\n--- What-if: Loosening parameters ---")
    for name, alt_params in [
        ("approach_atr=0.5", {**params, "approach_atr": 0.5}),
        ("decel_ratio=0.85", {**params, "decel_ratio": 0.85}),
        ("rsi_shift=3", {**params, "rsi_shift": 3}),
        ("all loosened", {**params, "approach_atr": 0.5, "decel_ratio": 0.85, "rsi_shift": 3}),
    ]:
        count = _quick_count(df, m5, fe, re_eng, pair, alt_params)
        print(f"  {name:25s}: ~{count} signals")

    return total_valid, trades


def _simulate_trade(df, entry_bar, direction, entry, sl, tp, pair,
                    spread_pips, max_hold=6):
    """Simple forward simulation: look ahead up to max_hold bars."""
    entry_cost = pips_to_price(spread_pips / 2, pair)  # half spread
    if direction == "long":
        entry += entry_cost
    else:
        entry -= entry_cost

    for j in range(1, min(max_hold + 1, len(df) - entry_bar)):
        bar = df.iloc[entry_bar + j]
        h = float(bar["high"])
        l = float(bar["low"])

        if direction == "long":
            if l <= sl:
                return price_to_pips(sl - entry, pair)
            if h >= tp:
                return price_to_pips(tp - entry, pair)
        else:
            if h >= sl:
                return price_to_pips(entry - sl, pair)
            if l <= tp:
                return price_to_pips(entry - tp, pair)

    # Time stop: close at last bar
    last_close = float(df.iloc[min(entry_bar + max_hold, len(df) - 1)]["close"])
    if direction == "long":
        return price_to_pips(last_close - entry, pair)
    else:
        return price_to_pips(entry - last_close, pair)


def _quick_count(df, m5, fe, re_eng, pair, params):
    """Quick signal count without full simulation."""
    approach_atr = params.get("approach_atr", 0.3)
    decel_bars = params.get("decel_bars", 3)
    decel_ratio = params.get("decel_ratio", 0.7)
    rsi_shift = params.get("rsi_shift", 5)
    count = 0

    for i in range(max(100, RANGE_WINDOW * 5 + 10), len(df), 3):  # sample every 3rd bar
        m1_chunk = df.iloc[max(0, i - 99):i + 1]
        if len(m1_chunk) < 50:
            continue
        ind_m1 = fe.compute(m1_chunk, "M1")
        atr = ind_m1.atr14
        if atr is None or atr <= 0:
            continue

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

        range_slice = m5.iloc[m5_idx - RANGE_WINDOW:m5_idx]
        if len(range_slice) < RANGE_WINDOW:
            continue
        range_high = float(range_slice["high"].max())
        range_low = float(range_slice["low"].min())

        bar = df.iloc[i]
        c = float(bar["close"])
        h = float(bar["high"])
        l = float(bar["low"])
        o = float(bar["open"])
        rsi = ind_m1.rsi14
        if rsi is None:
            continue

        recent = m1_chunk.iloc[-(decel_bars + 1):]
        bodies = [abs(float(r["close"]) - float(r["open"])) for _, r in recent.iterrows()]
        current_body = bodies[-1]
        avg_prev = sum(bodies[:-1]) / max(len(bodies) - 1, 1)

        # Top
        dist_top = range_high - c
        if (0 < dist_top <= approach_atr * atr) or (h >= range_high and c < range_high):
            if current_body <= avg_prev * decel_ratio and c < o:
                count += 1

        # Bottom
        dist_bot = c - range_low
        if (0 < dist_bot <= approach_atr * atr) or (l <= range_low and c > range_low):
            if current_body <= avg_prev * decel_ratio and c > o:
                count += 1

    return count * 3  # compensate for sampling every 3rd bar


def main():
    parser = argparse.ArgumentParser(description="Range Bounce Diagnostic")
    parser.add_argument("--pair", type=str, default=None)
    args = parser.parse_args()

    params = {
        "approach_atr": 0.5,
        "decel_bars": 3,
        "decel_ratio": 0.85,
        "rsi_shift": 3,
        "spread": 0.3,
        "sl_buffer_atr": 0.3,
        "tp_type": "mid_range",
    }

    pairs = [args.pair] if args.pair else list(DATA_FILES.keys())
    all_results = {}

    for pair in pairs:
        total, trades = analyze_pair(pair, params) or (0, [])
        all_results[pair] = (total, trades)

    print(f"\n{'='*70}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*70}")
    grand_total = sum(v[0] for v in all_results.values())
    grand_trades = []
    for pair, (total, trades) in all_results.items():
        grand_trades.extend(trades)
        print(f"  {pair}: {total} signals")
    print(f"  TOTAL: {grand_total} signals across all pairs")

    if grand_trades:
        gt = pd.DataFrame(grand_trades)
        total_pnl = gt["pnl_pips"].sum()
        wr = (gt["pnl_pips"] > 0).mean()
        print(f"  Combined WR: {wr:.0%}")
        print(f"  Combined PnL: {total_pnl:+.1f} pips")

    if grand_total >= 200:
        print(f"\n  VERDICT: Concept looks viable ({grand_total} signals). Worth building.")
    elif grand_total >= 50:
        print(f"\n  VERDICT: Moderate signal count. May need loosening or combination with Model A.")
    else:
        print(f"\n  VERDICT: Too few signals ({grand_total}). Concept needs rethinking.")


if __name__ == "__main__":
    main()
