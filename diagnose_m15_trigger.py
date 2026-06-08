"""M15 Trigger Diagnostic — Test the system with M15 as trigger timeframe.

Current system: M1 trigger + M5 context
M5 test: M5 trigger + M15 context
This test: M15 trigger + H1 context

Bigger moves, even lower cost impact, but longer hold times (~1-2 hours).

Uses existing M1 data → resamples to M15 and H1.

Usage:
    python diagnose_m15_trigger.py --pair EUR_USD
    python diagnose_m15_trigger.py  (all pairs)
"""

import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

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

# Regime on H1 (context)
REGIME_CONFIG = {
    "trend": {"ema_slope_thr": 0.15, "rsi_min": 52, "rsi_max": 78},
    "range": {"bb_width_thr": 0.004},
}

# Model A params (adapted for M15)
MODEL_A = {
    "compression_N": 8,
    "compression_atr_mult": 2.0,
    "breakout_buffer_atr": 0.10,
    "body_ratio_min": 0.55,
    "rsi_min_long": 55,
    "sl_atr": 0.8,
    "tp_R": 1.7,
    "retest_timeout": 3,
    "retest_tolerance_atr": 0.15,
}

# Model B params (Range bounce on M15 — range from H1)
MODEL_B = {
    "range_window": 12,  # 12 H1 candles = 12 hours
    "wick_ratio_min": 0.40,
    "wick_excess_atr": 0.50,
    "rsi_overbought": 65,
    "rsi_reversal_down": 60,
    "rsi_oversold": 35,
    "rsi_reversal_up": 40,
    "sl_buffer_atr": 0.3,
}

SPREAD_PIPS = 0.3
SLIPPAGE_PIPS = 0.1
MAX_HOLD_BARS = 6  # 6 M15 bars = 90 minutes


def resample(df_m1, n):
    """Resample M1 to Mn candles."""
    groups = len(df_m1) // n
    if groups == 0:
        return df_m1.copy()
    trimmed = df_m1.iloc[:groups * n].copy()
    trimmed["g"] = np.repeat(range(groups), n)
    result = trimmed.groupby("g").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).reset_index(drop=True)

    # Carry timestamps if available
    if "timestamp" in df_m1.columns:
        ts = trimmed.groupby("g")["timestamp"].first().reset_index(drop=True)
        result["timestamp"] = ts

    return result


def analyze_pair(pair: str):
    path = DATA_FILES[pair]
    if not Path(path).exists():
        print(f"  ERROR: {path} not found")
        return

    print(f"\n{'='*70}")
    print(f"  M15 TRIGGER DIAGNOSTIC: {pair}")
    print(f"{'='*70}")

    df_m1 = pd.read_csv(path)
    print(f"  M1 candles: {len(df_m1)}")

    # Resample
    m15 = resample(df_m1, 15)
    h1 = resample(df_m1, 60)
    print(f"  M15 candles: {len(m15)}")
    print(f"  H1 candles: {len(h1)}")

    fe = FeatureEngine()
    re_eng = RegimeEngine(REGIME_CONFIG)

    # Counters
    total_bars = 0
    regime_counts = defaultdict(int)

    # Model A
    a_eligible = 0
    a_compression_pass = 0
    a_breakout_pass = 0
    a_valid = 0
    compression_ratios = []

    # Model B
    b_eligible = 0
    b_poked_top = 0
    b_poked_bottom = 0
    b_wick_pass = 0
    b_rsi_pass = 0
    b_valid_top = 0
    b_valid_bottom = 0

    # Trades
    trades_a = []
    trades_b = []
    hourly = defaultdict(lambda: {"A": 0, "B": 0})

    print(f"  Analyzing {len(m15)} M15 bars...\n")

    for i in range(60, len(m15) - MAX_HOLD_BARS):
        total_bars += 1
        if total_bars % 1000 == 0:
            print(f"    ...{total_bars} M15 bars processed")

        # Hour
        hour = None
        if "timestamp" in m15.columns:
            try:
                hour = pd.Timestamp(m15.iloc[i]["timestamp"]).hour
            except:
                pass

        # M15 indicators (trigger)
        trig_chunk = m15.iloc[max(0, i - 99):i + 1]
        if len(trig_chunk) < 30:
            continue
        ind_trig = fe.compute(trig_chunk, "M15")
        atr = ind_trig.atr14
        if atr is None or atr <= 0:
            continue

        # H1 indicators (context/regime)
        h1_idx = i // 4
        if h1_idx < 21:
            continue
        h1_chunk = h1.iloc[max(0, h1_idx - 49):h1_idx + 1]
        if len(h1_chunk) < 20:
            continue
        ind_h1 = fe.compute(h1_chunk, "H1")
        close_h1 = float(h1_chunk.iloc[-1]["close"])

        # Regime (on H1)
        regime_result = re_eng.evaluate(ind_h1, close_h1)
        regime_name = regime_result.regime.value
        regime_counts[regime_name] += 1

        bar = m15.iloc[i]
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        cr = h - l
        if cr <= 0:
            continue

        # ===========================================================
        #  MODEL A: Compression + Breakout on M15 (Trend only)
        # ===========================================================
        if regime_result.regime in (Regime.TREND_UP, Regime.TREND_DOWN):
            a_eligible += 1
            N = MODEL_A["compression_N"]

            if len(trig_chunk) >= N + 1:
                lookback = trig_chunk.iloc[-(N + 1):-1]
                range_n = float(lookback["high"].max()) - float(lookback["low"].min())
                ratio = range_n / atr if atr > 0 else 999
                compression_ratios.append(ratio)

                if ratio <= MODEL_A["compression_atr_mult"]:
                    a_compression_pass += 1
                    hh = float(lookback["high"].max())
                    ll = float(lookback["low"].min())
                    buf = MODEL_A["breakout_buffer_atr"] * atr
                    body = abs(c - o)
                    body_ratio = body / cr if cr > 0 else 0

                    is_long = (regime_result.regime == Regime.TREND_UP and
                               c > hh + buf and body_ratio >= MODEL_A["body_ratio_min"])
                    is_short = (regime_result.regime == Regime.TREND_DOWN and
                                c < ll - buf and body_ratio >= MODEL_A["body_ratio_min"])

                    if is_long or is_short:
                        a_breakout_pass += 1

                        # Check RSI + MACD momentum
                        rsi = ind_trig.rsi14
                        macd = ind_trig.macd_hist
                        has_momentum = False
                        if is_long and rsi and rsi >= MODEL_A["rsi_min_long"] and macd and macd > 0:
                            has_momentum = True
                        if is_short and rsi and rsi <= (100 - MODEL_A["rsi_min_long"]) and macd and macd < 0:
                            has_momentum = True

                        if has_momentum:
                            a_valid += 1
                            direction = "long" if is_long else "short"

                            # Simulate
                            entry = c
                            risk = MODEL_A["sl_atr"] * atr
                            if direction == "long":
                                sl = entry - risk
                                tp = entry + risk * MODEL_A["tp_R"]
                            else:
                                sl = entry + risk
                                tp = entry - risk * MODEL_A["tp_R"]

                            pnl = _simulate(m15, i, direction, entry, sl, tp,
                                            pair, SPREAD_PIPS, MAX_HOLD_BARS)
                            trades_a.append({
                                "model": "A", "direction": direction,
                                "pnl_pips": pnl,
                                "risk_pips": price_to_pips(risk, pair),
                                "hour": hour,
                            })
                            if hour is not None:
                                hourly[hour]["A"] += 1

        # ===========================================================
        #  MODEL B: Range Bounce on M15 (Range only)
        # ===========================================================
        elif regime_result.regime == Regime.RANGE:
            b_eligible += 1
            rw = MODEL_B["range_window"]

            # Range from H1 (exclude current)
            if h1_idx >= rw + 1:
                range_slice = h1.iloc[h1_idx - rw:h1_idx]
                if len(range_slice) >= rw:
                    range_high = float(range_slice["high"].max())
                    range_low = float(range_slice["low"].min())
                    mid_range = (range_high + range_low) / 2
                    range_size = range_high - range_low

                    if range_size < atr * 0.5:
                        continue

                    rsi = ind_trig.rsi14
                    if rsi is None:
                        continue

                    # Previous RSI
                    if len(trig_chunk) >= 16:
                        prev_closes = trig_chunk["close"].astype(float)
                        delta = prev_closes.diff()
                        gain = delta.where(delta > 0, 0.0)
                        loss_s = (-delta).where(delta < 0, 0.0)
                        ag = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                        al = loss_s.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                        rs_p = ag.iloc[-2] / al.iloc[-2] if al.iloc[-2] != 0 else 100
                        rsi_prev = float(100 - (100 / (1 + rs_p)))
                    else:
                        rsi_prev = rsi

                    # --- TOP: Failed breakout / bounce ---
                    upper_wick = h - max(o, c)
                    wick_ratio_top = upper_wick / cr if cr > 0 else 0

                    if h > range_high and c < range_high:
                        b_poked_top += 1
                        wick_excess = (h - range_high) / atr

                        if wick_ratio_top >= MODEL_B["wick_ratio_min"]:
                            b_wick_pass += 1
                            if wick_excess <= MODEL_B["wick_excess_atr"]:
                                # RSI reversal
                                if (rsi_prev >= MODEL_B["rsi_overbought"]
                                        and rsi < MODEL_B["rsi_reversal_down"]):
                                    b_rsi_pass += 1
                                    b_valid_top += 1

                                    sl = h + MODEL_B["sl_buffer_atr"] * atr
                                    tp = mid_range
                                    risk = abs(sl - c)
                                    pnl = _simulate(m15, i, "short", c, sl, tp,
                                                    pair, SPREAD_PIPS, MAX_HOLD_BARS)
                                    trades_b.append({
                                        "model": "B", "direction": "short",
                                        "side": "top", "pnl_pips": pnl,
                                        "risk_pips": price_to_pips(risk, pair),
                                        "hour": hour,
                                    })
                                    if hour is not None:
                                        hourly[hour]["B"] += 1

                                # Relaxed: RSI just dropping (no extreme needed)
                                elif rsi < rsi_prev - 3:
                                    b_valid_top += 1
                                    sl = h + MODEL_B["sl_buffer_atr"] * atr
                                    tp = mid_range
                                    risk = abs(sl - c)
                                    pnl = _simulate(m15, i, "short", c, sl, tp,
                                                    pair, SPREAD_PIPS, MAX_HOLD_BARS)
                                    trades_b.append({
                                        "model": "B", "direction": "short",
                                        "side": "top", "pnl_pips": pnl,
                                        "risk_pips": price_to_pips(risk, pair),
                                        "hour": hour,
                                    })
                                    if hour is not None:
                                        hourly[hour]["B"] += 1

                    # --- BOTTOM: Failed breakout / bounce ---
                    lower_wick = min(o, c) - l
                    wick_ratio_bot = lower_wick / cr if cr > 0 else 0

                    if l < range_low and c > range_low:
                        b_poked_bottom += 1
                        wick_excess = (range_low - l) / atr

                        if wick_ratio_bot >= MODEL_B["wick_ratio_min"]:
                            b_wick_pass += 1
                            if wick_excess <= MODEL_B["wick_excess_atr"]:
                                if (rsi_prev <= MODEL_B["rsi_oversold"]
                                        and rsi > MODEL_B["rsi_reversal_up"]):
                                    b_rsi_pass += 1
                                    b_valid_bottom += 1

                                    sl = l - MODEL_B["sl_buffer_atr"] * atr
                                    tp = mid_range
                                    risk = abs(c - sl)
                                    pnl = _simulate(m15, i, "long", c, sl, tp,
                                                    pair, SPREAD_PIPS, MAX_HOLD_BARS)
                                    trades_b.append({
                                        "model": "B", "direction": "long",
                                        "side": "bottom", "pnl_pips": pnl,
                                        "risk_pips": price_to_pips(risk, pair),
                                        "hour": hour,
                                    })
                                    if hour is not None:
                                        hourly[hour]["B"] += 1

                                elif rsi > rsi_prev + 3:
                                    b_valid_bottom += 1
                                    sl = l - MODEL_B["sl_buffer_atr"] * atr
                                    tp = mid_range
                                    risk = abs(c - sl)
                                    pnl = _simulate(m15, i, "long", c, sl, tp,
                                                    pair, SPREAD_PIPS, MAX_HOLD_BARS)
                                    trades_b.append({
                                        "model": "B", "direction": "long",
                                        "side": "bottom", "pnl_pips": pnl,
                                        "risk_pips": price_to_pips(risk, pair),
                                        "hour": hour,
                                    })
                                    if hour is not None:
                                        hourly[hour]["B"] += 1

    # ===========================================================
    #  REPORT
    # ===========================================================
    analyzed = total_bars
    b_valid = b_valid_top + b_valid_bottom

    print(f"\n--- Regime (H1 context) ---")
    for k, v in sorted(regime_counts.items()):
        print(f"  {k:12s}: {v:5d} ({v/max(analyzed,1)*100:.1f}%)")

    print(f"\n--- Model A: Compression+Breakout on M15 (Trend: {a_eligible} bars) ---")
    print(f"  Compression pass: {a_compression_pass} ({a_compression_pass/max(a_eligible,1)*100:.1f}%)")
    print(f"  Breakout pass:    {a_breakout_pass}")
    print(f"  VALID (w/ momentum): {a_valid}")
    if compression_ratios:
        cr = pd.Series(compression_ratios)
        print(f"  Compression ratio: mean={cr.mean():.2f} median={cr.median():.2f}")
        for t in [1.5, 2.0, 2.5, 3.0]:
            print(f"    <= {t}: {(cr<=t).sum()} ({(cr<=t).mean()*100:.1f}%)")

    print(f"\n--- Model B: Range Bounce on M15 (Range: {b_eligible} bars) ---")
    print(f"  Poked top:     {b_poked_top}")
    print(f"  Poked bottom:  {b_poked_bottom}")
    print(f"  Wick pass:     {b_wick_pass}")
    print(f"  RSI pass:      {b_rsi_pass}")
    print(f"  VALID:         {b_valid} (top={b_valid_top}, bottom={b_valid_bottom})")

    # Performance
    all_trades = trades_a + trades_b
    days = max(analyzed // 14, 1)  # ~14 M15 bars per trading day (in overlap)

    for label, subset in [("Model A", trades_a), ("Model B", trades_b), ("Combined", all_trades)]:
        if not subset:
            print(f"\n--- {label}: No trades ---")
            continue
        tdf = pd.DataFrame(subset)
        wins = tdf[tdf["pnl_pips"] > 0]
        losses = tdf[tdf["pnl_pips"] <= 0]
        total_pnl = tdf["pnl_pips"].sum()
        wr = len(wins) / len(tdf)
        avg_pnl = tdf["pnl_pips"].mean()
        avg_win = wins["pnl_pips"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl_pips"].mean() if len(losses) > 0 else 0
        gp = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        gl = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
        pf = gp / gl if gl > 0 else float("inf")
        cost = SPREAD_PIPS + SLIPPAGE_PIPS
        pnl_before_cost = total_pnl + len(tdf) * cost
        slip_impact = (len(tdf) * cost) / pnl_before_cost * 100 if pnl_before_cost > 0 else 999

        print(f"\n--- {label}: Performance ---")
        print(f"  Trades:          {len(tdf)}")
        print(f"  Trades/day:      {len(tdf)/max(days,1):.1f}")
        print(f"  Win rate:        {wr:.0%}")
        print(f"  Total PnL:       {total_pnl:+.1f} pips")
        print(f"  Avg PnL:         {avg_pnl:+.2f} pips")
        print(f"  Avg winner:      {avg_win:+.2f} pips")
        print(f"  Avg loser:       {avg_loss:+.2f} pips")
        print(f"  Profit Factor:   {pf:.2f}")
        print(f"  Avg risk:        {tdf['risk_pips'].mean():.2f} pips")
        print(f"  Slippage impact: {slip_impact:.1f}%")

    # Hourly
    if any(hourly.values()):
        print(f"\n--- Hourly Distribution ---")
        print(f"  {'Hour':>4s}  {'A':>4s}  {'B':>4s}  {'Total':>5s}")
        for h in range(24):
            a_h = hourly[h]["A"]
            b_h = hourly[h]["B"]
            tot = a_h + b_h
            if tot > 0:
                bar = "#" * min(tot, 40)
                print(f"  {h:4d}  {a_h:4d}  {b_h:4d}  {tot:5d}  {bar}")

    # Comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON: M15 vs M5 vs M1 trigger ({pair})")
    print(f"{'='*70}")
    m1_ref = {"EUR_USD": (231, 404, 24.9), "USD_JPY": (243, 803, 14.5), "GBP_USD": (276, 577, 22.2)}
    m5_ref = {"EUR_USD": (40, 52.9, 23.2)}
    if pair in m1_ref:
        m1_trades, m1_pnl, m1_slip = m1_ref[pair]
        m15_total = len(all_trades)
        m15_pnl = sum(t["pnl_pips"] for t in all_trades)
        m5_trades, m5_pnl_ref, m5_slip = m5_ref.get(pair, (0, 0, 0))
        a_count = sum(1 for t in all_trades if t["model"] == "A")
        b_count_t = sum(1 for t in all_trades if t["model"] == "B")
        print(f"  {'':15s} {'M1':>10s} {'M5':>10s} {'M15':>10s}")
        print(f"  {'Trades':15s} {m1_trades:10d} {m5_trades:10d} {m15_total:10d}")
        print(f"  {'PnL (pips)':15s} {m1_pnl:10.1f} {m5_pnl_ref:10.1f} {m15_pnl:10.1f}")
        print(f"  {'Model A':15s} {m1_trades:10d} {'0':>10s} {a_count:10d}")
        print(f"  {'Model B':15s} {'0':>10s} {m5_trades:10d} {b_count_t:10d}")

    return len(all_trades), all_trades


def _simulate(df, entry_bar, direction, entry, sl, tp, pair,
              spread_pips, max_hold):
    """Forward simulation on M15 bars."""
    cost = pips_to_price(spread_pips / 2 + SLIPPAGE_PIPS / 2, pair)
    if direction == "long":
        entry += cost
    else:
        entry -= cost

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

    last = float(df.iloc[min(entry_bar + max_hold, len(df) - 1)]["close"])
    if direction == "long":
        return price_to_pips(last - entry, pair)
    else:
        return price_to_pips(entry - last, pair)


def main():
    parser = argparse.ArgumentParser(description="M5 Trigger Diagnostic")
    parser.add_argument("--pair", type=str, default=None)
    args = parser.parse_args()

    pairs = [args.pair] if args.pair else list(DATA_FILES.keys())
    results = {}

    for pair in pairs:
        count, trades = analyze_pair(pair) or (0, [])
        results[pair] = (count, trades)

    if len(pairs) > 1:
        print(f"\n{'='*70}")
        print(f"  OVERALL SUMMARY")
        print(f"{'='*70}")
        grand = []
        for pair, (count, trades) in results.items():
            grand.extend(trades)
            a_count = sum(1 for t in trades if t["model"] == "A")
            b_count = sum(1 for t in trades if t["model"] == "B")
            pnl = sum(t["pnl_pips"] for t in trades)
            print(f"  {pair}: {count} trades (A={a_count}, B={b_count}) | PnL={pnl:+.1f}")

        if grand:
            gt = pd.DataFrame(grand)
            print(f"\n  Total: {len(gt)} trades")
            print(f"  Total PnL: {gt['pnl_pips'].sum():+.1f} pips")
            print(f"  Win rate: {(gt['pnl_pips'] > 0).mean():.0%}")
            a_pnl = gt[gt["model"] == "A"]["pnl_pips"].sum()
            b_pnl = gt[gt["model"] == "B"]["pnl_pips"].sum()
            print(f"  Model A PnL: {a_pnl:+.1f} | Model B PnL: {b_pnl:+.1f}")


if __name__ == "__main__":
    main()
