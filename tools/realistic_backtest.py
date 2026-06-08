"""Realistic Multi-Pair Backtester — All pairs simultaneously with shared constraints.

Unlike combined_backtest.py which runs pairs separately then merges,
this iterates through time bar-by-bar across ALL pairs together:

- Shared max_concurrent (2 across all pairs, not per pair)
- Shared daily_loss limit (accumulated across all pairs)
- Shared cooldown and circuit breaker
- Real-time correlation guard (EUR/USD + GBP/USD same direction blocked)
- Combined equity curve and drawdown

This is the closest simulation to real Paper/Live trading.

Usage:
    python realistic_backtest.py
    python realistic_backtest.py --months 3
"""

import argparse
import sys
import time as time_module
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

from src.scalp_mode.engine.feature_engine import FeatureEngine, IndicatorSet
from src.scalp_mode.engine.regime_engine import RegimeEngine, Regime
from src.scalp_mode.engine.model_a import ModelATrigger, TriggerPhase, Direction
from src.scalp_mode.utils.pip_utils import pip_value, price_to_pips, pips_to_price


PAIRS = ["EUR_USD", "USD_JPY", "GBP_USD"]

CORRELATED_PAIRS = [frozenset({"EUR_USD", "GBP_USD"})]

MAX_SPREAD = {"EUR_USD": 0.8, "USD_JPY": 0.8, "GBP_USD": 1.0}


@dataclass
class OpenTrade:
    pair: str
    direction: str
    model: str
    entry_price: float
    sl_price: float
    tp_price: float
    units: int
    entry_bar: int
    entry_time: Optional[datetime]
    risk_amount: float
    spread_at_entry: float
    slippage: float
    sl_moved: bool = False
    is_borderline: bool = False


@dataclass
class ClosedTrade:
    pair: str
    direction: str
    model: str
    entry_time: Optional[datetime]
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: float
    pnl_pips: float
    pnl_pct: float
    exit_reason: str
    hold_seconds: int
    spread_at_entry: float


def main():
    parser = argparse.ArgumentParser(description="Realistic multi-pair backtest")
    parser.add_argument("--months", type=int, default=12, choices=[3, 6, 12, 24])
    parser.add_argument("--spread", type=float, default=0.3)
    parser.add_argument("--slippage", type=float, default=0.1)
    parser.add_argument("--nav", type=float, default=10000)
    args = parser.parse_args()

    suffix_map = {3: "3m", 6: "6m", 12: "12m", 24: "24m"}
    suffix = suffix_map.get(args.months, f"{args.months}m")

    # Load config
    from src.scalp_mode.config import Config
    config = Config("config/settings.yaml")
    sc = config.scalp

    # Risk params (SHARED across all pairs)
    risk_pct = sc["risk"]["risk_pct"]
    max_concurrent = sc["risk"]["max_concurrent"]  # 2 total, not per pair
    daily_loss_limit = sc["risk"]["daily_loss"]
    consec_loss_circuit = sc["risk"]["consec_loss_circuit"]
    circuit_cooldown_min = sc["risk"]["cooldown_minutes"]
    cooldown_same_dir_min = sc["risk"]["cooldown_same_pair_dir_min"]

    # Model A config
    model_a_cfg = sc["model_a"]
    tp_R = model_a_cfg["tp_R"]
    sl_atr = model_a_cfg["sl_atr"]
    time_stop_min = model_a_cfg["time_stop_min"]
    sl_move_thr = model_a_cfg["sl_move_threshold_R"]
    sl_move_target = model_a_cfg["sl_move_target_R"]
    sl_move_window = model_a_cfg["sl_move_window_min"]

    print("=" * 70)
    print(f"  REALISTIC MULTI-PAIR BACKTEST — {args.months} months")
    print(f"  Pairs: {', '.join(PAIRS)}")
    print(f"  NAV: £{args.nav:,.0f} | Risk: {risk_pct:.2%}/trade | Max concurrent: {max_concurrent}")
    print(f"  tp_R={tp_R} | sl_atr={sl_atr} | Daily loss limit: {daily_loss_limit:.1%}")
    print("=" * 70)

    # =========================================================
    # Load and pre-compute indicators for all pairs
    # =========================================================
    fe = FeatureEngine()
    regime_eng = RegimeEngine(sc["regime"])
    trigger_a = ModelATrigger(model_a_cfg)

    pair_data = {}
    min_len = float("inf")

    for pair in PAIRS:
        data_path = Path(f"data/{pair}_M1_{suffix}.csv")
        if not data_path.exists():
            print(f"  ERROR: {data_path} not found")
            sys.exit(1)

        df = pd.read_csv(data_path)
        timestamps = None
        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], utc=True)

        # Resample M5
        groups = len(df) // 5
        trimmed = df.iloc[:groups * 5].copy()
        trimmed["g"] = np.repeat(range(groups), 5)
        m5 = trimmed.groupby("g").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).reset_index(drop=True)

        # Pre-compute indicators
        print(f"  Pre-computing {pair}... ({len(df)} candles)")
        m1_series = fe.compute_series(df, "M1")
        m5_series = fe.compute_series(m5, "M5")

        pair_data[pair] = {
            "df": df, "m5": m5, "timestamps": timestamps,
            "m1_series": m1_series, "m5_series": m5_series,
        }
        min_len = min(min_len, len(df))

    # =========================================================
    # Helper functions
    # =========================================================
    def _val(series_dict, key, idx):
        s = series_dict.get(key)
        if s is None or idx >= len(s):
            return None
        v = s.iloc[idx]
        return None if pd.isna(v) else float(v)

    def _make_ind(series_dict, idx, timeframe="M1"):
        ind = IndicatorSet(
            ema20=_val(series_dict, "ema20", idx),
            ema50=_val(series_dict, "ema50", idx),
            atr14=_val(series_dict, "atr14", idx),
            rsi14=_val(series_dict, "rsi14", idx),
            bb_upper=_val(series_dict, "bb_upper", idx),
            bb_mid=_val(series_dict, "bb_mid", idx),
            bb_lower=_val(series_dict, "bb_lower", idx),
            bb_width=_val(series_dict, "bb_width", idx),
            ema_slope=_val(series_dict, "ema_slope", idx),
        )
        if timeframe == "M1" and "macd_hist" in series_dict:
            ind.macd_hist = _val(series_dict, "macd_hist", idx)
            ind.macd_hist_prev = _val(series_dict, "macd_hist", idx - 1) if idx >= 1 else None
            ind.macd_hist_prev2 = _val(series_dict, "macd_hist", idx - 2) if idx >= 2 else None
        return ind

    def _has_correlation_conflict(pair, direction, open_trades):
        for corr_set in CORRELATED_PAIRS:
            if pair in corr_set:
                partner = list(corr_set - {pair})[0]
                for ot in open_trades:
                    if ot.pair == partner and ot.direction == direction:
                        return True
        return False

    def _manage_trade(ot, bar, bar_idx):
        bars_held = bar_idx - ot.entry_bar
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])

        if ot.direction == "long":
            if l <= ot.sl_price:
                return {"exit_price": ot.sl_price, "reason": "sl_hit"}
            if h >= ot.tp_price:
                return {"exit_price": ot.tp_price, "reason": "tp_hit"}
            risk = ot.risk_amount
            current_R = (c - ot.entry_price) / risk if risk > 0 else 0
            if (not ot.sl_moved and current_R >= sl_move_thr
                    and sl_move_window[0] <= bars_held <= sl_move_window[1]):
                sl_offset = abs(sl_move_target) * risk
                ot.sl_price = ot.entry_price - sl_offset
                ot.sl_moved = True
        else:
            if h >= ot.sl_price:
                return {"exit_price": ot.sl_price, "reason": "sl_hit"}
            if l <= ot.tp_price:
                return {"exit_price": ot.tp_price, "reason": "tp_hit"}
            risk = ot.risk_amount
            current_R = (ot.entry_price - c) / risk if risk > 0 else 0
            if (not ot.sl_moved and current_R >= sl_move_thr
                    and sl_move_window[0] <= bars_held <= sl_move_window[1]):
                sl_offset = abs(sl_move_target) * risk
                ot.sl_price = ot.entry_price + sl_offset
                ot.sl_moved = True

        if bars_held >= time_stop_min:
            return {"exit_price": c, "reason": "time_stop"}

        return None

    # =========================================================
    # Main simulation loop
    # =========================================================
    nav = args.nav
    initial_nav = args.nav  # Fixed for sizing — no compounding
    total_pnl_gbp = 0.0
    open_trades: list[OpenTrade] = []
    closed_trades: list[ClosedTrade] = []

    # Shared state
    daily_pnl = 0.0
    current_day = None
    consec_losses = 0
    circuit_until = None
    last_trade_time = {}  # (pair, direction) → datetime

    # Stats
    correlation_blocked = 0
    concurrent_blocked = 0
    daily_loss_blocked = 0
    circuit_blocked = 0
    cooldown_blocked = 0
    monthly_pnl = defaultdict(float)

    equity = [nav]
    t0 = time_module.time()
    warmup = 60

    print(f"\n  Simulating {min_len} bars across {len(PAIRS)} pairs...")

    for i in range(warmup, min_len):
        if (i - warmup) % 50000 == 0 and i > warmup:
            elapsed = time_module.time() - t0
            print(f"    ...{i - warmup} bars | {len(closed_trades)} trades | {elapsed:.0f}s")

        # Get bar time from first pair
        bar_time = None
        ts = pair_data[PAIRS[0]]["timestamps"]
        if ts is not None and i < len(ts):
            bar_time = ts.iloc[i]
            if not isinstance(bar_time, datetime):
                try:
                    bar_time = pd.Timestamp(bar_time).to_pydatetime()
                    if bar_time.tzinfo is None:
                        bar_time = bar_time.replace(tzinfo=timezone.utc)
                except:
                    bar_time = None

        # Reset daily PnL at day boundary
        if bar_time:
            day = bar_time.date()
            if current_day is None:
                current_day = day
            elif day != current_day:
                daily_pnl = 0.0
                current_day = day

        # ---- Manage open trades (all pairs) ----
        to_close = []
        for ot in open_trades:
            bar = pair_data[ot.pair]["df"].iloc[i]
            result = _manage_trade(ot, bar, i)
            if result is not None:
                exit_price = result["exit_price"]
                if ot.direction == "long":
                    pnl_price = exit_price - ot.entry_price
                else:
                    pnl_price = ot.entry_price - exit_price
                pnl_pips = price_to_pips(pnl_price, ot.pair)
                pnl_gbp = pnl_price * ot.units  # Actual GBP profit/loss
                pnl_pct = pnl_gbp / initial_nav if initial_nav > 0 else 0

                entry_time = ot.entry_time or datetime(2025, 4, 1, tzinfo=timezone.utc)
                exit_time = bar_time or datetime(2025, 4, 1, tzinfo=timezone.utc)
                hold_sec = int((exit_time - entry_time).total_seconds()) if entry_time and exit_time else 0

                closed_trades.append(ClosedTrade(
                    pair=ot.pair, direction=ot.direction, model=ot.model,
                    entry_time=entry_time, exit_time=exit_time,
                    entry_price=ot.entry_price, exit_price=exit_price,
                    pnl_pips=round(pnl_pips, 2), pnl_pct=round(pnl_pct, 6),
                    exit_reason=result["reason"],
                    hold_seconds=max(hold_sec, 0),
                    spread_at_entry=ot.spread_at_entry,
                ))
                total_pnl_gbp += pnl_gbp
                nav = initial_nav + total_pnl_gbp  # Track actual NAV but size on initial
                daily_pnl += pnl_pct
                equity.append(nav)

                if bar_time:
                    monthly_pnl[bar_time.strftime("%Y-%m")] += pnl_pips

                # Track consecutive losses
                if pnl_pips <= 0:
                    consec_losses += 1
                    if consec_losses >= consec_loss_circuit and bar_time:
                        circuit_until = bar_time + timedelta(minutes=circuit_cooldown_min)
                else:
                    consec_losses = 0

                to_close.append(ot)

        for ot in to_close:
            open_trades.remove(ot)

        # ---- Check shared constraints before new trades ----
        if len(open_trades) >= max_concurrent:
            concurrent_blocked += 1
            continue

        if daily_pnl <= -daily_loss_limit:
            daily_loss_blocked += 1
            continue

        if circuit_until and bar_time and bar_time < circuit_until:
            circuit_blocked += 1
            continue

        # ---- Try each pair ----
        for pair in PAIRS:
            if len(open_trades) >= max_concurrent:
                break

            pd_info = pair_data[pair]
            df = pd_info["df"]

            if i >= len(df):
                continue

            # Skip if already have open trade on this pair
            if any(ot.pair == pair for ot in open_trades):
                continue

            # Spread
            spread_pips = args.spread

            # Cooldown check
            if bar_time:
                for direction in ["long", "short"]:
                    key = (pair, direction)
                    if key in last_trade_time:
                        elapsed_min = (bar_time - last_trade_time[key]).total_seconds() / 60
                        if elapsed_min < cooldown_same_dir_min:
                            continue

            # Indicators (pre-computed)
            if i < 50:
                continue
            ind_m1 = _make_ind(pd_info["m1_series"], i, "M1")

            m5_idx = i // 5
            if m5_idx < 20:
                continue
            ind_m5 = _make_ind(pd_info["m5_series"], m5_idx, "M5")

            # NaN check
            has_nan, _ = ind_m5.has_nan()
            if has_nan:
                continue
            has_nan, _ = ind_m1.has_nan()
            if has_nan:
                continue

            # Regime
            m5_df = pd_info["m5"]
            close_m5 = float(m5_df.iloc[m5_idx]["close"]) if m5_idx < len(m5_df) else None
            if close_m5 is None:
                continue
            regime_result = regime_eng.evaluate(ind_m5, close_m5)
            if regime_result.regime == Regime.NO_TRADE:
                continue
            if regime_result.regime == Regime.RANGE:
                continue  # Model B disabled

            # Trigger
            lookback_m1 = df.iloc[max(0, i - 99):i + 1]
            trigger = trigger_a.evaluate(lookback_m1, ind_m1, regime_result.regime, pair)
            if trigger.phase != TriggerPhase.VALID:
                continue

            # Correlation guard
            direction = trigger.direction.value
            if _has_correlation_conflict(pair, direction, open_trades):
                correlation_blocked += 1
                continue

            # Cooldown (refined with actual direction)
            if bar_time:
                key = (pair, direction)
                if key in last_trade_time:
                    elapsed_min = (bar_time - last_trade_time[key]).total_seconds() / 60
                    if elapsed_min < cooldown_same_dir_min:
                        cooldown_blocked += 1
                        continue

            # Position sizing
            stop_pips = trigger.risk_pips or 0
            if stop_pips <= 0:
                continue
            risk_amount = initial_nav * risk_pct  # Fixed sizing — no compounding
            units = int(risk_amount / (stop_pips * pip_value(pair)))
            if units <= 0:
                continue

            # Entry with costs
            entry_price = trigger.entry_price
            if direction == "long":
                entry_price += pips_to_price(args.slippage, pair)
            else:
                entry_price -= pips_to_price(args.slippage, pair)

            open_trades.append(OpenTrade(
                pair=pair, direction=direction, model="A",
                entry_price=entry_price,
                sl_price=trigger.sl_price,
                tp_price=trigger.tp_price,
                units=units, entry_bar=i, entry_time=bar_time,
                risk_amount=abs(entry_price - trigger.sl_price),
                spread_at_entry=spread_pips,
                slippage=args.slippage,
            ))

            if bar_time:
                last_trade_time[(pair, direction)] = bar_time

    # Close remaining
    for ot in open_trades:
        bar = pair_data[ot.pair]["df"].iloc[min_len - 1]
        c = float(bar["close"])
        if ot.direction == "long":
            pnl_price = c - ot.entry_price
        else:
            pnl_price = ot.entry_price - c
        pnl_pips = price_to_pips(pnl_price, ot.pair)
        closed_trades.append(ClosedTrade(
            pair=ot.pair, direction=ot.direction, model=ot.model,
            entry_time=ot.entry_time, exit_time=bar_time,
            entry_price=ot.entry_price, exit_price=c,
            pnl_pips=round(pnl_pips, 2), pnl_pct=0,
            exit_reason="end_of_data", hold_seconds=0,
            spread_at_entry=ot.spread_at_entry,
        ))

    elapsed = time_module.time() - t0

    # =========================================================
    # Report
    # =========================================================
    print(f"\n{'='*70}")
    print(f"  REALISTIC MULTI-PAIR RESULTS")
    print(f"  Simulation time: {elapsed:.1f}s")
    print(f"{'='*70}")

    total = len(closed_trades)
    if total == 0:
        print("  No trades executed.")
        return

    wins = sum(1 for t in closed_trades if t.pnl_pips > 0)
    total_pnl = sum(t.pnl_pips for t in closed_trades)
    wr = wins / total
    gp = sum(t.pnl_pips for t in closed_trades if t.pnl_pips > 0)
    gl = abs(sum(t.pnl_pips for t in closed_trades if t.pnl_pips <= 0))
    pf = gp / gl if gl > 0 else float("inf")

    # Max drawdown
    peak = equity[0]
    max_dd_pct = 0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd_pct:
            max_dd_pct = dd

    # Trading days
    dates = set()
    for t in closed_trades:
        if t.entry_time:
            dates.add(t.entry_time.date())
    trading_days = max(len(dates), 1)

    # Per pair
    print(f"\n--- Per Pair ---")
    print(f"  {'Pair':10s} {'Trades':>7s} {'WR':>6s} {'PnL pips':>10s} {'PnL GBP':>10s} {'Avg pip':>8s}")
    for pair in PAIRS:
        pt = [t for t in closed_trades if t.pair == pair]
        if pt:
            pw = sum(1 for t in pt if t.pnl_pips > 0)
            pp = sum(t.pnl_pips for t in pt)
            pg = sum(t.pnl_pct * initial_nav for t in pt)
            print(f"  {pair:10s} {len(pt):7d} {pw/len(pt):5.0%} {pp:+10.1f} {pg:+10.0f} {pp/len(pt):+8.2f}")

    # First 10 trades (verification)
    print(f"\n--- First 10 Trades (verification) ---")
    print(f"  {'#':>3s} {'Pair':8s} {'Dir':5s} {'Units':>7s} {'SL pips':>8s} {'PnL pip':>8s} {'PnL GBP':>9s} {'Exit':10s}")
    running_nav = initial_nav
    for j, t in enumerate(closed_trades[:10]):
        pnl_gbp = t.pnl_pct * initial_nav
        running_nav += pnl_gbp
        risk_gbp = initial_nav * risk_pct
        print(f"  {j+1:3d} {t.pair:8s} {t.direction:5s} {'—':>7s} {'—':>8s} {t.pnl_pips:+8.2f} {pnl_gbp:+9.2f} {t.exit_reason:10s}")

    print(f"\n--- Portfolio ---")
    print(f"  Total trades:        {total}")
    print(f"  Trades/day:          {total / trading_days:.1f}")
    print(f"  Win rate:            {wr:.0%}")
    print(f"  Total PnL:           {total_pnl:+.1f} pips")
    print(f"  Total PnL (GBP):     £{total_pnl_gbp:+,.2f}")
    print(f"  Profit Factor:       {pf:.2f}")
    print(f"  Initial NAV:         £{initial_nav:,.0f} (fixed — no compounding)")
    print(f"  Final NAV:           £{nav:,.2f}")
    print(f"  Return:              {total_pnl_gbp / initial_nav:.1%}")
    print(f"  Max Drawdown:        {max_dd_pct:.2%}")
    print(f"  Trading days:        {trading_days}")

    print(f"\n--- Shared Constraints Impact ---")
    print(f"  Correlation blocked: {correlation_blocked}")
    print(f"  Concurrent blocked:  {concurrent_blocked}")
    print(f"  Daily loss blocked:  {daily_loss_blocked}")
    print(f"  Circuit breaker:     {circuit_blocked}")
    print(f"  Cooldown blocked:    {cooldown_blocked}")

    # Exit reasons
    reasons = defaultdict(int)
    for t in closed_trades:
        reasons[t.exit_reason] += 1
    print(f"\n--- Exit Reasons ---")
    for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {r:20s}: {c}")

    # Monthly
    print(f"\n--- Monthly PnL (pips) ---")
    for month in sorted(monthly_pnl.keys()):
        pnl = monthly_pnl[month]
        status = "WIN" if pnl > 0 else "LOSS"
        bar = "+" * min(int(pnl / 50), 40) if pnl > 0 else "-" * min(int(abs(pnl) / 50), 40)
        print(f"  {month}: {pnl:+8.1f} {status:4s} {bar}")

    pm = sum(1 for v in monthly_pnl.values() if v > 0)
    tm = len(monthly_pnl)
    print(f"\n  Profitable months: {pm}/{tm} ({pm/max(tm,1):.0%})")

    # Yearly breakdown
    yearly_pnl = defaultdict(float)
    yearly_trades = defaultdict(int)
    yearly_wins = defaultdict(int)
    for t in closed_trades:
        if t.entry_time:
            year_key = t.entry_time.strftime("%Y")
            yearly_pnl[year_key] += t.pnl_pips
            yearly_trades[year_key] += 1
            if t.pnl_pips > 0:
                yearly_wins[year_key] += 1

    if len(yearly_pnl) > 1:
        print(f"\n--- Yearly Breakdown ---")
        print(f"  {'Year':6s} {'Trades':>7s} {'WR':>6s} {'PnL pips':>10s} {'PnL GBP':>10s}")
        for year in sorted(yearly_pnl.keys()):
            yt = yearly_trades[year]
            yw = yearly_wins[year]
            yp = yearly_pnl[year]
            yg = yp * (total_pnl_gbp / total_pnl) if total_pnl > 0 else 0
            print(f"  {year:6s} {yt:7d} {yw/max(yt,1):5.0%} {yp:+10.1f} {yg:+10.0f}")

    # Per-pair per-year
    if len(yearly_pnl) > 1:
        print(f"\n--- Per Pair Per Year ---")
        for pair in PAIRS:
            pt = [t for t in closed_trades if t.pair == pair]
            pair_yearly = defaultdict(lambda: {"trades": 0, "pnl": 0, "wins": 0})
            for t in pt:
                if t.entry_time:
                    yk = t.entry_time.strftime("%Y")
                    pair_yearly[yk]["trades"] += 1
                    pair_yearly[yk]["pnl"] += t.pnl_pips
                    if t.pnl_pips > 0:
                        pair_yearly[yk]["wins"] += 1
            print(f"  {pair}:")
            for year in sorted(pair_yearly.keys()):
                d = pair_yearly[year]
                wr = d["wins"] / max(d["trades"], 1)
                print(f"    {year}: {d['trades']:4d} trades | WR={wr:.0%} | PnL={d['pnl']:+.1f}")

    # Comparison
    print(f"\n--- vs Separate Backtests ---")
    separate_pnl = 1860 + 3578 + 2219  # tp_R=2.0 results
    print(f"  Separate (no shared constraints): +{separate_pnl} pips")
    print(f"  Realistic (shared constraints):   {total_pnl:+.0f} pips")
    print(f"  Difference:                       {total_pnl - separate_pnl:+.0f} pips ({(total_pnl - separate_pnl)/separate_pnl*100:+.1f}%)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
