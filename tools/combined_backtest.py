"""Combined Portfolio Backtest — All 3 pairs simultaneously.

Runs EUR_USD, USD_JPY, GBP_USD together and shows:
1. Combined PnL and equity curve
2. Correlation guard impact (EUR/USD + GBP/USD same direction blocked)
3. Max concurrent trades across all pairs
4. Combined drawdown
5. Daily/monthly P&L breakdown
6. Per-pair contribution

Usage:
    python combined_backtest.py
    python combined_backtest.py --months 3
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

from src.scalp_mode.backtest.backtester import Backtester, BacktestConfig
from src.scalp_mode.backtest.performance import PerformanceAnalyzer
from src.scalp_mode.backtest.go_nogo import GoNoGoEvaluator


PAIRS = ["EUR_USD", "USD_JPY", "GBP_USD"]

# Correlation pairs (from risk_manager.py)
CORRELATED = [
    frozenset({"EUR_USD", "GBP_USD"}),
]


def main():
    parser = argparse.ArgumentParser(description="Combined 3-pair backtest")
    parser.add_argument("--months", type=int, default=12, choices=[3, 12])
    parser.add_argument("--spread", type=float, default=0.3)
    parser.add_argument("--nav", type=float, default=10000)
    args = parser.parse_args()

    suffix = "3m" if args.months == 3 else "12m"

    # Load config
    from src.scalp_mode.config import Config
    config = Config("config/settings.yaml")
    scalp_cfg = config.scalp

    bt_config = BacktestConfig(
        initial_nav=args.nav,
        fixed_spread_pips=args.spread,
        slippage_pips=0.1,
        check_sessions=False,
        warmup_bars=60,
    )

    print("=" * 70)
    print(f"  COMBINED PORTFOLIO BACKTEST — {args.months} months")
    print(f"  Pairs: {', '.join(PAIRS)}")
    print(f"  NAV: £{args.nav:,.0f}  Spread: {args.spread} pips")
    print("=" * 70)

    # Run backtest for each pair
    all_trades = {}
    for pair in PAIRS:
        data_path = Path(f"data/{pair}_M1_{suffix}.csv")
        if not data_path.exists():
            print(f"  ERROR: {data_path} not found")
            sys.exit(1)

        df = pd.read_csv(data_path)
        timestamps = None
        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], utc=True)

        print(f"\n  Running {pair}... ({len(df)} candles)")
        backtester = Backtester(scalp_cfg, bt_config)
        trades = backtester.run(pair, df, timestamps)
        all_trades[pair] = trades
        print(f"    → {len(trades)} trades")

    # Combine all trades and sort by entry time
    combined = []
    for pair, trades in all_trades.items():
        for t in trades:
            combined.append(t)
    combined.sort(key=lambda t: t.entry_time)

    print(f"\n  Total raw trades: {len(combined)}")

    # Apply correlation guard: remove EUR_USD + GBP_USD same direction overlaps
    filtered = []
    blocked_corr = 0
    for trade in combined:
        # Check if any open filtered trade conflicts
        conflict = False
        for corr_set in CORRELATED:
            if trade.pair in corr_set:
                partner = (corr_set - {trade.pair}).pop()
                # Check if partner has an open trade in same direction
                for open_t in filtered:
                    if (open_t.pair == partner and
                            open_t.direction == trade.direction and
                            open_t.entry_time <= trade.entry_time < open_t.exit_time):
                        conflict = True
                        break
            if conflict:
                break
        if conflict:
            blocked_corr += 1
        else:
            filtered.append(trade)

    print(f"  Correlation blocked: {blocked_corr} trades")
    print(f"  After correlation: {len(filtered)} trades")

    # Check max concurrent trades
    max_concurrent = 0
    concurrent_hist = defaultdict(int)
    for i, trade in enumerate(filtered):
        concurrent = sum(
            1 for t in filtered
            if t.entry_time <= trade.entry_time < t.exit_time
        )
        max_concurrent = max(max_concurrent, concurrent)
        concurrent_hist[concurrent] += 1

    # Build equity curve
    nav = args.nav
    equity = [nav]
    equity_dates = [filtered[0].entry_time if filtered else datetime(2025, 4, 1, tzinfo=timezone.utc)]
    daily_pnl = defaultdict(float)
    monthly_pnl = defaultdict(float)

    for trade in filtered:
        pnl_amount = trade.pnl_pct * nav
        nav += pnl_amount
        equity.append(nav)
        equity_dates.append(trade.exit_time)

        day_key = trade.exit_time.strftime("%Y-%m-%d")
        month_key = trade.exit_time.strftime("%Y-%m")
        daily_pnl[day_key] += trade.pnl_pips
        monthly_pnl[month_key] += trade.pnl_pips

    # Max drawdown on equity
    peak = equity[0]
    max_dd = 0
    max_dd_pct = 0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd_pct:
            max_dd_pct = dd
            max_dd = peak - val

    # Per-pair stats
    pair_stats = {}
    for pair in PAIRS:
        pair_trades = [t for t in filtered if t.pair == pair]
        if pair_trades:
            wins = sum(1 for t in pair_trades if t.pnl_pips > 0)
            total_pnl = sum(t.pnl_pips for t in pair_trades)
            pair_stats[pair] = {
                "trades": len(pair_trades),
                "wr": wins / len(pair_trades),
                "pnl": total_pnl,
                "avg": total_pnl / len(pair_trades),
            }

    # Combined stats
    total_trades = len(filtered)
    total_pnl_pips = sum(t.pnl_pips for t in filtered)
    total_wins = sum(1 for t in filtered if t.pnl_pips > 0)
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    gross_profit = sum(t.pnl_pips for t in filtered if t.pnl_pips > 0)
    gross_loss = abs(sum(t.pnl_pips for t in filtered if t.pnl_pips <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Trading days
    if filtered:
        dates = set(t.entry_time.date() for t in filtered)
        trading_days = len(dates)
    else:
        trading_days = 1

    # Report
    print(f"\n{'='*70}")
    print(f"  COMBINED PORTFOLIO RESULTS")
    print(f"{'='*70}")

    print(f"\n--- Per Pair ---")
    print(f"  {'Pair':10s} {'Trades':>7s} {'WR':>6s} {'PnL':>10s} {'Avg':>8s} {'Contrib':>8s}")
    for pair in PAIRS:
        if pair in pair_stats:
            ps = pair_stats[pair]
            contrib = ps["pnl"] / total_pnl_pips * 100 if total_pnl_pips > 0 else 0
            print(f"  {pair:10s} {ps['trades']:7d} {ps['wr']:5.0%} {ps['pnl']:+10.1f} {ps['avg']:+8.2f} {contrib:7.0f}%")
    print(f"  {'TOTAL':10s} {total_trades:7d} {overall_wr:5.0%} {total_pnl_pips:+10.1f}")

    print(f"\n--- Portfolio Metrics ---")
    print(f"  Total PnL:           {total_pnl_pips:+.1f} pips")
    print(f"  Final NAV:           £{nav:,.2f} (started £{args.nav:,.0f})")
    print(f"  Return:              {(nav - args.nav) / args.nav:.1%}")
    print(f"  Win Rate:            {overall_wr:.0%}")
    print(f"  Profit Factor:       {pf:.2f}")
    print(f"  Max Drawdown:        {max_dd_pct:.2%} (£{max_dd:,.2f})")
    print(f"  Trading Days:        {trading_days}")
    print(f"  Trades/Day:          {total_trades / max(trading_days, 1):.1f}")
    print(f"  Max Concurrent:      {max_concurrent}")
    print(f"  Correlation Blocked: {blocked_corr} trades")

    # Monthly breakdown
    print(f"\n--- Monthly PnL (pips) ---")
    for month in sorted(monthly_pnl.keys()):
        pnl = monthly_pnl[month]
        bar = "+" * min(int(pnl / 50), 40) if pnl > 0 else "-" * min(int(abs(pnl) / 50), 40)
        status = "WIN" if pnl > 0 else "LOSS"
        print(f"  {month}: {pnl:+8.1f} {status:4s} {bar}")

    profitable_months = sum(1 for v in monthly_pnl.values() if v > 0)
    total_months = len(monthly_pnl)
    print(f"\n  Profitable months: {profitable_months}/{total_months} ({profitable_months/max(total_months,1):.0%})")

    # Estimated real-world performance
    print(f"\n--- Realistic Estimates (after Paper degradation) ---")
    for label, factor in [("Optimistic (50%)", 0.50), ("Realistic (35%)", 0.35), ("Conservative (20%)", 0.20)]:
        est_pnl = total_pnl_pips * factor
        est_nav = args.nav + (nav - args.nav) * factor
        print(f"  {label:25s}: {est_pnl:+.0f} pips → £{est_nav:,.0f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
