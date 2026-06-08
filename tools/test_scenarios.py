"""Comprehensive Backtest Suite — All pairs × Multiple scenarios.

Runs backtests across EUR_USD, USD_JPY, GBP_USD with different parameter
combinations to identify optimal settings, problems, and gaps.

Usage:
    python test_scenarios.py
    python test_scenarios.py --pairs EUR_USD USD_JPY
    python test_scenarios.py --quick  (fewer scenarios, faster)

Output: logs/scenario_results.csv + console summary
"""

import sys
import time
import itertools
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.scalp_mode.config import Config
from src.scalp_mode.backtest.backtester import Backtester, BacktestConfig
from src.scalp_mode.backtest.performance import PerformanceAnalyzer
from src.scalp_mode.backtest.go_nogo import GoNoGoEvaluator


# ============================================================
#  SCENARIO DEFINITIONS
# ============================================================

PAIRS = ["EUR_USD", "USD_JPY", "GBP_USD"]

DATA_FILES = {
    "EUR_USD": "data/EUR_USD_M1_3m.csv",
    "USD_JPY": "data/USD_JPY_M1_3m.csv",
    "GBP_USD": "data/GBP_USD_M1_3m.csv",
}

# Base config (current calibrated values)
BASE = {
    "ema_slope_thr": 0.15,
    "compression_atr_mult": 2.0,
    "tp_R": 1.7,
    "compression_N": 8,
    "body_ratio_min": 0.55,
    "rsi_min_long": 55,
    "sl_atr": 0.8,
    "breakout_buffer_atr": 0.10,
    "retest_timeout": 3,
    "retest_tolerance_atr": 0.15,
    "spread": 0.3,
    "slippage": 0.1,
    "model_b_enabled": False,
}

# Scenarios to test
SCENARIOS = {
    # === Current baseline ===
    "baseline": {},

    # === EMA slope sensitivity ===
    "slope_0.10": {"ema_slope_thr": 0.10},
    "slope_0.12": {"ema_slope_thr": 0.12},
    "slope_0.15": {},  # same as baseline
    "slope_0.18": {"ema_slope_thr": 0.18},
    "slope_0.20": {"ema_slope_thr": 0.20},

    # === Compression sensitivity ===
    "compress_1.5": {"compression_atr_mult": 1.5},
    "compress_2.0": {},  # same as baseline
    "compress_2.5": {"compression_atr_mult": 2.5},
    "compress_3.0": {"compression_atr_mult": 3.0},

    # === TP target sensitivity ===
    "tp_1.0": {"tp_R": 1.0},
    "tp_1.2": {"tp_R": 1.2},
    "tp_1.5": {"tp_R": 1.5},
    "tp_1.7": {},  # same as baseline
    "tp_2.0": {"tp_R": 2.0},
    "tp_2.5": {"tp_R": 2.5},

    # === SL sensitivity ===
    "sl_0.6": {"sl_atr": 0.6},
    "sl_0.8": {},  # same as baseline
    "sl_1.0": {"sl_atr": 1.0},
    "sl_1.2": {"sl_atr": 1.2},

    # === Spread sensitivity (execution cost) ===
    "spread_0.2": {"spread": 0.2},
    "spread_0.3": {},  # same as baseline
    "spread_0.5": {"spread": 0.5},
    "spread_0.8": {"spread": 0.8},

    # === Compression + TP combined ===
    "compress2.5_tp2.0": {"compression_atr_mult": 2.5, "tp_R": 2.0},
    "compress3.0_tp2.0": {"compression_atr_mult": 3.0, "tp_R": 2.0},

    # === Aggressive vs Conservative ===
    "conservative": {
        "ema_slope_thr": 0.20, "compression_atr_mult": 1.5,
        "tp_R": 1.0, "sl_atr": 0.8, "spread": 0.5,
    },
    "balanced": {
        "ema_slope_thr": 0.15, "compression_atr_mult": 2.0,
        "tp_R": 1.5, "sl_atr": 0.8, "spread": 0.3,
    },
    "aggressive": {
        "ema_slope_thr": 0.10, "compression_atr_mult": 3.0,
        "tp_R": 2.0, "sl_atr": 0.6, "spread": 0.2,
    },

    # === Model B scenarios ===
    "model_b_default": {"model_b_enabled": True},
    "model_b_loose": {
        "model_b_enabled": True,
        "wick_ratio_min": 0.40,
        "wick_excess_atr": 0.50,
    },

    # === Retest sensitivity ===
    "retest_2": {"retest_timeout": 2},
    "retest_3": {},  # baseline
    "retest_5": {"retest_timeout": 5},

    # === Body ratio sensitivity ===
    "body_0.45": {"body_ratio_min": 0.45},
    "body_0.55": {},  # baseline
    "body_0.65": {"body_ratio_min": 0.65},

    # === RSI threshold ===
    "rsi_50": {"rsi_min_long": 50},
    "rsi_55": {},  # baseline
    "rsi_60": {"rsi_min_long": 60},
}

QUICK_SCENARIOS = [
    "baseline", "slope_0.10", "slope_0.20",
    "compress_1.5", "compress_3.0",
    "tp_1.0", "tp_2.0",
    "spread_0.2", "spread_0.8",
    "aggressive", "conservative",
    "model_b_loose",
]


# ============================================================
#  RUNNER
# ============================================================

def build_scalp_config(overrides: dict) -> dict:
    """Build a complete scalp config from BASE + overrides."""
    params = {**BASE, **overrides}

    config = {
        "regime": {
            "trend": {
                "ema_slope_thr": params["ema_slope_thr"],
                "rsi_min": 52,
                "rsi_max": 78,
            },
            "range": {"bb_width_thr": 0.004},
        },
        "model_a": {
            "compression_N": params["compression_N"],
            "compression_atr_mult": params["compression_atr_mult"],
            "breakout_buffer_atr": params["breakout_buffer_atr"],
            "retest_timeout": params["retest_timeout"],
            "retest_tolerance_atr": params["retest_tolerance_atr"],
            "body_ratio_min": params["body_ratio_min"],
            "rsi_min_long": params["rsi_min_long"],
            "sl_atr": params["sl_atr"],
            "tp_R": params["tp_R"],
            "time_stop_min": 6,
            "sl_move_threshold_R": 0.8,
            "sl_move_target_R": -0.1,
            "sl_move_window_min": [2, 4],
        },
        "model_b": {
            "enabled": params.get("model_b_enabled", False),
            "range_window_M5": 12,
            "wick_ratio_min": params.get("wick_ratio_min", 0.60),
            "wick_excess_atr": params.get("wick_excess_atr", 0.25),
            "stop_spread_buffer_mult": 2.0,
            "stop_atr_buffer": 0.15,
            "rsi_overbought": 65,
            "rsi_reversal": 60,
            "rsi_oversold": 35,
            "rsi_reversal_up": 40,
            "target_type": "mid_range_or_bbmid",
        },
        "risk": {
            "risk_pct": 0.0025,
            "max_concurrent": 2,
            "cooldown_same_pair_dir_min": 10,
            "consec_loss_circuit": 3,
            "cooldown_minutes": 60,
            "trades_per_hour_pair": 3,
            "trades_per_hour_total": 6,
            "daily_loss": 0.01,
            "max_margin_pct": 0.08,
        },
        "orders": {
            "limit_ttl_seconds": 180,
            "fallback_market": True,
            "fallback_max_atr_distance": 0.3,
            "fallback_cooldown_min": 2,
            "price_bound_slippage": 0.2,
        },
        "costs": {
            "max_spread_pips": {
                "EUR_USD": 0.8,
                "USD_JPY": 0.8,
                "GBP_USD": 1.0,
            },
        },
    }
    return config, params


def run_one(pair: str, scenario_name: str, overrides: dict,
            data_cache: dict) -> dict:
    """Run a single backtest and return result dict."""
    scalp_config, params = build_scalp_config(overrides)

    bt_config = BacktestConfig(
        initial_nav=10000,
        fixed_spread_pips=params["spread"],
        slippage_pips=params.get("slippage", 0.1),
        check_sessions=False,
        warmup_bars=60,
        spread_model="session_variable",
    )

    backtester = Backtester(scalp_config, bt_config)

    df = data_cache[pair]
    timestamps = None
    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], utc=True)

    trades = backtester.run(pair, df, timestamps)

    analyzer = PerformanceAnalyzer()
    days = max(len(df) // 210, 1)
    metrics = analyzer.compute(trades, trading_days=days)

    evaluator = GoNoGoEvaluator()
    go_result = evaluator.backtest_to_paper(metrics)

    return {
        "pair": pair,
        "scenario": scenario_name,
        "trades": metrics.total_trades,
        "trades_per_day": round(metrics.total_trades / max(days, 1), 2),
        "win_rate": round(metrics.win_rate * 100, 1),
        "sharpe": round(metrics.sharpe_ratio, 2),
        "max_dd": round(metrics.max_drawdown_pct, 2),
        "profit_factor": round(metrics.profit_factor, 2),
        "total_pnl": round(metrics.total_pnl_pips, 1),
        "avg_pnl": round(metrics.avg_pnl_pips, 2),
        "avg_winner": round(metrics.avg_winner_pips, 2),
        "avg_loser": round(metrics.avg_loser_pips, 2),
        "slippage_impact": round(metrics.slippage_impact_pct, 1),
        "tp_count": metrics.tp_hit_count,
        "sl_count": metrics.sl_hit_count,
        "time_stop": metrics.time_stop_count,
        "borderline": metrics.borderline_count,
        "consec_losses": metrics.max_consecutive_losses,
        "verdict": go_result.verdict.value,
        "failed": ", ".join(go_result.failed_criteria) if go_result.failed_criteria else "",
        "stop": ", ".join(go_result.stop_criteria) if go_result.stop_criteria else "",
        # Parameters used
        "p_slope": params["ema_slope_thr"],
        "p_compress": params["compression_atr_mult"],
        "p_tp": params["tp_R"],
        "p_sl": params["sl_atr"],
        "p_spread": params["spread"],
        "p_model_b": params.get("model_b_enabled", False),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-pair scenario backtester")
    parser.add_argument("--pairs", nargs="+", default=PAIRS)
    parser.add_argument("--quick", action="store_true",
                        help="Run only key scenarios (faster)")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run a single scenario only")
    args = parser.parse_args()

    pairs = args.pairs
    if args.scenario:
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    elif args.quick:
        scenarios = {k: SCENARIOS[k] for k in QUICK_SCENARIOS if k in SCENARIOS}
    else:
        scenarios = SCENARIOS

    # Load data once
    print("Loading data...")
    data_cache = {}
    for pair in pairs:
        path = DATA_FILES.get(pair)
        if not path or not Path(path).exists():
            print(f"  WARNING: {path} not found — skipping {pair}")
            continue
        data_cache[pair] = pd.read_csv(path)
        print(f"  {pair}: {len(data_cache[pair])} candles")

    if not data_cache:
        print("ERROR: No data files found. Run fetch_historical.py first.")
        sys.exit(1)

    # Run all combinations
    total = len(data_cache) * len(scenarios)
    print(f"\nRunning {total} backtests ({len(data_cache)} pairs × {len(scenarios)} scenarios)...\n")

    results = []
    done = 0
    start_all = time.time()

    for scenario_name, overrides in scenarios.items():
        for pair in data_cache:
            done += 1
            t0 = time.time()
            try:
                result = run_one(pair, scenario_name, overrides, data_cache)
                results.append(result)
                elapsed = time.time() - t0
                verdict_icon = "GO" if result["verdict"] == "GO" else "NO" if result["verdict"] == "NO_GO" else "XX"
                print(f"  [{done}/{total}] {pair} | {scenario_name:25s} | "
                      f"{result['trades']:3d} trades | "
                      f"WR={result['win_rate']:5.1f}% | "
                      f"PnL={result['total_pnl']:7.1f} | "
                      f"Slip={result['slippage_impact']:5.1f}% | "
                      f"{verdict_icon} | {elapsed:.0f}s")
            except Exception as e:
                print(f"  [{done}/{total}] {pair} | {scenario_name:25s} | ERROR: {e}")
                results.append({
                    "pair": pair, "scenario": scenario_name,
                    "trades": 0, "verdict": "ERROR",
                    "failed": str(e),
                })

    total_time = time.time() - start_all
    print(f"\nCompleted {done} backtests in {total_time/60:.1f} minutes")

    # Save results
    df_results = pd.DataFrame(results)
    output_path = Path("logs/scenario_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # ============================================================
    #  ANALYSIS
    # ============================================================
    print("\n" + "=" * 80)
    print("  ANALYSIS SUMMARY")
    print("=" * 80)

    # 1. GO scenarios
    go_df = df_results[df_results["verdict"] == "GO"]
    print(f"\n--- GO Scenarios: {len(go_df)} of {len(df_results)} ---")
    if len(go_df) > 0:
        for _, row in go_df.sort_values("total_pnl", ascending=False).head(15).iterrows():
            print(f"  {row['pair']:8s} | {row['scenario']:25s} | "
                  f"trades={row['trades']:3d} | PnL={row['total_pnl']:7.1f} | "
                  f"Slip={row['slippage_impact']:5.1f}%")

    # 2. Best per pair
    print(f"\n--- Best PnL per Pair (GO only) ---")
    if len(go_df) > 0:
        for pair in pairs:
            pair_go = go_df[go_df["pair"] == pair]
            if len(pair_go) > 0:
                best = pair_go.loc[pair_go["total_pnl"].idxmax()]
                print(f"  {pair}: {best['scenario']} → "
                      f"PnL={best['total_pnl']:.1f} pips, "
                      f"{best['trades']} trades, "
                      f"Slip={best['slippage_impact']:.1f}%")

    # 3. STOP scenarios (no edge)
    stop_df = df_results[df_results["verdict"] == "STOP"]
    if len(stop_df) > 0:
        print(f"\n--- STOP Scenarios (no edge): {len(stop_df)} ---")
        for _, row in stop_df.head(10).iterrows():
            print(f"  {row['pair']:8s} | {row['scenario']:25s} | "
                  f"reason: {row.get('stop', row.get('failed', ''))}")

    # 4. Sensitivity analysis
    print(f"\n--- Sensitivity: EMA Slope ---")
    for s in ["slope_0.10", "slope_0.12", "slope_0.15", "slope_0.18", "slope_0.20"]:
        sub = df_results[df_results["scenario"] == s]
        if len(sub) > 0:
            avg_trades = sub["trades"].mean()
            avg_pnl = sub["total_pnl"].mean()
            avg_slip = sub["slippage_impact"].mean()
            go_count = (sub["verdict"] == "GO").sum()
            print(f"  {s:15s}: avg_trades={avg_trades:5.0f} | "
                  f"avg_PnL={avg_pnl:7.1f} | avg_Slip={avg_slip:5.1f}% | "
                  f"GO={go_count}/{len(sub)}")

    print(f"\n--- Sensitivity: Compression ---")
    for s in ["compress_1.5", "compress_2.0", "compress_2.5", "compress_3.0"]:
        sub = df_results[df_results["scenario"] == s]
        if len(sub) > 0:
            avg_trades = sub["trades"].mean()
            avg_pnl = sub["total_pnl"].mean()
            avg_slip = sub["slippage_impact"].mean()
            go_count = (sub["verdict"] == "GO").sum()
            print(f"  {s:15s}: avg_trades={avg_trades:5.0f} | "
                  f"avg_PnL={avg_pnl:7.1f} | avg_Slip={avg_slip:5.1f}% | "
                  f"GO={go_count}/{len(sub)}")

    print(f"\n--- Sensitivity: TP Target ---")
    for s in ["tp_1.0", "tp_1.2", "tp_1.5", "tp_1.7", "tp_2.0", "tp_2.5"]:
        sub = df_results[df_results["scenario"] == s]
        if len(sub) > 0:
            avg_trades = sub["trades"].mean()
            avg_pnl = sub["total_pnl"].mean()
            avg_slip = sub["slippage_impact"].mean()
            go_count = (sub["verdict"] == "GO").sum()
            print(f"  {s:15s}: avg_trades={avg_trades:5.0f} | "
                  f"avg_PnL={avg_pnl:7.1f} | avg_Slip={avg_slip:5.1f}% | "
                  f"GO={go_count}/{len(sub)}")

    print(f"\n--- Sensitivity: Spread Cost ---")
    for s in ["spread_0.2", "spread_0.3", "spread_0.5", "spread_0.8"]:
        sub = df_results[df_results["scenario"] == s]
        if len(sub) > 0:
            avg_trades = sub["trades"].mean()
            avg_pnl = sub["total_pnl"].mean()
            avg_slip = sub["slippage_impact"].mean()
            go_count = (sub["verdict"] == "GO").sum()
            print(f"  {s:15s}: avg_trades={avg_trades:5.0f} | "
                  f"avg_PnL={avg_pnl:7.1f} | avg_Slip={avg_slip:5.1f}% | "
                  f"GO={go_count}/{len(sub)}")

    print(f"\n--- Profiles ---")
    for s in ["conservative", "balanced", "aggressive"]:
        sub = df_results[df_results["scenario"] == s]
        if len(sub) > 0:
            for _, row in sub.iterrows():
                print(f"  {s:15s} {row['pair']:8s}: "
                      f"trades={row['trades']:3d} | PnL={row['total_pnl']:7.1f} | "
                      f"Slip={row['slippage_impact']:5.1f}% | {row['verdict']}")

    # 5. Model B impact
    print(f"\n--- Model B Impact ---")
    for s in ["baseline", "model_b_default", "model_b_loose"]:
        sub = df_results[df_results["scenario"] == s]
        if len(sub) > 0:
            for _, row in sub.iterrows():
                print(f"  {s:20s} {row['pair']:8s}: "
                      f"trades={row['trades']:3d} | PnL={row['total_pnl']:7.1f} | "
                      f"Slip={row['slippage_impact']:5.1f}% | {row['verdict']}")

    # 6. Issues detected
    print(f"\n--- Issues & Warnings ---")
    issues = []

    # Check borderline ratio
    for _, row in df_results.iterrows():
        if row.get("trades", 0) > 0 and row.get("borderline", 0) > 0:
            bl_ratio = row["borderline"] / row["trades"]
            if bl_ratio > 0.95:
                issues.append(
                    f"  HIGH BORDERLINE: {row['pair']} {row['scenario']} — "
                    f"{row['borderline']}/{row['trades']} ({bl_ratio:.0%}) borderline")

    # Check slippage sensitivity
    base_sub = df_results[df_results["scenario"] == "baseline"]
    high_spread = df_results[df_results["scenario"] == "spread_0.8"]
    for pair in pairs:
        b = base_sub[base_sub["pair"] == pair]
        h = high_spread[high_spread["pair"] == pair]
        if len(b) > 0 and len(h) > 0:
            b_pnl = b.iloc[0]["total_pnl"]
            h_pnl = h.iloc[0]["total_pnl"]
            if b_pnl > 0:
                drop = (b_pnl - h_pnl) / b_pnl * 100
                if drop > 50:
                    issues.append(
                        f"  FRAGILE EDGE: {pair} loses {drop:.0f}% PnL "
                        f"when spread goes from 0.3 to 0.8")

    # Check Model B
    for pair in pairs:
        base = df_results[(df_results["pair"] == pair) & (df_results["scenario"] == "baseline")]
        mb = df_results[(df_results["pair"] == pair) & (df_results["scenario"] == "model_b_loose")]
        if len(base) > 0 and len(mb) > 0:
            diff = mb.iloc[0]["trades"] - base.iloc[0]["trades"]
            if diff < 10:
                issues.append(
                    f"  MODEL B WEAK: {pair} — only +{diff} trades with Model B")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  No major issues detected.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
