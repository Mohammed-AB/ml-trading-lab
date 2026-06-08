#!/usr/bin/env python3
"""Process Historical Data — Generates agent-readable intelligence from raw OHLCV data.

Reads M1, M5, and Daily CSV files for all 8 pairs and outputs markdown
files with processed statistics that agents use as their knowledge base.

Usage:
    python tools/process_historical.py --data-dir ~/Downloads/FX-1-Minute-Data-master/forex_data

Output: data/brain/historical/*.md
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD",
    "EURGBP", "USDCHF", "USDCAD", "NZDUSD",
]

OANDA_PAIRS = {
    "EURUSD": "EUR_USD", "USDJPY": "USD_JPY", "GBPUSD": "GBP_USD",
    "AUDUSD": "AUD_USD", "EURGBP": "EUR_GBP", "USDCHF": "USD_CHF",
    "USDCAD": "USD_CAD", "NZDUSD": "NZD_USD",
}


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df


def pip_mult(pair: str) -> float:
    return 100.0 if "JPY" in pair else 10000.0


def compute_session_stats(df, pair):
    mult = pip_mult(pair)
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["range_pips"] = (df["high"] - df["low"]) * mult
    sessions = {
        "Asian (22:00-07:00)": df[df["hour"].isin(list(range(22, 24)) + list(range(0, 7)))],
        "London (07:00-12:00)": df[df["hour"].isin(range(7, 12))],
        "NY Overlap (12:00-16:00)": df[df["hour"].isin(range(12, 16))],
        "NY (16:00-21:00)": df[df["hour"].isin(range(16, 21))],
    }
    stats = {}
    for name, sdf in sessions.items():
        if len(sdf) == 0:
            continue
        stats[name] = {
            "avg_range_pips": round(sdf["range_pips"].mean(), 2),
            "median_range_pips": round(sdf["range_pips"].median(), 2),
            "candle_count": len(sdf),
        }
    return stats


def compute_hourly_stats(df, pair):
    mult = pip_mult(pair)
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["range_pips"] = (df["high"] - df["low"]) * mult
    hourly = {}
    for h in range(24):
        hdf = df[df["hour"] == h]
        if len(hdf) == 0:
            continue
        hourly[f"{h:02d}:00"] = {
            "avg_range": round(hdf["range_pips"].mean(), 2),
            "count": len(hdf),
        }
    return hourly


def compute_monthly_stats(df_daily, pair):
    mult = pip_mult(pair)
    df = df_daily.copy()
    df["month"] = df["datetime"].dt.month
    df["daily_range"] = (df["high"] - df["low"]) * mult
    df["daily_change"] = (df["close"] - df["open"]) * mult
    months = {}
    names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        mdf = df[df["month"] == m]
        if len(mdf) == 0:
            continue
        avg_change = mdf["daily_change"].mean()
        months[names[m]] = {
            "avg_daily_range": round(mdf["daily_range"].mean(), 1),
            "avg_daily_change": round(avg_change, 1),
            "direction": "bullish" if avg_change > 0 else "bearish",
        }
    return months


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_rsi_reliability(df, pair):
    mult = pip_mult(pair)
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"])
    df["future_5"] = df["close"].shift(-5)
    df["future_10"] = df["close"].shift(-10)
    df = df.dropna(subset=["rsi", "future_10"])
    results = {}
    for label, threshold, direction in [
        ("RSI < 25", 25, "long"), ("RSI < 30", 30, "long"),
        ("RSI > 70", 70, "short"), ("RSI > 75", 75, "short"),
    ]:
        if direction == "long":
            sub = df[df["rsi"] < threshold]
            if len(sub) < 50:
                continue
            bounce_5 = ((sub["future_5"] - sub["close"]) * mult).mean()
            bounce_10 = ((sub["future_10"] - sub["close"]) * mult).mean()
            win_rate = (sub["future_5"] > sub["close"]).mean()
        else:
            sub = df[df["rsi"] > threshold]
            if len(sub) < 50:
                continue
            bounce_5 = ((sub["close"] - sub["future_5"]) * mult).mean()
            bounce_10 = ((sub["close"] - sub["future_10"]) * mult).mean()
            win_rate = (sub["future_5"] < sub["close"]).mean()
        results[label] = {
            "occurrences": len(sub),
            "avg_move_5_candles": round(bounce_5, 2),
            "avg_move_10_candles": round(bounce_10, 2),
            "win_rate_5_candles": round(win_rate * 100, 1),
        }
    return results


def compute_ema_crossover_backtest(df, pair, fast=5, slow=40):
    mult = pip_mult(pair)
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["cross_up"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["cross_down"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (df["ema_fast"] < df["ema_slow"])

    trades = []
    cross_indices = df.index[(df["cross_up"] | df["cross_down"]) & df["atr"].notna()].tolist()

    for idx in cross_indices:
        row = df.loc[idx]
        direction = "long" if row["cross_up"] else "short"
        entry = row["close"]
        sl_dist = row["atr"] * 1.3549
        tp_dist = sl_dist * 0.5009
        if direction == "long":
            sl, tp = entry - sl_dist, entry + tp_dist
        else:
            sl, tp = entry + sl_dist, entry - tp_dist

        end_idx = min(idx + 20, len(df) - 1)
        future = df.iloc[idx + 1:end_idx + 1]
        result = "timeout"
        for _, frow in future.iterrows():
            if direction == "long":
                if frow["low"] <= sl:
                    result = "sl_hit"; break
                if frow["high"] >= tp:
                    result = "tp_hit"; break
            else:
                if frow["high"] >= sl:
                    result = "sl_hit"; break
                if frow["low"] <= tp:
                    result = "tp_hit"; break

        pnl = tp_dist * mult if result == "tp_hit" else (-sl_dist * mult if result == "sl_hit" else 0)
        trades.append({"result": result, "pnl_pips": round(pnl, 2),
                        "hour": row["datetime"].hour if hasattr(row["datetime"], "hour") else 0})

    if not trades:
        return {"total_trades": 0}
    wins = [t for t in trades if t["result"] == "tp_hit"]
    losses = [t for t in trades if t["result"] == "sl_hit"]

    session_results = {}
    for name, hours in [("Asian", list(range(22, 24)) + list(range(0, 7))),
                         ("London", list(range(7, 12))),
                         ("NY_Overlap", list(range(12, 16)))]:
        st = [t for t in trades if t["hour"] in hours]
        if st:
            sw = [t for t in st if t["result"] == "tp_hit"]
            session_results[name] = {
                "trades": len(st),
                "win_rate": round(len(sw) / len(st) * 100, 1),
                "total_pnl": round(sum(t["pnl_pips"] for t in st), 1),
            }

    return {
        "total_trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "total_pnl_pips": round(sum(t["pnl_pips"] for t in trades), 1),
        "avg_win": round(np.mean([t["pnl_pips"] for t in wins]), 2) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_pips"] for t in losses]), 2) if losses else 0,
        "session_breakdown": session_results,
    }


def generate_pair_report(pair, data_dir, output_dir):
    oanda_pair = OANDA_PAIRS[pair]
    print(f"  Processing {pair}...")
    m5_path = data_dir / "5min" / f"{pair}_5min.csv"
    daily_path = data_dir / "daily" / "all_5pairs_daily.csv"
    lines = [f"# {oanda_pair} Historical Intelligence\n",
             f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"]

    if m5_path.exists():
        print(f"    Loading M5...")
        df_m5 = load_csv(m5_path)
        years = (df_m5["datetime"].max() - df_m5["datetime"].min()).days / 365
        lines.append(f"Data: {len(df_m5):,} M5 candles over {years:.1f} years\n")

        print(f"    Session stats...")
        for name, s in compute_session_stats(df_m5, pair).items():
            lines.append(f"- **{name}**: avg {s['avg_range_pips']} pips ({s['candle_count']:,} candles)")
        lines.append("")

        print(f"    Hourly stats...")
        hourly = compute_hourly_stats(df_m5, pair)
        lines.append("## Best Hours (M5 avg range pips)\n")
        for h, s in sorted(hourly.items(), key=lambda x: x[1]["avg_range"], reverse=True)[:5]:
            lines.append(f"- **{h} UTC**: {s['avg_range']} pips")
        lines.append("")

        print(f"    RSI reliability...")
        for label, s in compute_rsi_reliability(df_m5, pair).items():
            lines.append(f"- **{label}**: {s['win_rate_5_candles']}% bounce rate, "
                          f"avg {s['avg_move_5_candles']:+.1f} pips in 5 candles "
                          f"({s['occurrences']:,} samples)")
        lines.append("")

        print(f"    EMA crossover backtest (may take a minute)...")
        bt = compute_ema_crossover_backtest(df_m5, pair)
        lines.append("## EMA Crossover Backtest (5/40)\n")
        if bt["total_trades"] > 0:
            lines.append(f"- {bt['total_trades']} trades | {bt['win_rate']}% WR | "
                          f"{bt['total_pnl_pips']:+.0f} pips")
            lines.append(f"- Avg win: {bt['avg_win']:+.1f} | Avg loss: {bt['avg_loss']:+.1f}")
            for sn, ss in bt.get("session_breakdown", {}).items():
                lines.append(f"  - {sn}: {ss['trades']} trades, {ss['win_rate']}% WR, "
                              f"{ss['total_pnl']:+.0f} pips")
        lines.append("")

    if daily_path.exists():
        print(f"    Daily patterns...")
        df_all = pd.read_csv(daily_path)
        if "pair" in df_all.columns:
            df_d = df_all[df_all["pair"] == pair].copy()
        else:
            df_d = df_all.copy()
        if len(df_d) > 0:
            df_d["datetime"] = pd.to_datetime(df_d["datetime"])
            monthly = compute_monthly_stats(df_d, pair)
            if monthly:
                lines.append("## Monthly Patterns\n")
                for m, s in monthly.items():
                    lines.append(f"- **{m}**: {s['avg_daily_range']:.0f} pips range, "
                                  f"{s['direction']} ({s['avg_daily_change']:+.1f}/day)")
                lines.append("")

    (output_dir / f"{oanda_pair}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"    Done: {oanda_pair}.md")


def generate_strategies_report(output_dir):
    lines = [
        "# Strategy Parameter Sweep — IN-SAMPLE ONLY (no proven edge)\n",
        "WARNING: these figures come from in-sample parameter sweeps and overfit.",
        "Honest out-of-sample evaluation (see docs/ARENA_LEADERBOARD.md) shows the",
        "rule strategies LOSE money — every profit-factor is below 1.0. Do NOT treat",
        "any number below as a live expectation; it is research material only.\n",
        "## Example config surfaced by the sweep: EMA Crossover\n",
        "- Params: EMA fast=5, slow=40, ATR*1.35, R:R=0.50",
        "- In-sample metrics are not predictive and are intentionally omitted.",
        "- Out-of-sample: unprofitable, like the rest of the rule set.",
        "",
        "## Note\n",
        "Per-pair and per-strategy rankings from the sweep are unstable and do not",
        "generalize. They are kept only to illustrate the research process.",
        "",
    ]
    (output_dir / "strategies.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"    Done: strategies.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="data/brain/historical")
    parser.add_argument("--pairs", nargs="*", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  HISTORICAL DATA PROCESSOR")
    print("=" * 50)

    for pair in (args.pairs or PAIRS):
        p = pair.upper()
        if p in PAIRS:
            generate_pair_report(p, data_dir, output_dir)

    generate_strategies_report(output_dir)
    print("\nDone! Files in:", output_dir)


if __name__ == "__main__":
    main()
