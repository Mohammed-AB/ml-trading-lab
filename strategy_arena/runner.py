"""Run book + research strategies, filter in-sample / OOS, aggregate leaderboard."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backtest_strategies import STRATEGIES, run_strategy, simulate_trades_vec  # noqa: E402

from scalp_mode.ml.bar_features import (  # noqa: E402
    SPREAD_PIPS_DEFAULT,
    pip_for_pair,
    spread_half_price,
)

from .config import OOS_START, V2_PAIRS  # noqa: E402
from .research import RESEARCH_STRATEGIES  # noqa: E402
from .research_pdf import RESEARCH_PDF_STRATEGIES  # noqa: E402
from .research_v2 import RESEARCH_V2_STRATEGIES  # noqa: E402

RESEARCH_ALL = (
    tuple(RESEARCH_STRATEGIES)
    + tuple(RESEARCH_V2_STRATEGIES)
    + tuple(RESEARCH_PDF_STRATEGIES)
)


def _spread_price_full(pair: str) -> float:
    pip = pip_for_pair(pair)
    return SPREAD_PIPS_DEFAULT.get(pair, 1.5) * pip


def run_research_strategy(
    name: str,
    sig_func,
    df: pd.DataFrame,
    pair: str,
    pip: float,
    spread: float,
    max_bars: int,
) -> list[dict]:
    result = sig_func(df, pair, pip, spread)
    idx, dirs, ent, sls, tps = result
    if not idx:
        return []
    h_arr = df["high"].values
    l_arr = df["low"].values
    c_arr = df["close"].values
    hours = df["hour"].values if "hour" in df.columns else np.zeros(len(df))
    tss = df["timestamp"].values
    hsp = spread_half_price(pair, SPREAD_PIPS_DEFAULT.get(pair, 1.5))
    sim = simulate_trades_vec(
        h_arr,
        l_arr,
        c_arr,
        idx,
        dirs,
        ent,
        sls,
        tps,
        max_bars,
        pip,
        half_spread=hsp,
    )
    trades: list[dict] = []
    for k in range(len(idx)):
        trades.append(
            {
                "strategy": name,
                "pair": pair,
                "direction": "long" if dirs[k] == 1 else "short",
                "hour": int(hours[idx[k]]),
                "pnl_pips": float(sim[k]["pnl_pips"]),
                "exit_reason": str(sim[k]["exit_reason"]),
                "bars_held": int(sim[k]["bars_held"]),
                "sl_pips": round(abs(ent[k] - sls[k]) / pip, 1),
                "tp_pips": round(abs(ent[k] - tps[k]) / pip, 1),
                "timestamp": str(tss[idx[k]]),
            }
        )
    return trades


def _parse_ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def filter_window(
    trades: list[dict],
    oos: bool,
    oos_start: pd.Timestamp,
    full: bool = False,
) -> list[dict]:
    if full:
        return trades
    oos_utc = pd.Timestamp(oos_start)
    if oos_utc.tz is None:
        oos_utc = oos_utc.tz_localize("UTC")
    else:
        oos_utc = oos_utc.tz_convert("UTC")
    out: list[dict] = []
    for t in trades:
        ts = pd.Timestamp(t["timestamp"])
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        if oos and ts >= oos_utc:
            out.append(t)
        elif (not oos) and ts < oos_utc:
            out.append(t)
    return out


def _metrics(trs: list[dict]) -> dict[str, Any]:
    if not trs:
        return {
            "n": 0,
            "wr": 0.0,
            "pf": 0.0,
            "total_pips": 0.0,
            "avg_pips": 0.0,
            "wins": 0,
        }
    pnl = [t["pnl_pips"] for t in trs]
    n = len(pnl)
    wins = sum(1 for p in pnl if p > 0)
    gp = sum(p for p in pnl if p > 0)
    gl = abs(sum(p for p in pnl if p <= 0))
    return {
        "n": n,
        "wr": 100.0 * wins / n,
        "pf": float(gp / gl) if gl else 99.0,
        "total_pips": float(sum(pnl)),
        "avg_pips": float(np.mean(pnl)),
        "wins": wins,
    }


def run_arena(
    data_dir: Path,
    oos: bool = True,
    only_research: bool = False,
    only_book: bool = False,
    full: bool = False,
) -> dict[str, Any]:
    from .loader import list_available_pairs, load_oanda_m1, prepare_m5  # noqa: WPS433

    oos_start = _parse_ts(OOS_START)
    pairs = list_available_pairs(data_dir)
    if not pairs:
        return {"error": f"No OANDA CSVs in {data_dir} — run scripts/fetch_historical.py --v2"}

    all_trades: list[dict] = []
    if full:
        oos_str = "full range (no IS/OOS split)"
    else:
        oos_str = "out-of-sample" if oos else "in-sample"

    for pair in sorted(p for p in pairs if p in V2_PAIRS):
        df_m1 = load_oanda_m1(data_dir, pair)
        if df_m1.empty or len(df_m1) < 2000:
            continue
        df_m5 = prepare_m5(df_m1, pair)
        if df_m5.empty or len(df_m5) < 200:
            continue
        pip = pip_for_pair(pair)
        spread = _spread_price_full(pair)

        if not only_research:
            for sname, sfunc, mb in STRATEGIES:
                if sname == "S15_VWAP":
                    tr = run_strategy(
                        sname, sfunc, df_m1, pair, pip, spread, mb
                    )
                else:
                    tr = run_strategy(
                        sname, sfunc, df_m5, pair, pip, spread, mb
                    )
                all_trades.extend(filter_window(tr, oos, oos_start, full=full))
        if not only_book:
            for sname, sfunc, mb in RESEARCH_ALL:
                tr = run_research_strategy(
                    sname, sfunc, df_m5, pair, pip, spread, mb
                )
                all_trades.extend(filter_window(tr, oos, oos_start, full=full))

    by_strat: dict[str, list[dict]] = defaultdict(list)
    for t in all_trades:
        by_strat[t["strategy"]].append(t)

    summary: dict[str, Any] = {
        "window": oos_str,
        "oos_start_utc": OOS_START,
        "pairs": pairs,
        "strategies": {},
    }
    rows = []
    for s, trs in sorted(by_strat.items(), key=lambda x: -len(x[1])):
        m = _metrics(trs)
        summary["strategies"][s] = m
        rows.append(
            (s, m["n"], m["wr"], m["pf"], m["total_pips"], m["avg_pips"])
        )
    rows.sort(key=lambda r: (r[3] if r[3] is not None else 0, r[4]), reverse=True)
    return {"trades": all_trades, "summary": summary, "ranked": rows}
