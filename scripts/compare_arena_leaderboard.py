#!/usr/bin/env python3
"""Merge two strategy_arena JSON summaries (e.g. IS + OOS) into a ranked comparison table.

Usage:
  python scripts/compare_arena_leaderboard.py is.json oos.json
  python scripts/compare_arena_leaderboard.py is.json oos.json -o docs/ARENA_LEADERBOARD.md

Expects JSON shaped like run_arena output with key \"summary\"[\"strategies\"][name] =
  {n, wr, pf, total_pips, avg_pips, wins}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_strategies(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    summ = data.get("summary") or {}
    return summ.get("strategies") or {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare IS vs OOS arena JSON summaries.")
    ap.add_argument("is_json", type=Path, help="In-sample JSON (--json from --in-sample run)")
    ap.add_argument("oos_json", type=Path, help="OOS JSON (default run)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Write markdown table")
    args = ap.parse_args()

    is_s = _load_strategies(args.is_json)
    oos_s = _load_strategies(args.oos_json)
    names = sorted(set(is_s) | set(oos_s), key=str)

    out_lines: list[str] = [
        "# Strategy arena: IS vs OOS\n",
        "| Strategy | IS N | IS PF | IS Pips | OOS N | OOS PF | OOS Pips |",
        "|----------|------|-------|---------|-------|--------|----------|",
    ]
    rank_rows: list[tuple[float, str]] = []
    for name in names:
        a = is_s.get(name, {})
        b = oos_s.get(name, {})
        is_n = int(a.get("n", 0))
        oos_n = int(b.get("n", 0))
        is_pf = float(a.get("pf", 0) or 0)
        oos_pf = float(b.get("pf", 0) or 0)
        is_p = float(a.get("total_pips", 0) or 0)
        oos_p = float(b.get("total_pips", 0) or 0)
        out_lines.append(
            f"| {name} | {is_n} | {is_pf:.2f} | {is_p:+.0f} | {oos_n} | {oos_pf:.2f} | {oos_p:+.0f} |"
        )
        score = (is_pf * min(is_n, 500) ** 0.5 + oos_pf * min(oos_n, 200) ** 0.5) / 2.0
        rank_rows.append((score, name))

    out_lines.append("\n## Ranked by blended score (IS+OOS PF with sqrt n cap)\n")
    for sc, name in sorted(rank_rows, key=lambda x: -x[0])[:80]:
        out_lines.append(f"- **{name}** — {sc:.2f}")

    text = "\n".join(out_lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
