#!/usr/bin/env python3
"""Purge Asian-session backfill noise + Chief diagnostic memos from lessons.

Context:
  On 2026-04-20 the initial `rebalance_lessons.py` run promoted 270 legacy
  learning-agent entries that had been stored at 0.00 confidence up to 0.60.
  Those entries were LLM-generated observations of the bot's own Asian-
  session skip-loops ("no_compression / no_breakout / skip / avoid ..."),
  not real trading edges. Once they were at 0.60 they dominated Strategy's
  top-40 lesson window (36/40 of the prompt slots were "don't trade X"
  rules), smothering the handful of real edges from backtest priors and
  postmortems.

  This script removes the contaminated entries so Strategy's prompt snaps
  back to: backtest priors (0.85+) + postmortems + genuinely new learning
  lessons generated post-2026-04-20.

Keeps:
  - Backtest priors (confidence >= 0.85)
  - Postmortem lessons (source == "postmortem")
  - Learning-agent lessons from the live agent (source == "learning_agent")

Drops:
  - source == "learning_agent_backfill" (the 270 Asian-session noise entries)
  - source == "chief_agent"             (diagnostic boardroom memos)
  - Anything still at 0.00 confidence with no source

Safe: writes `patterns.jsonl.purgebak` before rewriting, rewrites atomically.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path


KEEP_SOURCES = {"postmortem", "learning_agent"}
DROP_SOURCES = {"learning_agent_backfill", "chief_agent"}


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(
        "data/brain/lessons/patterns.jsonl")
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        return 1

    backup = path.with_suffix(path.suffix + ".purgebak")
    shutil.copy2(path, backup)
    print(f"Backup written to {backup}")

    kept: list[str] = []
    dropped_by_source: dict[str, int] = {}
    kept_by_source: dict[str, int] = {}

    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            lesson = json.loads(raw)
        except Exception:
            kept.append(raw)
            continue

        conf = float(lesson.get("confidence", 0) or 0)
        source = lesson.get("source", "")

        if source in DROP_SOURCES:
            dropped_by_source[source] = dropped_by_source.get(source, 0) + 1
            continue

        if conf >= 0.85:
            kept_by_source["backtest_prior"] = kept_by_source.get(
                "backtest_prior", 0) + 1
            kept.append(json.dumps(lesson, ensure_ascii=False))
            continue

        if source in KEEP_SOURCES:
            kept_by_source[source] = kept_by_source.get(source, 0) + 1
            kept.append(json.dumps(lesson, ensure_ascii=False))
            continue

        if conf == 0.0:
            dropped_by_source["unsourced_zero_conf"] = dropped_by_source.get(
                "unsourced_zero_conf", 0) + 1
            continue

        # Middle-confidence, unknown source — keep defensively.
        kept_by_source["other"] = kept_by_source.get("other", 0) + 1
        kept.append(json.dumps(lesson, ensure_ascii=False))

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(kept) + "\n")
    tmp.replace(path)

    print(f"\nKept {len(kept)} lessons:")
    for k, v in sorted(kept_by_source.items()):
        print(f"  {k}: {v}")
    print("Dropped:")
    for k, v in sorted(dropped_by_source.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
