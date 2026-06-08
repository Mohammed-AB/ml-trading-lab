#!/usr/bin/env python3
"""One-shot cleanup of data/brain/lessons/patterns.jsonl.

Demotes historical Chief memos (stored at 0.80 as a bug) to 0.30, and
promotes Learning-Agent lessons (stored at 0.00 as a bug) to 0.60.
Backtest priors >=0.85 are preserved untouched.

Safe to run while the bot is stopped; it rewrites atomically via a temp
file. Existing backup is kept at patterns.jsonl.bak.

Usage:
    python3 scripts/rebalance_lessons.py [PATH]

If PATH omitted, defaults to data/brain/lessons/patterns.jsonl in the
current working directory.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

CHIEF_NEW_CONF = 0.30
LEARNING_NEW_CONF = 0.60


def is_chief(lesson: dict) -> bool:
    if lesson.get("source") == "chief_agent":
        return True
    scope = lesson.get("scope") or {}
    if isinstance(scope, dict) and scope.get("agent") == "chief":
        return True
    pattern = (lesson.get("pattern") or "")
    if "BOARDROOM" in pattern or "**BOARDROOM" in pattern:
        return True
    return False


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(
        "data/brain/lessons/patterns.jsonl")
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        return 1

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    print(f"Backup written to {backup}")

    total = 0
    chief_bumped = 0
    learning_bumped = 0
    priors_kept = 0
    other_kept = 0

    lines = path.read_text().splitlines()
    rewritten: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            lesson = json.loads(line)
        except Exception:
            rewritten.append(line)
            continue
        total += 1
        conf = float(lesson.get("confidence", 0) or 0)

        if conf >= 0.85:
            priors_kept += 1
        elif is_chief(lesson):
            lesson["confidence"] = CHIEF_NEW_CONF
            if "source" not in lesson:
                lesson["source"] = "chief_agent"
            chief_bumped += 1
        elif conf == 0.0:
            lesson["confidence"] = LEARNING_NEW_CONF
            if "source" not in lesson:
                lesson["source"] = "learning_agent_backfill"
            learning_bumped += 1
        else:
            other_kept += 1

        rewritten.append(json.dumps(lesson, ensure_ascii=False))

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(rewritten) + "\n")
    tmp.replace(path)

    print(f"Processed {total} lessons:")
    print(f"  Chief memos   0.80 -> {CHIEF_NEW_CONF:.2f} : {chief_bumped}")
    print(f"  Learning      0.00 -> {LEARNING_NEW_CONF:.2f} : {learning_bumped}")
    print(f"  Backtest priors (>=0.85) kept: {priors_kept}")
    print(f"  Other (middle-conf) kept     : {other_kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
