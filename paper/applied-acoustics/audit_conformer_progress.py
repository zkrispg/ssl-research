#!/usr/bin/env python3
"""Audit whether the GCA Conformer progress file can be promoted to final.

This is intentionally read-only. It checks the current progress CSV for the
four deterministic GCA Conformer cells needed by the manuscript.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parents[1]
PROGRESS_CSV = REPO_ROOT / "runs" / "gca_conformer_det_seld_progress.csv"

TASKS = {
    "161": "MIC Conformer full",
    "162": "MIC Conformer no_geom",
    "171": "FOA Conformer full",
    "172": "FOA Conformer no_geom",
}
EXPECTED_SEEDS = {0, 1, 2, 3, 4}


def main() -> int:
    if not PROGRESS_CSV.exists():
        print(f"Missing {PROGRESS_CSV.relative_to(REPO_ROOT)}")
        return 1

    with PROGRESS_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task", "")).strip()
        if task in TASKS:
            by_task[task].append(row)

    print("GCA Conformer progress audit")
    print(f"Source: {PROGRESS_CSV.relative_to(REPO_ROOT)}")
    print()

    blockers = 0
    for task, label in TASKS.items():
        complete_seeds = set()
        missing_or_bad: list[int] = []
        for row in by_task.get(task, []):
            seed_text = str(row.get("seed", "")).strip()
            if not seed_text:
                continue
            seed = int(seed_text)
            if row.get("status") == "complete":
                complete_seeds.add(seed)
            else:
                missing_or_bad.append(seed)

        missing = sorted(EXPECTED_SEEDS - complete_seeds)
        complete = sorted(complete_seeds)
        print(f"- {task} {label}: complete seeds={complete}")
        if missing:
            blockers += 1
            print(f"  BLOCKER: missing complete seeds={missing}")
        if missing_or_bad:
            bad = sorted(set(missing_or_bad))
            print(f"  noted non-complete rows={bad}")

    print()
    if blockers:
        print("Progress cannot be promoted to final yet.")
        print("Do not generate gca_conformer_det_seld_final.* from this file.")
        return 1

    print("Progress has all required complete seeds and can be promoted manually.")
    print("Next: generate final .md/.csv/.json, then run plan_conformer_table_update.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
