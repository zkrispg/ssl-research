#!/usr/bin/env python3
"""Plan the final GCA Conformer manuscript table update.

This script is intentionally read-only. It validates that the final deterministic
GCA Conformer result files are present, computes the paired contrasts needed by
the manuscript, and prints the exact manuscript locations that should be updated.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parents[1]
RUNS_DIR = REPO_ROOT / "runs"

FINAL_MD = RUNS_DIR / "gca_conformer_det_seld_final.md"
FINAL_CSV = RUNS_DIR / "gca_conformer_det_seld_final.csv"
FINAL_JSON = RUNS_DIR / "gca_conformer_det_seld_final.json"

TASKS = {
    "161": "MIC Conformer full",
    "162": "MIC Conformer no_geom",
    "171": "FOA Conformer full",
    "172": "FOA Conformer no_geom",
}

PAIRS = [
    ("161", "162", "MIC Conformer"),
    ("171", "172", "FOA Conformer"),
]

TABLE_TARGETS = [
    "sections/04_experiments.tex: tab:dissociation Conformer rows",
    "sections/04_experiments.tex: tab:cross in-domain Conformer column",
    "sections/04_experiments.tex: tab:convbias GCA Conformer row",
    "sections/07_appendix.tex: tab:allcells Conformer rows",
]


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NA":
        return None
    return float(text)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values)


def paired_stats(deltas: list[float]) -> dict[str, float | str]:
    n = len(deltas)
    avg = mean(deltas)
    sd = sample_std(deltas)
    if n < 2 or sd == 0:
        return {"n": n, "mean": avg, "t": "NA", "p": "NA", "dz": "NA"}
    t_value = avg / (sd / math.sqrt(n))
    dz = avg / sd

    p_value: float | str = "NA"
    try:
        from scipy import stats  # type: ignore

        p_value = float(stats.ttest_1samp(deltas, 0.0).pvalue)
    except Exception:
        pass

    return {"n": n, "mean": avg, "t": t_value, "p": p_value, "dz": dz}


def format_number(value: float | str, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    return f"{value:+.{digits}f}" if value >= 0 else f"{value:.{digits}f}"


def load_rows() -> list[dict[str, str]]:
    missing = [p for p in (FINAL_MD, FINAL_CSV, FINAL_JSON) if not p.exists()]
    if missing:
        print("Missing final files:")
        for path in missing:
            print(f"  - {path.relative_to(REPO_ROOT)}")
        print()
        print("Copy these files into runs/ before updating manuscript numbers.")
        raise SystemExit(1)

    with FINAL_JSON.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "rows" not in payload:
        raise SystemExit("Final JSON does not contain a top-level 'rows' key.")

    with FINAL_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Final CSV has no rows.")
    return rows


def main() -> int:
    rows = load_rows()

    by_task_seed: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        task = str(row.get("task", "")).strip()
        if task not in TASKS:
            continue
        if row.get("status") != "complete":
            continue
        seed_text = row.get("seed")
        if seed_text is None or str(seed_text).strip() == "":
            continue
        by_task_seed[(task, int(seed_text))] = row

    print("Final GCA Conformer table update plan")
    print()

    print("Per-task summaries")
    for task, label in TASKS.items():
        task_rows = [row for (t, _), row in by_task_seed.items() if t == task]
        seeds = sorted(seed for (t, seed) in by_task_seed if t == task)
        print(f"- {task} {label}: n={len(task_rows)}, seeds={seeds}")
        if len(task_rows) != 5:
            print("  BLOCKER: expected 5 complete seeds before manuscript update.")
            continue
        for metric in ("seld", "f20", "doae", "rde"):
            values = [as_float(row.get(metric)) for row in task_rows]
            if any(v is None for v in values):
                print(f"  BLOCKER: missing {metric} values.")
                continue
            numeric = [float(v) for v in values if v is not None]
            print(f"  {metric}: {mean(numeric):.3f} +/- {sample_std(numeric):.3f}")

    print()
    print("Paired contrasts: full - no_geom")
    for full_task, no_geom_task, label in PAIRS:
        common_seeds = sorted(
            seed
            for (task, seed) in by_task_seed
            if task == full_task and (no_geom_task, seed) in by_task_seed
        )
        print(f"- {label}: paired n={len(common_seeds)}, seeds={common_seeds}")
        if len(common_seeds) != 5:
            print("  BLOCKER: expected 5 paired seeds before manuscript update.")
            continue
        for metric in ("seld", "f20", "doae", "rde"):
            deltas: list[float] = []
            for seed in common_seeds:
                full = as_float(by_task_seed[(full_task, seed)].get(metric))
                no_geom = as_float(by_task_seed[(no_geom_task, seed)].get(metric))
                if full is None or no_geom is None:
                    raise SystemExit(f"Missing {metric} for pair {label} seed {seed}.")
                deltas.append(full - no_geom)
            stats = paired_stats(deltas)
            p = stats["p"]
            p_text = "NA" if isinstance(p, str) else f"{p:.3f}"
            print(
                "  "
                f"delta {metric}: mean={format_number(stats['mean'])}, "
                f"t={format_number(stats['t'])}, p={p_text}, dz={format_number(stats['dz'])}"
            )

    print()
    print("Manuscript targets to update after reviewing these numbers")
    for target in TABLE_TARGETS:
        print(f"- {target}")

    print()
    print("After editing, run:")
    print("  ./paper/applied-acoustics/check_submission_state.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
