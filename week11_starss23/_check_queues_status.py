"""Print a compact status report for all currently running / completed queues.

Reports for each known queue:
  * progress JSON: how many cells done, current cell, errors
  * partial summary JSON (if any)
  * last 6 lines of the latest per-cell training log

Run with no arguments. Idempotent and safe to call concurrently with running
queues; it only reads files.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

RUNS = Path("D:/ssl-research/week11_starss23/runs")

# (display name, progress filename, summary filename, expected total cells)
QUEUE_FILES = [
    (
        "no_geom + full (N=5)",
        "multiseed_progress.json",
        "multiseed_summary.json",
        10,
    ),
    (
        "SpecAug ablation (N=5 x 2)",
        "multiseed_progress_specaug.json",
        "multiseed_summary_specaug.json",
        10,
    ),
    (
        "SELDnet baseline vanilla (N=3)",
        "multiseed_progress_seldnet_baseline_vanilla.json",
        "multiseed_summary_seldnet_baseline_vanilla.json",
        3,
    ),
    (
        "SELDnet baseline +SpecAug (N=3)",
        "multiseed_progress_seldnet_baseline_specaug.json",
        "multiseed_summary_seldnet_baseline_specaug.json",
        3,
    ),
]


def _tail(path: Path, n: int = 6) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    lines = text.splitlines()
    return lines[-n:] if len(lines) >= n else lines


def _seconds_to_human(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m = s / 60
    if m < 60:
        return f"{m:.1f}min"
    h = m / 60
    return f"{h:.2f}h"


def main() -> None:
    print(f"[status] generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    for name, prog_name, sum_name, expected_total in QUEUE_FILES:
        prog_path = RUNS / prog_name
        sum_path = RUNS / sum_name
        print("=" * 72)
        print(f"  {name}")
        print("=" * 72)

        if not prog_path.exists() and not sum_path.exists():
            print("  [not started]\n")
            continue

        if prog_path.exists():
            prog = json.loads(prog_path.read_text(encoding="utf-8"))
            done = len(prog.get("completed", []))
            errors = prog.get("errors", [])
            cur = prog.get("current")
            print(
                f"  progress: {done}/{expected_total} done  "
                f"errors: {len(errors)}  started: {prog.get('started_at', '?')}"
            )
            if cur is not None:
                print(f"  current : {cur}")
            for e in errors[-3:]:
                print(f"    err: {e}")

        if sum_path.exists():
            summary = json.loads(sum_path.read_text(encoding="utf-8"))
            status = summary.get("status", "in_progress")
            n_train = len(summary.get("trainings", []))
            n_eval = len(summary.get("evals", []))
            print(f"  summary : status={status}  trainings={n_train}  evals={n_eval}")
            best_losses = [
                t.get("best_eval_loss") for t in summary.get("trainings", [])
                if t.get("best_eval_loss") is not None
            ]
            elapsed = [
                t.get("elapsed_s") for t in summary.get("trainings", [])
                if t.get("elapsed_s")
            ]
            if best_losses:
                print(
                    f"            best_eval_loss range: "
                    f"[{min(best_losses):.5f}, {max(best_losses):.5f}]"
                )
            if elapsed:
                avg = sum(elapsed) / len(elapsed)
                remaining = (expected_total - n_train) * avg
                print(
                    f"            avg cell time: {_seconds_to_human(avg)}; "
                    f"est. remaining: {_seconds_to_human(remaining)}"
                )

        # Find the most recently modified per-cell log for this queue.
        if "specaug" in prog_name and "seldnet" not in prog_name:
            log_glob = "queue_specaug_*.log"
        elif "seldnet_baseline_vanilla" in prog_name:
            log_glob = "queue_seldnet_baseline_vanilla_*.log"
        elif "seldnet_baseline_specaug" in prog_name:
            log_glob = "queue_seldnet_baseline_specaug_*.log"
        else:
            log_glob = "queue_*_seed*.log"
        candidates = sorted(
            (p for p in RUNS.glob(log_glob)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Filter out the "stdout" / "stderr" logs (those are queue-runner stdio,
        # not per-cell training logs).
        candidates = [
            p for p in candidates
            if "stdout" not in p.name and "stderr" not in p.name
        ]
        if candidates:
            latest = candidates[0]
            mtime = time.strftime("%H:%M:%S", time.localtime(latest.stat().st_mtime))
            print(f"  latest log ({latest.name}, mtime={mtime}):")
            for line in _tail(latest, n=6):
                print(f"    | {line}")
        print()


if __name__ == "__main__":
    main()
