"""Chain three GPU queues end-to-end so the host can run unattended.

Flow:
    1. Poll ``multiseed_summary_specaug.json`` until ``status == "complete"``.
       If the file never becomes complete (e.g. queue dies), this script will
       eventually time out (default 30 hours) and abort with a clear error.
    2. Launch ``run_seldnet_baseline_queue.py --seeds 0 1 2`` and wait.
    3. Launch ``run_seldnet_baseline_queue.py --seeds 0 1 2 --specaug`` and wait.
    4. Run all relevant ``_pairwise_ttest.py`` comparisons and dump JSON
       artifacts the paper can quote.

Logs everything to ``runs/supervisor.log``. Uses ``subprocess.run`` so each
step blocks until the previous one finishes -- no GPU contention.

Usage:
    python -m week11_starss23._supervisor_chain
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO = Path("D:/ssl-research")
PYEXE = REPO / "venv" / "Scripts" / "python.exe"
RUNS = REPO / "week11_starss23" / "runs"
LOG = RUNS / "supervisor.log"

SPECAUG_SUMMARY = RUNS / "multiseed_summary_specaug.json"
POLL_INTERVAL_S = 120
MAX_WAIT_HOURS = 30


def _log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _wait_for_specaug_complete() -> bool:
    """Poll the SpecAug queue summary until it reports complete or timeout."""
    deadline = time.time() + MAX_WAIT_HOURS * 3600
    last_done = -1
    while time.time() < deadline:
        if SPECAUG_SUMMARY.exists():
            try:
                summary = json.loads(SPECAUG_SUMMARY.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
            n_train = len(summary.get("trainings", []))
            n_eval = len(summary.get("evals", []))
            status = summary.get("status", "?")
            if n_train != last_done:
                _log(f"SpecAug progress: trainings={n_train} evals={n_eval} status={status}")
                last_done = n_train
            if status == "complete":
                _log("SpecAug queue COMPLETE -- chaining to SELDnet baseline.")
                return True
        time.sleep(POLL_INTERVAL_S)
    _log("ERROR: SpecAug queue never reported complete within timeout.")
    return False


def _run_subprocess(name: str, cmd: list[str]) -> int:
    _log(f"[{name}] launching: {' '.join(cmd)}")
    log_path = RUNS / f"supervisor_{name}.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, check=False,
        )
    dt = time.time() - t0
    _log(f"[{name}] returncode={proc.returncode}  elapsed={dt/3600:.2f}h  log={log_path}")
    return proc.returncode


def _run_pairwise(name: str, a: str, b: str, seeds: list[int]) -> None:
    cmd = [
        str(PYEXE), "-u", "-m", "week11_starss23._pairwise_ttest",
        "--a", a, "--b", b, "--seeds", *map(str, seeds),
    ]
    _run_subprocess(f"pairwise_{name}", cmd)


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    _log("=" * 72)
    _log("supervisor chain started")
    _log(f"  SPECAUG_SUMMARY = {SPECAUG_SUMMARY}")
    _log(f"  poll interval   = {POLL_INTERVAL_S}s")
    _log(f"  max wait        = {MAX_WAIT_HOURS}h")

    if not _wait_for_specaug_complete():
        _log("aborting chain -- SpecAug queue did not finish.")
        return

    rc = _run_subprocess(
        "seldnet_baseline_vanilla",
        [str(PYEXE), "-u", "-m", "week11_starss23.run_seldnet_baseline_queue",
         "--seeds", "0", "1", "2"],
    )
    if rc != 0:
        _log(f"vanilla seldnet queue returned non-zero ({rc}); continuing anyway.")

    rc = _run_subprocess(
        "seldnet_baseline_specaug",
        [str(PYEXE), "-u", "-m", "week11_starss23.run_seldnet_baseline_queue",
         "--seeds", "0", "1", "2", "--specaug"],
    )
    if rc != 0:
        _log(f"specaug seldnet queue returned non-zero ({rc}); continuing anyway.")

    _log("All training queues finished. Running pairwise t-tests...")

    # SpecAug ablation: vanilla vs +SpecAug, both N=5
    _run_pairwise(
        "no_geom_specaug_ablation",
        a="no_geom:mc8_inmem",
        b="no_geom:mc8_inmem_specaug",
        seeds=[0, 1, 2, 3, 4],
    )
    _run_pairwise(
        "full_specaug_ablation",
        a="full:mc8_inmem",
        b="full:mc8_inmem_specaug",
        seeds=[0, 1, 2, 3, 4],
    )

    # SELDnet baseline vs no_geom and full (N=3)
    _run_pairwise(
        "seldnet_vs_no_geom_n3",
        a="seldnet_official:baseline_mc8_inmem",
        b="no_geom:mc8_inmem",
        seeds=[0, 1, 2],
    )
    _run_pairwise(
        "seldnet_vs_full_n3",
        a="seldnet_official:baseline_mc8_inmem",
        b="full:mc8_inmem",
        seeds=[0, 1, 2],
    )

    # SELDnet vanilla vs SELDnet +SpecAug (own SpecAug ablation, N=3)
    _run_pairwise(
        "seldnet_specaug_ablation_n3",
        a="seldnet_official:baseline_mc8_inmem",
        b="seldnet_official:baseline_mc8_inmem_specaug",
        seeds=[0, 1, 2],
    )

    _log("supervisor chain DONE.")


if __name__ == "__main__":
    main()
