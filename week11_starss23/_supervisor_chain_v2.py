"""Tier-1 + Tier-2 extension supervisor.

Chains the three new GPU queues, then runs the full grid of pairwise
t-tests, then runs the three CPU-only analysis scripts. All steps log to
``runs/supervisor_v2.log``. Each step is a separate ``subprocess.run`` so
nothing competes for the GPU.

Stages (sequential):

    G1: SELDnet baseline N = 5 extension          (~3.3 h GPU)
    G2: Weak SpecAug control (10 cells)           (~ 8.0 h GPU)
    G3: Capacity sweep (18 cells, xs / l / xl)    (~15.0 h GPU)
    P : 10 pairwise t-tests                        (~  1 min CPU)
    A : 3 analysis scripts                         (~  3 min CPU)

Total ~26 GPU-hours + a few minutes CPU. Resume-aware: each child queue
already skips cells whose artifacts exist, so re-running the supervisor
on a clean machine after a crash is safe.

Usage:
    python -m week11_starss23._supervisor_chain_v2
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
LOG = RUNS / "supervisor_v2.log"


def _log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run(name: str, cmd: list[str]) -> int:
    _log(f"[{name}] launching: {' '.join(cmd)}")
    log_path = RUNS / f"supervisor_v2_{name}.log"
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
    _run(f"pairwise_{name}", cmd)


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    _log("=" * 72)
    _log("supervisor v2 (Tier 1 + Tier 2 extension) started")

    # ----- GPU stages ---------------------------------------------------
    rc = _run("queue_g1", [str(PYEXE), "-u", "-m",
                            "week11_starss23.run_extension_queues",
                            "--queue", "G1"])
    if rc != 0:
        _log(f"G1 returned {rc}; continuing anyway.")

    rc = _run("queue_g2", [str(PYEXE), "-u", "-m",
                            "week11_starss23.run_extension_queues",
                            "--queue", "G2"])
    if rc != 0:
        _log(f"G2 returned {rc}; continuing anyway.")

    rc = _run("queue_g3", [str(PYEXE), "-u", "-m",
                            "week11_starss23.run_extension_queues",
                            "--queue", "G3"])
    if rc != 0:
        _log(f"G3 returned {rc}; continuing anyway.")

    _log("All GPU queues finished. Running pairwise t-tests (CPU)...")

    # ----- Pairwise t-tests ---------------------------------------------
    # SELDnet N = 5 cross-system comparisons
    _run_pairwise(
        "seldnet_vs_no_geom_n5",
        a="seldnet_official:baseline_mc8_inmem",
        b="no_geom:mc8_inmem",
        seeds=[0, 1, 2, 3, 4],
    )
    _run_pairwise(
        "seldnet_vs_full_n5",
        a="seldnet_official:baseline_mc8_inmem",
        b="full:mc8_inmem",
        seeds=[0, 1, 2, 3, 4],
    )
    _run_pairwise(
        "seldnet_specaug_ablation_n5",
        a="seldnet_official:baseline_mc8_inmem",
        b="seldnet_official:baseline_mc8_inmem_specaug",
        seeds=[0, 1, 2, 3, 4],
    )

    # Weak SpecAug ablations (vs vanilla, both variants)
    _run_pairwise(
        "no_geom_weak_specaug_ablation",
        a="no_geom:mc8_inmem",
        b="no_geom:mc8_inmem_specaug_weak",
        seeds=[0, 1, 2, 3, 4],
    )
    _run_pairwise(
        "full_weak_specaug_ablation",
        a="full:mc8_inmem",
        b="full:mc8_inmem_specaug_weak",
        seeds=[0, 1, 2, 3, 4],
    )

    # SpecAug strength comparison (weak vs strong, both variants)
    _run_pairwise(
        "no_geom_specaug_weak_vs_strong",
        a="no_geom:mc8_inmem_specaug_weak",
        b="no_geom:mc8_inmem_specaug",
        seeds=[0, 1, 2, 3, 4],
    )
    _run_pairwise(
        "full_specaug_weak_vs_strong",
        a="full:mc8_inmem_specaug_weak",
        b="full:mc8_inmem_specaug",
        seeds=[0, 1, 2, 3, 4],
    )

    # Capacity-stratified geometry ablations (no_geom vs full at each size)
    _run_pairwise(
        "geom_ablation_xs",
        a="no_geom_xs:cap_xs_mc8_inmem",
        b="full_xs:cap_xs_mc8_inmem",
        seeds=[0, 1, 2],
    )
    _run_pairwise(
        "geom_ablation_l",
        a="no_geom_l:cap_l_mc8_inmem",
        b="full_l:cap_l_mc8_inmem",
        seeds=[0, 1, 2],
    )
    _run_pairwise(
        "geom_ablation_xl",
        a="no_geom_xl:cap_xl_mc8_inmem",
        b="full_xl:cap_xl_mc8_inmem",
        seeds=[0, 1, 2],
    )

    _log("Pairwise t-tests done. Running analysis scripts...")

    # ----- Analysis scripts (CPU) ---------------------------------------
    _run("analysis_best_thr", [str(PYEXE), "-u", "-m",
                                "week11_starss23._analysis_best_threshold"])
    _run("analysis_bootstrap", [str(PYEXE), "-u", "-m",
                                 "week11_starss23._analysis_bootstrap"])
    _run("analysis_perclass", [str(PYEXE), "-u", "-m",
                                "week11_starss23._analysis_perclass"])

    _log("supervisor v2 chain DONE.")


if __name__ == "__main__":
    main()
