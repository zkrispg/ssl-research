"""Path B supervisor: chain capacity-N=5 + FOA + cross-dataset + analyses.

After the v2 chain finished (G1-G3 + pairwise t-tests + 3 analyses), the
v3 chain runs the Path-B additions:

    1. G5: capacity sweep N=5 extension (12 cells, ~10h GPU)
    2. G6: SELDnet FOA (5 cells, ~4-5h GPU; depends on FOA dataset being
       already extracted to STARSS23/foa_dev/)
    3. Cross-dataset evaluation on STARSS22 dev-test (zero-shot, ~30 min)
    4. Re-run analyses (best-threshold, bootstrap, perclass) including
       the new capacity N=5 cells. Perclass now reports
       Bonferroni-corrected significance and a Wilcoxon companion.
    5. New pairwise t-tests for the additional comparisons:
         - geom ablation at xs/l/xl, now at N=5
         - SELDnet MIC vs FOA (G6 paired with original SELDnet)

Designed to be **resume-safe**: if a queue's progress JSON is already
"complete", we skip it. This way we can re-run the supervisor after
crashes / restarts without re-training anything.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("D:/ssl-research")
PYEXE = REPO / "venv" / "Scripts" / "python.exe"
RUNS = REPO / "week11_starss23" / "runs"
LOG = RUNS / "supervisor_v3.log"


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run(name: str, cmd: list[str]) -> int:
    """Run a child step, redirecting its output to runs/supervisor_v3_<name>.log."""
    log_path = RUNS / f"supervisor_v3_{name}.log"
    _log(f"[{name}] launching: {' '.join(cmd)}")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = (time.time() - t0) / 3600.0
    _log(
        f"[{name}] returncode={proc.returncode}  elapsed={elapsed:.2f}h  "
        f"log={log_path}"
    )
    return proc.returncode


def _queue_complete(tag: str) -> bool:
    summary = RUNS / f"multiseed_summary_{tag}.json"
    if not summary.exists():
        return False
    try:
        d = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return d.get("status") == "complete"


def _maybe_run_queue(qkey: str, tag: str) -> int:
    if _queue_complete(tag):
        _log(f"[{qkey}] already complete -> skipping")
        return 0
    return _run(
        f"queue_{qkey.lower()}",
        [str(PYEXE), "-u", "-m", "week11_starss23.run_extension_queues",
         "--queue", qkey],
    )


def _run_pairwise(name: str, a: str, b: str, seeds: list[int]) -> None:
    cmd = [
        str(PYEXE), "-u", "-m", "week11_starss23._pairwise_ttest",
        "--a", a, "--b", b, "--seeds", *[str(s) for s in seeds],
    ]
    _run(f"pairwise_{name}", cmd)


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    _log("=" * 72)
    _log("supervisor v3 (Path B: capacity-N=5 + FOA + cross-dataset) started")

    # ---------- GPU queues ----------
    rc_g5 = _maybe_run_queue("G5", "capacity_sweep_n5_extension")
    rc_g6 = _maybe_run_queue("G6", "seldnet_foa")
    if rc_g5 != 0 or rc_g6 != 0:
        _log(f"WARNING: GPU queue returncode g5={rc_g5} g6={rc_g6}")

    # ---------- Pairwise t-tests (CPU) ----------
    _log("Running new Path-B pairwise t-tests (capacity@N=5 + MIC vs FOA)...")
    pairs: list[tuple[str, str, str, list[int]]] = [
        # capacity geom ablation now at N=5 paired
        ("geom_xs_n5",
         "no_geom_xs:cap_xs_mc8_inmem", "full_xs:cap_xs_mc8_inmem", [0, 1, 2, 3, 4]),
        ("geom_l_n5",
         "no_geom_l:cap_l_mc8_inmem", "full_l:cap_l_mc8_inmem", [0, 1, 2, 3, 4]),
        ("geom_xl_n5",
         "no_geom_xl:cap_xl_mc8_inmem", "full_xl:cap_xl_mc8_inmem", [0, 1, 2, 3, 4]),
        # SELDnet: MIC vs FOA (paired on seed)
        ("seldnet_mic_vs_foa_n5",
         "seldnet_official:baseline_mc8_inmem",
         "seldnet_official:foa_baseline_mc8_inmem", [0, 1, 2, 3, 4]),
    ]
    for tag, a, b, seeds in pairs:
        _run_pairwise(tag, a, b, seeds)

    # ---------- Cross-dataset zero-shot evaluation ----------
    _log("Running cross-dataset evaluation on STARSS22 dev-test...")
    _run("cross_dataset_starss22",
         [str(PYEXE), "-u", "-m", "week11_starss23._eval_cross_dataset",
          "--variants", "no_geom", "full",
          "--seeds", "0", "1", "2", "3", "4",
          "--suffix", "mc8_inmem",
          "--testset", "starss22-test"])
    # Also include the SELDnet baseline on the same testset.
    _run("cross_dataset_starss22_seldnet",
         [str(PYEXE), "-u", "-m", "week11_starss23._eval_cross_dataset",
          "--variants", "seldnet_official",
          "--seeds", "0", "1", "2", "3", "4",
          "--suffix", "baseline_mc8_inmem",
          "--testset", "starss22-test"])

    # ---------- Re-run augmented analyses ----------
    _log("Re-running CPU analyses (perclass with Bonferroni + Wilcoxon)...")
    _run("analysis_best_thr",
         [str(PYEXE), "-u", "-m", "week11_starss23._analysis_best_threshold"])
    _run("analysis_bootstrap",
         [str(PYEXE), "-u", "-m", "week11_starss23._analysis_bootstrap"])
    _run("analysis_perclass",
         [str(PYEXE), "-u", "-m", "week11_starss23._analysis_perclass"])

    _log("supervisor v3 chain DONE.")


if __name__ == "__main__":
    main()
