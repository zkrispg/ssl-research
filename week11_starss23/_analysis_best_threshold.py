"""T1b -- per-variant best-threshold paired analysis.

For each cell of every paired comparison we already saved
``eval_threshold_sweep.json`` with metrics at 5 activity thresholds
(0.10, 0.15, 0.18, 0.22, 0.30). The single-threshold paired t-tests
in :mod:`_pairwise_ttest` lock the operating point at thr = 0.18.
Reviewers may object: "if you tune thr per variant, does no_geom still
win?" This script answers that.

For every paired comparison set on disk:
    * Find the per-variant SELD-optimal threshold (min mean macro-SELD
      across seeds for that variant).
    * Re-do the paired t-test on F1 / ER / LE_CD / LR_CD / SELD with
      *each variant evaluated at its own optimal threshold*.
    * Save a JSON per comparison + a top-level summary table.

Output: ``runs/analysis_best_threshold.json``.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

RUNS = Path("D:/ssl-research/week11_starss23/runs")
OUT = RUNS / "analysis_best_threshold.json"
THRESHOLDS = ("0.10", "0.15", "0.18", "0.22", "0.30")
METRICS = ("f1", "er", "le_cd", "lr_cd", "seld")

# (label, variant_a, suffix_a, variant_b, suffix_b, seeds)
COMPARISONS: list[tuple[str, str, str, str, str, list[int]]] = [
    ("geom_n5",
     "no_geom", "mc8_inmem", "full", "mc8_inmem", [0, 1, 2, 3, 4]),
    ("seldnet_vs_no_geom_n5",
     "seldnet_official", "baseline_mc8_inmem", "no_geom", "mc8_inmem", [0, 1, 2, 3, 4]),
    ("seldnet_vs_full_n5",
     "seldnet_official", "baseline_mc8_inmem", "full", "mc8_inmem", [0, 1, 2, 3, 4]),
    ("strong_specaug_no_geom_n5",
     "no_geom", "mc8_inmem", "no_geom", "mc8_inmem_specaug", [0, 1, 2, 3, 4]),
    ("strong_specaug_full_n5",
     "full", "mc8_inmem", "full", "mc8_inmem_specaug", [0, 1, 2, 3, 4]),
    ("weak_specaug_no_geom_n5",
     "no_geom", "mc8_inmem", "no_geom", "mc8_inmem_specaug_weak", [0, 1, 2, 3, 4]),
    ("weak_specaug_full_n5",
     "full", "mc8_inmem", "full", "mc8_inmem_specaug_weak", [0, 1, 2, 3, 4]),
    ("geom_xs_n5",
     "no_geom_xs", "cap_xs_mc8_inmem", "full_xs", "cap_xs_mc8_inmem", [0, 1, 2, 3, 4]),
    ("geom_l_n5",
     "no_geom_l", "cap_l_mc8_inmem", "full_l", "cap_l_mc8_inmem", [0, 1, 2, 3, 4]),
    ("geom_xl_n5",
     "no_geom_xl", "cap_xl_mc8_inmem", "full_xl", "cap_xl_mc8_inmem", [0, 1, 2, 3, 4]),
    ("seldnet_mic_vs_foa_n5",
     "seldnet_official", "baseline_mc8_inmem",
     "seldnet_official", "foa_baseline_mc8_inmem", [0, 1, 2, 3, 4]),
]


def _load_sweep(variant: str, suffix: str, seed: int) -> dict | None:
    p = RUNS / f"{variant}_seed{seed}_{suffix}" / "eval_threshold_sweep.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _best_threshold(
    sweeps: list[dict], avg: str, metric: str, lower_better: bool
) -> str:
    """Pick the threshold that optimises ``metric`` averaged over seeds."""
    means = {}
    for thr in THRESHOLDS:
        vals = [s["thresholds"][thr][avg][metric] for s in sweeps]
        means[thr] = float(np.mean(vals))
    if lower_better:
        return min(means, key=means.get)
    return max(means, key=means.get)


def _paired_t(a: list[float], b: list[float]) -> dict[str, float]:
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    delta = arr_b - arr_a
    if len(delta) < 2:
        return {"t": float("nan"), "p": float("nan"),
                "mean_delta": float(delta.mean()), "std_delta": 0.0}
    res = stats.ttest_rel(arr_b, arr_a)
    return {
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "rel_pct": float(delta.mean() / arr_a.mean() * 100) if arr_a.mean() != 0 else float("nan"),
    }


def main() -> None:
    print(f"[analysis] best-threshold reanalysis -> {OUT}\n", flush=True)
    out: dict[str, Any] = {}

    for label, va, sa, vb, sb, seeds in COMPARISONS:
        sweeps_a = [_load_sweep(va, sa, s) for s in seeds]
        sweeps_b = [_load_sweep(vb, sb, s) for s in seeds]
        if any(x is None for x in sweeps_a + sweeps_b):
            print(f"[skip] {label}: missing sweep file(s)")
            out[label] = {"status": "missing"}
            continue

        comp: dict[str, Any] = {
            "a": {"variant": va, "suffix": sa},
            "b": {"variant": vb, "suffix": sb},
            "seeds": seeds,
        }
        for avg in ("macro", "micro"):
            avg_block: dict[str, Any] = {}
            # per-variant optimal threshold for SELD score (lower better)
            best_a = _best_threshold(sweeps_a, avg, "seld", lower_better=True)
            best_b = _best_threshold(sweeps_b, avg, "seld", lower_better=True)
            avg_block["best_thr_a_seld"] = best_a
            avg_block["best_thr_b_seld"] = best_b

            # Paired t-test at locked thr=0.18 vs each at its own optimum.
            for metric in METRICS:
                # Locked thr
                a_locked = [s["thresholds"]["0.18"][avg][metric] for s in sweeps_a]
                b_locked = [s["thresholds"]["0.18"][avg][metric] for s in sweeps_b]
                # Best-per-variant thr (optimized on SELD)
                a_best = [s["thresholds"][best_a][avg][metric] for s in sweeps_a]
                b_best = [s["thresholds"][best_b][avg][metric] for s in sweeps_b]
                avg_block[metric] = {
                    "thr_locked_0.18": {
                        "a_per_seed": a_locked,
                        "b_per_seed": b_locked,
                        **_paired_t(a_locked, b_locked),
                    },
                    "thr_best_per_variant": {
                        "a_thr": best_a, "b_thr": best_b,
                        "a_per_seed": a_best,
                        "b_per_seed": b_best,
                        **_paired_t(a_best, b_best),
                    },
                }
            comp[avg] = avg_block
        out[label] = comp

        # Print a compact one-liner per comparison (macro SELD).
        bp = comp["macro"]["seld"]
        loc = bp["thr_locked_0.18"]
        bst = bp["thr_best_per_variant"]
        print(
            f"[{label:>30}]  macro SELD  "
            f"locked: Δ={loc['mean_delta']:+.4f} p={loc['p']:.3f}  |  "
            f"best (a@{bst['a_thr']}, b@{bst['b_thr']}): "
            f"Δ={bst['mean_delta']:+.4f} p={bst['p']:.3f}",
            flush=True,
        )

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
