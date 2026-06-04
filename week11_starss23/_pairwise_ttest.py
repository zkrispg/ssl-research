"""Generalized paired t-test between any two (variant, suffix) cell sets.

Use this for cross-variant comparisons that the original
``_paired_ttest_report.py`` does not cover, e.g.

    # SELDnet baseline vs our full system (N = 3)
    python -m week11_starss23._pairwise_ttest \
        --a seldnet_official:baseline_mc8_inmem \
        --b full:mc8_inmem \
        --seeds 0 1 2

    # SpecAug ablation on the no_geom variant (N = 5)
    python -m week11_starss23._pairwise_ttest \
        --a no_geom:mc8_inmem \
        --b no_geom:mc8_inmem_specaug \
        --seeds 0 1 2 3 4

Each ``(variant, suffix)`` pair refers to the directory layout
``runs/<variant>_seed<seed>_<suffix>``. The script reads ``summary.json``
and ``eval_threshold_sweep.json`` from each cell, runs paired t-tests
across the supplied seeds, and prints / saves a paper-ready table.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

RUNS = Path("D:/ssl-research/week11_starss23/runs")
THRESHOLDS = ("0.10", "0.15", "0.18", "0.22", "0.30")
METRICS_M = ("f1", "er", "le_cd", "lr_cd", "seld")


@dataclass(frozen=True)
class CellGroup:
    """A set of cells sharing one (variant, suffix) but varying seed."""

    variant: str
    suffix: str

    @classmethod
    def parse(cls, spec: str) -> "CellGroup":
        if ":" not in spec:
            raise SystemExit(
                f"--a / --b must be 'variant:suffix' (got {spec!r})"
            )
        v, s = spec.split(":", 1)
        return cls(variant=v, suffix=s)

    def run_dir(self, seed: int) -> Path:
        return RUNS / f"{self.variant}_seed{seed}_{self.suffix}"

    @property
    def label(self) -> str:
        return f"{self.variant}/{self.suffix}"


def _load_cell(group: CellGroup, seed: int) -> dict[str, Any]:
    d = group.run_dir(seed)
    with (d / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    with (d / "eval_threshold_sweep.json").open("r", encoding="utf-8") as f:
        sweep = json.load(f)
    return {"summary": summary, "sweep": sweep}


def _paired_t(a: list[float], b: list[float]) -> dict[str, float]:
    """Paired t-test of (b - a)."""
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    delta = arr_b - arr_a
    if len(delta) < 2:
        return {
            "t": float("nan"),
            "p": float("nan"),
            "mean_delta": float(delta.mean()),
            "std_delta": 0.0,
            "rel_pct": float("nan"),
        }
    res = stats.ttest_rel(arr_b, arr_a)
    return {
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "rel_pct": float(delta.mean() / arr_a.mean() * 100) if arr_a.mean() != 0 else float("nan"),
    }


def _sig(p: float) -> str:
    if not np.isfinite(p):
        return "?"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return "ns"


def _format_label_for_filename(g: CellGroup) -> str:
    return f"{g.variant}_{g.suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="cell A as 'variant:suffix'")
    parser.add_argument("--b", required=True, help="cell B as 'variant:suffix'")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="JSON output path (default: runs/pairwise_<A>_vs_<B>_n<N>.json)",
    )
    args = parser.parse_args()

    a = CellGroup.parse(args.a)
    b = CellGroup.parse(args.b)
    seeds = tuple(args.seeds)
    if len(seeds) < 2:
        raise SystemExit(f"need at least 2 seeds, got {seeds}")

    print(f"[pairwise] A = {a.label}", flush=True)
    print(f"[pairwise] B = {b.label}", flush=True)
    print(f"[pairwise] seeds = {seeds}  (N = {len(seeds)} paired)\n", flush=True)

    cells_a = {s: _load_cell(a, s) for s in seeds}
    cells_b = {s: _load_cell(b, s) for s in seeds}

    a_loss = [cells_a[s]["summary"]["best_eval_loss"] for s in seeds]
    b_loss = [cells_b[s]["summary"]["best_eval_loss"] for s in seeds]
    eval_loss_test = _paired_t(a_loss, b_loss)

    dcase_tests: dict[str, dict[str, dict[str, dict]]] = {}
    for thr in THRESHOLDS:
        dcase_tests[thr] = {}
        for avg in ("macro", "micro"):
            avg_block: dict[str, dict] = {}
            for m in METRICS_M:
                aa = [cells_a[s]["sweep"]["thresholds"][thr][avg][m] for s in seeds]
                bb = [cells_b[s]["sweep"]["thresholds"][thr][avg][m] for s in seeds]
                avg_block[m] = {
                    "a_per_seed": aa,
                    "b_per_seed": bb,
                    **_paired_t(aa, bb),
                }
            dcase_tests[thr][avg] = avg_block

    final = {
        "a": {"variant": a.variant, "suffix": a.suffix},
        "b": {"variant": b.variant, "suffix": b.suffix},
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "best_eval_loss": {
            "a_per_seed": a_loss,
            "b_per_seed": b_loss,
            **eval_loss_test,
        },
        "dcase": dcase_tests,
    }
    out_path = (
        args.out
        if args.out is not None
        else RUNS
        / f"pairwise_{_format_label_for_filename(a)}_vs_{_format_label_for_filename(b)}_n{len(seeds)}.json"
    )
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}\n")

    # -------- ADPIT eval loss table --------
    print("=" * 84)
    print(f"ADPIT eval loss   B - A   (B = {b.label},  A = {a.label})")
    print("=" * 84)
    print(f"  {'seed':>5}  {'A':>10}  {'B':>10}  {'B - A':>11}  {'rel %':>8}")
    for s, av, bv in zip(seeds, a_loss, b_loss):
        d = bv - av
        print(
            f"  {s:>5}  {av:>10.5f}  {bv:>10.5f}  {d:>+11.5f}  "
            f"{(d / av * 100) if av != 0 else float('nan'):>+7.2f}%"
        )
    e = eval_loss_test
    print(
        f"  {'mean':>5}  {np.mean(a_loss):>10.5f}  {np.mean(b_loss):>10.5f}  "
        f"{e['mean_delta']:>+11.5f}  {e['rel_pct']:>+7.2f}%"
    )
    print(
        f"  paired t-test (B - A): t = {e['t']:+.3f}, p = {e['p']:.3f}, "
        f"sig = {_sig(e['p'])}"
    )

    # -------- DCASE metrics @ paper operating point --------
    print("\n" + "=" * 84)
    print(f"DCASE metrics @ thr=0.18, N = {len(seeds)} paired (B - A; B = {b.label})")
    print("=" * 84)
    thr = "0.18"
    print(
        f"  {'metric':>10}  {'avg':>5}  {'A_mean':>9}  {'B_mean':>9}  "
        f"{'Δ':>10}  {'rel%':>7}  {'t':>7}  {'p':>6}  sig"
    )
    for avg in ("macro", "micro"):
        for m in METRICS_M:
            d = dcase_tests[thr][avg][m]
            am = np.mean(d["a_per_seed"])
            bm = np.mean(d["b_per_seed"])
            print(
                f"  {m:>10}  {avg:>5}  {am:>9.4f}  {bm:>9.4f}  "
                f"{d['mean_delta']:>+10.4f}  {d['rel_pct']:>+6.2f}%  "
                f"{d['t']:>+7.2f}  {d['p']:>6.3f}  {_sig(d['p'])}"
            )


if __name__ == "__main__":
    main()
