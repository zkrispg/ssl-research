"""Aggregate per-(variant, seed) results and run paired t-tests.

Reads each cell's ``summary.json`` (best ADPIT eval loss) and
``eval_threshold_sweep.json`` (DCASE metrics at multiple thresholds),
then runs ``scipy.stats.ttest_rel`` on the paired ``full - no_geom``
deltas across the 3 seeds. Writes ``multiseed_paired_ttest.json`` and
prints a paper-ready table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

RUNS = Path("D:/ssl-research/week11_starss23/runs")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
THRESHOLDS = ("0.10", "0.15", "0.18", "0.22", "0.30")
METRICS_M = ("f1", "er", "le_cd", "lr_cd", "seld")
OUT_DEFAULT = RUNS / "multiseed_paired_ttest.json"


def _load_cell(variant: str, seed: int) -> dict[str, Any]:
    run_dir = RUNS / f"{variant}_seed{seed}_mc8_inmem"
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    with (run_dir / "eval_threshold_sweep.json").open("r", encoding="utf-8") as f:
        sweep = json.load(f)
    return {"summary": summary, "sweep": sweep}


def _paired_t(a: list[float], b: list[float]) -> dict[str, float]:
    """Paired t-test of (b - a). Returns t, p, mean delta, std delta."""
    arr_a = np.array(a, dtype=np.float64)
    arr_b = np.array(b, dtype=np.float64)
    delta = arr_b - arr_a
    if len(delta) < 2:
        return {"t": float("nan"), "p": float("nan"),
                "mean_delta": float(delta.mean()),
                "std_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "rel_pct": float("nan")}
    res = stats.ttest_rel(arr_b, arr_a)
    return {
        "t": float(res.statistic),
        "p": float(res.pvalue),
        "mean_delta": float(delta.mean()),
        "std_delta": float(delta.std(ddof=1)),
        "rel_pct": float(delta.mean() / arr_a.mean() * 100),
    }


def _available_seeds() -> tuple[int, ...]:
    """Return the seeds for which BOTH variants have summary + sweep on disk."""
    out: list[int] = []
    for s in DEFAULT_SEEDS:
        ok = True
        for v in ("no_geom", "full"):
            d = RUNS / f"{v}_seed{s}_mc8_inmem"
            if not (d / "summary.json").exists() or not (d / "eval_threshold_sweep.json").exists():
                ok = False
                break
        if ok:
            out.append(s)
    return tuple(out)


def main() -> None:
    import sys
    if len(sys.argv) > 1:
        seeds = tuple(int(x) for x in sys.argv[1:])
    else:
        seeds = _available_seeds()
    if len(seeds) < 2:
        print(f"[error] need at least 2 seeds with complete artifacts, "
              f"found {seeds}; aborting.")
        sys.exit(1)
    print(f"[t-test] using seeds = {seeds}  (N = {len(seeds)} paired)\n", flush=True)
    cells = {(v, s): _load_cell(v, s) for v in ("no_geom", "full") for s in seeds}

    # ----- 1. Best ADPIT eval loss --------------------------------------------
    no_geom_loss = [cells[("no_geom", s)]["summary"]["best_eval_loss"] for s in seeds]
    full_loss = [cells[("full", s)]["summary"]["best_eval_loss"] for s in seeds]
    eval_loss_test = _paired_t(no_geom_loss, full_loss)

    # ----- 2. DCASE metrics across thresholds ---------------------------------
    dcase_tests: dict[str, dict[str, dict[str, dict]]] = {}
    for thr in THRESHOLDS:
        dcase_tests[thr] = {}
        for avg in ("macro", "micro"):
            avg_block: dict[str, dict] = {}
            for m in METRICS_M:
                ng = [cells[("no_geom", s)]["sweep"]["thresholds"][thr][avg][m] for s in seeds]
                ft = [cells[("full", s)]["sweep"]["thresholds"][thr][avg][m] for s in seeds]
                avg_block[m] = {
                    "no_geom_per_seed": ng,
                    "full_per_seed": ft,
                    **_paired_t(ng, ft),
                }
            dcase_tests[thr][avg] = avg_block

    final = {
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "best_eval_loss": {
            "no_geom_per_seed": no_geom_loss,
            "full_per_seed": full_loss,
            **eval_loss_test,
        },
        "dcase": dcase_tests,
    }
    out_path = OUT_DEFAULT.with_name(f"multiseed_paired_ttest_n{len(seeds)}.json")
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    OUT_DEFAULT.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}\n[saved] {OUT_DEFAULT}\n")

    # ----- pretty table (paper-ready) -----------------------------------------
    print("=" * 80)
    print("ADPIT eval loss (lower is better)")
    print("=" * 80)
    print(f"  {'seed':>5}  {'no_geom':>10}  {'full':>10}  {'Δ':>10}  {'Δ rel':>8}")
    for s, ng, ft in zip(seeds, no_geom_loss, full_loss):
        d = ft - ng
        print(f"  {s:>5}  {ng:>10.5f}  {ft:>10.5f}  {d:>+10.5f}  {d/ng*100:>+7.2f}%")
    e = eval_loss_test
    print(f"  {'mean':>5}  {np.mean(no_geom_loss):>10.5f}  {np.mean(full_loss):>10.5f}  "
          f"{e['mean_delta']:>+10.5f}  {e['rel_pct']:>+7.2f}%")
    print(f"  paired t-test:  t = {e['t']:+.3f},  p = {e['p']:.3f}  (N={len(seeds)} paired)")
    sig = "***" if e["p"] < 0.001 else "**" if e["p"] < 0.01 else "*" if e["p"] < 0.05 else "ns"
    print(f"  significance @ alpha=0.05:  {sig}")

    print("\n" + "=" * 80)
    print(f"DCASE metrics @ thr=0.18 (paper operating point), N={len(seeds)} paired t-tests")
    print("=" * 80)
    thr = "0.18"
    print(f"  {'metric':>10}  {'avg':>5}  {'ng_mean':>9}  {'ft_mean':>9}  "
          f"{'Δ':>10}  {'rel%':>7}  {'t':>7}  {'p':>6}  sig")
    for avg in ("macro", "micro"):
        for m in METRICS_M:
            d = dcase_tests[thr][avg][m]
            ng_mean = np.mean(d["no_geom_per_seed"])
            ft_mean = np.mean(d["full_per_seed"])
            sig = (
                "***" if d["p"] < 0.001
                else "**" if d["p"] < 0.01
                else "*" if d["p"] < 0.05
                else "."  if d["p"] < 0.10
                else "ns"
            )
            print(
                f"  {m:>10}  {avg:>5}  {ng_mean:>9.4f}  {ft_mean:>9.4f}  "
                f"{d['mean_delta']:>+10.4f}  {d['rel_pct']:>+6.2f}%  "
                f"{d['t']:>+7.2f}  {d['p']:>6.3f}  {sig}"
            )

    print("\n" + "=" * 80)
    print("DCASE SELD micro across thresholds (overview)")
    print("=" * 80)
    print(f"  {'thr':>5}  {'ng_mean':>9}  {'ft_mean':>9}  {'Δ':>10}  {'rel%':>7}  {'p':>6}")
    for thr in THRESHOLDS:
        d = dcase_tests[thr]["micro"]["seld"]
        ng_mean = np.mean(d["no_geom_per_seed"])
        ft_mean = np.mean(d["full_per_seed"])
        print(f"  {thr:>5}  {ng_mean:>9.4f}  {ft_mean:>9.4f}  "
              f"{d['mean_delta']:>+10.4f}  {d['rel_pct']:>+6.2f}%  {d['p']:>6.3f}")


if __name__ == "__main__":
    main()
