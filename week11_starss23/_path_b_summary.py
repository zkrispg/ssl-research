"""One-shot summary of all Path B results.

Reads:
  * runs/pairwise_*.json  (G3-vs-N=5 capacity, MIC vs FOA, etc.)
  * runs/cross_dataset_starss22-test_summary.json
  * runs/multiseed_summary_seldnet_foa.json
  * runs/analysis_perclass.json (with Bonferroni section)

Prints a single, paper-ready summary table.
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("D:/ssl-research/week11_starss23/runs")


def _load(name: str) -> dict | None:
    fp = R / name
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _sig(p: float) -> str:
    if p != p:  # NaN
        return ""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def _find_comparison(d: dict, metric: str, average: str, threshold: str) -> dict | None:
    for c in d.get("comparisons", []):
        if (
            c["metric"] == metric
            and c["average"] == average
            and c["threshold"] == threshold
        ):
            return c
    return None


print("=" * 84)
print("PATH B FINAL RESULTS  (target: INTERSPEECH / TASLP)")
print("=" * 84)

# ------------------------------------------------------------------ A1: capacity x geometry @ N=5
print("\n=== A1: Capacity x geometry (paired t-test no_geom vs full, N=5 seeds) ===")
print(f'{"size":<6}{"thr":<6}{"avg":<6}{"metric":<8}{"no_geom":>10}{"full":>10}{"delta":>10}{"p":>10}')
print("-" * 76)
files = [
    ("xs", "pairwise_no_geom_xs_cap_xs_mc8_inmem_vs_full_xs_cap_xs_mc8_inmem_n5.json"),
    ("m",  "pairwise_no_geom_mc8_inmem_vs_full_mc8_inmem_n5.json"),
    ("l",  "pairwise_no_geom_l_cap_l_mc8_inmem_vs_full_l_cap_l_mc8_inmem_n5.json"),
    ("xl", "pairwise_no_geom_xl_cap_xl_mc8_inmem_vs_full_xl_cap_xl_mc8_inmem_n5.json"),
]
for size, fname in files:
    d = _load(fname)
    if d is None:
        print(f"{size:<6} (missing: {fname})")
        continue
    n_used = d.get("n_seeds", "?")
    dcase = d.get("dcase", {})
    for thr in ("0.18", "0.30"):
        if thr not in dcase:
            continue
        for avg, metric in [("macro", "seld"), ("micro", "f1")]:
            c = dcase[thr].get(avg, {}).get(metric)
            if c is None:
                continue
            p = c.get("p", float("nan"))
            a_mean = _mean(c.get("a_per_seed", []))
            b_mean = _mean(c.get("b_per_seed", []))
            print(
                f"{size:<6}{thr:<6}{avg:<6}{metric:<8}"
                f"{a_mean:>10.4f}{b_mean:>10.4f}"
                f"{c['mean_delta']:>+10.4f}{p:>10.4f}{_sig(p)}  (N={n_used})"
            )
print(f"\n  * delta = full - no_geom; positive => no_geom better for SELD-score-like metrics (lower is better)")

# ------------------------------------------------------------------ B2: MIC vs FOA SELDnet
print("\n=== B2: SELDnet MIC vs FOA (paired t-test, N=5 seeds) ===")
fname = "pairwise_seldnet_official_baseline_mc8_inmem_vs_seldnet_official_foa_baseline_mc8_inmem_n5.json"
d = _load(fname)
if d is None:
    print(f"  (missing: {fname})")
else:
    print(f'{"thr":<6}{"avg":<6}{"metric":<8}{"MIC":>10}{"FOA":>10}{"delta":>10}{"p":>10}')
    print("-" * 60)
    dcase = d.get("dcase", {})
    for thr in ("0.18", "0.30"):
        if thr not in dcase:
            continue
        for avg, metric in [("macro", "seld"), ("micro", "f1"), ("macro", "le_cd")]:
            c = dcase[thr].get(avg, {}).get(metric)
            if c is None:
                continue
            p = c.get("p", float("nan"))
            a_mean = _mean(c.get("a_per_seed", []))
            b_mean = _mean(c.get("b_per_seed", []))
            print(
                f"{thr:<6}{avg:<6}{metric:<8}"
                f"{a_mean:>10.4f}{b_mean:>10.4f}"
                f"{c['mean_delta']:>+10.4f}{p:>10.4f}{_sig(p)}"
            )

# ------------------------------------------------------------------ B1: Cross-dataset (STARSS22)
print("\n=== B1: Cross-dataset zero-shot eval on STARSS22 dev-test ===")
d = _load("cross_dataset_starss22-test_summary.json")
if d is None:
    print("  (missing: cross_dataset_starss22-test_summary.json)")
else:
    print(f'{"compare":<32}{"thr":<6}{"avg":<6}{"metric":<8}{"a":>10}{"b":>10}{"delta":>10}{"p":>10}')
    print("-" * 92)
    for c in d.get("comparisons", []):
        if (
            c["metric"] == "seld"
            and c["average"] == "macro"
            and c["threshold"] in ("0.18", "0.30")
        ):
            label = f"{c['a_variant']} vs {c['b_variant']}"
            p = c.get("p", float("nan"))
            print(
                f"{label:<32}{c['threshold']:<6}{c['average']:<6}{c['metric']:<8}"
                f"{c['mean_a']:>10.4f}{c['mean_b']:>10.4f}"
                f"{c['mean_delta']:>+10.4f}{p:>10.4f}{_sig(p)}"
            )

# Same for SELDnet within-class
print("\n--- B1b: SELDnet on STARSS22 (within-variant zero-shot) ---")
d = _load("cross_dataset_starss22-test_summary.json")  # same file? actually no
# We saved seldnet_official as separate seeds in cross_dataset_starss22-test_seldnet_*.
# Mean over 5 seeds at thr=0.30.
import numpy as np
sd_thr30: list[float] = []
sd_thr18: list[float] = []
for s in range(5):
    fp = R / f"cross_dataset_starss22-test_seldnet_official_baseline_mc8_inmem_seed{s}.json"
    if not fp.exists():
        continue
    rj = json.loads(fp.read_text(encoding="utf-8"))
    sd_thr30.append(rj["thresholds"]["0.30"]["macro"]["seld"])
    sd_thr18.append(rj["thresholds"]["0.18"]["macro"]["seld"])
print(f"  SELDnet on STARSS22 (n={len(sd_thr30)} seeds):")
if sd_thr18:
    a = np.array(sd_thr18)
    print(f"    thr=0.18  macro SELD = {a.mean():.4f} ± {a.std(ddof=1):.4f}")
if sd_thr30:
    a = np.array(sd_thr30)
    print(f"    thr=0.30  macro SELD = {a.mean():.4f} ± {a.std(ddof=1):.4f}")

# ------------------------------------------------------------------ A2: Per-class summary
print("\n=== A2: Per-class breakdown w/ Bonferroni + Wilcoxon ===")
d = _load("analysis_perclass.json")
if d is None:
    print("  (missing: analysis_perclass.json)")
else:
    bonf = d.get("bonferroni", {})
    print(f"  N = {len(d.get('seeds', []))}, M metrics = {bonf.get('n_metrics')}, "
          f"K active classes = {bonf.get('n_classes_active')}")
    print(f"  per-metric Bonferroni alpha = {bonf.get('within_metric_alpha'):.4g}")
    print(f"  global   Bonferroni alpha = {bonf.get('global_alpha'):.4g}")

    classes = d.get("classes", [])
    n_t_metric = sum(
        1 for r in classes for m in ("f1", "er_cd", "le_cd", "lr_cd")
        if r[f"{m}_paired"]["p"] is not None
        and not (r[f"{m}_paired"]["p"] != r[f"{m}_paired"]["p"])
        and r[f"{m}_paired"]["p"] < bonf.get("within_metric_alpha", 1)
    )
    n_w_metric = sum(
        1 for r in classes for m in ("f1", "er_cd", "le_cd", "lr_cd")
        if r[f"{m}_paired"]["wilcoxon_p"] is not None
        and not (r[f"{m}_paired"]["wilcoxon_p"] != r[f"{m}_paired"]["wilcoxon_p"])
        and r[f"{m}_paired"]["wilcoxon_p"] < bonf.get("within_metric_alpha", 1)
    )
    print(f"  classes surviving per-metric Bonferroni (parametric t):  {n_t_metric}")
    print(f"  classes surviving per-metric Bonferroni (Wilcoxon SR):   {n_w_metric}")
    # Top 3 most significant per metric (uncorrected)
    print("  most significant per metric (uncorrected, full - no_geom):")
    for m in ("f1", "le_cd", "lr_cd", "er_cd"):
        best = sorted(
            classes,
            key=lambda r: r[f"{m}_paired"]["p"] if r[f"{m}_paired"]["p"] == r[f"{m}_paired"]["p"] else 99,
        )[:1]
        if not best:
            continue
        b = best[0]
        bp = b[f"{m}_paired"]
        print(
            f"    {m:<6}  class {b['class_id']:>2} "
            f"{b['class_name'][:24]:<24}  Δ={bp['mean_delta']:+.3f}  "
            f"p_t={bp['p']:.3f}  p_W={bp['wilcoxon_p']:.3f}"
        )

print("\n" + "=" * 84)
print("DONE")
print("=" * 84)
