"""Path C analysis -- parse DCASE 2024 test_only logs and run paired stats.

Reads the per-cell test logs produced by `_run_path_c_full.ps1`:

    runs/
      dcase2024_100_repro_seed{S}_test.log     # FOA Stage 1 reproduce  (5 seeds: 0..4)
      dcase2024_110_ablate_seed{S}_test.log    # GCA full
      dcase2024_111_ablate_seed{S}_test.log    # GCA no_geom
      dcase2024_112_ablate_seed{S}_test.log    # no-GCA matched control

For each cell it extracts:
    F 20° (%), DOAE_CD (deg), Dist_err (m), RDE_CD (rel), SELD score.

Then computes:
  - Per-cell mean ± std across seeds
  - Paired t-test (110 vs 112) and (110 vs 111) for F1, DOAE_CD, RDE_CD
  - Cohen's d_z, bootstrap 95 % CI on the per-seed delta
  - Output: paths_c_results.json + path_c_summary.md (paste-ready table)

Usage:
    python _path_c_analyze.py
    python _path_c_analyze.py --runs-dir D:\\ssl-research\\week11_starss23\\runs

The script is safe to run incrementally: if a seed log is missing it is
treated as N/A and skipped from paired tests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import numpy as np
from scipy import stats  # type: ignore


# Cells we want to analyze.
CELLS = {
    "100_foa_repro":   "dcase2024_100_repro_seed{seed}_test.log",
    "110_gca_full":    "dcase2024_110_ablate_seed{seed}_test.log",
    "111_gca_nogeom":  "dcase2024_111_ablate_seed{seed}_test.log",
    "112_no_gca":      "dcase2024_112_ablate_seed{seed}_test.log",
    "113_vanilla_se":  "dcase2024_113_ablate_seed{seed}_test.log",
}
SEEDS = [0, 1, 2, 3, 4]

METRIC_PATTERNS = {
    # field name -> (regex, transform). Patterns are tolerant of mojibake
    # for the degree symbol that may appear between '20' and '(' (e.g.
    # '°' becomes '\ufffd\ufffd' under UTF-8-as-UTF-16 conversion).
    "F1":    (re.compile(r"F\s*20[^(]*\(.*\)\s*:\s*([\d.]+)\s*%"), float),
    "LE":    (re.compile(r"DOAE_CD\s*\(deg\)\s*:\s*([\d.]+)"),     float),
    "DE":    (re.compile(r"Dist_err\s*\(m\)\s*:\s*([\d.]+)"),      float),
    "RDE":   (re.compile(r"RDE_CD\s*\(rel\)\s*:\s*([\d.]+)"),      float),
    "SELD":  (re.compile(r"SELD score\s*\(.*\)\s*:\s*([\d.]+)"),   float),
}


def _read_text_any_encoding(path: str) -> str:
    """Read a log file with auto-encoding detection for the encodings
    we see in this project: UTF-8 (Python default), UTF-8-with-BOM
    (legacy Out-File on some PowerShells), and UTF-16-LE-with-BOM
    (Windows PowerShell 5's `Out-File -Encoding UTF8`, which actually
    writes UTF-16-LE -- see https://stackoverflow.com/q/40098771).
    """
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le", errors="replace")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_log(path: str) -> Optional[dict[str, float]]:
    if not os.path.isfile(path):
        return None
    text = _read_text_any_encoding(path)
    out: dict[str, float] = {}
    for key, (pat, cast) in METRIC_PATTERNS.items():
        m = pat.search(text)
        if m is None:
            return None  # missing any metric => incomplete log
        out[key] = cast(m.group(1))
    return out


def cohens_dz(deltas: list[float]) -> float:
    if len(deltas) < 2:
        return float("nan")
    arr = np.asarray(deltas, dtype=np.float64)
    s = arr.std(ddof=1)
    return float("nan") if s == 0 else float(arr.mean() / s)


def bootstrap_ci(deltas: list[float], n_boot: int = 5000, alpha: float = 0.05,
                 rng_seed: int = 0) -> tuple[float, float]:
    if len(deltas) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(deltas, dtype=np.float64)
    n = len(arr)
    boot_means = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_test(a_list: list[float], b_list: list[float]) -> dict:
    """Per-seed paired test between cell-A and cell-B values."""
    pairs = list(zip(a_list, b_list))
    if len(pairs) < 2:
        return {"n": len(pairs), "note": "insufficient pairs"}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    deltas = (a - b).tolist()
    n = len(deltas)

    t_stat, p_t = stats.ttest_rel(a, b)
    try:
        w_stat, p_w = stats.wilcoxon(a, b, zero_method="pratt")
    except ValueError:
        w_stat, p_w = float("nan"), float("nan")

    return {
        "n": n,
        "a_per_seed": a.tolist(),
        "b_per_seed": b.tolist(),
        "delta_per_seed": deltas,
        "delta_mean": float(np.mean(deltas)),
        "delta_std":  float(np.std(deltas, ddof=1)) if n > 1 else 0.0,
        "ttest_rel":  {"t": float(t_stat), "p": float(p_t)},
        "wilcoxon":   {"W": float(w_stat), "p": float(p_w)},
        "cohens_dz":  cohens_dz(deltas),
        "bootstrap_95ci": bootstrap_ci(deltas),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=r"D:\ssl-research\week11_starss23\runs")
    ap.add_argument("--out-json", default=r"D:\ssl-research\paper\path_c_results.json")
    ap.add_argument("--out-md",   default=r"D:\ssl-research\paper\path_c_summary.md")
    args = ap.parse_args()

    runs_dir = args.runs_dir

    # ---------------- harvest per-cell per-seed metrics ----------------
    raw: dict[str, dict[int, dict[str, float] | None]] = {c: {} for c in CELLS}
    for cell_name, fname_tmpl in CELLS.items():
        for s in SEEDS:
            path = os.path.join(runs_dir, fname_tmpl.format(seed=s))
            raw[cell_name][s] = parse_log(path)

    # Per-cell mean ± std summary
    summary: dict[str, dict] = {}
    for cell_name, by_seed in raw.items():
        avail = {s: m for s, m in by_seed.items() if m is not None}
        cell = {"n_seeds": len(avail), "missing_seeds": [s for s in SEEDS if s not in avail]}
        for metric in METRIC_PATTERNS:
            vals = [m[metric] for m in avail.values()]
            cell[metric] = {
                "values":   vals,
                "mean":     mean(vals) if vals else None,
                "std":      stdev(vals) if len(vals) > 1 else 0.0,
                "n":        len(vals),
            }
        summary[cell_name] = cell

    # ---------------- paired contrasts ----------------
    contrasts: dict[str, dict] = {}
    pair_specs = [
        ("110_gca_full",   "112_no_gca",     "GCA full vs no-GCA matched control"),
        ("110_gca_full",   "111_gca_nogeom", "GCA full vs no_geom (geometry contribution)"),
        ("111_gca_nogeom", "112_no_gca",     "no_geom GCA vs no-GCA (channel-attn alone)"),
        ("113_vanilla_se", "112_no_gca",     "Vanilla SE vs no-GCA (channel-attn-on-features alone)"),
        ("113_vanilla_se", "111_gca_nogeom", "Vanilla SE vs GCA no_geom (mlp gate vs Q/K/V over mics)"),
        ("110_gca_full",   "113_vanilla_se", "GCA full vs Vanilla SE (geometry-mic vs feature-channel)"),
    ]
    for cell_a, cell_b, descr in pair_specs:
        per_metric = {}
        for metric in ("F1", "LE", "RDE", "DE", "SELD"):
            shared = [s for s in SEEDS if raw[cell_a].get(s) and raw[cell_b].get(s)]
            a_vals = [raw[cell_a][s][metric] for s in shared]
            b_vals = [raw[cell_b][s][metric] for s in shared]
            per_metric[metric] = paired_test(a_vals, b_vals)
        contrasts[f"{cell_a}__vs__{cell_b}"] = {"description": descr, "metrics": per_metric}

    # ---------------- write JSON ----------------
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {"per_cell": summary, "contrasts": contrasts}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[saved] {args.out_json}")

    # ---------------- markdown summary ----------------
    lines: list[str] = []
    lines.append("# Path C results -- DCASE 2024 baseline + GCA ablation")
    lines.append("")
    lines.append("Auto-generated from DCASE test_only logs in `runs/`.")
    lines.append("")
    lines.append("## Per-cell mean ± std (seeds 0..4)")
    lines.append("")
    lines.append("| Cell                    | n  | F 20° (%)        | DOAE_CD (°)      | RDE              | Dist_err (m)     | SELD             |")
    lines.append("| ----------------------- | -- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |")
    for cell_name, cell in summary.items():
        n = cell["n_seeds"]
        def fmt(metric: str, scale: float = 1.0, nd: int = 2) -> str:
            d = cell[metric]
            if d["mean"] is None:
                return "n/a"
            return f"{scale*d['mean']:.{nd}f} ± {scale*d['std']:.{nd}f}"
        lines.append(
            f"| {cell_name:<23} | {n:<2} | "
            f"{fmt('F1'):<16} | {fmt('LE'):<16} | {fmt('RDE',1.0,3):<16} | {fmt('DE'):<16} | {fmt('SELD',1.0,3):<16} |"
        )
    lines.append("")
    lines.append("## Paired contrasts (across-seed)")
    lines.append("")
    for cname, c in contrasts.items():
        lines.append(f"### {cname}")
        lines.append(f"_{c['description']}_")
        lines.append("")
        lines.append("| Metric | n | mean Δ (A-B) | t (p)        | Wilcoxon W (p) | d_z   | bootstrap 95% CI    |")
        lines.append("| ------ | - | ------------ | ------------ | --------------- | ----- | ------------------- |")
        for metric in ("F1", "LE", "RDE", "DE", "SELD"):
            r = c["metrics"][metric]
            n = r.get("n", 0)
            if n < 2:
                lines.append(f"| {metric} | {n} | n/a | n/a | n/a | n/a | n/a |")
                continue
            ci = r["bootstrap_95ci"]
            lines.append(
                f"| {metric:<6} | {n} | {r['delta_mean']:+.3f} ± {r['delta_std']:.3f} | "
                f"t={r['ttest_rel']['t']:+.2f} (p={r['ttest_rel']['p']:.3f}) | "
                f"W={r['wilcoxon']['W']:.1f} (p={r['wilcoxon']['p']:.3f}) | "
                f"{r['cohens_dz']:+.2f} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |"
            )
        lines.append("")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {args.out_md}")

    # ---------------- print short status ----------------
    print()
    print("=" * 60)
    print("Per-cell completion status:")
    for cell_name, cell in summary.items():
        miss = cell.get("missing_seeds", [])
        miss_str = "" if not miss else f" (missing seeds: {miss})"
        print(f"  {cell_name:<24} n={cell['n_seeds']}/{len(SEEDS)}{miss_str}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
