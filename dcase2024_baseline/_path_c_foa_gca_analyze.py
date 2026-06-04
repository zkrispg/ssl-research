"""Path C / Tier VI: FOA-modality GCA cross-modality ablation analysis.

Reads test_logs for:
  - task 100 (FOA reproduce, no GCA)            -- job 'repro_seed{1..4}'   from week11 logs
  - task 130 (FOA + GCA full)                   -- job 'ablate_seed{1..3}'  from runs/
  - task 131 (FOA + GCA no_geom)                -- job 'ablate_seed{1..3}'  from runs/

Computes per-cell mean/std and paired contrasts on shared seeds.

Outputs:
    D:\\ssl-research\\paper\\path_c_foa_gca.json
    D:\\ssl-research\\paper\\path_c_foa_gca.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
from _path_c_analyze import parse_log  # type: ignore

LOG_DIRS = [
    Path(r"D:\ssl-research\runs"),
    Path(r"D:\ssl-research\week11_starss23\runs"),
]
OUT_PATH = Path(r"D:\ssl-research\paper")
OUT_PATH.mkdir(parents=True, exist_ok=True)


CELLS = [
    {"task": "100", "job_pattern": "repro_seed{seed}",   "name": "100_foa_no_gca",      "seeds": [1, 2, 3, 4]},
    {"task": "130", "job_pattern": "ablate_seed{seed}",  "name": "130_foa_gca_full",    "seeds": [1, 2, 3]},
    {"task": "131", "job_pattern": "ablate_seed{seed}",  "name": "131_foa_gca_nogeom",  "seeds": [1, 2, 3]},
]


def _resolve_log(name: str) -> Path:
    for d in LOG_DIRS:
        p = d / name
        if p.is_file(): return p
    return LOG_DIRS[0] / name


def cohens_dz(deltas):
    a = np.asarray(deltas, dtype=np.float64)
    if len(a) < 2: return float("nan")
    s = a.std(ddof=1)
    return float("nan") if s == 0 else float(a.mean() / s)


def bootstrap_ci(deltas, n_boot=5000, alpha=0.05, rng_seed=0):
    if len(deltas) < 2: return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(deltas, dtype=np.float64)
    bm = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return (float(np.percentile(bm, 100*alpha/2)), float(np.percentile(bm, 100*(1-alpha/2))))


def main() -> int:
    per_cell: dict[str, dict] = {}
    for spec in CELLS:
        seed_metrics = {}
        for s in spec["seeds"]:
            job = spec["job_pattern"].format(seed=s)
            log = _resolve_log(f"dcase2024_{spec['task']}_{job}_test.log")
            m = parse_log(str(log))
            if m is not None:
                seed_metrics[s] = m
        agg = {"n_seeds": len(seed_metrics), "per_seed": seed_metrics}
        for metric in ("F1", "LE", "DE", "RDE", "SELD"):
            vals = np.array([sm[metric] for sm in seed_metrics.values()])
            if len(vals):
                agg[metric] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
        per_cell[spec["name"]] = agg
        print(f"[{spec['name']}] n={agg['n_seeds']}, F1={agg.get('F1', {}).get('mean','n/a')}")

    # paired contrasts
    pairs = [
        ("130_foa_gca_full",   "100_foa_no_gca",    "FOA: GCA full vs no GCA (overall)"),
        ("130_foa_gca_full",   "131_foa_gca_nogeom","FOA: GCA full vs no_geom (geometry contribution)"),
        ("131_foa_gca_nogeom", "100_foa_no_gca",    "FOA: channel attention alone vs no GCA"),
    ]
    contrasts = {}
    for a_name, b_name, descr in pairs:
        a_seeds = per_cell[a_name].get("per_seed", {})
        b_seeds = per_cell[b_name].get("per_seed", {})
        shared = sorted(set(a_seeds.keys()) & set(b_seeds.keys()))
        c = {"description": descr, "n": len(shared), "shared_seeds": shared}
        if len(shared) >= 2:
            for metric in ("F1", "LE", "DE", "RDE", "SELD"):
                a_vals = np.array([a_seeds[s][metric] for s in shared])
                b_vals = np.array([b_seeds[s][metric] for s in shared])
                d = a_vals - b_vals
                t_stat, p_t = stats.ttest_rel(a_vals, b_vals)
                try:
                    w_stat, p_w = stats.wilcoxon(a_vals, b_vals)
                except Exception:
                    w_stat, p_w = float("nan"), float("nan")
                c[metric] = {
                    "a_per_seed":     a_vals.tolist(),
                    "b_per_seed":     b_vals.tolist(),
                    "delta_per_seed": d.tolist(),
                    "delta_mean":     float(d.mean()),
                    "delta_std":      float(d.std(ddof=1)),
                    "t":              float(t_stat), "p_t": float(p_t),
                    "w":              float(w_stat), "p_w": float(p_w),
                    "cohens_dz":      cohens_dz(d.tolist()),
                    "boot_ci_95":     list(bootstrap_ci(d.tolist())),
                }
        contrasts[f"{a_name}__vs__{b_name}"] = c

    payload = {"per_cell": per_cell, "contrasts": contrasts, "log_dirs_searched": [str(d) for d in LOG_DIRS]}
    out_json = OUT_PATH / "path_c_foa_gca.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    # write MD
    L = ["# Path C / FOA-modality GCA ablation (Tier VI)",
         "",
         "Cross-modality replication of Stage 3: trains the same GCA mechanism over the FOA",
         "ambisonic channels (W/X/Y/Z) where the geometry token encodes each channel's direction-",
         "of-max-response (W = origin, X/Y/Z = unit vectors along principal axes).",
         "",
         "## Per-cell"]
    L.append("| Cell | n | F 20\u00b0 (%) | DOAE_CD (\u00b0) | RDE | Dist_err (m) | SELD |")
    L.append("| ---- | - | --------- | ----------- | --- | ------------ | ---- |")
    for spec in CELLS:
        c = per_cell[spec["name"]]
        if c.get("n_seeds", 0) == 0:
            L.append(f"| {spec['name']} | 0 | n/a | n/a | n/a | n/a | n/a |"); continue
        f1 = c["F1"]; le = c["LE"]; rde = c["RDE"]; de = c["DE"]; seld = c["SELD"]
        L.append(f"| {spec['name']} | {c['n_seeds']} | {f1['mean']:.2f} \u00b1 {f1['std']:.2f} | "
                 f"{le['mean']:.2f} \u00b1 {le['std']:.2f} | {rde['mean']:.3f} \u00b1 {rde['std']:.3f} | "
                 f"{de['mean']:.2f} \u00b1 {de['std']:.2f} | {seld['mean']:.3f} \u00b1 {seld['std']:.3f} |")
    L.append("")
    L.append("## Paired contrasts (matched seeds)")
    for pkey, c in contrasts.items():
        L.append(f"### {pkey}")
        if c.get("n", 0) < 2:
            L.append(f"- {c.get('description')}: insufficient pairs (n={c.get('n', 0)})"); L.append(""); continue
        L.append(f"- _{c['description']}_, n={c['n']} matched seeds {c['shared_seeds']}")
        L.append("")
        L.append("| Metric | mean delta (A-B) | t (p_t) | Wilcoxon (p_w) | d_z | bootstrap 95% CI |")
        L.append("| ------ | ---------------- | ------- | -------------- | --- | ---------------- |")
        for m in ("F1", "LE", "RDE", "DE", "SELD"):
            r = c.get(m, {})
            if not r:
                L.append(f"| {m} | n/a | - | - | - | - |"); continue
            ci = r["boot_ci_95"]
            L.append(f"| {m} | {r['delta_mean']:+.3f} \u00b1 {r['delta_std']:.3f} | "
                     f"t={r['t']:+.2f} (p={r['p_t']:.3f}) | "
                     f"W={r['w']:.1f} (p={r['p_w']:.3f}) | "
                     f"{r['cohens_dz']:+.2f} | "
                     f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        L.append("")

    out_md = OUT_PATH / "path_c_foa_gca.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
