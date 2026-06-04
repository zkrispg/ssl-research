"""Path C / Tier VIII: unified (modality x architecture) 2x2 dissociation table.

Reports the geometry-prior contribution (GCA full vs GCA no_geom) on each of
the four (modality, architecture) cells:

  cell A: MIC + CRNN  -> 110 vs 111   (Stage 3, 5 seeds)
  cell B: FOA + CRNN  -> 130 vs 131   (Tier VI, 3 seeds)
  cell C: MIC + Xfm   -> 141 vs 142   (Tier VII, 3 seeds)
  cell D: FOA + Xfm   -> 151 vs 152   (Tier VIII, 3 seeds; new)

Outputs:
    D:\\ssl-research\\paper\\path_c_2x2.json
    D:\\ssl-research\\paper\\path_c_2x2.md
    D:\\ssl-research\\paper\\figs\\path_c_2x2_dissociation.png
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
FIG_DIR  = OUT_PATH / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH.mkdir(parents=True, exist_ok=True)


CELLS = [
    {"key": "MIC_CRNN", "modality": "MIC", "arch": "CRNN", "label": "MIC + CRNN",
     "gca_full":   {"task": "110", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
     "gca_nogeom": {"task": "111", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]}},
    {"key": "FOA_CRNN", "modality": "FOA", "arch": "CRNN", "label": "FOA + CRNN",
     "gca_full":   {"task": "130", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
     "gca_nogeom": {"task": "131", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]}},
    {"key": "MIC_CONF", "modality": "MIC", "arch": "Conformer", "label": "MIC + Conformer",
     "gca_full":   {"task": "161", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
     "gca_nogeom": {"task": "162", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]}},
    {"key": "FOA_CONF", "modality": "FOA", "arch": "Conformer", "label": "FOA + Conformer",
     "gca_full":   {"task": "171", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
     "gca_nogeom": {"task": "172", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]}},
    {"key": "MIC_XFM",  "modality": "MIC", "arch": "Xfm",  "label": "MIC + Xfm",
     "gca_full":   {"task": "141", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
     "gca_nogeom": {"task": "142", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]}},
    {"key": "FOA_XFM",  "modality": "FOA", "arch": "Xfm",  "label": "FOA + Xfm",
     "gca_full":   {"task": "151", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
     "gca_nogeom": {"task": "152", "job": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]}},
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


def collect_seeds(spec, job_pattern_key) -> dict[int, dict]:
    s = spec[job_pattern_key]
    out: dict[int, dict] = {}
    for seed in s["seeds"]:
        log = _resolve_log(f"dcase2024_{s['task']}_{s['job'].format(seed=seed)}_test.log")
        m = parse_log(str(log))
        if m is not None:
            out[seed] = m
    return out


def main() -> int:
    cells_payload: dict[str, dict] = {}
    for spec in CELLS:
        full = collect_seeds(spec, "gca_full")
        ngm  = collect_seeds(spec, "gca_nogeom")
        shared = sorted(set(full.keys()) & set(ngm.keys()))
        c = {
            "label":         spec["label"],
            "modality":      spec["modality"],
            "arch":          spec["arch"],
            "n_full":        len(full),
            "n_nogeom":      len(ngm),
            "shared_seeds":  shared,
        }
        for metric in ("F1", "LE", "DE", "RDE", "SELD"):
            f_vals = np.array([full[s][metric]  for s in full.keys()]) if full else np.array([])
            g_vals = np.array([ngm[s][metric]  for s in ngm.keys()])  if ngm  else np.array([])
            c[f"{metric}_full_mean"]   = float(np.mean(f_vals))                 if len(f_vals)    else None
            c[f"{metric}_full_std"]    = float(np.std(f_vals, ddof=1))          if len(f_vals)>1  else 0.0
            c[f"{metric}_nogeom_mean"] = float(np.mean(g_vals))                 if len(g_vals)    else None
            c[f"{metric}_nogeom_std"]  = float(np.std(g_vals, ddof=1))          if len(g_vals)>1  else 0.0
            if len(shared) >= 2:
                a = np.array([full[s][metric] for s in shared])
                b = np.array([ngm[s][metric]  for s in shared])
                d = a - b
                t_stat, p_t = stats.ttest_rel(a, b)
                try:
                    w_stat, p_w = stats.wilcoxon(a, b)
                except Exception:
                    w_stat, p_w = float("nan"), float("nan")
                c[metric] = {
                    "delta_mean":  float(d.mean()),
                    "delta_std":   float(d.std(ddof=1)),
                    "t":           float(t_stat),
                    "p_t":         float(p_t),
                    "w":           float(w_stat),
                    "p_w":         float(p_w),
                    "cohens_dz":   cohens_dz(d.tolist()),
                    "boot_ci_95":  list(bootstrap_ci(d.tolist())),
                }
            else:
                c[metric] = {"note": f"insufficient pairs (n={len(shared)})"}
        cells_payload[spec["key"]] = c
        print(f"[{spec['label']}] n_shared={len(shared)} F1.delta={c.get('F1', {}).get('delta_mean','-')} "
              f"DOAE.delta={c.get('LE', {}).get('delta_mean','-')}")

    payload = {"cells": cells_payload, "metric_order": ["F1", "LE", "DE", "RDE", "SELD"]}
    out_json = OUT_PATH / "path_c_2x2.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    L = ["# Path C / 2x2 dissociation: GCA geometry-prior effect across (modality, architecture)",
         "",
         "Each row reports the **paired contrast** (GCA full vs GCA no_geom) on the matched ",
         "seeds within a (modality, architecture) cell. The contrast isolates the geometry-bias ",
         "contribution from the underlying channel-attention mechanism.",
         "",
         "## DOAE_CD (deg) -- the headline metric",
         "| cell | n shared | DOAE GCA full | DOAE no_geom | delta DOAE | t (p_t) | d_z | bootstrap 95% CI |",
         "| ---- | -------- | ------------- | ------------ | ---------- | ------- | --- | ---------------- |"]
    for spec in CELLS:
        c = cells_payload[spec["key"]]
        if c["LE"].get("delta_mean") is None:
            L.append(f"| {spec['label']} | {len(c['shared_seeds'])} | n/a | n/a | n/a | - | - | - |"); continue
        m_full = c["LE_full_mean"]; s_full = c["LE_full_std"]
        m_ngm  = c["LE_nogeom_mean"]; s_ngm = c["LE_nogeom_std"]
        d = c["LE"]
        ci = d["boot_ci_95"]
        L.append(f"| **{spec['label']}** | {len(c['shared_seeds'])} | {m_full:.2f} \u00b1 {s_full:.2f} | "
                 f"{m_ngm:.2f} \u00b1 {s_ngm:.2f} | **{d['delta_mean']:+.2f}** | "
                 f"t={d['t']:+.2f} (p={d['p_t']:.3f}) | **{d['cohens_dz']:+.2f}** | "
                 f"[{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    L.append("")

    L += ["## F1 (%) -- detection",
          "| cell | n shared | F1 GCA full | F1 no_geom | delta F1 | t (p_t) | d_z |",
          "| ---- | -------- | ----------- | ---------- | -------- | ------- | --- |"]
    for spec in CELLS:
        c = cells_payload[spec["key"]]
        if c["F1"].get("delta_mean") is None:
            L.append(f"| {spec['label']} | {len(c['shared_seeds'])} | n/a | n/a | n/a | - | - |"); continue
        m_full = c["F1_full_mean"]; s_full = c["F1_full_std"]
        m_ngm  = c["F1_nogeom_mean"]; s_ngm = c["F1_nogeom_std"]
        d = c["F1"]
        L.append(f"| {spec['label']} | {len(c['shared_seeds'])} | {m_full:.2f} \u00b1 {s_full:.2f} | "
                 f"{m_ngm:.2f} \u00b1 {s_ngm:.2f} | {d['delta_mean']:+.2f} | "
                 f"t={d['t']:+.2f} (p={d['p_t']:.3f}) | {d['cohens_dz']:+.2f} |")
    L.append("")

    L += ["## SELD score -- joint metric (lower is better)",
          "| cell | n shared | SELD GCA full | SELD no_geom | delta SELD | t (p_t) | d_z |",
          "| ---- | -------- | ------------- | ------------ | ---------- | ------- | --- |"]
    for spec in CELLS:
        c = cells_payload[spec["key"]]
        if c["SELD"].get("delta_mean") is None:
            L.append(f"| {spec['label']} | {len(c['shared_seeds'])} | n/a | n/a | n/a | - | - |"); continue
        m_full = c["SELD_full_mean"]; s_full = c["SELD_full_std"]
        m_ngm  = c["SELD_nogeom_mean"]; s_ngm = c["SELD_nogeom_std"]
        d = c["SELD"]
        L.append(f"| {spec['label']} | {len(c['shared_seeds'])} | {m_full:.3f} \u00b1 {s_full:.3f} | "
                 f"{m_ngm:.3f} \u00b1 {s_ngm:.3f} | {d['delta_mean']:+.3f} | "
                 f"t={d['t']:+.2f} (p={d['p_t']:.3f}) | {d['cohens_dz']:+.2f} |")
    L.append("")

    out_md = OUT_PATH / "path_c_2x2.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"[saved] {out_md}")

    # ---------- 2x2 visualization
    try:
        plot_2x2(cells_payload, FIG_DIR / "path_c_2x2_dissociation.png")
    except Exception as e:
        print(f"[warn] plot failed: {e}")
    return 0


def plot_2x2(cells: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    labels = []; doae_d = []; doae_ci_lo = []; doae_ci_hi = []
    f1_d = []; f1_ci_lo = []; f1_ci_hi = []
    colors = []
    for spec in CELLS:
        c = cells[spec["key"]]
        labels.append(spec["label"])
        # color: blue for help, red for hurt, gray for null
        d = c["LE"].get("delta_mean")
        dz = c["LE"].get("cohens_dz", 0.0)
        if d is None or np.isnan(dz):
            colors.append("lightgray"); doae_d.append(np.nan); doae_ci_lo.append(np.nan); doae_ci_hi.append(np.nan)
        else:
            if d <= -2 and dz <= -1:   colors.append("tab:blue")
            elif d >= 2 and dz >=  1:  colors.append("tab:red")
            else:                      colors.append("lightgray")
            ci = c["LE"]["boot_ci_95"]
            doae_d.append(d); doae_ci_lo.append(ci[0]); doae_ci_hi.append(ci[1])
        # F1 alongside
        if c["F1"].get("delta_mean") is None:
            f1_d.append(np.nan); f1_ci_lo.append(np.nan); f1_ci_hi.append(np.nan)
        else:
            f1_d.append(c["F1"]["delta_mean"])
            ci2 = c["F1"]["boot_ci_95"]
            f1_ci_lo.append(ci2[0]); f1_ci_hi.append(ci2[1])

    # DOAE plot
    x = np.arange(len(labels))
    yerr = np.stack([np.asarray(doae_d) - np.asarray(doae_ci_lo),
                     np.asarray(doae_ci_hi) - np.asarray(doae_d)])
    axes[0].bar(x, doae_d, color=colors, edgecolor="black", linewidth=0.8, yerr=yerr, capsize=6)
    axes[0].axhline(0, color="black", lw=0.7, ls=":")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylabel("delta DOAE_CD (deg, GCA full - GCA no_geom)")
    axes[0].set_title("Geometry-prior contribution to DOAE_CD\n(modality x architecture dissociation)")
    axes[0].grid(axis="y", alpha=0.3)

    # F1 plot
    yerr2 = np.stack([np.asarray(f1_d) - np.asarray(f1_ci_lo),
                      np.asarray(f1_ci_hi) - np.asarray(f1_d)])
    axes[1].bar(x, f1_d, color="lightsteelblue", edgecolor="black", linewidth=0.8, yerr=yerr2, capsize=6)
    axes[1].axhline(0, color="black", lw=0.7, ls=":")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylabel("delta F 20\u00b0 (pp, GCA full - GCA no_geom)")
    axes[1].set_title("Geometry-prior contribution to F1")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Geometry-prior x (modality, architecture) dissociation -- "
                 "DOAE shows clean cross-cell pattern, F1 mostly inert",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[saved plot] {out_png}")


if __name__ == "__main__":
    sys.exit(main())
