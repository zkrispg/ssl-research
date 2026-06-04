"""Path C / Tier V (A): per-class breakdown of SELD metrics.

For each cell x seed, locates the existing `*_test_only` CSV dump folder
under results_audio/, re-runs the official SELD evaluator with `is_jackknife=False`
to get the `classwise_results` array (shape (7, n_classes) when eval_dist=True),
and aggregates per-cell mean/std and per-class paired contrasts.

Class metric layout (rows of classwise_results when eval_dist=True):
    0: ER (broadcast across classes)   <-- not class-specific, ignored here
    1: F1 (location-aware F-score)
    2: AngE (DOAE_CD, deg)
    3: DistE (m)
    4: RelDistE (rel)
    5: LR (location recall)
    6: SELD score (per class)

Outputs:
    D:\\ssl-research\\paper\\path_c_per_class.json
    D:\\ssl-research\\paper\\path_c_per_class.md
    D:\\ssl-research\\paper\\figs\\path_c_per_class_F1.png
    D:\\ssl-research\\paper\\figs\\path_c_per_class_DOAE.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats  # type: ignore

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
import parameters
from cls_compute_seld_results import ComputeSELDResults

DCASE_REPO  = Path(r"D:\ssl-research\dcase2024_baseline")
RESULTS_DIR = DCASE_REPO / "results_audio"
OUT_PATH    = Path(r"D:\ssl-research\paper")
FIG_DIR     = OUT_PATH / "figs"
OUT_PATH.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CELLS = [
    # MIC + CRNN (Stage 3 + 4)
    {"task": "110", "name": "110_gca_full",      "modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    {"task": "111", "name": "111_gca_nogeom",    "modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    {"task": "112", "name": "112_no_gca",        "modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    {"task": "113", "name": "113_vanilla_se",    "modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    # FOA + CRNN (Stage 1 + Tier VI)
    {"task": "100", "name": "100_foa_no_gca",    "modality": "foa", "seeds": [0, 1, 2, 3, 4],
     "job_pattern": "repro_seed{seed}"},
    {"task": "130", "name": "130_foa_gca_full",  "modality": "foa", "seeds": [0, 1, 2, 3, 4]},
    {"task": "131", "name": "131_foa_gca_nogeom","modality": "foa", "seeds": [0, 1, 2, 3, 4]},
    # MIC + Xfm (Tier VII)
    {"task": "140", "name": "140_xfm_no_gca",    "modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    {"task": "141", "name": "141_xfm_gca_full",  "modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    {"task": "142", "name": "142_xfm_gca_nogeom","modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    # FOA + Xfm (Tier VIII)
    {"task": "150", "name": "150_xfm_foa_no_gca",   "modality": "foa", "seeds": [0, 1, 2, 3, 4]},
    {"task": "151", "name": "151_xfm_foa_gca_full", "modality": "foa", "seeds": [0, 1, 2, 3, 4]},
    {"task": "152", "name": "152_xfm_foa_gca_nogeom","modality": "foa","seeds": [0, 1, 2, 3, 4]},
    # MIC + Conformer (journal Tranche 2)
    {"task": "160", "name": "160_conf_no_gca",    "modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    {"task": "161", "name": "161_conf_gca_full",  "modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    {"task": "162", "name": "162_conf_gca_nogeom","modality": "mic", "seeds": [0, 1, 2, 3, 4]},
    # FOA + Conformer (journal Tranche 2)
    {"task": "170", "name": "170_conf_foa_no_gca",   "modality": "foa", "seeds": [0, 1, 2, 3, 4]},
    {"task": "171", "name": "171_conf_foa_gca_full", "modality": "foa", "seeds": [0, 1, 2, 3, 4]},
    {"task": "172", "name": "172_conf_foa_gca_nogeom","modality": "foa","seeds": [0, 1, 2, 3, 4]},
]
SEEDS = [0, 1, 2, 3, 4]  # legacy default; per-cell `seeds` override

# STARSS23 13-class taxonomy (DCASE 2024 Task 3 dev-set classes)
CLASS_NAMES = [
    "femaleSpeech",  # 0
    "maleSpeech",    # 1
    "clapping",      # 2
    "telephone",     # 3
    "laughter",      # 4
    "domesticSnd",   # 5
    "footsteps",     # 6
    "doorOpen",      # 7
    "music",         # 8
    "instrument",    # 9
    "waterTap",      # 10
    "bell",          # 11
    "knock",         # 12
]

# rows we care about (skip ER which is broadcast)
ROW_NAMES = ["F1", "DOAE_CD", "DistE", "RelDistE", "LR", "SELD_per_class"]
ROW_INDICES = [1, 2, 3, 4, 5, 6]  # index into classwise_results
HIGHER_IS_BETTER = {"F1": True, "LR": True, "DOAE_CD": False, "DistE": False, "RelDistE": False, "SELD_per_class": False}


# --------------------------------------------------------------------- helpers


_TS_RE = re.compile(r"_(\d{14})_test_only$")


def find_test_only_dump(task_id: str, seed: int,
                        modality: str = "mic",
                        job_pattern: str = "ablate_seed{seed}") -> Optional[Path]:
    """Find the most-recent test_only dump for the given (task, seed, modality)."""
    job = job_pattern.format(seed=seed)
    feat_kind = "mic_gcc" if modality == "mic" else "foa"
    prefix = f"{task_id}_{job}_dev_split0_multiaccdoa_{feat_kind}_"
    candidates = []
    for d in RESULTS_DIR.iterdir():
        if not d.is_dir(): continue
        name = d.name
        if not name.startswith(prefix): continue
        if not name.endswith("_test_only"): continue
        m = _TS_RE.search(name)
        ts = m.group(1) if m else ""
        candidates.append((ts, d))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def evaluate_dump(dump_dir: Path, score_obj: ComputeSELDResults) -> dict:
    """Run SELD evaluator on a dump folder. Returns aggregate + classwise."""
    pred_count = sum(1 for _ in dump_dir.iterdir())
    if pred_count == 0:
        raise RuntimeError(f"empty dump dir {dump_dir}")
    res = score_obj.get_SELD_Results(str(dump_dir), is_jackknife=False)
    # res = (ER, F, AngE, DistE, RelDistE, LR, seld_scr, classwise) when eval_dist=True
    ER, F, AngE, DistE, RelDistE, LR, seld_scr, classwise = res
    cw = np.asarray(classwise, dtype=np.float64)  # (7, 13)
    return {
        "pred_files": pred_count,
        "aggregate": {
            "ER":          float(ER),
            "F1":          float(F),
            "AngE":        float(AngE),
            "DistE":       float(DistE),
            "RelDistE":    float(RelDistE),
            "LR":          float(LR),
            "SELD":        float(seld_scr),
        },
        "classwise":   cw.tolist(),
        "row_names":   ROW_NAMES,
        "class_names": CLASS_NAMES,
    }


# --------------------------------------------------------------------- stats


def cohens_dz(deltas):
    a = np.asarray(deltas, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) < 2: return float("nan")
    s = a.std(ddof=1)
    return float("nan") if s == 0 else float(a.mean() / s)


def paired_test(a_vals, b_vals) -> dict:
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return {"n": int(mask.sum()), "note": "insufficient pairs"}
    a, b = a[mask], b[mask]
    deltas = (a - b).tolist()
    t_stat, p_t = stats.ttest_rel(a, b)
    try:
        w_stat, p_w = stats.wilcoxon(a, b)
    except Exception:
        w_stat, p_w = float("nan"), float("nan")
    return {
        "n":               int(mask.sum()),
        "a_per_seed":      a.tolist(),
        "b_per_seed":      b.tolist(),
        "delta_per_seed":  deltas,
        "delta_mean":      float(np.mean(deltas)),
        "delta_std":       float(np.std(deltas, ddof=1)),
        "ttest_rel":       {"t": float(t_stat), "p": float(p_t)},
        "wilcoxon":        {"w": float(w_stat), "p": float(p_w)},
        "cohens_dz":       cohens_dz(deltas),
    }


# --------------------------------------------------------------------- main pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None,
                    help="Restrict to these task IDs (e.g. 110 112 113)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # All four cells share STARSS23 dev evaluation, so a single ComputeSELDResults
    # built from task 110's params reuses the same _ref_labels for everyone.
    print("[info] loading reference labels from STARSS23 metadata_dev (~10 s)...")
    cwd = os.getcwd()
    os.chdir(DCASE_REPO)  # so relative dataset_dir resolves
    p = parameters.get_params("110")
    score_obj = ComputeSELDResults(p)
    os.chdir(cwd)
    print(f"[info] {len(score_obj._ref_labels)} reference files loaded.")

    cells_filter = args.cells or [c["task"] for c in CELLS]
    seeds_filter = args.seeds  # None means use per-cell `seeds`

    per_ckpt: dict[str, dict] = {}
    for cell in CELLS:
        task_id = cell["task"]
        if task_id not in cells_filter:
            continue
        modality   = cell.get("modality", "mic")
        job_pattern = cell.get("job_pattern", "ablate_seed{seed}")
        cell_seeds = seeds_filter if seeds_filter is not None else cell.get("seeds", SEEDS)
        for s in cell_seeds:
            print(f"\n=== {task_id} seed {s} (modality={modality}) ===")
            dump = find_test_only_dump(task_id, s, modality=modality, job_pattern=job_pattern)
            if dump is None:
                print(f"  [skip] no test_only dump found")
                continue
            print(f"  dump: {dump.name}")
            try:
                r = evaluate_dump(dump, score_obj)
            except Exception as e:
                print(f"  [error] {e}")
                continue
            per_ckpt[f"{task_id}_seed{s}"] = r
            agg = r["aggregate"]
            print(f"  agg F1 = {100*agg['F1']:.2f}%, DOAE_CD = {agg['AngE']:.2f}, SELD = {agg['SELD']:.3f}")

    # ---------- aggregate per cell, per class
    print("\n[agg] aggregating per cell...")
    cell_results: dict[str, dict] = {}
    for cell in CELLS:
        if cell["task"] not in cells_filter: continue
        cell_seeds = seeds_filter if seeds_filter is not None else cell.get("seeds", SEEDS)
        per_seed_cw = []
        per_seed_keys = []
        for s in cell_seeds:
            k = f"{cell['task']}_seed{s}"
            if k in per_ckpt:
                per_seed_cw.append(np.asarray(per_ckpt[k]["classwise"], dtype=np.float64))
                per_seed_keys.append(s)
        if not per_seed_cw:
            cell_results[cell["name"]] = {"n_seeds": 0}
            continue
        stack = np.stack(per_seed_cw, axis=0)  # (S, 7, 13)
        # Replace ER row 0 (broadcast) with NaN -- not class-specific.
        stack[:, 0, :] = np.nan
        cell_results[cell["name"]] = {
            "n_seeds":      stack.shape[0],
            "seeds":        per_seed_keys,
            "rows":         ROW_NAMES,
            "row_indices":  ROW_INDICES,
            "classes":      CLASS_NAMES,
            "per_seed":     stack.tolist(),
            "mean":         np.nanmean(stack, axis=0).tolist(),  # (7, 13)
            "std":          np.nanstd(stack, axis=0, ddof=1).tolist() if stack.shape[0] > 1 else np.zeros((7, 13)).tolist(),
        }

    # ---------- per-class paired contrasts (compute for F1 and DOAE_CD)
    print("\n[stats] computing per-class contrasts...")
    pair_specs = [
        # MIC + CRNN (Stage 3 + 4)
        ("110_gca_full",   "112_no_gca",         "MIC+CRNN: GCA full vs no-GCA"),
        ("110_gca_full",   "111_gca_nogeom",     "MIC+CRNN: GCA full vs no_geom (geometry contribution)"),
        ("110_gca_full",   "113_vanilla_se",     "MIC+CRNN: GCA full vs Vanilla SE"),
        ("113_vanilla_se", "112_no_gca",         "MIC+CRNN: Vanilla SE vs no-GCA"),
        ("111_gca_nogeom", "112_no_gca",         "MIC+CRNN: GCA no_geom vs no-GCA"),
        # FOA + CRNN (Tier VI)
        ("130_foa_gca_full",   "131_foa_gca_nogeom", "FOA+CRNN: GCA full vs no_geom (geometry contribution)"),
        ("130_foa_gca_full",   "100_foa_no_gca",     "FOA+CRNN: GCA full vs no GCA"),
        # MIC + Xfm (Tier VII)
        ("141_xfm_gca_full",   "142_xfm_gca_nogeom", "MIC+Xfm: GCA full vs no_geom (geometry contribution)"),
        ("141_xfm_gca_full",   "140_xfm_no_gca",     "MIC+Xfm: GCA full vs no GCA"),
        # FOA + Xfm (Tier VIII)
        ("151_xfm_foa_gca_full", "152_xfm_foa_gca_nogeom", "FOA+Xfm: GCA full vs no_geom (geometry contribution)"),
        ("151_xfm_foa_gca_full", "150_xfm_foa_no_gca",     "FOA+Xfm: GCA full vs no GCA"),
        # MIC + Conformer (Tranche 2)
        ("161_conf_gca_full", "162_conf_gca_nogeom", "MIC+Conf: GCA full vs no_geom (geometry contribution)"),
        ("161_conf_gca_full", "160_conf_no_gca",     "MIC+Conf: GCA full vs no GCA"),
        # FOA + Conformer (Tranche 2)
        ("171_conf_foa_gca_full", "172_conf_foa_gca_nogeom", "FOA+Conf: GCA full vs no_geom (geometry contribution)"),
        ("171_conf_foa_gca_full", "170_conf_foa_no_gca",     "FOA+Conf: GCA full vs no GCA"),
    ]
    per_class_contrasts: dict[str, dict] = {}
    for a_name, b_name, descr in pair_specs:
        if a_name not in cell_results or b_name not in cell_results:
            continue
        if cell_results[a_name].get("n_seeds", 0) < 2 or cell_results[b_name].get("n_seeds", 0) < 2:
            continue
        a_seeds = cell_results[a_name]["seeds"]
        b_seeds = cell_results[b_name]["seeds"]
        shared = sorted(set(a_seeds) & set(b_seeds))
        if len(shared) < 2:
            continue
        a_stack = np.stack([np.asarray(per_ckpt[f"{cell_lookup_task(a_name)}_seed{s}"]["classwise"], dtype=np.float64)
                            for s in shared], axis=0)
        b_stack = np.stack([np.asarray(per_ckpt[f"{cell_lookup_task(b_name)}_seed{s}"]["classwise"], dtype=np.float64)
                            for s in shared], axis=0)
        # both stacks: (S, 7, 13)
        per_metric = {}
        for ri, rname in zip(ROW_INDICES, ROW_NAMES):
            per_class_test = []
            for ci, cname in enumerate(CLASS_NAMES):
                a_vals = a_stack[:, ri, ci]
                b_vals = b_stack[:, ri, ci]
                t = paired_test(a_vals, b_vals)
                t["class"] = cname
                per_class_test.append(t)
            per_metric[rname] = per_class_test
        per_class_contrasts[f"{a_name}__vs__{b_name}"] = {
            "description": descr,
            "shared_seeds": shared,
            "per_metric":   per_metric,
        }

    # ---------- save JSON + Markdown
    payload = {
        "per_ckpt":            per_ckpt,
        "per_cell":            cell_results,
        "per_class_contrasts": per_class_contrasts,
        "class_names":         CLASS_NAMES,
        "row_names":           ROW_NAMES,
    }
    out_json = OUT_PATH / "path_c_per_class.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    write_md(payload, OUT_PATH / "path_c_per_class.md")
    print(f"[saved] {OUT_PATH / 'path_c_per_class.md'}")

    # ---------- plots (F1 bar, DOAE bar)
    if not args.no_plot:
        try:
            plot_per_class_bars(payload, FIG_DIR)
        except Exception as e:
            print(f"[warn] plotting failed: {e}")

    return 0


def cell_lookup_task(name: str) -> str:
    for c in CELLS:
        if c["name"] == name: return c["task"]
    raise KeyError(name)


def write_md(payload: dict, path: Path) -> None:
    L = ["# Path C / per-class SELD breakdown",
         "",
         "Per-class metrics on STARSS23 dev-test (macro mode), aggregated across 5 seeds per cell.",
         "Source: `_path_c_per_class.py` re-evaluating existing `*_test_only` CSV dumps.",
         ""]

    # Per-cell F1 table
    cell_results = payload["per_cell"]
    L.append("## Per-class F1 (%, mean +/- std across seeds)")
    L.append("| class | " + " | ".join(c["name"] for c in CELLS) + " |")
    L.append("| ----- | " + " | ".join("-----" for _ in CELLS) + " |")
    for ci, cname in enumerate(CLASS_NAMES):
        row = [cname]
        for c in CELLS:
            cn = c["name"]
            if cn not in cell_results or cell_results[cn].get("n_seeds", 0) == 0:
                row.append("n/a"); continue
            mean = np.asarray(cell_results[cn]["mean"])  # (7, 13)
            std  = np.asarray(cell_results[cn]["std"])
            row.append(f"{100*mean[1, ci]:.1f} +/- {100*std[1, ci]:.1f}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # Per-cell DOAE_CD table
    L.append("## Per-class DOAE_CD (deg, mean +/- std across seeds)")
    L.append("| class | " + " | ".join(c["name"] for c in CELLS) + " |")
    L.append("| ----- | " + " | ".join("-----" for _ in CELLS) + " |")
    for ci, cname in enumerate(CLASS_NAMES):
        row = [cname]
        for c in CELLS:
            cn = c["name"]
            if cn not in cell_results or cell_results[cn].get("n_seeds", 0) == 0:
                row.append("n/a"); continue
            mean = np.asarray(cell_results[cn]["mean"])
            std  = np.asarray(cell_results[cn]["std"])
            v = mean[2, ci]
            if np.isnan(v):
                row.append("n/a")
            else:
                row.append(f"{v:.1f} +/- {std[2, ci]:.1f}")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # Paired per-class contrasts (F1)
    L.append("## Per-class F1 paired contrasts (delta = A - B in raw F1, signed)")
    for pair_key, pair in payload["per_class_contrasts"].items():
        L.append(f"### {pair_key}  ({pair['description']}, n={len(pair['shared_seeds'])} seeds)")
        L.append("| class | delta_mean (F1 pp) | t | p_t | p_wilcoxon | d_z |")
        L.append("| ----- | ------------------ | - | --- | ---------- | --- |")
        f1_rows = pair["per_metric"]["F1"]
        for r in f1_rows:
            if "delta_mean" not in r:
                L.append(f"| {r['class']} | n/a | - | - | - | - |"); continue
            dm = r["delta_mean"] * 100  # raw F1 -> percentage points
            t  = r["ttest_rel"]["t"]
            p_t = r["ttest_rel"]["p"]
            p_w = r["wilcoxon"]["p"]
            dz  = r["cohens_dz"]
            L.append(f"| {r['class']} | {dm:+.2f} | {t:+.2f} | {p_t:.3f} | {p_w:.3f} | {dz:+.2f} |")
        L.append("")

    # Paired per-class contrasts (DOAE_CD)
    L.append("## Per-class DOAE_CD paired contrasts (delta = A - B in deg)")
    for pair_key, pair in payload["per_class_contrasts"].items():
        L.append(f"### {pair_key}  ({pair['description']}, n={len(pair['shared_seeds'])} seeds)")
        L.append("| class | delta_mean (deg) | t | p_t | p_wilcoxon | d_z |")
        L.append("| ----- | ---------------- | - | --- | ---------- | --- |")
        rows = pair["per_metric"]["DOAE_CD"]
        for r in rows:
            if "delta_mean" not in r:
                L.append(f"| {r['class']} | n/a | - | - | - | - |"); continue
            dm = r["delta_mean"]
            t  = r["ttest_rel"]["t"]
            p_t = r["ttest_rel"]["p"]
            p_w = r["wilcoxon"]["p"]
            dz  = r["cohens_dz"]
            L.append(f"| {r['class']} | {dm:+.2f} | {t:+.2f} | {p_t:.3f} | {p_w:.3f} | {dz:+.2f} |")
        L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


def plot_per_class_bars(payload: dict, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_results = payload["per_cell"]
    cell_names = [c["name"] for c in CELLS if c["name"] in cell_results
                  and cell_results[c["name"]].get("n_seeds", 0) > 0]
    if not cell_names:
        print("[plot] no cell results, skipping")
        return

    # F1 bar chart
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    x = np.arange(len(CLASS_NAMES))
    w = 0.20
    colors = ["tab:red", "tab:orange", "tab:gray", "tab:blue"][:len(cell_names)]
    for i, cn in enumerate(cell_names):
        mean = np.asarray(cell_results[cn]["mean"])
        std  = np.asarray(cell_results[cn]["std"])
        f1_pct = 100 * mean[1]
        f1_std = 100 * std[1]
        ax.bar(x + (i - len(cell_names)/2 + 0.5) * w, f1_pct, w,
               yerr=f1_std, label=cn, color=colors[i], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("F 20° per class (%)")
    ax.set_title("Per-class F1 on STARSS23 dev-test (5 seeds, macro)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "path_c_per_class_F1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")

    # DOAE bar chart
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    for i, cn in enumerate(cell_names):
        mean = np.asarray(cell_results[cn]["mean"])
        std  = np.asarray(cell_results[cn]["std"])
        d = mean[2]
        d_std = std[2]
        ax.bar(x + (i - len(cell_names)/2 + 0.5) * w, d, w,
               yerr=d_std, label=cn, color=colors[i], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("DOAE_CD per class (deg)")
    ax.set_title("Per-class DOAE_CD on STARSS23 dev-test (5 seeds, macro)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "path_c_per_class_DOAE.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] {out}")


if __name__ == "__main__":
    sys.exit(main())
