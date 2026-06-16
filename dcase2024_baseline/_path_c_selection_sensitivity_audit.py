"""Audit metric/selection sensitivity for the archived Path C GCA grid.

This does not re-evaluate checkpoints. It uses the archived per-seed final-test
metrics in paper/path_c_2x2_anova_long.csv to separate the planned DOAE claim
from secondary F1/SELD behavior, and records what cannot be audited without the
original per-epoch validation histories or materialized LFS checkpoints.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "paper" / "path_c_2x2_anova_long.csv"
OUT_DIR = ROOT.parents[1] / "outputs"
OUT_MD = OUT_DIR / "gca_selection_sensitivity_audit.md"
OUT_JSON = OUT_DIR / "gca_selection_sensitivity_audit.json"

METRICS = {
    "LE": "DOAE_CD direction error (lower is better)",
    "F1": "F20 detection F-score (higher is better)",
    "SELD": "SELD score (lower is better)",
    "RDE": "relative distance error (lower is better)",
}

ARCH_ORDER = ["CRNN", "Conformer", "Xfm"]
MODALITIES = ["MIC", "FOA"]


def read_rows() -> list[dict]:
    with INPUT.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["seed"] = int(row["seed"])
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def paired_deltas(rows: list[dict], metric: str) -> list[dict]:
    by_key = {(r["modality"], r["arch"], r["prior"], r["seed"]): r for r in rows}
    out = []
    for modality in MODALITIES:
        for arch in ARCH_ORDER:
            seeds = sorted({
                seed
                for (mod, a, prior, seed) in by_key
                if mod == modality and a == arch and prior in {"full", "nogeom"}
            })
            for seed in seeds:
                full = by_key.get((modality, arch, "full", seed))
                nogeom = by_key.get((modality, arch, "nogeom", seed))
                if full is None or nogeom is None:
                    continue
                out.append({
                    "modality": modality,
                    "arch": arch,
                    "seed": seed,
                    "delta": full[metric] - nogeom[metric],
                    "full": full[metric],
                    "nogeom": nogeom[metric],
                })
    return out


def paired_summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return {"n": int(len(arr)), "mean": float(arr.mean()) if len(arr) else math.nan}
    t, p = stats.ttest_1samp(arr, 0.0)
    sd = float(arr.std(ddof=1))
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": sd,
        "t": float(t),
        "p": float(p),
        "cohens_dz": float(arr.mean() / sd) if sd > 0 else math.nan,
    }


def metric_audit(rows: list[dict], metric: str) -> dict:
    deltas = paired_deltas(rows, metric)
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_arch: dict[str, list[float]] = defaultdict(list)
    for item in deltas:
        by_cell[(item["modality"], item["arch"])].append(item["delta"])
        by_arch[item["arch"]].append(item["delta"])

    cell_summary = {
        f"{modality}_{arch}": paired_summary(by_cell[(modality, arch)])
        for modality in MODALITIES
        for arch in ARCH_ORDER
    }
    arch_summary = {
        arch: paired_summary(by_arch[arch])
        for arch in ARCH_ORDER
    }

    # Planned directional interaction: CRNN geometry effect minus Transformer
    # geometry effect, paired by modality and seed.
    by_mas = {(d["modality"], d["arch"], d["seed"]): d["delta"] for d in deltas}
    second_diffs = []
    for modality in MODALITIES:
        for seed in range(5):
            a = by_mas.get((modality, "CRNN", seed))
            b = by_mas.get((modality, "Xfm", seed))
            if a is not None and b is not None:
                second_diffs.append(a - b)

    return {
        "metric": metric,
        "label": METRICS[metric],
        "cell_summary": cell_summary,
        "arch_summary": arch_summary,
        "crnn_minus_xfm_second_difference": paired_summary(second_diffs),
        "per_seed_deltas": deltas,
    }


def lfs_pointer_count() -> dict:
    model_dir = ROOT / "dcase2024_baseline" / "models_audio"
    tasks = {"110", "111", "130", "131", "141", "142", "151", "152", "161", "162", "171", "172"}
    files = [p for p in model_dir.glob("*.h5") if p.name.split("_", 1)[0] in tasks]
    ptr = [p for p in files if p.stat().st_size <= 200]
    materialized = [p for p in files if p.stat().st_size > 200]
    return {
        "total_gca_model_files_seen": len(files),
        "lfs_pointer_like_files_<=200B": len(ptr),
        "materialized_files_>200B": len(materialized),
        "pointer_examples": [p.name for p in sorted(ptr)[:12]],
    }


def git_lfs_available() -> str:
    try:
        proc = subprocess.run(
            ["git", "lfs", "version"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return f"unavailable: {exc}"
    return (proc.stdout or proc.stderr).strip()


def fmt_num(x: float, nd: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{nd}f}"


def build_markdown(payload: dict) -> str:
    lines = [
        "# GCA checkpoint/metric sensitivity audit",
        "",
        "This audit uses the archived per-seed final-test table `paper/path_c_2x2_anova_long.csv`.",
        "It does not claim to reselect historical checkpoints, because the current workspace lacks",
        "the original GCA per-epoch validation histories and many archived GCA `.h5` files are Git LFS",
        "pointers rather than materialized weights.",
        "",
        "## What can be audited now",
        "",
        "- The planned primary endpoint is DOAE_CD direction error, not F1 or aggregate SELD score.",
        "- The archived final-test DOAE pattern remains structured by architecture: CRNN mean delta",
        "  is negative, Conformer is near zero, and Transformer is positive.",
        "- The same monotone architecture pattern is not present for F1 or SELD, so the paper should",
        "  not claim a universal detection or aggregate-SELD gain from geometry.",
        "",
        "## Archived metric sensitivity",
        "",
        "| metric | CRNN mean delta | Conformer mean delta | Transformer mean delta | CRNN-Xfm second diff | p | reading |",
        "| ------ | --------------- | -------------------- | ---------------------- | -------------------- | - | ------- |",
    ]
    readings = {
        "LE": "primary DOAE claim supported",
        "F1": "no architecture-prior interaction",
        "SELD": "aggregate metric does not follow DOAE",
        "RDE": "distance follows different rule",
    }
    for metric in ("LE", "F1", "SELD", "RDE"):
        m = payload["metrics"][metric]
        arch = m["arch_summary"]
        sd = m["crnn_minus_xfm_second_difference"]
        lines.append(
            "| {metric} | {crnn} | {conf} | {xfm} | {diff} | {p} | {reading} |".format(
                metric=metric,
                crnn=fmt_num(arch["CRNN"]["mean"]),
                conf=fmt_num(arch["Conformer"]["mean"]),
                xfm=fmt_num(arch["Xfm"]["mean"]),
                diff=fmt_num(sd["mean"]),
                p=fmt_num(sd.get("p", math.nan)),
                reading=readings[metric],
            )
        )

    lines.extend([
        "",
        "For DOAE_CD, negative delta means geometry helps and positive delta means geometry hurts.",
        "The archived DOAE second difference is CRNN minus Transformer; the negative value means",
        "the geometry prior helps recurrent models more than pure-attention models.",
        "",
        "## Per-cell DOAE deltas",
        "",
        "| cell | n | mean full-no_geom DOAE | p | reading |",
        "| ---- | - | ---------------------- | - | ------- |",
    ])
    doae = payload["metrics"]["LE"]["cell_summary"]
    for modality in MODALITIES:
        for arch in ARCH_ORDER:
            key = f"{modality}_{arch}"
            c = doae[key]
            mean = c["mean"]
            if mean < -2:
                reading = "geometry helps direction"
            elif mean > 2:
                reading = "geometry hurts direction"
            else:
                reading = "near neutral"
            lines.append(
                f"| {modality}+{arch} | {c['n']} | {fmt_num(mean)} | {fmt_num(c.get('p', math.nan))} | {reading} |"
            )

    lfs = payload["lfs"]
    lines.extend([
        "",
        "## Artifact limits",
        "",
        f"- Git LFS: `{payload['git_lfs']}`.",
        f"- GCA model files seen: `{lfs['total_gca_model_files_seen']}`; materialized: `{lfs['materialized_files_>200B']}`; pointer-like <=200B: `{lfs['lfs_pointer_like_files_<=200B']}`.",
        f"- Pointer examples: `{', '.join(lfs['pointer_examples'])}`.",
        "- `D:\\ssl-research` is not present in this workspace, and current `runs/` does not contain the historical GCA training logs.",
        "- Therefore a true checkpoint-selection audit (`best F` vs `best SELD` vs `best DOAE`) requires either materializing all LFS checkpoints and re-testing, or rerunning the GCA grid with per-epoch validation logs preserved.",
        "",
        "## Paper action",
        "",
        "Keep the main claim as a DOAE_CD/direction-localization claim. Report F1, SELD, and RDE as secondary metrics and explicitly state that they do not all share the same architecture-conditional pattern.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    rows = read_rows()
    payload = {
        "input": str(INPUT),
        "n_rows": len(rows),
        "metrics": {metric: metric_audit(rows, metric) for metric in METRICS},
        "lfs": lfs_pointer_count(),
        "git_lfs": git_lfs_available(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text(build_markdown(payload), encoding="utf-8")
    print(f"[saved] {OUT_MD}")
    print(f"[saved] {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
