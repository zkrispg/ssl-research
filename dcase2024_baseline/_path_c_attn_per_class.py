"""Path C / Tier V (C): per-class attention map aggregation.

For 110 (GCA full, geometry_bias=True) and 111 (GCA no_geom) seed-0
checkpoints, runs inference over a sweep of STARSS23 dev-test files,
captures the GCA's 4x4 self-attention matrix per feature-sequence
chunk (250 frames = 25 label frames), and aggregates a mean attention
matrix PER GROUND-TRUTH ACTIVE CLASS (the chunks supporting that class).

Question
--------
Does the geometry-bias prior (110) restrict attention pattern more
strongly for some sound classes than others? E.g., does it impose the
same mic-pair structure across ALL classes, or does it adapt by class?

Method
------
For each test file:
  1. Forward; cache (n_seqs, 4, 4) attention.
  2. Read metadata; for each chunk i, find the set of active class IDs
     in any of its 25 label frames.
  3. Add chunk_attn[i] to a (13, 4, 4) accumulator at every active class.

Outputs:
    D:\\ssl-research\\paper\\path_c_attn_per_class.json
    D:\\ssl-research\\paper\\figs\\path_c_attn_per_class.png
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")
import parameters
from _path_c_attn_viz import (
    build_and_load,
    patch_gca_to_cache,
    FEAT_DIR,
    MODEL_DIR,
)
from _path_c_per_class import CLASS_NAMES, find_test_only_dump
from _path_c_probe import META_DIR, list_dev_test_files

OUT_PATH = Path(r"D:\ssl-research\paper")
FIG_DIR  = OUT_PATH / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CELLS_WITH_GCA = [
    {"task": "110", "name": "110_gca_full",   "title": "GCA full (geometry_bias=True)"},
    {"task": "111", "name": "111_gca_nogeom", "title": "GCA no_geom (geometry_bias=False)"},
]
SEED = 0
LABEL_HOP_S = 0.1  # 10 Hz label rate


def load_class_activity_per_label_frame(stem: str, n_label_frames: int) -> np.ndarray:
    """Return (n_label_frames, 13) bool array: row t has class c set if
    metadata says class c was active at frame t (any source).
    """
    out = np.zeros((n_label_frames, 13), dtype=bool)
    for sub in ("dev-test-sony", "dev-test-tau"):
        cand = META_DIR / sub / f"{stem}.csv"
        if cand.is_file():
            csvpath = cand; break
    else:
        return out
    with open(csvpath, "r") as fh:
        for row in csv.reader(fh):
            try:
                fr = int(row[0]); cls = int(row[1])
            except (ValueError, IndexError):
                continue
            if 0 <= fr < n_label_frames and 0 <= cls < 13:
                out[fr, cls] = True
    return out


def run_one_cell(task_id: str, file_stems: list[str]) -> dict:
    p = parameters.get_params(task_id).copy()
    feat_seq_len = p["feature_sequence_length"]
    label_seq_len = p["label_sequence_length"]
    nb_mel = p["nb_mel_bins"]
    in_ch = 4 + 6
    n_classes = 13

    accum = np.zeros((n_classes, 4, 4), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    chunks_total = 0

    model = build_and_load(task_id, SEED)
    patch_gca_to_cache(model)

    for stem in file_stems:
        npy = FEAT_DIR / f"{stem}.npy"
        if not npy.is_file(): continue
        feat = np.load(npy)
        T_total = feat.shape[0]
        feat = feat.reshape(T_total, in_ch, nb_mel)
        n_seqs = (T_total + feat_seq_len - 1) // feat_seq_len
        pad_T = n_seqs * feat_seq_len - T_total
        if pad_T:
            feat = np.concatenate([feat, np.zeros((pad_T, in_ch, nb_mel), dtype=feat.dtype)], axis=0)
        feat = feat.reshape(n_seqs, feat_seq_len, in_ch, nb_mel).transpose(0, 2, 1, 3)
        xb = torch.from_numpy(feat).float()

        # reset cache
        gca = getattr(model, "gca", None)
        if gca is not None:
            gca._cached_attn = []; gca._cached_gate = []
        with torch.no_grad():
            _ = model(xb)
        if gca is None: break  # no attn to record
        attn = torch.cat(gca._cached_attn, dim=0).numpy()  # (n_seqs, 4, 4)

        # Map each seq to label frames.
        # feat_seq_len feature frames = label_seq_len label frames per chunk.
        # so chunk i covers label frames [i*label_seq_len, (i+1)*label_seq_len).
        # Total label frames are based on T_total // 5 (for the original audio).
        n_label_total = (T_total + 4) // 5  # ceil
        class_act = load_class_activity_per_label_frame(stem, n_label_total + label_seq_len)

        for i in range(n_seqs):
            lf_start = i * label_seq_len
            lf_end   = (i + 1) * label_seq_len
            chunk_act = class_act[lf_start:lf_end]   # (label_seq_len, 13)
            if chunk_act.size == 0: continue
            active_classes = np.where(chunk_act.any(axis=0))[0]
            for c in active_classes:
                accum[c] += attn[i]
                counts[c] += 1
            chunks_total += 1

    mean_per_class = np.zeros_like(accum)
    for c in range(n_classes):
        if counts[c] > 0:
            mean_per_class[c] = accum[c] / counts[c]

    return {
        "task_id":     task_id,
        "counts":      counts.tolist(),
        "chunks_total": int(chunks_total),
        "mean_per_class": mean_per_class.tolist(),
    }


def plot_grid(results: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cells = len(results)
    fig, axes = plt.subplots(
        nrows=n_cells, ncols=13,
        figsize=(2.0 * 13, 2.4 * n_cells), squeeze=False,
    )
    for r_idx, r in enumerate(results):
        attn_per = np.asarray(r["mean_per_class"])
        counts   = np.asarray(r["counts"])
        for c in range(13):
            ax = axes[r_idx, c]
            if counts[c] == 0:
                ax.text(0.5, 0.5, "n=0", ha="center", va="center")
                ax.axis("off")
                if r_idx == 0:
                    ax.set_title(CLASS_NAMES[c], fontsize=8)
                continue
            im = ax.imshow(attn_per[c], cmap="viridis", vmin=0, vmax=attn_per[c].max())
            ax.set_xticks([]); ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(f"{CLASS_NAMES[c]}\n(n={counts[c]})", fontsize=8)
            else:
                ax.set_title(f"n={counts[c]}", fontsize=7)
            for i in range(4):
                for j in range(4):
                    val = attn_per[c, i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val < attn_per[c].max() * 0.6 else "black",
                            fontsize=6)
        axes[r_idx, 0].set_ylabel(f"{r['task_id']}\nq mic", fontsize=10)

    fig.suptitle("GCA per-class attention (4x4) -- rows: cell, cols: 13 STARSS23 classes "
                 "(seed 0, aggregated over all dev-test chunks supporting each class)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[saved] {out_png}")


def plot_difference_grid(results: list[dict], out_png: Path) -> None:
    """Plot 110 - 111 attention difference per class (so we can see where
    the geometry prior changes the pattern)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.asarray(results[0]["mean_per_class"])  # 110
    b = np.asarray(results[1]["mean_per_class"])  # 111
    counts = np.minimum(np.asarray(results[0]["counts"]),
                        np.asarray(results[1]["counts"]))
    diff = a - b  # (13, 4, 4)

    fig, axes = plt.subplots(1, 13, figsize=(2.0 * 13, 2.4), squeeze=False)
    vmax = max(np.nanmax(np.abs(diff)), 1e-3)
    for c in range(13):
        ax = axes[0, c]
        if counts[c] == 0:
            ax.text(0.5, 0.5, "n=0", ha="center", va="center"); ax.axis("off")
            ax.set_title(CLASS_NAMES[c], fontsize=8); continue
        im = ax.imshow(diff[c], cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{CLASS_NAMES[c]}\n(n={counts[c]})", fontsize=8)
    fig.suptitle("Attention difference (110 - 111) per class -- where geometry prior reshapes attention",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    file_stems = list_dev_test_files()
    if args.max_files:
        file_stems = file_stems[: args.max_files]
    print(f"using {len(file_stems)} dev-test files")

    results = []
    for cell in CELLS_WITH_GCA:
        print(f"\n=== {cell['name']} (seed {SEED}) ===")
        ckpt = MODEL_DIR / f"{cell['task']}_ablate_seed{SEED}_dev_split0_multiaccdoa_mic_gcc_model.h5"
        if not ckpt.is_file():
            print(f"  [skip] missing {ckpt}")
            continue
        r = run_one_cell(cell["task"], file_stems)
        results.append(r)
        nz = sum(1 for c in r["counts"] if c > 0)
        print(f"  classes with chunks: {nz}/13, total chunks: {r['chunks_total']}")

    payload = {
        "class_names": CLASS_NAMES,
        "results":     results,
        "n_files":     len(file_stems),
        "seed":        SEED,
    }
    out_json = OUT_PATH / "path_c_attn_per_class.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {out_json}")

    if results:
        plot_grid(results, FIG_DIR / "path_c_attn_per_class.png")
        if len(results) >= 2:
            plot_difference_grid(results, FIG_DIR / "path_c_attn_per_class_diff.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
