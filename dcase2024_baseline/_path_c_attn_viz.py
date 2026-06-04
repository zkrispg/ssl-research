"""Path C / Tier IV: GCA attention map visualization.

For one or two representative STARSS23 dev-test segments, runs inference
with both 110 (geometry_bias=True, "full") and 111 (geometry_bias=False,
"no_geom") seed-0 checkpoints, captures GCA's per-mic gate and 4x4
self-attention matrix at every time step, and plots:

  1. SED activity timeline (max class probability per frame), shared between
     the two variants.
  2. Per-mic gate (4 mics x time) for each variant.
  3. Mean 4x4 attention matrix (over time) for each variant, side by side.

Output:
    D:\\ssl-research\\paper\\figs\\path_c_attn_<file_stem>_seed0.png
    D:\\ssl-research\\paper\\figs\\path_c_attn_<file_stem>_seed0.json   (raw caches)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")  # so week09_geometry_attn import works
import parameters
import seldnet_model

DCASE_REPO = Path(r"D:\ssl-research\dcase2024_baseline")
MODEL_DIR  = DCASE_REPO / "models_audio"
FEAT_DIR   = Path(r"D:\ssl-research\DCASE2024_SELD_dataset\seld_feat_label\mic_dev_norm")
OUT_DIR    = Path(r"D:\ssl-research\paper\figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick informative dev-test files spanning rooms + recording sites
TARGET_FILES = [
    "fold4_room23_mix001",  # sony / room 23
    "fold4_room24_mix005",  # sony / room 24
    "fold4_room10_mix001",  # tau / room 10
    "fold4_room8_mix003",   # tau / room 8
    "fold4_room16_mix007",  # tau / room 16
    "fold4_room2_mix001",   # tau / room 2
]

CELLS = [("110", "GCA full (geom)"),  ("111", "GCA no_geom"),  ("112", "no GCA control")]
SEED = 0


# --------------------------------------------------------------------- caching


def patch_gca_to_cache(model: torch.nn.Module) -> torch.nn.Module:
    """Monkey-patch the model's GCA module to cache attn + gate per call.

    After every forward, the module exposes:
      ._cached_attn  -> list of (B, M, M) tensors
      ._cached_gate  -> list of (B, M) tensors
    """
    gca = getattr(model, "gca", None)
    if gca is None:
        return model
    gca._cached_attn = []
    gca._cached_gate = []
    orig_forward = gca.forward

    def new_forward(x: torch.Tensor) -> torch.Tensor:
        # Re-implement forward to expose attn + gate intermediates
        if x.ndim != 5 or x.shape[2] != gca.M:
            raise ValueError(f"expected (B, C, {gca.M}, F, T), got {tuple(x.shape)}")
        B, C, M, F, T = x.shape
        per_mic = x.mean(dim=(3, 4)).transpose(1, 2)
        e = gca.feat_proj(per_mic)
        q = gca.q_proj(e)
        k_orig = gca.k_proj(e)
        v = gca.v_proj(e)
        if gca.geometry_bias:
            geom_bias = gca.geom_proj(gca.mic_geom)
            k = k_orig.unsqueeze(1) + geom_bias.unsqueeze(0)
            scores = torch.einsum("bqd,bqkd->bqk", q, k) / math.sqrt(gca.embed_dim)
        else:
            scores = torch.einsum("bqd,bkd->bqk", q, k_orig) / math.sqrt(gca.embed_dim)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bqk,bkd->bqd", attn, v)
        gate = torch.sigmoid(gca.gate_proj(ctx))
        gca._cached_attn.append(attn.detach().cpu())
        gca._cached_gate.append(gate.detach().cpu().squeeze(-1))
        gate_b = gate.view(B, 1, M, 1, 1)
        return x * gate_b

    gca.forward = new_forward  # type: ignore[assignment]
    gca._patched = True
    return model


# --------------------------------------------------------------------- model io


def build_and_load(task_id: str, seed: int) -> torch.nn.Module:
    p = parameters.get_params(task_id).copy()
    feat_seq_len = p["feature_sequence_length"]
    label_seq_len = p["label_sequence_length"]
    n_classes = p["unique_classes"]
    nb_mel = p["nb_mel_bins"]
    in_ch = 4 + 6  # MIC GCC

    in_shape  = (1, in_ch, feat_seq_len, nb_mel)
    out_shape = (1, label_seq_len, n_classes * 3 * 4)
    model = seldnet_model.SeldModel(in_shape, out_shape, p)
    model_path = MODEL_DIR / f"{task_id}_ablate_seed{seed}_dev_split0_multiaccdoa_mic_gcc_model.h5"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()
    return model


def run_inference_with_cache(task_id: str, seed: int, stem: str, p: dict):
    model = build_and_load(task_id, seed)
    patch_gca_to_cache(model)
    feat_seq_len = p["feature_sequence_length"]
    nb_mel = p["nb_mel_bins"]
    in_ch = 4 + 6

    npy = FEAT_DIR / f"{stem}.npy"
    feat = np.load(npy)
    T_total = feat.shape[0]
    feat = feat.reshape(T_total, in_ch, nb_mel)
    n_seqs = (T_total + feat_seq_len - 1) // feat_seq_len
    pad_T = n_seqs * feat_seq_len - T_total
    if pad_T:
        feat = np.concatenate([feat, np.zeros((pad_T, in_ch, nb_mel), dtype=feat.dtype)], axis=0)
    feat = feat.reshape(n_seqs, feat_seq_len, in_ch, nb_mel).transpose(0, 2, 1, 3)
    xb = torch.from_numpy(feat).float()

    with torch.no_grad():
        yb = model(xb)
    yb = yb.cpu().numpy().reshape(-1, p["unique_classes"] * 3 * 4)

    gca = getattr(model, "gca", None)
    if gca is None:
        return {
            "task_id": task_id, "stem": stem,
            "attn": None, "gate": None, "preds": yb, "n_seqs": n_seqs, "pad_T": pad_T,
        }
    # Cached lists are length = n_seqs (one per forward batch chunk after split)
    attn = torch.cat(gca._cached_attn, dim=0).numpy()  # (n_seqs, M, M)
    gate = torch.cat(gca._cached_gate, dim=0).numpy()  # (n_seqs, M)
    return {
        "task_id": task_id, "stem": stem,
        "attn": attn, "gate": gate, "preds": yb, "n_seqs": n_seqs, "pad_T": pad_T,
    }


# --------------------------------------------------------------------- plotting


def plot_one_file(stem: str, results: list[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cells = sum(1 for r in results if r is not None)
    fig, axes = plt.subplots(
        nrows=3, ncols=max(2, n_cells), figsize=(4.5 * max(2, n_cells), 9.0),
        gridspec_kw={"height_ratios": [1.0, 1.4, 1.4]},
    )
    if axes.ndim == 1:
        axes = axes[:, None]

    # Top row: prediction max-class-prob timeline (one panel, taken from
    # cell 110 since they share the same input)
    ref = results[0]
    preds = ref["preds"]  # (T_label_total, n_classes*3*4)
    n_classes = parameters.get_params("110")["unique_classes"]
    sed = preds[:, : n_classes]
    sed_max = np.max(np.abs(sed), axis=1)
    ax = axes[0, 0]
    ax.plot(sed_max, lw=0.7, color="tab:blue")
    ax.set_title(f"max(|ACCDDOA|) timeline -- {stem}")
    ax.set_xlabel("frame (10 Hz)"); ax.set_ylabel("max abs")
    for col in range(1, axes.shape[1]):
        axes[0, col].axis("off")

    # Middle row: per-mic gate over time, one column per cell
    for col, r in enumerate(results):
        ax = axes[1, col]
        if r is None or r["gate"] is None:
            ax.text(0.5, 0.5, "no GCA module", ha="center", va="center")
            ax.axis("off"); continue
        gate = r["gate"]  # (n_seqs, 4)
        # Repeat each seq's gate over its 250 frames so we get continuous time
        feat_seq_len = parameters.get_params("110")["feature_sequence_length"]
        gate_t = np.repeat(gate, feat_seq_len, axis=0)  # (T_total_padded, 4)
        for m in range(gate_t.shape[1]):
            ax.plot(gate_t[:, m], label=f"mic{m}", lw=0.8)
        ax.set_title(f"{r['task_id']} -- per-mic gate")
        ax.set_xlabel("feature frame"); ax.set_ylabel("sigmoid gate")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1)

    # Bottom row: mean attention matrix (4x4) across time
    for col, r in enumerate(results):
        ax = axes[2, col]
        if r is None or r["attn"] is None:
            ax.text(0.5, 0.5, "no GCA module", ha="center", va="center")
            ax.axis("off"); continue
        attn = r["attn"].mean(axis=0)  # (M, M)
        im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=attn.max())
        ax.set_title(f"{r['task_id']} -- mean attn (q rows, k cols)")
        ax.set_xticks(range(attn.shape[1])); ax.set_yticks(range(attn.shape[0]))
        ax.set_xlabel("key mic"); ax.set_ylabel("query mic")
        for i in range(attn.shape[0]):
            for j in range(attn.shape[1]):
                ax.text(j, i, f"{attn[i, j]:.2f}", ha="center", va="center",
                        color="white" if attn[i, j] < attn.max()*0.6 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"GCA attention diagnostic -- {stem} (seed {SEED})", fontsize=12)
    fig.tight_layout()
    out_png = outdir / f"path_c_attn_{stem}_seed{SEED}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out_png}")


# --------------------------------------------------------------------- main


def main() -> int:
    p110 = parameters.get_params("110")

    for stem in TARGET_FILES:
        npy = FEAT_DIR / f"{stem}.npy"
        if not npy.is_file():
            print(f"[skip] missing feature {npy}")
            continue
        print(f"\n--- {stem} ---")
        results = []
        for task_id, label in CELLS:
            print(f"  forward {task_id} ({label})")
            ckpt = MODEL_DIR / f"{task_id}_ablate_seed{SEED}_dev_split0_multiaccdoa_mic_gcc_model.h5"
            if not ckpt.is_file():
                print(f"    [skip] {ckpt} missing")
                results.append(None); continue
            r = run_inference_with_cache(task_id, SEED, stem, p110)
            results.append(r)

        plot_one_file(stem, results, OUT_DIR)

        # also dump JSON for paper Appendix
        payload = {}
        for r in results:
            if r is None: continue
            payload[r["task_id"]] = {
                "gate_per_seq":  r["gate"].tolist() if r["gate"] is not None else None,
                "attn_mean":     r["attn"].mean(axis=0).tolist() if r["attn"] is not None else None,
                "attn_per_seq":  r["attn"].tolist() if r["attn"] is not None else None,
            }
        out_json = OUT_DIR / f"path_c_attn_{stem}_seed{SEED}.json"
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  [saved] {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
