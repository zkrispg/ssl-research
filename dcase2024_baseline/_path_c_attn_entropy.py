"""A2: channel-attention entropy mechanism (MIC+Transformer).

Tests, on the cell where the geometry prior hurts SELD (MIC+Transformer:
141=full vs 142=no_geom), whether the geometry bias makes the GCA channel
attention more *peaked* (lower entropy) -- i.e. over-constrains the attention
toward a fixed geometry-derived pattern instead of letting it adapt.

For each seed and cell we capture GCA's per-query channel-attention distribution
(softmax over the M=4 channels) on STARSS23 dev-test segments, and report:
  - mean attention entropy H (nats; lower = more peaked / less adaptive)
  - paired full vs no_geom contrast across seeds
  - mean M x M attention matrices for the mechanism figure

Outputs:
  D:\\ssl-research\\paper\\path_c_attn_entropy.md
  D:\\ssl-research\\paper\\figs\\path_c_attn_entropy.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats  # type: ignore

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
import parameters  # noqa: E402
import _path_c_attn_viz as av  # noqa: E402

OUT_MD = Path(r"D:\ssl-research\paper\path_c_attn_entropy.md")
FIG = Path(r"D:\ssl-research\paper\figs\path_c_attn_entropy.png")
FIG.parent.mkdir(parents=True, exist_ok=True)

CELLS = [("141", "full"), ("142", "no_geom")]
SEEDS = [0, 1, 2]


def pick_files(n: int = 6) -> list[str]:
    have = [s for s in av.TARGET_FILES if (av.FEAT_DIR / f"{s}.npy").is_file()]
    if len(have) >= 2:
        return have[:n]
    # fallback: any fold4 (dev-test) files present
    cand = sorted(p.stem for p in av.FEAT_DIR.glob("fold4_*.npy"))
    return cand[:n]


def row_entropy(attn: np.ndarray) -> float:
    """Mean Shannon entropy (nats) of attention rows. attn: (N, M, M)."""
    p = np.clip(attn, 1e-12, 1.0)
    H = -(p * np.log(p)).sum(axis=-1)  # (N, M)
    return float(H.mean())


def main() -> int:
    files = pick_files()
    if not files:
        print("[error] no dev-test feature files found"); return 1
    print(f"[info] using {len(files)} files: {files}")
    p141 = parameters.get_params("141")

    ent = {"full": [], "no_geom": []}
    matsum = {"full": None, "no_geom": None}
    matn = {"full": 0, "no_geom": 0}

    for task_id, tag in CELLS:
        for seed in SEEDS:
            attns = []
            for stem in files:
                r = av.run_inference_with_cache(task_id, seed, stem, p141)
                if r["attn"] is None:
                    continue
                attns.append(r["attn"])
            if not attns:
                print(f"[warn] no attn for {tag} seed{seed}"); continue
            A = np.concatenate(attns, axis=0)  # (N, M, M)
            ent[tag].append(row_entropy(A))
            m = A.mean(axis=0)  # (M, M)
            matsum[tag] = m if matsum[tag] is None else matsum[tag] + m
            matn[tag] += 1
            print(f"[{tag} seed{seed}] H={ent[tag][-1]:.4f} nats over {A.shape[0]} rows-of-seqs")

    full = np.array(ent["full"]); ngm = np.array(ent["no_geom"])
    n = min(len(full), len(ngm))
    full, ngm = full[:n], ngm[:n]
    d = full - ngm
    t, p_t = stats.ttest_rel(full, ngm) if n >= 2 else (float("nan"), float("nan"))
    dz = float(d.mean() / d.std(ddof=1)) if n >= 2 and d.std(ddof=1) > 0 else float("nan")

    L = ["# A2 / channel-attention entropy mechanism (MIC+Transformer, 141 vs 142)",
         "",
         f"Files: {len(files)} STARSS23 dev-test segments. Seeds: {SEEDS}.",
         "Entropy is the mean Shannon entropy (nats) of the GCA per-query channel-",
         "attention distribution over the M=4 channels. Lower = more peaked = less",
         "adaptive (max possible = ln 4 = 1.386).",
         "",
         "| variant | n seeds | mean H (nats) | std |",
         "| ------- | ------- | ------------- | --- |",
         f"| full (geom)   | {len(full)} | {full.mean():.4f} | {full.std(ddof=1) if len(full)>1 else 0:.4f} |",
         f"| no_geom       | {len(ngm)} | {ngm.mean():.4f} | {ngm.std(ddof=1) if len(ngm)>1 else 0:.4f} |",
         "",
         f"**Paired contrast (full - no_geom):** delta H = {d.mean():+.4f} nats, "
         f"t={t:+.2f} (p={p_t:.3f}), d_z={dz:+.2f}.",
         "",
         ("**Reading:** the geometry bias makes channel attention *more peaked* "
          "(lower entropy), consistent with the prior over-constraining the "
          "Transformer's channel mixing toward a fixed layout."
          if d.mean() < 0 else
          "**Reading:** the geometry bias does not reduce attention entropy; "
          "the over-constraint hypothesis is not supported by this measure.")]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))

    # figure: mean attention matrices + entropy bars
    try:
        plot(matsum, matn, full, ngm)
    except Exception as e:
        print(f"[warn] plot failed: {e}")
    print(f"\n[saved] {OUT_MD}")
    return 0


def plot(matsum, matn, full, ngm) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats = {k: (matsum[k] / matn[k]) for k in matsum if matn[k]}
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    vmax = max(m.max() for m in mats.values())
    titles = {"full": "full (geometry bias)", "no_geom": "no\\_geom"}
    for ax, key in zip(axes[:2], ["full", "no_geom"]):
        im = ax.imshow(mats[key], vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(titles[key], fontsize=10)
        ax.set_xlabel("key channel"); ax.set_ylabel("query channel")
        ax.set_xticks(range(mats[key].shape[0])); ax.set_yticks(range(mats[key].shape[0]))
        for i in range(mats[key].shape[0]):
            for j in range(mats[key].shape[1]):
                ax.text(j, i, f"{mats[key][i,j]:.2f}", ha="center", va="center",
                        color="white" if mats[key][i, j] < vmax * 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    ax = axes[2]
    means = [full.mean(), ngm.mean()]
    errs = [full.std(ddof=1) if len(full) > 1 else 0, ngm.std(ddof=1) if len(ngm) > 1 else 0]
    ax.bar([0, 1], means, yerr=errs, capsize=5, color=["#c0392b", "#b0b0b0"],
           edgecolor="black", width=0.6)
    ax.axhline(np.log(4), color="gray", ls=":", lw=0.8)
    ax.text(1.4, np.log(4), "max (ln 4)", fontsize=7, va="bottom", ha="right", color="gray")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["full", "no_geom"])
    ax.set_ylabel("mean attention entropy (nats)")
    ax.set_title("lower = more peaked", fontsize=9)
    ax.set_ylim(0, np.log(4) * 1.08)
    fig.suptitle("GCA channel-attention, MIC+Transformer (mean over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {FIG}")


if __name__ == "__main__":
    raise SystemExit(main())
