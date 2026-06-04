"""A4: layer-wise linear probing of the MIC+Transformer cell.

Localizes WHERE the geometry-prior information loss arises. We probe direction
(sin/cos of az,el) from the representation at increasing depth:

  depth 0 = post-conv (input to the temporal stack; this is where GCA acts)
  depth 1..4 = output of each of the 4 TransformerEncoder layers

for 141 (geometry full) vs 142 (no_geom), seeds 0..2. If the geometry-biased
representation is already worse at depth 0 and the gap persists through depth,
the loss is localized to the pre-temporal GCA stage rather than created by the
Transformer. Runs on CPU to avoid contending with other GPU jobs.

Outputs:
  D:\\ssl-research\\paper\\path_c_probe_layerwise.md
  D:\\ssl-research\\paper\\figs\\path_c_probe_layerwise.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats  # type: ignore

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
import parameters  # noqa: E402
import _path_c_probe as pb  # noqa: E402

OUT_MD = Path(r"D:\ssl-research\paper\path_c_probe_layerwise.md")
FIG = Path(r"D:\ssl-research\paper\figs\path_c_probe_layerwise.png")
FIG.parent.mkdir(parents=True, exist_ok=True)

CELLS = [("141", "full"), ("142", "no_geom")]
SEEDS = [0, 1, 2]
MODALITY = "mic"


def extract_multi(model, npy: Path, p: dict):
    """Single forward; capture post-conv + each transformer layer output.
    Returns dict depth_name -> (T_label_total, D) feature matrix.
    """
    feat_seq_len = p["feature_sequence_length"]
    nb_mel = p["nb_mel_bins"]
    in_ch = 4 + (6 if MODALITY == "mic" else 3)

    caps: dict[str, list] = {}
    handles = []

    def mk(name):
        caps[name] = []
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            caps[name].append(o.detach().cpu())
        return hook

    handles.append(model.conv_block_list[-1].register_forward_hook(mk("conv")))
    layers = model.transformer_encoder.layers
    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(mk(f"L{li+1}")))

    feat = np.load(npy)
    T_total = feat.shape[0]
    feat = feat.reshape(T_total, in_ch, nb_mel)
    n_seqs = (T_total + feat_seq_len - 1) // feat_seq_len
    pad_T = n_seqs * feat_seq_len - T_total
    if pad_T:
        feat = np.concatenate([feat, np.zeros((pad_T, in_ch, nb_mel), dtype=feat.dtype)], axis=0)
    feat = feat.reshape(n_seqs, feat_seq_len, in_ch, nb_mel).transpose(0, 2, 1, 3)
    xb = torch.from_numpy(feat).float()
    try:
        with torch.no_grad():
            _ = model(xb)
    finally:
        for h in handles:
            h.remove()

    pad_T_label = (pad_T + 4) // 5
    out = {}
    for name, lst in caps.items():
        t = torch.cat(lst, dim=0)
        if t.ndim == 4:  # post-conv (n_seqs, C, T, F)
            n, C, d2, d3 = t.shape
            # T axis is the one equal to T_label (50); F is the other
            label_len = p["label_sequence_length"]
            if d2 == label_len:
                t = t.permute(0, 2, 1, 3)  # (n, T, C, F)
            else:
                t = t.permute(0, 3, 1, 2)
            pooled = torch.cat([t.mean(-1), t.amax(-1)], dim=-1)  # (n, T, 2C)
            arr = pooled.reshape(-1, pooled.shape[-1]).numpy()
        else:  # transformer layer (n_seqs, T, d_model)
            arr = t.reshape(-1, t.shape[-1]).numpy()
        if pad_T_label:
            arr = arr[: arr.shape[0] - pad_T_label]
        out[name] = arr
    return out


def probe_depths_one_ckpt(task_id, seed, stems):
    p = parameters.get_params(task_id).copy()
    try:
        model = pb.build_and_load(task_id, seed, modality=MODALITY).to("cpu")
    except FileNotFoundError as e:
        print(f"  [skip] {e}"); return None
    feat_dir = pb.feat_dir_for(MODALITY)
    depth_X: dict[str, list] = {}
    ys, fis = [], []
    for fi, stem in enumerate(stems):
        npy = feat_dir / f"{stem}.npy"
        feats = extract_multi(model, npy, p)
        T_label = feats["conv"].shape[0]
        az, el, mask = pb.load_az_el_targets(stem, T_label)
        if mask.sum() == 0:
            continue
        az_r = np.deg2rad(az[mask]); el_r = np.deg2rad(el[mask])
        yf = np.stack([np.sin(az_r), np.cos(az_r), np.sin(el_r), np.cos(el_r)], axis=1)
        ys.append(yf); fis.append(np.full(int(mask.sum()), fi, dtype=np.int32))
        for name, arr in feats.items():
            depth_X.setdefault(name, []).append(arr[mask])
    del model
    if not ys:
        return None
    y = np.concatenate(ys); fi = np.concatenate(fis)
    res = {}
    for name, parts in depth_X.items():
        X = np.concatenate(parts, axis=0)
        res[name] = pb.fit_probe_kfold(X, y, fi, n_signals=2, k=5)["mae_mean"]
    return res


def main() -> int:
    stems = pb.list_dev_test_files(MODALITY)
    print(f"[info] {len(stems)} dev-test files")
    depth_order = ["conv", "L1", "L2", "L3", "L4"]
    per = {tag: {d: [] for d in depth_order} for _, tag in CELLS}
    for task_id, tag in CELLS:
        for seed in SEEDS:
            print(f"[{tag} seed{seed}] probing depths ...")
            r = probe_depths_one_ckpt(task_id, seed, stems)
            if r is None:
                continue
            for d in depth_order:
                if d in r and not np.isnan(r[d]):
                    per[tag][d].append(r[d])
            print("   " + "  ".join(f"{d}={r.get(d, float('nan')):.2f}" for d in depth_order))

    L = ["# A4 / layer-wise probing: MIC+Transformer (141 full vs 142 no_geom)",
         "",
         "Angular probe MAE (deg; lower = direction more linearly decodable) at",
         "increasing depth. depth 'conv' = post-conv (input to temporal, where GCA",
         "acts); L1..L4 = TransformerEncoder layer outputs. Mean over seeds 0..2.",
         "",
         "| depth | full MAE | no_geom MAE | delta (full-no_geom) | d_z |",
         "| ----- | -------- | ----------- | -------------------- | --- |"]
    for d in depth_order:
        f = np.array(per["full"][d]); g = np.array(per["no_geom"][d])
        n = min(len(f), len(g))
        if n == 0:
            L.append(f"| {d} | n/a | n/a | n/a | - |"); continue
        f, g = f[:n], g[:n]
        delta = f - g
        dz = pb.cohens_dz(delta.tolist()) if n >= 2 else float("nan")
        L.append(f"| {d} | {f.mean():.2f} | {g.mean():.2f} | {delta.mean():+.2f} | {dz:+.2f} |")
    L += ["",
          "**Reading:** a geometry-prior gap (full worse) present already at 'conv'",
          "and persisting across L1..L4 localizes the harm to the pre-temporal GCA",
          "stage; the Transformer neither creates nor repairs it."]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    try:
        plot(per, depth_order)
    except Exception as e:
        print(f"[warn] plot failed: {e}")
    print(f"\n[saved] {OUT_MD}")
    return 0


def plot(per, depth_order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(depth_order))
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for tag, color in [("full", "#c0392b"), ("no_geom", "#2c6fbb")]:
        means = [np.mean(per[tag][d]) if per[tag][d] else np.nan for d in depth_order]
        errs = [np.std(per[tag][d], ddof=1) if len(per[tag][d]) > 1 else 0 for d in depth_order]
        ax.errorbar(x, means, yerr=errs, marker="o", capsize=3, label=tag, color=color, lw=1.6)
    ax.set_xticks(x); ax.set_xticklabels(["post-conv", "L1", "L2", "L3", "L4"], fontsize=8)
    ax.set_ylabel("probe angular MAE (deg)"); ax.set_xlabel("representation depth")
    ax.set_title("MIC+Transformer: where the direction info lives", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {FIG}")


if __name__ == "__main__":
    raise SystemExit(main())
