"""Path C / Tier III: linear probing of intermediate representations.

Question
--------
Does the geometry-bias prior in GCA full (task 110) cause the conv stack
to encode *less* generalizable directional information than plain channel
attention (task 111) or no attention (task 112)?

Method
------
For each of the 15 ckpts (110/111/112 x seeds 0..4):

  1. Hook the model to capture features just after the conv stack
     (input to the GRU): shape (B, C_red, T_label, F_red).
  2. Forward over STARSS23 dev-test, gather features.
  3. Build (X, y) dataset:
     - X: per-frame feature vector pooled over F_red (concat mean+max).
     - y: (sin az, cos az, sin el, cos el) for frames whose ground truth
          has exactly ONE active source. Frames with 0 or >=2 sources are
          dropped (clean probing).
  4. Train a linear ridge regressor predicting y from X with 5-fold CV
     across files; report mean angular MAE on validation in degrees.

Outputs
-------
    D:\\ssl-research\\paper\\path_c_probe.json
    D:\\ssl-research\\paper\\path_c_probe.md
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")
import parameters
import seldnet_model

DCASE_REPO = Path(r"D:\ssl-research\dcase2024_baseline")
MODEL_DIR  = DCASE_REPO / "models_audio"
FEAT_BASE  = Path(r"D:\ssl-research\DCASE2024_SELD_dataset\seld_feat_label")
FEAT_DIR_MIC = FEAT_BASE / "mic_dev_norm"
FEAT_DIR_FOA = FEAT_BASE / "foa_dev_norm"
META_DIR   = Path(r"D:\ssl-research\DCASE2024_SELD_dataset\metadata_dev")
OUT_PATH   = Path(r"D:\ssl-research\paper")
OUT_PATH.mkdir(parents=True, exist_ok=True)

CELLS = [
    # MIC + CRNN
    {"task": "110", "name": "110_gca_full",        "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "111", "name": "111_gca_nogeom",      "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "112", "name": "112_no_gca",          "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "113", "name": "113_vanilla_se",      "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    # FOA + CRNN
    {"task": "100", "name": "100_foa_no_gca",      "modality": "foa", "job_pattern": "repro_seed{seed}",  "seeds": [0, 1, 2, 3, 4]},
    {"task": "130", "name": "130_foa_gca_full",    "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "131", "name": "131_foa_gca_nogeom",  "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    # MIC + Xfm
    {"task": "140", "name": "140_xfm_no_gca",      "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "141", "name": "141_xfm_gca_full",    "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "142", "name": "142_xfm_gca_nogeom",  "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    # FOA + Xfm (Tier VIII)
    {"task": "150", "name": "150_xfm_foa_no_gca",     "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "151", "name": "151_xfm_foa_gca_full",   "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "152", "name": "152_xfm_foa_gca_nogeom", "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    # MIC + Conformer (journal Tranche 2)
    {"task": "160", "name": "160_conf_no_gca",     "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "161", "name": "161_conf_gca_full",   "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "162", "name": "162_conf_gca_nogeom", "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    # FOA + Conformer (journal Tranche 2)
    {"task": "170", "name": "170_conf_foa_no_gca",     "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "171", "name": "171_conf_foa_gca_full",   "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "172", "name": "172_conf_foa_gca_nogeom", "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
]
SEEDS = [0, 1, 2, 3, 4]
LABEL_HOP_S = 0.1  # 10 Hz label rate


# --------------------------------------------------------------------- helpers


def feat_dir_for(modality: str) -> Path:
    return FEAT_DIR_MIC if modality == "mic" else FEAT_DIR_FOA


def list_dev_test_files(modality: str = "mic") -> list[str]:
    """Return list of dev-test stems (fold4_*) which have BOTH a feature
    .npy and a metadata CSV.
    """
    feat_dir = feat_dir_for(modality)
    stems = []
    for sub in ("dev-test-sony", "dev-test-tau"):
        meta_dir = META_DIR / sub
        for csvfile in sorted(meta_dir.glob("*.csv")):
            stem = csvfile.stem
            if (feat_dir / f"{stem}.npy").is_file():
                stems.append(stem)
    return stems


def load_az_el_targets(stem: str, n_frames: int):
    """Read STARSS23 metadata CSV, return per-frame (az, el) for
    frames with exactly 1 active source. Mask: 1 if frame has unique
    source, 0 otherwise. Returns (az_arr, el_arr, mask)."""
    az_arr = np.zeros(n_frames, dtype=np.float32)
    el_arr = np.zeros(n_frames, dtype=np.float32)
    mask   = np.zeros(n_frames, dtype=bool)
    counts: dict[int, list[tuple[float, float]]] = {}
    sub_dirs = ("dev-test-sony", "dev-test-tau")
    for sub in sub_dirs:
        cand = META_DIR / sub / f"{stem}.csv"
        if cand.is_file():
            csvpath = cand
            break
    else:
        return az_arr, el_arr, mask
    with open(csvpath, "r") as fh:
        for row in csv.reader(fh):
            try:
                fr = int(row[0]); az = float(row[3]); el = float(row[4])
            except (ValueError, IndexError):
                continue
            counts.setdefault(fr, []).append((az, el))
    for fr, items in counts.items():
        if 0 <= fr < n_frames and len(items) == 1:
            az_arr[fr] = items[0][0]
            el_arr[fr] = items[0][1]
            mask[fr] = True
    return az_arr, el_arr, mask


# --------------------------------------------------------------------- model io


def build_and_load(task_id: str, seed: int,
                   modality: str = "mic",
                   job_pattern: str = "ablate_seed{seed}") -> torch.nn.Module:
    p = parameters.get_params(task_id).copy()
    feat_seq_len = p["feature_sequence_length"]
    label_seq_len = p["label_sequence_length"]
    n_classes = p["unique_classes"]
    nb_mel = p["nb_mel_bins"]
    in_ch = 4 + (6 if modality == "mic" else 3)
    in_shape  = (1, in_ch, feat_seq_len, nb_mel)
    out_shape = (1, label_seq_len, n_classes * 3 * 4)
    model = seldnet_model.SeldModel(in_shape, out_shape, p)
    feat_tag = "mic_gcc" if modality == "mic" else "foa"
    job = job_pattern.format(seed=seed)
    model_path = MODEL_DIR / f"{task_id}_{job}_dev_split0_multiaccdoa_{feat_tag}_model.h5"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()
    return model


def extract_post_conv_features(model: torch.nn.Module, feat_npy: Path, p: dict, device,
                               modality: str = "mic") -> np.ndarray:
    """Run forward, return (T_label, D) post-conv features pooled over F."""
    feat_seq_len = p["feature_sequence_length"]
    label_seq_len = p["label_sequence_length"]
    nb_mel = p["nb_mel_bins"]
    in_ch = 4 + (6 if modality == "mic" else 3)

    captured: list[torch.Tensor] = []
    # conv_block_list is a ModuleList (no own forward). Hook its LAST submodule
    # which is the final Dropout2d on (B, C, T, F) right before the GRU.
    target_layer = model.conv_block_list[-1]

    def hook(_mod, _inp, out):
        captured.append(out.detach().cpu())
    handle = target_layer.register_forward_hook(hook)

    feat = np.load(feat_npy)
    T_total = feat.shape[0]
    feat = feat.reshape(T_total, in_ch, nb_mel)
    n_seqs = (T_total + feat_seq_len - 1) // feat_seq_len
    pad_T = n_seqs * feat_seq_len - T_total
    if pad_T:
        feat = np.concatenate([feat, np.zeros((pad_T, in_ch, nb_mel), dtype=feat.dtype)], axis=0)
    feat = feat.reshape(n_seqs, feat_seq_len, in_ch, nb_mel).transpose(0, 2, 1, 3)
    xb = torch.from_numpy(feat).float().to(device)

    try:
        with torch.no_grad():
            _ = model(xb)
    finally:
        handle.remove()

    # captured may contain several tensors if conv_block_list is a Sequential
    # with multiple submodule outputs; simplest: take the LAST recorded output
    # and assume its shape is (n_seqs, C, T_seq, F_red) or (n_seqs, C, T_label, F_red).
    out = captured[-1]
    # Be flexible about shape ordering
    if out.ndim != 4:
        raise RuntimeError(f"Unexpected post-conv shape {tuple(out.shape)}; check hook target.")
    n_seqs, C, dim2, dim3 = out.shape
    # In DCASE 2024 SELDnet, after conv_block_list output is (B, C, T_label, F_red)
    # because t_pool divides T_seq down to T_label. Verify by checking divisibility:
    if dim2 == label_seq_len:
        T_dim, F_dim = 2, 3
    elif dim3 == label_seq_len:
        T_dim, F_dim = 3, 2
    else:
        # fall back: assume axis2 is time (most common after conv stack)
        T_dim, F_dim = 2, 3

    if T_dim == 2:
        # (n_seqs, C, T_label, F_red)
        out = out.permute(0, 2, 1, 3)  # (n_seqs, T_label, C, F_red)
    else:
        out = out.permute(0, 3, 1, 2)
    n_seqs2, T_label, C2, F_red = out.shape
    pooled_mean = out.mean(dim=-1)  # (n_seqs, T_label, C)
    pooled_max  = out.amax(dim=-1)
    pooled = torch.cat([pooled_mean, pooled_max], dim=-1)  # (n_seqs, T_label, 2*C)
    pooled = pooled.reshape(n_seqs2 * T_label, -1).numpy()
    # trim padding (label-resolution)
    pad_T_label = (pad_T + 4) // 5
    if pad_T_label:
        pooled = pooled[: pooled.shape[0] - pad_T_label]
    return pooled


# --------------------------------------------------------------------- probing


def angular_error_deg(pred_sincos: np.ndarray, gt_sincos: np.ndarray, n_signals: int) -> np.ndarray:
    """For each row, treat n_signals*2 components as (sin, cos) pairs;
    compute the angular error per pair in degrees and return the mean
    across pairs.
    """
    errs_per_pair = []
    for i in range(n_signals):
        ps = pred_sincos[:, 2 * i]; pc = pred_sincos[:, 2 * i + 1]
        gs = gt_sincos[:, 2 * i];   gc = gt_sincos[:, 2 * i + 1]
        # normalize predicted vectors so atan2 makes sense
        norm_p = np.sqrt(ps ** 2 + pc ** 2 + 1e-9)
        ps2, pc2 = ps / norm_p, pc / norm_p
        ang_p = np.arctan2(ps2, pc2)
        ang_g = np.arctan2(gs, gc)
        diff = np.angle(np.exp(1j * (ang_p - ang_g)))
        errs_per_pair.append(np.rad2deg(np.abs(diff)))
    return np.mean(np.stack(errs_per_pair), axis=0)


def fit_probe_kfold(X: np.ndarray, y: np.ndarray, file_index: np.ndarray, n_signals: int, k: int = 5) -> dict:
    """5-fold CV, splits by FILE (not random row) so file-level signal
    leakage doesn't inflate scores.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    unique_files = np.unique(file_index)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_files)
    folds = np.array_split(unique_files, k)

    fold_maes = []
    for fold_i, val_files in enumerate(folds):
        val_mask = np.isin(file_index, val_files)
        Xt, yt = X[~val_mask], y[~val_mask]
        Xv, yv = X[val_mask],  y[val_mask]
        if len(Xt) == 0 or len(Xv) == 0:
            continue
        sc = StandardScaler().fit(Xt)
        reg = Ridge(alpha=1.0).fit(sc.transform(Xt), yt)
        pred = reg.predict(sc.transform(Xv))
        errs = angular_error_deg(pred, yv, n_signals)
        fold_maes.append(float(errs.mean()))
    if not fold_maes:
        return {"mae_per_fold": [], "mae_mean": float("nan"), "mae_std": 0.0,
                "note": "no usable folds (all val sets empty)"}
    return {
        "mae_per_fold": fold_maes,
        "mae_mean":     float(np.mean(fold_maes)),
        "mae_std":      float(np.std(fold_maes, ddof=1)) if len(fold_maes) > 1 else 0.0,
    }


def probe_one_ckpt(task_id: str, seed: int, file_stems: list[str], device,
                   modality: str = "mic",
                   job_pattern: str = "ablate_seed{seed}") -> dict | None:
    p = parameters.get_params(task_id).copy()
    try:
        model = build_and_load(task_id, seed, modality=modality, job_pattern=job_pattern).to(device)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return None

    feat_dir = feat_dir_for(modality)
    Xs, ys, fis = [], [], []
    for fi, stem in enumerate(file_stems):
        npy = feat_dir / f"{stem}.npy"
        feat_post = extract_post_conv_features(model, npy, p, device, modality=modality)  # (T_label_total, D)
        T_label = feat_post.shape[0]
        az, el, mask = load_az_el_targets(stem, T_label)
        if mask.sum() == 0:
            continue
        Xf = feat_post[mask]
        # build target: (sin az, cos az, sin el, cos el) in radians
        az_r = np.deg2rad(az[mask]); el_r = np.deg2rad(el[mask])
        yf = np.stack([np.sin(az_r), np.cos(az_r), np.sin(el_r), np.cos(el_r)], axis=1)
        Xs.append(Xf); ys.append(yf); fis.append(np.full(Xf.shape[0], fi, dtype=np.int32))
    del model; torch.cuda.empty_cache()
    if not Xs:
        return None
    X = np.concatenate(Xs, axis=0); y = np.concatenate(ys, axis=0); fi = np.concatenate(fis)
    print(f"    probe X: {X.shape}, y: {y.shape}, files used: {len(np.unique(fi))}")
    return fit_probe_kfold(X, y, fi, n_signals=2, k=5)


# --------------------------------------------------------------------- aggregate


def cohens_dz(deltas):
    a = np.asarray(deltas, dtype=np.float64)
    s = a.std(ddof=1)
    return float("nan") if s == 0 else float(a.mean() / s)


def aggregate(per_ckpt: dict) -> dict:
    cells: dict[str, dict[int, float]] = {c["name"]: {} for c in CELLS}
    for cell in CELLS:
        seeds_use = cell.get("seeds", SEEDS)
        for s in seeds_use:
            key = f"{cell['task']}_seed{s}"
            if key in per_ckpt:
                cells[cell["name"]][s] = per_ckpt[key]["mae_mean"]
    summary = {}
    for cname, by_seed in cells.items():
        vals = list(by_seed.values())
        summary[cname] = {
            "n_seeds":    len(vals),
            "mae_per_seed": by_seed,
            "mae_mean":   float(np.mean(vals)) if vals else None,
            "mae_std":    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    contrasts = {}
    pair_specs = [
        # MIC + CRNN
        ("110_gca_full",   "112_no_gca",         "MIC+CRNN: GCA full vs no-GCA (probing)"),
        ("110_gca_full",   "111_gca_nogeom",     "MIC+CRNN: GCA full vs no_geom (geometry contribution, probing)"),
        ("111_gca_nogeom", "112_no_gca",         "MIC+CRNN: GCA no_geom vs no-GCA (probing)"),
        ("113_vanilla_se", "112_no_gca",         "MIC+CRNN: Vanilla SE vs no-GCA (probing)"),
        # FOA + CRNN
        ("130_foa_gca_full",   "131_foa_gca_nogeom", "FOA+CRNN: GCA full vs no_geom (probing)"),
        ("130_foa_gca_full",   "100_foa_no_gca",     "FOA+CRNN: GCA full vs no GCA (probing)"),
        # MIC + Xfm
        ("141_xfm_gca_full",   "142_xfm_gca_nogeom", "MIC+Xfm: GCA full vs no_geom (probing)"),
        ("141_xfm_gca_full",   "140_xfm_no_gca",     "MIC+Xfm: GCA full vs no GCA (probing)"),
        # FOA + Xfm
        ("151_xfm_foa_gca_full", "152_xfm_foa_gca_nogeom", "FOA+Xfm: GCA full vs no_geom (probing)"),
        ("151_xfm_foa_gca_full", "150_xfm_foa_no_gca",     "FOA+Xfm: GCA full vs no GCA (probing)"),
        # MIC + Conformer (Tranche 2)
        ("161_conf_gca_full", "162_conf_gca_nogeom", "MIC+Conf: GCA full vs no_geom (probing)"),
        ("161_conf_gca_full", "160_conf_no_gca",     "MIC+Conf: GCA full vs no GCA (probing)"),
        # FOA + Conformer (Tranche 2)
        ("171_conf_foa_gca_full", "172_conf_foa_gca_nogeom", "FOA+Conf: GCA full vs no_geom (probing)"),
        ("171_conf_foa_gca_full", "170_conf_foa_no_gca",     "FOA+Conf: GCA full vs no GCA (probing)"),
    ]
    for a, b, descr in pair_specs:
        shared = sorted(set(cells[a].keys()) & set(cells[b].keys()))
        if len(shared) < 2:
            contrasts[f"{a}__vs__{b}"] = {"n": len(shared), "note": "insufficient pairs"}; continue
        a_vals = np.array([cells[a][s] for s in shared])
        b_vals = np.array([cells[b][s] for s in shared])
        deltas = (a_vals - b_vals).tolist()
        t_stat, p_t = stats.ttest_rel(a_vals, b_vals)
        contrasts[f"{a}__vs__{b}"] = {
            "description":     descr,
            "n":               len(shared),
            "a_per_seed":      a_vals.tolist(),
            "b_per_seed":      b_vals.tolist(),
            "delta_per_seed":  deltas,
            "delta_mean":      float(np.mean(deltas)),
            "delta_std":       float(np.std(deltas, ddof=1)),
            "ttest_rel":       {"t": float(t_stat), "p": float(p_t)},
            "cohens_dz":       cohens_dz(deltas),
        }
    return {"per_cell": summary, "contrasts": contrasts}


def write_md(payload: dict, path: Path) -> None:
    L = ["# Path C linear probing -- post-conv (B, C, T_label, F_red) -> (sin az, cos az, sin el, cos el)",
         "",
         "Probe: Ridge(alpha=1) on standardized pooled features (mean+max over F_red).",
         "5-fold CV, folds split by FILE. Lower angular MAE = representation encodes location more linearly.",
         "",
         "## Per-cell"]
    L.append("| cell | n | MAE mean (deg) | MAE std (deg) |")
    L.append("| ---- | - | -------------- | ------------- |")
    for cname, c in payload["per_cell"].items():
        m = c["mae_mean"]
        if m is None:
            L.append(f"| {cname} | {c['n_seeds']} | n/a | n/a |"); continue
        L.append(f"| {cname} | {c['n_seeds']} | {m:.2f} | {c['mae_std']:.2f} |")
    L += ["", "## Paired contrasts"]
    for cname, c in payload["contrasts"].items():
        L.append(f"### {cname}")
        if c.get("n", 0) < 2:
            L.append("- insufficient pairs"); L.append(""); continue
        L.append(f"- _{c['description']}_")
        L.append(f"- n={c['n']}, mean delta = {c['delta_mean']:+.2f} +/- {c['delta_std']:.2f} deg")
        L.append(f"- t = {c['ttest_rel']['t']:+.2f} (p = {c['ttest_rel']['p']:.4f})")
        L.append(f"- Cohen's d_z = {c['cohens_dz']:+.2f}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------- main


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None, help="Restrict to these task IDs.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=None,
                    help="Use at most N dev-test files (debug only)")
    args = ap.parse_args()

    cells_filter = args.cells
    seeds_filter = args.seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-compute file lists per modality (same stems work for both;
    # only changes which feature .npy is loaded).
    file_stems_mic = list_dev_test_files("mic")
    file_stems_foa = list_dev_test_files("foa")
    if args.max_files:
        file_stems_mic = file_stems_mic[: args.max_files]
        file_stems_foa = file_stems_foa[: args.max_files]
    print(f"using {len(file_stems_mic)} MIC dev-test files, {len(file_stems_foa)} FOA dev-test files")

    per_ckpt: dict[str, dict] = {}
    for cell in CELLS:
        task_id = cell["task"]
        if cells_filter is not None and task_id not in cells_filter:
            continue
        modality   = cell.get("modality", "mic")
        jobpat     = cell.get("job_pattern", "ablate_seed{seed}")
        cell_seeds = seeds_filter if seeds_filter is not None else cell.get("seeds", SEEDS)
        stems = file_stems_mic if modality == "mic" else file_stems_foa
        for s in cell_seeds:
            print(f"\n=== {task_id} seed {s} (modality={modality}) ===")
            r = probe_one_ckpt(task_id, s, stems, device, modality=modality, job_pattern=jobpat)
            if r is None:
                continue
            per_ckpt[f"{task_id}_seed{s}"] = r
            print(f"    MAE = {r['mae_mean']:.2f} +/- {r['mae_std']:.2f} deg")

    payload = aggregate(per_ckpt)
    payload["per_ckpt"] = per_ckpt
    out_json = OUT_PATH / "path_c_probe.json"
    out_md   = OUT_PATH / "path_c_probe.md"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_md(payload, out_md)
    print(f"\n[saved] {out_json}\n[saved] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
