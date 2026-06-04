"""Path C cross-dataset evaluation: STARSS22 dev-test on DCASE 2024 ckpts.

Pipeline:
  1. Augment STARSS22 metadata CSVs (5 cols -> 6 cols by appending dummy
     distance, so ComputeSELDResults' DCASE 2024 6-col loader works).
     Since lad_dist_thresh = lad_reldist_thresh = inf during cross-eval,
     the distance dimension is effectively ignored for matching; we only
     trust F1/AE/SELD numbers. Dist_err and RDE_CD are NOT reported.

  2. Extract MIC GCC features for STARSS22 dev-test wavs into a
     dedicated feat_label_dir (different from the STARSS23 one to keep
     filenames disjoint), normalized using the *STARSS23 train scaler*
     (copy `mic_wts` over). This gives true zero-shot transfer:
     train data on STARSS23, eval features whitened with STARSS23 stats.

  3. Iterate over all 15 ckpts (110/111/112 x seeds 0..4):
     - Build SeldModel with the matching task params (so use_gca/etc match)
     - Load ckpt
     - For each STARSS22 file, load its normalized feature, slice into
       250-frame seq batches, run inference, decode Multi-ACCDDOA, dump
       per-file DCASE-format CSV
     - Run ComputeSELDResults pointed at STARSS22 metadata, save metrics

  4. Aggregate per-cell mean/std + paired contrasts (110 vs 112,
     110 vs 111, 111 vs 112) and write JSON + Markdown summary.

Outputs:
    D:\\ssl-research\\paper\\path_c_cross_starss22.json
    D:\\ssl-research\\paper\\path_c_cross_starss22.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
import cls_data_generator
import cls_feature_class
import parameters
import seldnet_model
from cls_compute_seld_results import ComputeSELDResults

# Paths
STARSS22_DCASE_ROOT = Path(r"D:\ssl-research\STARSS22_DCASE")
STARSS22_AUGMENTED_META = STARSS22_DCASE_ROOT / "metadata_dev_dist_padded"
STARSS22_FEAT_DIR = STARSS22_DCASE_ROOT / "seld_feat_label"

STARSS23_FEAT_DIR = Path(r"D:\ssl-research\DCASE2024_SELD_dataset\seld_feat_label")
STARSS23_MIC_WTS = STARSS23_FEAT_DIR / "mic_wts"
STARSS23_FOA_WTS = STARSS23_FEAT_DIR / "foa_wts"

DCASE_REPO = Path(r"D:\ssl-research\dcase2024_baseline")
MODEL_DIR = DCASE_REPO / "models_audio"
RESULTS_OUT = Path(r"D:\ssl-research\paper")

# Cells to evaluate. Each row may override (modality, job_pattern, seeds).
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
# Default seeds when ad-hoc cell entries lack a `seeds` field.
SEEDS = [0, 1, 2, 3, 4]

# Cross-eval threshold settings: distance-blind matching
CROSS_PARAMS_OVERRIDES = {
    "lad_dist_thresh": float("inf"),
    "lad_reldist_thresh": float("inf"),
}

# ---------------------------------------------------------------------- helpers


def augment_metadata_with_dummy_distance() -> None:
    """STARSS22 CSVs: 5 cols (frame, cls, src, az, el) -> 6 cols by
    appending dummy distance=100 (any positive float; lad_dist_thresh=inf
    means distance never disqualifies a match).
    """
    if STARSS22_AUGMENTED_META.exists():
        n = sum(1 for _ in STARSS22_AUGMENTED_META.glob("**/*.csv"))
        print(f"[meta] already padded ({n} files), skipping")
        return

    print("[meta] augmenting STARSS22 metadata with dummy distance column ...")
    src_root = STARSS22_DCASE_ROOT / "metadata_dev"
    for split_dir in src_root.iterdir():
        if not split_dir.is_dir():
            continue
        out_dir = STARSS22_AUGMENTED_META / split_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for csv_in in split_dir.glob("*.csv"):
            with open(csv_in, "r", newline="") as fin, \
                 open(out_dir / csv_in.name, "w", newline="") as fout:
                w = csv.writer(fout)
                for row in csv.reader(fin):
                    if len(row) == 5:
                        row.append("100")  # dummy distance in cm
                    w.writerow(row)
    n = sum(1 for _ in STARSS22_AUGMENTED_META.glob("**/*.csv"))
    print(f"[meta] padded {n} CSVs")


def extract_starss22_features(task_id: str = "110", modality: str = "mic") -> bool:
    """Run unnormalized feature extraction + reuse STARSS23's scaler for
    zero-shot transfer. `modality` selects MIC (logmel + GCC) or FOA
    (logmel + intensity vector). Returns True if features are usable
    afterwards, False if audio is missing and we should skip cells of
    this modality."""
    p = parameters.get_params(task_id).copy()
    p["dataset_dir"] = str(STARSS22_DCASE_ROOT)
    p["feat_label_dir"] = str(STARSS22_FEAT_DIR)
    p["dataset"] = modality
    p["use_salsalite"] = False

    feat_cls = cls_feature_class.FeatureClass(p, is_eval=False)

    norm_dir = Path(feat_cls.get_normalized_feat_dir())
    norm_dir.mkdir(parents=True, exist_ok=True)

    n_existing = len(list(norm_dir.glob("*.npy")))
    if n_existing >= 50:
        print(f"[feat] STARSS22 {modality.upper()} normalized features already exist ({n_existing}), skipping")
        return True

    audio_dir = STARSS22_DCASE_ROOT / (f"{modality}_dev")
    has_audio = audio_dir.is_dir() and any(audio_dir.glob("**/*.wav"))
    if not has_audio:
        print(f"[feat] STARSS22 {modality.upper()} audio not present at {audio_dir}; skipping {modality.upper()} cells.")
        return False

    print(f"[feat] extracting raw STARSS22 {modality.upper()} features ...")
    feat_cls.extract_all_feature()

    # Copy STARSS23 normalization scaler before preprocess_features so
    # is_eval=True uses STARSS23 stats.
    wts_src = STARSS23_MIC_WTS if modality == "mic" else STARSS23_FOA_WTS
    wts_target = Path(feat_cls.get_normalized_wts_file())
    wts_target.parent.mkdir(parents=True, exist_ok=True)
    if not wts_target.exists():
        shutil.copy2(wts_src, wts_target)
        print(f"[feat] copied STARSS23 {wts_src.name} -> {wts_target}")

    feat_cls._is_eval = True
    feat_cls.preprocess_features()
    print(f"[feat] STARSS22 {modality.upper()} normalized features ready")
    return True


# ---------------------------------------------------------------------- inference


def _decode_multi_accdoa_to_dict(out: np.ndarray, params: dict) -> dict:
    """Reproduce train_seldnet.eval_epoch's multi-ACCDDOA -> output_dict
    decode for one file's worth of model output.

    Args:
        out: (n_seqs, label_seq_len, n_classes*3*4) numpy array.
        params: hyperparameter dict (uses 'unique_classes' and 'thresh_unify').
    Returns:
        output_dict {frame: [[cls, x, y, z, dist], ...]}
    """
    from cls_compute_seld_results import reshape_3Dto2D
    from train_seldnet import (
        get_multi_accdoa_labels, determine_similar_location,
    )

    n_classes = params["unique_classes"]
    thresh_unify = params["thresh_unify"]

    sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, \
        sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(out, n_classes)
    sed_pred0  = reshape_3Dto2D(sed_pred0);  doa_pred0  = reshape_3Dto2D(doa_pred0);  dist_pred0  = reshape_3Dto2D(dist_pred0)
    sed_pred1  = reshape_3Dto2D(sed_pred1);  doa_pred1  = reshape_3Dto2D(doa_pred1);  dist_pred1  = reshape_3Dto2D(dist_pred1)
    sed_pred2  = reshape_3Dto2D(sed_pred2);  doa_pred2  = reshape_3Dto2D(doa_pred2);  dist_pred2  = reshape_3Dto2D(dist_pred2)

    output_dict: dict[int, list[list]] = {}
    nC = n_classes
    for frame_cnt in range(sed_pred0.shape[0]):
        for class_cnt in range(sed_pred0.shape[1]):
            f01 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, thresh_unify, nC)
            f12 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, thresh_unify, nC)
            f20 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, thresh_unify, nC)
            total = f01 + f12 + f20
            if total == 0:
                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                    output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt + nC], doa_pred0[frame_cnt][class_cnt + 2 * nC], dist_pred0[frame_cnt][class_cnt]])
                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                    output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt + nC], doa_pred1[frame_cnt][class_cnt + 2 * nC], dist_pred1[frame_cnt][class_cnt]])
                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                    output_dict.setdefault(frame_cnt, []).append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt + nC], doa_pred2[frame_cnt][class_cnt + 2 * nC], dist_pred2[frame_cnt][class_cnt]])
            elif total == 1:
                output_dict.setdefault(frame_cnt, [])
                if f01:
                    if sed_pred2[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt + nC], doa_pred2[frame_cnt][class_cnt + 2 * nC], dist_pred2[frame_cnt][class_cnt]])
                    doa_pred_fc  = (doa_pred0[frame_cnt]  + doa_pred1[frame_cnt])  / 2
                    dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nC], doa_pred_fc[class_cnt + 2 * nC], dist_pred_fc[class_cnt]])
                elif f12:
                    if sed_pred0[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt + nC], doa_pred0[frame_cnt][class_cnt + 2 * nC], dist_pred0[frame_cnt][class_cnt]])
                    doa_pred_fc  = (doa_pred1[frame_cnt]  + doa_pred2[frame_cnt])  / 2
                    dist_pred_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nC], doa_pred_fc[class_cnt + 2 * nC], dist_pred_fc[class_cnt]])
                elif f20:
                    if sed_pred1[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt + nC], doa_pred1[frame_cnt][class_cnt + 2 * nC], dist_pred1[frame_cnt][class_cnt]])
                    doa_pred_fc  = (doa_pred2[frame_cnt]  + doa_pred0[frame_cnt])  / 2
                    dist_pred_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nC], doa_pred_fc[class_cnt + 2 * nC], dist_pred_fc[class_cnt]])
            elif total >= 2:
                output_dict.setdefault(frame_cnt, [])
                doa_pred_fc  = (doa_pred0[frame_cnt]  + doa_pred1[frame_cnt]  + doa_pred2[frame_cnt])  / 3
                dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nC], doa_pred_fc[class_cnt + 2 * nC], dist_pred_fc[class_cnt]])
    return output_dict


def evaluate_one_ckpt(task_id: str, job_id: str, dump_root: Path,
                      modality: str = "mic") -> dict | None:
    """Inference + DCASE eval on STARSS22 dev-test for one ckpt."""
    p = parameters.get_params(task_id).copy()
    p.update(CROSS_PARAMS_OVERRIDES)

    feat_tag = "mic_gcc" if modality == "mic" else "foa"
    out_tag = "multiaccdoa"
    unique_name = f"{task_id}_{job_id}_dev_split0_{out_tag}_{feat_tag}"
    model_path = MODEL_DIR / f"{unique_name}_model.h5"
    if not model_path.is_file():
        print(f"  [skip] missing ckpt {model_path}")
        return None

    feat_subdir = "mic_dev_norm" if modality == "mic" else "foa_dev_norm"
    feat_dir = STARSS22_FEAT_DIR / feat_subdir
    feat_files = sorted(feat_dir.glob("*.npy"))
    # Cross-eval is on the DCASE test split (fold4 / dev-test). Some feature
    # extractions also produce dev-train (fold3) features for which no reference
    # label exists -> KeyError in scoring. Keep only files with a matching ref.
    feat_files = [f for f in feat_files if f.stem.startswith("fold4")]
    if not feat_files:
        print(f"  [skip] no STARSS22 dev-test features at {feat_dir}")
        return None

    n_classes = p["unique_classes"]
    feat_seq_len = p["feature_sequence_length"]
    label_seq_len = p["label_sequence_length"]
    nb_mel = p["nb_mel_bins"]

    # MIC: 4 mel + 6 GCC. FOA: 4 mel + 3 intensity vector.
    in_ch = 4 + (6 if modality == "mic" else 3)
    in_shape = (1, in_ch, feat_seq_len, nb_mel)
    out_shape = (1, label_seq_len, n_classes * 3 * 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = seldnet_model.SeldModel(in_shape, out_shape, p).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()

    # FeatureClass for write_output_format_file (DCASE CSV writer)
    p_writer = p.copy()
    p_writer["dataset_dir"] = str(STARSS22_DCASE_ROOT)
    p_writer["feat_label_dir"] = str(STARSS22_FEAT_DIR)
    p_writer["dataset"] = modality
    feat_writer = cls_feature_class.FeatureClass(p_writer, is_eval=False)

    dump_dir = dump_root / unique_name
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with torch.no_grad():
        for npy in feat_files:
            feat = np.load(npy)
            T_total = feat.shape[0]
            feat = feat.reshape(T_total, in_ch, nb_mel)
            n_seqs = (T_total + feat_seq_len - 1) // feat_seq_len
            pad_T = n_seqs * feat_seq_len - T_total
            if pad_T:
                feat = np.concatenate([feat, np.zeros((pad_T, in_ch, nb_mel), dtype=feat.dtype)], axis=0)
            feat = feat.reshape(n_seqs, feat_seq_len, in_ch, nb_mel).transpose(0, 2, 1, 3)
            xb = torch.from_numpy(feat).float().to(device)
            yb = model(xb)  # (n_seqs, label_seq_len, n_classes*3*4)
            yb_np = yb.detach().cpu().numpy()

            output_dict = _decode_multi_accdoa_to_dict(yb_np, p)
            # trim padding in label resolution: label rate is feat rate / t_pool; in DCASE 2024 mic+ma, t_pool[0]=5
            pad_T_label = (pad_T + 4) // 5
            if pad_T_label:
                T_label_total = n_seqs * label_seq_len
                trim_after = T_label_total - pad_T_label
                output_dict = {k: v for k, v in output_dict.items() if k < trim_after}

            stem = npy.stem
            feat_writer.write_output_format_file(str(dump_dir / f"{stem}.csv"), output_dict)
    print(f"  inference {unique_name} took {time.time() - t0:.1f}s")

    # Compute SELD metrics with STARSS22 ground truth
    p2 = p.copy()
    score_obj = ComputeSELDResults(p2, ref_files_folder=str(STARSS22_AUGMENTED_META))
    use_jackknife = False
    test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise = \
        score_obj.get_SELD_Results(str(dump_dir), is_jackknife=use_jackknife)

    return {
        "ckpt": unique_name,
        "F1":   float(test_F),
        "ER":   float(test_ER),
        "LE":   float(test_LE),
        "LR":   float(test_LR),
        "SELD": float(test_seld_scr),
        # Distance metrics are meaningless on STARSS22 (dummy refs); we
        # record them but do not use them in paired stats.
        "DE_unreliable":  float(test_dist_err),
        "RDE_unreliable": float(test_rel_dist_err),
    }


# ---------------------------------------------------------------------- aggregate


def cohens_dz(deltas):
    a = np.asarray(deltas, dtype=np.float64)
    s = a.std(ddof=1)
    return float("nan") if s == 0 else float(a.mean() / s)


def bootstrap_ci(deltas, n_boot=5000, alpha=0.05, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    a = np.asarray(deltas, dtype=np.float64)
    n = len(a)
    boot = a[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def aggregate(per_ckpt: dict) -> dict:
    cells: dict[str, dict[int, dict | None]] = {c["name"]: {} for c in CELLS}
    for cell in CELLS:
        seeds_use = cell.get("seeds", SEEDS)
        jobpat = cell.get("job_pattern", "ablate_seed{seed}")
        for s in seeds_use:
            key = f"{cell['task']}_{jobpat.format(seed=s)}"
            cells[cell["name"]][s] = per_ckpt.get(key)

    summary = {}
    for cell in CELLS:
        cname = cell["name"]
        seeds_use = cell.get("seeds", SEEDS)
        by_seed = cells[cname]
        avail = {s: v for s, v in by_seed.items() if v is not None}
        out = {"n_seeds": len(avail), "missing_seeds": [s for s in seeds_use if s not in avail]}
        for m in ("F1", "LE", "LR", "SELD"):
            vals = [v[m] for v in avail.values()]
            out[m] = {
                "values": vals,
                "mean":   mean(vals) if vals else None,
                "std":    stdev(vals) if len(vals) > 1 else 0.0,
                "n":      len(vals),
            }
        summary[cname] = out

    contrasts = {}
    pair_specs = [
        # MIC + CRNN
        ("110_gca_full",    "112_no_gca",         "MIC+CRNN: GCA full vs no-GCA (cross)"),
        ("110_gca_full",    "111_gca_nogeom",     "MIC+CRNN: GCA full vs no_geom (geometry contribution, cross)"),
        ("111_gca_nogeom",  "112_no_gca",         "MIC+CRNN: GCA no_geom vs no-GCA (cross)"),
        ("113_vanilla_se",  "112_no_gca",         "MIC+CRNN: Vanilla SE vs no-GCA (cross)"),
        # FOA + CRNN
        ("130_foa_gca_full",   "131_foa_gca_nogeom", "FOA+CRNN: GCA full vs no_geom (geometry contribution, cross)"),
        ("130_foa_gca_full",   "100_foa_no_gca",     "FOA+CRNN: GCA full vs no GCA (cross)"),
        # MIC + Xfm
        ("141_xfm_gca_full",   "142_xfm_gca_nogeom", "MIC+Xfm: GCA full vs no_geom (geometry contribution, cross)"),
        ("141_xfm_gca_full",   "140_xfm_no_gca",     "MIC+Xfm: GCA full vs no GCA (cross)"),
        # FOA + Xfm
        ("151_xfm_foa_gca_full", "152_xfm_foa_gca_nogeom", "FOA+Xfm: GCA full vs no_geom (geometry contribution, cross)"),
        ("151_xfm_foa_gca_full", "150_xfm_foa_no_gca",     "FOA+Xfm: GCA full vs no GCA (cross)"),
        # MIC + Conformer (Tranche 2)
        ("161_conf_gca_full", "162_conf_gca_nogeom", "MIC+Conf: GCA full vs no_geom (geometry contribution, cross)"),
        ("161_conf_gca_full", "160_conf_no_gca",     "MIC+Conf: GCA full vs no GCA (cross)"),
        # FOA + Conformer (Tranche 2)
        ("171_conf_foa_gca_full", "172_conf_foa_gca_nogeom", "FOA+Conf: GCA full vs no_geom (geometry contribution, cross)"),
        ("171_conf_foa_gca_full", "170_conf_foa_no_gca",     "FOA+Conf: GCA full vs no GCA (cross)"),
    ]
    for cell_a, cell_b, descr in pair_specs:
        if cell_a not in cells or cell_b not in cells: continue
        per_metric = {}
        for m in ("F1", "LE", "LR", "SELD"):
            seeds_a = set(cells[cell_a].keys())
            seeds_b = set(cells[cell_b].keys())
            shared = [s for s in sorted(seeds_a & seeds_b)
                      if cells[cell_a].get(s) and cells[cell_b].get(s)]
            a_vals = [cells[cell_a][s][m] for s in shared]
            b_vals = [cells[cell_b][s][m] for s in shared]
            n = len(a_vals)
            if n < 2:
                per_metric[m] = {"n": n, "note": "insufficient pairs"}
                continue
            a_arr, b_arr = np.array(a_vals), np.array(b_vals)
            deltas = (a_arr - b_arr).tolist()
            t_stat, p_t = stats.ttest_rel(a_arr, b_arr)
            try:
                w_stat, p_w = stats.wilcoxon(a_arr, b_arr, zero_method="pratt")
            except ValueError:
                w_stat, p_w = float("nan"), float("nan")
            per_metric[m] = {
                "n":               n,
                "a_per_seed":      a_arr.tolist(),
                "b_per_seed":      b_arr.tolist(),
                "delta_per_seed":  deltas,
                "delta_mean":      float(np.mean(deltas)),
                "delta_std":       float(np.std(deltas, ddof=1)) if n > 1 else 0.0,
                "ttest_rel":       {"t": float(t_stat), "p": float(p_t)},
                "wilcoxon":        {"W": float(w_stat), "p": float(p_w)},
                "cohens_dz":       cohens_dz(deltas),
                "bootstrap_95ci":  bootstrap_ci(deltas),
            }
        contrasts[f"{cell_a}__vs__{cell_b}"] = {"description": descr, "metrics": per_metric}
    return {"per_cell": summary, "contrasts": contrasts}


def write_md_summary(payload: dict, path: Path) -> None:
    lines = ["# Path C cross-dataset eval -- STARSS22 dev-test (zero-shot)",
             "",
             "Models trained on STARSS23 dev-train, eval on STARSS22 dev-test (54 clips).",
             "Distance dimension not available in STARSS22; lad_dist_thresh=inf so",
             "DOAE_CD/F1 use az/el matching only. Dist_err/RDE not reported.",
             "",
             "## Per-cell mean +/- std (seeds 0..4)",
             "",
             "| Cell                    | n  | F 20deg (%)   | DOAE_CD (deg) | LR_CD          | SELD            |",
             "| ----------------------- | -- | ------------- | ------------- | -------------- | --------------- |"]
    for cname, cell in payload["per_cell"].items():
        n = cell["n_seeds"]
        def fmt(m, scale=1.0, nd=2):
            d = cell[m]
            if d["mean"] is None: return "n/a"
            return f"{scale*d['mean']:.{nd}f} +/- {scale*d['std']:.{nd}f}"
        lines.append(f"| {cname:<23} | {n:<2} | {fmt('F1', 100):<13} | {fmt('LE'):<13} | {fmt('LR'):<14} | {fmt('SELD',1.0,3):<15} |")
    lines.append("")
    lines.append("## Paired contrasts (seeds matched, cross-dataset)")
    lines.append("")
    for cname, c in payload["contrasts"].items():
        lines += [f"### {cname}", f"_{c['description']}_", "",
                  "| Metric | n | mean delta | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |",
                  "| ------ | - | ---------- | ----- | --------------- | --- | ---------------- |"]
        for m in ("F1", "LE", "LR", "SELD"):
            r = c["metrics"][m]
            n = r.get("n", 0)
            if n < 2:
                lines.append(f"| {m} | {n} | n/a | n/a | n/a | n/a | n/a |"); continue
            ci = r["bootstrap_95ci"]
            lines.append(
                f"| {m:<6} | {n} | {r['delta_mean']:+.3f} +/- {r['delta_std']:.3f} | "
                f"t={r['ttest_rel']['t']:+.2f} (p={r['ttest_rel']['p']:.3f}) | "
                f"W={r['wilcoxon']['W']:.1f} (p={r['wilcoxon']['p']:.3f}) | "
                f"{r['cohens_dz']:+.2f} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-only", action="store_true",
                    help="Only do meta padding + feature extraction, skip inference.")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="Restrict to these task IDs (default: 110 111 112).")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Restrict to these seeds (default: 0..4).")
    args = ap.parse_args()

    augment_metadata_with_dummy_distance()
    # Feature extraction for both modalities (idempotent if already cached).
    # Each call returns False if audio missing -- we still proceed with whichever
    # modality has features; missing-modality cells will be skipped per-iteration.
    have_mic = extract_starss22_features("110", modality="mic")
    have_foa = extract_starss22_features("100", modality="foa")
    if not (have_mic or have_foa):
        print("[error] no STARSS22 features available for either modality; aborting.")
        return 1

    if args.prep_only:
        print("\n[prep-only] done. Re-run without --prep-only to do inference.")
        return 0

    cells_filter = args.cells  # None = all
    seeds_filter = args.seeds  # None = per-cell defaults

    dump_root = STARSS22_DCASE_ROOT / "predictions_starss22"
    dump_root.mkdir(parents=True, exist_ok=True)

    per_ckpt: dict[str, dict] = {}
    for cell in CELLS:
        task_id = cell["task"]
        if cells_filter is not None and task_id not in cells_filter:
            continue
        modality   = cell.get("modality", "mic")
        if modality == "mic" and not have_mic:
            print(f"[skip cell] {task_id}: MIC features unavailable"); continue
        if modality == "foa" and not have_foa:
            print(f"[skip cell] {task_id}: FOA features unavailable"); continue
        jobpat     = cell.get("job_pattern", "ablate_seed{seed}")
        cell_seeds = seeds_filter if seeds_filter is not None else cell.get("seeds", SEEDS)
        for s in cell_seeds:
            job_id = jobpat.format(seed=s)
            print(f"\n=== [{task_id} {job_id} | modality={modality}] ===")
            res = evaluate_one_ckpt(task_id, job_id, dump_root, modality=modality)
            if res is not None:
                per_ckpt[f"{task_id}_{job_id}"] = res
                print(f"  F1={res['F1']*100:.2f}% AE={res['LE']:.2f} LR={res['LR']:.2f} SELD={res['SELD']:.3f}")

    # Merge with any previously-computed per_ckpt so a partial-cell rerun
    # (e.g. --cells 140 141 142) does not clobber other cells' results.
    out_json = RESULTS_OUT / "path_c_cross_starss22.json"
    out_md   = RESULTS_OUT / "path_c_cross_starss22.md"
    if out_json.is_file():
        try:
            prev = json.loads(out_json.read_text(encoding="utf-8")).get("per_ckpt", {})
            merged = dict(prev)
            merged.update(per_ckpt)  # new results win on conflict
            print(f"[merge] previous ckpts={len(prev)}, new={len(per_ckpt)}, merged={len(merged)}")
            per_ckpt = merged
        except Exception as e:
            print(f"[merge] could not read previous results ({e}); writing fresh")

    payload = aggregate(per_ckpt)
    payload["per_ckpt"] = per_ckpt
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_md_summary(payload, out_md)
    print(f"\n[saved] {out_json}\n[saved] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
