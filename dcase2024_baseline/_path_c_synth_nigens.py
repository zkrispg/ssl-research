"""Path C / E (synth stress test): zero-shot evaluation of STARSS23-trained
ckpts on TAU-NIGENS-SSE-2021 (synthetic, real SRIRs).

Validates whether the (modality, architecture, prior) dissociation pattern
survives a heavy domain shift to *synthetic* but acoustically-realistic data.

NIGENS-2021 class taxonomy (12) is partially compatible with STARSS23-24's
13-class taxonomy. We remap the 7 overlapping classes:

  NIGENS 0  alarm           --> STARSS 11 bell
  NIGENS 5  female speech   --> STARSS 0  femaleSpeech
  NIGENS 6  footsteps       --> STARSS 6  footsteps
  NIGENS 7  knocking        --> STARSS 12 knock
  NIGENS 9  male speech     --> STARSS 1  maleSpeech
  NIGENS 10 ringing phone   --> STARSS 3  telephone
  NIGENS 11 piano           --> STARSS 9  instrument

Other NIGENS classes (1, 2, 3, 4, 8) have no STARSS counterpart and their
events are dropped from ground truth before evaluation. The model predicts
13-class STARSS outputs and is matched against the remapped 7-class ground
truth at SELD-eval time.

Pipeline:
  1. Pad NIGENS metadata to STARSS24 6-col format with class remap +
     dummy distance.
  2. Extract MIC + FOA features (logmel + GCC / IV) from NIGENS audio,
     normalize with STARSS23 train scaler (true zero-shot).
  3. Iterate ckpts, run inference on NIGENS dev-test, dump CSVs.
  4. Run ComputeSELDResults against remapped metadata.
  5. Aggregate per-cell, per-cell paired contrasts.

Outputs:
    D:\\ssl-research\\paper\\path_c_synth_nigens.json
    D:\\ssl-research\\paper\\path_c_synth_nigens.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
import cls_data_generator
import cls_feature_class
import parameters
import seldnet_model
from cls_compute_seld_results import ComputeSELDResults

# ---------------------------------------------------------------------- paths

NIGENS_ROOT     = Path(r"D:\ssl-research\TAU_NIGENS_SSE_2021")
NIGENS_RAW_META = NIGENS_ROOT / "metadata_dev"
NIGENS_REMAP_META = NIGENS_ROOT / "metadata_dev_starss_remap"
NIGENS_FEAT_DIR = NIGENS_ROOT / "seld_feat_label"
NIGENS_FOA_AUDIO = NIGENS_ROOT / "foa_dev"
NIGENS_MIC_AUDIO = NIGENS_ROOT / "mic_dev"

STARSS23_FEAT_DIR = Path(r"D:\ssl-research\DCASE2024_SELD_dataset\seld_feat_label")
STARSS23_MIC_WTS  = STARSS23_FEAT_DIR / "mic_wts"
STARSS23_FOA_WTS  = STARSS23_FEAT_DIR / "foa_wts"

DCASE_REPO   = Path(r"D:\ssl-research\dcase2024_baseline")
MODEL_DIR    = DCASE_REPO / "models_audio"
RESULTS_OUT  = Path(r"D:\ssl-research\paper")

# ---------------------------------------------------------------------- mapping

# NIGENS-2021 class index --> STARSS24 class index (None means drop).
NIGENS2STARSS = {
    0:  11,   # alarm           -> bell
    1:  None, # crying baby     -> drop
    2:  None, # crash           -> drop
    3:  None, # barking dog     -> drop
    4:  None, # female scream   -> drop
    5:  0,    # female speech   -> femaleSpeech
    6:  6,    # footsteps       -> footsteps
    7:  12,   # knocking        -> knock
    8:  None, # male scream     -> drop
    9:  1,    # male speech     -> maleSpeech
    10: 3,    # ringing phone   -> telephone
    11: 9,    # piano           -> instrument
}
NIGENS_KEPT_STARSS_CLASSES = sorted(
    [c for c in NIGENS2STARSS.values() if c is not None])  # [0, 1, 3, 6, 9, 11, 12]

# ---------------------------------------------------------------------- cells

# Each cell: task, name, modality (mic|foa), job pattern, seeds.
CELLS = [
    {"task": "100", "name": "100_foa_no_gca",         "modality": "foa", "job_pattern": "repro_seed{seed}",  "seeds": [0, 1, 2, 3, 4]},
    {"task": "110", "name": "110_gca_full",           "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "111", "name": "111_gca_nogeom",         "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "112", "name": "112_no_gca",             "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2, 3, 4]},
    {"task": "130", "name": "130_foa_gca_full",       "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [1, 2, 3]},
    {"task": "131", "name": "131_foa_gca_nogeom",     "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [1, 2, 3]},
    {"task": "140", "name": "140_xfm_no_gca",         "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2]},
    {"task": "141", "name": "141_xfm_gca_full",       "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2]},
    {"task": "142", "name": "142_xfm_gca_nogeom",     "modality": "mic", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2]},
    {"task": "150", "name": "150_xfm_foa_no_gca",     "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2]},
    {"task": "151", "name": "151_xfm_foa_gca_full",   "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2]},
    {"task": "152", "name": "152_xfm_foa_gca_nogeom", "modality": "foa", "job_pattern": "ablate_seed{seed}", "seeds": [0, 1, 2]},
]

# Distance dimension is meaningless on NIGENS (no GT distance);
# threshold to inf so distance never disqualifies a match.
CROSS_PARAMS_OVERRIDES = {
    "lad_dist_thresh":    float("inf"),
    "lad_reldist_thresh": float("inf"),
}

# ---------------------------------------------------------------------- prep


def maybe_unzip_audio() -> bool:
    """Extract foa_dev.zip / mic_dev.zip into NIGENS_ROOT if not done.
    Returns True if at least FOA dev audio is available afterwards."""
    foa_extracted = NIGENS_FOA_AUDIO.is_dir() and any(NIGENS_FOA_AUDIO.glob("**/*.wav"))
    mic_extracted = NIGENS_MIC_AUDIO.is_dir() and any(NIGENS_MIC_AUDIO.glob("**/*.wav"))

    foa_zip   = NIGENS_ROOT / "foa_dev.zip"
    foa_z01   = NIGENS_ROOT / "foa_dev.z01"
    mic_zip   = NIGENS_ROOT / "mic_dev.zip"
    mic_z01   = NIGENS_ROOT / "mic_dev.z01"

    # Split-zip extraction: requires merging .z01 + .zip into a single .zip
    # using `zip -F` (Linux/macOS) or 7-Zip on Windows. We use 7z if available.
    def extract_split(prefix: str) -> bool:
        zip_main = NIGENS_ROOT / f"{prefix}.zip"
        zip_part = NIGENS_ROOT / f"{prefix}.z01"
        if not zip_main.is_file() or not zip_part.is_file():
            print(f"[unzip] missing pieces of {prefix} (zip={zip_main.is_file()}, z01={zip_part.is_file()})")
            return False
        # Try 7-Zip CLI
        candidates = [
            r"D:\ssl-research\tools\7zip\7z.exe",
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
            "7z.exe",
        ]
        sevenzip = next((c for c in candidates if Path(c).is_file() or shutil.which(c)), None)
        if not sevenzip:
            print(f"[unzip] 7z not found; cannot extract split-zip {prefix}.")
            return False
        out_dir = NIGENS_ROOT
        # Merge first to make a contiguous zip via 7z's "join" archives feature, then extract.
        # 7z handles split zips via the .zip extension entry point if both .z01 and .zip are in the same dir.
        cmd = [sevenzip, "x", "-y", str(zip_main), f"-o{out_dir}"]
        print(f"[unzip] running {cmd}")
        rc = os.system(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        return rc == 0

    if not foa_extracted:
        ok = extract_split("foa_dev")
        if not ok:
            print("[unzip] FOA extraction failed; cannot proceed.")
            return False
    if not mic_extracted:
        ok = extract_split("mic_dev")
        if not ok:
            print("[unzip] MIC extraction failed; will proceed FOA-only.")
    return True


def remap_metadata() -> None:
    """Convert NIGENS-2021 5-col CSVs into STARSS24 6-col CSVs with class
    remap + dummy distance. Drops events from un-mapped classes."""
    if NIGENS_REMAP_META.exists() and any(NIGENS_REMAP_META.glob("**/*.csv")):
        n = sum(1 for _ in NIGENS_REMAP_META.glob("**/*.csv"))
        print(f"[meta] remapped already exists ({n} CSVs); skipping")
        return
    NIGENS_REMAP_META.mkdir(parents=True, exist_ok=True)
    n_kept_total = 0; n_drop_total = 0; n_files = 0
    for sub in NIGENS_RAW_META.iterdir():
        if not sub.is_dir(): continue
        out_dir = NIGENS_REMAP_META / sub.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in sub.glob("*.csv"):
            n_files += 1
            with open(src, "r", newline="") as fin, \
                 open(out_dir / src.name, "w", newline="") as fout:
                w = csv.writer(fout)
                for row in csv.reader(fin):
                    if len(row) < 5: continue
                    try:
                        frame = int(row[0]); cls_n = int(row[1]); src_idx = int(row[2])
                        az = float(row[3]); el = float(row[4])
                    except (ValueError, IndexError):
                        continue
                    cls_s = NIGENS2STARSS.get(cls_n)
                    if cls_s is None:
                        n_drop_total += 1; continue
                    n_kept_total += 1
                    w.writerow([frame, cls_s, src_idx, int(round(az)), int(round(el)), 100])
    print(f"[meta] processed {n_files} CSVs, kept {n_kept_total} events, dropped {n_drop_total}")


# ---------------------------------------------------------------------- features


def extract_features(modality: str = "foa") -> bool:
    """Extract logmel + (GCC/IV) for NIGENS dev-test audio. Reuse STARSS23's
    train scaler for normalization (zero-shot transfer)."""
    feat_subdir = "mic_dev_norm" if modality == "mic" else "foa_dev_norm"
    out_dir = NIGENS_FEAT_DIR / feat_subdir
    if out_dir.exists():
        n = len(list(out_dir.glob("*.npy")))
        if n >= 50:
            print(f"[feat] NIGENS {modality.upper()} normalized features ready ({n}); skipping")
            return True

    p = parameters.get_params("110" if modality == "mic" else "100").copy()
    p["dataset_dir"]    = str(NIGENS_ROOT)
    p["feat_label_dir"] = str(NIGENS_FEAT_DIR)
    p["dataset"]        = modality
    p["use_salsalite"]  = False
    feat_cls = cls_feature_class.FeatureClass(p, is_eval=False)
    Path(feat_cls.get_normalized_feat_dir()).mkdir(parents=True, exist_ok=True)
    print(f"[feat] extracting NIGENS {modality.upper()} raw features ...")
    feat_cls.extract_all_feature()

    wts_src    = STARSS23_MIC_WTS if modality == "mic" else STARSS23_FOA_WTS
    wts_target = Path(feat_cls.get_normalized_wts_file())
    wts_target.parent.mkdir(parents=True, exist_ok=True)
    if not wts_target.exists():
        shutil.copy2(wts_src, wts_target)
        print(f"[feat] copied STARSS23 {wts_src.name} -> {wts_target}")
    feat_cls._is_eval = True
    feat_cls.preprocess_features()
    print(f"[feat] NIGENS {modality.upper()} normalized features ready")
    return True


# ---------------------------------------------------------------------- inference


def _decode_multi_accdoa_to_dict(out: np.ndarray, params: dict) -> dict:
    """(reused from STARSS22 cross-eval)."""
    from cls_compute_seld_results import reshape_3Dto2D
    from train_seldnet import (
        get_multi_accdoa_labels, determine_similar_location,
    )
    n_classes = params["unique_classes"]
    thresh_unify = params["thresh_unify"]
    sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, \
        sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(out, n_classes)
    sed_pred0 = reshape_3Dto2D(sed_pred0); doa_pred0 = reshape_3Dto2D(doa_pred0); dist_pred0 = reshape_3Dto2D(dist_pred0)
    sed_pred1 = reshape_3Dto2D(sed_pred1); doa_pred1 = reshape_3Dto2D(doa_pred1); dist_pred1 = reshape_3Dto2D(dist_pred1)
    sed_pred2 = reshape_3Dto2D(sed_pred2); doa_pred2 = reshape_3Dto2D(doa_pred2); dist_pred2 = reshape_3Dto2D(dist_pred2)

    output_dict: dict[int, list[list]] = {}
    nC = n_classes
    for fr in range(sed_pred0.shape[0]):
        for cc in range(sed_pred0.shape[1]):
            f01 = determine_similar_location(sed_pred0[fr][cc], sed_pred1[fr][cc], doa_pred0[fr], doa_pred1[fr], cc, thresh_unify, nC)
            f12 = determine_similar_location(sed_pred1[fr][cc], sed_pred2[fr][cc], doa_pred1[fr], doa_pred2[fr], cc, thresh_unify, nC)
            f20 = determine_similar_location(sed_pred2[fr][cc], sed_pred0[fr][cc], doa_pred2[fr], doa_pred0[fr], cc, thresh_unify, nC)
            total = f01 + f12 + f20
            if total == 0:
                if sed_pred0[fr][cc] > 0.5: output_dict.setdefault(fr, []).append([cc, doa_pred0[fr][cc], doa_pred0[fr][cc + nC], doa_pred0[fr][cc + 2 * nC], dist_pred0[fr][cc]])
                if sed_pred1[fr][cc] > 0.5: output_dict.setdefault(fr, []).append([cc, doa_pred1[fr][cc], doa_pred1[fr][cc + nC], doa_pred1[fr][cc + 2 * nC], dist_pred1[fr][cc]])
                if sed_pred2[fr][cc] > 0.5: output_dict.setdefault(fr, []).append([cc, doa_pred2[fr][cc], doa_pred2[fr][cc + nC], doa_pred2[fr][cc + 2 * nC], dist_pred2[fr][cc]])
            elif total == 1:
                output_dict.setdefault(fr, [])
                if f01:
                    if sed_pred2[fr][cc] > 0.5:
                        output_dict[fr].append([cc, doa_pred2[fr][cc], doa_pred2[fr][cc + nC], doa_pred2[fr][cc + 2 * nC], dist_pred2[fr][cc]])
                    doa_avg = (doa_pred0[fr] + doa_pred1[fr]) / 2
                    dist_avg = (dist_pred0[fr] + dist_pred1[fr]) / 2
                    output_dict[fr].append([cc, doa_avg[cc], doa_avg[cc + nC], doa_avg[cc + 2 * nC], dist_avg[cc]])
                elif f12:
                    if sed_pred0[fr][cc] > 0.5:
                        output_dict[fr].append([cc, doa_pred0[fr][cc], doa_pred0[fr][cc + nC], doa_pred0[fr][cc + 2 * nC], dist_pred0[fr][cc]])
                    doa_avg = (doa_pred1[fr] + doa_pred2[fr]) / 2
                    dist_avg = (dist_pred1[fr] + dist_pred2[fr]) / 2
                    output_dict[fr].append([cc, doa_avg[cc], doa_avg[cc + nC], doa_avg[cc + 2 * nC], dist_avg[cc]])
                elif f20:
                    if sed_pred1[fr][cc] > 0.5:
                        output_dict[fr].append([cc, doa_pred1[fr][cc], doa_pred1[fr][cc + nC], doa_pred1[fr][cc + 2 * nC], dist_pred1[fr][cc]])
                    doa_avg = (doa_pred2[fr] + doa_pred0[fr]) / 2
                    dist_avg = (dist_pred2[fr] + dist_pred0[fr]) / 2
                    output_dict[fr].append([cc, doa_avg[cc], doa_avg[cc + nC], doa_avg[cc + 2 * nC], dist_avg[cc]])
            elif total >= 2:
                output_dict.setdefault(fr, [])
                doa_avg = (doa_pred0[fr] + doa_pred1[fr] + doa_pred2[fr]) / 3
                dist_avg = (dist_pred0[fr] + dist_pred1[fr] + dist_pred2[fr]) / 3
                output_dict[fr].append([cc, doa_avg[cc], doa_avg[cc + nC], doa_avg[cc + 2 * nC], dist_avg[cc]])
    return output_dict


def evaluate_one_ckpt(task_id: str, job_id: str, modality: str, dump_root: Path) -> Optional[dict]:
    p = parameters.get_params(task_id).copy()
    p.update(CROSS_PARAMS_OVERRIDES)
    feat_tag = "mic_gcc" if modality == "mic" else "foa"
    out_tag  = "multiaccdoa"
    unique_name = f"{task_id}_{job_id}_dev_split0_{out_tag}_{feat_tag}"
    model_path = MODEL_DIR / f"{unique_name}_model.h5"
    if not model_path.is_file():
        print(f"  [skip] missing ckpt {model_path}"); return None
    feat_subdir = "mic_dev_norm" if modality == "mic" else "foa_dev_norm"
    feat_dir = NIGENS_FEAT_DIR / feat_subdir
    feat_files = sorted(feat_dir.glob("*.npy"))
    if not feat_files:
        print(f"  [skip] no NIGENS features at {feat_dir}"); return None

    n_classes = p["unique_classes"]; feat_seq_len = p["feature_sequence_length"]
    label_seq_len = p["label_sequence_length"]; nb_mel = p["nb_mel_bins"]
    in_ch = 4 + (6 if modality == "mic" else 3)
    in_shape = (1, in_ch, feat_seq_len, nb_mel)
    out_shape = (1, label_seq_len, n_classes * 3 * 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = seldnet_model.SeldModel(in_shape, out_shape, p).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()

    p_writer = p.copy()
    p_writer["dataset_dir"]    = str(NIGENS_ROOT)
    p_writer["feat_label_dir"] = str(NIGENS_FEAT_DIR)
    p_writer["dataset"]        = modality
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
            yb = model(xb).detach().cpu().numpy()
            output_dict = _decode_multi_accdoa_to_dict(yb, p)
            pad_T_label = (pad_T + 4) // 5
            if pad_T_label:
                T_total_label = n_seqs * label_seq_len
                trim_after = T_total_label - pad_T_label
                output_dict = {k: v for k, v in output_dict.items() if k < trim_after}
            stem = npy.stem
            feat_writer.write_output_format_file(str(dump_dir / f"{stem}.csv"), output_dict)
    print(f"  inference {unique_name} took {time.time() - t0:.1f}s")

    p2 = p.copy()
    score_obj = ComputeSELDResults(p2, ref_files_folder=str(NIGENS_REMAP_META))
    res = score_obj.get_SELD_Results(str(dump_dir), is_jackknife=False)
    test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise = res
    cw = np.asarray(classwise, dtype=np.float64)  # (7, 13)

    # Restricted-class metric: average classwise rows over the 7 mapped
    # STARSS classes only. Row layout: 0=ER (broadcast), 1=F1, 2=AngE,
    # 3=DistE, 4=RelDistE, 5=LR, 6=SELD per-class.
    kept_idx = NIGENS_KEPT_STARSS_CLASSES
    F1_kept   = float(np.nanmean(cw[1, kept_idx]))
    LE_kept   = float(np.nanmean(cw[2, kept_idx]))
    LR_kept   = float(np.nanmean(cw[5, kept_idx]))
    SELD_kept = float(np.nanmean(cw[6, kept_idx]))

    return {
        "ckpt": unique_name,
        "agg_F1":   float(test_F),
        "agg_ER":   float(test_ER),
        "agg_LE":   float(test_LE),
        "agg_LR":   float(test_LR),
        "agg_SELD": float(test_seld_scr),
        # Restricted to the 7 NIGENS-mapped STARSS classes.
        "kept_F1":   F1_kept,
        "kept_LE":   LE_kept,
        "kept_LR":   LR_kept,
        "kept_SELD": SELD_kept,
        "kept_classes": kept_idx,
    }


# ---------------------------------------------------------------------- aggregate


def cohens_dz(deltas):
    a = np.asarray(deltas, dtype=np.float64)
    s = a.std(ddof=1)
    return float("nan") if s == 0 else float(a.mean() / s)


def aggregate(per_ckpt: dict) -> dict:
    cells: dict[str, dict[int, dict]] = {c["name"]: {} for c in CELLS}
    for cell in CELLS:
        jp = cell.get("job_pattern", "ablate_seed{seed}")
        for s in cell.get("seeds", []):
            key = f"{cell['task']}_{jp.format(seed=s)}"
            cells[cell["name"]][s] = per_ckpt.get(key)

    summary = {}
    for cname, by_seed in cells.items():
        avail = {s: v for s, v in by_seed.items() if v is not None}
        out_cell = {"n_seeds": len(avail), "missing_seeds": []}
        for m in ("agg_F1", "agg_LE", "agg_LR", "agg_SELD",
                  "kept_F1", "kept_LE", "kept_LR", "kept_SELD"):
            vals = [v[m] for v in avail.values() if v is not None]
            out_cell[m] = {
                "values": vals,
                "mean":   mean(vals) if vals else None,
                "std":    stdev(vals) if len(vals) > 1 else 0.0,
                "n":      len(vals),
            }
        summary[cname] = out_cell

    contrasts = {}
    pair_specs = [
        # FOA + CRNN
        ("130_foa_gca_full",   "131_foa_gca_nogeom", "FOA+CRNN: GCA full vs no_geom (synth)"),
        ("130_foa_gca_full",   "100_foa_no_gca",     "FOA+CRNN: GCA full vs no GCA (synth)"),
        # MIC + CRNN
        ("110_gca_full",       "111_gca_nogeom",     "MIC+CRNN: GCA full vs no_geom (synth)"),
        ("110_gca_full",       "112_no_gca",         "MIC+CRNN: GCA full vs no GCA (synth)"),
        # MIC + Xfm
        ("141_xfm_gca_full",   "142_xfm_gca_nogeom", "MIC+Xfm: GCA full vs no_geom (synth)"),
        ("141_xfm_gca_full",   "140_xfm_no_gca",     "MIC+Xfm: GCA full vs no GCA (synth)"),
        # FOA + Xfm
        ("151_xfm_foa_gca_full", "152_xfm_foa_gca_nogeom", "FOA+Xfm: GCA full vs no_geom (synth)"),
        ("151_xfm_foa_gca_full", "150_xfm_foa_no_gca",     "FOA+Xfm: GCA full vs no GCA (synth)"),
    ]
    for ca, cb, descr in pair_specs:
        if ca not in cells or cb not in cells: continue
        per_metric = {}
        for m in ("kept_F1", "kept_LE", "kept_LR", "kept_SELD"):
            shared = sorted(set(cells[ca].keys()) & set(cells[cb].keys()))
            shared = [s for s in shared if cells[ca].get(s) and cells[cb].get(s)]
            n = len(shared)
            if n < 2:
                per_metric[m] = {"n": n, "note": "insufficient pairs"}; continue
            a = np.array([cells[ca][s][m] for s in shared])
            b = np.array([cells[cb][s][m] for s in shared])
            deltas = (a - b).tolist()
            t_stat, p_t = stats.ttest_rel(a, b)
            per_metric[m] = {
                "n": n,
                "delta_mean": float(np.mean(deltas)),
                "delta_std":  float(np.std(deltas, ddof=1)),
                "t":          float(t_stat),
                "p_t":        float(p_t),
                "cohens_dz":  cohens_dz(deltas),
            }
        contrasts[f"{ca}__vs__{cb}"] = {"description": descr, "metrics": per_metric}
    return {"per_cell": summary, "contrasts": contrasts}


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()

    print("[step 1] unzip audio if needed ...")
    if not maybe_unzip_audio():
        print("[error] cannot unzip; aborting"); return 1

    print("\n[step 2] remap NIGENS metadata to STARSS taxonomy ...")
    remap_metadata()

    print("\n[step 3] feature extraction ...")
    if NIGENS_FOA_AUDIO.is_dir(): extract_features("foa")
    if NIGENS_MIC_AUDIO.is_dir(): extract_features("mic")
    else: print("[warn] no MIC audio; skipping MIC cells")

    if args.prep_only:
        print("\n[prep-only] done."); return 0

    cells_filter = args.cells
    seeds_filter = args.seeds

    dump_root = NIGENS_ROOT / "predictions_nigens"
    dump_root.mkdir(parents=True, exist_ok=True)

    per_ckpt: dict[str, dict] = {}
    for cell in CELLS:
        task_id = cell["task"]
        if cells_filter is not None and task_id not in cells_filter: continue
        modality = cell.get("modality", "mic")
        # If MIC features missing, skip MIC cells gracefully.
        if modality == "mic" and not NIGENS_MIC_AUDIO.is_dir(): continue
        jp = cell.get("job_pattern", "ablate_seed{seed}")
        cell_seeds = seeds_filter if seeds_filter is not None else cell.get("seeds", [0,1,2,3,4])
        for s in cell_seeds:
            job_id = jp.format(seed=s)
            print(f"\n=== [{task_id} {job_id} | modality={modality}] ===")
            res = evaluate_one_ckpt(task_id, job_id, modality, dump_root)
            if res is not None:
                per_ckpt[f"{task_id}_{job_id}"] = res
                print(f"  agg F1={res['agg_F1']*100:.2f}% kept F1={res['kept_F1']*100:.2f}% "
                      f"agg LE={res['agg_LE']:.2f} kept LE={res['kept_LE']:.2f}")

    # Merge with any previously-saved per_ckpt so partial (per-modality) runs
    # accumulate instead of overwriting each other.
    out_json = RESULTS_OUT / "path_c_synth_nigens.json"
    out_md   = RESULTS_OUT / "path_c_synth_nigens.md"
    if out_json.is_file():
        try:
            prev = json.loads(out_json.read_text(encoding="utf-8")).get("per_ckpt", {})
            merged = dict(prev)
            merged.update(per_ckpt)  # new results take precedence
            n_added = len(set(per_ckpt) - set(prev))
            n_kept  = len(set(prev) - set(per_ckpt))
            print(f"[merge] carried over {n_kept} prior ckpt(s), added/updated {len(per_ckpt)}")
            per_ckpt = merged
        except Exception as e:
            print(f"[merge] could not merge prior results: {e}")

    payload = aggregate(per_ckpt)
    payload["per_ckpt"] = per_ckpt
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    L = ["# Path C / E (synth): TAU-NIGENS-SSE-2021 zero-shot evaluation",
         "",
         "Models trained on STARSS23 dev-train, evaluated on TAU-NIGENS-SSE-2021 dev-test.",
         "NIGENS uses 12 classes; we remap 7 overlapping classes to STARSS taxonomy and",
         "drop events from the other 5 classes (crying baby, crash, barking dog, female",
         "scream, male scream). Distance is meaningless on NIGENS; lad_dist_thresh=inf.",
         "",
         "Metrics labeled `kept_*` are averaged over only the 7 mapped STARSS classes;",
         "`agg_*` are the standard 13-class average.",
         "",
         "## Per-cell mean +/- std (kept-class subset)",
         "",
         "| Cell                       | n  | F1 (%)            | LE (deg)         | SELD             |",
         "| -------------------------- | -- | ----------------- | ---------------- | ---------------- |"]
    for cname, c in payload["per_cell"].items():
        n = c["n_seeds"]
        if n == 0:
            L.append(f"| {cname:<26} | 0  | n/a               | n/a              | n/a              |"); continue
        f = c["kept_F1"]; le = c["kept_LE"]; sl = c["kept_SELD"]
        L.append(f"| {cname:<26} | {n:<2} | {100*f['mean']:.2f} \u00b1 {100*f['std']:.2f}     | {le['mean']:.2f} \u00b1 {le['std']:.2f}    | {sl['mean']:.3f} \u00b1 {sl['std']:.3f}  |")
    L.append("")
    L.append("## Paired contrasts (synth, kept-class subset)")
    L.append("")
    for cname, c in payload["contrasts"].items():
        L.append(f"### {cname}")
        L.append(f"_{c['description']}_\n")
        L.append("| Metric    | n | delta mean | t (p) | d_z |")
        L.append("| --------- | - | ---------- | ----- | --- |")
        for m in ("kept_F1", "kept_LE", "kept_SELD"):
            r = c["metrics"].get(m, {"n": 0, "note": "n/a"})
            if r.get("n", 0) < 2:
                L.append(f"| {m:<9} | {r.get('n',0)} | n/a | - | - |"); continue
            L.append(f"| {m:<9} | {r['n']} | {r['delta_mean']:+.3f} \u00b1 {r['delta_std']:.3f} | t={r['t']:+.2f} (p={r['p_t']:.3f}) | {r['cohens_dz']:+.2f} |")
        L.append("")
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[saved] {out_json}\n[saved] {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
