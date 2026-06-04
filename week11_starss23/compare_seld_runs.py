"""Compare multiple trained SELD checkpoints with shared decode/threshold.

Loads every checkpoint, runs each on the same dev split with identical
``activity_threshold`` / ``nms_tol_deg``, and prints a side-by-side
DCASE Task 3 table. Useful for paired comparisons such as
``full`` vs ``no_geom`` (the core RQ of the ICASSP paper).

Usage:
    python -m week11_starss23.compare_seld_runs \
        --ckpts runs/no_geom_seed0_mc8_inmem/best.pt \
                runs/full_seed0_mc8_inmem/best.pt \
        --activity-threshold 0.18 \
        --out-json runs/compare_no_geom_vs_full_seed0.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from week11_starss23.evaluate_seld import load_checkpoint
from week11_starss23.seld_features import SeldFeatureConfig
from week11_starss23.seld_metrics import (
    DcaseSeldStats,
    decode_pred_to_events,
    target_to_events,
)
from week11_starss23.starss_dataset import StarssDataset

DEFAULT_AUDIO_DIR = Path("D:/ssl-research/data/STARSS23/mic_dev")
DEFAULT_META_DIR = Path("D:/ssl-research/data/STARSS23/metadata_dev/metadata_dev")
DEFAULT_CACHE_DIR = Path("D:/ssl-research/data/STARSS23/feat_cache")


def evaluate_ckpt(
    ckpt_path: Path,
    *,
    split: str,
    audio_dir: Path,
    metadata_dir: Path,
    cache_dir: Path,
    activity_threshold: float,
    nms_tol_deg: float,
    tolerance_deg: float,
    device: torch.device,
) -> dict:
    """Evaluate one checkpoint and return all DCASE metrics in a dict."""
    model, feat_cfg, blob = load_checkpoint(ckpt_path, device)
    n_classes = blob["model_cfg"]["n_classes"]
    variant = blob["args"].get("variant", "?")
    seed = blob["args"].get("seed", "?")

    ds = StarssDataset(
        audio_dir=audio_dir,
        metadata_dir=metadata_dir,
        split=split,
        feature_config=feat_cfg,
        clip_seconds=None,
        random_crop=False,
        cache_dir=cache_dir,
    )
    stats = DcaseSeldStats(tolerance_deg=tolerance_deg, n_classes=n_classes)
    t0 = time.perf_counter()
    with torch.no_grad():
        for k in range(len(ds)):
            sample = ds[k]
            feats = sample["features"].unsqueeze(0).to(device)
            target = sample["target"].numpy()
            out_flat = model(feats)["accdoa"][0].cpu().numpy()
            T = out_flat.shape[0]
            n_tracks = out_flat.shape[1] // (3 * n_classes)
            pred = out_flat.reshape(T, n_tracks, 3, n_classes)
            pred_frames = decode_pred_to_events(
                pred, activity_threshold=activity_threshold, nms_tol_deg=nms_tol_deg
            )
            gt_frames = target_to_events(target)
            for p, g in zip(pred_frames, gt_frames):
                stats.add_frame(p, g)
    dt = time.perf_counter() - t0

    macro = stats.summary("macro")
    micro = stats.summary("micro")
    return {
        "ckpt": str(ckpt_path),
        "variant": variant,
        "seed": seed,
        "n_clips": len(ds),
        "n_classes": n_classes,
        "best_eval_loss": blob.get("history", [{}])[-1].get("eval_loss", float("nan"))
        if blob.get("history") else float("nan"),
        "elapsed_s": dt,
        "macro": macro,
        "micro": micro,
    }


def fmt_row(label: str, m: dict) -> str:
    macro = m["macro"]
    micro = m["micro"]
    return (
        f"{label:>20}  "
        f"{macro['f1']:>5.3f}  {micro['f1']:>5.3f}  "
        f"{macro['er']:>6.2f}  {micro['er']:>6.2f}  "
        f"{macro['le_cd']:>5.1f}  {micro['le_cd']:>5.1f}  "
        f"{macro['lr_cd']:>5.3f}  {micro['lr_cd']:>5.3f}  "
        f"{macro['seld']:>6.3f}  {micro['seld']:>6.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpts", nargs="+", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--activity-threshold", type=float, default=0.18)
    parser.add_argument("--nms-tol-deg", type=float, default=15.0)
    parser.add_argument("--tolerance-deg", type=float, default=20.0)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  thr={args.activity_threshold}  nms={args.nms_tol_deg}deg  "
          f"tol={args.tolerance_deg}deg  split={args.split}")

    rows: list[dict] = []
    for ckpt in args.ckpts:
        print(f"\n[eval] {ckpt}", flush=True)
        m = evaluate_ckpt(
            ckpt,
            split=args.split,
            audio_dir=args.audio_dir,
            metadata_dir=args.metadata_dir,
            cache_dir=args.cache_dir,
            activity_threshold=args.activity_threshold,
            nms_tol_deg=args.nms_tol_deg,
            tolerance_deg=args.tolerance_deg,
            device=device,
        )
        print(f"  variant={m['variant']}  seed={m['seed']}  elapsed={m['elapsed_s']:.1f}s")
        rows.append(m)

    print(f"\n{'method':>20}  {'F1m':>5}  {'F1u':>5}  "
          f"{'ERm':>6}  {'ERu':>6}  {'LEm':>5}  {'LEu':>5}  "
          f"{'LRm':>5}  {'LRu':>5}  {'SELDm':>6}  {'SELDu':>6}")
    for m in rows:
        label = f"{m['variant']}_seed{m['seed']}"
        print(fmt_row(label, m))

    # Pairwise diff against the first row
    if len(rows) >= 2:
        ref = rows[0]
        print(f"\n[paired delta vs {ref['variant']}_seed{ref['seed']}]")
        print(f"{'method':>20}  {'dF1m':>+7}  {'dF1u':>+7}  {'dSELDm':>+7}  {'dSELDu':>+7}")
        for m in rows[1:]:
            label = f"{m['variant']}_seed{m['seed']}"
            d_f1m = m["macro"]["f1"] - ref["macro"]["f1"]
            d_f1u = m["micro"]["f1"] - ref["micro"]["f1"]
            d_sm = m["macro"]["seld"] - ref["macro"]["seld"]
            d_su = m["micro"]["seld"] - ref["micro"]["seld"]
            print(f"{label:>20}  {d_f1m:>+7.3f}  {d_f1u:>+7.3f}  "
                  f"{d_sm:>+7.3f}  {d_su:>+7.3f}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps({"args": {k: str(v) if isinstance(v, (Path, list)) else v for k, v in vars(args).items()},
                        "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
