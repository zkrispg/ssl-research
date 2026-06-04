"""DCASE SELD evaluation for a trained STARSS23 checkpoint.

Loads a ``best.pt`` saved by :mod:`train_seld`, runs the model on the
dev-test split (full-clip), decodes the Multi-ACCDOA predictions, and
reports DCASE Task 3 metrics: F1, ER, LE_CD, LR_CD, SELD score.

Usage:
    python -m week11_starss23.evaluate_seld --ckpt PATH/best.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from week11_starss23.seld_features import SeldFeatureConfig
from week11_starss23.seld_metrics import (
    DcaseSeldStats,
    decode_pred_to_events,
    target_to_events,
)
from week11_starss23.seld_model import SeldCRNN, SeldModelConfig, default_uca4_positions
from week11_starss23.starss_dataset import StarssDataset

DEFAULT_AUDIO_DIR = Path("D:/ssl-research/data/STARSS23/mic_dev")
DEFAULT_META_DIR = Path("D:/ssl-research/data/STARSS23/metadata_dev/metadata_dev")
DEFAULT_CACHE_DIR = Path("D:/ssl-research/data/STARSS23/feat_cache")


def load_checkpoint(ckpt_path: Path, device: torch.device):
    """Load a SELD checkpoint, dispatching on ``model_type``.

    Old checkpoints (saved before ``model_type`` was added) are assumed
    to be :class:`SeldCRNN` for backwards compatibility.
    """
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    feat_cfg = SeldFeatureConfig(**blob["feat_cfg"])
    model_type = blob.get("model_type", "seld_crnn")
    if model_type == "seldnet_official":
        from week11_starss23.seldnet_official import (
            SeldNetOfficial,
            SeldNetOfficialConfig,
        )
        model_cfg = SeldNetOfficialConfig(**blob["model_cfg"])
        model = SeldNetOfficial(model_cfg)
    else:
        model_cfg = SeldModelConfig(**blob["model_cfg"])
        model = SeldCRNN(model_cfg, mic_positions=default_uca4_positions())
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model, feat_cfg, blob


@torch.no_grad()
def evaluate_one_clip(
    model: SeldCRNN,
    sample: dict,
    stats: DcaseSeldStats,
    device: torch.device,
    activity_threshold: float,
    nms_tol_deg: float,
) -> tuple[int, int]:
    """Predict one clip and accumulate frame-level metrics. Returns ``(n_pred_events, n_gt_events)``."""
    feats = sample["features"].unsqueeze(0).to(device)
    target = sample["target"].numpy()  # (T_label, 6, 4, n_classes)
    out_flat = model(feats)["accdoa"][0].cpu().numpy()  # (T_label, n_tracks*3*n_classes)
    T = out_flat.shape[0]
    n_classes = target.shape[-1]
    n_tracks = out_flat.shape[1] // (3 * n_classes)
    pred_tensor = out_flat.reshape(T, n_tracks, 3, n_classes)

    pred_frames = decode_pred_to_events(
        pred_tensor, activity_threshold=activity_threshold, nms_tol_deg=nms_tol_deg
    )
    gt_frames = target_to_events(target)

    n_pred_events = sum(len(f) for f in pred_frames)
    n_gt_events = sum(len(f) for f in gt_frames)

    for p, g in zip(pred_frames, gt_frames):
        stats.add_frame(p, g)
    return n_pred_events, n_gt_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="path to best.pt")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--split", choices=("train", "test"), default="test",
        help="which dev split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--activity-threshold", type=float, default=0.5,
        help="ACCDOA magnitude threshold for active prediction"
    )
    parser.add_argument(
        "--nms-tol-deg", type=float, default=15.0,
        help="same-class predictions within this angular distance get merged"
    )
    parser.add_argument(
        "--tolerance-deg", type=float, default=20.0,
        help="DCASE location tolerance for TP/FP/FN matching"
    )
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat_cfg, blob = load_checkpoint(args.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ckpt] {args.ckpt}")
    print(f"[ckpt] variant={blob['args']['variant']}  seed={blob['args']['seed']}  "
          f"trained_for={blob.get('epoch', '?')} epochs  params={n_params:,}")
    print(f"[device] {device}")

    n_classes = blob["model_cfg"]["n_classes"]
    stats = DcaseSeldStats(
        tolerance_deg=args.tolerance_deg, n_classes=n_classes
    )

    ds = StarssDataset(
        audio_dir=args.audio_dir,
        metadata_dir=args.metadata_dir,
        split=args.split,
        feature_config=feat_cfg,
        clip_seconds=None,  # full clip evaluation
        random_crop=False,
        cache_dir=args.cache_dir,
    )
    print(f"[data] split={args.split}  {len(ds)} clips")

    t0 = time.perf_counter()
    total_pred = total_gt = 0
    for k in range(len(ds)):
        sample = ds[k]
        n_p, n_g = evaluate_one_clip(
            model, sample, stats, device,
            args.activity_threshold, args.nms_tol_deg,
        )
        total_pred += n_p
        total_gt += n_g
        if (k + 1) % 10 == 0 or k == len(ds) - 1:
            print(
                f"  [{k+1:3d}/{len(ds)}] {sample['name']}  "
                f"pred_evt={n_p}  gt_evt={n_g}  "
                f"frames={stats.n_frames}",
                flush=True,
            )
    dt = time.perf_counter() - t0

    macro = stats.summary("macro")
    micro = stats.summary("micro")
    per = stats.per_class_metrics()

    print(f"\n[done] elapsed={dt:.1f}s  total_pred={total_pred}  total_gt={total_gt}")
    print(f"[metrics] macro: F1={macro['f1']:.3f}  ER={macro['er']:.3f}  "
          f"LE={macro['le_cd']:.1f}deg  LR={macro['lr_cd']:.3f}  SELD={macro['seld']:.3f}")
    print(f"[metrics] micro: F1={micro['f1']:.3f}  ER={micro['er']:.3f}  "
          f"LE={micro['le_cd']:.1f}deg  LR={micro['lr_cd']:.3f}  SELD={micro['seld']:.3f}")

    print("\n[per-class (active only)]")
    print(f"  {'class':>5}  {'n_ref':>5}  {'F1':>6}  {'ER':>6}  {'LE':>6}  {'LR':>6}")
    for c in range(n_classes):
        n_ref = int(stats.n_ref_per_class[c])
        if n_ref == 0:
            continue
        print(
            f"  {c:>5}  {n_ref:>5}  {per['f1'][c]:>6.3f}  {per['er_cd'][c]:>6.3f}  "
            f"{per['le_cd'][c]:>6.1f}  {per['lr_cd'][c]:>6.3f}"
        )

    if args.out_json is not None:
        result = {
            "ckpt": str(args.ckpt),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "macro": macro,
            "micro": micro,
            "per_class": {
                k: per[k].tolist() for k in ("precision", "recall", "f1", "le_cd", "lr_cd", "er_cd")
            },
            "n_ref_per_class": stats.n_ref_per_class.tolist(),
            "tp_per_class": stats.tp_per_class.tolist(),
            "fp_per_class": stats.fp_per_class.tolist(),
            "fn_per_class": stats.fn_per_class.tolist(),
            "elapsed_s": dt,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
