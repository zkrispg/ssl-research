"""Sweep activity-threshold and report SELD at each threshold."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from week11_starss23.evaluate_seld import load_checkpoint
from week11_starss23.seld_metrics import (
    DcaseSeldStats,
    decode_pred_to_events,
    target_to_events,
)
from week11_starss23.starss_dataset import StarssDataset

DEFAULT_CKPT = "D:/ssl-research/week11_starss23/runs/no_geom_seed0_mc8_inmem/best.pt"
THRESHOLDS = [0.05, 0.10, 0.12, 0.15, 0.18, 0.22, 0.30]


def main() -> None:
    import sys
    ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat_cfg, blob = load_checkpoint(ckpt, device)
    print(f"[ckpt] {ckpt}", flush=True)
    n_classes = blob["model_cfg"]["n_classes"]

    ds = StarssDataset(
        audio_dir=Path("D:/ssl-research/data/STARSS23/mic_dev"),
        metadata_dir=Path("D:/ssl-research/data/STARSS23/metadata_dev/metadata_dev"),
        split="test",
        feature_config=feat_cfg,
        clip_seconds=None,
        random_crop=False,
        cache_dir=Path("D:/ssl-research/data/STARSS23/feat_cache"),
    )

    print(f"Caching predictions for {len(ds)} clips ...", flush=True)
    cached_preds: list[np.ndarray] = []
    cached_targets: list[np.ndarray] = []
    with torch.no_grad():
        for k in range(len(ds)):
            sample = ds[k]
            feats = sample["features"].unsqueeze(0).to(device)
            target = sample["target"].numpy()
            out_flat = model(feats)["accdoa"][0].cpu().numpy()
            T = out_flat.shape[0]
            n_tracks = out_flat.shape[1] // (3 * n_classes)
            cached_preds.append(out_flat.reshape(T, n_tracks, 3, n_classes))
            cached_targets.append(target)
            if (k + 1) % 20 == 0:
                print(f"  {k+1}/{len(ds)}", flush=True)

    print(f"\n{'thr':>5}  {'F1m':>5}  {'F1u':>5}  {'ERm':>6}  {'ERu':>6}  "
          f"{'LEm':>6}  {'LEu':>6}  {'LRm':>5}  {'LRu':>5}  {'SELDu':>6}")
    for thr in THRESHOLDS:
        stats = DcaseSeldStats(tolerance_deg=20.0, n_classes=n_classes)
        for pred, tgt in zip(cached_preds, cached_targets):
            pred_frames = decode_pred_to_events(pred, activity_threshold=thr, nms_tol_deg=15.0)
            gt_frames = target_to_events(tgt)
            for p, g in zip(pred_frames, gt_frames):
                stats.add_frame(p, g)
        macro = stats.summary("macro")
        micro = stats.summary("micro")
        print(
            f"{thr:>5.2f}  "
            f"{macro['f1']:>5.3f}  {micro['f1']:>5.3f}  "
            f"{macro['er']:>6.3f}  {micro['er']:>6.3f}  "
            f"{macro['le_cd']:>6.1f}  {micro['le_cd']:>6.1f}  "
            f"{macro['lr_cd']:>5.3f}  {micro['lr_cd']:>5.3f}  "
            f"{micro['seld']:>6.3f}"
        )


if __name__ == "__main__":
    main()
