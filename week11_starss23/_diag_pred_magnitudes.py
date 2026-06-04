"""Quick diagnostic: print prediction magnitude distribution.

Helps determine whether the model collapsed to silence or whether the
activity threshold is just too high.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from week11_starss23.evaluate_seld import load_checkpoint
from week11_starss23.starss_dataset import StarssDataset
from week11_starss23.seld_features import SeldFeatureConfig

CKPT = Path("D:/ssl-research/week11_starss23/runs/no_geom_seed0_real15/best.pt")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat_cfg, blob = load_checkpoint(CKPT, device)
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

    all_mags = []
    all_active_mags = []  # only counting (frame, track, class) where target is active
    n_clips_to_check = 5
    for k in range(min(n_clips_to_check, len(ds))):
        sample = ds[k]
        feats = sample["features"].unsqueeze(0).to(device)
        target = sample["target"].numpy()  # (T, 6, 4, C)
        with torch.no_grad():
            out_flat = model(feats)["accdoa"][0].cpu().numpy()
        T = out_flat.shape[0]
        n_tracks = out_flat.shape[1] // (3 * n_classes)
        pred = out_flat.reshape(T, n_tracks, 3, n_classes)
        mags = np.linalg.norm(pred, axis=2)  # (T, n_tracks, C)
        all_mags.append(mags.flatten())

        # Active mask from target (any of the 6 dummy slots active for that class)
        target_active = (target[:, :, 0, :] >= 0.5).any(axis=1)  # (T, C)
        for t in range(T):
            for c in range(n_classes):
                if target_active[t, c]:
                    all_active_mags.append(mags[t, :, c].max())

    all_mags = np.concatenate(all_mags)
    all_active_mags = np.array(all_active_mags) if all_active_mags else np.array([np.nan])

    print(f"n_clips_inspected: {min(n_clips_to_check, len(ds))}")
    print(f"\n=== ALL pred magnitudes ===")
    print(f"  mean   {all_mags.mean():.4f}")
    print(f"  std    {all_mags.std():.4f}")
    print(f"  max    {all_mags.max():.4f}")
    print(f"  p99    {np.percentile(all_mags, 99):.4f}")
    print(f"  p99.9  {np.percentile(all_mags, 99.9):.4f}")
    print(f"  > 0.10: {(all_mags > 0.10).mean()*100:.4f}%")
    print(f"  > 0.20: {(all_mags > 0.20).mean()*100:.4f}%")
    print(f"  > 0.50: {(all_mags > 0.50).mean()*100:.4f}%")

    print(f"\n=== pred magnitudes WHERE TARGET IS ACTIVE ===")
    print(f"  n_active_target_cells: {len(all_active_mags)}")
    if len(all_active_mags) > 0 and not np.isnan(all_active_mags[0]):
        print(f"  mean   {all_active_mags.mean():.4f}")
        print(f"  std    {all_active_mags.std():.4f}")
        print(f"  max    {all_active_mags.max():.4f}")
        print(f"  p50    {np.percentile(all_active_mags, 50):.4f}")
        print(f"  p90    {np.percentile(all_active_mags, 90):.4f}")
        print(f"  > 0.10: {(all_active_mags > 0.10).mean()*100:.2f}%")
        print(f"  > 0.20: {(all_active_mags > 0.20).mean()*100:.2f}%")
        print(f"  > 0.50: {(all_active_mags > 0.50).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
