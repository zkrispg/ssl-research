"""End-to-end smoke test on a real STARSS23 clip.

Reads one WAV + matching CSV through the full Day 1-4 pipeline:
    seld_features.load_multichannel_audio
    seld_features.extract_seld_features
    seld_labels.parse_starss_csv
    seld_labels.events_to_multi_accdoa
    seld_model.SeldCRNN forward pass (CPU + GPU)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from week11_starss23.seld_features import (
    SeldFeatureConfig,
    crop_features_to_label_frames,
    extract_seld_features,
    load_multichannel_audio,
    num_label_frames,
)
from week11_starss23.seld_labels import (
    estimate_num_frames,
    events_to_multi_accdoa,
    parse_starss_csv,
)
from week11_starss23.seld_model import SeldCRNN, SeldModelConfig
from week11_starss23.starss_dataset import StarssDataset

DATA_ROOT = Path("D:/ssl-research/data/STARSS23")
AUDIO_DIR = DATA_ROOT / "mic_dev"
META_DIR = DATA_ROOT / "metadata_dev" / "metadata_dev"


def main() -> None:
    cfg = SeldFeatureConfig()

    # ---- 1) load one real STARSS23 clip ---------------------------------
    wav = next((AUDIO_DIR / "dev-train-sony").glob("fold3_*.wav"))
    csv = META_DIR / "dev-train-sony" / f"{wav.stem}.csv"
    print(f"[clip] {wav.name}")
    t0 = time.perf_counter()
    audio = load_multichannel_audio(wav, target_fs=cfg.fs, n_mics=cfg.n_mics)
    t_load = time.perf_counter() - t0
    duration_s = audio.shape[1] / cfg.fs
    print(f"  shape={audio.shape}  duration={duration_s:.1f}s  load={t_load:.2f}s")

    # ---- 2) feature extraction ------------------------------------------
    t0 = time.perf_counter()
    features = extract_seld_features(audio, cfg)
    n_label = num_label_frames(audio.shape[1], cfg)
    features = crop_features_to_label_frames(features, n_label, cfg)
    t_feat = time.perf_counter() - t0
    print(
        f"  features: {features.shape}  T_label={n_label}  feat_extract={t_feat:.2f}s"
    )
    assert features.shape == (10, n_label * 5, 64), f"unexpected feat shape {features.shape}"

    # ---- 3) parse + label tensor ----------------------------------------
    events = parse_starss_csv(csv)
    target = events_to_multi_accdoa(events, num_frames=n_label)
    print(
        f"  events={len(events)}  target.shape={tuple(target.shape)}  "
        f"active_frames={int((target.abs().sum(dim=(1,2,3)) > 0).sum())}"
    )

    # ---- 4) full Dataset path -------------------------------------------
    ds = StarssDataset(
        AUDIO_DIR, META_DIR, split="train", feature_config=cfg, clip_seconds=5.0,
        random_crop=True, seed=0,
    )
    print(f"\n[StarssDataset] split=train -> {len(ds)} clips")
    t0 = time.perf_counter()
    sample = ds[0]
    t_ds = time.perf_counter() - t0
    print(
        f"  features={tuple(sample['features'].shape)}  "
        f"target={tuple(sample['target'].shape)}  load+extract={t_ds:.2f}s"
    )

    # ---- 5) full SELD model forward (CPU) -------------------------------
    print("\n[Model forward CPU]")
    model = SeldCRNN(SeldModelConfig(use_gca=True, gca_geometry_bias=False))
    x = sample["features"].unsqueeze(0)
    t0 = time.perf_counter()
    out = model(x)["accdoa"]
    print(f"  in={tuple(x.shape)}  out={tuple(out.shape)}  cpu_fwd={time.perf_counter()-t0:.3f}s")

    # ---- 6) full SELD model forward (GPU if available) ------------------
    if torch.cuda.is_available():
        print("\n[Model forward GPU]")
        model = model.cuda()
        x_gpu = x.cuda()
        # warm-up
        for _ in range(3):
            _ = model(x_gpu)["accdoa"]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            out = model(x_gpu)["accdoa"]
        torch.cuda.synchronize()
        print(
            f"  out={tuple(out.shape)}  gpu_fwd_avg={(time.perf_counter()-t0)/5*1000:.1f} ms"
        )
    else:
        print("\n[Model forward GPU] -- CUDA not available, skip")

    print("\n[OK] real STARSS23 end-to-end smoke test passed")


if __name__ == "__main__":
    main()
