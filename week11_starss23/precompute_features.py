"""Precompute STARSS23 SELD features and cache them to disk.

Walks both train and test splits, runs the full feature pipeline once
per clip, and stores ``(features, n_label_frames)`` to
``cache_dir/{name}_{cfg_hash}.npz``. Idempotent -- already-cached clips
are skipped.

Usage:
    python -m week11_starss23.precompute_features [--cache-dir PATH]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from week11_starss23.seld_features import (
    SeldFeatureConfig,
    crop_features_to_label_frames,
    extract_seld_features,
    load_multichannel_audio,
    num_label_frames,
)
from week11_starss23.starss_dataset import _config_hash, discover_clips

DEFAULT_AUDIO_DIR = Path("D:/ssl-research/data/STARSS23/mic_dev")
DEFAULT_META_DIR = Path("D:/ssl-research/data/STARSS23/metadata_dev/metadata_dev")
DEFAULT_CACHE_DIR = Path("D:/ssl-research/data/STARSS23/feat_cache")


def precompute_split(
    audio_dir: Path,
    metadata_dir: Path,
    cache_dir: Path,
    split: str,
    cfg: SeldFeatureConfig,
) -> tuple[int, int, float]:
    """Returns ``(n_processed, n_skipped, elapsed_s)``."""
    cfg_hash = _config_hash(cfg)
    clips = discover_clips(metadata_dir, audio_dir, split=split)  # type: ignore[arg-type]
    n_proc = n_skip = 0
    t_start = time.perf_counter()

    for k, clip in enumerate(clips, 1):
        cache_path = cache_dir / f"{clip.name}_{cfg_hash}.npz"
        if cache_path.exists():
            n_skip += 1
            continue
        if not clip.has_audio():
            print(f"  [skip] {clip.name}: no audio file")
            continue

        t0 = time.perf_counter()
        audio = load_multichannel_audio(
            clip.audio_path, target_fs=cfg.fs, n_mics=cfg.n_mics
        )
        n_label = num_label_frames(audio.shape[1], cfg)
        features = extract_seld_features(audio, cfg)
        features = crop_features_to_label_frames(features, n_label, cfg)

        np.savez_compressed(
            cache_path,
            features=features,
            n_label_frames=np.int32(n_label),
        )
        dt = time.perf_counter() - t0
        n_proc += 1
        dur = audio.shape[1] / cfg.fs
        print(
            f"  [{split} {k:3d}/{len(clips)}] {clip.name}  "
            f"{dur:5.1f}s -> {features.shape}  cached in {dt:5.2f}s",
            flush=True,
        )

    return n_proc, n_skip, time.perf_counter() - t_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_META_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()

    cfg = SeldFeatureConfig()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir : {args.cache_dir}")
    print(f"Cfg hash  : {_config_hash(cfg)}")
    print(f"Cfg       : fs={cfg.fs}, n_fft={cfg.n_fft}, hop={cfg.hop_samples}, "
          f"label_hop={cfg.label_hop_samples}, n_mels={cfg.n_mels}, "
          f"n_gcc_lags={cfg.n_gcc_lags}")
    print()

    total_proc = total_skip = 0
    total_t = 0.0
    for split in ("train", "test"):
        print(f"--- split={split} ---")
        n_p, n_s, dt = precompute_split(
            args.audio_dir, args.metadata_dir, args.cache_dir, split, cfg
        )
        print(f"  -> processed {n_p}, skipped {n_s}, elapsed {dt:.1f}s\n")
        total_proc += n_p
        total_skip += n_s
        total_t += dt

    cache_files = list(args.cache_dir.glob(f"*_{_config_hash(cfg)}.npz"))
    cache_size_mb = sum(f.stat().st_size for f in cache_files) / 1024 / 1024
    print(
        f"Done. processed={total_proc}, skipped={total_skip}, "
        f"cache files={len(cache_files)}, "
        f"cache size={cache_size_mb:.1f} MB, elapsed={total_t:.1f}s"
    )


if __name__ == "__main__":
    main()
