"""Unit tests for week11_starss23.starss_dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from week11_starss23.seld_features import SeldFeatureConfig
from week11_starss23.seld_labels import (
    NUM_AXIS,
    NUM_DCASE2023_CLASSES,
    NUM_TRACK_DUMMY,
)
from week11_starss23.starss_dataset import (
    StarssClipMeta,
    StarssDataset,
    _config_hash,
    discover_clips,
)


# ---------------------------------------------------------------------------
# Synthetic dataset fixture (works without downloaded STARSS23 audio)
# ---------------------------------------------------------------------------


def _build_synthetic_starss(
    root: Path,
    fs: int = 24_000,
    duration_s: float = 6.0,
    n_clips_per_split: dict[str, int] | None = None,
) -> tuple[Path, Path]:
    """Construct a fake mini-STARSS23 layout under ``root``.

    Returns ``(audio_dir, metadata_dir)``.
    """
    if n_clips_per_split is None:
        n_clips_per_split = {
            "dev-train-sony": 2,
            "dev-train-tau": 1,
            "dev-test-sony": 1,
            "dev-test-tau": 1,
        }
    audio_dir = root / "mic_dev"
    metadata_dir = root / "metadata_dev"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    n_samples = int(fs * duration_s)
    n_label_frames = int(duration_s / 0.1)
    rng = np.random.default_rng(123)

    for split_dir, n_clips in n_clips_per_split.items():
        (audio_dir / split_dir).mkdir(parents=True, exist_ok=True)
        (metadata_dir / split_dir).mkdir(parents=True, exist_ok=True)
        fold = "fold3" if "train" in split_dir else "fold4"
        for k in range(1, n_clips + 1):
            stem = f"{fold}_room1_mix{k:03d}"
            audio = rng.standard_normal((n_samples, 4)).astype(np.float32) * 0.05
            sf.write(audio_dir / split_dir / f"{stem}.wav", audio, fs, subtype="FLOAT")

            # Drop a couple of events at known frames/classes.
            csv_path = metadata_dir / split_dir / f"{stem}.csv"
            csv_path.write_text(
                "1,0,0,0,0,200\n"
                f"{n_label_frames // 2},5,0,30,5,150\n"
                f"{n_label_frames - 1},2,0,-45,-10,250\n",
                encoding="utf-8",
            )
    return audio_dir, metadata_dir


# ---------------------------------------------------------------------------
# discover_clips
# ---------------------------------------------------------------------------


def test_discover_clips_train(tmp_path: Path):
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path)
    clips = discover_clips(metadata_dir, audio_dir, split="train")
    names = sorted(c.name for c in clips)
    assert names == [
        "fold3_room1_mix001",  # sony 1
        "fold3_room1_mix001",  # tau 1
        "fold3_room1_mix002",  # sony 2
    ]
    splits = {c.split for c in clips}
    assert splits == {"train"}


def test_discover_clips_test(tmp_path: Path):
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path)
    clips = discover_clips(metadata_dir, audio_dir, split="test")
    splits = {c.split for c in clips}
    locs = {c.location for c in clips}
    assert splits == {"test"}
    assert locs == {"sony", "tau"}
    assert len(clips) == 2


def test_discover_clips_all(tmp_path: Path):
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path)
    clips = discover_clips(metadata_dir, audio_dir, split="all")
    assert len(clips) == 5  # 2 + 1 + 1 + 1


def test_discover_clips_metadata_only(tmp_path: Path):
    _, metadata_dir = _build_synthetic_starss(tmp_path)
    clips = discover_clips(metadata_dir, audio_dir=None, split="train")
    assert all(c.audio_path is None for c in clips)
    assert all(not c.has_audio() for c in clips)


def test_discover_clips_invalid_split_raises(tmp_path: Path):
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path)
    with pytest.raises(ValueError, match="split must be one of"):
        discover_clips(metadata_dir, audio_dir, split="bogus")  # type: ignore[arg-type]


def test_discover_clips_skips_missing_dirs(tmp_path: Path):
    """Don't crash if e.g. dev-test-tau has not been downloaded yet."""
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path)
    # Remove dev-test-tau so it does not exist.
    import shutil

    shutil.rmtree(metadata_dir / "dev-test-tau")
    clips = discover_clips(metadata_dir, audio_dir, split="test")
    locs = {c.location for c in clips}
    assert locs == {"sony"}


# ---------------------------------------------------------------------------
# _config_hash stability
# ---------------------------------------------------------------------------


def test_config_hash_deterministic():
    cfg = SeldFeatureConfig()
    assert _config_hash(cfg) == _config_hash(cfg)


def test_config_hash_changes_with_config():
    h1 = _config_hash(SeldFeatureConfig(n_mels=64))
    h2 = _config_hash(SeldFeatureConfig(n_mels=128, n_gcc_lags=128))
    assert h1 != h2


# ---------------------------------------------------------------------------
# StarssDataset basic behaviour
# ---------------------------------------------------------------------------


def test_dataset_len_matches_discovery(tmp_path: Path):
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path)
    ds = StarssDataset(
        audio_dir, metadata_dir, split="train", clip_seconds=2.0, random_crop=False
    )
    assert len(ds) == 3


def test_dataset_getitem_shape(tmp_path: Path):
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path)
    cfg = SeldFeatureConfig()
    ds = StarssDataset(
        audio_dir,
        metadata_dir,
        split="train",
        feature_config=cfg,
        clip_seconds=2.0,
        random_crop=False,
    )
    sample = ds[0]
    assert isinstance(sample["features"], torch.Tensor)
    assert isinstance(sample["target"], torch.Tensor)
    n_label = int(2.0 / cfg.label_hop_s)
    assert sample["target"].shape == (n_label, NUM_TRACK_DUMMY, NUM_AXIS, NUM_DCASE2023_CLASSES)
    assert sample["features"].shape[0] == cfg.n_feature_channels()
    assert sample["features"].shape[1] == n_label * cfg.feature_per_label_ratio
    assert sample["features"].shape[2] == cfg.n_mels
    assert sample["features"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32


def test_dataset_full_clip_when_clip_seconds_is_none(tmp_path: Path):
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path, duration_s=6.0)
    cfg = SeldFeatureConfig()
    ds = StarssDataset(
        audio_dir, metadata_dir, split="train", feature_config=cfg, clip_seconds=None
    )
    sample = ds[0]
    expected_label = int(6.0 / cfg.label_hop_s)
    assert sample["target"].shape[0] == expected_label
    assert sample["features"].shape[1] == expected_label * cfg.feature_per_label_ratio


def test_dataset_random_crop_varies_across_calls(tmp_path: Path):
    """Same idx, different rng draws -> different features (probabilistically)."""
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path, duration_s=10.0)
    cfg = SeldFeatureConfig()
    ds = StarssDataset(
        audio_dir,
        metadata_dir,
        split="train",
        feature_config=cfg,
        clip_seconds=1.0,
        random_crop=True,
        seed=0,
    )
    a = ds[0]["features"].numpy()
    # Re-draw rng without rebuilding ds: call __getitem__ again.
    different = False
    for _ in range(8):
        b = ds[0]["features"].numpy()
        if not np.array_equal(a, b):
            different = True
            break
    assert different, "random crop never differed across 8 draws"


def test_dataset_short_clip_pads(tmp_path: Path):
    """When the clip is shorter than clip_seconds, pad to fixed length."""
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path, duration_s=2.0)
    cfg = SeldFeatureConfig()
    ds = StarssDataset(
        audio_dir,
        metadata_dir,
        split="train",
        feature_config=cfg,
        clip_seconds=4.0,  # longer than the clip
        random_crop=False,
    )
    sample = ds[0]
    expected_label = int(4.0 / cfg.label_hop_s)
    assert sample["target"].shape[0] == expected_label
    assert sample["features"].shape[1] == expected_label * cfg.feature_per_label_ratio


def test_dataset_caches_features_to_disk(tmp_path: Path):
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path, duration_s=2.0)
    cache = tmp_path / "cache"
    cfg = SeldFeatureConfig()
    ds = StarssDataset(
        audio_dir,
        metadata_dir,
        split="train",
        feature_config=cfg,
        clip_seconds=None,
        cache_dir=cache,
    )
    _ = ds[0]
    cache_files = list(cache.glob("*.npz"))
    assert len(cache_files) == 1
    assert _config_hash(cfg) in cache_files[0].name

    # Second access should hit cache (no error if WAV deleted).
    name = ds.clips[0].name
    (audio_dir / "dev-train-sony" / f"{name}.wav").unlink()
    sample2 = ds[0]
    # We deleted the audio for index 0 -> cache must be the source of features.
    assert sample2["features"].shape[0] == cfg.n_feature_channels()


def test_dataset_missing_audio_raises(tmp_path: Path):
    _, metadata_dir = _build_synthetic_starss(tmp_path)
    ds = StarssDataset(
        audio_dir=None, metadata_dir=metadata_dir, split="train", clip_seconds=2.0
    )
    with pytest.raises(FileNotFoundError, match="audio missing"):
        _ = ds[0]


def test_dataset_filter_drops_clips(tmp_path: Path):
    audio_dir, metadata_dir = _build_synthetic_starss(tmp_path)
    ds = StarssDataset(
        audio_dir, metadata_dir, split="all", clip_seconds=None
    )
    n0 = len(ds)
    ds.filter_(lambda c: c.location == "sony")
    assert len(ds) < n0
    assert all(c.location == "sony" for c in ds.iter_metadata())


# ---------------------------------------------------------------------------
# Real STARSS23 metadata pairing smoke test (no audio required)
# ---------------------------------------------------------------------------


_REAL_METADATA_DIR = Path("D:/ssl-research/data/STARSS23/metadata_dev/metadata_dev")


@pytest.mark.skipif(not _REAL_METADATA_DIR.exists(), reason="real STARSS metadata not on disk")
def test_real_starss23_split_counts():
    """We saw earlier: 90 train (40+50) and 78 test (30+48) clips."""
    train = discover_clips(_REAL_METADATA_DIR, audio_dir=None, split="train")
    test = discover_clips(_REAL_METADATA_DIR, audio_dir=None, split="test")
    assert len(train) == 90
    assert len(test) == 78
    assert {c.location for c in train} == {"sony", "tau"}
    assert {c.location for c in test} == {"sony", "tau"}
    # All train should be fold3, all test fold4 (DCASE convention).
    assert all(c.name.startswith("fold3") for c in train)
    assert all(c.name.startswith("fold4") for c in test)
