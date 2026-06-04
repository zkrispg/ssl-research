"""PyTorch Dataset for STARSS23 (DCASE 2023/2024 Task 3 SELD).

Directory layout expected on disk:

    <metadata_dir>/
        dev-train-sony/fold3_room*_mix*.csv
        dev-train-tau/fold3_room*_mix*.csv
        dev-test-sony/fold4_room*_mix*.csv
        dev-test-tau/fold4_room*_mix*.csv

    <audio_dir>/
        dev-train-sony/fold3_room*_mix*.wav
        ...

The CSV ``stem`` matches the WAV ``stem`` exactly. We use the parent
directory name to determine the split (``train`` vs ``test``) and the
recording location (``sony`` vs ``tau``).

For training we typically crop a random ``clip_seconds`` window per
sample; for evaluation we return the full clip. Feature extraction is
expensive (~ a few seconds per minute of audio), so we support an
optional on-disk ``.npz`` cache keyed by clip name *and* a hash of the
feature configuration.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from week11_starss23.seld_features import (
    SeldFeatureConfig,
    crop_features_to_label_frames,
    extract_seld_features,
    load_multichannel_audio,
    num_label_frames,
)
from week11_starss23.seld_labels import (
    NUM_AXIS,
    NUM_DCASE2023_CLASSES,
    NUM_TRACK_DUMMY,
    events_to_multi_accdoa,
    parse_starss_csv,
)

Split = Literal["train", "test", "all"]
_SPLIT_DIRS: dict[Split, tuple[str, ...]] = {
    "train": ("dev-train-sony", "dev-train-tau"),
    "test": ("dev-test-sony", "dev-test-tau"),
    "all": (
        "dev-train-sony",
        "dev-train-tau",
        "dev-test-sony",
        "dev-test-tau",
    ),
}


@dataclass(frozen=True)
class StarssClipMeta:
    """Lightweight pointer to a clip without loading audio or labels."""

    name: str  # e.g. "fold4_room23_mix001"
    csv_path: Path
    audio_path: Path | None  # None if audio_dir was not supplied
    split: str  # "train" | "test"
    location: str  # "sony" | "tau"

    def has_audio(self) -> bool:
        return self.audio_path is not None and self.audio_path.exists()


def _split_and_loc_from_dirname(dirname: str) -> tuple[str, str]:
    """``"dev-train-sony"`` -> ``("train", "sony")``."""
    parts = dirname.split("-")
    if len(parts) < 3 or parts[0] != "dev":
        raise ValueError(f"unexpected metadata dir name {dirname!r}")
    return parts[1], parts[2]


def discover_clips(
    metadata_dir: str | Path,
    audio_dir: str | Path | None,
    split: Split = "train",
) -> list[StarssClipMeta]:
    """Enumerate clips for the requested split.

    Args:
        metadata_dir: parent of ``dev-train-sony/`` etc.
        audio_dir: parent of ``dev-train-sony/`` etc., or ``None`` if audio
            is not yet downloaded (in which case ``audio_path`` will be
            ``None`` and ``__getitem__`` will fail).
        split: ``"train"``, ``"test"``, or ``"all"``.
    """
    metadata_dir = Path(metadata_dir)
    audio_dir = Path(audio_dir) if audio_dir is not None else None
    if split not in _SPLIT_DIRS:
        raise ValueError(f"split must be one of {list(_SPLIT_DIRS)}, got {split!r}")

    out: list[StarssClipMeta] = []
    for dirname in _SPLIT_DIRS[split]:
        meta_subdir = metadata_dir / dirname
        if not meta_subdir.is_dir():
            continue
        s, loc = _split_and_loc_from_dirname(dirname)
        for csv in sorted(meta_subdir.glob("*.csv")):
            audio_path: Path | None = None
            if audio_dir is not None:
                audio_path = audio_dir / dirname / f"{csv.stem}.wav"
            out.append(
                StarssClipMeta(
                    name=csv.stem,
                    csv_path=csv,
                    audio_path=audio_path,
                    split=s,
                    location=loc,
                )
            )
    return out


def _config_hash(cfg: SeldFeatureConfig) -> str:
    blob = json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:10]


class StarssDataset(Dataset):
    """STARSS23 SELD dataset returning features + Multi-ACCDOA targets.

    Each sample is a ``dict``:

        {
            "features": Tensor (n_ch, T_features, n_mels),
            "target":   Tensor (n_label_frames, 6, 4, n_classes),
            "name":     str,
            "split":    str,
            "location": str,
        }

    Args:
        audio_dir: parent directory of ``dev-{train,test}-{sony,tau}/``
            containing 4-channel WAV files. Pass ``None`` for a
            metadata-only dataset (useful for unit-testing label paths).
        metadata_dir: parent directory of ``dev-{train,test}-{sony,tau}/``
            containing the CSV annotations.
        split: which clips to include.
        feature_config: feature extraction hyper-parameters.
        clip_seconds: random crop window in seconds. Set to ``None`` to
            return the full clip (use this for evaluation).
        random_crop: if True, crop is randomised; if False, takes
            ``[0, clip_seconds]``. Ignored when ``clip_seconds is None``.
        cache_dir: if given, extracted features are persisted to
            ``cache_dir/{name}_{cfg_hash}.npz`` and reloaded on subsequent
            calls.
        num_classes: class-axis size of the target tensor.
        seed: torch.Generator seed used for random cropping.
        crops_per_clip: how many independent random crops to yield per clip
            per epoch. ``__len__`` becomes ``n_clips * crops_per_clip``;
            each ``__getitem__`` call still performs a fresh random crop
            (the duplicates are not cached). Useful when ``random_crop``
            is True and clips are much longer than ``clip_seconds`` so the
            single-crop epoch undersamples the data. Has no effect when
            ``clip_seconds is None`` (full-clip mode).
        in_memory: if True, decoded ``(features, n_label_frames)`` are
            cached in a process-local dict the first time each clip is
            accessed. This makes multi-crop / multi-epoch training much
            faster at the cost of RAM (~50 MB per clip).
    """

    def __init__(
        self,
        audio_dir: str | Path | None,
        metadata_dir: str | Path,
        *,
        split: Split = "train",
        feature_config: SeldFeatureConfig | None = None,
        clip_seconds: float | None = 5.0,
        random_crop: bool = True,
        cache_dir: str | Path | None = None,
        num_classes: int = NUM_DCASE2023_CLASSES,
        seed: int = 0,
        crops_per_clip: int = 1,
        in_memory: bool = False,
    ) -> None:
        self.cfg = feature_config or SeldFeatureConfig()
        self.clip_seconds = clip_seconds
        self.random_crop = random_crop
        self.num_classes = num_classes
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rng = np.random.default_rng(seed)
        if crops_per_clip < 1:
            raise ValueError(f"crops_per_clip must be >= 1, got {crops_per_clip}")
        # Multi-crop only makes sense when we actually crop and randomise.
        if crops_per_clip > 1 and (clip_seconds is None or not random_crop):
            crops_per_clip = 1
        self.crops_per_clip = crops_per_clip
        self.in_memory = in_memory
        self._memcache: dict[str, tuple[np.ndarray, int]] = {}
        self.clips: list[StarssClipMeta] = discover_clips(
            metadata_dir=metadata_dir, audio_dir=audio_dir, split=split
        )
        if not self.clips:
            warnings.warn(f"no clips found for split={split} under {metadata_dir}")

    # -- standard Dataset interface ------------------------------------------------

    def __len__(self) -> int:
        return len(self.clips) * self.crops_per_clip

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx % len(self.clips)]
        feature, label_frames_total = self._extract_features_for_clip(clip)
        events = parse_starss_csv(clip.csv_path)
        target_full = events_to_multi_accdoa(
            events, num_frames=label_frames_total, num_classes=self.num_classes
        )

        # Now align features and target to the (possibly cropped) window.
        feature, target = self._maybe_crop(feature, target_full)
        return {
            "features": torch.from_numpy(np.ascontiguousarray(feature)),
            "target": target,
            "name": clip.name,
            "split": clip.split,
            "location": clip.location,
        }

    # -- helpers ------------------------------------------------------------------

    def _extract_features_for_clip(
        self, clip: StarssClipMeta
    ) -> tuple[np.ndarray, int]:
        """Return ``(features, n_label_frames_total)``; cached on disk and (optionally) in RAM."""
        if self.in_memory and clip.name in self._memcache:
            return self._memcache[clip.name]

        cache_path: Path | None = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{clip.name}_{_config_hash(self.cfg)}.npz"
            if cache_path.exists():
                with np.load(cache_path) as nz:
                    out = (nz["features"], int(nz["n_label_frames"]))
                if self.in_memory:
                    self._memcache[clip.name] = out
                return out

        if not clip.has_audio():
            raise FileNotFoundError(
                f"audio missing for {clip.name}: "
                f"{clip.audio_path}. Did you forget to extract mic_dev.zip?"
            )
        audio = load_multichannel_audio(
            clip.audio_path, target_fs=self.cfg.fs, n_mics=self.cfg.n_mics
        )
        n_label_frames_total = num_label_frames(audio.shape[1], self.cfg)
        features = extract_seld_features(audio, self.cfg)
        features = crop_features_to_label_frames(features, n_label_frames_total, self.cfg)

        if cache_path is not None:
            np.savez_compressed(
                cache_path, features=features, n_label_frames=np.int32(n_label_frames_total)
            )
        out = (features, n_label_frames_total)
        if self.in_memory:
            self._memcache[clip.name] = out
        return out

    def _maybe_crop(
        self, features: np.ndarray, target: torch.Tensor
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Apply random / fixed crop to (features, target) consistently."""
        if self.clip_seconds is None:
            return features, target

        n_label_total = target.shape[0]
        clip_label_frames = int(round(self.clip_seconds / self.cfg.label_hop_s))
        if clip_label_frames <= 0:
            raise ValueError(f"clip_seconds={self.clip_seconds} too short")

        if n_label_total <= clip_label_frames:
            # Pad zero-events at end so downstream sees expected shape.
            return self._pad_to_clip_length(features, target, clip_label_frames)

        max_start_label = n_label_total - clip_label_frames
        if self.random_crop:
            start_label = int(self._rng.integers(0, max_start_label + 1))
        else:
            start_label = 0
        ratio = self.cfg.feature_per_label_ratio
        start_feat = start_label * ratio
        end_feat = start_feat + clip_label_frames * ratio
        end_label = start_label + clip_label_frames
        return (
            features[:, start_feat:end_feat, :],
            target[start_label:end_label],
        )

    def _pad_to_clip_length(
        self,
        features: np.ndarray,
        target: torch.Tensor,
        clip_label_frames: int,
    ) -> tuple[np.ndarray, torch.Tensor]:
        n_label_total = target.shape[0]
        pad_label = clip_label_frames - n_label_total
        ratio = self.cfg.feature_per_label_ratio
        pad_feat = pad_label * ratio
        n_ch = features.shape[0]
        n_mics = self.cfg.n_mics
        pad_feat_arr = np.concatenate(
            [
                np.full((n_mics, pad_feat, features.shape[2]), np.log(1e-8), dtype=features.dtype),
                np.zeros((n_ch - n_mics, pad_feat, features.shape[2]), dtype=features.dtype),
            ],
            axis=0,
        )
        features_padded = np.concatenate([features, pad_feat_arr], axis=1)
        target_padded = torch.cat(
            [target, torch.zeros((pad_label, NUM_TRACK_DUMMY, NUM_AXIS, target.shape[-1]))],
            dim=0,
        )
        return features_padded, target_padded

    # -- introspection helpers (handy for tests / debug) --------------------------

    def iter_metadata(self) -> Iterator[StarssClipMeta]:
        return iter(self.clips)

    def filter_(self, predicate) -> "StarssDataset":
        """Mutating ``filter`` returning ``self`` for chaining."""
        self.clips = [c for c in self.clips if predicate(c)]
        return self
