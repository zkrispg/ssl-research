"""Wrapper dataset that adds the source-count label and runtime augmentation.

Instead of duplicating the W5 simulator, we precompute the W5 dataset
(features, soft spectrum, padded azimuths) and wrap it with a thin layer
that:

* derives the source count K from the unpadded azimuth array;
* in training mode, applies :func:`aug.channel_rotate` with a uniformly
  random shift (4x effective augmentation for UCA4) and optional
  :func:`aug.spec_augment`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_W5 = Path(__file__).resolve().parent.parent / "week05_multi_source"
if str(_W5) not in sys.path:
    sys.path.insert(0, str(_W5))

from multi_dataset import (  # noqa: E402
    MultiSourceConfig,
    precompute_multi_source_dataset,
)

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from aug import channel_rotate, spec_augment  # noqa: E402


@dataclass
class AugConfig:
    enable_channel_rotate: bool = True
    enable_spec_augment: bool = True
    spec_n_freq_masks: int = 2
    spec_max_freq_mask: int = 20
    spec_n_time_masks: int = 2
    spec_max_time_mask: int = 5


class MultiTaskDataset(Dataset):
    """Wraps a precomputed W5 ``TensorDataset`` with augmentation + count.

    Yields ``(feat, spectrum_label, az_padded, k)`` where ``k`` is the
    integer source count in ``[1, max_k]``.
    """

    def __init__(
        self,
        base: torch.utils.data.TensorDataset,
        n_mics: int,
        n_classes: int,
        max_k: int,
        aug: AugConfig | None = None,
        train_mode: bool = True,
        seed_base: int = 0,
    ) -> None:
        self.base = base
        self.n_mics = n_mics
        self.n_classes = n_classes
        self.max_k = max_k
        self.aug = aug or AugConfig()
        self.train_mode = train_mode
        self.seed_base = seed_base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        feat, label, az_padded = self.base[idx]
        # Source count from the padded azimuth array.
        k = int((~torch.isnan(az_padded)).sum().item())

        if self.train_mode and self.aug.enable_channel_rotate:
            rng = np.random.default_rng(self.seed_base + idx)
            shift = int(rng.integers(0, self.n_mics))
            feat, label = channel_rotate(
                feat, label, shift, self.n_mics, self.n_classes
            )

        if self.train_mode and self.aug.enable_spec_augment:
            rng = np.random.default_rng(self.seed_base + idx + 9999)
            feat = spec_augment(
                feat,
                rng,
                n_freq_masks=self.aug.spec_n_freq_masks,
                max_freq_mask=self.aug.spec_max_freq_mask,
                n_time_masks=self.aug.spec_n_time_masks,
                max_time_mask=self.aug.spec_max_time_mask,
            )

        return feat, label, az_padded, torch.tensor(k - 1, dtype=torch.long)


def build_datasets(
    cfg_train: MultiSourceConfig,
    cfg_val: MultiSourceConfig,
    aug: AugConfig | None = None,
) -> tuple[MultiTaskDataset, MultiTaskDataset]:
    """Precompute the underlying W5 dataset once, return train/val wrappers."""
    train_base = precompute_multi_source_dataset(cfg_train, desc="precompute train")
    val_base = precompute_multi_source_dataset(cfg_val, desc="precompute val  ")
    train_ds = MultiTaskDataset(
        train_base,
        n_mics=cfg_train.mic_positions.shape[0],
        n_classes=int((cfg_train.az_grid_deg[1] - cfg_train.az_grid_deg[0]) // cfg_train.az_grid_deg[2]),
        max_k=cfg_train.max_k,
        aug=aug,
        train_mode=True,
    )
    val_ds = MultiTaskDataset(
        val_base,
        n_mics=cfg_val.mic_positions.shape[0],
        n_classes=int((cfg_val.az_grid_deg[1] - cfg_val.az_grid_deg[0]) // cfg_val.az_grid_deg[2]),
        max_k=cfg_val.max_k,
        aug=None,
        train_mode=False,
    )
    return train_ds, val_ds
