"""Multi-source phase-map dataset.

Each sample is a multi-frame phase tensor ``(2, M, F, T)`` paired with a
sparse multi-hot label vector over a discretized azimuth grid (default
72 bins of 5 degrees each). The number of active sources varies between
``min_k`` and ``max_k`` per call.

Soft labels: each true azimuth is encoded with a Gaussian bump on the
class grid (sigma defaults to 5 degrees, matching the bin width). This
lets the BCE loss tolerate small angular discretization noise instead of
penalizing the network for predicting an adjacent class.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

_W2 = Path(__file__).resolve().parent.parent / "week02_classical"
_W3 = Path(__file__).resolve().parent.parent / "week03_cnn_doa"
for p in (_W2, _W3):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from features import phase_features  # noqa: E402

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from multi_source_data import (  # noqa: E402
    sample_distinct_azimuths,
    simulate_freefield_multi,
    simulate_room_multi,
)


@dataclass
class MultiSourceConfig:
    mic_positions: np.ndarray
    n_samples: int = 2000
    fs: int = 16000
    duration: float = 1.0
    n_fft: int = 512
    hop_length: int = 256
    az_grid_deg: tuple[int, int, int] = (-180, 180, 5)  # (start, stop, step)
    min_k: int = 1
    max_k: int = 3
    min_separation_deg: float = 30.0
    snr_range_db: tuple[float, float] = (-5.0, 30.0)
    rt60_range: tuple[float, float] | None = None
    reverb_prob: float = 0.5
    label_sigma_deg: float = 5.0
    seed_base: int = 0
    source_band: tuple[float, float] | None = None
    az_class_weights: np.ndarray | None = field(default=None, repr=False)


def make_grid(start: int, stop: int, step: int) -> np.ndarray:
    return np.arange(start, stop, step, dtype=np.float32)


def soft_label_vector(
    azimuths_deg: np.ndarray, grid: np.ndarray, sigma_deg: float
) -> np.ndarray:
    """Build a multi-hot soft label by superposing Gaussian bumps.

    Output is element-wise capped at 1 so that overlapping sources do not
    push individual class targets above the BCE-supported range.
    """
    label = np.zeros_like(grid, dtype=np.float32)
    for az in azimuths_deg:
        diff = ((grid - float(az) + 180.0) % 360.0) - 180.0
        bump = np.exp(-(diff ** 2) / (2.0 * sigma_deg ** 2)).astype(np.float32)
        label = np.maximum(label, bump)
    return label


def hard_label_vector(azimuths_deg: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Per-source nearest-class one-hot, OR-ed across sources."""
    label = np.zeros_like(grid, dtype=np.float32)
    for az in azimuths_deg:
        diff = ((grid - float(az) + 180.0) % 360.0) - 180.0
        idx = int(np.argmin(np.abs(diff)))
        label[idx] = 1.0
    return label


class MultiSourcePhaseDataset(Dataset):
    """Online generator. Use :func:`precompute_multi_source_dataset` for
    multi-epoch training."""

    def __init__(self, cfg: MultiSourceConfig):
        self.cfg = cfg
        self.grid = make_grid(*cfg.az_grid_deg)
        self.n_classes = len(self.grid)

    def __len__(self) -> int:
        return self.cfg.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed_base + idx)
        K = int(rng.integers(cfg.min_k, cfg.max_k + 1))
        azs = sample_distinct_azimuths(
            rng, K, min_separation_deg=cfg.min_separation_deg
        )
        snr = float(rng.uniform(*cfg.snr_range_db))

        use_reverb = (
            cfg.rt60_range is not None and rng.random() < cfg.reverb_prob
        )
        if use_reverb:
            rt60 = float(rng.uniform(*cfg.rt60_range))
            mix, _ = simulate_room_multi(
                mic_positions=cfg.mic_positions,
                azimuths_deg=azs,
                rt60=rt60,
                fs=cfg.fs,
                duration=cfg.duration,
                snr_db=snr,
                source_band=cfg.source_band,
                seed=cfg.seed_base + idx,
            )
        else:
            mix, _ = simulate_freefield_multi(
                mic_positions=cfg.mic_positions,
                azimuths_deg=azs,
                fs=cfg.fs,
                duration=cfg.duration,
                snr_db=snr,
                source_band=cfg.source_band,
                seed=cfg.seed_base + idx,
            )

        feat = phase_features(
            mix, fs=cfg.fs, n_fft=cfg.n_fft, hop_length=cfg.hop_length
        )  # (2, M, F, T)

        label = soft_label_vector(azs, self.grid, cfg.label_sigma_deg)
        # Ground-truth azimuths padded to ``max_k`` with NaN for collation.
        az_padded = np.full((cfg.max_k,), np.nan, dtype=np.float32)
        az_padded[: len(azs)] = azs

        return (
            torch.from_numpy(feat).float(),
            torch.from_numpy(label).float(),
            torch.from_numpy(az_padded).float(),
        )


def precompute_multi_source_dataset(
    cfg: MultiSourceConfig, desc: str = "precompute"
) -> TensorDataset:
    """Materialize all (feature, label, azimuths) triples into RAM."""
    online = MultiSourcePhaseDataset(cfg)
    n = len(online)
    x0, y0, a0 = online[0]
    feats = torch.zeros((n,) + tuple(x0.shape), dtype=torch.float32)
    labels = torch.zeros((n, len(online.grid)), dtype=torch.float32)
    azs = torch.zeros((n, cfg.max_k), dtype=torch.float32)
    feats[0], labels[0], azs[0] = x0, y0, a0
    t0 = time.time()
    for i in tqdm(range(1, n), desc=desc, mininterval=2.0):
        x, y, a = online[i]
        feats[i] = x
        labels[i] = y
        azs[i] = a
    print(f"[{desc}] generated {n} multi-source samples in {time.time() - t0:.1f}s")
    return TensorDataset(feats, labels, azs)
