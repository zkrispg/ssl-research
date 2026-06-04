"""On-the-fly synthetic phase-map dataset for DOA estimation.

A single sample is a one-frame phase map ``(2, M, F)`` paired with an integer
class label corresponding to the source azimuth bin. Each call to
``__getitem__`` simulates a new free-field recording at a random azimuth
drawn from a uniform grid, with random SNR drawn from a configurable range,
and returns the sin/cos phase features for one randomly-chosen STFT frame.

This follows the Chakrabarty & Habets (2019) recipe of training on noise
signals: each sample uses a different noise realization, so the network
sees an essentially unlimited variety of patterns.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_W2 = Path(__file__).resolve().parent.parent / "week02_classical"
if str(_W2) not in sys.path:
    sys.path.insert(0, str(_W2))

from simulate_array import simulate_freefield, simulate_room  # noqa: E402

_W3 = Path(__file__).resolve().parent
if str(_W3) not in sys.path:
    sys.path.insert(0, str(_W3))

from features import phase_features, random_frame  # noqa: E402

import time
from tqdm import tqdm
from torch.utils.data import TensorDataset


@dataclass
class DatasetConfig:
    mic_positions: np.ndarray
    n_samples: int = 20000
    fs: int = 16000
    duration: float = 0.5
    n_fft: int = 512
    hop_length: int = 256
    azimuth_grid_deg: tuple[int, int, int] = (-180, 180, 5)  # (start, stop, step)
    snr_range_db: tuple[float, float] = (0.0, 30.0)
    rt60_range: tuple[float, float] | None = None  # if set, half samples reverb
    seed_base: int = 0
    source_band: tuple[float, float] | None = None  # None = full-band noise


def azimuth_classes(start: int, stop: int, step: int) -> np.ndarray:
    """Discrete azimuth grid as a 1-D numpy array of degrees."""
    return np.arange(start, stop, step, dtype=np.float32)


def class_to_azimuth(cls_idx: int, grid: np.ndarray) -> float:
    return float(grid[cls_idx])


def azimuth_to_class(az_deg: float, grid: np.ndarray) -> int:
    """Snap a continuous azimuth to its nearest class on the wrapped grid."""
    diff = ((grid - az_deg + 180.0) % 360.0) - 180.0
    return int(np.argmin(np.abs(diff)))


class PhaseMapDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.grid = azimuth_classes(*cfg.azimuth_grid_deg)
        self.n_classes = len(self.grid)

    def __len__(self) -> int:
        return self.cfg.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed_base + idx)

        cls_idx = int(rng.integers(0, self.n_classes))
        az = float(self.grid[cls_idx])
        snr = float(rng.uniform(*cfg.snr_range_db))

        use_reverb = cfg.rt60_range is not None and rng.random() < 0.5
        if use_reverb:
            rt60 = float(rng.uniform(*cfg.rt60_range))
            signals, _ = simulate_room(
                mic_positions=cfg.mic_positions,
                azimuth_deg=az,
                rt60=rt60,
                fs=cfg.fs,
                duration=cfg.duration,
                snr_db=snr,
                source_band=cfg.source_band,
                seed=cfg.seed_base + idx,
            )
        else:
            signals, _ = simulate_freefield(
                mic_positions=cfg.mic_positions,
                azimuth_deg=az,
                fs=cfg.fs,
                duration=cfg.duration,
                snr_db=snr,
                source_band=cfg.source_band,
                seed=cfg.seed_base + idx,
            )

        feat = phase_features(
            signals,
            fs=cfg.fs,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        frame = random_frame(feat, rng)  # (2, M, F)

        return torch.from_numpy(frame).float(), torch.tensor(cls_idx, dtype=torch.long)


def precompute_phase_dataset(cfg: DatasetConfig, desc: str = "precompute") -> TensorDataset:
    """Materialize all (feature, label) pairs of a PhaseMapDataset into RAM.

    Online generation is the right design for ``rt60_range > 0`` because each
    pyroomacoustics call is slow; pre-computing once and re-using the same
    features across epochs cuts CPU wall time by an order of magnitude when
    ``epochs > 1``.
    """
    online = PhaseMapDataset(cfg)
    n = len(online)
    feats = torch.zeros(
        (n, 2, cfg.mic_positions.shape[0], cfg.n_fft // 2 + 1),
        dtype=torch.float32,
    )
    labels = torch.zeros(n, dtype=torch.long)
    t0 = time.time()
    for i in tqdm(range(n), desc=desc, mininterval=2.0):
        x, y = online[i]
        feats[i] = x
        labels[i] = y
    print(f"[{desc}] generated {n} samples in {time.time() - t0:.1f}s")
    return TensorDataset(feats, labels)
