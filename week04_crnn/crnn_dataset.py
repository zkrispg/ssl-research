"""Multi-frame phase-map dataset for the W4 CRNN.

Each sample is a full ``(2, M, F, T)`` STFT phase tensor (sin/cos of phase)
together with the source azimuth in radians. Compared to W3's single-frame
classification, this lets the GRU integrate phase information across time,
which is crucial for reverberant conditions where the direct path's phase
is corrupted in any single frame but recoverable from many.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

_W2 = Path(__file__).resolve().parent.parent / "week02_classical"
if str(_W2) not in sys.path:
    sys.path.insert(0, str(_W2))
_W3 = Path(__file__).resolve().parent.parent / "week03_cnn_doa"
if str(_W3) not in sys.path:
    sys.path.insert(0, str(_W3))

from features import phase_features  # noqa: E402
from simulate_array import simulate_freefield, simulate_room  # noqa: E402


@dataclass
class MultiFrameConfig:
    mic_positions: np.ndarray
    n_samples: int = 4000
    fs: int = 16000
    duration: float = 1.0
    n_fft: int = 512
    hop_length: int = 256
    azimuth_range_deg: tuple[float, float] = (-180.0, 180.0)
    snr_range_db: tuple[float, float] = (0.0, 30.0)
    rt60_range: tuple[float, float] | None = None
    reverb_prob: float = 0.5
    seed_base: int = 0
    source_band: tuple[float, float] | None = None  # full-band noise


class MultiFramePhaseDataset(Dataset):
    """Online dataset; prefer :func:`precompute_multi_frame_dataset` to cache
    all features in RAM for fast multi-epoch training."""

    def __init__(self, cfg: MultiFrameConfig):
        self.cfg = cfg

    def __len__(self) -> int:
        return self.cfg.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed_base + idx)
        az_deg = float(rng.uniform(*cfg.azimuth_range_deg))
        snr = float(rng.uniform(*cfg.snr_range_db))

        use_reverb = (
            cfg.rt60_range is not None and rng.random() < cfg.reverb_prob
        )
        if use_reverb:
            rt60 = float(rng.uniform(*cfg.rt60_range))
            signals, _ = simulate_room(
                mic_positions=cfg.mic_positions,
                azimuth_deg=az_deg,
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
                azimuth_deg=az_deg,
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
        )  # (2, M, F, T)
        return torch.from_numpy(feat).float(), torch.tensor(
            np.radians(az_deg), dtype=torch.float32
        )


def precompute_multi_frame_dataset(
    cfg: MultiFrameConfig, desc: str = "precompute"
) -> TensorDataset:
    """Materialize the dataset into RAM. Must be done once for fast training."""
    online = MultiFramePhaseDataset(cfg)
    n = len(online)
    # Determine T from a probe sample
    x0, _ = online[0]
    feats = torch.zeros((n,) + tuple(x0.shape), dtype=torch.float32)
    az_rad = torch.zeros(n, dtype=torch.float32)
    feats[0] = x0
    az_rad[0] = (
        torch.tensor(0.0)
        if False
        else online[0][1]  # already produced; keep as placeholder
    )
    feats[0], az_rad[0] = x0, online[0][1]

    t0 = time.time()
    for i in tqdm(range(n), desc=desc, mininterval=2.0):
        x, y = online[i]
        feats[i] = x
        az_rad[i] = y
    print(f"[{desc}] generated {n} multi-frame samples in {time.time() - t0:.1f}s")
    return TensorDataset(feats, az_rad)


def az_to_xy(az_rad: torch.Tensor) -> torch.Tensor:
    """Convert azimuth in radians to a (B, 2) Cartesian unit vector."""
    return torch.stack([torch.cos(az_rad), torch.sin(az_rad)], dim=-1)


def xy_to_az_deg(xy: torch.Tensor) -> torch.Tensor:
    """Inverse: (..., 2) -> azimuth in degrees in (-180, 180]."""
    return torch.rad2deg(torch.atan2(xy[..., 1], xy[..., 0]))
