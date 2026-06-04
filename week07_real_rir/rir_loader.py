"""Loader interface for measured (real) multichannel RIRs.

This module provides the data plumbing for stage 2 of W7: plugging an
externally-measured RIR dataset into our evaluation harness. The actual
download and dataset-specific parsing is left to small adapter scripts
in ``rir_datasets/`` (one per dataset such as MIRD, REVERB, ACE).

The shared interface is :class:`RirRecord`: a single multichannel RIR
``h`` of shape ``(M, T_h)`` together with metadata (azimuth, room id,
RT60 if known). Test mixtures are produced by convolving any compatible
dry source with the RIR and summing per-channel signals across K
sources sampled from the bank.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_W1 = Path(__file__).resolve().parent.parent / "week01_gcc_phat"
if str(_W1) not in sys.path:
    sys.path.insert(0, str(_W1))

from simulate import make_source  # noqa: E402


@dataclass
class RirRecord:
    """One measured multichannel impulse response."""

    h: np.ndarray  # shape (M, T_h), float32
    azimuth_deg: float
    room_id: str
    rt60: float | None = None
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if self.h.ndim != 2:
            raise ValueError(f"RIR must be (M, T); got shape {self.h.shape}")
        self.h = self.h.astype(np.float32)


def convolve_multichannel(source: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve a 1-D source with an ``(M, T_h)`` RIR. Returns ``(M, N+T_h-1)``."""
    M, T_h = h.shape
    out = np.zeros((M, source.shape[-1] + T_h - 1), dtype=np.float32)
    src_fft_len = source.shape[-1] + T_h - 1
    src_fft = np.fft.rfft(source, n=src_fft_len)
    for m in range(M):
        h_fft = np.fft.rfft(h[m], n=src_fft_len)
        out[m] = np.fft.irfft(src_fft * h_fft, n=src_fft_len).astype(np.float32)
    return out


def synthesize_real_rir_mixture(
    rir_records: list[RirRecord],
    duration: float = 1.0,
    fs: int = 16000,
    snr_db: float = 30.0,
    source_band: tuple[float, float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Build a K-source mixture by convolving dry sources with measured RIRs.

    Args:
        rir_records: List of :class:`RirRecord` (one per source).
        duration, fs: Output length.
        snr_db: SNR of additive Gaussian noise.
        source_band: Bandwidth of synthesized dry source.
        seed: RNG seed.

    Returns:
        ``(signals, azimuths_deg)`` where ``signals`` is ``(M, N)`` and
        ``azimuths_deg`` is the per-source ground truth.
    """
    rng = np.random.default_rng(seed + 1)
    n = int(duration * fs)
    azimuths = []
    M = rir_records[0].h.shape[0]
    mix = np.zeros((M, n), dtype=np.float32)
    for k, rec in enumerate(rir_records):
        if rec.h.shape[0] != M:
            raise ValueError(
                f"All RIRs must share the same M; got {rec.h.shape[0]} vs {M}"
            )
        src = make_source(duration, fs, band=source_band, seed=seed + 1000 * (k + 1))
        conv = convolve_multichannel(src, rec.h)
        if conv.shape[1] >= n:
            conv = conv[:, :n]
        else:
            pad = np.zeros((M, n - conv.shape[1]), dtype=np.float32)
            conv = np.concatenate([conv, pad], axis=1)
        mix += conv
        azimuths.append(rec.azimuth_deg)

    sig_power = float(np.mean(mix[0] ** 2))
    if sig_power > 0:
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        mix = mix + rng.standard_normal(mix.shape).astype(np.float32) * float(np.sqrt(noise_power))
    return mix, azimuths


def sample_distinct_rirs(
    rng: np.random.Generator,
    bank: list[RirRecord],
    n_sources: int,
    min_separation_deg: float = 30.0,
    max_tries: int = 200,
) -> list[RirRecord]:
    """Pick ``n_sources`` RIRs whose azimuths are mutually well-separated."""
    if n_sources > len(bank):
        raise ValueError(f"Bank has only {len(bank)} RIRs; need {n_sources}")
    chosen: list[RirRecord] = []
    indices: set[int] = set()
    for _ in range(max_tries):
        if len(chosen) == n_sources:
            return chosen
        idx = int(rng.integers(0, len(bank)))
        if idx in indices:
            continue
        cand = bank[idx]
        ok = True
        for c in chosen:
            wrap = ((cand.azimuth_deg - c.azimuth_deg + 180.0) % 360.0) - 180.0
            if abs(wrap) < min_separation_deg:
                ok = False
                break
        if ok:
            chosen.append(cand)
            indices.add(idx)
    if len(chosen) == n_sources:
        return chosen
    raise RuntimeError(
        f"Failed to draw {n_sources} well-separated RIRs after {max_tries} tries"
    )
