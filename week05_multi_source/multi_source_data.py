"""Multi-source signal simulator for SSL.

Each call generates a multi-channel recording with K independent
far-field sources at distinct azimuths, sharing the same UCA. Both an
anechoic free-field model and a reverberant pyroomacoustics shoebox
model are supported.

Source independence is enforced by giving each source its own dry
waveform (different RNG seed). All sources are equal energy by default
to keep the localization problem balanced.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_W1 = Path(__file__).resolve().parent.parent / "week01_gcc_phat"
_W2 = Path(__file__).resolve().parent.parent / "week02_classical"
for p in (_W1, _W2):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from simulate import make_source  # noqa: E402

SPEED_OF_SOUND = 343.0

_SOURCE_GENERATOR = make_source


def set_source_generator(fn) -> None:
    """Override the per-source dry-waveform generator (used in W10 to swap
    band-limited noise for synthetic speech-like signals). The replacement
    must have signature ``(duration, fs, band, seed) -> np.ndarray``.
    """
    global _SOURCE_GENERATOR
    _SOURCE_GENERATOR = fn


def reset_source_generator() -> None:
    global _SOURCE_GENERATOR
    _SOURCE_GENERATOR = make_source


def sample_distinct_azimuths(
    rng: np.random.Generator,
    n_sources: int,
    az_range_deg: tuple[float, float] = (-180.0, 180.0),
    min_separation_deg: float = 30.0,
    max_tries: int = 200,
) -> np.ndarray:
    """Sample ``n_sources`` distinct azimuths with a minimum angular gap.

    Falls back to a uniform-grid placement if rejection sampling fails,
    so the function never raises in practice.
    """
    azs: list[float] = []
    for _ in range(max_tries):
        if len(azs) == n_sources:
            return np.asarray(azs, dtype=np.float32)
        cand = float(rng.uniform(*az_range_deg))
        ok = True
        for a in azs:
            wrap = ((cand - a + 180.0) % 360.0) - 180.0
            if abs(wrap) < min_separation_deg:
                ok = False
                break
        if ok:
            azs.append(cand)
    # Uniform fallback covers the worst case where rejection failed.
    step = 360.0 / max(n_sources, 1)
    base = float(rng.uniform(-180.0, 180.0))
    return np.asarray(
        [((base + i * step + 180.0) % 360.0) - 180.0 for i in range(n_sources)],
        dtype=np.float32,
    )


def _freefield_one_source(
    mic_positions: np.ndarray,
    azimuth_deg: float,
    source: np.ndarray,
    fs: int,
) -> np.ndarray:
    """Apply per-mic propagation delays of a single far-field source."""
    n = source.shape[-1]
    SRC = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    theta = np.radians(azimuth_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    M = mic_positions.shape[0]
    out = np.zeros((M, n), dtype=np.float32)
    for m in range(M):
        delay = -float(np.dot(mic_positions[m], u)) / SPEED_OF_SOUND
        SIG_m = SRC * np.exp(-1j * 2 * np.pi * freqs * delay)
        out[m] = np.fft.irfft(SIG_m, n=n).astype(np.float32)
    return out


def simulate_freefield_multi(
    mic_positions: np.ndarray,
    azimuths_deg: np.ndarray,
    fs: int = 16000,
    duration: float = 1.0,
    snr_db: float = 30.0,
    source_band: tuple[float, float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulate K-source free-field recording.

    Returns ``(signals, dry_sources)`` where ``signals`` is ``(M, N)`` and
    ``dry_sources`` is a list of length ``len(azimuths_deg)`` containing
    the per-source dry waveforms.
    """
    rng = np.random.default_rng(seed + 7919)
    K = len(azimuths_deg)
    sources = [
        _SOURCE_GENERATOR(duration, fs, band=source_band, seed=seed + 1000 * (k + 1))
        for k in range(K)
    ]
    M = mic_positions.shape[0]
    n = int(duration * fs)
    mix = np.zeros((M, n), dtype=np.float32)
    for src, az in zip(sources, azimuths_deg):
        mix += _freefield_one_source(mic_positions, float(az), src, fs)

    sig_power = float(np.mean(mix[0] ** 2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = float(np.sqrt(noise_power))
    if noise_std > 0:
        mix = mix + rng.standard_normal(mix.shape).astype(np.float32) * noise_std
    return mix, sources


def simulate_room_multi(
    mic_positions: np.ndarray,
    azimuths_deg: np.ndarray,
    rt60: float = 0.3,
    room_dim: tuple[float, float, float] = (6.0, 5.0, 3.0),
    array_center: tuple[float, float, float] | None = None,
    source_distance: float = 1.5,
    fs: int = 16000,
    duration: float = 1.0,
    snr_db: float = 30.0,
    source_band: tuple[float, float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """K-source reverberant shoebox simulation via pyroomacoustics."""
    import pyroomacoustics as pra

    rng = np.random.default_rng(seed + 7919)
    K = len(azimuths_deg)
    sources = [
        _SOURCE_GENERATOR(duration, fs, band=source_band, seed=seed + 1000 * (k + 1))
        for k in range(K)
    ]

    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        list(room_dim),
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        air_absorption=False,
    )

    if array_center is None:
        array_center = (room_dim[0] / 2, room_dim[1] / 2, room_dim[2] / 2)
    cx, cy, cz = array_center

    mic_positions_3d = np.zeros((mic_positions.shape[0], 3))
    mic_positions_3d[:, 0] = mic_positions[:, 0] + cx
    mic_positions_3d[:, 1] = mic_positions[:, 1] + cy
    mic_positions_3d[:, 2] = cz
    room.add_microphone_array(mic_positions_3d.T)

    for src, az_deg in zip(sources, azimuths_deg):
        theta = np.radians(float(az_deg))
        pos = np.array(
            [
                cx + source_distance * np.cos(theta),
                cy + source_distance * np.sin(theta),
                cz,
            ]
        )
        room.add_source(pos, signal=src)
    room.simulate()

    signals = room.mic_array.signals.astype(np.float32)
    n = int(duration * fs)
    if signals.shape[1] >= n:
        signals = signals[:, :n]
    else:
        pad = np.zeros((signals.shape[0], n - signals.shape[1]), dtype=np.float32)
        signals = np.concatenate([signals, pad], axis=1)

    sig_power = float(np.mean(signals[0] ** 2))
    if sig_power > 0:
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise_std = float(np.sqrt(noise_power))
        signals = signals + rng.standard_normal(signals.shape).astype(np.float32) * noise_std
    return signals, sources
