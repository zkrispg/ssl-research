"""Multi-channel signal simulators.

Two simulators are provided:

1. :func:`simulate_freefield` — anechoic plane-wave model. A single far-field
   source produces signals at each microphone with only the direct-path
   propagation delay. This is the simplest setting and the one used for
   sanity-checking SRP-PHAT and MUSIC.

2. :func:`simulate_room` — a rectangular ("shoebox") room with reverberation,
   built on top of ``pyroomacoustics``. This is closer to the conditions in
   which SSL methods are actually evaluated and is used in W3+ to study
   robustness to RT60.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_W1 = Path(__file__).resolve().parent.parent / "week01_gcc_phat"
if str(_W1) not in sys.path:
    sys.path.insert(0, str(_W1))

from simulate import make_source

SPEED_OF_SOUND = 343.0


def simulate_freefield(
    mic_positions: np.ndarray,
    azimuth_deg: float,
    fs: int = 16000,
    duration: float = 1.0,
    snr_db: float = 30.0,
    source_band: tuple[float, float] | None = (300.0, 3400.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate an M-channel anechoic recording of a far-field source.

    Args:
        mic_positions: ``(M, 2)`` mic coordinates in the xy-plane (meters).
        azimuth_deg: Source azimuth in degrees, measured counter-clockwise
            from +x.
        fs: Sampling rate.
        duration: Length in seconds.
        snr_db: SNR of additive white Gaussian noise (independent per channel).
        source_band: Bandwidth of the source signal as ``(low, high)`` Hz, or
            ``None`` for full-band white noise.
        seed: RNG seed.

    Returns:
        ``(signals, source)`` where ``signals`` has shape ``(M, N)`` and
        ``source`` is the dry source waveform of length ``N``.
    """
    rng = np.random.default_rng(seed + 7919)
    n = int(duration * fs)
    source = make_source(duration, fs, band=source_band, seed=seed)

    SRC = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    theta = np.radians(azimuth_deg)
    u = np.array([np.cos(theta), np.sin(theta)])

    M = mic_positions.shape[0]
    signals = np.zeros((M, n), dtype=np.float32)
    for m in range(M):
        delay = -float(np.dot(mic_positions[m], u)) / SPEED_OF_SOUND
        SIG_m = SRC * np.exp(-1j * 2 * np.pi * freqs * delay)
        signals[m] = np.fft.irfft(SIG_m, n=n).astype(np.float32)

    sig_power = float(np.mean(signals[0] ** 2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = float(np.sqrt(noise_power))
    if noise_std > 0:
        signals = signals + rng.standard_normal(signals.shape).astype(np.float32) * noise_std

    return signals, source


def simulate_room(
    mic_positions: np.ndarray,
    azimuth_deg: float,
    room_dim: tuple[float, float, float] = (6.0, 5.0, 3.0),
    array_center: tuple[float, float, float] | None = None,
    source_distance: float = 1.5,
    rt60: float = 0.3,
    fs: int = 16000,
    duration: float = 1.0,
    snr_db: float = 30.0,
    source_band: tuple[float, float] | None = (300.0, 3400.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a reverberant shoebox room recording.

    The mic array (assumed 2D in the xy-plane) is placed at ``array_center``
    (default = room center, height = half room height). The source is placed
    at ``source_distance`` meters from the array center, on the same horizontal
    plane, in the direction of ``azimuth_deg``.

    The RT60 target is reached via Sabine's formula and the
    ``inverse_sabine`` helper from pyroomacoustics; the actual achieved
    RT60 may differ slightly.
    """
    import pyroomacoustics as pra

    rng = np.random.default_rng(seed + 7919)
    source = make_source(duration, fs, band=source_band, seed=seed)

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

    theta = np.radians(azimuth_deg)
    src_pos = np.array(
        [
            cx + source_distance * np.cos(theta),
            cy + source_distance * np.sin(theta),
            cz,
        ]
    )
    room.add_source(src_pos, signal=source)
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

    return signals, source
