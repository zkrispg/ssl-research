"""Diverse out-of-distribution acoustic simulation for generalization tests.

This module extends the W5 multi-source simulator with much wider sampling
ranges over every acoustic factor of variation:

* Room dimensions: random ``(4-10, 4-8, 2.5-4)`` meters
* RT60: ``0.2-1.0`` seconds (training was capped at 0.5)
* Source-to-array distance: ``0.5-3`` meters (training was 1.5 m fixed)
* Array center offset from room center: random ``+/- 1`` m in x and y
* Optional per-source level variation: ``+/-6`` dB

The microphone array geometry (UCA4, radius 4 cm) is kept identical to
W5/W6 training so that the W6 model can be evaluated zero-shot under
these new conditions; only the acoustic context changes.

The result is a benchmark where every test sample comes from an
acoustic environment unseen during training -- exactly the setting that
makes a sim-trained model's generalization measurable.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_W1 = Path(__file__).resolve().parent.parent / "week01_gcc_phat"
_W2 = Path(__file__).resolve().parent.parent / "week02_classical"
_W5 = Path(__file__).resolve().parent.parent / "week05_multi_source"
for p in (_W1, _W2, _W5):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from multi_source_data import sample_distinct_azimuths  # noqa: E402
from simulate import make_source  # noqa: E402

SPEED_OF_SOUND = 343.0


@dataclass
class DiverseRoomSampler:
    """Random acoustic-environment sampler used by :func:`simulate_diverse`."""

    room_x_range: tuple[float, float] = (4.0, 10.0)
    room_y_range: tuple[float, float] = (4.0, 8.0)
    room_z_range: tuple[float, float] = (2.5, 4.0)
    rt60_range: tuple[float, float] = (0.20, 1.0)
    distance_range: tuple[float, float] = (0.5, 3.0)
    array_offset_range: tuple[float, float] = (-1.0, 1.0)
    array_height_range: tuple[float, float] = (1.0, 2.0)
    per_source_level_db: float = 0.0  # set > 0 to randomize per-source gain
    seed_base: int = 0

    def sample(self, rng: np.random.Generator) -> dict:
        room_dim = (
            float(rng.uniform(*self.room_x_range)),
            float(rng.uniform(*self.room_y_range)),
            float(rng.uniform(*self.room_z_range)),
        )
        rt60 = float(rng.uniform(*self.rt60_range))
        # Make sure the array stays inside the room with some margin.
        max_off_x = max(0.0, room_dim[0] / 2.0 - 0.3)
        max_off_y = max(0.0, room_dim[1] / 2.0 - 0.3)
        off_x = float(np.clip(rng.uniform(*self.array_offset_range), -max_off_x, max_off_x))
        off_y = float(np.clip(rng.uniform(*self.array_offset_range), -max_off_y, max_off_y))
        array_height = float(rng.uniform(*self.array_height_range))
        array_height = float(np.clip(array_height, 0.5, room_dim[2] - 0.5))
        array_center = (
            room_dim[0] / 2 + off_x,
            room_dim[1] / 2 + off_y,
            array_height,
        )
        distance = float(rng.uniform(*self.distance_range))
        return {
            "room_dim": room_dim,
            "rt60": rt60,
            "array_center": array_center,
            "source_distance": distance,
        }


def simulate_diverse(
    mic_positions: np.ndarray,
    azimuths_deg: np.ndarray,
    sampler: DiverseRoomSampler,
    fs: int = 16000,
    duration: float = 1.0,
    snr_db: float = 30.0,
    source_band: tuple[float, float] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """One reverberant K-source mixture with random room/distance/etc.

    Returns ``(signals, info)`` where ``info`` contains the sampled
    acoustic parameters so they can be logged or used for stratified
    evaluation later.
    """
    import pyroomacoustics as pra

    rng = np.random.default_rng(seed + 13)
    params = sampler.sample(rng)
    room_dim = params["room_dim"]
    rt60 = params["rt60"]
    array_center = params["array_center"]
    distance = params["source_distance"]

    # Try the requested RT60; if pyroomacoustics rejects it (room too
    # small for that absorption), retry with a slightly larger RT60.
    e_absorption = max_order = None
    for _ in range(5):
        try:
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            break
        except ValueError:
            rt60 = min(rt60 * 1.3, 1.5)
    if e_absorption is None:
        raise RuntimeError(f"pyroomacoustics rejected RT60 search for {room_dim}")

    room = pra.ShoeBox(
        list(room_dim),
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
        air_absorption=False,
    )

    cx, cy, cz = array_center
    mic_3d = np.zeros((mic_positions.shape[0], 3))
    mic_3d[:, 0] = mic_positions[:, 0] + cx
    mic_3d[:, 1] = mic_positions[:, 1] + cy
    mic_3d[:, 2] = cz
    room.add_microphone_array(mic_3d.T)

    K = len(azimuths_deg)
    sources = [
        make_source(duration, fs, band=source_band, seed=seed + 1000 * (k + 1))
        for k in range(K)
    ]

    if sampler.per_source_level_db > 0:
        gains = 10.0 ** (
            rng.uniform(-sampler.per_source_level_db, sampler.per_source_level_db, K) / 20.0
        )
    else:
        gains = np.ones(K, dtype=np.float64)

    for src, az_deg, gain in zip(sources, azimuths_deg, gains):
        theta = np.radians(float(az_deg))
        pos = np.array(
            [
                cx + distance * np.cos(theta),
                cy + distance * np.sin(theta),
                cz,
            ]
        )
        # Make sure the source is inside the room.
        pos[0] = float(np.clip(pos[0], 0.2, room_dim[0] - 0.2))
        pos[1] = float(np.clip(pos[1], 0.2, room_dim[1] - 0.2))
        room.add_source(pos, signal=(src * gain).astype(np.float32))
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
        signals = signals + (
            rng.standard_normal(signals.shape).astype(np.float32) * noise_std
        )

    info = dict(params)
    info["snr_db"] = snr_db
    info["azimuths_deg"] = list(map(float, azimuths_deg))
    info["per_source_gain"] = list(map(float, gains))
    return signals, info
