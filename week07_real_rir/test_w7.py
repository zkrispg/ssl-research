"""Sanity tests for W7 OOD simulator and RIR loader."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))
sys.path.insert(0, str(Path(__file__).parent.parent / "week05_multi_source"))

from diverse_simulator import DiverseRoomSampler, simulate_diverse
from geometry import uniform_circular_array
from multi_source_data import sample_distinct_azimuths
from rir_loader import (
    RirRecord,
    convolve_multichannel,
    sample_distinct_rirs,
    synthesize_real_rir_mixture,
)


def test_diverse_sampler_ranges():
    sampler = DiverseRoomSampler(seed_base=0)
    rng = np.random.default_rng(0)
    sizes = []
    rt60s = []
    distances = []
    for _ in range(50):
        params = sampler.sample(rng)
        sizes.append(params["room_dim"])
        rt60s.append(params["rt60"])
        distances.append(params["source_distance"])

    sizes_arr = np.asarray(sizes)
    assert sizes_arr[:, 0].min() >= 4.0 - 1e-6
    assert sizes_arr[:, 0].max() <= 10.0 + 1e-6
    assert sizes_arr[:, 1].min() >= 4.0 - 1e-6
    assert sizes_arr[:, 2].min() >= 2.5 - 1e-6
    assert min(rt60s) >= 0.20 - 1e-6
    assert max(rt60s) <= 1.0 + 1e-6
    assert min(distances) >= 0.5 - 1e-6
    assert max(distances) <= 3.0 + 1e-6


def test_simulate_diverse_runs():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    sampler = DiverseRoomSampler()
    rng = np.random.default_rng(0)
    azs = sample_distinct_azimuths(rng, n_sources=2, min_separation_deg=30.0)
    sig, info = simulate_diverse(
        mic_positions=mics,
        azimuths_deg=azs,
        sampler=sampler,
        fs=16000,
        duration=0.5,
        snr_db=20.0,
        seed=0,
    )
    assert sig.shape == (4, 8000)
    assert "room_dim" in info and "rt60" in info


def test_convolve_multichannel_basic():
    src = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    h = np.array([[0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    out = convolve_multichannel(src, h)
    assert out.shape == (2, 6)
    assert abs(out[0, 1] - 1.0) < 1e-5
    assert abs(out[0, 2] - 0.5) < 1e-5
    assert abs(out[1, 2] - 1.0) < 1e-5


def test_synthesize_real_rir_mixture_shape():
    M = 4
    h = np.random.randn(M, 800).astype(np.float32) * 0.001
    h[:, 100] = 1.0  # direct path
    rec1 = RirRecord(h=h, azimuth_deg=30.0, room_id="test")
    rec2 = RirRecord(h=h, azimuth_deg=-60.0, room_id="test")
    sig, azs = synthesize_real_rir_mixture(
        [rec1, rec2], duration=0.5, fs=16000, snr_db=30.0, seed=0
    )
    assert sig.shape == (4, 8000)
    assert len(azs) == 2 and 30.0 in azs


def test_sample_distinct_rirs():
    bank = [
        RirRecord(h=np.zeros((4, 100), dtype=np.float32), azimuth_deg=a, room_id="x")
        for a in (-90, -60, -30, 0, 30, 60, 90, 120)
    ]
    rng = np.random.default_rng(0)
    chosen = sample_distinct_rirs(rng, bank, n_sources=3, min_separation_deg=30.0)
    assert len(chosen) == 3
    azs = [c.azimuth_deg for c in chosen]
    for i in range(3):
        for j in range(i + 1, 3):
            wrap = ((azs[i] - azs[j] + 180) % 360) - 180
            assert abs(wrap) >= 30.0 - 1e-6
