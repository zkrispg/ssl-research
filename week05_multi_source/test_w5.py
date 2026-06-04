"""Sanity tests for W5 multi-source SSL."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))

from geometry import uniform_circular_array
from multi_baselines import find_peaks_circular, music_multi, srp_phat_multi
from multi_dataset import (
    MultiSourceConfig,
    MultiSourcePhaseDataset,
    hard_label_vector,
    make_grid,
    soft_label_vector,
)
from multi_eval import LocalizationStats, greedy_match
from multi_model import MultiSourceCRNN, count_parameters
from multi_source_data import (
    sample_distinct_azimuths,
    simulate_freefield_multi,
)


def test_distinct_azimuth_separation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        for k in (1, 2, 3, 4):
            azs = sample_distinct_azimuths(rng, n_sources=k, min_separation_deg=30.0)
            assert len(azs) == k
            for i in range(k):
                for j in range(i + 1, k):
                    diff = ((azs[i] - azs[j] + 180) % 360) - 180
                    assert abs(diff) >= 25.0  # allow small fallback margin


def test_simulate_freefield_multi_shape():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    azs = np.array([0.0, 90.0, -135.0])
    sig, sources = simulate_freefield_multi(
        mics, azimuths_deg=azs, fs=16000, duration=0.5, snr_db=30.0
    )
    assert sig.shape == (4, 8000)
    assert len(sources) == 3


def test_label_vector():
    grid = make_grid(-180, 180, 5)
    soft = soft_label_vector(np.array([0.0, 90.0]), grid, sigma_deg=5.0)
    assert soft.max() <= 1.0 + 1e-6
    # Strongest activation should be at the GT bins
    bins0 = int(np.argmin(np.abs(grid - 0.0)))
    bins90 = int(np.argmin(np.abs(grid - 90.0)))
    assert soft[bins0] > 0.95 and soft[bins90] > 0.95
    hard = hard_label_vector(np.array([0.0, 90.0]), grid)
    assert int(hard.sum()) == 2


def test_dataset_one_sample():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    cfg = MultiSourceConfig(
        mic_positions=mics,
        n_samples=4,
        snr_range_db=(20.0, 30.0),
        duration=0.5,
        max_k=3,
    )
    ds = MultiSourcePhaseDataset(cfg)
    feat, label, azs = ds[0]
    assert feat.shape[:3] == (2, 4, 257)
    assert label.shape == (72,)
    assert azs.shape == (3,)
    n_active = int((~torch.isnan(azs)).sum().item())
    assert 1 <= n_active <= 3


def test_model_forward_shape():
    model = MultiSourceCRNN(n_mics=4, n_freq=257, n_classes=72)
    n_params = count_parameters(model)
    assert n_params < 100_000
    x = torch.randn(2, 2, 4, 257, 30)
    y = model(x)
    assert y.shape == (2, 30, 72)


def test_peak_picker_recovers_ground_truth():
    grid = make_grid(-180, 180, 5)
    spectrum = np.zeros_like(grid)
    for az in (-90.0, 30.0, 120.0):
        bump = np.exp(-(((grid - az + 180) % 360 - 180) ** 2) / (2 * 5.0 ** 2))
        spectrum = np.maximum(spectrum, bump)
    peaks = find_peaks_circular(spectrum, grid, n_peaks=3, rel_threshold=0.3)
    assert len(peaks) == 3
    sorted_peaks = np.sort(peaks)
    expected = np.sort(np.array([-90.0, 30.0, 120.0]))
    err = np.abs(sorted_peaks - expected).max()
    assert err <= 5.0


def test_srp_and_music_multi_recover_two_sources():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    azs = np.array([-45.0, 90.0])
    sig, _ = simulate_freefield_multi(
        mics, azimuths_deg=azs, fs=16000, duration=1.0, snr_db=30.0,
    )
    srp_pred = srp_phat_multi(sig, mics, fs=16000, n_sources=2, rel_threshold=0.5)
    mus_pred = music_multi(sig, mics, fs=16000, n_sources=2, rel_threshold=0.3)
    for pred in (srp_pred, mus_pred):
        assert len(pred) == 2
        # Greedy match within 10 deg
        diff_matrix = np.abs(((pred[:, None] - azs[None, :] + 180) % 360) - 180)
        # Each GT must have a match within 10 deg
        for j in range(2):
            assert diff_matrix[:, j].min() < 10.0


def test_greedy_match_basic():
    matches, up, ug = greedy_match(
        np.array([10.0, 100.0]),
        np.array([95.0, 12.0]),
        tolerance_deg=20.0,
    )
    assert len(matches) == 2 and not up and not ug
    # Tolerance enforced
    matches, up, ug = greedy_match(
        np.array([10.0, 100.0, 200.0]),
        np.array([12.0]),
        tolerance_deg=20.0,
    )
    assert len(matches) == 1


def test_localization_stats_aggregation():
    stats = LocalizationStats()
    # Sample 1: 2 GT, 2 pred, both correct
    stats.add_sample(np.array([0.0, 90.0]), np.array([2.0, 88.0]))
    # Sample 2: 1 GT, 2 pred -> 1 TP, 1 FP
    stats.add_sample(np.array([45.0, 130.0]), np.array([46.0]))
    # Sample 3: 2 GT, 1 pred -> 1 TP, 1 FN
    stats.add_sample(np.array([45.0]), np.array([45.0, 130.0]))
    summary = stats.summary()
    assert summary["n_samples"] == 3
    # TP=2+1+1=4, FP=0+1+0=1, FN=0+0+1=1
    assert summary["f1"] == 2 * (4 / 5) * (4 / 5) / ((4 / 5) + (4 / 5))
