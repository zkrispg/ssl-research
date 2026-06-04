"""Sanity tests for W3 components."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))

from dataset import DatasetConfig, PhaseMapDataset, azimuth_classes, azimuth_to_class
from features import phase_features
from geometry import uniform_circular_array
from model import PhaseMapCNN, count_parameters


def test_phase_features_shape():
    fs = 16000
    n = fs
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((4, n)).astype(np.float32)
    feat = phase_features(sig, fs=fs, n_fft=512, hop_length=256)
    # (2 sin/cos, M, F, T)
    assert feat.shape[0] == 2
    assert feat.shape[1] == 4
    assert feat.shape[2] == 257
    assert feat.shape[3] > 0


def test_azimuth_class_roundtrip():
    grid = azimuth_classes(-180, 180, 5)
    assert len(grid) == 72
    for az_true in [-175.0, -90.0, 0.0, 12.5, 90.0, 174.5]:
        cls = azimuth_to_class(az_true, grid)
        # Snap should be within step/2 = 2.5 deg
        diff = ((grid[cls] - az_true + 180) % 360) - 180
        assert abs(diff) <= 2.6


def test_dataset_one_sample():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    cfg = DatasetConfig(mic_positions=mics, n_samples=4, snr_range_db=(20.0, 30.0))
    ds = PhaseMapDataset(cfg)
    x, y = ds[0]
    assert x.shape == (2, 4, 257)
    assert x.dtype == torch.float32
    assert 0 <= int(y.item()) < 72


def test_model_forward():
    model = PhaseMapCNN(n_mics=4, n_freq=257, n_classes=72)
    n_params = count_parameters(model)
    assert n_params < 60_000, f"model too large: {n_params}"
    x = torch.randn(8, 2, 4, 257)
    logits = model(x)
    assert logits.shape == (8, 72)


def test_model_overfits_tiny_set():
    """Sanity check: the model should overfit a small fixed batch quickly,
    confirming gradients flow correctly through all layers."""
    torch.manual_seed(0)
    model = PhaseMapCNN(n_mics=4, n_freq=257, n_classes=72)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = torch.nn.CrossEntropyLoss()

    rng = np.random.default_rng(0)
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    grid = azimuth_classes(-180, 180, 5)
    # Small fixed dataset
    xs = []
    ys = []
    cfg = DatasetConfig(
        mic_positions=mics, n_samples=8, snr_range_db=(40.0, 40.0), seed_base=42
    )
    ds = PhaseMapDataset(cfg)
    for i in range(8):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs)
    Y = torch.stack(ys)

    initial_loss = float(criterion(model(X), Y).item())
    for _ in range(40):
        optim.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optim.step()
    final_loss = float(loss.item())
    assert final_loss < initial_loss * 0.3, (
        f"model failed to overfit tiny set: {initial_loss:.3f} -> {final_loss:.3f}"
    )
