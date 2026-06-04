"""Sanity tests for W4 CRNN."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))

from crnn_dataset import (
    MultiFrameConfig,
    MultiFramePhaseDataset,
    az_to_xy,
    xy_to_az_deg,
)
from geometry import uniform_circular_array
from crnn_model import CRNNDoa, count_parameters


def test_xy_az_roundtrip():
    az_rad = torch.linspace(-np.pi + 0.01, np.pi - 0.01, 17)
    xy = az_to_xy(az_rad)
    az_back_deg = xy_to_az_deg(xy)
    err = (az_back_deg - torch.rad2deg(az_rad)).abs()
    assert torch.all(err < 1e-3)


def test_dataset_one_sample():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    cfg = MultiFrameConfig(mic_positions=mics, n_samples=4, snr_range_db=(20.0, 30.0))
    ds = MultiFramePhaseDataset(cfg)
    x, y = ds[0]
    # x: (2, 4, 257, T)
    assert x.shape[0] == 2 and x.shape[1] == 4 and x.shape[2] == 257
    assert x.shape[3] >= 10
    assert -np.pi - 1e-3 <= float(y) <= np.pi + 1e-3


def test_model_forward():
    model = CRNNDoa(n_mics=4, n_freq=257)
    n_params = count_parameters(model)
    assert n_params < 100_000, f"model too large: {n_params}"
    x = torch.randn(2, 2, 4, 257, 30)
    y = model(x)
    assert y.shape == (2, 30, 2)


def test_model_overfits_tiny_set():
    """The CRNN should overfit a tiny batch within ~80 steps."""
    torch.manual_seed(0)
    model = CRNNDoa(n_mics=4, n_freq=257)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)

    mics = uniform_circular_array(n_mics=4, radius=0.04)
    cfg = MultiFrameConfig(
        mic_positions=mics,
        n_samples=4,
        snr_range_db=(40.0, 40.0),
        seed_base=1234,
    )
    ds = MultiFramePhaseDataset(cfg)
    xs = []
    ys = []
    for i in range(4):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs)
    Y_az = torch.stack(ys)
    Y_xy = az_to_xy(Y_az)  # (B, 2)

    # Repeat target across time
    initial_pred = model(X)  # (B, T, 2)
    T = initial_pred.shape[1]
    Y_xy_seq = Y_xy.unsqueeze(1).expand(-1, T, -1)
    initial_loss = float(torch.mean((initial_pred - Y_xy_seq) ** 2).item())

    for _ in range(80):
        pred = model(X)
        loss = torch.mean((pred - Y_xy_seq) ** 2)
        optim.zero_grad()
        loss.backward()
        optim.step()
    final_loss = float(loss.item())
    assert final_loss < initial_loss * 0.3, (
        f"CRNN failed to overfit tiny set: {initial_loss:.4f} -> {final_loss:.4f}"
    )
