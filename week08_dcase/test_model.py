"""Tests for MultiAccdoaCRNN."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from multi_accdoa_model import MultiAccdoaCRNN, count_parameters
from multi_accdoa import adpit_loss_batch, az_to_xy, make_gt_xy


def test_model_forward_shapes():
    model = MultiAccdoaCRNN(n_mics=4, n_freq=257, n_tracks=3, max_k=3)
    n_params = count_parameters(model)
    assert n_params < 110_000, f"too large: {n_params}"
    x = torch.randn(2, 2, 4, 257, 30)
    out = model(x)
    assert out["accdoa"].shape == (2, 30, 3, 2)
    assert out["count"].shape == (2, 3)


def test_model_overfits_tiny_set_with_adpit():
    """Joint ADPIT + CE must drive both heads to overfit a tiny batch."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = MultiAccdoaCRNN(n_mics=4, n_freq=257, n_tracks=3, max_k=3)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    ce = torch.nn.CrossEntropyLoss()

    feats = []
    az_padded = []
    ks = []
    for i in range(4):
        f = torch.randn(2, 4, 257, 20)
        K = 1 + (i % 2)  # K = 1 or 2
        azs = [-90.0 + 60.0 * j + 10.0 * (i + 1) for j in range(K)]
        padded = azs + [float("nan")] * (3 - K)
        feats.append(f)
        az_padded.append(padded)
        ks.append(K - 1)
    X = torch.stack(feats)
    az_pad_t = torch.tensor(az_padded)
    gt_xy = make_gt_xy(az_pad_t)
    K_target = torch.tensor(ks, dtype=torch.long)

    initial = float(adpit_loss_batch(model(X)["accdoa"], gt_xy).item())
    for _ in range(80):
        out = model(X)
        loss_accdoa = adpit_loss_batch(out["accdoa"], gt_xy)
        loss_count = ce(out["count"], K_target)
        loss = loss_accdoa + loss_count
        optim.zero_grad()
        loss.backward()
        optim.step()
    final = float(adpit_loss_batch(model(X)["accdoa"], gt_xy).item())
    assert final < initial * 0.4, f"failed: {initial:.4f} -> {final:.4f}"
