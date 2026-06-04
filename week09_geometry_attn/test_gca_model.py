"""Tests for GCAMultiTaskCRNN."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))

from geometry import uniform_circular_array
from gca_model import GCAMultiTaskCRNN, count_parameters


def test_gca_model_forward_shape():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    model = GCAMultiTaskCRNN(mic_positions=mics, n_classes=72, max_k=3)
    x = torch.randn(2, 2, 4, 257, 30)
    out = model(x)
    assert out["spectrum"].shape == (2, 30, 72)
    assert out["count"].shape == (2, 3)


def test_gca_adds_few_parameters():
    """Adding GCA must increase param count by < 8% of W6 backbone."""
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    full = GCAMultiTaskCRNN(mic_positions=mics)
    backbone_only = full.backbone
    n_full = count_parameters(full)
    n_back = count_parameters(backbone_only)
    overhead = (n_full - n_back) / n_back
    assert overhead < 0.08, f"too much overhead: {overhead:.2%}"


def test_geometry_bias_changes_outputs():
    """The two ablation variants should produce different outputs."""
    torch.manual_seed(0)
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    m_geo = GCAMultiTaskCRNN(mic_positions=mics, geometry_bias=True)
    m_plain = GCAMultiTaskCRNN(mic_positions=mics, geometry_bias=False)
    backbone_state = m_geo.backbone.state_dict()
    m_plain.backbone.load_state_dict(backbone_state)
    x = torch.randn(2, 2, 4, 257, 8)
    a = m_geo(x)["spectrum"]
    b = m_plain(x)["spectrum"]
    assert not torch.allclose(a, b, atol=1e-5)


def test_overfit_tiny_batch():
    """Sanity: model can overfit a tiny batch on the spectrum head."""
    torch.manual_seed(0)
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    model = GCAMultiTaskCRNN(mic_positions=mics)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    bce = torch.nn.BCEWithLogitsLoss()

    x = torch.randn(4, 2, 4, 257, 12)
    y = torch.zeros(4, 12, 72)
    for b in range(4):
        idx = (b * 11) % 72
        y[b, :, idx] = 1.0

    initial = float(bce(model(x)["spectrum"], y).item())
    for _ in range(60):
        out = model(x)
        loss = bce(out["spectrum"], y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    final = float(bce(model(x)["spectrum"], y).item())
    assert final < initial * 0.4, f"failed: {initial:.4f} -> {final:.4f}"
