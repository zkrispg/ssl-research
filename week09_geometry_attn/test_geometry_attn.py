"""Unit tests for Geometry-aware Channel Attention."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))

from geometry import uniform_circular_array, uniform_linear_array
from geometry_attn import (
    GeometryAwareChannelAttention,
    count_parameters,
    mic_pair_geometry,
)


def test_mic_pair_geometry_uca4_distances():
    """UCA4 with radius 4 cm: adjacent dist = sqrt(2)*4 cm, opposite = 8 cm."""
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    geom = mic_pair_geometry(mics)
    assert geom.shape == (4, 4, 4)
    np.testing.assert_allclose(np.diag(geom[..., 2]), 0.0, atol=1e-6)
    adj = geom[0, 1, 2]
    opp = geom[0, 2, 2]
    np.testing.assert_allclose(adj, np.sqrt(2) * 0.04, atol=1e-6)
    np.testing.assert_allclose(opp, 0.08, atol=1e-6)


def test_mic_pair_geometry_symmetric_distance_antisymmetric_offset():
    mics = uniform_linear_array(n_mics=4, spacing=0.05)
    geom = mic_pair_geometry(mics)
    np.testing.assert_allclose(geom[..., 2], geom[..., 2].T, atol=1e-6)
    np.testing.assert_allclose(geom[..., 0] + geom[..., 0].T, 0.0, atol=1e-6)
    np.testing.assert_allclose(geom[..., 1] + geom[..., 1].T, 0.0, atol=1e-6)


def test_forward_shape_uca4():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    gca = GeometryAwareChannelAttention(mics)
    x = torch.randn(2, 2, 4, 257, 30)
    y = gca(x)
    assert y.shape == x.shape


def test_param_count_small():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    gca = GeometryAwareChannelAttention(mics, embed_dim=16)
    n_p = count_parameters(gca)
    assert n_p < 3500, f"too many params: {n_p}"


def test_geometry_off_collapses_to_plain_attention():
    """With geometry_bias=False the geom_proj layer should not exist."""
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    gca = GeometryAwareChannelAttention(mics, geometry_bias=False)
    assert gca.geom_proj is None
    x = torch.randn(1, 2, 4, 257, 8)
    y = gca(x)
    assert y.shape == x.shape


def test_geometry_changes_output():
    """Same x but different mic_positions -> different gate, hence different output."""
    torch.manual_seed(0)
    x = torch.randn(2, 2, 4, 257, 30)
    mics_a = uniform_circular_array(n_mics=4, radius=0.04)
    mics_b = uniform_linear_array(n_mics=4, spacing=0.04)
    gca_a = GeometryAwareChannelAttention(mics_a)
    gca_b = GeometryAwareChannelAttention(mics_b)
    gca_b.load_state_dict(gca_a.state_dict())  # share trainable weights
    y_a = gca_a(x)
    y_b = gca_b(x)
    assert not torch.allclose(y_a, y_b, atol=1e-5)


def test_gate_in_zero_one():
    mics = uniform_circular_array(n_mics=4, radius=0.04)
    gca = GeometryAwareChannelAttention(mics)
    x = torch.randn(2, 2, 4, 257, 30)
    y = gca(x)
    ratio = y / (x + 1e-9)
    assert (ratio.abs() <= 1.0 + 1e-5).all(), \
        f"ratio out of [0,1]: max={float(ratio.abs().max()):.4f}"
