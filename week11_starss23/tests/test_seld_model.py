"""Unit tests for week11_starss23.seld_model."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from week09_geometry_attn.geometry_attn import count_parameters
from week11_starss23.seld_model import (
    SeldCRNN,
    SeldModelConfig,
    default_uca4_positions,
    make_default_seld_model,
)


# ---------------------------------------------------------------------------
# Sanity helpers
# ---------------------------------------------------------------------------


def _expect_label_T(T_feat: int, ratio: int) -> int:
    return T_feat // ratio


# ---------------------------------------------------------------------------
# default_uca4_positions
# ---------------------------------------------------------------------------


def test_default_uca4_positions_has_expected_radius_and_count():
    pos = default_uca4_positions(radius_m=0.04)
    assert pos.shape == (4, 2)
    radii = np.linalg.norm(pos, axis=1)
    np.testing.assert_allclose(radii, 0.04, atol=1e-6)


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


def test_seld_model_forward_shape_default():
    cfg = SeldModelConfig()
    model = SeldCRNN(cfg)
    B, T_feat = 2, 50  # 50 / ratio=5 = 10 label frames
    x = torch.randn(B, cfg.in_channels, T_feat, cfg.n_freq_bins)
    out = model(x)
    assert "accdoa" in out
    assert out["accdoa"].shape == (
        B,
        _expect_label_T(T_feat, cfg.feature_per_label_ratio),
        cfg.n_tracks * 3 * cfg.n_classes,
    )


def test_seld_model_forward_dtype_finite():
    model = make_default_seld_model()
    x = torch.randn(1, 10, 25, 64) * 0.1
    out = model(x)
    assert out["accdoa"].dtype == torch.float32
    assert torch.isfinite(out["accdoa"]).all()


def test_seld_model_reshape_accdoa():
    model = make_default_seld_model()
    flat = torch.randn(2, 10, model.n_accdoa_outputs)
    reshaped = model.reshape_accdoa(flat)
    assert reshaped.shape == (2, 10, model.cfg.n_tracks, 3, model.cfg.n_classes)


def test_seld_model_handles_different_T():
    model = make_default_seld_model()
    for T_feat in (15, 30, 50, 100):
        x = torch.randn(1, 10, T_feat, 64) * 0.1
        out = model(x)
        assert out["accdoa"].shape[1] == T_feat // model.cfg.feature_per_label_ratio


def test_seld_model_handles_batch_size_1_and_4():
    model = make_default_seld_model()
    for B in (1, 4):
        x = torch.randn(B, 10, 25, 64) * 0.1
        out = model(x)
        assert out["accdoa"].shape[0] == B


# ---------------------------------------------------------------------------
# Distance head
# ---------------------------------------------------------------------------


def test_seld_model_with_distance_head():
    cfg = SeldModelConfig(use_distance_head=True)
    model = SeldCRNN(cfg)
    x = torch.randn(2, cfg.in_channels, 25, cfg.n_freq_bins) * 0.1
    out = model(x)
    assert "distance" in out
    assert out["distance"].shape == (2, 5, cfg.n_classes)
    # Softplus output is non-negative.
    assert (out["distance"] >= 0).all()


def test_seld_model_no_distance_head_by_default():
    model = make_default_seld_model()
    x = torch.randn(1, 10, 25, 64) * 0.1
    out = model(x)
    assert "distance" not in out


# ---------------------------------------------------------------------------
# GCA integration
# ---------------------------------------------------------------------------


def test_seld_model_with_gca_full():
    """GCA on, geometry bias on -> full W9 path."""
    cfg = SeldModelConfig(use_gca=True, gca_geometry_bias=True)
    model = SeldCRNN(cfg)
    assert model.gca is not None
    assert model.gca.geometry_bias is True
    assert model.gca.geom_proj is not None
    x = torch.randn(2, cfg.in_channels, 25, cfg.n_freq_bins) * 0.1
    out = model(x)
    assert out["accdoa"].shape == (2, 5, cfg.n_tracks * 3 * cfg.n_classes)


def test_seld_model_with_gca_no_geom():
    """GCA on, geometry bias off -> ablation variant."""
    cfg = SeldModelConfig(use_gca=True, gca_geometry_bias=False)
    model = SeldCRNN(cfg)
    assert model.gca is not None
    assert model.gca.geometry_bias is False
    assert model.gca.geom_proj is None
    x = torch.randn(2, cfg.in_channels, 25, cfg.n_freq_bins) * 0.1
    out = model(x)
    assert out["accdoa"].shape == (2, 5, cfg.n_tracks * 3 * cfg.n_classes)


def test_seld_model_without_gca():
    cfg = SeldModelConfig(use_gca=False)
    model = SeldCRNN(cfg)
    assert model.gca is None


def test_gca_actually_changes_outputs():
    """W9 'full' and W9 'no_geom' must produce different outputs on the same input."""
    torch.manual_seed(0)
    cfg_full = SeldModelConfig(use_gca=True, gca_geometry_bias=True)
    cfg_no_geom = SeldModelConfig(use_gca=True, gca_geometry_bias=False)
    model_full = SeldCRNN(cfg_full).eval()
    model_no = SeldCRNN(cfg_no_geom).eval()
    x = torch.randn(1, cfg_full.in_channels, 30, cfg_full.n_freq_bins) * 0.1
    with torch.no_grad():
        out_full = model_full(x)["accdoa"]
        out_no = model_no(x)["accdoa"]
    # Not identical (different parameter init AND structurally different geom_proj).
    assert not torch.allclose(out_full, out_no, atol=1e-6)


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------


def test_parameter_count_reasonable_no_gca():
    """Default backbone should be ~200-700K parameters for ICASSP-budget model."""
    model = make_default_seld_model()
    p = count_parameters(model)
    assert 100_000 <= p <= 1_500_000, f"unexpected param count {p}"


def test_gca_overhead_is_small():
    """Per W9 paper claims, GCA adds ~1-3 K parameters."""
    cfg_no_gca = SeldModelConfig(use_gca=False)
    cfg_full = SeldModelConfig(use_gca=True, gca_geometry_bias=True)
    cfg_no_geom = SeldModelConfig(use_gca=True, gca_geometry_bias=False)
    p_no_gca = count_parameters(SeldCRNN(cfg_no_gca))
    p_full = count_parameters(SeldCRNN(cfg_full))
    p_no_geom = count_parameters(SeldCRNN(cfg_no_geom))
    overhead_full = p_full - p_no_gca
    overhead_no_geom = p_no_geom - p_no_gca
    # Full > no_geom because of the extra geom_proj layer.
    assert overhead_full > overhead_no_geom
    # Both overheads should be small relative to the backbone.
    assert overhead_full < p_no_gca * 0.05
    # The geometry-bias path itself adds only a small linear layer (4 -> embed_dim).
    geom_only_overhead = overhead_full - overhead_no_geom
    assert geom_only_overhead < 1000  # 4*16 + 16 bias = 80 params, with extras some hundreds


# ---------------------------------------------------------------------------
# Backward / training-readiness
# ---------------------------------------------------------------------------


def test_seld_model_backward_runs():
    model = make_default_seld_model(use_gca=True)
    x = torch.randn(2, 10, 25, 64, requires_grad=True) * 0.1
    out = model(x)["accdoa"]
    loss = out.pow(2).mean()
    loss.backward()
    # Verify some gradients flowed.
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    assert len(grad_norms) > 0
    assert max(grad_norms) > 0
    # Sanity: not all gradients exploded.
    assert max(grad_norms) < 1e6


def test_seld_model_gpu_compatible():
    """Smoke test: model runs on GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = make_default_seld_model(use_gca=True).cuda()
    x = torch.randn(1, 10, 25, 64, device="cuda") * 0.1
    out = model(x)["accdoa"]
    assert out.is_cuda
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_seld_model_rejects_wrong_in_channels_at_forward():
    model = make_default_seld_model()
    bad = torch.randn(1, 5, 25, 64)  # 5 channels instead of 10
    with pytest.raises(ValueError, match="expected 10 input channels"):
        model(bad)


def test_seld_model_rejects_wrong_dim_at_forward():
    model = make_default_seld_model()
    bad = torch.randn(10, 25, 64)  # missing batch dim
    with pytest.raises(ValueError, match="expected 4-D input"):
        model(bad)


def test_seld_model_validates_freq_pool_divisibility():
    cfg = SeldModelConfig(n_freq_bins=64, f_pool_size=(4, 4, 3))  # 4*4*3=48 doesn't divide 64
    with pytest.raises(ValueError, match="must be divisible"):
        SeldCRNN(cfg)


def test_seld_model_validates_freq_pool_length():
    cfg = SeldModelConfig(f_pool_size=(4, 4))  # only 2 entries
    with pytest.raises(ValueError, match="must have length 3"):
        SeldCRNN(cfg)
