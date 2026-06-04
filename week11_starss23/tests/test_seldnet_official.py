"""Unit tests for the strict DCASE 2023 SELDnet baseline reproduction."""
from __future__ import annotations

import torch

from week11_starss23.seldnet_official import (
    SeldNetOfficial,
    SeldNetOfficialConfig,
    make_default_seldnet_official,
)


def test_default_config_matches_published_baseline() -> None:
    """The defaults must match the values published for DCASE 2023 Task 3."""
    cfg = SeldNetOfficialConfig()
    assert cfg.cnn_filters == 64
    assert cfg.f_pool_size == (4, 4, 2)
    assert cfg.rnn_hidden == 128
    assert cfg.rnn_layers == 2
    assert cfg.fc_hidden == 128
    assert cfg.feature_per_label_ratio == 5
    assert cfg.n_classes == 13
    assert cfg.n_tracks == 3


def test_forward_shapes_match_label_resolution() -> None:
    """Conv block 1 should collapse the time axis to label resolution."""
    cfg = SeldNetOfficialConfig(in_channels=10, n_freq_bins=64)
    model = SeldNetOfficial(cfg)
    B, T_feat = 2, 250  # 250 = 50 * 5  (label hop = 5)
    x = torch.randn(B, cfg.in_channels, T_feat, cfg.n_freq_bins)
    out = model(x)
    assert "accdoa" in out
    accdoa = out["accdoa"]
    expected_t_label = T_feat // cfg.feature_per_label_ratio
    expected_out = cfg.n_tracks * 3 * cfg.n_classes
    assert accdoa.shape == (B, expected_t_label, expected_out)


def test_reshape_accdoa_round_trip() -> None:
    cfg = SeldNetOfficialConfig()
    model = SeldNetOfficial(cfg)
    B, T = 1, 50
    flat = torch.randn(B, T, cfg.n_tracks * 3 * cfg.n_classes)
    reshaped = model.reshape_accdoa(flat)
    assert reshaped.shape == (B, T, cfg.n_tracks, 3, cfg.n_classes)
    assert torch.equal(reshaped.flatten(2), flat)


def test_output_in_tanh_range() -> None:
    """Final ``tanh`` should keep the head output within [-1, 1]."""
    cfg = SeldNetOfficialConfig()
    model = SeldNetOfficial(cfg)
    x = torch.randn(2, cfg.in_channels, 100, cfg.n_freq_bins)
    out = model(x)["accdoa"]
    assert torch.all(out >= -1.0)
    assert torch.all(out <= 1.0)


def test_param_count_in_baseline_range() -> None:
    """SELDnet 2023 baseline is between ~500K and ~1M params with these defaults."""
    model = make_default_seldnet_official()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Conservatively wide bound: rejects bugs that would create a 10M model
    # but tolerates small deviations from the exact published count.
    assert 400_000 < n_params < 1_500_000, f"unexpected param count: {n_params:,}"


def test_no_gca_attribute_exists() -> None:
    """The class is meant to be GCA-free; reviewers should be able to verify it."""
    model = make_default_seldnet_official()
    assert not hasattr(model, "gca"), "official baseline must not have a GCA module"


def test_invalid_input_channels_raises() -> None:
    cfg = SeldNetOfficialConfig(in_channels=10)
    model = SeldNetOfficial(cfg)
    bad_x = torch.randn(1, 9, 50, cfg.n_freq_bins)
    try:
        model(bad_x)
    except ValueError as exc:
        assert "input channels" in str(exc)
    else:
        raise AssertionError("expected ValueError for wrong channel count")


def test_invalid_input_dim_raises() -> None:
    model = make_default_seldnet_official()
    bad_x = torch.randn(1, 10, 50)  # missing freq dim
    try:
        model(bad_x)
    except ValueError as exc:
        assert "4-D" in str(exc)
    else:
        raise AssertionError("expected ValueError for 3-D input")


def test_invalid_freq_pool_raises() -> None:
    """``n_freq_bins`` not divisible by prod(f_pool_size) should fail loudly."""
    cfg = SeldNetOfficialConfig(n_freq_bins=63, f_pool_size=(4, 4, 2))
    try:
        SeldNetOfficial(cfg)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError for bad freq config")


def test_gradient_flow() -> None:
    """A backward pass must populate gradients on every parameter."""
    model = make_default_seldnet_official()
    x = torch.randn(1, model.cfg.in_channels, 50, model.cfg.n_freq_bins)
    loss = model(x)["accdoa"].pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any((g.abs() > 0).any() for g in grads if g is not None)


def test_eval_mode_is_deterministic() -> None:
    """In ``eval()`` mode dropout is off, so two forward passes match."""
    torch.manual_seed(0)
    model = make_default_seldnet_official()
    model.eval()
    x = torch.randn(1, model.cfg.in_channels, 50, model.cfg.n_freq_bins)
    with torch.no_grad():
        a = model(x)["accdoa"]
        b = model(x)["accdoa"]
    assert torch.equal(a, b)
