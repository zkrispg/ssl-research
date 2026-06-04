"""Unit tests for class-coupled ADPIT loss."""
from __future__ import annotations

import math

import pytest
import torch

from week11_starss23.seld_labels import doa_to_xyz
from week11_starss23.seld_loss import (
    ClassCoupledAdpitLoss,
    _activity_coupled,
    _build_target_patterns,
    select_best_pattern,
)


# ---------------------------------------------------------------------------
# Activity coupling
# ---------------------------------------------------------------------------


def test_activity_coupled_inactive_zeros():
    target = torch.zeros(1, 1, 6, 4, 13)
    coupled = _activity_coupled(target)
    assert coupled.shape == (1, 1, 6, 3, 13)
    assert coupled.abs().sum().item() == 0.0


def test_activity_coupled_active_passes_xyz():
    target = torch.zeros(1, 1, 6, 4, 13)
    target[0, 0, 0, 0, 5] = 1.0  # activity for class 5, slot A0
    target[0, 0, 0, 1:4, 5] = torch.tensor([0.5, 0.6, 0.7])
    coupled = _activity_coupled(target)
    assert torch.allclose(coupled[0, 0, 0, :, 5], torch.tensor([0.5, 0.6, 0.7]))
    # All other slots stay zero.
    coupled[0, 0, 0, :, 5] = 0.0
    assert coupled.abs().sum().item() == 0.0


def test_activity_coupled_inactive_overrides_xyz():
    target = torch.zeros(1, 1, 6, 4, 13)
    target[0, 0, 0, 1:4, 5] = torch.tensor([0.5, 0.6, 0.7])  # xyz set, activity 0
    coupled = _activity_coupled(target)
    assert coupled.abs().sum().item() == 0.0


def test_activity_coupled_rejects_wrong_shape():
    target = torch.zeros(1, 1, 6, 5, 13)  # axis 5 instead of 4
    with pytest.raises(ValueError, match="expected target shape"):
        _activity_coupled(target)


# ---------------------------------------------------------------------------
# Pattern construction
# ---------------------------------------------------------------------------


def test_pattern_count_is_13():
    coupled = torch.randn(1, 1, 6, 3, 13)
    patterns = _build_target_patterns(coupled)
    assert patterns.shape == (13, 1, 1, 9, 13)


def test_pattern_a_is_a0_triplicate():
    coupled = torch.zeros(1, 1, 6, 3, 13)
    coupled[0, 0, 0, :, 5] = torch.tensor([0.7, -0.3, 0.6])  # A0 active for class 5
    patterns = _build_target_patterns(coupled)
    pattern_a = patterns[0]  # (1, 1, 9, 13)
    expected = torch.cat([coupled[0, 0, 0, :, 5]] * 3)  # (9,)
    assert torch.allclose(pattern_a[0, 0, :, 5], expected)


def test_pattern_c_permutations_are_distinct():
    coupled = torch.zeros(1, 1, 6, 3, 13)
    coupled[0, 0, 3, :, 5] = torch.tensor([1.0, 0.0, 0.0])  # C0
    coupled[0, 0, 4, :, 5] = torch.tensor([0.0, 1.0, 0.0])  # C1
    coupled[0, 0, 5, :, 5] = torch.tensor([0.0, 0.0, 1.0])  # C2
    patterns = _build_target_patterns(coupled)
    # Patterns 7..12 are the 6 C perms; should all be different from each other.
    perms = [tuple(patterns[k][0, 0, :, 5].tolist()) for k in range(7, 13)]
    assert len(set(perms)) == 6


# ---------------------------------------------------------------------------
# Loss properties
# ---------------------------------------------------------------------------


def _make_target_a0(n_classes: int = 13, active_class: int = 5, az: float = 0.0, el: float = 0.0):
    """Build a (1, 1, 6, 4, C) target with a single A0 source."""
    target = torch.zeros(1, 1, 6, 4, n_classes)
    x, y, z = doa_to_xyz(az, el)
    target[0, 0, 0, 0, active_class] = 1.0
    target[0, 0, 0, 1, active_class] = x
    target[0, 0, 0, 2, active_class] = y
    target[0, 0, 0, 3, active_class] = z
    return target


def test_loss_zero_when_pred_matches_pattern_a():
    """If pred is the activity-coupled A0 triplicate, loss must be 0."""
    target = _make_target_a0(active_class=5, az=30.0, el=10.0)
    coupled = _activity_coupled(target)
    A0 = coupled[:, :, 0]  # (1, 1, 3, 13)
    pred_4d = torch.cat([A0, A0, A0], dim=2)  # (1, 1, 9, 13)
    loss = ClassCoupledAdpitLoss()(pred_4d, target)
    assert loss.item() < 1e-7


def test_loss_zero_when_pred_matches_via_flat():
    """Same test but with flat (1, 1, 117) pred shape."""
    target = _make_target_a0(active_class=2, az=-45.0, el=15.0)
    coupled = _activity_coupled(target)
    A0 = coupled[:, :, 0]  # (1, 1, 3, 13)
    pred_flat = torch.cat([A0, A0, A0], dim=2).reshape(1, 1, -1)
    loss = ClassCoupledAdpitLoss()(pred_flat, target)
    assert loss.item() < 1e-7


def test_loss_invariant_to_track_permutation_for_b():
    """Two same-class sources at B0 and B1: loss must be the same whether
    the prediction places them as (B0, B0, B1), (B1, B0, B0), etc."""
    target = torch.zeros(1, 1, 6, 4, 13)
    # B0 + B1 active for class 4.
    target[0, 0, 1, 0, 4] = 1.0  # B0 activity
    target[0, 0, 1, 1, 4] = 0.7  # B0 x
    target[0, 0, 1, 2, 4] = 0.3
    target[0, 0, 2, 0, 4] = 1.0  # B1 activity
    target[0, 0, 2, 1, 4] = -0.6
    target[0, 0, 2, 2, 4] = 0.4

    coupled = _activity_coupled(target)
    B0 = coupled[:, :, 1]
    B1 = coupled[:, :, 2]
    fn = ClassCoupledAdpitLoss()

    pred_a = torch.cat([B0, B0, B1], dim=2)  # one of 6 B perms
    pred_b = torch.cat([B1, B0, B0], dim=2)  # different perm
    pred_c = torch.cat([B0, B1, B0], dim=2)
    l_a = fn(pred_a, target).item()
    l_b = fn(pred_b, target).item()
    l_c = fn(pred_c, target).item()
    assert abs(l_a) < 1e-7
    assert abs(l_b) < 1e-7
    assert abs(l_c) < 1e-7


def test_loss_picks_correct_pattern_for_single_source():
    target = _make_target_a0(active_class=8, az=90.0, el=0.0)
    coupled = _activity_coupled(target)
    A0 = coupled[:, :, 0]
    pred = torch.cat([A0, A0, A0], dim=2)
    chosen = select_best_pattern(pred, target)
    # For class 8, pattern 0 (A0A0A0) should be optimal.
    assert chosen[0, 0, 8].item() == 0


def test_loss_silent_class_loss_is_zero_when_pred_zero():
    """If a class is silent everywhere, predicting zero gives zero loss for that class."""
    target = torch.zeros(1, 1, 6, 4, 13)
    pred = torch.zeros(1, 1, 9, 13)
    fn = ClassCoupledAdpitLoss(reduction="none")
    per_btc = fn(pred, target)  # (1, 1, 13)
    assert torch.allclose(per_btc, torch.zeros_like(per_btc))


def test_loss_silent_class_penalises_nonzero_pred():
    target = torch.zeros(1, 1, 6, 4, 13)
    pred = torch.zeros(1, 1, 9, 13)
    pred[0, 0, 0, 5] = 0.5  # ghost prediction for class 5
    fn = ClassCoupledAdpitLoss(reduction="none")
    per_btc = fn(pred, target)
    # All other classes still 0, class 5 should be > 0.
    assert per_btc[0, 0, 5].item() > 0
    other_mean = per_btc[0, 0, [c for c in range(13) if c != 5]].mean().item()
    assert other_mean < 1e-7


def test_loss_classes_are_independent():
    """Two classes with their own ADPIT pattern should each pay their own price."""
    target = torch.zeros(1, 1, 6, 4, 13)
    # Class 2: A0 active.
    target[0, 0, 0, 0, 2] = 1.0
    target[0, 0, 0, 1, 2] = 1.0  # +x
    # Class 7: A0 active different DOA.
    target[0, 0, 0, 0, 7] = 1.0
    target[0, 0, 0, 2, 7] = 1.0  # +y

    coupled = _activity_coupled(target)
    A0 = coupled[:, :, 0]
    pred = torch.cat([A0, A0, A0], dim=2)  # (1, 1, 9, 13)
    fn = ClassCoupledAdpitLoss(reduction="none")
    per_btc = fn(pred, target)
    assert per_btc[0, 0, 2].item() < 1e-7
    assert per_btc[0, 0, 7].item() < 1e-7


def test_loss_backward_runs():
    target = _make_target_a0(active_class=0, az=10.0, el=0.0)
    pred = torch.zeros(1, 1, 9, 13, requires_grad=True)
    loss = ClassCoupledAdpitLoss()(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0


def test_loss_reduction_none_shape():
    target = torch.zeros(2, 3, 6, 4, 13)
    pred = torch.zeros(2, 3, 9, 13)
    fn = ClassCoupledAdpitLoss(reduction="none")
    out = fn(pred, target)
    assert out.shape == (2, 3, 13)


def test_loss_invalid_reduction_raises():
    with pytest.raises(ValueError, match="reduction must be"):
        ClassCoupledAdpitLoss(reduction="sum")


def test_loss_rejects_incompatible_pred_dim():
    fn = ClassCoupledAdpitLoss()
    target = torch.zeros(1, 1, 6, 4, 13)
    bad = torch.zeros(1, 1, 100)  # 100 not divisible by 3*13=39
    with pytest.raises(ValueError, match="not divisible"):
        fn(bad, target)


def test_loss_seld_model_integration():
    """End-to-end: SeldCRNN forward + ADPIT backward on synthetic batch."""
    from week11_starss23.seld_model import SeldCRNN, SeldModelConfig

    cfg = SeldModelConfig(use_gca=True, gca_geometry_bias=True)
    model = SeldCRNN(cfg)
    x = torch.randn(2, cfg.in_channels, 25, cfg.n_freq_bins) * 0.1
    target = torch.zeros(2, 5, 6, 4, cfg.n_classes)
    target[0, 1, 0, 0, 3] = 1.0  # class 3 active in batch 0, frame 1
    target[0, 1, 0, 1:4, 3] = torch.tensor([1.0, 0.0, 0.0])

    out = model(x)["accdoa"]  # (2, 5, 117)
    loss = ClassCoupledAdpitLoss()(out, target)
    assert loss.item() >= 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
