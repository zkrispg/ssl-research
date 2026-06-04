"""Tests for Multi-ACCDOA + ADPIT."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from multi_accdoa import (
    adpit_loss_batch,
    adpit_loss_sample,
    az_to_xy,
    decode_multi_accdoa,
    make_gt_xy,
    surjective_assignments,
)


def test_surjective_assignments_count():
    # Exactly the documented numbers
    assert len(surjective_assignments(K=1, N=3)) == 1
    assert len(surjective_assignments(K=2, N=3)) == 6
    assert len(surjective_assignments(K=3, N=3)) == 6


def test_surjective_assignments_cover_all_sources():
    for K in (1, 2, 3):
        for f in surjective_assignments(K, N=3):
            assert set(f) == set(range(K))


def test_adpit_perfect_match_single_source():
    """K=1 perfect prediction: loss should be ~0."""
    az = 30.0
    xy = torch.tensor(az_to_xy(np.array([az])))  # (1, 2)
    pred = xy.unsqueeze(0).expand(5, 3, -1).clone()  # (T=5, N=3, 2)
    loss = adpit_loss_sample(pred, xy)
    assert loss.item() < 1e-6


def test_adpit_perfect_match_two_sources_with_duplication():
    """K=2: model predicts two distinct sources + duplicates one. Loss ~0."""
    az = np.array([0.0, 90.0])
    xy = torch.tensor(az_to_xy(az))  # (2, 2)
    # Make pred such that tracks 0,1 = sources 0,1 and track 2 duplicates source 0
    pred = torch.stack([xy[0], xy[1], xy[0]], dim=0)  # (N=3, 2)
    pred = pred.unsqueeze(0).expand(4, -1, -1).clone()  # (T=4, N=3, 2)
    loss = adpit_loss_sample(pred, xy)
    assert loss.item() < 1e-6


def test_adpit_three_sources_permutation_invariant():
    """K=3: any permutation of perfectly-predicted sources yields zero loss."""
    az = np.array([-90.0, 30.0, 150.0])
    xy = torch.tensor(az_to_xy(az))  # (3, 2)
    # Try every permutation
    from itertools import permutations
    for perm in permutations(range(3)):
        pred_xy = xy[list(perm)]
        pred = pred_xy.unsqueeze(0).expand(2, -1, -1).clone()  # (T=2, N=3, 2)
        loss = adpit_loss_sample(pred, xy)
        assert loss.item() < 1e-6, f"perm={perm}: loss={loss.item()}"


def test_adpit_silent_target_zero():
    """K=0 silent: pred should be pushed to (0, 0)."""
    pred = torch.zeros(2, 3, 2)
    gt = torch.empty(0, 2)
    loss = adpit_loss_sample(pred, gt)
    assert loss.item() == 0.0
    pred2 = torch.ones(2, 3, 2)
    loss2 = adpit_loss_sample(pred2, gt)
    assert loss2.item() == 1.0


def test_adpit_batch():
    az_padded = torch.tensor([[0.0, float("nan"), float("nan")],
                              [0.0, 90.0, float("nan")]])
    gt_xy = make_gt_xy(az_padded)
    # K=1 sample tracks all match source 0; K=2 sample uses (s0, s1, s0) etc.
    pred = torch.zeros(2, 4, 3, 2)
    src0 = torch.tensor(az_to_xy(np.array([0.0]))[0])
    src1 = torch.tensor(az_to_xy(np.array([90.0]))[0])
    # Batch 0 (K=1): all tracks predict source 0
    pred[0, :, 0] = src0
    pred[0, :, 1] = src0
    pred[0, :, 2] = src0
    # Batch 1 (K=2): tracks (src0, src1, src0)
    pred[1, :, 0] = src0
    pred[1, :, 1] = src1
    pred[1, :, 2] = src0
    loss = adpit_loss_batch(pred, gt_xy)
    assert loss.item() < 1e-6


def test_make_gt_xy_inactive_preserves_nan():
    az = torch.tensor([[0.0, 90.0, float("nan")]])
    gt = make_gt_xy(az)
    assert gt.shape == (1, 3, 2)
    assert not torch.isnan(gt[0, 0]).any()
    assert not torch.isnan(gt[0, 1]).any()
    assert torch.isnan(gt[0, 2]).all()


def test_decode_multi_accdoa_returns_active_only():
    """Inactive tracks (low magnitude) should be filtered out."""
    pred = torch.zeros(1, 4, 3, 2)
    src0 = torch.tensor(az_to_xy(np.array([45.0]))[0])
    pred[0, :, 0] = src0  # active
    pred[0, :, 1] = src0 * 0.1  # below threshold
    pred[0, :, 2] = torch.tensor(az_to_xy(np.array([-30.0]))[0])  # active
    decoded = decode_multi_accdoa(pred, activity_threshold=0.5)
    assert len(decoded) == 1
    azs = decoded[0]
    assert len(azs) == 2
    assert any(abs(((a - 45.0 + 180) % 360) - 180) < 1.0 for a in azs)
    assert any(abs(((a - (-30.0) + 180) % 360) - 180) < 1.0 for a in azs)


def test_decode_multi_accdoa_nms_dedups():
    """Two tracks with same azimuth should collapse to one peak."""
    pred = torch.zeros(1, 2, 3, 2)
    src = torch.tensor(az_to_xy(np.array([45.0]))[0])
    pred[0, :, 0] = src
    pred[0, :, 1] = src
    pred[0, :, 2] = src * 0.05  # silent
    decoded = decode_multi_accdoa(pred, activity_threshold=0.5, nms_tol_deg=20.0)
    assert len(decoded[0]) == 1
