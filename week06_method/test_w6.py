"""Sanity tests for W6 augmentation, dataset wrapper, and multi-task model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week02_classical"))
sys.path.insert(0, str(Path(__file__).parent.parent / "week05_multi_source"))

from aug import channel_rotate, spec_augment
from geometry import uniform_circular_array
from multi_dataset import MultiSourceConfig, soft_label_vector, make_grid
from multi_task_dataset import AugConfig, MultiTaskDataset
from multi_task_model import MultiTaskCRNN, count_parameters
from torch.utils.data import TensorDataset


def test_channel_rotate_label_consistency():
    """Rotating a single-source spectrum by k mics shifts its peak by 90*k deg."""
    grid = make_grid(-180, 180, 5)
    n_classes = len(grid)
    label = torch.from_numpy(soft_label_vector(np.array([0.0]), grid, sigma_deg=5.0))
    feat = torch.zeros((2, 4, 257, 30))

    for k in range(4):
        _, label_rot = channel_rotate(feat, label, k=k, n_mics=4, n_classes=n_classes)
        peak_idx = int(torch.argmax(label_rot).item())
        peak_az = float(grid[peak_idx])
        # Expected: 0 - 90*k, wrapped to (-180, 180]
        expected = ((0.0 - 90.0 * k + 180.0) % 360.0) - 180.0
        diff = ((peak_az - expected + 180.0) % 360.0) - 180.0
        assert abs(diff) < 1.0, f"k={k} peak={peak_az} expected={expected}"


def test_channel_rotate_feature_consistency():
    """Rotating the mic axis cycles through all four orderings."""
    feat = torch.arange(2 * 4 * 4 * 3, dtype=torch.float32).reshape(2, 4, 4, 3)
    label = torch.zeros(72)
    for k in range(4):
        rotated, _ = channel_rotate(feat, label, k=k, n_mics=4, n_classes=72)
        # rotated[:, m, :, :] should equal feat[:, (m+k) % 4, :, :]
        for m in range(4):
            assert torch.equal(rotated[:, m], feat[:, (m + k) % 4]), (
                f"k={k} m={m} mismatch"
            )


def test_spec_augment_zeros_some_bins():
    rng = np.random.default_rng(0)
    feat = torch.ones(2, 4, 257, 30)
    out = spec_augment(feat, rng, n_freq_masks=2, max_freq_mask=20,
                       n_time_masks=2, max_time_mask=5)
    # At least some bins should now be zero.
    assert (out == 0).any().item()
    assert out.shape == feat.shape


def test_multi_task_model_forward_and_params():
    model = MultiTaskCRNN(n_mics=4, n_freq=257, n_classes=72, max_k=3)
    n_params = count_parameters(model)
    assert n_params < 110_000
    x = torch.randn(2, 2, 4, 257, 30)
    out = model(x)
    assert "spectrum" in out and "count" in out
    assert out["spectrum"].shape == (2, 30, 72)
    assert out["count"].shape == (2, 3)


def test_multi_task_model_overfits_tiny_set():
    """Joint loss must drive both heads to overfit a single batch of 4 samples."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    grid = make_grid(-180, 180, 5)

    feats, labels, ks = [], [], []
    for i in range(4):
        f = torch.randn(2, 4, 257, 20)
        az = float(rng.uniform(-90, 90))
        lbl = torch.from_numpy(soft_label_vector(np.array([az]), grid, 5.0))
        feats.append(f)
        labels.append(lbl)
        ks.append(0)  # K=1 -> class 0
    X = torch.stack(feats)
    Y = torch.stack(labels)
    K = torch.tensor(ks, dtype=torch.long)

    model = MultiTaskCRNN(n_mics=4, n_freq=257, n_classes=72, max_k=3)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.full((72,), 12.0))
    ce = torch.nn.CrossEntropyLoss()

    initial_total = None
    for step in range(60):
        out = model(X)
        spec = out["spectrum"]
        T = spec.shape[1]
        target_spec = Y.unsqueeze(1).expand(-1, T, -1)
        loss_spec = bce(spec, target_spec)
        loss_count = ce(out["count"], K)
        loss = loss_spec + loss_count
        if initial_total is None:
            initial_total = float(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    final_total = float(loss.item())
    assert final_total < initial_total * 0.4, (
        f"failed to overfit tiny batch: {initial_total:.3f} -> {final_total:.3f}"
    )


def test_multi_task_dataset_wrapper():
    """Wrapper yields the right shapes including the integer K label."""
    feats = torch.randn(8, 2, 4, 257, 20)
    labels = torch.zeros(8, 72)
    azs = torch.full((8, 3), float("nan"))
    for i in range(8):
        n = 1 + (i % 3)
        for j in range(n):
            azs[i, j] = float(j) * 30.0 - 90.0
    base = TensorDataset(feats, labels, azs)
    ds = MultiTaskDataset(
        base, n_mics=4, n_classes=72, max_k=3,
        aug=AugConfig(enable_channel_rotate=True, enable_spec_augment=False),
        train_mode=True,
    )
    feat, lbl, az_pad, k = ds[3]
    assert feat.shape == (2, 4, 257, 20)
    assert lbl.shape == (72,)
    assert az_pad.shape == (3,)
    assert int(k.item()) in (0, 1, 2)
