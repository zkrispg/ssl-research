"""Unit tests for week11_starss23.seld_augment."""
from __future__ import annotations

import pytest
import torch

from week11_starss23.seld_augment import (
    SpecAugment,
    SpecAugmentConfig,
    specaug_config_for_strength,
)


# ---------------------------------------------------------------------------
# Strength presets (Tier 2 ablation)
# ---------------------------------------------------------------------------

def test_strength_strong_matches_default():
    cfg = specaug_config_for_strength("strong")
    default = SpecAugmentConfig()
    assert cfg == default


def test_strength_weak_is_half_of_strong():
    weak = specaug_config_for_strength("weak")
    strong = specaug_config_for_strength("strong")
    assert weak.time_mask_max == strong.time_mask_max // 2
    assert weak.freq_mask_max == strong.freq_mask_max // 2
    assert weak.n_time_masks == strong.n_time_masks
    assert weak.n_freq_masks == strong.n_freq_masks


def test_strength_off_disables_masks():
    cfg = specaug_config_for_strength("off")
    assert cfg.time_mask_max == 0
    assert cfg.freq_mask_max == 0
    assert cfg.n_time_masks == 0
    assert cfg.n_freq_masks == 0


def test_strength_unknown_raises():
    with pytest.raises(ValueError):
        specaug_config_for_strength("medium")
    with pytest.raises(ValueError):
        specaug_config_for_strength("")


def test_strength_aliases_accepted():
    """The 'strong' preset can also be requested as 'default' or 'dcase'."""
    a = specaug_config_for_strength("strong")
    b = specaug_config_for_strength("default")
    c = specaug_config_for_strength("dcase")
    assert a == b == c


def test_strength_off_module_is_identity_on_training_mode():
    """Even in training mode, an 'off' SpecAugment should pass inputs unchanged."""
    cfg = specaug_config_for_strength("off")
    aug = SpecAugment(cfg).train()
    x = torch.randn(2, 10, 50, 64)
    y = aug(x)
    assert torch.equal(x, y)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_default_is_valid():
    SpecAugmentConfig()


def test_config_rejects_p_out_of_range():
    with pytest.raises(ValueError):
        SpecAugmentConfig(p=1.5)
    with pytest.raises(ValueError):
        SpecAugmentConfig(p=-0.1)


def test_config_rejects_negative_mask_size():
    with pytest.raises(ValueError):
        SpecAugmentConfig(time_mask_max=-1)
    with pytest.raises(ValueError):
        SpecAugmentConfig(freq_mask_max=-3)


def test_config_rejects_negative_n_masks():
    with pytest.raises(ValueError):
        SpecAugmentConfig(n_time_masks=-1)


# ---------------------------------------------------------------------------
# Forward shape / dtype
# ---------------------------------------------------------------------------


def test_forward_preserves_shape_3d():
    cfg = SpecAugmentConfig()
    aug = SpecAugment(cfg).train()
    x = torch.randn(10, 100, 64)
    y = aug(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_forward_preserves_shape_batched():
    cfg = SpecAugmentConfig()
    aug = SpecAugment(cfg).train()
    x = torch.randn(4, 10, 100, 64)
    y = aug(x)
    assert y.shape == x.shape


def test_forward_rejects_invalid_ndim():
    aug = SpecAugment().train()
    with pytest.raises(ValueError):
        aug(torch.randn(10, 64))  # 2-D


# ---------------------------------------------------------------------------
# Eval-mode no-op
# ---------------------------------------------------------------------------


def test_eval_mode_is_noop():
    cfg = SpecAugmentConfig(n_time_masks=10, time_mask_max=80,
                            n_freq_masks=10, freq_mask_max=60)
    aug = SpecAugment(cfg).eval()
    x = torch.randn(10, 100, 64)
    y = aug(x)
    torch.testing.assert_close(y, x)


def test_p_zero_is_noop():
    # Probability of *applying* the augmentation is 0.
    cfg = SpecAugmentConfig(p=0.0)
    aug = SpecAugment(cfg).train()
    x = torch.randn(10, 100, 64)
    y = aug(x)
    torch.testing.assert_close(y, x)


# ---------------------------------------------------------------------------
# Masking actually zeros things
# ---------------------------------------------------------------------------


def test_time_mask_actually_zeros_at_least_one_slice():
    cfg = SpecAugmentConfig(
        n_time_masks=2, time_mask_max=20,
        n_freq_masks=0, freq_mask_max=0,
    )
    aug = SpecAugment(cfg).train()
    torch.manual_seed(0)
    x = torch.ones(10, 100, 64)
    y = aug(x)
    # At least some time slice should now be zero across all channels.
    zero_per_t = (y == 0).all(dim=(0, 2))  # (T,)
    assert int(zero_per_t.sum()) > 0


def test_freq_mask_actually_zeros_logmel_only_by_default():
    cfg = SpecAugmentConfig(
        n_time_masks=0, time_mask_max=0,
        n_freq_masks=2, freq_mask_max=8,
        n_logmel_channels=4, apply_freq_mask_to_gcc=False,
    )
    aug = SpecAugment(cfg).train()
    torch.manual_seed(0)
    x = torch.ones(10, 100, 64)
    y = aug(x)
    # Channels 0-3 (log-mel) should have at least one fully-zero freq column.
    logmel_zero_per_f = (y[:4] == 0).all(dim=(0, 1))  # (F,)
    assert int(logmel_zero_per_f.sum()) > 0
    # Channels 4-9 (GCC-PHAT) should be untouched.
    torch.testing.assert_close(y[4:], x[4:])


def test_freq_mask_can_apply_to_gcc():
    cfg = SpecAugmentConfig(
        n_time_masks=0, time_mask_max=0,
        n_freq_masks=2, freq_mask_max=8,
        n_logmel_channels=4, apply_freq_mask_to_gcc=True,
    )
    aug = SpecAugment(cfg).train()
    torch.manual_seed(0)
    x = torch.ones(10, 100, 64)
    y = aug(x)
    # All 10 channels should now have at least one fully-zero freq column.
    all_zero_per_f = (y == 0).all(dim=(0, 1))
    assert int(all_zero_per_f.sum()) > 0


# ---------------------------------------------------------------------------
# Mask sizes are bounded
# ---------------------------------------------------------------------------


def test_total_zeroed_time_bounded():
    cfg = SpecAugmentConfig(
        n_time_masks=2, time_mask_max=20,
        n_freq_masks=0, freq_mask_max=0,
    )
    aug = SpecAugment(cfg).train()
    torch.manual_seed(42)
    x = torch.ones(10, 100, 64)
    y = aug(x)
    # Each of the two masks zeros at most 20 time frames; total <= 40.
    zero_per_t = (y == 0).all(dim=(0, 2))
    assert int(zero_per_t.sum()) <= 2 * cfg.time_mask_max


# ---------------------------------------------------------------------------
# Independence across batch
# ---------------------------------------------------------------------------


def test_batched_samples_get_independent_masks():
    cfg = SpecAugmentConfig(n_time_masks=2, time_mask_max=20,
                            n_freq_masks=2, freq_mask_max=16)
    aug = SpecAugment(cfg).train()
    torch.manual_seed(0)
    x = torch.ones(4, 10, 100, 64)
    y = aug(x)
    # The 4 batched samples should not all have identical mask patterns.
    diffs = (y != y[0:1]).any().item()
    assert diffs


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_gradient_flows_through_unmasked_regions():
    cfg = SpecAugmentConfig(n_time_masks=1, time_mask_max=10,
                            n_freq_masks=1, freq_mask_max=8)
    aug = SpecAugment(cfg).train()
    torch.manual_seed(7)
    x = torch.randn(10, 100, 64, requires_grad=True)
    y = aug(x)
    loss = y.pow(2).sum()
    loss.backward()
    # Unmasked positions should receive non-zero gradient.
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
