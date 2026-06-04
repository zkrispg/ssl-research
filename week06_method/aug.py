"""Data augmentations for UCA multi-source SSL.

Two augmentations are implemented; both are applied at the feature level
(after STFT phase computation) and operate on a single sample at a time:

* :func:`channel_rotate` -- exact symmetry of the uniform circular array.
  Cyclically permuting the mic order by ``k`` positions corresponds to
  rotating the azimuth coordinate frame by ``-k * (360 / M)`` degrees.
  We apply the same rotation to the spatial-spectrum label so that
  ground truth and observation stay consistent. Since this is a
  zero-cost augmentation that preserves the physics exactly, it
  effectively multiplies the training set by ``M``.

* :func:`spec_augment` -- standard time/frequency masking on the
  per-channel STFT (Park et al., 2019). Implemented for the sin/cos
  phase representation: the same mask is applied to both sin and cos
  channels of every microphone so the channel-relative phase
  information is preserved on the unmasked bins.
"""
from __future__ import annotations

import numpy as np
import torch


def channel_rotate(
    feat: torch.Tensor,
    label: torch.Tensor,
    k: int,
    n_mics: int,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cyclic mic permutation + matching label rotation.

    Args:
        feat: ``(2, M, F, T)`` tensor.
        label: ``(n_classes,)`` soft multi-hot vector.
        k: number of positions to roll (in [0, n_mics-1]).
        n_mics: number of mics; assumed equal to feature M dim.
        n_classes: number of azimuth bins; must be divisible by n_mics.

    Returns:
        ``(rotated_feat, rotated_label)``.
    """
    if k == 0:
        return feat, label
    if n_classes % n_mics != 0:
        raise ValueError(
            f"n_classes={n_classes} must be divisible by n_mics={n_mics}"
        )
    bin_shift = k * (n_classes // n_mics)
    rotated_feat = torch.roll(feat, shifts=-k, dims=1)  # mic axis
    rotated_label = torch.roll(label, shifts=-bin_shift, dims=-1)
    return rotated_feat, rotated_label


def spec_augment(
    feat: torch.Tensor,
    rng: np.random.Generator,
    n_freq_masks: int = 2,
    max_freq_mask: int = 20,
    n_time_masks: int = 2,
    max_time_mask: int = 5,
) -> torch.Tensor:
    """Random time and frequency masking on a phase feature tensor.

    Args:
        feat: ``(2, M, F, T)`` tensor (sin/cos channels).
        rng: numpy RNG for sampling mask widths and positions.
        n_freq_masks: number of frequency masks to apply.
        max_freq_mask: each freq mask spans up to this many bins.
        n_time_masks: number of time masks.
        max_time_mask: each time mask spans up to this many frames.

    Returns:
        Tensor with masked regions zeroed out (in place).
    """
    feat = feat.clone()
    F = feat.shape[2]
    T = feat.shape[3]
    for _ in range(n_freq_masks):
        f = int(rng.integers(0, max_freq_mask + 1))
        if f == 0 or f >= F:
            continue
        f0 = int(rng.integers(0, F - f))
        feat[:, :, f0 : f0 + f, :] = 0.0
    for _ in range(n_time_masks):
        t = int(rng.integers(0, max_time_mask + 1))
        if t == 0 or t >= T:
            continue
        t0 = int(rng.integers(0, T - t))
        feat[:, :, :, t0 : t0 + t] = 0.0
    return feat
