"""SpecAugment-style data augmentation for STARSS23 SELD features.

Implements time and frequency masking on a 10-channel feature tensor of
shape ``(n_ch, T, F)`` (or batched ``(B, n_ch, T, F)``):

* Channels 0-3: 4-mic log-mel (real frequency axis -> freq masks valid).
* Channels 4-9: 6 GCC-PHAT pairs (lag axis, not freq -> freq masks
  optional). Time masks apply to both because time is meaningful for
  both feature types.

References:
    Park et al., "SpecAugment", Interspeech 2019.
    DCASE 2023 Task 3 baseline applies SpecAugment to log-mel only.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SpecAugmentConfig:
    """Hyperparameters for ``SpecAugment``.

    Defaults follow the DCASE 2023 Task 3 baseline (mild masking suitable
    for ~ minute-long clips at 100 ms label hop).
    """

    n_time_masks: int = 2
    time_mask_max: int = 50  # max consecutive time frames per mask
    n_freq_masks: int = 2
    freq_mask_max: int = 16  # max consecutive freq bins per mask
    n_logmel_channels: int = 4  # apply freq mask only to the first N channels
    apply_freq_mask_to_gcc: bool = False  # if True, apply freq mask to all channels
    p: float = 1.0  # probability of applying augmentation per call

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        if self.time_mask_max < 0 or self.freq_mask_max < 0:
            raise ValueError("mask_max sizes must be non-negative")
        if self.n_time_masks < 0 or self.n_freq_masks < 0:
            raise ValueError("n_*_masks must be non-negative")


class SpecAugment(nn.Module):
    """SpecAugment for SELD log-mel + GCC-PHAT inputs.

    Applies time and frequency masking only when ``self.training`` is True
    and a Bernoulli ``p`` draw succeeds. In eval mode this module is a
    no-op.

    Args:
        cfg: hyperparameters; see :class:`SpecAugmentConfig`.
    """

    def __init__(self, cfg: SpecAugmentConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SpecAugmentConfig()

    # NOTE: implemented in pure PyTorch for autograd compatibility.
    # We zero-fill rather than mean-fill for simplicity; both are
    # acceptable in the literature.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if self.cfg.p < 1.0 and torch.rand(1).item() > self.cfg.p:
            return x

        cfg = self.cfg
        squeeze_back = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        elif x.ndim != 4:
            raise ValueError(f"expected 3-D or 4-D tensor, got {x.shape}")

        x = x.clone()
        B, C, T, F = x.shape
        n_logmel = min(cfg.n_logmel_channels, C)

        # ---- time masks (all channels, per-sample independent draws) ----
        for _ in range(cfg.n_time_masks):
            if cfg.time_mask_max == 0:
                break
            for b in range(B):
                t_len = int(torch.randint(0, cfg.time_mask_max + 1, (1,)).item())
                if t_len == 0:
                    continue
                t_start = int(torch.randint(0, max(1, T - t_len + 1), (1,)).item())
                x[b, :, t_start:t_start + t_len, :] = 0.0

        # ---- frequency masks (log-mel channels only by default) ---------
        for _ in range(cfg.n_freq_masks):
            if cfg.freq_mask_max == 0:
                break
            for b in range(B):
                f_len = int(torch.randint(0, cfg.freq_mask_max + 1, (1,)).item())
                if f_len == 0:
                    continue
                f_start = int(torch.randint(0, max(1, F - f_len + 1), (1,)).item())
                if cfg.apply_freq_mask_to_gcc:
                    x[b, :, :, f_start:f_start + f_len] = 0.0
                else:
                    x[b, :n_logmel, :, f_start:f_start + f_len] = 0.0

        return x.squeeze(0) if squeeze_back else x

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"n_time_masks={c.n_time_masks}, time_mask_max={c.time_mask_max}, "
            f"n_freq_masks={c.n_freq_masks}, freq_mask_max={c.freq_mask_max}, "
            f"n_logmel_channels={c.n_logmel_channels}, p={c.p}"
        )


# ---------------------------------------------------------------------------
# Strength presets for the ICASSP ablation
# ---------------------------------------------------------------------------

def specaug_config_for_strength(strength: str) -> SpecAugmentConfig:
    """Return a :class:`SpecAugmentConfig` for the named strength preset.

    Presets:
        * ``"strong"``  -- DCASE 2023 baseline default (Park et al. 2019).
                          2 time masks @ 50 frames, 2 freq masks @ 16 bins.
        * ``"weak"``    -- half-strength control (mask widths halved). Used
                          to verify that the catastrophic SpecAug result on
                          STARSS23 is not strength-cherry-picked.
        * ``"off"``     -- masks disabled (forward becomes identity even in
                          training mode); useful for debug / sanity checks.

    Raises:
        ValueError: if ``strength`` is not one of the above.
    """
    s = strength.lower()
    if s in {"strong", "default", "dcase"}:
        return SpecAugmentConfig()  # canonical defaults
    if s == "weak":
        return SpecAugmentConfig(
            n_time_masks=2,
            time_mask_max=25,
            n_freq_masks=2,
            freq_mask_max=8,
        )
    if s == "off":
        return SpecAugmentConfig(
            n_time_masks=0,
            time_mask_max=0,
            n_freq_masks=0,
            freq_mask_max=0,
        )
    raise ValueError(f"unknown specaug strength: {strength!r} (use strong/weak/off)")
