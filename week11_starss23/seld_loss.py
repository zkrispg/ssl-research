"""Class-coupled ADPIT loss for STARSS23 / DCASE Multi-ACCDOA.

Extends Shimada 2022 (ICASSP) ADPIT to the 3-D, 13-class STARSS23 setup.

Conventions
-----------
Target tensor (from :mod:`week11_starss23.seld_labels`):

    ``target.shape == (B, T, 6, 4, C)``

with the 6 "track-dummy" slots ``[A0, B0, B1, C0, C1, C2]`` and the 4
axes ``[activity, x, y, z]``. Activity is 0/1.

Prediction tensor (from :class:`SeldCRNN`):

    ``pred.shape == (B, T, n_tracks * 3 * C) = (B, T, 9, C)`` after
    reshape, with the 9 axis ordered ``(track, axis)``: track 0 xyz,
    track 1 xyz, track 2 xyz.

The ADPIT loss enumerates the 13 admissible track-to-source assignment
patterns

    1   (A): one source -> all 3 tracks duplicate A0
    6   (B): two same-class sources at B0, B1 -> tracks pick one of
                   B0B0B1, B0B1B0, B0B1B1, B1B0B0, B1B0B1, B1B1B0
    6   (C): three same-class sources at C0, C1, C2 -> tracks pick one
                   permutation of the three

per ``(batch, frame, class)`` and selects the assignment with the lowest
MSE. The selection is independent across classes within a frame, since
each class has its own activity bookkeeping in STARSS23.
"""
from __future__ import annotations

import torch
from torch import nn


# Indexing into the 6 dummy track slots in the target tensor.
_A0, _B0, _B1, _C0, _C1, _C2 = 0, 1, 2, 3, 4, 5


def _activity_coupled(target: torch.Tensor) -> torch.Tensor:
    """``(B, T, 6, 4, C)`` target -> ``(B, T, 6, 3, C)`` activity-coupled xyz.

    ``activity_coupled[..., k, :, c] = target[..., k, 0, c] * target[..., k, 1:4, c]``,
    i.e. xyz multiplied by per-class binary activity.
    """
    if target.dim() != 5 or target.shape[3] != 4:
        raise ValueError(f"expected target shape (B,T,6,4,C), got {tuple(target.shape)}")
    activity = target[:, :, :, 0:1, :]  # (B, T, 6, 1, C)
    xyz = target[:, :, :, 1:4, :]  # (B, T, 6, 3, C)
    return activity * xyz  # (B, T, 6, 3, C)


def _build_target_patterns(coupled: torch.Tensor) -> torch.Tensor:
    """Stack the 13 ADPIT target patterns along a new leading axis.

    Args:
        coupled: ``(B, T, 6, 3, C)`` activity-coupled targets.

    Returns:
        ``(13, B, T, 9, C)`` -- 13 patterns, each ``(B, T, 9, C)`` where
        the 9 axis is ``(track0_x, track0_y, track0_z, track1_x, ...)``.
    """
    A0 = coupled[:, :, _A0]  # (B, T, 3, C)
    B0 = coupled[:, :, _B0]
    B1 = coupled[:, :, _B1]
    C0 = coupled[:, :, _C0]
    C1 = coupled[:, :, _C1]
    C2 = coupled[:, :, _C2]

    def cat3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, c], dim=2)  # (B, T, 9, C)

    patterns = [
        cat3(A0, A0, A0),  # 0 -- single source, duplicate
        cat3(B0, B0, B1),  # 1 -- 2 sources, B perms
        cat3(B0, B1, B0),  # 2
        cat3(B0, B1, B1),  # 3
        cat3(B1, B0, B0),  # 4
        cat3(B1, B0, B1),  # 5
        cat3(B1, B1, B0),  # 6
        cat3(C0, C1, C2),  # 7 -- 3 sources, C perms
        cat3(C0, C2, C1),  # 8
        cat3(C1, C0, C2),  # 9
        cat3(C1, C2, C0),  # 10
        cat3(C2, C0, C1),  # 11
        cat3(C2, C1, C0),  # 12
    ]
    return torch.stack(patterns, dim=0)  # (13, B, T, 9, C)


class ClassCoupledAdpitLoss(nn.Module):
    """ADPIT loss compatible with Multi-ACCDOA + 13-class STARSS23 targets.

    Args:
        reduction: ``"mean"`` (default) or ``"none"``.

    Forward signature:
        ``loss(pred, target) -> Tensor``

        * ``pred``  : ``(B, T, n_tracks * 3 * C)`` *flat* output of the
                       ACCDOA head, or pre-reshaped ``(B, T, 9, C)``.
        * ``target``: ``(B, T, 6, 4, C)`` activity + xyz targets.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            B, T, total = pred.shape
            C = target.shape[-1]
            if total % (3 * C) != 0:
                raise ValueError(
                    f"pred last dim {total} not divisible by 3*n_classes={3*C}"
                )
            n_tracks = total // (3 * C)
            if n_tracks * 3 * C != total:
                raise ValueError(
                    f"pred dim {total} incompatible with n_classes={C}"
                )
            pred_view = pred.view(B, T, n_tracks, 3, C).reshape(B, T, n_tracks * 3, C)
        elif pred.dim() == 4:
            pred_view = pred  # already (B, T, 9, C)
        else:
            raise ValueError(f"unsupported pred shape {tuple(pred.shape)}")

        coupled = _activity_coupled(target)  # (B, T, 6, 3, C)
        patterns = _build_target_patterns(coupled)  # (13, B, T, 9, C)

        # ADPIT padding (Shimada 2022): for each pattern X, the effective
        # target adds the canonical patterns from the *other* overlap
        # regimes. This prevents the silent-regime trivial-zero from
        # winning the argmin and zeroing out the gradient.
        target_A = patterns[0]  # (B, T, 9, C)
        target_B_canon = patterns[1]  # B0B0B1, the canonical 2-source pattern
        target_C_canon = patterns[7]  # C0C1C2, the canonical 3-source pattern
        pad4A = target_B_canon + target_C_canon  # for pattern A
        pad4B = target_A + target_C_canon  # for any B perm
        pad4C = target_A + target_B_canon  # for any C perm
        pads = torch.stack(
            [pad4A] + [pad4B] * 6 + [pad4C] * 6, dim=0
        )  # (13, B, T, 9, C)
        effective_targets = patterns + pads

        # Per-pattern (B, T, C) MSE: average over the 9-axis.
        diff = pred_view.unsqueeze(0) - effective_targets  # (13, B, T, 9, C)
        per_pattern = diff.pow(2).mean(dim=3)  # (13, B, T, C)

        # Per-(B, T, C) min over patterns.
        min_loss, _ = per_pattern.min(dim=0)  # (B, T, C)

        if self.reduction == "mean":
            return min_loss.mean()
        return min_loss


def select_best_pattern(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return the chosen ADPIT pattern index per ``(B, T, C)``.

    Useful for diagnostics (which pattern fired most per class?).
    """
    if pred.dim() == 3:
        B, T, total = pred.shape
        C = target.shape[-1]
        n_tracks = total // (3 * C)
        pred_view = pred.view(B, T, n_tracks, 3, C).reshape(B, T, n_tracks * 3, C)
    else:
        pred_view = pred

    coupled = _activity_coupled(target)
    patterns = _build_target_patterns(coupled)
    target_A = patterns[0]
    target_B_canon = patterns[1]
    target_C_canon = patterns[7]
    pad4A = target_B_canon + target_C_canon
    pad4B = target_A + target_C_canon
    pad4C = target_A + target_B_canon
    pads = torch.stack([pad4A] + [pad4B] * 6 + [pad4C] * 6, dim=0)
    effective_targets = patterns + pads
    diff = pred_view.unsqueeze(0) - effective_targets
    per_pattern = diff.pow(2).mean(dim=3)  # (13, B, T, C)
    return per_pattern.argmin(dim=0)  # (B, T, C)
