"""Multi-ACCDOA representation and ADPIT loss for multi-source SSL.

Each model output frame produces ``N`` "tracks", each track being a 2-D
ACCDOA vector ``(x, y) = (cos theta, sin theta)`` whose magnitude
encodes per-source activity (~1 when active, ~0 when silent). For a
sample with ``K`` active ground-truth sources, ADPIT (Auxiliary
Duplicating Permutation Invariant Training, Shimada et al. 2022) trains
the network by minimizing the MSE over **all surjective track-to-source
assignments** -- i.e. each source must be covered by at least one
track, and any extra tracks duplicate one of the sources.

For our single-class problem with ``N = max_K = 3`` and ``K in {1, 2, 3}``
the surjection count is small:

* ``K = 1``: 1 assignment   (all three tracks predict the only source)
* ``K = 2``: 6 assignments  (two-of-three duplicate placements)
* ``K = 3``: 6 assignments  (the 3! permutations)

Reference: K. Shimada et al., "Multi-ACCDOA: Localizing and Detecting
Overlapping Sounds from the Same Class with Auxiliary Duplicating
Permutation Invariant Training," ICASSP 2022.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import torch


def surjective_assignments(K: int, N: int) -> list[tuple[int, ...]]:
    """All surjective functions ``f: range(N) -> range(K)``.

    Returns each as a tuple of length ``N`` whose elements are source
    indices in ``[0, K)``. Pre-computed offline; both ``K`` and ``N``
    are small.
    """
    if K == 0:
        return [tuple([0] * N)]  # placeholder; silent sample handled separately
    if K > N:
        raise ValueError(f"K={K} > N={N}: no surjection exists")
    out: list[tuple[int, ...]] = []
    for f in product(range(K), repeat=N):
        if len(set(f)) == K:
            out.append(f)
    return out


# Cache assignments for the typical (K, N) sizes used in training.
_ASSIGNMENT_CACHE: dict[tuple[int, int], torch.LongTensor] = {}


def get_assignment_tensor(K: int, N: int) -> torch.LongTensor:
    """Return ``(num_assignments, N)`` long tensor of source indices."""
    key = (K, N)
    if key not in _ASSIGNMENT_CACHE:
        assigns = surjective_assignments(K, N)
        _ASSIGNMENT_CACHE[key] = torch.tensor(assigns, dtype=torch.long)
    return _ASSIGNMENT_CACHE[key]


def adpit_loss_sample(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    silent_target: float = 0.0,
) -> torch.Tensor:
    """ADPIT loss for one sample.

    Args:
        pred_xy: ``(T, N, 2)`` predicted ACCDOA vectors per frame and track.
        gt_xy: ``(K, 2)`` ground-truth (cos theta, sin theta) per source,
            ``K = 0`` means silence.
        silent_target: Target value for tracks when ``K == 0``. Defaults to 0.

    Returns:
        Scalar loss tensor.
    """
    T, N, _ = pred_xy.shape
    K = gt_xy.shape[0]
    if K == 0:
        target = torch.full_like(pred_xy, silent_target)
        return torch.mean((pred_xy - target) ** 2)

    assigns = get_assignment_tensor(K, N).to(pred_xy.device)  # (A, N)
    # For each assignment, build target (N, 2) = gt_xy[assigns_a]
    target = gt_xy[assigns]  # (A, N, 2)
    # Broadcast across T: (T, A, N, 2)
    target = target.unsqueeze(0).expand(T, -1, -1, -1)
    pred = pred_xy.unsqueeze(1).expand(-1, target.shape[1], -1, -1)
    # Per-assignment per-frame MSE: (A,)
    per_assign = torch.mean((pred - target) ** 2, dim=(0, 2, 3))
    return torch.min(per_assign)


def adpit_loss_batch(
    pred: torch.Tensor,
    gt_padded: torch.Tensor,
) -> torch.Tensor:
    """ADPIT loss averaged over a batch.

    Args:
        pred: ``(B, T, N, 2)`` predicted ACCDOAs.
        gt_padded: ``(B, max_K, 2)`` ground-truth (cos, sin) per source.
            Inactive entries are filled with NaN (use :func:`make_gt_xy`).

    Returns:
        Scalar mean loss.
    """
    B = pred.shape[0]
    losses = []
    for b in range(B):
        valid = ~torch.isnan(gt_padded[b, :, 0])
        gt = gt_padded[b][valid]  # (K, 2)
        losses.append(adpit_loss_sample(pred[b], gt))
    return torch.stack(losses).mean()


def az_to_xy(az_deg: np.ndarray) -> np.ndarray:
    """Convert azimuth (deg) array to ``(..., 2)`` (cos, sin) array."""
    az_rad = np.radians(az_deg)
    return np.stack([np.cos(az_rad), np.sin(az_rad)], axis=-1).astype(np.float32)


def make_gt_xy(az_padded_deg: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, max_K)`` padded azimuth (deg, NaN if inactive) to
    ``(B, max_K, 2)`` (cos, sin) with NaN preserved on inactive slots."""
    out = torch.full((*az_padded_deg.shape, 2), float("nan"), dtype=torch.float32)
    valid = ~torch.isnan(az_padded_deg)
    az_rad = torch.deg2rad(az_padded_deg[valid])
    out[valid] = torch.stack([torch.cos(az_rad), torch.sin(az_rad)], dim=-1)
    return out


def decode_multi_accdoa(
    pred: torch.Tensor,
    activity_threshold: float = 0.5,
    nms_tol_deg: float = 25.0,
) -> list[np.ndarray]:
    """Decode batched per-frame ACCDOA predictions to azimuth lists.

    Steps:
        1. Average predictions across the time axis.
        2. For each track, decide whether it is active by checking the
           per-track magnitude against ``activity_threshold``.
        3. Convert active tracks to azimuths via ``atan2``.
        4. Run a simple NMS to remove duplicates within ``nms_tol_deg``.

    Args:
        pred: ``(B, T, N, 2)`` raw output (no nonlinearity).
        activity_threshold: minimum length to consider a track active.
        nms_tol_deg: angular distance under which two active tracks are
            merged into one.

    Returns:
        List of length ``B``. Each entry is a 1-D array of estimated
        azimuths (degrees) for that sample.
    """
    pred_np = pred.detach().cpu().numpy()
    avg = pred_np.mean(axis=1)  # (B, N, 2)
    activities = np.linalg.norm(avg, axis=-1)  # (B, N)
    azimuths = np.degrees(np.arctan2(avg[..., 1], avg[..., 0]))  # (B, N)

    out: list[np.ndarray] = []
    for b in range(pred_np.shape[0]):
        keep_mask = activities[b] >= activity_threshold
        if not keep_mask.any():
            out.append(np.empty(0, dtype=np.float32))
            continue
        # Sort tracks by activity (descending) for NMS priority.
        idx = np.argsort(activities[b])[::-1]
        idx = idx[keep_mask[idx]]
        chosen: list[float] = []
        for i in idx:
            cand = float(azimuths[b][i])
            too_close = False
            for c in chosen:
                wrap = ((cand - c + 180.0) % 360.0) - 180.0
                if abs(wrap) < nms_tol_deg:
                    too_close = True
                    break
            if not too_close:
                chosen.append(cand)
        out.append(np.asarray(chosen, dtype=np.float32))
    return out
