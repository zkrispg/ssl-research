"""STARSS23 metadata parser + Multi-ACCDOA label tensor builder.

STARSS23 / DCASE 2023 Task 3 metadata format (CSV, no header):

    frame_index, class_id, source_id, azimuth_deg, elevation_deg, distance_cm

  * ``frame_index`` is **1-indexed** at 100 ms label resolution (10 fps).
  * ``class_id`` ranges over ``[0, 13)`` for the DCASE 2023 catalogue
    (13 sound event classes; see :data:`DCASE2023_CLASSES`).
  * ``source_id`` distinguishes simultaneous instances of the same class.
  * ``azimuth_deg`` is in ``(-180, 180]`` with the DCASE convention
    (0 = front, positive = left / counter-clockwise viewed from above).
  * ``elevation_deg`` is in ``[-90, 90]`` (positive = up).
  * ``distance_cm`` is a positive integer (DCASE 2024 distance task uses this).

This module converts a CSV into a *Multi-ACCDOA target tensor* that follows
the SELDnet 2022 reference layout:

    ``target.shape == (T_frames, num_track_dummy=6, num_axis=4, num_classes)``

with ``num_axis = 4`` arranged as ``[activity, x, y, z]`` (activity is 0/1,
xyz is the unit Cartesian DOA). The six "track dummies" cover the three
overlap regimes used by ADPIT (Shimada et al. 2022):

    Track 0  -- A0  : up to 1 source per class            (no same-class overlap)
    Track 1  -- B0  : first  of 2 simultaneous same-class
    Track 2  -- B1  : second of 2 simultaneous same-class
    Track 3  -- C0  : first  of 3 simultaneous same-class
    Track 4  -- C1  : second of 3 simultaneous same-class
    Track 5  -- C2  : third  of 3 simultaneous same-class

DCASE-style class-coupled ADPIT later picks the minimum loss over the 13
(= 1 + 6 + 6) admissible track assignments per class.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

# DCASE 2023 / STARSS23 sound event classes (13 classes, ID 0..12).
# Source: https://dcase.community/challenge2023/task-sound-event-localization-and-detection
DCASE2023_CLASSES: tuple[str, ...] = (
    "Female speech, woman speaking",  # 0
    "Male speech, man speaking",  # 1
    "Clapping",  # 2
    "Telephone",  # 3
    "Laughter",  # 4
    "Domestic sounds",  # 5
    "Walk, footsteps",  # 6
    "Door, open or close",  # 7
    "Music",  # 8
    "Musical instrument",  # 9
    "Water tap, faucet",  # 10
    "Bell",  # 11
    "Knock",  # 12
)
NUM_DCASE2023_CLASSES = len(DCASE2023_CLASSES)

LABEL_HOP_S = 0.1  # 100 ms label resolution (DCASE Task 3 standard)
NUM_TRACK_DUMMY = 6  # 1 (A) + 2 (B) + 3 (C); see module docstring
NUM_AXIS = 4  # [activity, x, y, z]


@dataclass(frozen=True)
class FrameEvent:
    """One CSV row, *0-indexed* frame and standardised types."""

    frame_idx: int  # 0-indexed
    class_id: int
    source_id: int
    azimuth_deg: float
    elevation_deg: float
    distance_cm: float

    @classmethod
    def from_row(cls, row: Iterable[str | int | float]) -> "FrameEvent":
        """Parse a CSV row (1-indexed frame in input) into a FrameEvent."""
        vals = list(row)
        return cls(
            frame_idx=int(vals[0]) - 1,  # convert to 0-indexed
            class_id=int(vals[1]),
            source_id=int(vals[2]),
            azimuth_deg=float(vals[3]),
            elevation_deg=float(vals[4]),
            distance_cm=float(vals[5]) if len(vals) > 5 else 0.0,
        )


def parse_starss_csv(csv_path: str | Path) -> list[FrameEvent]:
    """Parse a STARSS23 metadata CSV file into a list of :class:`FrameEvent`.

    Lines are expected to be comma-separated, *no header*. Empty lines and
    lines starting with ``#`` are ignored.
    """
    csv_path = Path(csv_path)
    events: list[FrameEvent] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                raise ValueError(
                    f"Malformed line in {csv_path}: {raw!r} (need >=5 fields)"
                )
            events.append(FrameEvent.from_row(parts))
    return events


def doa_to_xyz(azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Convert (azimuth, elevation) in degrees to a Cartesian unit vector.

    Uses the DCASE Task 3 convention:
      * azimuth = 0  -> +x  (front)
      * azimuth = 90 -> +y  (left, counter-clockwise from above)
      * elevation = 0  -> horizontal plane (xy)
      * elevation = 90 -> +z (up)

    Returns ``(x, y, z)`` with ``x**2 + y**2 + z**2 == 1`` (up to fp).
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    cos_el = np.cos(el)
    return (float(cos_el * np.cos(az)), float(cos_el * np.sin(az)), float(np.sin(el)))


def doa_array_to_xyz(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Vectorised version of :func:`doa_to_xyz`. Returns ``(..., 3)`` array."""
    az = np.radians(azimuth_deg.astype(np.float64))
    el = np.radians(elevation_deg.astype(np.float64))
    cos_el = np.cos(el)
    return np.stack(
        [cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)], axis=-1
    ).astype(np.float32)


def events_to_multi_accdoa(
    events: list[FrameEvent],
    num_frames: int,
    num_classes: int = NUM_DCASE2023_CLASSES,
) -> torch.Tensor:
    """Convert frame events to a Multi-ACCDOA target tensor.

    Args:
        events: list of :class:`FrameEvent` (0-indexed frames).
        num_frames: total number of label frames in the clip; events with
            ``frame_idx >= num_frames`` are dropped (with a warning skipped
            silently for now -- the dataset class is responsible for matching
            audio length to label length).
        num_classes: dimension of the class axis (default 13 for DCASE 2023).

    Returns:
        ``target`` tensor of shape ``(T_frames, 6, 4, num_classes)`` with
        the 4 axes laid out as ``[activity, x, y, z]``.

    Notes:
        For each frame and class, we group concurrent same-class sources
        and place them into the A/B/C track buckets. If a frame contains
        more than 3 sources of the same class (rare), the extras are
        dropped (last in CSV order).
    """
    target = torch.zeros((num_frames, NUM_TRACK_DUMMY, NUM_AXIS, num_classes), dtype=torch.float32)

    # Group events by (frame, class).
    grouped: dict[tuple[int, int], list[FrameEvent]] = {}
    for ev in events:
        if ev.frame_idx < 0 or ev.frame_idx >= num_frames:
            continue
        if ev.class_id < 0 or ev.class_id >= num_classes:
            continue
        grouped.setdefault((ev.frame_idx, ev.class_id), []).append(ev)

    for (frame, cls), group in grouped.items():
        # Stable order (by source_id) so that "first / second / third" is deterministic.
        group_sorted = sorted(group, key=lambda e: e.source_id)
        n = len(group_sorted)

        if n == 1:
            track_slots = [0]  # A0
        elif n == 2:
            track_slots = [1, 2]  # B0, B1
        else:  # n >= 3 -> C0,C1,C2 ; drop extras
            track_slots = [3, 4, 5]
            group_sorted = group_sorted[:3]

        for slot, ev in zip(track_slots, group_sorted):
            x, y, z = doa_to_xyz(ev.azimuth_deg, ev.elevation_deg)
            target[frame, slot, 0, cls] = 1.0  # activity
            target[frame, slot, 1, cls] = x
            target[frame, slot, 2, cls] = y
            target[frame, slot, 3, cls] = z

    return target


def estimate_num_frames(events: list[FrameEvent], min_frames: int = 0) -> int:
    """Cheap heuristic when the audio file is not yet available.

    Returns ``max(min_frames, max(frame_idx) + 1)``. Useful for unit tests
    on metadata-only fixtures; production code should pass the audio-derived
    length.
    """
    if not events:
        return min_frames
    return max(min_frames, max(ev.frame_idx for ev in events) + 1)


def decode_multi_accdoa(
    pred: torch.Tensor,
    activity_threshold: float = 0.5,
) -> list[list[dict]]:
    """Decode a Multi-ACCDOA *prediction* tensor into per-frame event lists.

    Mirror of :func:`events_to_multi_accdoa`; used at evaluation / inference.

    Args:
        pred: tensor shape ``(B, T, 3, 3, num_classes)`` -- 3 active tracks
            (after class-coupled ADPIT collapses A/B/C into 3 emitted
            tracks), each with 3D ACCDOA per class. The activity is the
            *magnitude* of the (x, y, z) vector (not stored separately, by
            convention).
        activity_threshold: minimum magnitude for a (track, class) entry to
            count as active.

    Returns:
        Nested list: ``out[batch][frame]`` is a list of dicts
        ``{"class_id", "azimuth_deg", "elevation_deg"}``.

    Note:
        We do not yet do unification across simultaneous same-class
        detections (DCASE ``thresh_unify=15deg``); that lives in the
        evaluation module.
    """
    if pred.dim() != 5:
        raise ValueError(f"expected 5-D pred, got shape {tuple(pred.shape)}")
    pred_np = pred.detach().cpu().numpy()  # (B, T, N=3, 3, C)
    B, T, N, _, C = pred_np.shape

    out: list[list[dict]] = []
    for b in range(B):
        per_frame: list[dict] = []
        for t in range(T):
            entries: list[dict] = []
            for n in range(N):
                for c in range(C):
                    vec = pred_np[b, t, n, :, c]
                    mag = float(np.linalg.norm(vec))
                    if mag < activity_threshold:
                        continue
                    az = float(np.degrees(np.arctan2(vec[1], vec[0])))
                    el = float(np.degrees(np.arcsin(np.clip(vec[2] / max(mag, 1e-8), -1.0, 1.0))))
                    entries.append({"class_id": c, "azimuth_deg": az, "elevation_deg": el})
            per_frame.append(entries)
        out.append(per_frame)
    return out
