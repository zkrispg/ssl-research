"""DCASE SELD metrics for STARSS23: class-aware, 3-D angular tolerance.

Implements the four DCASE Task 3 metrics with per-class location-aware
matching:

* **F1_CD**   class-dependent F-score (macro by default).
* **ER_CD**   class-dependent error rate ``(D + I + S) / N_ref``.
* **LE_CD**   class-dependent localisation error (mean angular error on TPs).
* **LR_CD**   class-dependent localisation recall ``TP / N_ref``.
* **SELD score** ``0.25 * (ER + (1 - F1) + LE / 180 + (1 - LR))`` (lower better).

Matching is greedy per class, per frame:
    1. Compute the 3-D angular distance between each (pred, gt) pair of the
       same class.
    2. Sort pair-distances ascending and assign greedily until either side
       runs out or the next-best pair exceeds the tolerance.
    3. Matched-and-within-tolerance pairs count as TPs; remaining preds
       become FPs (insertions); remaining gts become FNs (deletions).

Class substitutions ``S`` are always 0 because matching is class-restricted.

Reference: A. Mesaros et al., "Joint Measurement of Localization and
Detection of Sound Events," IEEE WASPAA 2019; DCASE 2022/2023 Task 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from week11_starss23.seld_labels import doa_to_xyz


def angular_distance_xyz(a: np.ndarray, b: np.ndarray) -> float:
    """Return angular distance in degrees between two unit-norm 3-D vectors."""
    cos_a = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def angular_distance_deg(az1: float, el1: float, az2: float, el2: float) -> float:
    """Angular distance between two (azimuth, elevation) directions, in degrees."""
    a = np.array(doa_to_xyz(az1, el1))
    b = np.array(doa_to_xyz(az2, el2))
    return angular_distance_xyz(a, b)


@dataclass
class SeldEvent:
    """One predicted or ground-truth (class, DOA) entry."""

    class_id: int
    azimuth_deg: float
    elevation_deg: float

    def to_xyz(self) -> np.ndarray:
        return np.array(doa_to_xyz(self.azimuth_deg, self.elevation_deg), dtype=np.float32)


def greedy_match_one_class(
    preds: list[SeldEvent],
    gts: list[SeldEvent],
    tolerance_deg: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy 1:1 matching of predictions and ground truth within one class.

    Returns:
        ``(matches, unmatched_pred, unmatched_gt)`` where ``matches`` is a
        list of ``(pred_idx, gt_idx, angular_err_deg)`` for accepted pairs
        (within tolerance), and the two index lists are the unmatched
        items.
    """
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))

    pairs: list[tuple[int, int, float]] = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            d = angular_distance_xyz(p.to_xyz(), g.to_xyz())
            pairs.append((pi, gi, d))
    pairs.sort(key=lambda t: t[2])

    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for pi, gi, d in pairs:
        if pi in used_p or gi in used_g:
            continue
        if d > tolerance_deg:
            break  # all remaining pairs have d > tolerance
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, d))

    unmatched_p = [i for i in range(len(preds)) if i not in used_p]
    unmatched_g = [i for i in range(len(gts)) if i not in used_g]
    return matches, unmatched_p, unmatched_g


@dataclass
class DcaseSeldStats:
    """Aggregator for class-aware DCASE Task 3 metrics."""

    tolerance_deg: float = 20.0
    n_classes: int = 13

    tp_per_class: np.ndarray = field(init=False)
    fp_per_class: np.ndarray = field(init=False)
    fn_per_class: np.ndarray = field(init=False)
    n_ref_per_class: np.ndarray = field(init=False)
    sum_angular_err_per_class: np.ndarray = field(init=False)
    n_angular_err_per_class: np.ndarray = field(init=False)
    n_frames: int = 0

    def __post_init__(self) -> None:
        self.tp_per_class = np.zeros(self.n_classes, dtype=np.int64)
        self.fp_per_class = np.zeros(self.n_classes, dtype=np.int64)
        self.fn_per_class = np.zeros(self.n_classes, dtype=np.int64)
        self.n_ref_per_class = np.zeros(self.n_classes, dtype=np.int64)
        self.sum_angular_err_per_class = np.zeros(self.n_classes, dtype=np.float64)
        self.n_angular_err_per_class = np.zeros(self.n_classes, dtype=np.int64)

    # ----- accumulation -------------------------------------------------------

    def add_frame(self, preds: Iterable[SeldEvent], gts: Iterable[SeldEvent]) -> None:
        """Add one frame of predictions and ground truth (lists of events)."""
        self.n_frames += 1
        preds_by_cls: dict[int, list[SeldEvent]] = {}
        gts_by_cls: dict[int, list[SeldEvent]] = {}
        for p in preds:
            preds_by_cls.setdefault(p.class_id, []).append(p)
        for g in gts:
            gts_by_cls.setdefault(g.class_id, []).append(g)
            self.n_ref_per_class[g.class_id] += 1

        all_classes = set(preds_by_cls.keys()) | set(gts_by_cls.keys())
        for c in all_classes:
            ps = preds_by_cls.get(c, [])
            gs = gts_by_cls.get(c, [])
            matches, unmatched_p, unmatched_g = greedy_match_one_class(
                ps, gs, self.tolerance_deg
            )
            self.tp_per_class[c] += len(matches)
            self.fp_per_class[c] += len(unmatched_p)
            self.fn_per_class[c] += len(unmatched_g)
            for _, _, d in matches:
                self.sum_angular_err_per_class[c] += d
                self.n_angular_err_per_class[c] += 1

    # ----- summarisation ------------------------------------------------------

    def per_class_metrics(self) -> dict[str, np.ndarray]:
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = self.tp_per_class / np.maximum(
                self.tp_per_class + self.fp_per_class, 1
            )
            recall = self.tp_per_class / np.maximum(
                self.tp_per_class + self.fn_per_class, 1
            )
            f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
            le = np.where(
                self.n_angular_err_per_class > 0,
                self.sum_angular_err_per_class / np.maximum(self.n_angular_err_per_class, 1),
                180.0,  # max angular error if no TPs (penalises silent class)
            )
            lr = self.tp_per_class / np.maximum(self.n_ref_per_class, 1)
            er = (self.fp_per_class + self.fn_per_class) / np.maximum(
                self.n_ref_per_class, 1
            )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "le_cd": le,
            "lr_cd": lr,
            "er_cd": er,
        }

    def summary(self, average: str = "macro") -> dict[str, float]:
        """Reduce per-class metrics to scalars.

        Args:
            average: ``"macro"`` (DCASE default) or ``"micro"``.

        Returns:
            Dict with keys ``f1``, ``er``, ``le_cd``, ``lr_cd``, ``seld``.
        """
        per = self.per_class_metrics()

        # Only consider classes that appear at least once in ground truth
        # (DCASE convention; otherwise silent classes drag macro-LE to 180).
        active = self.n_ref_per_class > 0
        if not active.any():
            return {"f1": 0.0, "er": 1.0, "le_cd": 180.0, "lr_cd": 0.0, "seld": 1.0}

        if average == "macro":
            f1 = float(per["f1"][active].mean())
            le = float(per["le_cd"][active].mean())
            lr = float(per["lr_cd"][active].mean())
            er = float(per["er_cd"][active].mean())
        elif average == "micro":
            tp = int(self.tp_per_class.sum())
            fp = int(self.fp_per_class.sum())
            fn = int(self.fn_per_class.sum())
            n_ref = max(int(self.n_ref_per_class.sum()), 1)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            er = (fp + fn) / n_ref
            n_err = max(int(self.n_angular_err_per_class.sum()), 1)
            le = float(self.sum_angular_err_per_class.sum() / n_err)
            lr = tp / n_ref
        else:
            raise ValueError(f"average must be 'macro' or 'micro', got {average!r}")

        seld = 0.25 * (er + (1 - f1) + le / 180.0 + (1 - lr))
        return {
            "f1": f1,
            "er": float(er),
            "le_cd": le,
            "lr_cd": lr,
            "seld": float(seld),
        }


def decode_pred_to_events(
    pred: np.ndarray,
    activity_threshold: float = 0.5,
    nms_tol_deg: float = 15.0,
) -> list[list[SeldEvent]]:
    """Decode one clip's prediction tensor into per-frame event lists.

    Args:
        pred: ``(T, n_tracks, 3, n_classes)`` array of ACCDOA predictions.
            Track magnitude is the per-class activity; direction is the DOA.
        activity_threshold: minimum vector norm for a (track, class) entry
            to count as active.
        nms_tol_deg: same-class predictions within this angular distance
            are merged (DCASE ``thresh_unify``).

    Returns:
        ``out[t]`` is a list of :class:`SeldEvent` for frame ``t``.
    """
    if pred.ndim != 4:
        raise ValueError(f"expected (T, N, 3, C), got {pred.shape}")
    T, N, _, C = pred.shape
    out: list[list[SeldEvent]] = []
    for t in range(T):
        per_class: dict[int, list[tuple[float, float, float]]] = {}
        for n in range(N):
            for c in range(C):
                vec = pred[t, n, :, c]
                mag = float(np.linalg.norm(vec))
                if mag < activity_threshold:
                    continue
                az = float(np.degrees(np.arctan2(vec[1], vec[0])))
                el = float(
                    np.degrees(np.arcsin(np.clip(vec[2] / max(mag, 1e-8), -1.0, 1.0)))
                )
                per_class.setdefault(c, []).append((mag, az, el))

        events: list[SeldEvent] = []
        for c, items in per_class.items():
            # Sort by magnitude (highest "confidence" first) for greedy NMS.
            items.sort(reverse=True)
            kept_xyz: list[np.ndarray] = []
            for _, az, el in items:
                xyz = np.array(doa_to_xyz(az, el), dtype=np.float32)
                if any(angular_distance_xyz(xyz, k) < nms_tol_deg for k in kept_xyz):
                    continue
                kept_xyz.append(xyz)
                events.append(SeldEvent(class_id=c, azimuth_deg=az, elevation_deg=el))
        out.append(events)
    return out


def target_to_events(target: np.ndarray) -> list[list[SeldEvent]]:
    """Convert a Multi-ACCDOA *target* tensor to per-frame event lists.

    Args:
        target: ``(T, 6, 4, n_classes)`` ground-truth tensor (the output of
            :func:`week11_starss23.seld_labels.events_to_multi_accdoa`).
    """
    if target.ndim != 4:
        raise ValueError(f"expected (T, 6, 4, C), got {target.shape}")
    T, n_dummy, axes, C = target.shape
    if axes != 4:
        raise ValueError(f"expected axis dim 4 ([activity, x, y, z]), got {axes}")
    out: list[list[SeldEvent]] = []
    for t in range(T):
        events: list[SeldEvent] = []
        for n in range(n_dummy):
            for c in range(C):
                if target[t, n, 0, c] < 0.5:
                    continue
                xyz = target[t, n, 1:4, c]
                az = float(np.degrees(np.arctan2(xyz[1], xyz[0])))
                el_input = float(xyz[2] / max(np.linalg.norm(xyz), 1e-8))
                el = float(np.degrees(np.arcsin(np.clip(el_input, -1.0, 1.0))))
                events.append(SeldEvent(class_id=c, azimuth_deg=az, elevation_deg=el))
        out.append(events)
    return out
