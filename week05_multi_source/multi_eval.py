"""Localization metrics for multi-source DOA estimation.

We follow the standard SELD evaluation protocol: each predicted azimuth
is matched at most once to a ground-truth azimuth, where a match is
valid only if the angular distance is below ``tolerance_deg``. The
matching minimizes total angular error via greedy nearest-neighbor
assignment, which is exact when the number of sources is small (<=4).

Metrics computed:

* Precision = TP / (TP + FP)
* Recall    = TP / (TP + FN)
* F1        = 2 * P * R / (P + R)
* MAE_TP    = mean angular error over true positives only
* count_acc = fraction of samples with predicted == true source count

These align with the metrics reported in SELD / DCASE Task 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def angular_distance_matrix(preds_deg: np.ndarray, gts_deg: np.ndarray) -> np.ndarray:
    """``(P, G)`` matrix of absolute wrapped angular distances in degrees."""
    p = np.asarray(preds_deg, dtype=np.float32).reshape(-1, 1)
    g = np.asarray(gts_deg, dtype=np.float32).reshape(1, -1)
    return np.abs(((p - g + 180.0) % 360.0) - 180.0)


def greedy_match(
    preds_deg: np.ndarray,
    gts_deg: np.ndarray,
    tolerance_deg: float = 20.0,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy nearest-neighbor matching with a max-distance constraint.

    Returns:
        ``(matches, unmatched_preds, unmatched_gts)`` where ``matches`` is
        a list of ``(pred_idx, gt_idx, distance_deg)`` tuples, all
        distances at most ``tolerance_deg``.
    """
    if len(preds_deg) == 0 or len(gts_deg) == 0:
        return [], list(range(len(preds_deg))), list(range(len(gts_deg)))

    dist = angular_distance_matrix(preds_deg, gts_deg)
    P, G = dist.shape
    used_p, used_g = set(), set()
    matches: list[tuple[int, int, float]] = []

    flat_idx = np.argsort(dist, axis=None)
    for k in flat_idx:
        i, j = int(k // G), int(k % G)
        if i in used_p or j in used_g:
            continue
        d = float(dist[i, j])
        if d > tolerance_deg:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, d))
        if len(used_p) == P or len(used_g) == G:
            break

    unmatched_preds = [i for i in range(P) if i not in used_p]
    unmatched_gts = [j for j in range(G) if j not in used_g]
    return matches, unmatched_preds, unmatched_gts


@dataclass
class LocalizationStats:
    """Aggregator for batch-level localization metrics."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    angular_errors: list[float] = None  # type: ignore[assignment]
    correct_count: int = 0
    n_samples: int = 0

    def __post_init__(self) -> None:
        if self.angular_errors is None:
            self.angular_errors = []

    def add_sample(
        self,
        preds_deg: np.ndarray,
        gts_deg: np.ndarray,
        tolerance_deg: float = 20.0,
    ) -> None:
        matches, unmatched_p, unmatched_g = greedy_match(
            preds_deg, gts_deg, tolerance_deg=tolerance_deg
        )
        self.tp += len(matches)
        self.fp += len(unmatched_p)
        self.fn += len(unmatched_g)
        for _, _, d in matches:
            self.angular_errors.append(d)
        self.correct_count += int(len(preds_deg) == len(gts_deg))
        self.n_samples += 1

    def summary(self) -> dict[str, float]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mae = float(np.mean(self.angular_errors)) if self.angular_errors else float("nan")
        count_acc = self.correct_count / max(self.n_samples, 1)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mae_tp_deg": mae,
            "count_acc": float(count_acc),
            "n_samples": int(self.n_samples),
        }
