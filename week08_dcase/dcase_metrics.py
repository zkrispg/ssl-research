"""DCASE SELD-style metrics for single-class multi-source DOA estimation.

The DCASE Challenge Task 3 evaluates sound event localization and detection
with four metrics, all computed with a tolerance-based location-aware
matching:

* **F1**: F-score of location-aware true positives (tolerance ``T_deg``).
* **ER**: error rate ``(D + I + S) / N_ref`` where D=deletions (missed
  ground-truth sources), I=insertions (extra predictions), and S=class
  substitutions (always 0 for single-class SSL).
* **LE_CD** (Localization Error, class-dependent): mean angular error on
  TPs in degrees. Smaller is better.
* **LR_CD** (Localization Recall, class-dependent): ``TP / N_ref``. The
  fraction of ground-truth sources that are both detected and localized
  within tolerance.

For our single-class problem, F1 reduces to the F1 we already report,
LE_CD is the same as MAE_TP, and LR_CD equals recall. ER is the new
metric introduced here.

Reference: A. Mesaros et al., "Joint Measurement of Localization and
Detection of Sound Events," IEEE WASPAA 2019.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_W5 = Path(__file__).resolve().parent.parent / "week05_multi_source"
if str(_W5) not in sys.path:
    sys.path.insert(0, str(_W5))

from multi_eval import greedy_match  # noqa: E402


@dataclass
class DcaseSeldStats:
    """Aggregator for DCASE SELD-style metrics.

    Add one prediction/ground-truth pair per call to :meth:`add_sample`,
    then read the four-metric summary from :meth:`summary`. The class
    keeps its own running counts and is unrelated to (but consistent
    with) ``LocalizationStats`` from W5: both compute the same TP/FP/FN
    on the same matching, so F1 will agree.
    """

    tolerance_deg: float = 20.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_ref: int = 0
    angular_errors: list[float] = field(default_factory=list)
    n_samples: int = 0
    correct_count: int = 0

    def add_sample(self, preds_deg: np.ndarray, gts_deg: np.ndarray) -> None:
        matches, unmatched_p, unmatched_g = greedy_match(
            preds_deg, gts_deg, tolerance_deg=self.tolerance_deg
        )
        self.tp += len(matches)
        self.fp += len(unmatched_p)
        self.fn += len(unmatched_g)
        self.n_ref += len(gts_deg)
        for _, _, d in matches:
            self.angular_errors.append(d)
        self.correct_count += int(len(preds_deg) == len(gts_deg))
        self.n_samples += 1

    def summary(self) -> dict[str, float]:
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        # ER counts a substitution as a single error rather than D+I, but
        # for single-class SSL there are no class substitutions; so
        # ER = (D + I) / N_ref where D=FN, I=FP.
        er = (self.fn + self.fp) / max(self.n_ref, 1)
        le_cd = float(np.mean(self.angular_errors)) if self.angular_errors else float("nan")
        lr_cd = self.tp / max(self.n_ref, 1)
        count_acc = self.correct_count / max(self.n_samples, 1)
        return {
            "F1": float(f1),
            "ER": float(er),
            "LE_CD": float(le_cd),
            "LR_CD": float(lr_cd),
            "precision": float(precision),
            "recall": float(recall),
            "count_acc": float(count_acc),
            "n_samples": int(self.n_samples),
            "n_ref": int(self.n_ref),
        }


def format_summary(name: str, summary: dict) -> str:
    """One-line representation suitable for tabular printing."""
    return (
        f"{name:<22}  "
        f"F1={summary['F1']:.3f}  "
        f"ER={summary['ER']:.3f}  "
        f"LE={summary['LE_CD']:6.2f}  "
        f"LR={summary['LR_CD']:.3f}  "
        f"count={summary['count_acc']:.3f}  "
        f"n={summary['n_samples']}"
    )


def overall_seld_score(summary: dict) -> float:
    """DCASE SELD overall score (lower is better).

    The combined metric used to rank DCASE Task 3 systems:

        SELD = 0.25 * (ER + (1 - F1) + LE_norm + (1 - LR_CD))

    where ``LE_norm = LE_CD / 180`` to put the angular error on [0, 1]
    when the tolerance was not exceeded. Returns ``inf`` if ``LE_CD``
    is NaN (no TPs were found).
    """
    le = summary["LE_CD"]
    if not np.isfinite(le):
        return float("inf")
    le_norm = min(le / 180.0, 1.0)
    score = 0.25 * (
        summary["ER"]
        + (1.0 - summary["F1"])
        + le_norm
        + (1.0 - summary["LR_CD"])
    )
    return float(score)
