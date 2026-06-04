"""Sanity tests for DCASE SELD metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dcase_metrics import DcaseSeldStats, overall_seld_score


def test_perfect_predictions():
    stats = DcaseSeldStats(tolerance_deg=20.0)
    stats.add_sample(np.array([0.0, 90.0]), np.array([0.0, 90.0]))
    stats.add_sample(np.array([45.0]), np.array([45.0]))
    s = stats.summary()
    assert s["F1"] == 1.0
    assert s["ER"] == 0.0
    assert s["LR_CD"] == 1.0
    assert s["LE_CD"] < 1e-6
    assert overall_seld_score(s) < 1e-3


def test_total_miss():
    stats = DcaseSeldStats(tolerance_deg=20.0)
    stats.add_sample(np.empty(0), np.array([0.0, 90.0]))
    stats.add_sample(np.empty(0), np.array([45.0]))
    s = stats.summary()
    assert s["F1"] == 0.0
    assert s["ER"] == 1.0  # 3 deletions / 3 GT
    assert s["LR_CD"] == 0.0
    assert not np.isfinite(s["LE_CD"])
    assert overall_seld_score(s) == float("inf")


def test_extra_predictions_increase_er():
    stats = DcaseSeldStats(tolerance_deg=20.0)
    # 1 GT, 3 predictions, 1 matches: TP=1, FP=2, FN=0, N_ref=1
    stats.add_sample(np.array([0.0, 90.0, -90.0]), np.array([0.0]))
    s = stats.summary()
    assert s["F1"] == 2 * (1 / 3) * 1.0 / ((1 / 3) + 1.0)
    # ER = (FN=0 + FP=2) / N_ref=1 = 2.0
    assert abs(s["ER"] - 2.0) < 1e-6
    assert s["LR_CD"] == 1.0


def test_partial_localization_error():
    stats = DcaseSeldStats(tolerance_deg=20.0)
    # GT at 0; pred at 15 (inside tolerance) -> TP with LE=15
    stats.add_sample(np.array([15.0]), np.array([0.0]))
    s = stats.summary()
    assert s["F1"] == 1.0
    assert abs(s["LE_CD"] - 15.0) < 1e-6


def test_out_of_tolerance_is_fp_and_fn():
    stats = DcaseSeldStats(tolerance_deg=20.0)
    # GT at 0; pred at 90 (outside tolerance) -> FP+FN
    stats.add_sample(np.array([90.0]), np.array([0.0]))
    s = stats.summary()
    assert s["F1"] == 0.0
    assert s["ER"] == 2.0  # 1 FN + 1 FP / 1 N_ref


def test_overall_score_lower_is_better():
    perfect = {"F1": 1.0, "ER": 0.0, "LE_CD": 0.0, "LR_CD": 1.0}
    bad = {"F1": 0.5, "ER": 0.4, "LE_CD": 30.0, "LR_CD": 0.5}
    assert overall_seld_score(perfect) < overall_seld_score(bad)
