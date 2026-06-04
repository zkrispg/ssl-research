"""Unit tests for week11_starss23.seld_metrics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from week11_starss23.seld_labels import doa_to_xyz, events_to_multi_accdoa, FrameEvent
from week11_starss23.seld_metrics import (
    DcaseSeldStats,
    SeldEvent,
    angular_distance_deg,
    angular_distance_xyz,
    decode_pred_to_events,
    greedy_match_one_class,
    target_to_events,
)


# ---------------------------------------------------------------------------
# Angular distance
# ---------------------------------------------------------------------------


def test_angular_distance_xyz_zero_for_same_vector():
    a = np.array([1.0, 0, 0])
    assert angular_distance_xyz(a, a) < 1e-4


def test_angular_distance_xyz_180_for_opposite():
    a = np.array([1.0, 0, 0])
    b = np.array([-1.0, 0, 0])
    assert math.isclose(angular_distance_xyz(a, b), 180.0, abs_tol=1e-3)


def test_angular_distance_xyz_90_orthogonal():
    a = np.array([1.0, 0, 0])
    b = np.array([0.0, 1.0, 0])
    assert math.isclose(angular_distance_xyz(a, b), 90.0, abs_tol=1e-3)


def test_angular_distance_deg_consistent_with_xyz():
    az1, el1, az2, el2 = 30.0, 5.0, -45.0, 10.0
    d = angular_distance_deg(az1, el1, az2, el2)
    a = np.array(doa_to_xyz(az1, el1))
    b = np.array(doa_to_xyz(az2, el2))
    np.testing.assert_allclose(d, angular_distance_xyz(a, b), atol=1e-5)


# ---------------------------------------------------------------------------
# greedy_match_one_class
# ---------------------------------------------------------------------------


def test_greedy_match_perfect_pair():
    preds = [SeldEvent(0, 30.0, 0.0)]
    gts = [SeldEvent(0, 30.0, 0.0)]
    matches, up, ug = greedy_match_one_class(preds, gts, tolerance_deg=20.0)
    assert len(matches) == 1
    assert matches[0][0] == 0 and matches[0][1] == 0
    assert matches[0][2] < 1e-3
    assert up == [] and ug == []


def test_greedy_match_outside_tolerance_unmatched():
    preds = [SeldEvent(0, 30.0, 0.0)]
    gts = [SeldEvent(0, 90.0, 0.0)]  # 60 deg away
    matches, up, ug = greedy_match_one_class(preds, gts, tolerance_deg=20.0)
    assert matches == []
    assert up == [0]
    assert ug == [0]


def test_greedy_match_multi_pair_optimal_assignment():
    preds = [SeldEvent(0, 0.0, 0.0), SeldEvent(0, 90.0, 0.0)]
    gts = [SeldEvent(0, 89.0, 0.0), SeldEvent(0, 1.0, 0.0)]
    matches, up, ug = greedy_match_one_class(preds, gts, tolerance_deg=20.0)
    assert len(matches) == 2
    # Pred 0 (0deg) -> Gt 1 (1deg), Pred 1 (90deg) -> Gt 0 (89deg).
    assignments = {(p, g) for p, g, _ in matches}
    assert assignments == {(0, 1), (1, 0)}


def test_greedy_match_extra_pred_becomes_fp():
    preds = [SeldEvent(0, 0.0, 0.0), SeldEvent(0, 100.0, 0.0)]
    gts = [SeldEvent(0, 0.0, 0.0)]
    matches, up, ug = greedy_match_one_class(preds, gts, tolerance_deg=20.0)
    assert len(matches) == 1
    assert up == [1]
    assert ug == []


def test_greedy_match_extra_gt_becomes_fn():
    preds = [SeldEvent(0, 0.0, 0.0)]
    gts = [SeldEvent(0, 0.0, 0.0), SeldEvent(0, 90.0, 0.0)]
    matches, up, ug = greedy_match_one_class(preds, gts, tolerance_deg=20.0)
    assert len(matches) == 1
    assert up == []
    assert ug == [1]


# ---------------------------------------------------------------------------
# DcaseSeldStats
# ---------------------------------------------------------------------------


def test_dcase_stats_perfect_predictions_yield_perfect_metrics():
    stats = DcaseSeldStats(tolerance_deg=20.0, n_classes=13)
    for _ in range(10):
        ev = [SeldEvent(0, 30.0, 0.0), SeldEvent(5, -45.0, 10.0)]
        stats.add_frame(preds=ev, gts=ev)
    s = stats.summary(average="macro")
    assert math.isclose(s["f1"], 1.0, abs_tol=1e-6)
    assert s["er"] < 1e-6
    assert s["le_cd"] < 1.0
    assert math.isclose(s["lr_cd"], 1.0, abs_tol=1e-6)
    assert s["seld"] < 0.01


def test_dcase_stats_completely_wrong_class_yields_no_match():
    stats = DcaseSeldStats(tolerance_deg=20.0, n_classes=13)
    gt = [SeldEvent(0, 30.0, 0.0)]
    pred = [SeldEvent(7, 30.0, 0.0)]  # right DOA but wrong class
    stats.add_frame(preds=pred, gts=gt)
    s = stats.summary(average="macro")
    # Class 0 has FN, class 7 has FP, neither has TP.
    assert s["f1"] == 0.0
    assert s["lr_cd"] == 0.0
    # Per-class ER macro: only class 0 had refs, ER=1.0 for class 0; class 7 had 0 refs (skipped in macro).
    assert s["er"] >= 1.0


def test_dcase_stats_empty_no_crash():
    stats = DcaseSeldStats()
    stats.add_frame(preds=[], gts=[])
    s = stats.summary()
    # No GT means default-to-silence summary.
    assert s["f1"] == 0.0
    assert s["seld"] == 1.0


def test_dcase_stats_micro_vs_macro_average():
    stats = DcaseSeldStats(tolerance_deg=20.0, n_classes=3)
    # Class 0: 1 perfect TP per frame x 9 frames.
    for _ in range(9):
        stats.add_frame([SeldEvent(0, 0.0, 0.0)], [SeldEvent(0, 0.0, 0.0)])
    # Class 1: 1 missed gt, 0 preds (FN).
    stats.add_frame([], [SeldEvent(1, 30.0, 0.0)])
    macro = stats.summary("macro")
    micro = stats.summary("micro")
    # Macro F1 = (1.0 + 0.0) / 2 = 0.5
    assert math.isclose(macro["f1"], 0.5, abs_tol=1e-6)
    # Micro: tp=9, fp=0, fn=1 -> p=1, r=9/10=0.9, F1 ~ 0.947
    assert micro["f1"] > macro["f1"]


def test_dcase_stats_le_cd_is_average_of_tp_errors():
    stats = DcaseSeldStats(tolerance_deg=20.0, n_classes=2)
    # Two frames with class 0, one matched at 0deg error, one at 10deg error.
    stats.add_frame([SeldEvent(0, 0.0, 0.0)], [SeldEvent(0, 0.0, 0.0)])
    stats.add_frame([SeldEvent(0, 0.0, 0.0)], [SeldEvent(0, 10.0, 0.0)])
    per = stats.per_class_metrics()
    # Class 0 TPs are (0, 10), mean = 5.0 deg
    assert math.isclose(per["le_cd"][0], 5.0, abs_tol=1e-2)


def test_dcase_stats_seld_score_is_average_of_four():
    """SELD score = mean(ER, 1-F1, LE/180, 1-LR)."""
    stats = DcaseSeldStats(tolerance_deg=20.0, n_classes=2)
    # Class 0: 1 TP, 1 FP, 0 FN out of 1 ref -> p=0.5, r=1.0, F1=0.667, ER=1, LR=1
    stats.add_frame(
        [SeldEvent(0, 0.0, 0.0), SeldEvent(0, 90.0, 0.0)],  # 1 TP + 1 FP
        [SeldEvent(0, 0.0, 0.0)],
    )
    s = stats.summary("micro")
    expected_seld = 0.25 * (s["er"] + (1 - s["f1"]) + s["le_cd"] / 180.0 + (1 - s["lr_cd"]))
    assert math.isclose(s["seld"], expected_seld, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# decode_pred_to_events / target_to_events
# ---------------------------------------------------------------------------


def test_decode_pred_to_events_inactive_below_threshold():
    pred = np.zeros((3, 3, 3, 13), dtype=np.float32)
    out = decode_pred_to_events(pred, activity_threshold=0.5)
    assert all(len(frame) == 0 for frame in out)


def test_decode_pred_to_events_recovers_doa():
    pred = np.zeros((1, 3, 3, 13), dtype=np.float32)
    az_in = 30.0
    el_in = 10.0
    x, y, z = doa_to_xyz(az_in, el_in)
    pred[0, 0, 0, 5] = x
    pred[0, 0, 1, 5] = y
    pred[0, 0, 2, 5] = z
    out = decode_pred_to_events(pred)
    assert len(out) == 1
    assert len(out[0]) == 1
    e = out[0][0]
    assert e.class_id == 5
    assert math.isclose(e.azimuth_deg, az_in, abs_tol=0.5)
    assert math.isclose(e.elevation_deg, el_in, abs_tol=0.5)


def test_decode_pred_nms_merges_close_same_class():
    pred = np.zeros((1, 3, 3, 13), dtype=np.float32)
    # Two near-identical (az=30, el=0) entries on different tracks for same class.
    x, y, z = doa_to_xyz(30.0, 0.0)
    pred[0, 0, 0, 7] = x
    pred[0, 0, 1, 7] = y
    pred[0, 0, 2, 7] = z
    pred[0, 1, 0, 7] = x  # track 1: same DOA
    pred[0, 1, 1, 7] = y
    pred[0, 1, 2, 7] = z
    out = decode_pred_to_events(pred, nms_tol_deg=15.0)
    # NMS should keep only one entry for class 7 in this frame.
    assert len(out[0]) == 1


def test_target_to_events_round_trip():
    """events -> multi_accdoa -> target_to_events should return the same events."""
    src = [
        FrameEvent(frame_idx=0, class_id=3, source_id=0, azimuth_deg=45.0, elevation_deg=10.0, distance_cm=200),
        FrameEvent(frame_idx=0, class_id=11, source_id=0, azimuth_deg=-30.0, elevation_deg=-5.0, distance_cm=150),
    ]
    target = events_to_multi_accdoa(src, num_frames=1)
    decoded = target_to_events(target.numpy())
    assert len(decoded) == 1
    by_class = {e.class_id: e for e in decoded[0]}
    assert set(by_class) == {3, 11}
    assert math.isclose(by_class[3].azimuth_deg, 45.0, abs_tol=0.5)
    assert math.isclose(by_class[3].elevation_deg, 10.0, abs_tol=0.5)
    assert math.isclose(by_class[11].azimuth_deg, -30.0, abs_tol=0.5)


# ---------------------------------------------------------------------------
# End-to-end metric pipeline smoke
# ---------------------------------------------------------------------------


def test_end_to_end_perfect_prediction_seld_zero():
    """Pred matches GT exactly -> SELD score should be ~ 0."""
    src = [
        FrameEvent(frame_idx=0, class_id=3, source_id=0, azimuth_deg=45.0, elevation_deg=10.0, distance_cm=200),
        FrameEvent(frame_idx=0, class_id=11, source_id=0, azimuth_deg=-30.0, elevation_deg=-5.0, distance_cm=150),
    ]
    target = events_to_multi_accdoa(src, num_frames=1)
    target_np = target.numpy()
    # Build a synthetic prediction from target's activity-coupled xyz
    # placed in track 0.
    pred = np.zeros((1, 3, 3, 13), dtype=np.float32)
    for c in (3, 11):
        # Find which dummy slot has activity.
        for n_dummy in range(target_np.shape[1]):
            if target_np[0, n_dummy, 0, c] >= 0.5:
                pred[0, 0, :, c] = target_np[0, n_dummy, 1:4, c]
                break

    pred_events = decode_pred_to_events(pred)
    gt_events = target_to_events(target_np)

    stats = DcaseSeldStats(tolerance_deg=20.0, n_classes=13)
    for p_frame, g_frame in zip(pred_events, gt_events):
        stats.add_frame(p_frame, g_frame)

    s = stats.summary()
    assert math.isclose(s["f1"], 1.0, abs_tol=1e-3)
    assert s["er"] < 1e-3
    assert s["le_cd"] < 1.0  # less than 1 deg
    assert math.isclose(s["lr_cd"], 1.0, abs_tol=1e-3)
    assert s["seld"] < 0.01
