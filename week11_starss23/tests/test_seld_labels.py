"""Unit tests for week11_starss23.seld_labels."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from week11_starss23.seld_labels import (
    DCASE2023_CLASSES,
    NUM_DCASE2023_CLASSES,
    NUM_TRACK_DUMMY,
    NUM_AXIS,
    FrameEvent,
    decode_multi_accdoa,
    doa_array_to_xyz,
    doa_to_xyz,
    estimate_num_frames,
    events_to_multi_accdoa,
    parse_starss_csv,
)


# ---------------------------------------------------------------------------
# doa_to_xyz / doa_array_to_xyz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "az,el,expected",
    [
        (0.0, 0.0, (1.0, 0.0, 0.0)),  # front
        (90.0, 0.0, (0.0, 1.0, 0.0)),  # left
        (-90.0, 0.0, (0.0, -1.0, 0.0)),  # right
        (180.0, 0.0, (-1.0, 0.0, 0.0)),  # back
        (0.0, 90.0, (0.0, 0.0, 1.0)),  # straight up
        (0.0, -90.0, (0.0, 0.0, -1.0)),  # straight down
    ],
)
def test_doa_to_xyz_cardinal_directions(az, el, expected):
    x, y, z = doa_to_xyz(az, el)
    assert math.isclose(x, expected[0], abs_tol=1e-6)
    assert math.isclose(y, expected[1], abs_tol=1e-6)
    assert math.isclose(z, expected[2], abs_tol=1e-6)


def test_doa_to_xyz_unit_norm():
    rng = np.random.default_rng(0)
    for _ in range(100):
        az = rng.uniform(-180, 180)
        el = rng.uniform(-90, 90)
        x, y, z = doa_to_xyz(az, el)
        assert math.isclose(x * x + y * y + z * z, 1.0, abs_tol=1e-6)


def test_doa_array_to_xyz_matches_scalar():
    rng = np.random.default_rng(42)
    az = rng.uniform(-180, 180, size=10)
    el = rng.uniform(-90, 90, size=10)
    arr = doa_array_to_xyz(az, el)
    assert arr.shape == (10, 3)
    assert arr.dtype == np.float32
    for i in range(10):
        x, y, z = doa_to_xyz(float(az[i]), float(el[i]))
        np.testing.assert_allclose(arr[i], (x, y, z), atol=1e-5)


# ---------------------------------------------------------------------------
# FrameEvent.from_row
# ---------------------------------------------------------------------------


def test_frame_event_from_row_six_fields():
    ev = FrameEvent.from_row(["1", "8", "0", "14", "0", "392"])
    assert ev.frame_idx == 0  # 1-indexed -> 0-indexed
    assert ev.class_id == 8
    assert ev.source_id == 0
    assert ev.azimuth_deg == 14.0
    assert ev.elevation_deg == 0.0
    assert ev.distance_cm == 392.0


def test_frame_event_from_row_five_fields_no_distance():
    ev = FrameEvent.from_row(["10", "1", "2", "-45.5", "12.0"])
    assert ev.frame_idx == 9
    assert ev.distance_cm == 0.0  # missing -> default 0


# ---------------------------------------------------------------------------
# parse_starss_csv
# ---------------------------------------------------------------------------


def test_parse_starss_csv_real_format(tmp_path: Path):
    p = tmp_path / "fold4_room23_mix001.csv"
    p.write_text(
        "1,8,0,14,0,392\n"
        "1,5,0,-37,-18,205\n"
        "2,8,0,14,0,392\n"
        "2,1,1,-88,2,66\n"
        "\n"  # blank line in the middle
        "# comment line that should be skipped\n"
        "3,5,0,-37,-18,205\n",
        encoding="utf-8",
    )
    events = parse_starss_csv(p)
    assert len(events) == 5
    assert events[0].class_id == 8
    assert events[0].frame_idx == 0  # 1 -> 0
    assert events[2].azimuth_deg == 14.0
    assert events[3].source_id == 1
    assert events[-1].frame_idx == 2  # 3 -> 2


def test_parse_starss_csv_malformed(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed line"):
        parse_starss_csv(p)


def test_parse_starss_csv_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_starss_csv(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# events_to_multi_accdoa
# ---------------------------------------------------------------------------


def test_events_to_multi_accdoa_shape_and_dtype():
    events = [FrameEvent(frame_idx=0, class_id=0, source_id=0, azimuth_deg=0, elevation_deg=0, distance_cm=100)]
    target = events_to_multi_accdoa(events, num_frames=10)
    assert target.shape == (10, NUM_TRACK_DUMMY, NUM_AXIS, NUM_DCASE2023_CLASSES)
    assert target.dtype == torch.float32


def test_events_to_multi_accdoa_single_source_goes_to_a0():
    """One source per (frame, class) -> placed in track 0 (A0)."""
    events = [
        FrameEvent(frame_idx=0, class_id=8, source_id=0, azimuth_deg=14, elevation_deg=0, distance_cm=392),
    ]
    target = events_to_multi_accdoa(events, num_frames=2)

    # Track 0 (A0), class 8 should be active.
    assert target[0, 0, 0, 8].item() == 1.0
    # Cartesian conversion of (14 deg, 0 deg).
    x, y, z = doa_to_xyz(14.0, 0.0)
    assert math.isclose(target[0, 0, 1, 8].item(), x, abs_tol=1e-5)
    assert math.isclose(target[0, 0, 2, 8].item(), y, abs_tol=1e-5)
    assert math.isclose(target[0, 0, 3, 8].item(), z, abs_tol=1e-5)

    # Other tracks for that (frame, class) are zero.
    for n in range(1, NUM_TRACK_DUMMY):
        assert target[0, n, :, 8].abs().sum().item() == 0.0
    # Other frames are zero.
    assert target[1].abs().sum().item() == 0.0


def test_events_to_multi_accdoa_two_same_class_goes_to_b0_b1():
    """Two simultaneous same-class sources -> tracks 1 (B0) and 2 (B1)."""
    events = [
        FrameEvent(frame_idx=0, class_id=1, source_id=0, azimuth_deg=10, elevation_deg=0, distance_cm=100),
        FrameEvent(frame_idx=0, class_id=1, source_id=1, azimuth_deg=170, elevation_deg=0, distance_cm=200),
    ]
    target = events_to_multi_accdoa(events, num_frames=1)

    assert target[0, 1, 0, 1].item() == 1.0  # B0
    assert target[0, 2, 0, 1].item() == 1.0  # B1
    assert target[0, 0, 0, 1].item() == 0.0  # A0 NOT used
    for n in (3, 4, 5):
        assert target[0, n, :, 1].abs().sum().item() == 0.0  # C* NOT used


def test_events_to_multi_accdoa_three_same_class_goes_to_c012():
    events = [
        FrameEvent(frame_idx=0, class_id=2, source_id=i, azimuth_deg=30 * i, elevation_deg=0, distance_cm=100)
        for i in range(3)
    ]
    target = events_to_multi_accdoa(events, num_frames=1)
    for n in (3, 4, 5):
        assert target[0, n, 0, 2].item() == 1.0
    for n in (0, 1, 2):
        assert target[0, n, :, 2].abs().sum().item() == 0.0


def test_events_to_multi_accdoa_drops_overflow_above_three():
    """More than 3 same-class sources in one frame -> drop the excess."""
    events = [
        FrameEvent(frame_idx=0, class_id=2, source_id=i, azimuth_deg=20 * i, elevation_deg=0, distance_cm=100)
        for i in range(5)
    ]
    target = events_to_multi_accdoa(events, num_frames=1)
    # Only C0/C1/C2 active, no other tracks.
    assert target[0, 3, 0, 2].item() == 1.0
    assert target[0, 4, 0, 2].item() == 1.0
    assert target[0, 5, 0, 2].item() == 1.0


def test_events_to_multi_accdoa_different_classes_independent():
    """Two sources of *different* classes both go to A0, but on different class slots."""
    events = [
        FrameEvent(frame_idx=0, class_id=0, source_id=0, azimuth_deg=10, elevation_deg=0, distance_cm=100),
        FrameEvent(frame_idx=0, class_id=8, source_id=0, azimuth_deg=-50, elevation_deg=15, distance_cm=300),
    ]
    target = events_to_multi_accdoa(events, num_frames=1)
    assert target[0, 0, 0, 0].item() == 1.0  # class 0 in A0
    assert target[0, 0, 0, 8].item() == 1.0  # class 8 in A0


def test_events_to_multi_accdoa_drops_out_of_range_frames():
    events = [
        FrameEvent(frame_idx=99, class_id=0, source_id=0, azimuth_deg=0, elevation_deg=0, distance_cm=100),
    ]
    target = events_to_multi_accdoa(events, num_frames=10)
    assert target.abs().sum().item() == 0.0  # silently dropped, no crash


def test_events_to_multi_accdoa_drops_invalid_class():
    events = [
        FrameEvent(frame_idx=0, class_id=99, source_id=0, azimuth_deg=0, elevation_deg=0, distance_cm=100),
    ]
    target = events_to_multi_accdoa(events, num_frames=1, num_classes=NUM_DCASE2023_CLASSES)
    assert target.abs().sum().item() == 0.0


def test_events_to_multi_accdoa_round_trip_via_decode():
    """Build a target, decode it, expect the original DOAs back."""
    events = [
        FrameEvent(frame_idx=0, class_id=3, source_id=0, azimuth_deg=45.0, elevation_deg=10.0, distance_cm=200),
        FrameEvent(frame_idx=0, class_id=11, source_id=0, azimuth_deg=-30.0, elevation_deg=-5.0, distance_cm=150),
    ]
    target = events_to_multi_accdoa(events, num_frames=1)

    # Synthesise the "model output" 3-track form by collapsing A0/B0/C0 into
    # the first emitted track. For our single-source case that's just track 0.
    pred = torch.zeros((1, 1, 3, 3, NUM_DCASE2023_CLASSES))
    # The decode_multi_accdoa expects (B, T, 3 emitted, 3 xyz, C).
    # Map A0 (target track 0) -> emitted track 0, copying xyz only:
    pred[0, 0, 0, :, :] = target[0, 0, 1:, :]  # axis 1: -> xyz
    decoded = decode_multi_accdoa(pred, activity_threshold=0.5)

    assert len(decoded) == 1  # batch size 1
    assert len(decoded[0]) == 1  # 1 frame
    found_classes = {e["class_id"] for e in decoded[0][0]}
    assert found_classes == {3, 11}
    # Check angles roughly match (decode does atan2 / asin on unit vectors).
    by_class = {e["class_id"]: e for e in decoded[0][0]}
    assert math.isclose(by_class[3]["azimuth_deg"], 45.0, abs_tol=0.5)
    assert math.isclose(by_class[3]["elevation_deg"], 10.0, abs_tol=0.5)
    assert math.isclose(by_class[11]["azimuth_deg"], -30.0, abs_tol=0.5)
    assert math.isclose(by_class[11]["elevation_deg"], -5.0, abs_tol=0.5)


# ---------------------------------------------------------------------------
# estimate_num_frames
# ---------------------------------------------------------------------------


def test_estimate_num_frames_empty():
    assert estimate_num_frames([]) == 0
    assert estimate_num_frames([], min_frames=100) == 100


def test_estimate_num_frames_with_events():
    events = [
        FrameEvent(frame_idx=5, class_id=0, source_id=0, azimuth_deg=0, elevation_deg=0, distance_cm=0),
        FrameEvent(frame_idx=20, class_id=0, source_id=0, azimuth_deg=0, elevation_deg=0, distance_cm=0),
    ]
    assert estimate_num_frames(events) == 21
    assert estimate_num_frames(events, min_frames=10) == 21
    assert estimate_num_frames(events, min_frames=50) == 50


# ---------------------------------------------------------------------------
# Real STARSS23 metadata smoke test
# ---------------------------------------------------------------------------


_REAL_METADATA_DIR = Path("D:/ssl-research/data/STARSS23/metadata_dev/metadata_dev")


@pytest.mark.skipif(not _REAL_METADATA_DIR.exists(), reason="STARSS23 metadata not on disk")
def test_real_starss23_metadata_smoke():
    """Smoke test against an actual STARSS23 CSV from the dev set."""
    csvs = list(_REAL_METADATA_DIR.rglob("*.csv"))
    assert len(csvs) > 0, "no CSVs found under metadata_dev"

    # Parse the first one.
    sample = csvs[0]
    events = parse_starss_csv(sample)
    assert len(events) > 0
    # All class IDs should be valid for DCASE 2023.
    cls_ids = {ev.class_id for ev in events}
    assert cls_ids.issubset(set(range(NUM_DCASE2023_CLASSES))), (
        f"unexpected class IDs {cls_ids - set(range(NUM_DCASE2023_CLASSES))} in {sample}"
    )
    # Azimuth in (-180, 180], elevation in [-90, 90].
    for ev in events:
        assert -180 <= ev.azimuth_deg <= 180
        assert -90 <= ev.elevation_deg <= 90
    # Build the target tensor on a generous frame budget.
    n_frames = estimate_num_frames(events, min_frames=600)
    target = events_to_multi_accdoa(events, n_frames)
    assert target.shape == (n_frames, NUM_TRACK_DUMMY, NUM_AXIS, NUM_DCASE2023_CLASSES)
    assert target.abs().sum().item() > 0  # something is active
