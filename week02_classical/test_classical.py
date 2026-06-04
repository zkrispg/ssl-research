"""Sanity tests for SRP-PHAT and MUSIC."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from geometry import uniform_circular_array, uniform_linear_array
from music import music
from simulate_array import simulate_freefield
from srp_phat import srp_phat


@pytest.fixture
def uca4():
    return uniform_circular_array(n_mics=4, radius=0.04)


@pytest.fixture
def uca8():
    return uniform_circular_array(n_mics=8, radius=0.05)


@pytest.fixture
def ula4():
    return uniform_linear_array(n_mics=4, spacing=0.04)


@pytest.mark.parametrize("angle", [-150.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0])
def test_srp_phat_uca4_freefield(uca4, angle):
    """4-mic UCA + free-field + 30 dB SNR + speech band: < 3 deg error."""
    signals, _ = simulate_freefield(
        mic_positions=uca4,
        azimuth_deg=angle,
        fs=16000,
        duration=1.0,
        snr_db=30.0,
    )
    est, _, _ = srp_phat(signals, mic_positions=uca4, fs=16000)
    err = ((est - angle + 180) % 360) - 180  # wrap to [-180, 180)
    assert abs(err) < 3.0, f"angle={angle} got {est}, err={err}"


@pytest.mark.parametrize("angle", [-150.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0])
def test_music_uca4_freefield(uca4, angle):
    """MUSIC on the same setup should be at least as accurate as SRP-PHAT."""
    signals, _ = simulate_freefield(
        mic_positions=uca4,
        azimuth_deg=angle,
        fs=16000,
        duration=1.0,
        snr_db=30.0,
    )
    est, _, _ = music(
        signals,
        mic_positions=uca4,
        fs=16000,
        n_sources=1,
        freq_band=(300.0, 3400.0),
    )
    err = ((est - angle + 180) % 360) - 180
    assert abs(err) < 3.0, f"angle={angle} got {est}, err={err}"


def test_srp_phat_low_snr_robustness(uca8):
    """8-mic UCA at SNR=0 dB averaged over angles: MAE < 5 deg.

    More mics improve robustness via the cross-pair sum in SRP-PHAT.
    """
    angles = np.arange(-150, 151, 30, dtype=float)
    errors = []
    for i, ang in enumerate(angles):
        signals, _ = simulate_freefield(
            mic_positions=uca8,
            azimuth_deg=float(ang),
            fs=16000,
            duration=1.0,
            snr_db=0.0,
            seed=i,
        )
        est, _, _ = srp_phat(signals, mic_positions=uca8, fs=16000)
        err = ((est - ang + 180) % 360) - 180
        errors.append(abs(err))
    mae = float(np.mean(errors))
    assert mae < 5.0, f"MAE = {mae:.2f}"


def test_ula4_front_back_ambiguity_resolved_by_grid(ula4):
    """Linear arrays have a front-back ambiguity. We verify that constraining
    the search grid to the front half-plane removes it."""
    angle = 30.0
    signals, _ = simulate_freefield(
        mic_positions=ula4,
        azimuth_deg=angle,
        fs=16000,
        duration=1.0,
        snr_db=30.0,
    )
    est, _, _ = srp_phat(
        signals,
        mic_positions=ula4,
        fs=16000,
        azimuth_grid_deg=np.arange(-90.0, 90.0, 1.0),
    )
    err = abs(est - angle)
    assert err < 3.0, f"got est={est}, err={err}"
