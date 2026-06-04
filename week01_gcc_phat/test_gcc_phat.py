"""Sanity tests for the GCC-PHAT implementation.

Run with: ``pytest week01_gcc_phat/test_gcc_phat.py -v``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gcc_phat import doa_to_tdoa, gcc_phat, tdoa_to_doa
from simulate import simulate_two_mic


def test_doa_tdoa_roundtrip():
    for theta in [-60.0, -30.0, 0.0, 30.0, 45.0, 60.0]:
        tau = doa_to_tdoa(theta, mic_distance=0.1)
        theta_back = tdoa_to_doa(tau, mic_distance=0.1)
        assert abs(theta - theta_back) < 1e-6


def test_gcc_phat_zero_tdoa():
    fs = 16000
    sig1, sig2, true_tau = simulate_two_mic(
        azimuth_deg=0.0, mic_distance=0.1, fs=fs, snr_db=40.0
    )
    tau_hat, _ = gcc_phat(sig1, sig2, fs=fs, max_tau=0.1 / 343.0 * 1.05)
    assert abs(tau_hat - true_tau) * 1e6 < 20.0  # within 20 us


@pytest.mark.parametrize("angle", [-60.0, -30.0, 30.0, 60.0])
def test_gcc_phat_high_snr_accuracy(angle):
    """GCC-PHAT MAE at 30 dB SNR is typically 1-3 deg. The arcsin
    nonlinearity amplifies sub-sample quantization error at large angles."""
    fs = 16000
    mic_distance = 0.1
    sig1, sig2, true_tau = simulate_two_mic(
        azimuth_deg=angle, mic_distance=mic_distance, fs=fs, snr_db=30.0
    )
    tau_hat, _ = gcc_phat(
        sig1, sig2, fs=fs, max_tau=mic_distance / 343.0 * 1.05, interp=16
    )
    est = tdoa_to_doa(tau_hat, mic_distance)
    assert abs(est - angle) < 3.0, f"angle={angle} got {est}"


@pytest.mark.parametrize("angle", [-60.0, -30.0, 0.0, 30.0, 60.0])
def test_gcc_phat_wideband_subdegree_accuracy(angle):
    """Algorithm-level correctness: with full-band white noise and high SNR,
    GCC-PHAT should achieve sub-degree accuracy at all azimuths. This isolates
    the algorithm from the bandwidth-limited resolution floor seen with
    speech-band signals."""
    fs = 16000
    mic_distance = 0.1
    sig1, sig2, _ = simulate_two_mic(
        azimuth_deg=angle,
        mic_distance=mic_distance,
        fs=fs,
        snr_db=60.0,
        source_band=None,
    )
    tau_hat, _ = gcc_phat(
        sig1, sig2, fs=fs, max_tau=mic_distance / 343.0 * 1.05, interp=64
    )
    est = tdoa_to_doa(tau_hat, mic_distance)
    assert abs(est - angle) < 1.0, f"angle={angle} got {est}"


def test_gcc_phat_moderate_snr_robustness():
    """At SNR=10 dB, errors should still average below ~3 deg."""
    fs = 16000
    mic_distance = 0.1
    angles = np.arange(-60, 61, 10, dtype=float)
    errors = []
    for i, theta in enumerate(angles):
        sig1, sig2, _ = simulate_two_mic(
            azimuth_deg=float(theta),
            mic_distance=mic_distance,
            fs=fs,
            snr_db=10.0,
            seed=i,
        )
        tau_hat, _ = gcc_phat(
            sig1, sig2, fs=fs, max_tau=mic_distance / 343.0 * 1.05
        )
        est = tdoa_to_doa(tau_hat, mic_distance)
        errors.append(abs(est - float(theta)))
    mae = float(np.mean(errors))
    assert mae < 3.0, f"MAE at 10 dB SNR was {mae:.2f} deg, expected < 3"
