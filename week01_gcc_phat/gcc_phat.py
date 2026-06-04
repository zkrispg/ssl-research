"""GCC-PHAT: Generalized Cross-Correlation with Phase Transform.

Reference:
    Knapp & Carter (1976), "The Generalized Correlation Method for Estimation
    of Time Delay", IEEE Trans. ASSP.

Given two microphone signals x1(t), x2(t), the cross-power spectrum is
    S12(f) = X1(f) * conj(X2(f))
The PHAT weighting normalizes the magnitude so only phase information remains:
    R12(f) = S12(f) / |S12(f)|
The TDOA between the two microphones is the peak location of the inverse
Fourier transform of R12(f).

This implementation supports fractional-sample resolution via FFT zero-padding
(``interp`` parameter), which is the standard trick in the literature.
"""
from __future__ import annotations

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s, at ~20 deg C


def gcc_phat_full(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: int,
    max_tau: float | None = None,
    interp: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the full GCC-PHAT cross-correlation function.

    Args:
        sig1, sig2: 1-D arrays of equal length.
        fs: Sampling rate in Hz.
        max_tau: Crop the output to lags within [-max_tau, +max_tau] seconds.
            Pass ``None`` to keep all lags.
        interp: FFT zero-padding factor for sub-sample resolution.

    Returns:
        ``(cc, lags)`` where ``cc[k]`` is the GCC-PHAT correlation at lag
        ``lags[k]`` (in seconds). Lags are sorted in ascending order, so the
        peak at ``argmax(|cc|)`` directly gives ``tau`` under the convention
        "tau > 0 means sig2 is delayed relative to sig1".
    """
    n = sig1.shape[-1] + sig2.shape[-1]

    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R /= np.abs(R) + 1e-15

    cc_raw = np.fft.irfft(R, n=interp * n)

    max_shift = interp * n // 2
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    cropped = np.concatenate((cc_raw[-max_shift:], cc_raw[: max_shift + 1]))
    cc = cropped[::-1]
    lags = (np.arange(len(cc)) - max_shift) / float(interp * fs)
    return cc, lags


def gcc_phat(
    sig1: np.ndarray,
    sig2: np.ndarray,
    fs: int,
    max_tau: float | None = None,
    interp: int = 16,
) -> tuple[float, np.ndarray]:
    """Estimate TDOA between two signals using GCC-PHAT.

    Returns:
        ``(tau, cc)`` where ``tau`` is the estimated TDOA in seconds (positive
        means sig2 is delayed relative to sig1) and ``cc`` is the
        lag-ascending cross-correlation function for visualization.
    """
    cc, lags = gcc_phat_full(sig1, sig2, fs=fs, max_tau=max_tau, interp=interp)
    peak_idx = int(np.argmax(np.abs(cc)))
    tau = float(lags[peak_idx])
    return tau, cc


def tdoa_to_doa(tau: float, mic_distance: float) -> float:
    """Convert TDOA to direction-of-arrival angle (far-field assumption).

    Geometry: two omnidirectional microphones on the x-axis, separated by
    ``mic_distance`` meters. A plane wave arriving from azimuth ``theta``
    (measured from broadside, the +y direction) creates a TDOA
        tau = (mic_distance / c) * sin(theta)
    so theta = arcsin(tau * c / mic_distance).

    Args:
        tau: TDOA in seconds.
        mic_distance: Distance between mics in meters.

    Returns:
        Azimuth in degrees, in [-90, 90]. Returns NaN if ``tau`` is
        outside the physically possible range.
    """
    sin_theta = tau * SPEED_OF_SOUND / mic_distance
    if abs(sin_theta) > 1.0:
        return float("nan")
    return float(np.degrees(np.arcsin(sin_theta)))


def doa_to_tdoa(theta_deg: float, mic_distance: float) -> float:
    """Inverse of :func:`tdoa_to_doa`. Used by the simulator."""
    return mic_distance * np.sin(np.radians(theta_deg)) / SPEED_OF_SOUND
