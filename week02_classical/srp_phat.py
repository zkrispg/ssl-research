"""SRP-PHAT: Steered Response Power with PHAT weighting.

Reference:
    DiBiase, Silverman, Brandstein (2001), "Robust Localization in
    Reverberant Rooms," in Microphone Arrays.

For an array of M microphones, SRP-PHAT computes the GCC-PHAT cross-
correlation R_ij(tau) for each microphone pair (i, j). For each candidate
direction-of-arrival theta, it predicts the corresponding TDOA between
each pair of mics under the far-field plane-wave model and sums the GCC-PHAT
values evaluated at those predicted lags:

    SRP(theta) = sum_{i<j} R_ij(tau_ij(theta))

where tau_ij(theta) = (p_i - p_j) . u(theta) / c, p_m is the position of
mic m, and u(theta) = (cos theta, sin theta) is the unit vector pointing
towards the source.

The estimate is the theta that maximizes SRP. This implementation is
restricted to azimuth-only (2D) localization.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_W1 = Path(__file__).resolve().parent.parent / "week01_gcc_phat"
if str(_W1) not in sys.path:
    sys.path.insert(0, str(_W1))

from gcc_phat import gcc_phat_full  # noqa: E402

SPEED_OF_SOUND = 343.0


def srp_phat(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    azimuth_grid_deg: np.ndarray | None = None,
    max_tau: float | None = None,
    interp: int = 16,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate the source azimuth using SRP-PHAT.

    Args:
        signals: ``(M, N)`` multichannel waveforms.
        mic_positions: ``(M, 2)`` mic coordinates in meters.
        fs: Sampling rate.
        azimuth_grid_deg: Candidate azimuths in degrees. Default is a 1-degree
            grid covering the full circle ``[-180, 180)``.
        max_tau: Crop GCC-PHAT to this maximum lag in seconds. Default is
            ``aperture / c``.
        interp: GCC-PHAT sub-sample interpolation factor.

    Returns:
        ``(best_azimuth_deg, srp_spectrum, azimuth_grid_deg)``. ``srp_spectrum``
        has the same shape as ``azimuth_grid_deg``.
    """
    if azimuth_grid_deg is None:
        azimuth_grid_deg = np.arange(-180.0, 180.0, 1.0)

    M = signals.shape[0]
    if max_tau is None:
        diffs = mic_positions[:, None, :] - mic_positions[None, :, :]
        max_tau = float(np.max(np.linalg.norm(diffs, axis=-1))) / SPEED_OF_SOUND * 1.05

    pairs = list(combinations(range(M), 2))
    cc_per_pair: list[np.ndarray] = []
    lags_per_pair: list[np.ndarray] = []
    for i, j in pairs:
        cc, lags = gcc_phat_full(signals[i], signals[j], fs=fs, max_tau=max_tau, interp=interp)
        cc_per_pair.append(cc)
        lags_per_pair.append(lags)

    azimuth_rad = np.radians(azimuth_grid_deg)
    cos_t = np.cos(azimuth_rad)
    sin_t = np.sin(azimuth_rad)

    srp = np.zeros_like(azimuth_grid_deg, dtype=float)
    for k_pair, (i, j) in enumerate(pairs):
        delta = mic_positions[i] - mic_positions[j]
        tau_ij = (delta[0] * cos_t + delta[1] * sin_t) / SPEED_OF_SOUND
        cc = cc_per_pair[k_pair]
        lags = lags_per_pair[k_pair]
        # Clip to valid lag range so np.interp does not extrapolate.
        tau_clip = np.clip(tau_ij, lags[0], lags[-1])
        srp += np.interp(tau_clip, lags, cc)

    best_idx = int(np.argmax(srp))
    return float(azimuth_grid_deg[best_idx]), srp, np.asarray(azimuth_grid_deg, dtype=float)
