"""Microphone array geometries.

All positions are returned in meters as ``(M, 2)`` arrays in the xy-plane.
The convention used in the rest of the project:

* The array is centered at the origin.
* The +x axis is "front" of the array; +y is to its left (right-hand rule
  with z up).
* Source azimuth ``theta`` is measured counter-clockwise from +x, so
  ``theta = 0`` is "directly in front", ``theta = +90`` is to the left.
"""
from __future__ import annotations

import numpy as np


def uniform_circular_array(n_mics: int, radius: float) -> np.ndarray:
    """Uniform circular array with ``n_mics`` mics on a circle of given radius.

    Mic 0 is at angle 0 (i.e., on the +x axis); the rest are spaced uniformly
    counter-clockwise.
    """
    angles = np.arange(n_mics) * 2.0 * np.pi / n_mics
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def uniform_linear_array(n_mics: int, spacing: float) -> np.ndarray:
    """Uniform linear array along the y-axis, centered at the origin.

    Placing the array along y means broadside is +x, which keeps the
    azimuth convention consistent with :func:`uniform_circular_array`.
    """
    ys = (np.arange(n_mics) - (n_mics - 1) / 2.0) * spacing
    return np.stack([np.zeros(n_mics), ys], axis=1)


def array_aperture(mic_positions: np.ndarray) -> float:
    """Maximum pairwise distance between mics. Used for ``max_tau``."""
    diffs = mic_positions[:, None, :] - mic_positions[None, :, :]
    return float(np.max(np.linalg.norm(diffs, axis=-1)))
