"""Multi-source extensions of SRP-PHAT and MUSIC, plus peak picking.

For both classical methods we already compute a 360-bin spatial spectrum
in W2; multi-source localization is just a matter of finding the K
strongest peaks instead of the global maximum.

A peak qualifies if it is a local maximum on the wrapped 1D azimuth
spectrum and exceeds either an absolute threshold or a relative
fraction of the global max. Peaks within ``min_separation_deg`` of a
stronger peak are suppressed (non-maximum suppression).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_W2 = Path(__file__).resolve().parent.parent / "week02_classical"
if str(_W2) not in sys.path:
    sys.path.insert(0, str(_W2))

from music import music  # noqa: E402
from srp_phat import srp_phat  # noqa: E402


def find_peaks_circular(
    spectrum: np.ndarray,
    grid_deg: np.ndarray,
    n_peaks: int | None = None,
    rel_threshold: float = 0.3,
    min_separation_deg: float = 25.0,
) -> np.ndarray:
    """Find azimuth peaks on a wrapped 1-D spectrum.

    Args:
        spectrum: ``(K,)`` array of non-negative response values.
        grid_deg: ``(K,)`` corresponding azimuth in degrees.
        n_peaks: If given, return at most this many peaks. If ``None``,
            return all peaks above the threshold.
        rel_threshold: Minimum response, as a fraction of the maximum
            response, required for a candidate peak.
        min_separation_deg: Suppress weaker peaks whose azimuth is within
            this distance of a stronger one.

    Returns:
        Array of azimuths of selected peaks, sorted by descending response.
    """
    s = np.asarray(spectrum)
    k = len(s)
    if k == 0:
        return np.empty(0, dtype=np.float32)

    is_local_max = (s >= np.roll(s, 1)) & (s >= np.roll(s, -1))
    candidates = np.where(is_local_max)[0]
    if candidates.size == 0:
        candidates = np.array([int(np.argmax(s))])

    cand_vals = s[candidates]
    threshold = float(rel_threshold * np.max(s))
    keep = cand_vals >= threshold
    candidates = candidates[keep]
    if candidates.size == 0:
        return np.empty(0, dtype=np.float32)

    order = np.argsort(s[candidates])[::-1]
    sorted_idx = candidates[order]

    selected: list[int] = []
    for idx in sorted_idx:
        az = float(grid_deg[idx])
        too_close = False
        for chosen in selected:
            wrap = ((float(grid_deg[chosen]) - az + 180.0) % 360.0) - 180.0
            if abs(wrap) < min_separation_deg:
                too_close = True
                break
        if not too_close:
            selected.append(int(idx))
        if n_peaks is not None and len(selected) >= n_peaks:
            break

    return np.asarray([float(grid_deg[i]) for i in selected], dtype=np.float32)


def srp_phat_multi(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    n_sources: int | None = None,
    rel_threshold: float = 0.5,
    min_separation_deg: float = 25.0,
) -> np.ndarray:
    """Run SRP-PHAT and return the K strongest peaks as azimuths."""
    _, spectrum, grid = srp_phat(signals, mic_positions=mic_positions, fs=fs)
    return find_peaks_circular(
        spectrum,
        grid,
        n_peaks=n_sources,
        rel_threshold=rel_threshold,
        min_separation_deg=min_separation_deg,
    )


def music_multi(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    n_sources: int = 1,
    rel_threshold: float = 0.5,
    min_separation_deg: float = 25.0,
) -> np.ndarray:
    """Run MUSIC with a signal subspace of size ``n_sources`` and pick peaks."""
    _, spectrum, grid = music(
        signals,
        mic_positions=mic_positions,
        fs=fs,
        n_sources=n_sources,
    )
    return find_peaks_circular(
        spectrum,
        grid,
        n_peaks=n_sources,
        rel_threshold=rel_threshold,
        min_separation_deg=min_separation_deg,
    )
