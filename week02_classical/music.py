"""Broadband MUSIC for far-field azimuth estimation.

Reference:
    Schmidt (1986), "Multiple Emitter Location and Signal Parameter
    Estimation," IEEE Trans. AP.

Algorithm (per frequency bin f):

1. Compute the spatial covariance matrix
       R(f) = (1/T) * sum_t X(t, f) X(t, f)^H
   where X(t, f) is the M-channel STFT vector.
2. Eigendecompose ``R(f)`` and split into a signal subspace (top ``n_sources``
   eigenvectors) and a noise subspace ``E_n(f)``.
3. For each candidate direction theta, build the steering vector
       a(theta, f)_m = exp(+j 2 pi f * (p_m . u(theta)) / c)
   and evaluate the MUSIC pseudospectrum
       P(theta, f) = 1 / (a^H E_n E_n^H a).

Broadband fusion: sum the denominator across frequencies and take the
reciprocal. This is the "incoherent" combination which is simple, robust,
and the most common choice for speech sources.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

SPEED_OF_SOUND = 343.0


def _compute_stft(
    signals: np.ndarray,
    fs: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, freqs) where X has shape (M, F, T)."""
    freqs, _, X = stft(
        signals,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window="hann",
        boundary=None,
        padded=False,
        axis=-1,
    )
    return X.astype(np.complex64), freqs.astype(np.float32)


def music(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    azimuth_grid_deg: np.ndarray | None = None,
    n_sources: int = 1,
    n_fft: int = 512,
    hop_length: int = 256,
    freq_band: tuple[float, float] = (300.0, 3400.0),
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate the source azimuth using broadband MUSIC.

    Args:
        signals: ``(M, N)`` multichannel waveforms.
        mic_positions: ``(M, 2)`` mic coordinates in meters.
        fs: Sampling rate.
        azimuth_grid_deg: Candidate azimuths in degrees. Default is a
            1-degree grid covering ``[-180, 180)``.
        n_sources: Number of sources (size of the signal subspace).
        n_fft: STFT window length.
        hop_length: STFT hop length.
        freq_band: ``(low, high)`` Hz; only frequency bins in this band
            contribute to the broadband pseudospectrum.

    Returns:
        ``(best_azimuth_deg, pseudospectrum, azimuth_grid_deg)``.
    """
    if azimuth_grid_deg is None:
        azimuth_grid_deg = np.arange(-180.0, 180.0, 1.0)

    M = signals.shape[0]
    X, freqs = _compute_stft(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    # X shape: (M, F, T)

    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    f_indices = np.where(band_mask)[0]
    if len(f_indices) == 0:
        raise ValueError(f"No frequency bins in band {freq_band} for fs={fs}, n_fft={n_fft}")

    azimuth_rad = np.radians(azimuth_grid_deg)
    cos_t = np.cos(azimuth_rad)
    sin_t = np.sin(azimuth_rad)
    # mic_proj[m, k] = p_m . u(theta_k)
    mic_proj = mic_positions[:, 0:1] * cos_t[None, :] + mic_positions[:, 1:2] * sin_t[None, :]
    # shape (M, K)

    denom = np.zeros(len(azimuth_grid_deg), dtype=np.float64)

    for f_idx in f_indices:
        f_hz = float(freqs[f_idx])
        Xf = X[:, f_idx, :]  # (M, T)
        T = Xf.shape[-1]
        if T == 0:
            continue

        Rf = (Xf @ Xf.conj().T) / T  # (M, M)
        # Hermitian eigendecomposition; eigenvalues ascending.
        eigvals, eigvecs = np.linalg.eigh(Rf)
        # Noise subspace: smallest (M - n_sources) eigenvectors.
        En = eigvecs[:, : M - n_sources]  # (M, M - K)

        phase = 2 * np.pi * f_hz * mic_proj / SPEED_OF_SOUND  # (M, K)
        a = np.exp(1j * phase) / np.sqrt(M)  # (M, K), normalized

        proj = En.conj().T @ a  # (M-K, K)
        denom += np.sum(np.abs(proj) ** 2, axis=0).astype(np.float64)

    pseudospectrum = 1.0 / (denom + 1e-12)
    best_idx = int(np.argmax(pseudospectrum))
    return float(azimuth_grid_deg[best_idx]), pseudospectrum, np.asarray(azimuth_grid_deg, dtype=float)
