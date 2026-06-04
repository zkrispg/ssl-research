"""Multi-channel STFT features for CNN-based DOA estimation.

Following Chakrabarty & Habets (2019), the SSL CNN is fed only with phase
information from the STFT. We use the sin/cos of the phase as a 2-channel
representation, which avoids the wraparound discontinuity at +/- pi while
preserving all phase content.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft


def multichannel_stft(
    signals: np.ndarray,
    fs: int,
    n_fft: int = 512,
    hop_length: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the multichannel STFT.

    Args:
        signals: ``(M, N)`` waveforms.
        fs, n_fft, hop_length: STFT parameters.

    Returns:
        ``(X, freqs)`` where ``X`` has shape ``(M, F, T)`` complex64 and
        ``freqs`` is the per-bin frequency in Hz.
    """
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


def phase_features(
    signals: np.ndarray,
    fs: int,
    n_fft: int = 512,
    hop_length: int = 256,
    return_freqs: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute (sin, cos) of the STFT phase for each microphone.

    Returns:
        Array of shape ``(2, M, F, T)`` where channel 0 is sin(phase) and
        channel 1 is cos(phase). If ``return_freqs`` is True, also returns
        the frequency axis.
    """
    X, freqs = multichannel_stft(signals, fs=fs, n_fft=n_fft, hop_length=hop_length)
    phase = np.angle(X).astype(np.float32)
    feat = np.stack([np.sin(phase), np.cos(phase)], axis=0).astype(np.float32)
    if return_freqs:
        return feat, freqs
    return feat


def random_frame(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pick one random time frame from a feature tensor of shape (..., T)."""
    t = features.shape[-1]
    idx = int(rng.integers(0, t))
    return features[..., idx]
