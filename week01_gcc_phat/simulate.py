"""Synthetic two-microphone signal generator.

Generates a far-field plane-wave signal arriving from a specified azimuth,
applies the corresponding TDOA between two omnidirectional microphones, and
optionally adds white noise. Used to verify the GCC-PHAT implementation
against a known ground truth before moving to real or simulated room data.
"""
from __future__ import annotations

import numpy as np

from gcc_phat import doa_to_tdoa


def make_source(
    duration: float,
    fs: int,
    band: tuple[float, float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Generate a band-limited noise source signal.

    Args:
        duration: Length in seconds.
        fs: Sampling rate.
        band: ``(low_hz, high_hz)`` band-pass range, or ``None`` for full-band
            white noise. Wideband sources are the best case for GCC-PHAT.
            Speech-band 300-3400 Hz reflects telephony quality, while
            100-8000 Hz reflects typical wideband ASR conditions.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    src = rng.standard_normal(n).astype(np.float32)
    if band is not None:
        SRC = np.fft.rfft(src)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        keep = (freqs >= band[0]) & (freqs <= band[1])
        SRC[~keep] = 0.0
        src = np.fft.irfft(SRC, n=n).astype(np.float32)
    src /= np.max(np.abs(src)) + 1e-9
    return src


def make_speech_like_source(duration: float, fs: int, seed: int = 0) -> np.ndarray:
    """Telephony-band 300-3400 Hz noise. Kept for backwards compatibility."""
    return make_source(duration, fs, band=(300.0, 3400.0), seed=seed)


def simulate_two_mic(
    azimuth_deg: float,
    mic_distance: float = 0.1,
    fs: int = 16000,
    duration: float = 1.0,
    snr_db: float = 30.0,
    source: np.ndarray | None = None,
    source_band: tuple[float, float] | None = (300.0, 3400.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate a two-channel recording of a far-field source.

    Args:
        azimuth_deg: True direction of arrival, in degrees, measured from
            the array broadside (perpendicular to the mic axis). Range
            [-90, 90].
        mic_distance: Distance between mics in meters.
        fs: Sampling rate.
        duration: Signal length in seconds.
        snr_db: Signal-to-noise ratio of additive Gaussian noise.
        source: Optional pre-existing source signal. If None, generates a
            speech-like band-limited noise.
        seed: RNG seed for reproducibility.

    Returns:
        ``(sig1, sig2, true_tau)`` where ``true_tau`` is the ground-truth
        TDOA in seconds. The first channel is the reference (no delay), the
        second channel is delayed by ``true_tau``.
    """
    rng = np.random.default_rng(seed + 1)

    if source is None:
        source = make_source(duration, fs, band=source_band, seed=seed)

    true_tau = doa_to_tdoa(azimuth_deg, mic_distance)
    delay_samples = true_tau * fs

    n = source.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    SRC = np.fft.rfft(source)
    SIG2 = SRC * np.exp(-1j * 2 * np.pi * freqs * delay_samples / fs)
    sig2 = np.fft.irfft(SIG2, n=n).astype(np.float32)
    sig1 = source.astype(np.float32)

    sig_power = np.mean(sig1**2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power)
    sig1 = sig1 + rng.standard_normal(n).astype(np.float32) * noise_std
    sig2 = sig2 + rng.standard_normal(n).astype(np.float32) * noise_std

    return sig1, sig2, true_tau
