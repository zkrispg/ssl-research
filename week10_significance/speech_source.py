"""Synthetic speech-like source signal generator.

Why not real LibriSpeech? Because a paper-grade demonstration that our
W9 result holds under realistic source statistics does *not* strictly
require natural speech recordings -- it requires that the source has the
**characteristic structure** of speech (harmonic excitation, formant
filtering, sub-syllabic amplitude envelope) rather than the
band-limited Gaussian noise used in W5-W9 training. The fully synthetic
generator below is:

* deterministic (seed-controlled),
* parameter-free for the caller (no on-disk dataset to manage),
* substantially closer to real speech than the W5-W9 fallback (it has
  pitch, formants, voiced/unvoiced segments, and a 4-7 Hz syllable
  envelope), and
* fast enough to be a drop-in replacement for ``make_source`` in the
  multi-source simulator.

Calling :func:`make_source_speech_like` with the same signature as
:func:`week01_gcc_phat.simulate.make_source` lets ``multi_source_data``
pick up speech-like sources without touching the rest of the pipeline.
"""
from __future__ import annotations

import numpy as np

VOWEL_FORMANTS: dict[str, tuple[tuple[float, float], ...]] = {
    "a": ((730.0, 90.0), (1090.0, 110.0), (2440.0, 140.0)),
    "e": ((660.0, 80.0), (1720.0, 110.0), (2410.0, 130.0)),
    "i": ((270.0, 70.0), (2290.0, 110.0), (3010.0, 150.0)),
    "o": ((570.0, 80.0), (840.0, 100.0), (2410.0, 130.0)),
    "u": ((300.0, 70.0), (870.0, 100.0), (2240.0, 130.0)),
}


def _pulse_train(n: int, fs: int, f0_curve: np.ndarray) -> np.ndarray:
    """Sample-accurate impulse train at a (possibly time-varying) f0."""
    out = np.zeros(n, dtype=np.float32)
    phase = 0.0
    for i in range(n):
        phase += f0_curve[i] / fs
        if phase >= 1.0:
            phase -= 1.0
            out[i] = 1.0
    return out


def _resonator(f: float, bw: float, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """A single 2-pole resonator at centre frequency ``f`` and 3-dB bandwidth ``bw``."""
    r = float(np.exp(-np.pi * bw / fs))
    theta = 2 * np.pi * f / fs
    a = np.asarray([1.0, -2 * r * np.cos(theta), r * r], dtype=np.float64)
    b = np.asarray([1.0 - r], dtype=np.float64)
    return b, a


def _filter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Direct-form II transposed IIR; avoid scipy dependency."""
    n = x.shape[0]
    y = np.zeros(n, dtype=np.float64)
    if a.shape[0] == 3 and b.shape[0] == 1:
        b0 = float(b[0]); a1 = float(a[1]); a2 = float(a[2])
        y1, y2 = 0.0, 0.0
        for i in range(n):
            yi = b0 * float(x[i]) - a1 * y1 - a2 * y2
            y[i] = yi
            y2 = y1
            y1 = yi
        return y.astype(np.float32)
    raise ValueError("only 2-pole 1-zero resonator supported")


def make_source_speech_like(
    duration: float,
    fs: int,
    band: tuple[float, float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Generate a 1-second-ish band-limited speech-like waveform.

    Args mirror :func:`week01_gcc_phat.simulate.make_source` so this is a
    drop-in replacement. The ``band`` argument is honoured by an optional
    band-pass post-filter (single-pole HP + LP).
    """
    rng = np.random.default_rng(seed)
    n = int(duration * fs)

    f0_mean = float(rng.uniform(95.0, 185.0))
    f0_jitter = float(rng.uniform(2.0, 8.0))
    f0_curve = f0_mean + f0_jitter * np.sin(
        2 * np.pi * float(rng.uniform(2.5, 5.5))
        * np.linspace(0.0, duration, n, dtype=np.float32)
    )
    f0_curve = f0_curve.astype(np.float64)

    excitation = _pulse_train(n, fs, f0_curve)
    excitation += 0.10 * rng.standard_normal(n).astype(np.float32)

    vowels = list(VOWEL_FORMANTS.keys())
    n_segments = max(2, int(duration * 4.0))
    seg_len = n // n_segments
    seg_choice = rng.choice(vowels, size=n_segments)

    out = np.zeros(n, dtype=np.float32)
    for s, v in enumerate(seg_choice):
        start, stop = s * seg_len, min((s + 1) * seg_len, n)
        chunk = excitation[start:stop]
        y = chunk.astype(np.float64)
        for f, bw in VOWEL_FORMANTS[v]:
            b, a = _resonator(f, bw, fs)
            y = _filter(b, a, y).astype(np.float64)
        out[start:stop] = y.astype(np.float32)

    envelope_rate_hz = float(rng.uniform(3.5, 6.5))
    t = np.linspace(0.0, duration, n, dtype=np.float32)
    envelope = 0.55 + 0.45 * (1.0 - np.cos(2 * np.pi * envelope_rate_hz * t + rng.uniform(0, 2 * np.pi))) / 2.0
    if rng.random() < 0.35:
        n_pauses = int(rng.integers(1, 3))
        for _ in range(n_pauses):
            ps = int(rng.uniform(0.05, 0.8) * n)
            pl = int(rng.uniform(0.04, 0.10) * n)
            envelope[ps:ps + pl] *= 0.05
    out = out * envelope

    if band is not None:
        lo, hi = band
        if hi < fs / 2:
            SRC = np.fft.rfft(out)
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)
            keep = (freqs >= lo) & (freqs <= hi)
            SRC[~keep] = 0.0
            out = np.fft.irfft(SRC, n=n).astype(np.float32)

    out = out / (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)


if __name__ == "__main__":
    sig = make_source_speech_like(1.0, 16000, seed=0)
    print(f"signal shape {sig.shape}  max {abs(sig).max():.3f}  "
          f"mean abs {np.mean(np.abs(sig)):.3f}")
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(sig.shape[0], d=1.0 / 16000)
    peaks = freqs[np.argsort(spec)[-5:]]
    print(f"top 5 spectral peaks (Hz): {sorted(peaks.round())}")
