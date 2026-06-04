"""STARSS23 SELD feature extraction: log-mel + GCC-PHAT (mic) / IV (FOA).

Follows the SELDnet 2022 / DCASE Task 3 reference convention:

    Sample rate          : 24 000 Hz
    STFT hop             :   480 samples (20 ms)
    Label hop            : 2 400 samples (100 ms = 5x feature hop)
    Mel bins             :    64 (50 Hz .. 12 kHz)
    GCC-PHAT lags        :    64 (centered, +/- 32 samples)

Two array formats are supported via ``array_type``:

    "mic"  (default)  Microphone array (Eigenmike-style 4-mic). Stack:
                       4 log-mel + 6 mic-pair GCC-PHAT = 10 channels.

    "foa"             First-Order Ambisonics (W, X, Y, Z). Stack:
                       4 log-mel + 3 mel-pooled intensity-vector dims (Ix,
                       Iy, Iz, energy-normalised, projected to mel scale).
                       Total = 7 channels. The DCASE 2023 baseline default.

Output tensor convention (per audio clip):

    feat.shape == (n_channels, T_features, n_freq=64)

with ``T_features`` at the *feature* (20 ms) resolution -- 5x denser than
the label resolution. The model is responsible for pooling 5x in time to
align with the Multi-ACCDOA target produced by ``seld_labels.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

DEFAULT_FS = 24_000
DEFAULT_HOP_S = 0.02
DEFAULT_LABEL_HOP_S = 0.1
DEFAULT_N_FFT = 1024
DEFAULT_N_MELS = 64
DEFAULT_GCC_LAGS = 64  # centered, +/- 32 samples
DEFAULT_FMIN = 50.0
DEFAULT_FMAX_RATIO = 0.5  # nyquist


@dataclass(frozen=True)
class SeldFeatureConfig:
    """Hyper-parameters for SELD feature extraction.

    ``array_type`` selects between microphone array (default) and First-Order
    Ambisonics. The feature stack and channel count depend on this choice;
    see :func:`extract_seld_features` for layout details.
    """

    fs: int = DEFAULT_FS
    n_fft: int = DEFAULT_N_FFT
    hop_s: float = DEFAULT_HOP_S
    label_hop_s: float = DEFAULT_LABEL_HOP_S
    n_mels: int = DEFAULT_N_MELS
    n_gcc_lags: int = DEFAULT_GCC_LAGS
    fmin: float = DEFAULT_FMIN
    fmax: float | None = None  # None -> fs/2
    n_mics: int = 4
    array_type: str = "mic"  # "mic" (default, GCC-PHAT) or "foa" (intensity vector)

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_s * self.fs))

    @property
    def label_hop_samples(self) -> int:
        return int(round(self.label_hop_s * self.fs))

    @property
    def feature_per_label_ratio(self) -> int:
        ratio = int(round(self.label_hop_s / self.hop_s))
        if abs(ratio * self.hop_s - self.label_hop_s) > 1e-9:
            raise ValueError(
                f"label_hop_s ({self.label_hop_s}) must be an integer multiple "
                f"of hop_s ({self.hop_s})"
            )
        return ratio

    def n_feature_channels(self) -> int:
        if self.array_type == "mic":
            n_pairs = self.n_mics * (self.n_mics - 1) // 2
            return self.n_mics + n_pairs
        if self.array_type == "foa":
            # 4 log-mel (W,X,Y,Z) + 3 intensity-vector dims (Ix,Iy,Iz)
            if self.n_mics != 4:
                raise ValueError(
                    f"array_type=foa requires n_mics=4 (W,X,Y,Z), got {self.n_mics}"
                )
            return self.n_mics + 3
        raise ValueError(f"unknown array_type: {self.array_type!r}")

    def effective_fmax(self) -> float:
        return float(self.fmax) if self.fmax is not None else self.fs * DEFAULT_FMAX_RATIO


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------


def load_multichannel_audio(
    wav_path: str | Path,
    target_fs: int = DEFAULT_FS,
    n_mics: int = 4,
) -> np.ndarray:
    """Load a multi-channel WAV, resampling to ``target_fs`` if needed.

    Returns:
        ``audio`` of shape ``(n_mics, n_samples)``, dtype float32 in [-1, 1].

    Raises:
        ValueError: if the file does not have at least ``n_mics`` channels.
    """
    wav_path = Path(wav_path)
    audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")  # (n_samples, n_ch)
    if audio.shape[1] < n_mics:
        raise ValueError(
            f"{wav_path} has only {audio.shape[1]} channels, need {n_mics}"
        )
    audio = audio[:, :n_mics].T  # (n_mics, n_samples)
    if sr != target_fs:
        # librosa.resample expects (..., n_samples)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_fs, axis=-1)
        audio = audio.astype(np.float32, copy=False)
    return audio


# ---------------------------------------------------------------------------
# STFT (shared between log-mel and GCC-PHAT)
# ---------------------------------------------------------------------------


def _stft_per_channel(audio: np.ndarray, cfg: SeldFeatureConfig) -> np.ndarray:
    """Compute STFT for each channel.

    Args:
        audio: ``(n_mics, n_samples)`` float32.

    Returns:
        Complex ``(n_mics, n_freq=n_fft/2+1, T_features)`` array.
    """
    out = []
    for ch in range(audio.shape[0]):
        spec = librosa.stft(
            y=audio[ch],
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_samples,
            win_length=cfg.n_fft,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
        out.append(spec)
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Log-Mel
# ---------------------------------------------------------------------------


def extract_logmel(
    audio: np.ndarray | None = None,
    *,
    cfg: SeldFeatureConfig = SeldFeatureConfig(),
    stft: np.ndarray | None = None,
) -> np.ndarray:
    """Per-channel log-mel spectrogram.

    Either ``audio`` or a pre-computed ``stft`` (complex) must be supplied.
    Returns:
        ``(n_mics, T_features, n_mels)`` float32.
    """
    if stft is None:
        if audio is None:
            raise ValueError("must pass either audio or stft")
        stft = _stft_per_channel(audio, cfg)
    n_mics, n_freq, T = stft.shape
    power = (np.abs(stft) ** 2).astype(np.float32)
    mel_filterbank = librosa.filters.mel(
        sr=cfg.fs,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.effective_fmax(),
    ).astype(np.float32)  # (n_mels, n_freq)
    mel = np.einsum("mf,cft->cmt", mel_filterbank, power)  # (n_mics, n_mels, T)
    log_mel = np.log(mel + 1e-8).astype(np.float32)
    return np.transpose(log_mel, (0, 2, 1))  # (n_mics, T, n_mels)


# ---------------------------------------------------------------------------
# GCC-PHAT (FFT-based, per frame)
# ---------------------------------------------------------------------------


def _gcc_phat_pair_from_stft(
    spec_i: np.ndarray,
    spec_j: np.ndarray,
    n_fft: int,
    n_lags: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Centred GCC-PHAT for one mic pair, using existing STFT.

    Args:
        spec_i, spec_j: ``(n_freq, T)`` complex spectrograms.
        n_fft: FFT length used to build the spectrograms (assumed even).
        n_lags: desired number of centred lag samples (kept symmetric).
        eps: phase-transform denom floor.

    Returns:
        ``(T, n_lags)`` real-valued, with lag 0 at index ``n_lags // 2``.
    """
    cpsd = spec_i * np.conj(spec_j)
    cpsd_norm = cpsd / (np.abs(cpsd) + eps)
    # Reconstruct the full negative-frequency half via Hermitian symmetry,
    # then IRFFT with explicit n=n_fft so we get exactly n_fft real samples.
    corr = np.fft.irfft(cpsd_norm, n=n_fft, axis=0)  # (n_fft, T)
    # Re-centre lag 0 at index n_fft // 2 (currently at index 0).
    corr = np.fft.fftshift(corr, axes=0)  # lag 0 -> index n_fft//2
    half = n_lags // 2
    centre = n_fft // 2
    if n_fft < n_lags:
        raise ValueError(f"n_fft ({n_fft}) must be >= n_lags ({n_lags})")
    sliced = corr[centre - half : centre - half + n_lags, :]  # (n_lags, T)
    return sliced.T.astype(np.float32)


def extract_gcc_phat(
    audio: np.ndarray | None = None,
    *,
    cfg: SeldFeatureConfig = SeldFeatureConfig(),
    stft: np.ndarray | None = None,
) -> np.ndarray:
    """6-pair (for 4 mics) GCC-PHAT feature.

    Returns:
        ``(n_pairs, T_features, n_lags)`` float32.

    Pair ordering matches :func:`itertools.combinations(range(n_mics), 2)`,
    i.e. ``(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`` for ``n_mics = 4``.
    """
    if stft is None:
        if audio is None:
            raise ValueError("must pass either audio or stft")
        stft = _stft_per_channel(audio, cfg)
    n_mics = stft.shape[0]
    pairs = list(combinations(range(n_mics), 2))
    out = []
    for (i, j) in pairs:
        gcc = _gcc_phat_pair_from_stft(stft[i], stft[j], cfg.n_fft, cfg.n_gcc_lags)
        out.append(gcc)
    return np.stack(out, axis=0)  # (n_pairs, T, n_lags)


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def extract_intensity_vector(
    stft: np.ndarray,
    cfg: SeldFeatureConfig,
) -> np.ndarray:
    """Mel-pooled FOA intensity vector (Ix, Iy, Iz), energy-normalised.

    For first-order Ambisonics with channels ordered (W, X, Y, Z),

        I_d(t,f) = Re( STFT_W(t,f) * conj(STFT_d(t,f)) )      d in {X,Y,Z}

    Each component is divided by the total energy
    ``|W|**2 + |X|**2 + |Y|**2 + |Z|**2`` (with floor) to give a
    direction-of-arrival proxy that lives in [-0.5, 0.5] (roughly). The
    per-bin DOA cube is then projected to the mel scale so it can be
    stacked with the log-mel block on the same final-axis grid.

    Args:
        stft: complex ``(4, n_freq, T)`` from :func:`_stft_per_channel` for
            FOA-ordered audio.

    Returns:
        ``(3, T, n_mels)`` float32.
    """
    if stft.shape[0] != 4:
        raise ValueError(f"intensity_vector expects 4 FOA channels, got {stft.shape[0]}")
    w = stft[0]
    energy = (
        np.abs(w) ** 2
        + np.abs(stft[1]) ** 2
        + np.abs(stft[2]) ** 2
        + np.abs(stft[3]) ** 2
    )
    energy = np.maximum(energy, 1e-8).astype(np.float32)  # (n_freq, T)

    iv_components: list[np.ndarray] = []
    for d in range(1, 4):
        iv_d = np.real(w * np.conj(stft[d])).astype(np.float32)  # (n_freq, T)
        iv_d = iv_d / energy
        iv_components.append(iv_d)
    iv_full = np.stack(iv_components, axis=0)  # (3, n_freq, T)

    mel_filterbank = librosa.filters.mel(
        sr=cfg.fs,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.effective_fmax(),
    ).astype(np.float32)  # (n_mels, n_freq)
    iv_mel = np.einsum("mf,cft->cmt", mel_filterbank, iv_full)  # (3, n_mels, T)
    return np.transpose(iv_mel, (0, 2, 1))  # (3, T, n_mels)


def extract_seld_features(
    audio: np.ndarray,
    cfg: SeldFeatureConfig = SeldFeatureConfig(),
) -> np.ndarray:
    """Compute combined SELD features for one clip.

    Layout depends on ``cfg.array_type``:

        "mic": 4 log-mel + 6 mic-pair GCC-PHAT = 10 channels.
        "foa": 4 log-mel (W,X,Y,Z) + 3 intensity-vector dims = 7 channels.

    Args:
        audio: ``(n_mics, n_samples)`` float32 at ``cfg.fs``.

    Returns:
        Feature tensor of shape
        ``(cfg.n_feature_channels(), T_features, cfg.n_mels)``.

        Note: ``n_mels`` and ``n_gcc_lags`` must be equal so that the two
        feature blocks share the same last-axis size; this is checked.
        For FOA the intensity vector is mel-pooled to ``n_mels``.
    """
    if cfg.n_mels != cfg.n_gcc_lags:
        raise ValueError(
            f"n_mels ({cfg.n_mels}) must equal n_gcc_lags ({cfg.n_gcc_lags}) "
            f"so log-mel and GCC-PHAT can be stacked along the channel axis"
        )
    if audio.shape[0] != cfg.n_mics:
        raise ValueError(
            f"audio has {audio.shape[0]} channels, cfg.n_mics = {cfg.n_mics}"
        )
    stft = _stft_per_channel(audio, cfg)  # (n_mics, n_freq, T)
    logmel = extract_logmel(stft=stft, cfg=cfg)  # (n_mics, T, n_mels)
    if cfg.array_type == "mic":
        spatial = extract_gcc_phat(stft=stft, cfg=cfg)  # (n_pairs, T, n_lags=n_mels)
    elif cfg.array_type == "foa":
        spatial = extract_intensity_vector(stft=stft, cfg=cfg)  # (3, T, n_mels)
    else:
        raise ValueError(f"unknown array_type: {cfg.array_type!r}")
    return np.concatenate([logmel, spatial], axis=0).astype(np.float32)


def num_label_frames(n_audio_samples: int, cfg: SeldFeatureConfig) -> int:
    """Compute the *label*-resolution frame count for an audio clip."""
    return int(np.floor(n_audio_samples / cfg.label_hop_samples))


def crop_features_to_label_frames(
    features: np.ndarray,
    n_label_frames: int,
    cfg: SeldFeatureConfig,
) -> np.ndarray:
    """Trim feature tensor so it covers exactly ``n_label_frames`` labels.

    The model pools time by ``cfg.feature_per_label_ratio``; this helper
    discards the trailing partial-label frames so that everything aligns.
    """
    ratio = cfg.feature_per_label_ratio
    target_T = n_label_frames * ratio
    if features.shape[1] < target_T:
        # Pad with -log eps (log silence) for log-mel + zeros for GCC.
        n_mics = cfg.n_mics
        pad_T = target_T - features.shape[1]
        pad_logmel = np.full(
            (n_mics, pad_T, features.shape[2]), np.log(1e-8), dtype=features.dtype
        )
        pad_gcc = np.zeros(
            (features.shape[0] - n_mics, pad_T, features.shape[2]), dtype=features.dtype
        )
        pad = np.concatenate([pad_logmel, pad_gcc], axis=0)
        return np.concatenate([features, pad], axis=1)
    return features[:, :target_T, :]
