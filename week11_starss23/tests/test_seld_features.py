"""Unit tests for week11_starss23.seld_features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from week11_starss23.seld_features import (
    DEFAULT_FS,
    SeldFeatureConfig,
    crop_features_to_label_frames,
    extract_gcc_phat,
    extract_intensity_vector,
    extract_logmel,
    extract_seld_features,
    load_multichannel_audio,
    num_label_frames,
    _stft_per_channel,
)


# ---------------------------------------------------------------------------
# SeldFeatureConfig
# ---------------------------------------------------------------------------


def test_config_defaults_match_dcase_seldnet():
    cfg = SeldFeatureConfig()
    assert cfg.fs == 24_000
    assert cfg.n_mels == 64
    assert cfg.n_gcc_lags == 64
    assert cfg.hop_samples == 480  # 20 ms @ 24 kHz
    assert cfg.label_hop_samples == 2400  # 100 ms @ 24 kHz
    assert cfg.feature_per_label_ratio == 5
    assert cfg.n_feature_channels() == 10  # 4 mics + C(4,2) = 6 pairs


def test_config_label_hop_must_be_multiple_of_hop():
    bad = SeldFeatureConfig(hop_s=0.03, label_hop_s=0.1)  # 0.1 / 0.03 not integer
    with pytest.raises(ValueError, match="integer multiple"):
        _ = bad.feature_per_label_ratio


def test_config_n_feature_channels_scales_with_mics():
    cfg = SeldFeatureConfig(n_mics=8)
    assert cfg.n_feature_channels() == 8 + 8 * 7 // 2  # 8 + 28 = 36


# ---------------------------------------------------------------------------
# load_multichannel_audio
# ---------------------------------------------------------------------------


def _write_synthetic_wav(path: Path, fs: int, duration_s: float, n_channels: int = 4):
    n = int(fs * duration_s)
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((n, n_channels)).astype(np.float32) * 0.1
    sf.write(str(path), audio, fs, subtype="FLOAT")


def test_load_multichannel_audio_native_sr(tmp_path: Path):
    p = tmp_path / "x.wav"
    _write_synthetic_wav(p, fs=24_000, duration_s=1.0, n_channels=4)
    a = load_multichannel_audio(p, target_fs=24_000, n_mics=4)
    assert a.shape == (4, 24_000)
    assert a.dtype == np.float32


def test_load_multichannel_audio_resamples(tmp_path: Path):
    p = tmp_path / "x_44k.wav"
    _write_synthetic_wav(p, fs=44_100, duration_s=1.0, n_channels=4)
    a = load_multichannel_audio(p, target_fs=24_000, n_mics=4)
    assert a.shape[0] == 4
    # Resampling produces ~target_fs * duration samples (allow small margin).
    assert abs(a.shape[1] - 24_000) <= 50


def test_load_multichannel_audio_drops_extra_channels(tmp_path: Path):
    p = tmp_path / "x_8ch.wav"
    _write_synthetic_wav(p, fs=24_000, duration_s=0.5, n_channels=8)
    a = load_multichannel_audio(p, target_fs=24_000, n_mics=4)
    assert a.shape == (4, 12_000)


def test_load_multichannel_audio_too_few_channels_raises(tmp_path: Path):
    p = tmp_path / "x_2ch.wav"
    _write_synthetic_wav(p, fs=24_000, duration_s=0.5, n_channels=2)
    with pytest.raises(ValueError, match="need 4"):
        load_multichannel_audio(p, target_fs=24_000, n_mics=4)


# ---------------------------------------------------------------------------
# extract_logmel
# ---------------------------------------------------------------------------


def test_extract_logmel_shape_and_dtype():
    cfg = SeldFeatureConfig()
    audio = np.random.RandomState(0).randn(4, cfg.fs * 1).astype(np.float32) * 0.1
    mel = extract_logmel(audio, cfg=cfg)
    expected_T = audio.shape[1] // cfg.hop_samples + 1  # librosa center=True
    assert mel.shape[0] == 4
    assert mel.shape[2] == cfg.n_mels
    assert abs(mel.shape[1] - expected_T) <= 2  # +/- 1 from edge effects
    assert mel.dtype == np.float32


def test_extract_logmel_finite_values():
    cfg = SeldFeatureConfig()
    audio = np.random.RandomState(0).randn(4, cfg.fs * 1).astype(np.float32) * 0.01
    mel = extract_logmel(audio, cfg=cfg)
    assert np.all(np.isfinite(mel))


def test_extract_logmel_silence_floor():
    cfg = SeldFeatureConfig()
    audio = np.zeros((4, cfg.fs), dtype=np.float32)
    mel = extract_logmel(audio, cfg=cfg)
    assert np.all(mel < 0)  # log of small number is negative
    # Silence should be about log(1e-8) ~ -18.4
    assert abs(mel.mean() - np.log(1e-8)) < 1.0


# ---------------------------------------------------------------------------
# extract_gcc_phat
# ---------------------------------------------------------------------------


def test_extract_gcc_phat_shape():
    cfg = SeldFeatureConfig()
    audio = np.random.RandomState(0).randn(4, cfg.fs * 1).astype(np.float32) * 0.1
    gcc = extract_gcc_phat(audio, cfg=cfg)
    assert gcc.shape[0] == 6  # C(4, 2)
    assert gcc.shape[2] == cfg.n_gcc_lags
    assert gcc.dtype == np.float32


def test_extract_gcc_phat_peak_at_known_delay():
    """Synthesize ch1 = ch0 delayed by D samples; GCC-PHAT must peak at lag D."""
    cfg = SeldFeatureConfig(n_mics=2)
    rng = np.random.default_rng(1)
    n = cfg.fs * 1
    src = rng.standard_normal(n).astype(np.float32) * 0.5
    delay = 10
    ch0 = src
    ch1 = np.concatenate([np.zeros(delay, dtype=np.float32), src[: n - delay]])
    audio = np.stack([ch0, ch1], axis=0)
    gcc = extract_gcc_phat(audio, cfg=cfg)  # (1, T, 64)
    centre = cfg.n_gcc_lags // 2
    # Average over time; GCC-PHAT(ch0, ch1) peaks at lag = -delay because we
    # use ch0 * conj(ch1) which corresponds to argmax over (ch0 lagging ch1).
    avg = gcc[0].mean(axis=0)
    peak_lag = int(np.argmax(avg)) - centre
    assert peak_lag == -delay, f"expected peak at -{delay}, got {peak_lag}"


def test_extract_gcc_phat_zero_delay_peaks_at_centre():
    cfg = SeldFeatureConfig(n_mics=2)
    rng = np.random.default_rng(2)
    n = cfg.fs * 1
    src = rng.standard_normal(n).astype(np.float32) * 0.5
    audio = np.stack([src, src], axis=0)
    gcc = extract_gcc_phat(audio, cfg=cfg)
    centre = cfg.n_gcc_lags // 2
    avg = gcc[0].mean(axis=0)
    assert int(np.argmax(avg)) == centre


def test_extract_gcc_phat_pair_ordering_matches_combinations():
    """For 4 mics, pair 0 should be (0,1), pair 5 should be (2,3)."""
    cfg = SeldFeatureConfig()
    rng = np.random.default_rng(3)
    n = cfg.fs
    src = rng.standard_normal(n).astype(np.float32) * 0.5
    # Construct: ch1, ch2, ch3 are delayed copies with distinct delays.
    audio = np.zeros((4, n), dtype=np.float32)
    audio[0] = src
    audio[1] = np.concatenate([np.zeros(5, dtype=np.float32), src[:-5]])  # 5 sample delay
    audio[2] = np.concatenate([np.zeros(15, dtype=np.float32), src[:-15]])  # 15 sample delay
    audio[3] = np.concatenate([np.zeros(2, dtype=np.float32), src[:-2]])  # 2 sample delay
    gcc = extract_gcc_phat(audio, cfg=cfg)  # (6, T, 64)
    centre = cfg.n_gcc_lags // 2
    expected_pair_delays = {
        0: -5,  # (0,1) -> ch0 leads by 5
        1: -15,  # (0,2)
        2: -2,  # (0,3)
        3: -10,  # (1,2): ch1 = +5, ch2 = +15 -> (1,2) lag = 5 - 15 = -10
        4: 3,  # (1,3): ch1 = +5, ch3 = +2 -> 5 - 2 = +3
        5: 13,  # (2,3): ch2 = +15, ch3 = +2 -> 15 - 2 = +13
    }
    for p, expected_delay in expected_pair_delays.items():
        avg = gcc[p].mean(axis=0)
        peak = int(np.argmax(avg)) - centre
        assert peak == expected_delay, (
            f"pair {p}: expected peak at {expected_delay}, got {peak}"
        )


# ---------------------------------------------------------------------------
# extract_seld_features (combined)
# ---------------------------------------------------------------------------


def test_extract_seld_features_shape():
    cfg = SeldFeatureConfig()
    audio = np.random.RandomState(0).randn(4, cfg.fs * 2).astype(np.float32) * 0.1
    feat = extract_seld_features(audio, cfg=cfg)
    assert feat.shape[0] == 10  # 4 logmel + 6 gcc
    assert feat.shape[2] == 64
    assert feat.dtype == np.float32


def test_extract_seld_features_first_4_channels_are_logmel():
    cfg = SeldFeatureConfig()
    audio = np.random.RandomState(0).randn(4, cfg.fs).astype(np.float32) * 0.1
    feat = extract_seld_features(audio, cfg=cfg)
    logmel_only = extract_logmel(audio, cfg=cfg)
    np.testing.assert_allclose(feat[:4], logmel_only, atol=1e-5)


def test_extract_seld_features_last_6_channels_are_gcc():
    cfg = SeldFeatureConfig()
    audio = np.random.RandomState(0).randn(4, cfg.fs).astype(np.float32) * 0.1
    feat = extract_seld_features(audio, cfg=cfg)
    gcc_only = extract_gcc_phat(audio, cfg=cfg)
    np.testing.assert_allclose(feat[4:], gcc_only, atol=1e-5)


def test_extract_seld_features_rejects_mismatched_n_mels_n_gcc_lags():
    cfg = SeldFeatureConfig(n_mels=64, n_gcc_lags=32)
    audio = np.zeros((4, 24_000), dtype=np.float32)
    with pytest.raises(ValueError, match="must equal n_gcc_lags"):
        extract_seld_features(audio, cfg=cfg)


def test_extract_seld_features_rejects_wrong_n_mics():
    cfg = SeldFeatureConfig(n_mics=4)
    audio = np.zeros((2, 24_000), dtype=np.float32)
    with pytest.raises(ValueError, match="cfg.n_mics"):
        extract_seld_features(audio, cfg=cfg)


# ---------------------------------------------------------------------------
# Frame-count helpers / cropping
# ---------------------------------------------------------------------------


def test_num_label_frames():
    cfg = SeldFeatureConfig()
    # 2.0 s -> 20 label frames at 100 ms hop
    assert num_label_frames(2 * cfg.fs, cfg) == 20
    # 1.05 s -> floor(1.05/0.1) = 10
    assert num_label_frames(int(1.05 * cfg.fs), cfg) == 10


def test_crop_features_to_label_frames_truncates():
    cfg = SeldFeatureConfig()
    n_ch = cfg.n_feature_channels()
    feat = np.random.RandomState(0).randn(n_ch, 50, cfg.n_mels).astype(np.float32)
    cropped = crop_features_to_label_frames(feat, n_label_frames=8, cfg=cfg)
    assert cropped.shape == (n_ch, 8 * 5, cfg.n_mels)


def test_crop_features_to_label_frames_pads_short_clips():
    cfg = SeldFeatureConfig()
    n_ch = cfg.n_feature_channels()
    # Too-short feature sequence: 12 frames, want 4 labels = 20 frames -> pad 8.
    feat = np.random.RandomState(0).randn(n_ch, 12, cfg.n_mels).astype(np.float32)
    cropped = crop_features_to_label_frames(feat, n_label_frames=4, cfg=cfg)
    assert cropped.shape == (n_ch, 20, cfg.n_mels)
    # Padding region of log-mel channels should be log(1e-8); GCC channels 0.
    pad_logmel_mean = cropped[:4, 12:].mean()
    pad_gcc_mean = cropped[4:, 12:].mean()
    assert abs(pad_logmel_mean - np.log(1e-8)) < 0.01
    assert abs(pad_gcc_mean) < 0.01


# ---------------------------------------------------------------------------
# Feature/label alignment guarantee
# ---------------------------------------------------------------------------


def test_features_and_labels_have_compatible_temporal_alignment():
    """A 2-second clip should yield 20 label frames and (after crop) 100 feature frames."""
    cfg = SeldFeatureConfig()
    audio = np.random.RandomState(0).randn(4, 2 * cfg.fs).astype(np.float32) * 0.1
    feat = extract_seld_features(audio, cfg=cfg)
    n_labels = num_label_frames(audio.shape[1], cfg)
    cropped = crop_features_to_label_frames(feat, n_labels, cfg)
    assert n_labels == 20
    assert cropped.shape[1] == 100  # 20 * 5


# ---------------------------------------------------------------------------
# FOA: intensity vector + array_type=foa dispatch
# ---------------------------------------------------------------------------


def test_foa_config_has_seven_channels():
    cfg = SeldFeatureConfig(array_type="foa")
    # 4 log-mel (W,X,Y,Z) + 3 IV (Ix,Iy,Iz) = 7
    assert cfg.n_feature_channels() == 7


def test_foa_config_rejects_non_4_mics():
    with pytest.raises(ValueError, match="array_type=foa requires n_mics=4"):
        SeldFeatureConfig(array_type="foa", n_mics=8).n_feature_channels()


def test_unknown_array_type_raises():
    with pytest.raises(ValueError, match="unknown array_type"):
        SeldFeatureConfig(array_type="binaural").n_feature_channels()


def test_intensity_vector_shape_and_dtype():
    cfg = SeldFeatureConfig(array_type="foa")
    audio = np.random.RandomState(0).randn(4, cfg.fs).astype(np.float32) * 0.1
    stft = _stft_per_channel(audio, cfg)
    iv = extract_intensity_vector(stft=stft, cfg=cfg)
    assert iv.shape[0] == 3  # Ix, Iy, Iz
    assert iv.shape[2] == cfg.n_mels
    assert iv.dtype == np.float32
    assert np.all(np.isfinite(iv))


def test_intensity_vector_silence_is_zero():
    """All-zero FOA -> energy floor kicks in -> IV ~ 0."""
    cfg = SeldFeatureConfig(array_type="foa")
    audio = np.zeros((4, cfg.fs), dtype=np.float32)
    stft = _stft_per_channel(audio, cfg)
    iv = extract_intensity_vector(stft=stft, cfg=cfg)
    # IV components are in [-0.5, 0.5] range typically; for silence they
    # collapse to numerator=0 / energy_floor -> exactly 0.
    assert np.allclose(iv, 0.0)


def test_intensity_vector_directional_signal_axis_alignment():
    """For a pure plane wave from +X (front), I_x should dominate I_y/I_z.

    With FOA convention W=omni, X=front, Y=left, Z=up, a source at +X
    yields W and X both ~equal cos(theta=0)=1 in amplitude (with W
    normalisation factor), while Y and Z = 0. Hence Re(W*conj(X)) > 0
    while Re(W*conj(Y)) ~ Re(W*conj(Z)) ~ 0.
    """
    cfg = SeldFeatureConfig(array_type="foa")
    rng = np.random.default_rng(7)
    n = cfg.fs
    src = rng.standard_normal(n).astype(np.float32) * 0.5
    audio = np.zeros((4, n), dtype=np.float32)
    audio[0] = src       # W
    audio[1] = src       # X (full energy in front direction)
    audio[2] = src * 0.0 # Y silent
    audio[3] = src * 0.0 # Z silent
    stft = _stft_per_channel(audio, cfg)
    iv = extract_intensity_vector(stft=stft, cfg=cfg)
    ix, iy, iz = iv[0].mean(), iv[1].mean(), iv[2].mean()
    # Relative dominance of Ix is the substantive claim; mel-filter Slaney
    # normalisation makes the absolute scale ~0.02 even for a unit-strength
    # axis-aligned source. We assert the sign and the relative gap.
    assert ix > 0.0
    assert ix > abs(iy) + 0.001
    assert ix > abs(iz) + 0.001


def test_extract_seld_features_foa_layout():
    """First 4 channels = log-mel; last 3 = mel-pooled intensity vector."""
    cfg = SeldFeatureConfig(array_type="foa")
    audio = np.random.RandomState(11).randn(4, cfg.fs).astype(np.float32) * 0.1
    feat = extract_seld_features(audio, cfg=cfg)
    assert feat.shape[0] == 7
    assert feat.shape[2] == cfg.n_mels
    assert feat.dtype == np.float32

    logmel_only = extract_logmel(audio, cfg=cfg)
    np.testing.assert_allclose(feat[:4], logmel_only, atol=1e-5)
    stft = _stft_per_channel(audio, cfg)
    iv_only = extract_intensity_vector(stft=stft, cfg=cfg)
    np.testing.assert_allclose(feat[4:], iv_only, atol=1e-5)


def test_extract_seld_features_foa_and_mic_have_different_channel_counts():
    audio = np.random.RandomState(0).randn(4, DEFAULT_FS).astype(np.float32) * 0.1
    mic = extract_seld_features(audio, cfg=SeldFeatureConfig(array_type="mic"))
    foa = extract_seld_features(audio, cfg=SeldFeatureConfig(array_type="foa"))
    assert mic.shape[0] == 10
    assert foa.shape[0] == 7
