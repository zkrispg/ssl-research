"""SELD model for STARSS23: CRNN backbone + Multi-ACCDOA head.

The architecture follows the SELDnet 2022 reference (Adavanne et al.) but
is adapted to the user's pipeline:

    Input:    (B, in_channels=10, T_feat, n_freq=64)
                  4 channels of per-mic log-mel
                + 6 channels of GCC-PHAT (one per mic pair)

    Optional: Geometry-aware Channel Attention (W9 GCA) on the 4 log-mel
              channels only -- the GCC channels already encode geometry.

    CNN:      3 x [Conv2d(3x3) + BN + ReLU + MaxPool + Dropout]
              First MaxPool collapses time by ``feature_per_label_ratio``
              so the post-CNN time axis is at *label resolution*.

    GRU:      2-layer bidirectional GRU + multiplicative gate.

    Heads:    Multi-ACCDOA  -> (B, T_label, n_tracks * 3 * n_classes)
              (Optional)    distance head -> (B, T_label, n_classes)

The Multi-ACCDOA output is a flat vector that should be reshaped to
``(B, T_label, n_tracks, 3, n_classes)`` for decoding and ADPIT loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from week09_geometry_attn.geometry_attn import (
    GeometryAwareChannelAttention,
    count_parameters,
)


@dataclass(frozen=True)
class SeldModelConfig:
    """Hyper-parameters for :class:`SeldCRNN`."""

    in_channels: int = 10
    n_mics: int = 4
    n_classes: int = 13
    n_tracks: int = 3  # emitted ACCDOA tracks (DCASE convention: 3)
    n_freq_bins: int = 64
    feature_per_label_ratio: int = 5
    cnn_filters: int = 64
    f_pool_size: tuple[int, ...] = (4, 4, 2)  # cumulative product must divide n_freq_bins
    rnn_hidden: int = 128
    rnn_layers: int = 2
    dropout: float = 0.05
    use_distance_head: bool = False  # DCASE 2024 distance task

    # GCA options
    use_gca: bool = False
    gca_geometry_bias: bool = True
    gca_embed_dim: int = 16


def default_uca4_positions(radius_m: float = 0.04) -> np.ndarray:
    """4-mic UCA in the xy plane, mics at 0, 90, 180, 270 deg."""
    angles = np.array([0.0, 90.0, 180.0, 270.0])
    rad = np.radians(angles)
    return np.stack([radius_m * np.cos(rad), radius_m * np.sin(rad)], axis=1).astype(
        np.float32
    )


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        pool: tuple[int, int],
        dropout: float,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pool(torch.relu_(self.bn(self.conv(x)))))


class SeldCRNN(nn.Module):
    """SELDnet-style CRNN with optional GCA prefix.

    Output convention:
        ``forward(x)`` returns a dict with key ``"accdoa"`` of shape
        ``(B, T_label, n_tracks * 3 * n_classes)`` ready for class-coupled
        ADPIT loss (Shimada 2022). When ``use_distance_head`` is True the
        dict also contains ``"distance": (B, T_label, n_classes)``.
    """

    def __init__(
        self,
        cfg: SeldModelConfig | None = None,
        mic_positions: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        cfg = cfg or SeldModelConfig()
        self.cfg = cfg

        # Optional GCA on the per-mic log-mel block.
        if cfg.use_gca:
            if mic_positions is None:
                mic_positions = default_uca4_positions()
            self.gca: GeometryAwareChannelAttention | None = GeometryAwareChannelAttention(
                mic_positions=mic_positions,
                in_channels=1,
                embed_dim=cfg.gca_embed_dim,
                geometry_bias=cfg.gca_geometry_bias,
            )
        else:
            self.gca = None

        # ---- 3 conv blocks --------------------------------------------------
        # First block pools time by feature_per_label_ratio (default 5x) so that
        # the GRU sees label-resolution frames; subsequent blocks pool only freq.
        t_pool = (cfg.feature_per_label_ratio, 1, 1)
        f_pool = cfg.f_pool_size
        if len(f_pool) != 3:
            raise ValueError(f"f_pool_size must have length 3, got {f_pool}")
        prod_f_pool = int(np.prod(f_pool))
        if cfg.n_freq_bins % prod_f_pool != 0:
            raise ValueError(
                f"n_freq_bins ({cfg.n_freq_bins}) must be divisible by "
                f"prod(f_pool_size)={prod_f_pool}"
            )
        post_freq = cfg.n_freq_bins // prod_f_pool
        if post_freq < 1:
            raise ValueError(
                f"f_pool_size {f_pool} reduces n_freq_bins {cfg.n_freq_bins} below 1"
            )

        self.conv_blocks = nn.Sequential(
            _ConvBlock(cfg.in_channels, cfg.cnn_filters, (t_pool[0], f_pool[0]), cfg.dropout),
            _ConvBlock(cfg.cnn_filters, cfg.cnn_filters, (t_pool[1], f_pool[1]), cfg.dropout),
            _ConvBlock(cfg.cnn_filters, cfg.cnn_filters, (t_pool[2], f_pool[2]), cfg.dropout),
        )

        # ---- GRU ------------------------------------------------------------
        gru_in = cfg.cnn_filters * post_freq
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
        )

        # ---- Heads ----------------------------------------------------------
        self.accdoa_head = nn.Linear(
            cfg.rnn_hidden, cfg.n_tracks * 3 * cfg.n_classes
        )
        self.distance_head = (
            nn.Linear(cfg.rnn_hidden, cfg.n_classes) if cfg.use_distance_head else None
        )

    # ------------------------------------------------------------------ helpers

    @property
    def n_accdoa_outputs(self) -> int:
        return self.cfg.n_tracks * 3 * self.cfg.n_classes

    def reshape_accdoa(self, accdoa_flat: torch.Tensor) -> torch.Tensor:
        """``(B, T, N*3*C)`` -> ``(B, T, N, 3, C)`` for decoding."""
        B, T, _ = accdoa_flat.shape
        return accdoa_flat.view(B, T, self.cfg.n_tracks, 3, self.cfg.n_classes)

    # ------------------------------------------------------------------ forward

    def _maybe_apply_gca(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GCA on the per-mic log-mel block; leave GCC pairs untouched.

        Args:
            x: ``(B, in_channels, T, F)``.
        """
        if self.gca is None:
            return x
        n_mics = self.cfg.n_mics
        mic_block = x[:, :n_mics]  # (B, n_mics, T, F)
        rest = x[:, n_mics:]
        # GCA expects (B, C, M, F, T) with C the per-mic feature-channel count
        # (here 1: a single log-mel value per (mic, freq, time)).
        B, M, T, F = mic_block.shape
        mic_for_gca = mic_block.permute(0, 2, 3, 1).contiguous()  # (B, T, F, M) -- not yet right
        # Need (B, C=1, M, F, T):
        mic_for_gca = mic_block.unsqueeze(1).permute(0, 1, 2, 4, 3).contiguous()
        # mic_block is (B, M, T, F); unsqueeze(1) -> (B, 1, M, T, F); permute to put F before T:
        # (0,1,2,4,3) -> (B, 1, M, F, T) -- correct.
        gated = self.gca(mic_for_gca)  # (B, 1, M, F, T)
        # back to (B, M, T, F):
        mic_block_gated = gated.squeeze(1).permute(0, 1, 3, 2).contiguous()
        return torch.cat([mic_block_gated, rest], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the SELD model.

        Args:
            x: ``(B, in_channels, T_feat, n_freq)`` feature tensor.

        Returns:
            Dict with key ``"accdoa"`` of shape
            ``(B, T_label, n_tracks * 3 * n_classes)``. If
            ``use_distance_head`` is enabled, also has key ``"distance"``
            of shape ``(B, T_label, n_classes)``.
        """
        if x.dim() != 4:
            raise ValueError(f"expected 4-D input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"expected {self.cfg.in_channels} input channels, got {x.shape[1]}"
            )

        x = self._maybe_apply_gca(x)
        x = self.conv_blocks(x)  # (B, C, T_label, post_freq)
        # Reshape to (B, T_label, C*post_freq)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T_label, C, post_freq)
        x = x.flatten(2)  # (B, T_label, C*post_freq)

        h, _ = self.gru(x)  # (B, T_label, 2*rnn_hidden)
        # SELDnet-style multiplicative gate: tanh + split-and-multiply.
        h = torch.tanh(h)
        rh = self.cfg.rnn_hidden
        h = h[..., rh:] * h[..., :rh]  # (B, T_label, rnn_hidden)

        accdoa = torch.tanh(self.accdoa_head(h))  # (B, T_label, n_tracks*3*n_classes)
        out: dict[str, torch.Tensor] = {"accdoa": accdoa}
        if self.distance_head is not None:
            # Distance is positive; softplus keeps it >= 0.
            out["distance"] = torch.nn.functional.softplus(self.distance_head(h))
        return out


def make_default_seld_model(
    use_gca: bool = False,
    gca_geometry_bias: bool = True,
    n_classes: int = 13,
) -> SeldCRNN:
    """Construct a SeldCRNN with the standard ICASSP-experiment defaults."""
    cfg = SeldModelConfig(
        use_gca=use_gca,
        gca_geometry_bias=gca_geometry_bias,
        n_classes=n_classes,
    )
    return SeldCRNN(cfg)
