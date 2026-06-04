"""Strict reproduction of the DCASE 2023 SELDnet baseline.

This module exists so that the ICASSP paper can compare our pipeline to
*the published baseline*, not just to our own ablations. The architecture
mirrors the official baseline released alongside DCASE 2023 Task 3
(Adavanne et al. 2018; Politis et al. 2022 STARSS22; baseline updates in
the seld-dcase2023 repository).

Architecture (same defaults as the official repo):

    Input:    (B, in_channels, T_feat, F=64)
              For our MIC-array pipeline we feed the *same* 10-channel
              tensor used everywhere else (4 log-mel + 6 GCC-PHAT) so the
              comparison is matched on inputs.

    Conv block 1: Conv2d(64, 3x3, pad=1) + BN + ReLU + MaxPool(5x4) + Dropout
    Conv block 2: Conv2d(64, 3x3, pad=1) + BN + ReLU + MaxPool(1x4) + Dropout
    Conv block 3: Conv2d(64, 3x3, pad=1) + BN + ReLU + MaxPool(1x2) + Dropout
    -> tensor (B, 64, T_label, 2), flatten freq -> (B, T_label, 128)

    BiGRU x 2:  hidden=128 per direction (output dim = 256)
                inter-layer dropout matches conv dropout

    FC 1:       Linear(256 -> 128) + ReLU + Dropout
    FC 2:       Linear(128 -> N_tracks * 3 * N_classes) + tanh

Differences from our :class:`SeldCRNN` ``baseline`` variant:

    * Two FC layers with ReLU bottleneck instead of the SELDnet
      multiplicative gate ``h[..., rh:] * h[..., :rh]``.
    * No GCA prefix at all (``baseline`` already disables GCA, but this
      file removes the option entirely so reviewers can verify the model
      definition is byte-for-byte the published one).

The output dictionary uses key ``"accdoa"`` of shape
``(B, T_label, n_tracks * 3 * n_classes)`` so the existing ADPIT loss,
decoder, and metric pipeline can be reused without modification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SeldNetOfficialConfig:
    """Hyper-parameters matching the DCASE 2023 Task 3 SELDnet baseline."""

    in_channels: int = 10
    n_classes: int = 13
    n_tracks: int = 3
    n_freq_bins: int = 64
    feature_per_label_ratio: int = 5

    # Frozen at the published values; we expose them only for unit tests.
    cnn_filters: int = 64
    f_pool_size: tuple[int, int, int] = (4, 4, 2)
    rnn_hidden: int = 128
    rnn_layers: int = 2
    fc_hidden: int = 128
    dropout: float = 0.05


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        pool: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pool(torch.relu_(self.bn(self.conv(x)))))


class SeldNetOfficial(nn.Module):
    """DCASE 2023 SELDnet baseline -- strict reproduction.

    Output:
        ``forward(x)`` returns ``{"accdoa": (B, T_label, N*3*C)}`` ready
        for class-coupled Multi-ACCDOA decoding and ADPIT loss.
    """

    def __init__(self, cfg: SeldNetOfficialConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or SeldNetOfficialConfig()
        self.cfg = cfg

        prod_f_pool = int(np.prod(cfg.f_pool_size))
        if cfg.n_freq_bins % prod_f_pool != 0:
            raise ValueError(
                f"n_freq_bins ({cfg.n_freq_bins}) must be divisible by "
                f"prod(f_pool_size)={prod_f_pool}"
            )
        post_freq = cfg.n_freq_bins // prod_f_pool

        # Time pool: collapse to label resolution in the first conv block.
        t_pool = (cfg.feature_per_label_ratio, 1, 1)

        self.conv_blocks = nn.Sequential(
            _ConvBlock(
                cfg.in_channels, cfg.cnn_filters,
                (t_pool[0], cfg.f_pool_size[0]), cfg.dropout,
            ),
            _ConvBlock(
                cfg.cnn_filters, cfg.cnn_filters,
                (t_pool[1], cfg.f_pool_size[1]), cfg.dropout,
            ),
            _ConvBlock(
                cfg.cnn_filters, cfg.cnn_filters,
                (t_pool[2], cfg.f_pool_size[2]), cfg.dropout,
            ),
        )

        gru_in = cfg.cnn_filters * post_freq
        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
        )

        # FC 1: 2*rnn_hidden -> fc_hidden (the canonical 256 -> 128 step)
        # FC 2: fc_hidden -> N*3*C
        self.fc1 = nn.Linear(2 * cfg.rnn_hidden, cfg.fc_hidden)
        self.fc_dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(cfg.fc_hidden, cfg.n_tracks * 3 * cfg.n_classes)

    @property
    def n_accdoa_outputs(self) -> int:
        return self.cfg.n_tracks * 3 * self.cfg.n_classes

    def reshape_accdoa(self, accdoa_flat: torch.Tensor) -> torch.Tensor:
        B, T, _ = accdoa_flat.shape
        return accdoa_flat.view(B, T, self.cfg.n_tracks, 3, self.cfg.n_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"expected 4-D input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"expected {self.cfg.in_channels} input channels, got {x.shape[1]}"
            )

        x = self.conv_blocks(x)  # (B, C, T_label, post_freq)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T_label, C, post_freq)
        x = x.flatten(2)  # (B, T_label, C*post_freq)

        h, _ = self.gru(x)  # (B, T_label, 2*rnn_hidden)
        h = self.fc_dropout(torch.relu_(self.fc1(h)))  # (B, T_label, fc_hidden)
        accdoa = torch.tanh(self.fc2(h))  # (B, T_label, N*3*C)
        return {"accdoa": accdoa}


def make_default_seldnet_official(n_classes: int = 13) -> SeldNetOfficial:
    """Construct the DCASE 2023 baseline with default hyper-parameters."""
    return SeldNetOfficial(SeldNetOfficialConfig(n_classes=n_classes))
