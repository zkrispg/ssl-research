"""CNN classifier for single-frame phase-map DOA estimation.

Architecture follows Chakrabarty & Habets (2019), simplified for CPU
training:

    Input: (B, 2, M, F)   (sin/cos of phase, M mics, F frequency bins)
    -> 3 conv2d layers with kernel (2, 1) that progressively reduce the M
       dimension from M to 1, fully fusing inter-mic phase relationships.
    -> 2 conv1d layers along F with kernel 5 (broadband modeling).
    -> global average pool over F.
    -> dropout + dense -> n_classes logits.

The total parameter count is ~40K, suitable for CPU training in <30 minutes.
"""
from __future__ import annotations

import torch
from torch import nn


class PhaseMapCNN(nn.Module):
    def __init__(
        self,
        n_mics: int = 4,
        n_freq: int = 257,
        n_classes: int = 72,
        n_filters_2d: int = 32,
        n_filters_1d: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if n_mics < 2:
            raise ValueError("Need at least 2 mics")

        self.n_mics = n_mics
        self.n_freq = n_freq
        self.n_classes = n_classes

        # M-1 spatial conv layers, each reducing the mic dim by 1.
        spatial_blocks = []
        in_ch = 2
        for _ in range(n_mics - 1):
            spatial_blocks += [
                nn.Conv2d(in_ch, n_filters_2d, kernel_size=(2, 1)),
                nn.BatchNorm2d(n_filters_2d),
                nn.ReLU(inplace=True),
            ]
            in_ch = n_filters_2d
        self.spatial = nn.Sequential(*spatial_blocks)

        self.freq_conv = nn.Sequential(
            nn.Conv1d(n_filters_2d, n_filters_1d, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_filters_1d),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_filters_1d, n_filters_1d, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_filters_1d),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters_1d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, M, F) -> (B, n_classes)."""
        h = self.spatial(x)  # (B, C, 1, F)
        h = h.squeeze(2)  # (B, C, F)
        h = self.freq_conv(h)  # (B, n_filters_1d, F)
        h = h.mean(dim=-1)  # (B, n_filters_1d)
        h = self.dropout(h)
        return self.fc(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
