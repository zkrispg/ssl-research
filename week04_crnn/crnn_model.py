"""Multi-frame CRNN with ACCDOA-style output for azimuth regression.

Architecture:

    Input: (B, 2, M, F, T)   (sin/cos phase, M mics, F freq bins, T frames)

    1. Per-frame spatial conv: (M-1) Conv2d layers with kernel (2, 1)
       collapse the M dimension to 1 while learning inter-mic phase
       relationships. Operates on each frame independently by reshaping
       the time axis into the batch dimension.

    2. Per-frame frequency conv: a Conv1d with kernel 5 along F, then a
       global average pool over F gives a fixed-size embedding per frame.

    3. Temporal recurrence: a single bidirectional GRU consumes the
       sequence of per-frame embeddings.

    4. ACCDOA head: a linear layer outputs a 2-D vector (cos theta, sin theta)
       for each frame. Magnitude of the vector implicitly encodes
       confidence; direction encodes the predicted azimuth.

Inference: average the per-frame ACCDOA vectors of an utterance and take
``atan2(y, x)``. This continuous output bypasses the 5-degree class grid
floor of the W3 CNN.
"""
from __future__ import annotations

import torch
from torch import nn


class CRNNDoa(nn.Module):
    def __init__(
        self,
        n_mics: int = 4,
        n_freq: int = 257,
        spatial_filters: int = 32,
        freq_filters: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_mics = n_mics
        self.n_freq = n_freq

        spatial_layers = []
        in_ch = 2
        for _ in range(n_mics - 1):
            spatial_layers += [
                nn.Conv2d(in_ch, spatial_filters, kernel_size=(2, 1)),
                nn.BatchNorm2d(spatial_filters),
                nn.ReLU(inplace=True),
            ]
            in_ch = spatial_filters
        self.spatial = nn.Sequential(*spatial_layers)

        self.freq = nn.Sequential(
            nn.Conv1d(spatial_filters, freq_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(freq_filters),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        self.gru = nn.GRU(
            input_size=freq_filters,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, M, F, T) -> (B, T, 2) per-frame ACCDOA vectors."""
        B, C, M, F, T = x.shape
        # Move T into batch for per-frame conv2d
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, T, 2, M, F)
        x = x.view(B * T, C, M, F)
        x = self.spatial(x)  # (B*T, sp, 1, F)
        x = x.squeeze(2)  # (B*T, sp, F)
        x = self.freq(x).squeeze(-1)  # (B*T, freq_filters)
        x = x.view(B, T, -1)
        x, _ = self.gru(x)  # (B, T, 2*gru_hidden)
        return self.head(x)  # (B, T, 2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
