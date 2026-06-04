"""CRNN that outputs a spatial pseudo-spectrum for multi-source DOA.

Architecture is the W4 CRNN with the head replaced:

    Input  : (B, 2, M, F, T)          sin/cos phase
    Body   : 3 spatial conv2d + freq conv1d + global F pool + bidirectional GRU
    Head   : Linear -> (B, T, n_classes) of sigmoid logits

Output is a per-frame, per-class logit. Inference averages sigmoid
probabilities across frames and uses peak picking to recover the
estimated source azimuths.
"""
from __future__ import annotations

import torch
from torch import nn


class MultiSourceCRNN(nn.Module):
    def __init__(
        self,
        n_mics: int = 4,
        n_freq: int = 257,
        n_classes: int = 72,
        spatial_filters: int = 32,
        freq_filters: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_mics = n_mics
        self.n_freq = n_freq
        self.n_classes = n_classes

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
            nn.Linear(gru_hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, M, F, T) -> (B, T, n_classes) logits."""
        B, C, M, F, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, M, F)
        x = self.spatial(x).squeeze(2)  # (B*T, sp, F)
        x = self.freq(x).squeeze(-1)  # (B*T, freq_filters)
        x = x.view(B, T, -1)
        x, _ = self.gru(x)  # (B, T, 2*gru_hidden)
        return self.head(x)  # (B, T, n_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
