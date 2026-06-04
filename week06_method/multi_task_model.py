"""W6 multi-task CRNN: spatial pseudo-spectrum + source count.

The body is identical to the W5 model. We add a second head that takes
the pooled per-utterance representation (mean over GRU outputs across
time) and predicts the source count via softmax over ``max_k`` classes.
"""
from __future__ import annotations

import torch
from torch import nn


class MultiTaskCRNN(nn.Module):
    def __init__(
        self,
        n_mics: int = 4,
        n_freq: int = 257,
        n_classes: int = 72,
        max_k: int = 3,
        spatial_filters: int = 32,
        freq_filters: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_mics = n_mics
        self.n_freq = n_freq
        self.n_classes = n_classes
        self.max_k = max_k

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

        self.spectrum_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, n_classes),
        )

        # The count head pools over time before classifying.
        self.count_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, max_k),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, 2, M, F, T) -> dict with 'spectrum' (B,T,C) and 'count' (B,K)."""
        B, C, M, F, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, M, F)
        x = self.spatial(x).squeeze(2)
        x = self.freq(x).squeeze(-1)
        x = x.view(B, T, -1)
        h, _ = self.gru(x)  # (B, T, 2*gru_hidden)
        spectrum_logits = self.spectrum_head(h)  # (B, T, n_classes)
        pooled = h.mean(dim=1)  # (B, 2*gru_hidden)
        count_logits = self.count_head(pooled)  # (B, max_k)
        return {"spectrum": spectrum_logits, "count": count_logits}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
