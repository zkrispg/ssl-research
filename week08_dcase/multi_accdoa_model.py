"""W8 model: W6 backbone with Multi-ACCDOA head + auxiliary count head.

Architectural change relative to W6: the 72-bin sigmoid spatial spectrum
head is replaced by a Multi-ACCDOA regression head that outputs ``N=3``
``(x, y)`` ACCDOA vectors per frame, trained with ADPIT loss. The
auxiliary source-count head is kept (still helpful as a regularizer
even though the count is implicitly recoverable from track activities).
"""
from __future__ import annotations

import torch
from torch import nn


class MultiAccdoaCRNN(nn.Module):
    """Multi-track ACCDOA CRNN for multi-source DOA estimation.

    Body identical to W6's :class:`MultiTaskCRNN`; differs in the head:

    * ``accdoa_head`` outputs a ``(B, T, N, 2)`` regression target.
    * ``count_head`` outputs a ``(B, max_K)`` softmax (auxiliary).
    """

    def __init__(
        self,
        n_mics: int = 4,
        n_freq: int = 257,
        n_tracks: int = 3,
        max_k: int = 3,
        spatial_filters: int = 32,
        freq_filters: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_mics = n_mics
        self.n_freq = n_freq
        self.n_tracks = n_tracks
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

        self.accdoa_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, n_tracks * 2),
        )

        self.count_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, max_k),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, 2, M, F, T) -> {'accdoa': (B, T, N, 2), 'count': (B, max_K)}."""
        B, C, M, F, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(B * T, C, M, F)
        x = self.spatial(x).squeeze(2)
        x = self.freq(x).squeeze(-1)
        x = x.view(B, T, -1)
        h, _ = self.gru(x)  # (B, T, 2*gru_hidden)
        accdoa = self.accdoa_head(h).view(B, T, self.n_tracks, 2)
        pooled = h.mean(dim=1)
        count = self.count_head(pooled)
        return {"accdoa": accdoa, "count": count}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
