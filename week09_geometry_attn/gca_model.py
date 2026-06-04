"""W9 model: W6 multi-task CRNN with Geometry-aware Channel Attention.

Architecture
------------
The W9 model is the W6 :class:`MultiTaskCRNN` (sigmoid spatial-spectrum
head + auxiliary source-count head) preceded by a
:class:`GeometryAwareChannelAttention` (GCA) preprocessor. GCA operates
directly on the ``(B, C=2, M, F, T)`` phase tensor, computes a per-mic
sigmoid gate from a single-head self-attention over mics whose keys are
biased by the inter-mic ``(dx, dy, distance, bearing)`` geometry, and
returns the input multiplied by that gate.

Because GCA is a pure preprocessor that does not modify the W6 backbone
itself, we expose ``self.backbone`` for clean ablation:

* W6 baseline weights can be loaded into ``model.backbone`` directly.
* Setting ``geometry_bias=False`` collapses GCA into a plain
  channel-attention (same parameters and structure, no geometry).

The added overhead is on the order of 1.5 K parameters (~2 % of W6),
keeping the controlled comparison fair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "week06_method"))

from geometry_attn import GeometryAwareChannelAttention  # noqa: E402
from multi_task_model import MultiTaskCRNN  # noqa: E402


class GCAMultiTaskCRNN(nn.Module):
    """Geometry-aware multi-task CRNN.

    Args:
        mic_positions: ``(M, 2)`` or ``(M, 3)`` array of mic coordinates
            in metres.
        n_freq: number of STFT frequency bins (typically 257 for
            ``n_fft=512``).
        n_classes: number of discrete azimuth bins (e.g. 72 for 5-deg
            grid over 360 degrees).
        max_k: maximum number of simultaneous sources (count head
            outputs ``max_k`` logits).
        gca_embed_dim: hidden dimension for GCA attention.
        geometry_bias: when ``False`` GCA reduces to plain
            channel-attention (used for ablation).
        spatial_filters / freq_filters / gru_hidden / dropout:
            hyper-parameters forwarded to :class:`MultiTaskCRNN`.
    """

    def __init__(
        self,
        mic_positions: np.ndarray,
        n_freq: int = 257,
        n_classes: int = 72,
        max_k: int = 3,
        gca_embed_dim: int = 16,
        geometry_bias: bool = True,
        spatial_filters: int = 32,
        freq_filters: int = 64,
        gru_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mic_positions = np.asarray(mic_positions, dtype=np.float32)
        self.n_mics = self.mic_positions.shape[0]
        self.n_freq = n_freq
        self.n_classes = n_classes
        self.max_k = max_k
        self.geometry_bias = geometry_bias

        self.gca = GeometryAwareChannelAttention(
            mic_positions=self.mic_positions,
            in_channels=2,
            embed_dim=gca_embed_dim,
            geometry_bias=geometry_bias,
        )
        self.backbone = MultiTaskCRNN(
            n_mics=self.n_mics,
            n_freq=n_freq,
            n_classes=n_classes,
            max_k=max_k,
            spatial_filters=spatial_filters,
            freq_filters=freq_filters,
            gru_hidden=gru_hidden,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: ``(B, 2, M, F, T)`` -> ``{'spectrum', 'count'}`` (see W6)."""
        x = self.gca(x)
        return self.backbone(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
