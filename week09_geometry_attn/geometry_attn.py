"""Geometry-aware Channel Attention (GCA) for multi-mic SSL.

Motivation
----------
Classical SSL (GCC-PHAT, SRP-PHAT, MUSIC) relies directly on the
*physical positions* of the microphones: every formula involves the
inter-mic distance and bearing. Deep models such as our W6 baseline,
in contrast, treat the M-channel phase tensor as just a stack of input
channels and have to learn the array geometry implicitly from data.
This works for the small UCA4 we trained on but is obviously not
ideal -- the model has no way to know that mic 0 and mic 2 are 8 cm
apart along x while mic 0 and mic 1 are sqrt(2)*4 cm apart at 45 deg.

GCA adds a tiny attention module *before* W6's spatial convolution
stack. It computes per-mic channel embeddings from the phase features
(via global pooling), runs a single-head self-attention over the M mics,
and biases the attention keys with a *geometry token* derived from the
relative mic-pair positions ``(dx, dy, distance, bearing)``. The output
is a per-mic gate that re-weights the phase channels.

Properties
----------
* Tiny (~1-3 K parameters) -- adds <5 % to the W6 backbone.
* Fully differentiable; geometry tokens are deterministic functions of
  ``mic_positions`` (registered as a buffer, not a parameter).
* Drops in cleanly with ``GeometryAwareChannelAttention(mic_positions)``
  and exposes a ``forward(x)`` that returns ``x * gate``.
* Can be ablated by setting ``geometry_bias=False``, which collapses GCA
  into a plain SE-style channel attention (no geometry knowledge).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def mic_pair_geometry(mic_positions: np.ndarray) -> np.ndarray:
    """Compute pairwise geometry features for an array of M mics.

    Returns an ``(M, M, 4)`` float32 array whose entry ``[i, j]`` is

        ``(dx_ij, dy_ij, distance_ij, bearing_ij_rad)``

    where ``(dx_ij, dy_ij)`` is the 2-D vector from mic i to mic j (we
    project onto the array plane and ignore z), ``distance_ij`` is the
    Euclidean inter-mic distance (m), and ``bearing_ij_rad`` is
    ``atan2(dy, dx)`` in radians. The diagonal is filled with zeros.
    """
    pos = np.asarray(mic_positions, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[0] < 2 or pos.shape[1] < 2:
        raise ValueError(f"expected (M, >=2) array, got shape {pos.shape}")
    if pos.shape[1] >= 3:
        pos = pos[:, :2]
    M = pos.shape[0]
    diff = pos[None, :, :] - pos[:, None, :]  # (M, M, 2)
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    bearing = np.arctan2(diff[..., 1:2], diff[..., 0:1])
    geom = np.concatenate([diff, dist, bearing], axis=-1).astype(np.float32)
    np.fill_diagonal(geom[..., 0], 0.0)
    np.fill_diagonal(geom[..., 1], 0.0)
    np.fill_diagonal(geom[..., 2], 0.0)
    np.fill_diagonal(geom[..., 3], 0.0)
    return geom


class GeometryAwareChannelAttention(nn.Module):
    """Single-head self-attention over the mic dimension with a
    geometry bias on the keys.

    Args:
        mic_positions: ``(M, 2)`` or ``(M, 3)`` array of mic coordinates
            in metres. Stored as a buffer.
        in_channels: number of feature channels in the phase tensor
            (typically 2 for sin/cos).
        embed_dim: hidden dimension of the per-mic and per-pair
            embeddings.
        geometry_bias: when ``False`` the key has no geometry term and
            the module reduces to a plain channel attention (used for
            ablation).
    """

    def __init__(
        self,
        mic_positions: np.ndarray | None = None,
        in_channels: int = 2,
        embed_dim: int = 16,
        geometry_bias: bool = True,
        pair_geom: np.ndarray | None = None,
    ) -> None:
        """If ``pair_geom`` is provided (shape ``(M, M, 4)``), it is used
        directly and ``mic_positions`` is ignored. This lets non-MIC
        modalities (e.g. FOA ambisonic channels) supply a custom
        geometry feature: the rest of the module is unchanged.
        """
        super().__init__()
        if pair_geom is not None:
            geom_np = np.asarray(pair_geom, dtype=np.float32)
            if geom_np.ndim != 3 or geom_np.shape[-1] != 4:
                raise ValueError(
                    f"pair_geom must have shape (M, M, 4); got {geom_np.shape}"
                )
        else:
            if mic_positions is None:
                raise ValueError("must supply either mic_positions or pair_geom")
            geom_np = mic_pair_geometry(mic_positions)
        self.M = geom_np.shape[0]
        self.embed_dim = embed_dim
        self.geometry_bias = geometry_bias

        self.register_buffer(
            "mic_geom", torch.from_numpy(geom_np), persistent=False
        )

        self.feat_proj = nn.Linear(in_channels, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.geom_proj = nn.Linear(4, embed_dim) if geometry_bias else None
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GCA to a phase feature tensor.

        Args:
            x: ``(B, C, M, F, T)`` where ``C`` is ``in_channels`` (e.g. 2
                for sin/cos), ``M`` is the number of mics.

        Returns:
            Tensor of the same shape, modulated by a per-mic
            sigmoid gate broadcast over ``F`` and ``T``.
        """
        if x.ndim != 5 or x.shape[2] != self.M:
            raise ValueError(
                f"expected (B, C, {self.M}, F, T), got {tuple(x.shape)}"
            )
        B, C, M, F, T = x.shape

        per_mic = x.mean(dim=(3, 4)).transpose(1, 2)  # (B, M, C)
        e = self.feat_proj(per_mic)  # (B, M, D)
        q = self.q_proj(e)  # (B, M, D)
        k = self.k_proj(e)  # (B, M, D)
        v = self.v_proj(e)  # (B, M, D)
        if self.geometry_bias:
            geom_bias = self.geom_proj(self.mic_geom)  # (M, M, D)
            k = k.unsqueeze(1) + geom_bias.unsqueeze(0)  # (B, M_q, M_k, D)
            scores = torch.einsum("bqd,bqkd->bqk", q, k) / math.sqrt(self.embed_dim)
        else:
            scores = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(self.embed_dim)
        attn = torch.softmax(scores, dim=-1)  # (B, M_q, M_k)
        ctx = torch.einsum("bqk,bkd->bqd", attn, v)  # (B, M, D)

        gate = torch.sigmoid(self.gate_proj(ctx))  # (B, M, 1)
        gate = gate.view(B, 1, M, 1, 1)
        return x * gate


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
