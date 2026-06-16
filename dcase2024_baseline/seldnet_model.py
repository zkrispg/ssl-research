# The SELDnet architecture
#
# This file extends the official DCASE 2024 baseline with optional
# Geometry-Aware Channel Attention (GCA) on the first n_mics input
# channels (per-mic log-mel block). GCA is only meaningful for MIC array
# inputs whose first 4 channels correspond to physical microphones; on
# FOA inputs (W/X/Y/Z spherical-harmonic features) GCA should be left off.
#
# The official architecture/state-dict is preserved when use_gca=False so
# the synthetic-pretrained .h5 checkpoint loads with strict=True.
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from IPython import embed  # noqa: F401
except ImportError:
    def embed(*args, **kwargs):  # type: ignore
        raise RuntimeError("IPython.embed() requested but IPython not installed")

# Make the SSL-research GCA module importable regardless of cwd or drive letter.
_SSL_REPO_ROOTS = [
    str(Path(__file__).resolve().parents[1]),
    r"D:\ssl-research",
]
for _SSL_REPO_ROOT in reversed(_SSL_REPO_ROOTS):
    if _SSL_REPO_ROOT not in sys.path:
        sys.path.insert(0, _SSL_REPO_ROOT)

try:
    from week09_geometry_attn.geometry_attn import (  # type: ignore
        GeometryAwareChannelAttention,
    )
    _HAS_GCA = True
except ImportError:
    GeometryAwareChannelAttention = None  # type: ignore
    _HAS_GCA = False


def foa_ambisonic_pair_geometry() -> np.ndarray:
    """Pairwise geometry features for first-order ambisonic channels.

    Channel order: W (omnidirectional), X (dipole +x), Y (dipole +y),
    Z (dipole +z). Returns an (M=4, M=4, 4) float32 tensor whose entry
    [i, j] is ``(dx, dy, dz, both_directional)``:

      - ``(dx, dy, dz)`` = direction-of-max-response of channel j
        minus that of channel i (W is treated as the origin).
      - ``both_directional`` = 1.0 if both i and j are directional
        (X/Y/Z), else 0.0. Encodes the special-cased role of W.

    This is a 4-d feature so the GCA `geom_proj: Linear(4, embed_dim)`
    is shared with the MIC geometry tokenizer.
    """
    dirs = np.array([
        [0.0, 0.0, 0.0],  # W: omni (origin)
        [1.0, 0.0, 0.0],  # X: dipole +x
        [0.0, 1.0, 0.0],  # Y: dipole +y
        [0.0, 0.0, 1.0],  # Z: dipole +z
    ], dtype=np.float32)
    M = dirs.shape[0]
    is_dir = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)  # 1 for X/Y/Z
    geom = np.zeros((M, M, 4), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            geom[i, j, :3] = dirs[j] - dirs[i]
            geom[i, j, 3]  = is_dir[i] * is_dir[j]  # 1 only if both directional
    return geom


def starss23_tetrahedral_mic_positions(radius_m: float = 0.042) -> np.ndarray:
    """4-mic regular tetrahedral arrangement at the given circumradius.

    Mirrors the Eigenmike's 4-channel subset used by the STARSS23 MIC
    dataset. The exact xyz values do not matter for GCA: only the
    pairwise distance / bearing matter, which are invariant to rotation /
    translation. A regular tetrahedron of circumradius 0.042 m matches
    the documented Eigenmike radius.
    """
    # Vertices of a regular tetrahedron centered at origin, scaled to
    # circumradius `radius_m`. Reference vertices on a unit cube:
    base = np.array(
        [
            [+1.0, +1.0, +1.0],
            [+1.0, -1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
        ],
        dtype=np.float64,
    )
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    return (base[:, :2] * radius_m).astype(np.float32)  # use 2-D xy for GCA


def per_channel_geometry_vector(modality: str) -> np.ndarray:
    """Per-channel layout descriptor for the convolutional-bias geometry
    injection (a SECOND injection mechanism, alternative to GCA's
    attention-key bias).

    Returns a flat ``(G,)`` vector of layout constants, one group per input
    microphone / ambisonic channel:
      - MIC: the tetrahedral mic xy positions (rotation/translation-normalized),
        ``G = 4*2 = 8``.
      - FOA: each channel's direction-of-max-response (W = origin, X/Y/Z unit
        axes), ``G = 4*3 = 12``.

    This carries the same array-layout information as GCA but is injected by a
    different mechanism (a learned linear projection added as a per-filter bias
    to the first conv feature maps), letting us test whether the
    architecture-graded geometry effect is specific to GCA-style injection or
    generalizes across injection mechanisms.
    """
    if str(modality).lower() == 'foa':
        dirs = np.array([
            [0.0, 0.0, 0.0],  # W: omni (origin)
            [1.0, 0.0, 0.0],  # X: dipole +x
            [0.0, 1.0, 0.0],  # Y: dipole +y
            [0.0, 0.0, 1.0],  # Z: dipole +z
        ], dtype=np.float32)
        return dirs.reshape(-1)
    base = np.array(
        [
            [+1.0, +1.0, +1.0],
            [+1.0, -1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
        ],
        dtype=np.float64,
    )
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    return base[:, :2].astype(np.float32).reshape(-1)


class MSELoss_ADPIT(object):
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*4*13]
            target: [batch_size, frames, num_track_dummy=6, num_axis=5, num_class=13]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZD)=4, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*4, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*4, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class VanillaSEBlock(nn.Module):
    """Plain Squeeze-and-Excitation block over feature channels.

    Operates on (B, C, T, F) tensors and produces a per-channel sigmoid
    gate. NO geometric structure of any kind. Used as the channel-level
    counterpart to GCA's per-mic attention so we can disentangle
    'channel attention helps' from 'per-mic attention with geometry helps'.

    Args:
        in_channels: number of feature channels (C).
        reduction: bottleneck reduction ratio.
    """

    def __init__(self, in_channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, in_channels, bias=True)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"VanillaSEBlock expected (B, C, T, F), got {tuple(x.shape)}")
        B, C, T, F_ = x.shape
        if C != self.in_channels:
            raise ValueError(f"channel mismatch: model expected {self.in_channels}, got {C}")
        pooled = x.mean(dim=(2, 3))           # (B, C)
        z = F.relu(self.fc1(pooled))          # (B, C/r)
        gate = torch.sigmoid(self.fc2(z))     # (B, C)
        return x * gate.view(B, C, 1, 1)


class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params=params

        # ---------------- Optional GCA prefix (MIC arrays only) ---------------
        # When enabled, GCA gates the first `n_mics` input channels (per-mic
        # log-mel block) using a single-head self-attention with geometry
        # bias derived from a tetrahedral mic arrangement. Setting
        # gca_geometry_bias=False ablates GCA into a plain SE-style channel
        # attention, which is our 'no_geom' control variant.
        self.use_gca = bool(params.get('use_gca', False))
        self.gca_n_mics = int(params.get('gca_n_mics', 4))
        # `gca_modality` controls the geometry token: "mic" uses the tetrahedral
        # mic xy positions (default, MIC mode), "foa" uses ambisonic channel
        # direction vectors (W/X/Y/Z). These produce 4-d pair geometry features
        # so the rest of GCA is unchanged.
        self.gca_modality = str(params.get('gca_modality', 'mic')).lower()
        if self.use_gca:
            if not _HAS_GCA:
                raise RuntimeError(
                    "GCA requested but week09_geometry_attn module not importable; "
                    "ensure the ssl-research repo root is on sys.path."
                )
            if self.gca_modality == 'foa':
                # FOA path: 4 ambisonic channels (W/X/Y/Z). Geometry token
                # encodes each channel's direction-of-max-response. We treat
                # the W (omnidirectional) channel as the origin and X/Y/Z as
                # unit vectors along the principal axes; pair geometry shape
                # (4, 4, 4) matches the MIC tokenizer interface.
                pair_geom = foa_ambisonic_pair_geometry()
                self.gca = GeometryAwareChannelAttention(
                    pair_geom=pair_geom,
                    in_channels=1,
                    embed_dim=int(params.get('gca_embed_dim', 16)),
                    geometry_bias=bool(params.get('gca_geometry_bias', True)),
                )
            else:
                if params.get('dataset', 'foa') == 'foa':
                    raise ValueError(
                        "use_gca with dataset='foa' requires gca_modality='foa'."
                    )
                mic_pos = starss23_tetrahedral_mic_positions()
                self.gca = GeometryAwareChannelAttention(
                    mic_positions=mic_pos,
                    in_channels=1,
                    embed_dim=int(params.get('gca_embed_dim', 16)),
                    geometry_bias=bool(params.get('gca_geometry_bias', True)),
                )
        else:
            self.gca = None

        # ---------- Optional vanilla SE-block on the full input channels ----------
        # Mutually exclusive with GCA. Operates on the full (B, in_C, T, F) tensor
        # right before the conv stack, gates each of the `in_C` input channels
        # (4 logmel + 6 GCC for MIC). This is the channel-level counterpart to
        # GCA's per-mic gating; only differs from GCA no_geom in two ways:
        #   1. scope is over feature channels, not mic dim
        #   2. mechanism is plain MLP, not Q/K/V self-attention
        self.use_se_block = bool(params.get('use_se_block', False))
        if self.use_se_block:
            if self.use_gca:
                raise ValueError("use_se_block and use_gca are mutually exclusive controls.")
            in_ch = int(in_feat_shape[1])
            self.se_block = VanillaSEBlock(
                in_channels=in_ch,
                reduction=int(params.get('se_block_reduction', 2)),
            )
        else:
            self.se_block = None

        # ---------- Optional convolutional-bias geometry injection ----------
        # A SECOND geometry-prior injection mechanism, distinct from GCA. The
        # per-channel layout descriptor is mapped by a learned linear projection
        # and ADDED as a per-filter bias to the first conv feature maps. This is
        # mechanistically different from GCA (which biases channel-attention
        # keys) but carries the same array-layout information, so it tests
        # whether the architecture-graded effect generalizes across injection
        # mechanisms. full and no_geom share an IDENTICAL parameter count: the
        # no_geom variant feeds zeros through the same projection (its output is
        # then exactly zero), so the only difference is the geometry signal.
        self.geometry_mode = str(params.get('geometry_mode', 'none')).lower()
        self.geom_concat_full = bool(params.get('gca_geometry_bias', True))
        if self.geometry_mode == 'convbias':
            if self.use_gca or self.use_se_block:
                raise ValueError(
                    "geometry_mode='convbias' is exclusive with use_gca/use_se_block."
                )
            geom_vec = per_channel_geometry_vector(self.gca_modality)
            self.register_buffer(
                'geom_concat_vec', torch.from_numpy(geom_vec), persistent=False
            )
            self.geom_proj_concat = nn.Linear(
                int(geom_vec.shape[0]), int(params['nb_cnn2d_filt']), bias=False
            )
        else:
            self.geom_proj_concat = None

        self.conv_block_list = nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        # `temporal_arch` selects the temporal modeling stack between the conv
        # block and the FNN head:
        #   "gru_mhsa" (default) -> 2x bidirectional GRU + 2x MHSA  (DCASE 2024 baseline)
        #   "transformer"        -> Linear projection + N TransformerEncoder layers
        # The transformer-only variant drops recurrent processing entirely and
        # replaces it with self-attention, letting us test whether the GCA
        # finding generalizes to a non-CRNN backbone.
        self.temporal_arch = str(params.get('temporal_arch', 'gru_mhsa')).lower()
        if self.temporal_arch == 'transformer':
            # Replace GRU with a Linear projection (gru_input_dim -> rnn_size).
            self.gru = None
            self.input_proj = nn.Linear(self.gru_input_dim, params['rnn_size'])
            n_xfm_layers = int(params.get('nb_transformer_blocks', 4))
            xfm_layer = nn.TransformerEncoderLayer(
                d_model=params['rnn_size'],
                nhead=params['nb_heads'],
                dim_feedforward=int(params.get('xfm_ff_dim', 4 * params['rnn_size'])),
                dropout=params['dropout_rate'],
                batch_first=True,
                activation='gelu',
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(xfm_layer, num_layers=n_xfm_layers)
            self.conformer = None
            # MHSA blocks are unused in this branch (kept empty for state-dict
            # compatibility on the GRU baseline path).
            self.mhsa_block_list = nn.ModuleList()
            self.layer_norm_list = nn.ModuleList()
        elif self.temporal_arch == 'conformer':
            # Conformer temporal stack: Linear projection + N Conformer blocks
            # (macaron FFN + MHSA + depthwise-conv module). A third backbone
            # point on the architecture axis -- a conv/attention hybrid -- to
            # test whether the geometry-prior effect tracks the attention
            # mechanism rather than any one specific model.
            import torchaudio
            self.gru = None
            self.input_proj = nn.Linear(self.gru_input_dim, params['rnn_size'])
            n_conf_layers = int(params.get('nb_conformer_blocks', 4))
            self.conformer = torchaudio.models.Conformer(
                input_dim=params['rnn_size'],
                num_heads=params['nb_heads'],
                ffn_dim=int(params.get('conformer_ff_dim', 4 * params['rnn_size'])),
                num_layers=n_conf_layers,
                depthwise_conv_kernel_size=int(params.get('conformer_conv_kernel', 31)),
                dropout=params['dropout_rate'],
            )
            self.transformer_encoder = None
            self.mhsa_block_list = nn.ModuleList()
            self.layer_norm_list = nn.ModuleList()
        else:
            self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
            self.input_proj = None
            self.transformer_encoder = None
            self.conformer = None

            self.mhsa_block_list = nn.ModuleList()
            self.layer_norm_list = nn.ModuleList()
            for mhsa_cnt in range(params['nb_self_attn_layers']):
                self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
                self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        # fusion layers
        if in_vid_feat_shape is not None:
            self.visual_embed_to_d_model = nn.Linear(in_features = int(in_vid_feat_shape[2]*in_vid_feat_shape[3]), out_features = self.params['rnn_size'] )
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.params['rnn_size'], nhead=self.params['nb_heads'], batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.params['nb_transformer_layers'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def _maybe_apply_gca(self, x: torch.Tensor) -> torch.Tensor:
        """Gate the first `n_mics` per-mic channels with GCA. Other
        channels (GCC-PHAT pairs, IV components, etc.) pass through.

        Args:
            x: ``(B, in_channels, T_feat, F)`` feature tensor.
        """
        if self.gca is None:
            return x
        n_mics = self.gca_n_mics
        if x.shape[1] < n_mics:
            return x
        mic_block = x[:, :n_mics]            # (B, M, T, F)
        rest = x[:, n_mics:]                  # (B, in-M, T, F)
        # GCA expects (B, C=1, M, F, T):
        mic_for_gca = mic_block.unsqueeze(1).permute(0, 1, 2, 4, 3).contiguous()
        gated = self.gca(mic_for_gca)         # (B, 1, M, F, T)
        mic_block_gated = gated.squeeze(1).permute(0, 1, 3, 2).contiguous()
        return torch.cat([mic_block_gated, rest], dim=1)

    def _maybe_apply_se(self, x: torch.Tensor) -> torch.Tensor:
        """Apply VanillaSEBlock to (B, C, T, F) input if enabled."""
        if self.se_block is None:
            return x
        return self.se_block(x)

    def _maybe_add_geom_bias(self, x: torch.Tensor) -> torch.Tensor:
        """Add the geometry-derived per-filter bias to the first conv feature
        maps (convbias injection). The no_geom variant feeds zeros, so the
        projection contributes exactly zero while keeping the parameter count
        matched to full."""
        if self.geometry_mode != 'convbias' or self.geom_proj_concat is None:
            return x
        vec = self.geom_concat_vec if self.geom_concat_full \
            else torch.zeros_like(self.geom_concat_vec)
        bias = self.geom_proj_concat(vec)          # (nb_cnn2d_filt,)
        return x + bias.view(1, -1, 1, 1)

    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        x = self._maybe_apply_gca(x)
        x = self._maybe_apply_se(x)
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
            if conv_cnt == 0:
                x = self._maybe_add_geom_bias(x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        if self.temporal_arch == 'transformer':
            x = self.input_proj(x)              # (B, T, rnn_size)
            x = self.transformer_encoder(x)     # (B, T, rnn_size)
        elif self.temporal_arch == 'conformer':
            x = self.input_proj(x)              # (B, T, rnn_size)
            lengths = torch.full((x.shape[0],), x.shape[1],
                                 dtype=torch.long, device=x.device)
            x, _ = self.conformer(x, lengths)   # (B, T, rnn_size)
        else:
            (x, _) = self.gru(x)
            x = torch.tanh(x)
            x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

            for mhsa_cnt in range(len(self.mhsa_block_list)):
                x_attn_in = x
                x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
                x = x + x_attn_in
                x = self.layer_norm_list[mhsa_cnt](x)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            x = self.transformer_decoder(x, vid_feat)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x)
        # the below-commented code applies tanh for doa and relu for distance estimates respectively in multi-accdoa scenarios.
        # they can be uncommented and used, but there is no significant changes in the results.
        #doa = doa.reshape(doa.size(0), doa.size(1), 3, 4, 13)
        #doa1 = doa[:, :, :, :3, :]
        #dist = doa[:, :, :, 3:, :]

        #doa1 = self.doa_act(doa1)
        #dist = self.dist_act(dist)
        #doa2 = torch.cat((doa1, dist), dim=3)

        #doa2 = doa2.reshape((doa.size(0), doa.size(1), -1))
        #return doa2
        return doa
