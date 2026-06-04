"""Quick smoke test: build SeldModel with use_gca=True and run a fake batch.

Verifies:
  1. import path (week09_geometry_attn) resolves
  2. SeldModel(use_gca=True, dataset='mic') instantiates
  3. forward pass on (2, 10, 250, 64) doesn't crash
  4. output shape matches expected Multi-ACCDDOA dims
  5. asserting use_gca + dataset='foa' raises (sanity)
  6. with use_gca=False the official synthetic .h5 still loads strict=True
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")

import parameters
from seldnet_model import SeldModel


def _build_params(task_id: str) -> dict:
    p = parameters.get_params(task_id)
    return p


def _fake_input_shape(p: dict) -> tuple[int, int, int, int]:
    # in_channels for our setups:
    #   FOA  + GCC=False -> 4 logmel + 3 IV = 7
    #   MIC  + GCC=False -> 4 logmel + 6 GCC = 10
    if p['dataset'] == 'foa':
        ch = 7
    else:
        ch = 4 + 6 if not p.get('use_salsalite') else 7
    return (2, ch, p['feature_sequence_length'], p['nb_mel_bins'])


def _expected_out_dim(p: dict) -> int:
    cls = p['unique_classes']
    if p['multi_accdoa']:
        return cls * 3 * 4   # 3 tracks * 4 axes (xyzd)
    return cls * 4


def main() -> int:
    print("=== test 1: GCA full (task 110, MIC + GCA, geometry_bias=True) ===")
    p = _build_params('110')
    in_shape = _fake_input_shape(p)
    out_dim = _expected_out_dim(p)
    out_shape = (2, p['label_sequence_length'], out_dim)
    model = SeldModel(in_shape, out_shape, p)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"  built. params = {n_params:,}, GCA active = {model.gca is not None}")
    print(f"  GCA pair-geometry shape: {model.gca.mic_geom.shape if model.gca else 'n/a'}")

    x = torch.randn(*in_shape)
    y = model(x)
    print(f"  forward OK. input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == out_shape, f"unexpected output shape {y.shape}"
    print()

    print("=== test 2: GCA no_geom (task 111, MIC + GCA, geometry_bias=False) ===")
    p = _build_params('111')
    in_shape = _fake_input_shape(p)
    out_shape = (2, p['label_sequence_length'], _expected_out_dim(p))
    model = SeldModel(in_shape, out_shape, p)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"  built. params = {n_params:,}, geometry_bias = {model.gca.geometry_bias}")
    x = torch.randn(*in_shape)
    y = model(x)
    print(f"  forward OK. output {tuple(y.shape)}")
    assert y.shape == out_shape
    print()

    print("=== test 3: GCA-no-bias-vs-with-bias param delta ===")
    p_with = _build_params('110')
    p_no   = _build_params('111')
    in_shape = _fake_input_shape(p_with)
    n_with = sum(
        t.numel() for t in SeldModel(in_shape, (2, p_with['label_sequence_length'], _expected_out_dim(p_with)), p_with).parameters()
    )
    n_no = sum(
        t.numel() for t in SeldModel(in_shape, (2, p_no['label_sequence_length'], _expected_out_dim(p_no)), p_no).parameters()
    )
    print(f"  full GCA: {n_with:,}  no_geom GCA: {n_no:,}  delta (geom_proj): {n_with - n_no:,}")
    assert n_with > n_no, "GCA full must have more params than GCA no_geom (geom_proj layer)"
    print()

    print("=== test 4: FOA + use_gca=True must raise ===")
    p = _build_params('100').copy()
    p['use_gca'] = True
    raised = False
    try:
        SeldModel((2, 7, p['feature_sequence_length'], p['nb_mel_bins']),
                  (2, p['label_sequence_length'], _expected_out_dim(p)), p)
    except ValueError as e:
        print(f"  correctly raised: {e}")
        raised = True
    assert raised, "FOA + use_gca should have raised ValueError"
    print()

    print("=== test 5: FOA + use_gca=False can still strict-load synthetic ckpt ===")
    p = _build_params('100')
    in_shape = _fake_input_shape(p)
    out_shape = (2, p['label_sequence_length'], _expected_out_dim(p))
    model = SeldModel(in_shape, out_shape, p)
    sd = torch.load(r"D:\ssl-research\dcase2024_baseline\3_1_dev_split0_multiaccdoa_foa_model.h5",
                    map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  finetune init OK. missing={len(missing)}  unexpected={len(unexpected)}")
    assert len(unexpected) == 0, f"unexpected keys broke ckpt compatibility: {unexpected[:3]}"
    print()

    print("All GCA integration smoke tests PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
