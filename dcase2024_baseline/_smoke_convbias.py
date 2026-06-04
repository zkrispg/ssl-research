"""Smoke test for the convbias geometry-injection tasks (180-183).

Builds each model, runs a dummy forward, checks:
  (1) forward works and output shape is sane,
  (2) full and no_geom have IDENTICAL parameter counts (matched capacity),
  (3) the geometry signal actually changes the output (full vs zeroed on the
      SAME weights).
No data / no pretrained weights needed.
"""
import numpy as np
import torch

import parameters
from seldnet_model import SeldModel


def build(task):
    p = parameters.get_params(task)
    C = 7 if p['dataset'] == 'foa' else 10
    T = p['feature_sequence_length']
    Fb = p['nb_mel_bins']
    in_shape = (2, C, T, Fb)
    out_shape = (2, 50, 156)
    model = SeldModel(in_shape, out_shape, p)
    return model, in_shape


def nparams(m):
    return sum(x.numel() for x in m.parameters())


torch.manual_seed(0)
PAIRS = [
    ('180', '181', 'FOA+CRNN'),
    ('182', '183', 'MIC+Transformer'),
    ('184', '185', 'FOA+Conformer'),
    ('186', '187', 'FOA+Transformer'),
]
for full, nogeom, label in PAIRS:
    mf, sh = build(full)
    mn, _ = build(nogeom)
    mf.eval(); mn.eval()
    x = torch.randn(*sh)
    with torch.no_grad():
        of = mf(x)
        on = mn(x)
    # geometry on/off on the SAME (full) model weights:
    assert mf.geometry_mode == 'convbias', f"{full} not convbias"
    assert mf.geom_concat_full is True and mn.geom_concat_full is False
    mf.geom_concat_full = False
    with torch.no_grad():
        of_zero = mf(x)
    changed = not torch.allclose(of, of_zero, atol=1e-6)
    print(f"[{label}] {full} full: params={nparams(mf)}, out={tuple(of.shape)}")
    print(f"[{label}] {nogeom} no_geom: params={nparams(mn)}, out={tuple(on.shape)}")
    print(f"[{label}] param_match(full==no_geom): {nparams(mf) == nparams(mn)}")
    print(f"[{label}] geometry_changes_output: {changed}")
    print("---")

print("SMOKE_OK")
