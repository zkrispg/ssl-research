"""Smoke test: Conformer variant build/forward + non-strict ckpt load."""
from __future__ import annotations
import os, sys
sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")
os.chdir(r"D:\ssl-research\dcase2024_baseline")

import torch
import parameters
import seldnet_model

for task in ("160", "161", "162", "170", "171", "172"):
    p = parameters.get_params(task).copy()
    in_ch = 7 if p["dataset"] == "foa" else 10
    feat_seq_len = p.get("feature_sequence_length", 250)
    label_seq_len = p["label_sequence_length"]
    n_classes = p.get("unique_classes", 13)
    nb_mel = p["nb_mel_bins"]
    in_shape = (1, in_ch, feat_seq_len, nb_mel)
    out_shape = (1, label_seq_len, n_classes * 3 * 4)
    print(f"\n=== task {task} | arch={p.get('temporal_arch','gru_mhsa')} | "
          f"dataset={p['dataset']} | use_gca={p.get('use_gca', False)} | "
          f"bias={p.get('gca_geometry_bias','n/a')} | gca_mod={p.get('gca_modality','mic')} ===")
    model = seldnet_model.SeldModel(in_shape, out_shape, p)
    nparams = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  trainable params = {nparams:,}")
    print(f"  has GRU: {model.gru is not None}, has Conformer: {model.conformer is not None}")
    x = torch.randn(*in_shape)
    with torch.no_grad():
        y = model(x)
    print(f"  forward OK: x{tuple(x.shape)} -> y{tuple(y.shape)}")
    assert tuple(y.shape) == out_shape, f"output shape mismatch: {tuple(y.shape)} vs {out_shape}"

    ckpt_path = p.get("pretrained_model_weights", "")
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        miss, unex = model.load_state_dict(sd, strict=False)
        kept = len(sd) - len(unex)
        print(f"  load {ckpt_path} (strict=False): kept {kept}/{len(sd)} keys, "
              f"missing={len(miss)} unexpected={len(unex)}")
        conf_keys = [k for k in miss if k.startswith("conformer.") or k.startswith("input_proj.")]
        print(f"    -> conformer/proj keys in 'missing' (random init): {len(conf_keys)}")

print("\nALL CONFORMER SMOKE TESTS PASSED.")
