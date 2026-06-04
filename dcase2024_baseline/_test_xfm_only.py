"""Smoke test: Transformer-only variant build/forward + non-strict ckpt load."""
from __future__ import annotations
import os, sys
sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")
os.chdir(r"D:\ssl-research\dcase2024_baseline")

import numpy as np
import torch
import parameters
import seldnet_model

for task in ("140", "141", "142", "112"):  # 112 = baseline GRU + no GCA for sanity
    p = parameters.get_params(task).copy()
    in_ch = 7 if p["dataset"] == "foa" else 10
    feat_seq_len = p.get("feature_sequence_length", 250)
    label_seq_len = p["label_sequence_length"]
    n_classes = p.get("unique_classes", 13)
    nb_mel = p["nb_mel_bins"]
    in_shape = (1, in_ch, feat_seq_len, nb_mel)
    out_shape = (1, label_seq_len, n_classes * 3 * 4)
    print(f"\n=== task {task} | arch={p.get('temporal_arch','gru_mhsa')} | "
          f"use_gca={p.get('use_gca', False)} | bias={p.get('gca_geometry_bias','n/a')} ===")
    model = seldnet_model.SeldModel(in_shape, out_shape, p)
    nparams = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  trainable params = {nparams:,}")
    print(f"  has GRU: {model.gru is not None}, has Xfm: {model.transformer_encoder is not None}")
    x = torch.randn(*in_shape)
    with torch.no_grad():
        y = model(x)
    print(f"  forward OK: x{tuple(x.shape)} -> y{tuple(y.shape)}")

    # Non-strict load from synthetic ckpt
    ckpt_path = p.get("pretrained_model_weights", "")
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        miss, unex = model.load_state_dict(sd, strict=False)
        kept = len(sd) - len(unex)
        print(f"  load {ckpt_path} (strict=False): kept {kept}/{len(sd)} keys, "
              f"missing={len(miss)} unexpected={len(unex)}")
        if model.transformer_encoder is not None:
            xfm_keys = [k for k in miss if k.startswith("transformer_encoder.") or k.startswith("input_proj.")]
            print(f"    -> Xfm/proj keys in 'missing' (random init): {len(xfm_keys)}")
