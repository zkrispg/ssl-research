"""Smoke test: FOA + Transformer-only + GCA variants (Tier VIII)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")
os.chdir(r"D:\ssl-research\dcase2024_baseline")

import numpy as np
import torch
import parameters
import seldnet_model

for task in ("150", "151", "152"):
    p = parameters.get_params(task).copy()
    in_ch = 7  # FOA: 4 mel + 3 IV
    feat_seq_len = p.get("feature_sequence_length", 250)
    label_seq_len = p["label_sequence_length"]
    n_classes = p.get("unique_classes", 13)
    nb_mel = p["nb_mel_bins"]
    in_shape = (1, in_ch, feat_seq_len, nb_mel)
    out_shape = (1, label_seq_len, n_classes * 3 * 4)
    print(f"\n=== task {task} | dataset={p['dataset']} | arch={p.get('temporal_arch')} | "
          f"use_gca={p.get('use_gca', False)} | mod={p.get('gca_modality','n/a')} | "
          f"bias={p.get('gca_geometry_bias','n/a')} ===")
    model = seldnet_model.SeldModel(in_shape, out_shape, p)
    nparams = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  trainable params = {nparams:,}")
    print(f"  has GRU: {model.gru is not None}, has Xfm: {model.transformer_encoder is not None}, "
          f"has GCA: {model.gca is not None}")
    if model.gca is not None:
        print(f"  GCA M={model.gca.M}, geom_bias={model.gca.geometry_bias}")
    x = torch.randn(*in_shape)
    with torch.no_grad():
        y = model(x)
    print(f"  forward OK: x{tuple(x.shape)} -> y{tuple(y.shape)}")

    ckpt_path = p.get("pretrained_model_weights", "")
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        miss, unex = model.load_state_dict(sd, strict=False)
        kept = len(sd) - len(unex)
        print(f"  ckpt: kept {kept}/{len(sd)} keys, missing={len(miss)} unexpected={len(unex)}")
