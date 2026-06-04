"""Smoke test: FOA + GCA model build and forward."""
from __future__ import annotations
import os, sys
sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")
os.chdir(r"D:\ssl-research\dcase2024_baseline")

import numpy as np
import torch
import parameters
import seldnet_model

for task in ("130", "131", "100"):
    p = parameters.get_params(task).copy()
    print("\n=== task", task, "modality:", p.get("gca_modality", "n/a"),
          "use_gca:", p.get("use_gca", False),
          "geometry_bias:", p.get("gca_geometry_bias", "n/a"), "===")
    # FOA: 4 W/X/Y/Z log-mel + 3 intensity vector = 7. MIC: 4 mel + 6 GCC = 10.
    in_ch = 7 if p["dataset"] == "foa" else 10
    feat_seq_len = p.get("feature_sequence_length", 250)
    label_seq_len = p["label_sequence_length"]
    n_classes = p.get("unique_classes", 13)
    nb_mel = p["nb_mel_bins"]

    in_shape  = (1, in_ch, feat_seq_len, nb_mel)
    out_shape = (1, label_seq_len, n_classes * 3 * 4)
    model = seldnet_model.SeldModel(in_shape, out_shape, p)
    nparams = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  model built. trainable params = {nparams}")
    if model.gca is not None:
        print(f"  GCA module: M={model.gca.M}, embed_dim={model.gca.embed_dim}, "
              f"geometry_bias={model.gca.geometry_bias}, "
              f"mic_geom shape={tuple(model.gca.mic_geom.shape)}")
        # Quick check of a couple geom entries
        g = model.gca.mic_geom.cpu().numpy()
        print(f"  geom[0,1] = {g[0, 1]}  (W->X if FOA, mic0->mic1 if MIC)")
        print(f"  geom[1,2] = {g[1, 2]}  (X->Y if FOA, mic1->mic2 if MIC)")
    else:
        print("  GCA: None (control)")
    # Forward
    x = torch.randn(*in_shape)
    with torch.no_grad():
        y = model(x)
    print(f"  forward OK: x{tuple(x.shape)} -> y{tuple(y.shape)}")

    # Test loading the synthetic FOA pretrained ckpt (non-strict)
    ckpt_path = "3_1_dev_split0_multiaccdoa_foa_model.h5"
    if p["dataset"] == "foa" and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        miss, unex = model.load_state_dict(sd, strict=False)
        kept = sum(1 for k in sd.keys() if k not in unex)
        print(f"  synth-pretrained load (strict=False): kept {kept}/{len(sd)} keys, "
              f"missing={len(miss)} unexpected={len(unex)}")
        if model.gca is not None:
            gca_keys = [k for k in miss if k.startswith("gca.")]
            print(f"    -> GCA keys in 'missing' (random init): {len(gca_keys)} of {len(miss)}")
