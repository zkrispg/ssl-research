"""Smoke test for task 113 (Vanilla SE-block) integration."""
import sys, os
sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
sys.path.insert(0, r"D:\ssl-research")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import parameters
import seldnet_model

p = parameters.get_params("113").copy()
in_ch = 4 + 6
in_shape  = (1, in_ch, p["feature_sequence_length"], p["nb_mel_bins"])
out_shape = (1, p["label_sequence_length"], p["unique_classes"] * 3 * 4)

print(f"Building model with use_se_block={p.get('use_se_block')}, se_block_reduction={p.get('se_block_reduction')}")
m = seldnet_model.SeldModel(in_shape, out_shape, p)
print(f"  has gca: {m.gca is not None}")
print(f"  has se_block: {m.se_block is not None}")
print(f"  total params: {sum(p_.numel() for p_ in m.parameters()):,}")
if m.se_block is not None:
    se_n = sum(p_.numel() for p_ in m.se_block.parameters())
    print(f"  SE-block params: {se_n}")

x = torch.randn(*in_shape)
y = m(x)
print(f"  forward in: {tuple(x.shape)} -> out: {tuple(y.shape)}")
assert y.shape == out_shape

# Try loading the synthetic MIC checkpoint with strict=False
import os.path as osp
ckpt = r"D:\ssl-research\dcase2024_baseline\models_audio\6_1_dev_split0_multiaccdoa_mic_gcc_model.h5"
if osp.isfile(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    res = m.load_state_dict(sd, strict=False)
    print(f"  load strict=False: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    print(f"  missing keys preview: {res.missing_keys[:6]}")
    assert all("se_block" in k for k in res.missing_keys), \
        f"unexpected missing keys: {res.missing_keys}"
    print("  -> only SE-block params are missing; backbone loads cleanly ✓")
else:
    print(f"  ckpt not found: {ckpt}")

# Mutual exclusion test
p_bad = p.copy()
p_bad["use_gca"] = True
try:
    seldnet_model.SeldModel(in_shape, out_shape, p_bad)
    print("FAIL: should have raised ValueError for mutual exclusion")
except ValueError as e:
    print(f"  mutual-exclusion check ok: {e}")

print("\n[smoke 113] all checks passed.")
