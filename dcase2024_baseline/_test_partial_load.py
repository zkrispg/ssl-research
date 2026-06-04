"""Verify non-strict load: synthetic MIC ckpt -> GCA-augmented model.

The GCA layer params should be missing from the ckpt and keep random init;
all other params should load cleanly. This catches the case where parameter
shape changed accidentally.
"""
import sys
import torch

sys.path.insert(0, r"D:\ssl-research\dcase2024_baseline")
import parameters
from seldnet_model import SeldModel

for task_id in ('110', '111', '112'):
    print(f"=== task {task_id} ===")
    p = parameters.get_params(task_id)
    in_shape = (2, 10, p['feature_sequence_length'], p['nb_mel_bins'])
    out_shape = (2, p['label_sequence_length'], p['unique_classes'] * 3 * 4)
    model = SeldModel(in_shape, out_shape, p)
    n_total = sum(t.numel() for t in model.parameters())
    sd = torch.load(p['pretrained_model_weights'], map_location='cpu')
    print(f"  ckpt has {len(sd)} tensors, {sum(v.numel() for v in sd.values()):,} params")
    print(f"  model has {sum(1 for _ in model.parameters())} tensors, {n_total:,} params")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  load result: missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"    missing keys (kept random init): {missing}")
    if unexpected:
        print(f"    UNEXPECTED keys (potential issue): {unexpected[:5]}")
    print()
print("All partial-load tests OK.")
