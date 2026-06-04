"""Quick demo: parameter counts + end-to-end shapes."""
import torch

from week09_geometry_attn.geometry_attn import count_parameters
from week11_starss23.seld_model import SeldCRNN, SeldModelConfig


print("=== Parameter counts ===")
configs = [
    ("SELDnet baseline (no GCA)", SeldModelConfig(use_gca=False)),
    ("W9 full (GCA + geom)", SeldModelConfig(use_gca=True, gca_geometry_bias=True)),
    ("W9 no_geom (GCA - geom)", SeldModelConfig(use_gca=True, gca_geometry_bias=False)),
    (
        "+ distance head (DCASE 2024)",
        SeldModelConfig(use_gca=True, gca_geometry_bias=True, use_distance_head=True),
    ),
]
for name, cfg in configs:
    p = count_parameters(SeldCRNN(cfg))
    print(f"  {name:40s}: {p:>9,}")

print()
print("=== End-to-end shapes ===")
m = SeldCRNN(SeldModelConfig(use_gca=True))
x = torch.randn(2, 10, 100, 64) * 0.1
out = m(x)
acc = out["accdoa"]
re = m.reshape_accdoa(acc)
print(f"  Input  : (2, 10, 100, 64)")
print(f"  ACCDOA flat   : {tuple(acc.shape)}")
print(f"  ACCDOA reshape: {tuple(re.shape)}  (B, T, n_tracks, 3 xyz, n_classes)")
