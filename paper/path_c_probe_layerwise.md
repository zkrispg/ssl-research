# A4 / layer-wise probing: MIC+Transformer (141 full vs 142 no_geom)

Angular probe MAE (deg; lower = direction more linearly decodable) at
increasing depth. depth 'conv' = post-conv (input to temporal, where GCA
acts); L1..L4 = TransformerEncoder layer outputs. Mean over seeds 0..2.

| depth | full MAE | no_geom MAE | delta (full-no_geom) | d_z |
| ----- | -------- | ----------- | -------------------- | --- |
| conv | 26.54 | 26.16 | +0.38 | +2.91 |
| L1 | 16.78 | 16.78 | +0.01 | +0.01 |
| L2 | 17.37 | 17.05 | +0.32 | +0.55 |
| L3 | 17.82 | 17.51 | +0.32 | +0.52 |
| L4 | 18.11 | 17.62 | +0.48 | +1.58 |

**Reading:** a geometry-prior gap (full worse) present already at 'conv'
and persisting across L1..L4 localizes the harm to the pre-temporal GCA
stage; the Transformer neither creates nor repairs it.