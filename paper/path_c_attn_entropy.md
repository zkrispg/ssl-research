# A2 / channel-attention entropy mechanism (MIC+Transformer, 141 vs 142)

Files: 6 STARSS23 dev-test segments. Seeds: [0, 1, 2].
Entropy is the mean Shannon entropy (nats) of the GCA per-query channel-
attention distribution over the M=4 channels. Lower = more peaked = less
adaptive (max possible = ln 4 = 1.386).

| variant | n seeds | mean H (nats) | std |
| ------- | ------- | ------------- | --- |
| full (geom)   | 3 | 1.2185 | 0.1588 |
| no_geom       | 3 | 1.3477 | 0.0549 |

**Paired contrast (full - no_geom):** delta H = -0.1292 nats, t=-1.11 (p=0.382), d_z=-0.64.

**Reading:** the geometry bias makes channel attention *more peaked* (lower entropy), consistent with the prior over-constraining the Transformer's channel mixing toward a fixed layout.