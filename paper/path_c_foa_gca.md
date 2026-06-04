# Path C / FOA-modality GCA ablation (Tier VI)

Cross-modality replication of Stage 3: trains the same GCA mechanism over the FOA
ambisonic channels (W/X/Y/Z) where the geometry token encodes each channel's direction-
of-max-response (W = origin, X/Y/Z = unit vectors along principal axes).

## Per-cell
| Cell | n | F 20° (%) | DOAE_CD (°) | RDE | Dist_err (m) | SELD |
| ---- | - | --------- | ----------- | --- | ------------ | ---- |
| 100_foa_no_gca | 4 | 13.25 ± 0.72 | 42.63 ± 5.62 | 0.275 ± 0.013 | 0.56 ± 0.11 | 0.526 ± 0.011 |
| 130_foa_gca_full | 3 | 12.57 ± 0.70 | 39.79 ± 2.47 | 0.267 ± 0.006 | 0.53 ± 0.02 | 0.532 ± 0.003 |
| 131_foa_gca_nogeom | 3 | 12.75 ± 0.72 | 47.51 ± 6.90 | 0.287 ± 0.015 | 0.66 ± 0.11 | 0.536 ± 0.015 |

## Paired contrasts (matched seeds)
### 130_foa_gca_full__vs__100_foa_no_gca
- _FOA: GCA full vs no GCA (overall)_, n=3 matched seeds [1, 2, 3]

| Metric | mean delta (A-B) | t (p_t) | Wilcoxon (p_w) | d_z | bootstrap 95% CI |
| ------ | ---------------- | ------- | -------------- | --- | ---------------- |
| F1 | -0.560 ± 0.424 | t=-2.29 (p=0.150) | W=0.0 (p=0.250) | -1.32 | [-1.050, -0.310] |
| LE | -3.520 ± 5.186 | t=-1.18 (p=0.361) | W=1.0 (p=0.500) | -0.68 | [-9.310, +0.700] |
| RDE | -0.003 ± 0.012 | t=-0.50 (p=0.667) | W=2.0 (p=1.000) | -0.29 | [-0.010, +0.010] |
| DE | -0.027 ± 0.136 | t=-0.34 (p=0.767) | W=3.0 (p=1.000) | -0.20 | [-0.180, +0.080] |
| SELD | +0.009 ± 0.013 | t=+1.20 (p=0.353) | W=0.0 (p=0.250) | +0.69 | [+0.001, +0.024] |

### 130_foa_gca_full__vs__131_foa_gca_nogeom
- _FOA: GCA full vs no_geom (geometry contribution)_, n=3 matched seeds [1, 2, 3]

| Metric | mean delta (A-B) | t (p_t) | Wilcoxon (p_w) | d_z | bootstrap 95% CI |
| ------ | ---------------- | ------- | -------------- | --- | ---------------- |
| F1 | -0.180 ± 1.178 | t=-0.26 (p=0.816) | W=3.0 (p=1.000) | -0.15 | [-0.890, +1.180] |
| LE | -7.723 ± 4.455 | t=-3.00 (p=0.095) | W=0.0 (p=0.250) | -1.73 | [-10.400, -2.580] |
| RDE | -0.020 ± 0.010 | t=-3.46 (p=0.074) | W=0.0 (p=0.250) | -2.00 | [-0.030, -0.010] |
| DE | -0.127 ± 0.117 | t=-1.88 (p=0.201) | W=0.0 (p=0.500) | -1.08 | [-0.230, +0.000] |
| SELD | -0.004 ± 0.017 | t=-0.41 (p=0.724) | W=2.0 (p=0.750) | -0.23 | [-0.020, +0.014] |

### 131_foa_gca_nogeom__vs__100_foa_no_gca
- _FOA: channel attention alone vs no GCA_, n=3 matched seeds [1, 2, 3]

| Metric | mean delta (A-B) | t (p_t) | Wilcoxon (p_w) | d_z | bootstrap 95% CI |
| ------ | ---------------- | ------- | -------------- | --- | ---------------- |
| F1 | -0.380 ± 1.018 | t=-0.65 (p=0.584) | W=2.0 (p=0.750) | -0.37 | [-1.490, +0.510] |
| LE | +4.203 ± 5.795 | t=+1.26 (p=0.336) | W=0.0 (p=0.250) | +0.73 | [+0.630, +10.890] |
| RDE | +0.017 ± 0.015 | t=+1.89 (p=0.199) | W=0.0 (p=0.500) | +1.09 | [+0.000, +0.030] |
| DE | +0.100 ± 0.062 | t=+2.77 (p=0.109) | W=0.0 (p=0.250) | +1.60 | [+0.050, +0.170] |
| SELD | +0.013 ± 0.007 | t=+3.22 (p=0.085) | W=0.0 (p=0.250) | +1.86 | [+0.008, +0.021] |
