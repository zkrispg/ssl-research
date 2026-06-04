# Path C / 2x2 ANOVA: formal interaction test for (modality, arch, prior)

OLS factorial ANOVA on per-seed metrics with three between-cell
categorical factors (modality, arch, prior) and seed as the unit of
replication. Type-II SS. Reference levels: modality=MIC, arch=CRNN, prior=nogeom.

Note: cells have 3-5 seeds each (unbalanced). The key statistic is
the **3-way interaction term** modality:arch:prior, which tests whether
the prior's effect direction depends on (modality, arch).

## LE (DOAE_CD)
- n_obs = 60, R^2 = 0.241

| factor | F | PR(>F) |
| ------ | - | ------ |
| C(modality, Treatment('MIC')) | 1.52 | 0.2238 |
| C(arch, Treatment('CRNN')) | 3.08 | 0.0552 . |
| C(prior, Treatment('nogeom')) | 0.02 | 0.8755 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')) | 1.20 | 0.3102 |
| C(modality, Treatment('MIC')):C(prior, Treatment('nogeom')) | 0.26 | 0.6124 |
| C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 1.94 | 0.1546 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 0.51 | 0.6018 |

### Within-cell paired contrasts (full - nogeom) on LE
| cell | n | delta mean | t (p_t) | d_z |
| ---- | - | ---------- | ------- | --- |
| MIC+CRNN | 5 | -0.882 ± 12.998 | t=-0.15 (p=0.887) | -0.07 |
| MIC+Xfm | 5 | +5.698 ± 7.980 | t=+1.60 (p=0.186) | +0.71 |
| MIC+Conformer | 5 | -1.832 ± 11.777 | t=-0.35 (p=0.745) | -0.16 |
| FOA+CRNN | 5 | -5.024 ± 4.871 | t=-2.31 (p=0.082) | -1.03 |
| FOA+Xfm | 5 | +2.570 ± 6.662 | t=+0.86 (p=0.437) | +0.39 |
| FOA+Conformer | 5 | +0.878 ± 6.499 | t=+0.30 (p=0.778) | +0.14 |

## F1
- n_obs = 60, R^2 = 0.805

| factor | F | PR(>F) |
| ------ | - | ------ |
| C(modality, Treatment('MIC')) | 36.91 | 0.0000 *** |
| C(arch, Treatment('CRNN')) | 79.18 | 0.0000 *** |
| C(prior, Treatment('nogeom')) | 0.02 | 0.8977 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')) | 1.11 | 0.3386 |
| C(modality, Treatment('MIC')):C(prior, Treatment('nogeom')) | 1.16 | 0.2875 |
| C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 0.02 | 0.9844 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 0.03 | 0.9670 |

### Within-cell paired contrasts (full - nogeom) on F1
| cell | n | delta mean | t (p_t) | d_z |
| ---- | - | ---------- | ------- | --- |
| MIC+CRNN | 5 | -0.402 ± 0.362 | t=-2.49 (p=0.068) | -1.11 |
| MIC+Xfm | 5 | -0.494 ± 0.806 | t=-1.37 (p=0.242) | -0.61 |
| MIC+Conformer | 5 | -0.474 ± 3.727 | t=-0.28 (p=0.790) | -0.13 |
| FOA+CRNN | 5 | +0.150 ± 1.109 | t=+0.30 (p=0.777) | +0.14 |
| FOA+Xfm | 5 | +0.376 ± 1.307 | t=+0.64 (p=0.555) | +0.29 |
| FOA+Conformer | 5 | +0.550 ± 2.845 | t=+0.43 (p=0.688) | +0.19 |

## SELD
- n_obs = 60, R^2 = 0.651

| factor | F | PR(>F) |
| ------ | - | ------ |
| C(modality, Treatment('MIC')) | 2.28 | 0.1375 |
| C(arch, Treatment('CRNN')) | 40.25 | 0.0000 *** |
| C(prior, Treatment('nogeom')) | 0.00 | 0.9772 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')) | 1.23 | 0.3027 |
| C(modality, Treatment('MIC')):C(prior, Treatment('nogeom')) | 3.13 | 0.0833 . |
| C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 0.02 | 0.9799 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 0.52 | 0.5991 |

### Within-cell paired contrasts (full - nogeom) on SELD
| cell | n | delta mean | t (p_t) | d_z |
| ---- | - | ---------- | ------- | --- |
| MIC+CRNN | 5 | +0.003 ± 0.024 | t=+0.24 (p=0.823) | +0.11 |
| MIC+Xfm | 5 | +0.031 ± 0.050 | t=+1.38 (p=0.238) | +0.62 |
| MIC+Conformer | 5 | +0.041 ± 0.168 | t=+0.55 (p=0.612) | +0.25 |
| FOA+CRNN | 5 | -0.007 ± 0.013 | t=-1.17 (p=0.306) | -0.52 |
| FOA+Xfm | 5 | -0.033 ± 0.054 | t=-1.37 (p=0.242) | -0.61 |
| FOA+Conformer | 5 | -0.033 ± 0.078 | t=-0.94 (p=0.398) | -0.42 |

## RDE
- n_obs = 60, R^2 = 0.163

| factor | F | PR(>F) |
| ------ | - | ------ |
| C(modality, Treatment('MIC')) | 1.34 | 0.2532 |
| C(arch, Treatment('CRNN')) | 1.84 | 0.1696 |
| C(prior, Treatment('nogeom')) | 3.17 | 0.0816 . |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')) | 0.07 | 0.9295 |
| C(modality, Treatment('MIC')):C(prior, Treatment('nogeom')) | 0.39 | 0.5364 |
| C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 0.20 | 0.8164 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 0.10 | 0.9078 |

### Within-cell paired contrasts (full - nogeom) on RDE
| cell | n | delta mean | t (p_t) | d_z |
| ---- | - | ---------- | ------- | --- |
| MIC+CRNN | 5 | -0.016 ± 0.062 | t=-0.58 (p=0.594) | -0.26 |
| MIC+Xfm | 5 | -0.002 ± 0.056 | t=-0.08 (p=0.940) | -0.04 |
| MIC+Conformer | 5 | -0.008 ± 0.048 | t=-0.38 (p=0.726) | -0.17 |
| FOA+CRNN | 5 | -0.024 ± 0.017 | t=-3.21 (p=0.033) | -1.43 |
| FOA+Xfm | 5 | -0.020 ± 0.020 | t=-2.24 (p=0.089) | -1.00 |
| FOA+Conformer | 5 | -0.010 ± 0.028 | t=-0.79 (p=0.473) | -0.35 |

## DE
- n_obs = 60, R^2 = 0.301

| factor | F | PR(>F) |
| ------ | - | ------ |
| C(modality, Treatment('MIC')) | 3.01 | 0.0893 . |
| C(arch, Treatment('CRNN')) | 4.13 | 0.0221 * |
| C(prior, Treatment('nogeom')) | 2.61 | 0.1125 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')) | 1.30 | 0.2822 |
| C(modality, Treatment('MIC')):C(prior, Treatment('nogeom')) | 0.54 | 0.4660 |
| C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 1.77 | 0.1814 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 0.04 | 0.9580 |

### Within-cell paired contrasts (full - nogeom) on DE
| cell | n | delta mean | t (p_t) | d_z |
| ---- | - | ---------- | ------- | --- |
| MIC+CRNN | 5 | -0.084 ± 0.301 | t=-0.62 (p=0.567) | -0.28 |
| MIC+Xfm | 5 | +0.004 ± 0.159 | t=+0.06 (p=0.958) | +0.03 |
| MIC+Conformer | 5 | +0.020 ± 0.098 | t=+0.46 (p=0.671) | +0.20 |
| FOA+CRNN | 5 | -0.110 ± 0.087 | t=-2.81 (p=0.048) | -1.26 |
| FOA+Xfm | 5 | -0.018 ± 0.047 | t=-0.85 (p=0.441) | -0.38 |
| FOA+Conformer | 5 | -0.032 ± 0.036 | t=-2.01 (p=0.115) | -0.90 |
