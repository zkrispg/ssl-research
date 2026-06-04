# Path C linear probing -- post-conv (B, C, T_label, F_red) -> (sin az, cos az, sin el, cos el)

Probe: Ridge(alpha=1) on standardized pooled features (mean+max over F_red).
5-fold CV, folds split by FILE. Lower angular MAE = representation encodes location more linearly.

## Per-cell
| cell | n | MAE mean (deg) | MAE std (deg) |
| ---- | - | -------------- | ------------- |
| 110_gca_full | 5 | 28.50 | 0.50 |
| 111_gca_nogeom | 5 | 28.37 | 0.81 |
| 112_no_gca | 5 | 28.64 | 0.48 |
| 113_vanilla_se | 5 | 29.14 | 0.47 |
| 100_foa_no_gca | 5 | 20.89 | 0.62 |
| 130_foa_gca_full | 5 | 20.62 | 0.61 |
| 131_foa_gca_nogeom | 5 | 20.89 | 0.67 |
| 140_xfm_no_gca | 5 | 26.32 | 1.00 |
| 141_xfm_gca_full | 5 | 26.11 | 1.10 |
| 142_xfm_gca_nogeom | 5 | 26.27 | 0.73 |
| 150_xfm_foa_no_gca | 5 | 17.96 | 0.50 |
| 151_xfm_foa_gca_full | 5 | 17.93 | 0.49 |
| 152_xfm_foa_gca_nogeom | 5 | 17.89 | 0.65 |
| 160_conf_no_gca | 5 | 28.42 | 2.90 |
| 161_conf_gca_full | 5 | 29.88 | 2.68 |
| 162_conf_gca_nogeom | 5 | 28.35 | 1.95 |
| 170_conf_foa_no_gca | 5 | 19.18 | 1.00 |
| 171_conf_foa_gca_full | 5 | 18.69 | 0.58 |
| 172_conf_foa_gca_nogeom | 5 | 19.28 | 1.28 |

## Paired contrasts
### 110_gca_full__vs__112_no_gca
- _MIC+CRNN: GCA full vs no-GCA (probing)_
- n=5, mean delta = -0.14 +/- 0.87 deg
- t = -0.36 (p = 0.7392)
- Cohen's d_z = -0.16

### 110_gca_full__vs__111_gca_nogeom
- _MIC+CRNN: GCA full vs no_geom (geometry contribution, probing)_
- n=5, mean delta = +0.12 +/- 1.02 deg
- t = +0.27 (p = 0.8028)
- Cohen's d_z = +0.12

### 111_gca_nogeom__vs__112_no_gca
- _MIC+CRNN: GCA no_geom vs no-GCA (probing)_
- n=5, mean delta = -0.26 +/- 0.62 deg
- t = -0.94 (p = 0.3991)
- Cohen's d_z = -0.42

### 113_vanilla_se__vs__112_no_gca
- _MIC+CRNN: Vanilla SE vs no-GCA (probing)_
- n=5, mean delta = +0.51 +/- 0.56 deg
- t = +2.01 (p = 0.1147)
- Cohen's d_z = +0.90

### 130_foa_gca_full__vs__131_foa_gca_nogeom
- _FOA+CRNN: GCA full vs no_geom (probing)_
- n=5, mean delta = -0.27 +/- 0.59 deg
- t = -1.04 (p = 0.3563)
- Cohen's d_z = -0.47

### 130_foa_gca_full__vs__100_foa_no_gca
- _FOA+CRNN: GCA full vs no GCA (probing)_
- n=5, mean delta = -0.27 +/- 0.50 deg
- t = -1.21 (p = 0.2938)
- Cohen's d_z = -0.54

### 141_xfm_gca_full__vs__142_xfm_gca_nogeom
- _MIC+Xfm: GCA full vs no_geom (probing)_
- n=5, mean delta = -0.16 +/- 1.43 deg
- t = -0.25 (p = 0.8123)
- Cohen's d_z = -0.11

### 141_xfm_gca_full__vs__140_xfm_no_gca
- _MIC+Xfm: GCA full vs no GCA (probing)_
- n=5, mean delta = -0.21 +/- 1.85 deg
- t = -0.25 (p = 0.8131)
- Cohen's d_z = -0.11

### 151_xfm_foa_gca_full__vs__152_xfm_foa_gca_nogeom
- _FOA+Xfm: GCA full vs no_geom (probing)_
- n=5, mean delta = +0.05 +/- 0.86 deg
- t = +0.12 (p = 0.9094)
- Cohen's d_z = +0.05

### 151_xfm_foa_gca_full__vs__150_xfm_foa_no_gca
- _FOA+Xfm: GCA full vs no GCA (probing)_
- n=5, mean delta = -0.03 +/- 0.26 deg
- t = -0.27 (p = 0.7995)
- Cohen's d_z = -0.12

### 161_conf_gca_full__vs__162_conf_gca_nogeom
- _MIC+Conf: GCA full vs no_geom (probing)_
- n=5, mean delta = +1.53 +/- 3.88 deg
- t = +0.88 (p = 0.4288)
- Cohen's d_z = +0.39

### 161_conf_gca_full__vs__160_conf_no_gca
- _MIC+Conf: GCA full vs no GCA (probing)_
- n=5, mean delta = +1.47 +/- 2.86 deg
- t = +1.15 (p = 0.3158)
- Cohen's d_z = +0.51

### 171_conf_foa_gca_full__vs__172_conf_foa_gca_nogeom
- _FOA+Conf: GCA full vs no_geom (probing)_
- n=5, mean delta = -0.59 +/- 1.13 deg
- t = -1.18 (p = 0.3047)
- Cohen's d_z = -0.53

### 171_conf_foa_gca_full__vs__170_conf_foa_no_gca
- _FOA+Conf: GCA full vs no GCA (probing)_
- n=5, mean delta = -0.49 +/- 0.84 deg
- t = -1.30 (p = 0.2649)
- Cohen's d_z = -0.58
