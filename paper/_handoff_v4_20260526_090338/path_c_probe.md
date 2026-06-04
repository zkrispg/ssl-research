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

## Paired contrasts
### 110_gca_full__vs__112_no_gca
- _GCA full vs no-GCA control (probing)_
- n=5, mean delta = -0.14 +/- 0.87 deg
- t = -0.36 (p = 0.7392)
- Cohen's d_z = -0.16

### 110_gca_full__vs__111_gca_nogeom
- _GCA full vs no_geom (geometry contribution, probing)_
- n=5, mean delta = +0.12 +/- 1.02 deg
- t = +0.27 (p = 0.8028)
- Cohen's d_z = +0.12

### 111_gca_nogeom__vs__112_no_gca
- _no_geom GCA vs no-GCA (channel-attn alone, probing)_
- n=5, mean delta = -0.26 +/- 0.62 deg
- t = -0.94 (p = 0.3991)
- Cohen's d_z = -0.42

### 113_vanilla_se__vs__112_no_gca
- _Vanilla SE vs no-GCA (probing)_
- n=5, mean delta = +0.51 +/- 0.56 deg
- t = +2.01 (p = 0.1147)
- Cohen's d_z = +0.90

### 113_vanilla_se__vs__111_gca_nogeom
- _Vanilla SE vs GCA no_geom (probing)_
- n=5, mean delta = +0.77 +/- 1.07 deg
- t = +1.60 (p = 0.1839)
- Cohen's d_z = +0.72

### 110_gca_full__vs__113_vanilla_se
- _GCA full vs Vanilla SE (probing)_
- n=5, mean delta = -0.65 +/- 0.64 deg
- t = -2.26 (p = 0.0867)
- Cohen's d_z = -1.01
