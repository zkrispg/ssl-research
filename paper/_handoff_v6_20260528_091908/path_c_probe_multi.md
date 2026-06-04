# Path C / multi-source linear probing

Probe target: 2 sources -> 8-d sin/cos vector (sorted by GT azimuth).
Eval: Hungarian-matched mean angular error (deg).

## Per-cell
| cell | n | MAE mean (deg) | MAE std (deg) |
| ---- | - | -------------- | ------------- |
| 110_gca_full | 5 | 30.32 | 0.34 |
| 111_gca_nogeom | 5 | 30.39 | 0.43 |
| 112_no_gca | 5 | 30.55 | 0.11 |
| 113_vanilla_se | 5 | 30.58 | 0.17 |

## Paired contrasts
### 110_gca_full__vs__112_no_gca
- _GCA full vs no-GCA (multi-src probing)_
- n=5, mean delta = -0.23 +/- 0.29 deg
- t = -1.81 (p_t = 0.1437, p_w = 0.3125)
- Cohen's d_z = -0.81

### 110_gca_full__vs__111_gca_nogeom
- _GCA full vs no_geom (multi-src probing)_
- n=5, mean delta = -0.08 +/- 0.60 deg
- t = -0.29 (p_t = 0.7892, p_w = 0.8125)
- Cohen's d_z = -0.13

### 111_gca_nogeom__vs__112_no_gca
- _no_geom GCA vs no-GCA (multi-src probing)_
- n=5, mean delta = -0.15 +/- 0.48 deg
- t = -0.73 (p_t = 0.5065, p_w = 0.6250)
- Cohen's d_z = -0.33

### 113_vanilla_se__vs__112_no_gca
- _Vanilla SE vs no-GCA (multi-src probing)_
- n=5, mean delta = +0.03 +/- 0.24 deg
- t = +0.33 (p_t = 0.7597, p_w = 0.8125)
- Cohen's d_z = +0.15

### 113_vanilla_se__vs__111_gca_nogeom
- _Vanilla SE vs GCA no_geom (multi-src probing)_
- n=5, mean delta = +0.19 +/- 0.39 deg
- t = +1.10 (p_t = 0.3347, p_w = 0.4375)
- Cohen's d_z = +0.49

### 110_gca_full__vs__113_vanilla_se
- _GCA full vs Vanilla SE (multi-src probing)_
- n=5, mean delta = -0.27 +/- 0.46 deg
- t = -1.30 (p_t = 0.2622, p_w = 0.3125)
- Cohen's d_z = -0.58
