# Path C / E (synth): TAU-NIGENS-SSE-2021 zero-shot evaluation

Models trained on STARSS23 dev-train, evaluated on TAU-NIGENS-SSE-2021 dev-test.
NIGENS uses 12 classes; we remap 7 overlapping classes to STARSS taxonomy and
drop events from the other 5 classes (crying baby, crash, barking dog, female
scream, male scream). Distance is meaningless on NIGENS; lad_dist_thresh=inf.

Metrics labeled `kept_*` are averaged over only the 7 mapped STARSS classes;
`agg_*` are the standard 13-class average.

## Per-cell mean +/- std (kept-class subset)

| Cell                       | n  | F1 (%)            | LE (deg)         | SELD             |
| -------------------------- | -- | ----------------- | ---------------- | ---------------- |
| 100_foa_no_gca             | 4  | 1.89 ± 0.30     | 58.20 ± 2.37    | 0.674 ± 0.017  |
| 110_gca_full               | 5  | 1.92 ± 0.43     | 60.21 ± 9.71    | 0.720 ± 0.025  |
| 111_gca_nogeom             | 5  | 2.09 ± 0.18     | 59.99 ± 7.97    | 0.683 ± 0.012  |
| 112_no_gca                 | 5  | 2.07 ± 0.42     | 60.72 ± 9.60    | 0.700 ± 0.011  |
| 130_foa_gca_full           | 3  | 1.96 ± 0.53     | 59.41 ± 7.24    | 0.660 ± 0.010  |
| 131_foa_gca_nogeom         | 3  | 1.81 ± 0.52     | 61.55 ± 4.86    | 0.680 ± 0.015  |
| 140_xfm_no_gca             | 3  | 1.43 ± 0.15     | 64.16 ± 2.02    | 0.684 ± 0.005  |
| 141_xfm_gca_full           | 3  | 0.96 ± 0.38     | 65.12 ± 4.53    | 0.726 ± 0.051  |
| 142_xfm_gca_nogeom         | 3  | 1.24 ± 0.09     | 64.00 ± 2.00    | 0.681 ± 0.016  |
| 150_xfm_foa_no_gca         | 3  | 0.98 ± 0.24     | 51.74 ± 8.61    | 0.757 ± 0.044  |
| 151_xfm_foa_gca_full       | 3  | 1.12 ± 0.28     | 58.59 ± 3.14    | 0.681 ± 0.010  |
| 152_xfm_foa_gca_nogeom     | 3  | 1.23 ± 0.14     | 57.33 ± 6.71    | 0.724 ± 0.042  |

## Paired contrasts (synth, kept-class subset)

### 130_foa_gca_full__vs__131_foa_gca_nogeom
_FOA+CRNN: GCA full vs no_geom (synth)_

| Metric    | n | delta mean | t (p) | d_z |
| --------- | - | ---------- | ----- | --- |
| kept_F1   | 3 | +0.001 ± 0.003 | t=+0.89 (p=0.467) | +0.51 |
| kept_LE   | 3 | -2.139 ± 3.611 | t=-1.03 (p=0.413) | -0.59 |
| kept_SELD | 3 | -0.020 ± 0.011 | t=-3.04 (p=0.093) | -1.76 |

### 130_foa_gca_full__vs__100_foa_no_gca
_FOA+CRNN: GCA full vs no GCA (synth)_

| Metric    | n | delta mean | t (p) | d_z |
| --------- | - | ---------- | ----- | --- |
| kept_F1   | 3 | -0.001 ± 0.004 | t=-0.23 (p=0.843) | -0.13 |
| kept_LE   | 3 | +0.307 ± 8.725 | t=+0.06 (p=0.957) | +0.04 |
| kept_SELD | 3 | -0.017 ± 0.027 | t=-1.11 (p=0.384) | -0.64 |

### 110_gca_full__vs__111_gca_nogeom
_MIC+CRNN: GCA full vs no_geom (synth)_

| Metric    | n | delta mean | t (p) | d_z |
| --------- | - | ---------- | ----- | --- |
| kept_F1   | 5 | -0.002 ± 0.003 | t=-1.08 (p=0.341) | -0.48 |
| kept_LE   | 5 | +0.221 ± 15.312 | t=+0.03 (p=0.976) | +0.01 |
| kept_SELD | 5 | +0.037 ± 0.022 | t=+3.78 (p=0.019) | +1.69 |

### 110_gca_full__vs__112_no_gca
_MIC+CRNN: GCA full vs no GCA (synth)_

| Metric    | n | delta mean | t (p) | d_z |
| --------- | - | ---------- | ----- | --- |
| kept_F1   | 5 | -0.001 ± 0.006 | t=-0.54 (p=0.615) | -0.24 |
| kept_LE   | 5 | -0.512 ± 14.739 | t=-0.08 (p=0.942) | -0.03 |
| kept_SELD | 5 | +0.020 ± 0.034 | t=+1.36 (p=0.246) | +0.61 |

### 141_xfm_gca_full__vs__142_xfm_gca_nogeom
_MIC+Xfm: GCA full vs no_geom (synth)_

| Metric    | n | delta mean | t (p) | d_z |
| --------- | - | ---------- | ----- | --- |
| kept_F1   | 3 | -0.003 ± 0.003 | t=-1.41 (p=0.295) | -0.81 |
| kept_LE   | 3 | +1.119 ± 6.439 | t=+0.30 (p=0.792) | +0.17 |
| kept_SELD | 3 | +0.045 ± 0.039 | t=+2.00 (p=0.184) | +1.15 |

### 141_xfm_gca_full__vs__140_xfm_no_gca
_MIC+Xfm: GCA full vs no GCA (synth)_

| Metric    | n | delta mean | t (p) | d_z |
| --------- | - | ---------- | ----- | --- |
| kept_F1   | 3 | -0.005 ± 0.003 | t=-2.95 (p=0.098) | -1.70 |
| kept_LE   | 3 | +0.956 ± 2.687 | t=+0.62 (p=0.600) | +0.36 |
| kept_SELD | 3 | +0.042 ± 0.046 | t=+1.58 (p=0.255) | +0.91 |

### 151_xfm_foa_gca_full__vs__152_xfm_foa_gca_nogeom
_FOA+Xfm: GCA full vs no_geom (synth)_

| Metric    | n | delta mean | t (p) | d_z |
| --------- | - | ---------- | ----- | --- |
| kept_F1   | 3 | -0.001 ± 0.003 | t=-0.71 (p=0.553) | -0.41 |
| kept_LE   | 3 | +1.268 ± 8.540 | t=+0.26 (p=0.821) | +0.15 |
| kept_SELD | 3 | -0.043 ± 0.048 | t=-1.55 (p=0.261) | -0.89 |

### 151_xfm_foa_gca_full__vs__150_xfm_foa_no_gca
_FOA+Xfm: GCA full vs no GCA (synth)_

| Metric    | n | delta mean | t (p) | d_z |
| --------- | - | ---------- | ----- | --- |
| kept_F1   | 3 | +0.001 ± 0.004 | t=+0.61 (p=0.601) | +0.35 |
| kept_LE   | 3 | +6.856 ± 11.754 | t=+1.01 (p=0.419) | +0.58 |
| kept_SELD | 3 | -0.077 ± 0.039 | t=-3.40 (p=0.077) | -1.96 |
