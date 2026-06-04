# Path C / 2x2 dissociation: GCA geometry-prior effect across (modality, architecture)

Each row reports the **paired contrast** (GCA full vs GCA no_geom) on the matched 
seeds within a (modality, architecture) cell. The contrast isolates the geometry-bias 
contribution from the underlying channel-attention mechanism.

## DOAE_CD (deg) -- the headline metric
| cell | n shared | DOAE GCA full | DOAE no_geom | delta DOAE | t (p_t) | d_z | bootstrap 95% CI |
| ---- | -------- | ------------- | ------------ | ---------- | ------- | --- | ---------------- |
| **MIC + CRNN** | 5 | 45.38 ± 6.38 | 46.26 ± 7.50 | **-0.88** | t=-0.15 (p=0.887) | **-0.07** | [-11.26, +9.27] |
| **FOA + CRNN** | 5 | 40.11 ± 2.20 | 45.14 ± 5.93 | **-5.02** | t=-2.31 (p=0.082) | **-1.03** | [-8.75, -1.30] |
| **MIC + Conformer** | 5 | 45.94 ± 9.15 | 47.77 ± 5.30 | **-1.83** | t=-0.35 (p=0.745) | **-0.16** | [-11.04, +7.38] |
| **FOA + Conformer** | 5 | 48.71 ± 6.82 | 47.83 ± 4.89 | **+0.88** | t=+0.30 (p=0.778) | **+0.14** | [-4.70, +5.44] |
| **MIC + Xfm** | 5 | 47.96 ± 5.19 | 42.26 ± 3.06 | **+5.70** | t=+1.60 (p=0.186) | **+0.71** | [-1.31, +10.91] |
| **FOA + Xfm** | 5 | 42.67 ± 5.14 | 40.10 ± 4.23 | **+2.57** | t=+0.86 (p=0.437) | **+0.39** | [-2.72, +7.86] |

## F1 (%) -- detection
| cell | n shared | F1 GCA full | F1 no_geom | delta F1 | t (p_t) | d_z |
| ---- | -------- | ----------- | ---------- | -------- | ------- | --- |
| MIC + CRNN | 5 | 9.59 ± 0.41 | 9.99 ± 0.54 | -0.40 | t=-2.49 (p=0.068) | -1.11 |
| FOA + CRNN | 5 | 12.79 ± 0.65 | 12.64 ± 0.60 | +0.15 | t=+0.30 (p=0.777) | +0.14 |
| MIC + Conformer | 5 | 4.29 ± 2.78 | 4.76 ± 2.87 | -0.47 | t=-0.28 (p=0.790) | -0.13 |
| FOA + Conformer | 5 | 7.22 ± 1.15 | 6.67 ± 2.24 | +0.55 | t=+0.43 (p=0.688) | +0.19 |
| MIC + Xfm | 5 | 9.13 ± 0.75 | 9.63 ± 0.59 | -0.49 | t=-1.37 (p=0.242) | -0.61 |
| FOA + Xfm | 5 | 11.13 ± 0.50 | 10.75 ± 1.09 | +0.38 | t=+0.64 (p=0.555) | +0.29 |

## SELD score -- joint metric (lower is better)
| cell | n shared | SELD GCA full | SELD no_geom | delta SELD | t (p_t) | d_z |
| ---- | -------- | ------------- | ------------ | ---------- | ------- | --- |
| MIC + CRNN | 5 | 0.547 ± 0.026 | 0.545 ± 0.024 | +0.003 | t=+0.24 (p=0.823) | +0.11 |
| FOA + CRNN | 5 | 0.531 ± 0.002 | 0.538 ± 0.011 | -0.007 | t=-1.17 (p=0.306) | -0.52 |
| MIC + Conformer | 5 | 0.740 ± 0.080 | 0.698 ± 0.111 | +0.041 | t=+0.55 (p=0.612) | +0.25 |
| FOA + Conformer | 5 | 0.651 ± 0.022 | 0.684 ± 0.065 | -0.033 | t=-0.94 (p=0.398) | -0.42 |
| MIC + Xfm | 5 | 0.630 ± 0.055 | 0.599 ± 0.030 | +0.031 | t=+1.38 (p=0.238) | +0.62 |
| FOA + Xfm | 5 | 0.598 ± 0.034 | 0.631 ± 0.071 | -0.033 | t=-1.37 (p=0.242) | -0.61 |
