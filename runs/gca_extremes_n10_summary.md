# GCA Extremes n=10 Summary

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | seeds | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---|---:|---:|---:|---:|
| 130 | 10 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 0.505 +/- 0.005 | 11.288 +/- 0.807 | 47.688 +/- 2.076 | 0.258 +/- 0.016 |
| 131 | 10 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 0.503 +/- 0.007 | 11.978 +/- 0.568 | 45.413 +/- 2.801 | 0.262 +/- 0.010 |
| 141 | 10 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 0.565 +/- 0.013 | 7.962 +/- 1.501 | 44.047 +/- 5.235 | 0.278 +/- 0.010 |
| 142 | 10 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 0.557 +/- 0.022 | 7.629 +/- 1.020 | 46.363 +/- 5.320 | 0.275 +/- 0.024 |

## Paired Contrasts

| label | pair | n | delta SELD | delta F20 | delta DOAE | delta RDE | t(DOAE) | p(DOAE) | dz(DOAE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FOA_CRNN | 130-131 | 10 | +0.002 | -0.690 | +2.275 | -0.004 | +2.629 | 0.027 | +0.831 |
| MIC_Transformer | 141-142 | 10 | +0.007 | +0.333 | -2.316 | +0.003 | -1.072 | 0.311 | -0.339 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | RDE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 130 | 0 | det_gca_seed0 | complete | 53 | seld | 0.508 | 11.050 | 49.150 | 0.250 | e326b4a7ee06b2ce |
| 131 | 0 | det_gca_seed0 | complete | 43 | seld | 0.509 | 11.830 | 46.730 | 0.270 | e326b4a7ee06b2ce |
| 141 | 0 | det_gca_seed0 | complete | 51 | seld | 0.555 | 11.050 | 46.190 | 0.300 | e326b4a7ee06b2ce |
| 142 | 0 | det_gca_seed0 | complete | 42 | seld | 0.562 | 6.480 | 43.950 | 0.280 | e326b4a7ee06b2ce |
| 130 | 1 | det_gca_seed1 | complete | 40 | seld | 0.507 | 12.010 | 48.030 | 0.260 | e326b4a7ee06b2ce |
| 131 | 1 | det_gca_seed1 | complete | 40 | seld | 0.496 | 11.400 | 44.510 | 0.240 | e326b4a7ee06b2ce |
| 141 | 1 | det_gca_seed1 | complete | 58 | seld | 0.550 | 7.810 | 37.150 | 0.290 | e326b4a7ee06b2ce |
| 142 | 1 | det_gca_seed1 | complete | 58 | seld | 0.535 | 8.980 | 48.830 | 0.320 | e326b4a7ee06b2ce |
| 130 | 2 | det_gca_seed2 | complete | 53 | seld | 0.506 | 10.190 | 46.810 | 0.240 | e326b4a7ee06b2ce |
| 131 | 2 | det_gca_seed2 | complete | 39 | seld | 0.508 | 12.240 | 48.000 | 0.270 | e326b4a7ee06b2ce |
| 141 | 2 | det_gca_seed2 | complete | 53 | seld | 0.565 | 7.970 | 51.280 | 0.270 | e326b4a7ee06b2ce |
| 142 | 2 | det_gca_seed2 | complete | 54 | seld | 0.566 | 6.790 | 55.220 | 0.230 | e326b4a7ee06b2ce |
| 130 | 3 | det_gca_seed3 | complete | 59 | seld | 0.505 | 12.100 | 46.870 | 0.260 | e326b4a7ee06b2ce |
| 131 | 3 | det_gca_seed3 | complete | 50 | seld | 0.510 | 11.250 | 48.280 | 0.260 | e326b4a7ee06b2ce |
| 141 | 3 | det_gca_seed3 | complete | 54 | seld | 0.546 | 9.450 | 40.490 | 0.280 | e326b4a7ee06b2ce |
| 142 | 3 | det_gca_seed3 | complete | 55 | seld | 0.523 | 6.120 | 45.920 | 0.260 | e326b4a7ee06b2ce |
| 130 | 4 | det_gca_seed4 | complete | 45 | seld | 0.506 | 10.550 | 44.950 | 0.260 | e326b4a7ee06b2ce |
| 131 | 4 | det_gca_seed4 | complete | 45 | seld | 0.498 | 12.080 | 43.310 | 0.260 | e326b4a7ee06b2ce |
| 141 | 4 | det_gca_seed4 | complete | 56 | seld | 0.569 | 6.280 | 47.220 | 0.280 | e326b4a7ee06b2ce |
| 142 | 4 | det_gca_seed4 | complete | 54 | seld | 0.546 | 9.220 | 40.380 | 0.270 | e326b4a7ee06b2ce |
| 130 | 5 | det_gca_seed5 | complete | 45 | seld | 0.491 | 10.830 | 52.040 | 0.290 | e326b4a7ee06b2ce |
| 131 | 5 | det_gca_seed5 | complete | 54 | seld | 0.504 | 12.560 | 46.600 | 0.270 | e326b4a7ee06b2ce |
| 141 | 5 | det_gca_seed5 | complete | 56 | seld | 0.557 | 8.610 | 46.510 | 0.270 | e326b4a7ee06b2ce |
| 142 | 5 | det_gca_seed5 | complete | 48 | seld | 0.583 | 7.640 | 37.490 | 0.270 | e326b4a7ee06b2ce |
| 130 | 6 | det_gca_seed6 | complete | 44 | seld | 0.507 | 11.290 | 49.440 | 0.240 | e326b4a7ee06b2ce |
| 131 | 6 | det_gca_seed6 | complete | 57 | seld | 0.510 | 11.290 | 49.150 | 0.260 | e326b4a7ee06b2ce |
| 141 | 6 | det_gca_seed6 | complete | 49 | seld | 0.590 | 7.500 | 41.730 | 0.270 | e326b4a7ee06b2ce |
| 142 | 6 | det_gca_seed6 | complete | 50 | seld | 0.595 | 7.110 | 44.670 | 0.260 | e326b4a7ee06b2ce |
| 130 | 7 | det_gca_seed7 | complete | 38 | seld | 0.504 | 12.790 | 46.120 | 0.270 | e326b4a7ee06b2ce |
| 131 | 7 | det_gca_seed7 | complete | 51 | seld | 0.496 | 12.670 | 43.840 | 0.250 | e326b4a7ee06b2ce |
| 141 | 7 | det_gca_seed7 | complete | 58 | seld | 0.568 | 6.490 | 48.680 | 0.270 | e326b4a7ee06b2ce |
| 142 | 7 | det_gca_seed7 | complete | 58 | seld | 0.569 | 7.910 | 49.140 | 0.290 | e326b4a7ee06b2ce |
| 130 | 8 | det_gca_seed8 | complete | 49 | seld | 0.505 | 10.660 | 47.550 | 0.240 | e326b4a7ee06b2ce |
| 131 | 8 | det_gca_seed8 | complete | 45 | seld | 0.493 | 12.740 | 40.190 | 0.270 | e326b4a7ee06b2ce |
| 141 | 8 | det_gca_seed8 | complete | 40 | seld | 0.579 | 8.180 | 35.000 | 0.270 | e326b4a7ee06b2ce |
| 142 | 8 | det_gca_seed8 | complete | 57 | seld | 0.558 | 8.180 | 45.480 | 0.280 | e326b4a7ee06b2ce |
| 130 | 9 | det_gca_seed9 | complete | 55 | seld | 0.507 | 11.410 | 45.920 | 0.270 | e326b4a7ee06b2ce |
| 131 | 9 | det_gca_seed9 | complete | 50 | seld | 0.503 | 11.720 | 43.520 | 0.270 | e326b4a7ee06b2ce |
| 141 | 9 | det_gca_seed9 | complete | 48 | seld | 0.566 | 6.280 | 46.220 | 0.280 | e326b4a7ee06b2ce |
| 142 | 9 | det_gca_seed9 | complete | 52 | seld | 0.537 | 7.860 | 52.550 | 0.290 | e326b4a7ee06b2ce |
