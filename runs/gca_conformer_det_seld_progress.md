# GCA Conformer Deterministic Run Progress

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---:|---:|---:|---:|
| 161 | 4 | 0.666 | 4.280 | 47.505 | 0.270 |
| 162 | 4 | 0.688 | 3.050 | 47.500 | 0.305 |
| 171 | 4 | 0.647 | 6.100 | 50.975 | 0.273 |
| 172 | 3 | 0.692 | 4.120 | 54.297 | 0.267 |

## Paired Deltas

| pair | n | delta SELD | delta F20 | delta DOAE | delta RDE |
|---|---:|---:|---:|---:|---:|
| 161-162 | 4 | -0.021 | +1.230 | +0.005 | -0.035 |
| 171-172 | 3 | -0.019 | +1.573 | -3.707 | +0.010 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---|
| 161 | 0 | det_gca_seed0 | complete | 58 | seld | 0.604 | 3.570 | 43.740 | e326b4a7ee06b2ce |
| 162 | 0 | det_gca_seed0 | complete | 57 | seld | 0.646 | 4.070 | 41.090 | e326b4a7ee06b2ce |
| 171 | 0 | det_gca_seed0 | complete | 57 | seld | 0.613 | 6.270 | 53.120 | e326b4a7ee06b2ce |
| 172 | 0 | det_gca_seed0 | complete | 58 | seld | 0.682 | 5.580 | 52.930 | e326b4a7ee06b2ce |
| 161 | 1 | det_gca_seed1 | complete | 53 | seld | 0.734 | 1.760 | 44.830 | e326b4a7ee06b2ce |
| 162 | 1 | det_gca_seed1 | complete | 51 | seld | 0.726 | 2.970 | 45.290 | e326b4a7ee06b2ce |
| 171 | 1 | det_gca_seed1 | complete | 55 | seld | 0.645 | 5.700 | 52.700 | e326b4a7ee06b2ce |
| 172 | 1 | det_gca_seed1 | complete | 52 | seld | 0.660 | 4.230 | 59.200 | e326b4a7ee06b2ce |
| 161 | 2 | det_gca_seed2 | complete | 57 | seld | 0.685 | 6.370 | 53.460 | e326b4a7ee06b2ce |
| 162 | 2 | det_gca_seed2 | complete | 55 | seld | 0.637 | 2.440 | 51.660 | e326b4a7ee06b2ce |
| 171 | 2 | det_gca_seed2 | complete | 49 | seld | 0.761 | 5.110 | 45.950 | e326b4a7ee06b2ce |
| 172 | 2 | det_gca_seed2 | complete | 56 | seld | 0.734 | 2.550 | 50.760 | e326b4a7ee06b2ce |
| 161 | 3 | det_gca_seed3 | complete | 56 | seld | 0.641 | 5.420 | 47.990 | e326b4a7ee06b2ce |
| 162 | 3 | det_gca_seed3 | complete | 58 | seld | 0.741 | 2.720 | 51.960 | e326b4a7ee06b2ce |
| 171 | 3 | det_gca_seed3 | complete | 55 | seld | 0.567 | 7.320 | 52.130 | e326b4a7ee06b2ce |
| 172 | 3 | det_gca_seed3 | missing | 56 | seld | NA | NA | NA | e326b4a7ee06b2ce |
| 161 | 4 | det_gca_seed4 | missing | NA | NA | NA | NA | NA | NA |
| 162 | 4 | det_gca_seed4 | missing | NA | NA | NA | NA | NA | NA |
| 171 | 4 | det_gca_seed4 | missing | NA | NA | NA | NA | NA | NA |
| 172 | 4 | det_gca_seed4 | missing | NA | NA | NA | NA | NA | NA |
