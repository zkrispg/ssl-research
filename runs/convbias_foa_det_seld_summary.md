# Convbias FOA Run Summary

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---:|---:|---:|---:|
| 184 | 3 | 0.735 | 2.897 | 58.977 | 0.263 |
| 185 | 3 | 0.703 | 3.807 | 51.390 | 0.270 |
| 186 | 3 | 0.572 | 8.953 | 43.800 | 0.300 |
| 187 | 3 | 0.568 | 10.067 | 38.990 | 0.280 |

## Paired Deltas

| pair | n | delta SELD | delta F20 | delta DOAE | delta RDE |
|---|---:|---:|---:|---:|---:|
| 184-185 | 3 | 0.032 | -0.910 | 7.587 | -0.007 |
| 186-187 | 3 | 0.003 | -1.113 | 4.810 | 0.020 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---|
| 184 | 0 | det_seld_seed0 | complete | 56 | seld | 0.785 | 0.780 | 60.870 | e326b4a7ee06b2ce |
| 185 | 0 | det_seld_seed0 | complete | 56 | seld | 0.728 | 4.130 | 49.340 | e326b4a7ee06b2ce |
| 186 | 0 | det_seld_seed0 | complete | 57 | seld | 0.580 | 9.320 | 38.530 | e326b4a7ee06b2ce |
| 187 | 0 | det_seld_seed0 | complete | 47 | seld | 0.578 | 11.370 | 41.970 | e326b4a7ee06b2ce |
| 184 | 1 | det_seld_seed1 | complete | 56 | seld | 0.631 | 6.680 | 46.940 | e326b4a7ee06b2ce |
| 185 | 1 | det_seld_seed1 | complete | 55 | seld | 0.607 | 6.110 | 52.390 | e326b4a7ee06b2ce |
| 186 | 1 | det_seld_seed1 | complete | 54 | seld | 0.570 | 7.510 | 48.000 | e326b4a7ee06b2ce |
| 187 | 1 | det_seld_seed1 | complete | 54 | seld | 0.555 | 8.460 | 45.930 | e326b4a7ee06b2ce |
| 184 | 2 | det_seld_seed2 | complete | 57 | seld | 0.789 | 1.230 | 69.120 | e326b4a7ee06b2ce |
| 185 | 2 | det_seld_seed2 | complete | 57 | seld | 0.773 | 1.180 | 52.440 | e326b4a7ee06b2ce |
| 186 | 2 | det_seld_seed2 | complete | 54 | seld | 0.565 | 10.030 | 44.870 | e326b4a7ee06b2ce |
| 187 | 2 | det_seld_seed2 | complete | 59 | seld | 0.572 | 10.370 | 29.070 | e326b4a7ee06b2ce |
