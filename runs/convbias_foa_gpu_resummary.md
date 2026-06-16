# Convbias FOA Run Summary

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---:|---:|---:|---:|
| 184 | 3 | 0.672 | 7.013 | 51.243 | 0.283 |
| 185 | 3 | 0.668 | 7.080 | 50.930 | 0.320 |
| 186 | 3 | 0.649 | 10.867 | 35.637 | 0.337 |
| 187 | 3 | 0.665 | 10.617 | 38.530 | 0.327 |

## Paired Deltas

| pair | n | delta SELD | delta F20 | delta DOAE | delta RDE |
|---|---:|---:|---:|---:|---:|
| 184-185 | 3 | 0.004 | -0.067 | 0.313 | -0.037 |
| 186-187 | 3 | -0.016 | 0.250 | -2.893 | 0.010 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---|
| 184 | 0 | gpu_seed0 | complete | 54 | NA | 0.719 | 6.970 | 41.860 | NA |
| 185 | 0 | gpu_seed0 | complete | 57 | NA | 0.656 | 7.890 | 50.710 | NA |
| 186 | 0 | gpu_seed0 | complete | 47 | NA | 0.625 | 11.510 | 38.300 | NA |
| 187 | 0 | gpu_seed0 | complete | 39 | NA | 0.661 | 10.020 | 37.100 | NA |
| 184 | 1 | gpu_seed1 | complete | 57 | NA | 0.619 | 7.140 | 58.040 | NA |
| 185 | 1 | gpu_seed1 | complete | 52 | NA | 0.639 | 7.150 | 60.750 | NA |
| 186 | 1 | gpu_seed1 | complete | 51 | NA | 0.573 | 10.630 | 35.950 | NA |
| 187 | 1 | gpu_seed1 | complete | 35 | NA | 0.714 | 9.690 | 37.720 | NA |
| 184 | 2 | gpu_seed2 | complete | 56 | NA | 0.678 | 6.930 | 53.830 | NA |
| 185 | 2 | gpu_seed2 | complete | 59 | NA | 0.710 | 6.200 | 41.330 | NA |
| 186 | 2 | gpu_seed2 | complete | 32 | NA | 0.749 | 10.460 | 32.660 | NA |
| 187 | 2 | gpu_seed2 | complete | 54 | NA | 0.621 | 12.140 | 40.770 | NA |
