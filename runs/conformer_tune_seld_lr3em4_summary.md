# Conformer tuning seld lr3em4 Summary

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | seeds | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---|---:|---:|---:|---:|
| 171 | 2 | [0, 1] | 0.583 +/- 0.006 | 7.995 +/- 0.856 | 37.800 +/- 2.857 | 0.265 +/- 0.007 |
| 172 | 2 | [0, 1] | 0.589 +/- 0.001 | 8.295 +/- 1.223 | 39.100 +/- 1.527 | 0.290 +/- 0.028 |

## Paired Contrasts

| label | pair | n | delta SELD | delta F20 | delta DOAE | delta RDE | t(DOAE) | p(DOAE) | dz(DOAE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FOA_Conformer_seld_lr3em4 | 171-172 | 2 | -0.007 | -0.300 | -1.300 | -0.025 | -1.383 | 0.399 | -0.978 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | RDE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 171 | 0 | tune_seld_lr3em4_seed0 | complete | 55 | seld | 0.587 | 7.390 | 39.820 | 0.260 | e326b4a7ee06b2ce |
| 172 | 0 | tune_seld_lr3em4_seed0 | complete | 55 | seld | 0.589 | 7.430 | 40.180 | 0.270 | e326b4a7ee06b2ce |
| 171 | 1 | tune_seld_lr3em4_seed1 | complete | 58 | seld | 0.578 | 8.600 | 35.780 | 0.270 | e326b4a7ee06b2ce |
| 172 | 1 | tune_seld_lr3em4_seed1 | complete | 56 | seld | 0.590 | 9.160 | 38.020 | 0.310 | e326b4a7ee06b2ce |
