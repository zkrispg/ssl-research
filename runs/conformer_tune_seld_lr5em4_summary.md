# Conformer tuning seld lr5em4 Summary

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | seeds | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---|---:|---:|---:|---:|
| 171 | 2 | [0, 1] | 0.569 +/- 0.013 | 8.765 +/- 0.346 | 43.765 +/- 8.549 | 0.265 +/- 0.007 |
| 172 | 2 | [0, 1] | 0.589 +/- 0.007 | 7.900 +/- 0.283 | 39.325 +/- 3.175 | 0.285 +/- 0.007 |

## Paired Contrasts

| label | pair | n | delta SELD | delta F20 | delta DOAE | delta RDE | t(DOAE) | p(DOAE) | dz(DOAE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FOA_Conformer_seld_lr5em4 | 171-172 | 2 | -0.020 | +0.865 | +4.440 | -0.020 | +1.168 | 0.451 | +0.826 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | RDE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 171 | 0 | tune_seld_lr5em4_seed0 | complete | 55 | seld | 0.578 | 8.520 | 37.720 | 0.260 | e326b4a7ee06b2ce |
| 172 | 0 | tune_seld_lr5em4_seed0 | complete | 55 | seld | 0.584 | 8.100 | 37.080 | 0.280 | e326b4a7ee06b2ce |
| 171 | 1 | tune_seld_lr5em4_seed1 | complete | 50 | seld | 0.559 | 9.010 | 49.810 | 0.270 | e326b4a7ee06b2ce |
| 172 | 1 | tune_seld_lr5em4_seed1 | complete | 53 | seld | 0.594 | 7.700 | 41.570 | 0.290 | e326b4a7ee06b2ce |
