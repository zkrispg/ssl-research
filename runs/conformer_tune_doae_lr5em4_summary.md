# Conformer tuning doae lr5em4 Summary

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | seeds | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---|---:|---:|---:|---:|
| 171 | 2 | [0, 1] | 0.927 +/- 0.064 | 0.805 +/- 1.138 | 29.860 +/- 1.344 | 0.585 +/- 0.304 |
| 172 | 2 | [0, 1] | 0.796 +/- 0.250 | 4.480 +/- 6.336 | 24.785 +/- 6.343 | 0.585 +/- 0.375 |

## Paired Contrasts

| label | pair | n | delta SELD | delta F20 | delta DOAE | delta RDE | t(DOAE) | p(DOAE) | dz(DOAE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FOA_Conformer_doae_lr5em4 | 171-172 | 2 | +0.131 | -3.675 | +5.075 | +0.000 | +1.436 | 0.387 | +1.015 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | RDE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 171 | 0 | tune_doae_lr5em4_seed0 | complete | 0 | doae | 0.973 | 0.000 | 28.910 | 0.800 | e326b4a7ee06b2ce |
| 172 | 0 | tune_doae_lr5em4_seed0 | complete | 0 | doae | 0.973 | 0.000 | 20.300 | 0.850 | e326b4a7ee06b2ce |
| 171 | 1 | tune_doae_lr5em4_seed1 | complete | 14 | doae | 0.882 | 1.610 | 30.810 | 0.370 | e326b4a7ee06b2ce |
| 172 | 1 | tune_doae_lr5em4_seed1 | complete | 49 | doae | 0.620 | 8.960 | 29.270 | 0.320 | e326b4a7ee06b2ce |
