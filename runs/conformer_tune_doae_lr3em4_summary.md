# Conformer tuning doae lr3em4 Summary

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | seeds | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---|---:|---:|---:|---:|
| 171 | 2 | [0, 1] | 0.875 +/- 0.070 | 1.640 +/- 2.192 | 24.515 +/- 3.231 | 0.325 +/- 0.106 |
| 172 | 2 | [0, 1] | 0.877 +/- 0.057 | 1.275 +/- 1.520 | 30.130 +/- 0.113 | 0.280 +/- 0.057 |

## Paired Contrasts

| label | pair | n | delta SELD | delta F20 | delta DOAE | delta RDE | t(DOAE) | p(DOAE) | dz(DOAE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FOA_Conformer_doae_lr3em4 | 171-172 | 2 | -0.003 | +0.365 | -5.615 | +0.045 | -2.374 | 0.254 | -1.679 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | RDE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 171 | 0 | tune_doae_lr3em4_seed0 | complete | 2 | doae | 0.924 | 0.090 | 22.230 | 0.400 | e326b4a7ee06b2ce |
| 172 | 0 | tune_doae_lr3em4_seed0 | complete | 2 | doae | 0.918 | 0.200 | 30.210 | 0.240 | e326b4a7ee06b2ce |
| 171 | 1 | tune_doae_lr3em4_seed1 | complete | 12 | doae | 0.825 | 3.190 | 26.800 | 0.250 | e326b4a7ee06b2ce |
| 172 | 1 | tune_doae_lr3em4_seed1 | complete | 14 | doae | 0.837 | 2.350 | 30.050 | 0.320 | e326b4a7ee06b2ce |
