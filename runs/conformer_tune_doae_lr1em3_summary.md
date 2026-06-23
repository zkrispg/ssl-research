# Conformer tuning doae lr1em3 Summary

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | seeds | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---|---:|---:|---:|---:|
| 171 | 2 | [0, 1] | 0.946 +/- 0.015 | 0.235 +/- 0.205 | 28.925 +/- 0.969 | 0.380 +/- 0.269 |
| 172 | 2 | [0, 1] | 0.885 +/- 0.110 | 0.805 +/- 0.983 | 30.260 +/- 8.598 | 0.380 +/- 0.057 |

## Paired Contrasts

| label | pair | n | delta SELD | delta F20 | delta DOAE | delta RDE | t(DOAE) | p(DOAE) | dz(DOAE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FOA_Conformer_doae_lr1em3 | 171-172 | 2 | +0.060 | -0.570 | -1.335 | -0.000 | -0.247 | 0.846 | -0.175 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | RDE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 171 | 0 | tune_doae_lr1em3_seed0 | complete | 3 | doae | 0.956 | 0.380 | 29.610 | 0.190 | e326b4a7ee06b2ce |
| 172 | 0 | tune_doae_lr1em3_seed0 | complete | 41 | doae | 0.808 | 1.500 | 36.340 | 0.340 | e326b4a7ee06b2ce |
| 171 | 1 | tune_doae_lr1em3_seed1 | complete | 13 | doae | 0.935 | 0.090 | 28.240 | 0.570 | e326b4a7ee06b2ce |
| 172 | 1 | tune_doae_lr1em3_seed1 | complete | 5 | doae | 0.963 | 0.110 | 24.180 | 0.420 | e326b4a7ee06b2ce |
