# Conformer tuning seld lr1em3 Summary

Positive delta means full-geometry task minus matched no-geometry control.
For DOAE/SELD/RDE lower is better; for F20 higher is better.

## Per Task

| task | n | seeds | SELD mean | F20 mean | DOAE mean | RDE mean |
|---|---:|---|---:|---:|---:|---:|
| 171 | 2 | [0, 1] | 0.629 +/- 0.023 | 5.985 +/- 0.403 | 52.910 +/- 0.297 | 0.265 +/- 0.021 |
| 172 | 2 | [0, 1] | 0.671 +/- 0.016 | 4.905 +/- 0.955 | 56.065 +/- 4.434 | 0.255 +/- 0.007 |

## Paired Contrasts

| label | pair | n | delta SELD | delta F20 | delta DOAE | delta RDE | t(DOAE) | p(DOAE) | dz(DOAE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FOA_Conformer_seld_lr1em3 | 171-172 | 2 | -0.042 | +1.080 | -3.155 | +0.010 | -0.943 | 0.519 | -0.667 |

## Cells

| task | seed | job | status | best epoch | best metric | SELD | F20 | DOAE | RDE | digest |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 171 | 0 | tune_seld_lr1em3_seed0 | complete | 57 | seld | 0.613 | 6.270 | 53.120 | 0.280 | e326b4a7ee06b2ce |
| 172 | 0 | tune_seld_lr1em3_seed0 | complete | 58 | seld | 0.682 | 5.580 | 52.930 | 0.250 | e326b4a7ee06b2ce |
| 171 | 1 | tune_seld_lr1em3_seed1 | complete | 55 | seld | 0.645 | 5.700 | 52.700 | 0.250 | e326b4a7ee06b2ce |
| 172 | 1 | tune_seld_lr1em3_seed1 | complete | 52 | seld | 0.660 | 4.230 | 59.200 | 0.260 | e326b4a7ee06b2ce |
