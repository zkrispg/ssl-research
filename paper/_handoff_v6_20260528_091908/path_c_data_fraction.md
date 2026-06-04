# Path C / data-fraction sweep: when does the geometry prior help?

Compares 110 (GCA full, geometry prior) vs 112 (no GCA, matched control)
trained on subsets of STARSS23 dev-train at three fractions:
100% (existing Stage 3), 50% (tasks 120/121), 25% (tasks 122/123).

Hypothesis test: at low data, the geometry prior should regularize and
*help* the model; at high data, the prior should over-constrain and *hurt*
the model. A positive *interaction* between data-fraction and prior would
manifest as a non-zero slope in delta = (GCA - no-GCA) vs fraction.

## Per-fraction summary
| fraction | n pairs | F1 GCA (%) | F1 no-GCA (%) | delta F1 (pp) | t | p_t | d_z |
| -------- | ------- | ---------- | ------------- | ------------- | - | --- | --- |
| 25% | 5 | 6.92 +/- 0.42 | 7.16 +/- 0.42 | -0.25 | -1.26 | 0.2761 | -0.56 |
| 50% | 5 | 8.50 +/- 0.71 | 9.22 +/- 0.43 | -0.72 | -2.20 | 0.0928 | -0.98 |
| 100% | 5 | 9.59 +/- 0.41 | 9.72 +/- 0.46 | -0.13 | -0.84 | 0.4480 | -0.38 |

## Per-fraction SELD score (lower is better)
| fraction | n pairs | SELD GCA | SELD no-GCA | delta SELD | t | p_t | d_z |
| -------- | ------- | -------- | ----------- | ---------- | - | --- | --- |
| 25% | 5 | 0.655 +/- 0.027 | 0.652 +/- 0.024 | +0.003 | +0.20 | 0.8477 | +0.09 |
| 50% | 5 | 0.574 +/- 0.045 | 0.544 +/- 0.023 | +0.030 | +1.03 | 0.3627 | +0.46 |
| 100% | 5 | 0.547 +/- 0.026 | 0.579 +/- 0.031 | -0.031 | -1.33 | 0.2555 | -0.59 |

## Interpretation
- **Positive delta F1** at fraction f means the geometry prior helps at f.
- **Negative delta F1** means the geometry prior hurts at f.
- A monotonically increasing delta as fraction decreases would support the
  'prior helps when data is scarce' hypothesis (expected behavior of an
  inductive bias acting as regularizer).
