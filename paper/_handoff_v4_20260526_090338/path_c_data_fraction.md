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
| 25% | 3 | 6.80 +/- 0.51 | 7.13 +/- 0.46 | -0.33 | -4.25 | 0.0512 | -2.45 |
| 50% | 3 | 8.91 +/- 0.39 | 9.13 +/- 0.47 | -0.23 | -1.05 | 0.4043 | -0.61 |
| 100% | 5 | 9.59 +/- 0.41 | 9.72 +/- 0.46 | -0.13 | -0.84 | 0.4480 | -0.38 |

## Per-fraction SELD score (lower is better)
| fraction | n pairs | SELD GCA | SELD no-GCA | delta SELD | t | p_t | d_z |
| -------- | ------- | -------- | ----------- | ---------- | - | --- | --- |
| 25% | 3 | 0.646 +/- 0.034 | 0.659 +/- 0.022 | -0.013 | -1.34 | 0.3126 | -0.77 |
| 50% | 3 | 0.580 +/- 0.007 | 0.538 +/- 0.019 | +0.042 | +3.08 | 0.0914 | +1.78 |
| 100% | 5 | 0.547 +/- 0.026 | 0.579 +/- 0.031 | -0.031 | -1.33 | 0.2555 | -0.59 |

## Interpretation
- **Positive delta F1** at fraction f means the geometry prior helps at f.
- **Negative delta F1** means the geometry prior hurts at f.
- A monotonically increasing delta as fraction decreases would support the
  'prior helps when data is scarce' hypothesis (expected behavior of an
  inductive bias acting as regularizer).
