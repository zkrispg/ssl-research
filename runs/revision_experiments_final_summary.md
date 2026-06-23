# Revision Experiments Final Summary

Last updated: 2026-06-23

This note consolidates the two post-revision experiment blocks added after
`2932629`.

## Stage 1: GCA extremes to n=10

Output files:

- `runs/gca_extremes_n10_summary.md`
- `runs/gca_extremes_n10_summary.csv`
- `runs/gca_extremes_n10_summary.json`

Completed: `2026-06-20 13:52:48`

Main result:

- `FOA_CRNN (130-131)`: `delta DOAE = +2.275`, `p = 0.027`
- `MIC_Transformer (141-142)`: `delta DOAE = -2.316`, `p = 0.311`

Interpretation:

- GCA is not a universally helpful prior.
- In the FOA CRNN extreme, `full` is significantly worse than `no_geom`.
- In the MIC Transformer extreme, `full` trends better, but the contrast is not
  statistically significant at `n=10`.

## Stage 2: Conformer tuning sweep

Output files:

- `runs/conformer_tune_seld_lr1em3_summary.*`
- `runs/conformer_tune_seld_lr5em4_summary.*`
- `runs/conformer_tune_seld_lr3em4_summary.*`
- `runs/conformer_tune_doae_lr1em3_summary.*`
- `runs/conformer_tune_doae_lr5em4_summary.*`
- `runs/conformer_tune_doae_lr3em4_summary.*`

Completed: `2026-06-23 02:10:30`

Working conclusion:

- `best_metric=seld` is the only sensible primary operating point.
- `best_metric=doae` can reduce angular error, but often collapses `F20` and
  produces much worse overall `SELD`.
- The most balanced Conformer pilot setting is `best_metric=seld, lr=3e-4`.

Recommended Conformer reference point:

| task | setting | SELD | F20 | DOAE | note |
|---|---|---:|---:|---:|---|
| 171 | `seld + lr=3e-4` | 0.587 / 0.578 | 7.39 / 8.60 | 39.82 / 35.78 | stable across 2 seeds |
| 172 | `seld + lr=3e-4` | 0.589 / 0.590 | 7.43 / 9.16 | 40.18 / 38.02 | stable across 2 seeds |

Interpretation:

- Conformer remains a neutral-to-weak-help middle point.
- The tuning sweep does not turn Conformer into a new strong geometry-success
  case.
- `doae` early stopping should not be used as the paper's main result setting.

## Paper-facing summary

The safest manuscript claim after all revision experiments is:

> Geometry-prior effects are architecture- and modality-dependent. The GCA
> effect is not a monotonic gain: some settings are neutral, some weakly
> helpful, and some harmful. Conformer remains a middle point, and aggressive
> DOAE-driven checkpoint selection is not a valid replacement for balanced SELD
> model selection.
