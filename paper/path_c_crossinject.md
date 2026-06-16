# Cross-injection check: convbias geometry prior versus GCA

A *second* geometry-injection mechanism (`convbias`: geometry as a learned 
per-filter conv-feature bias) re-measures the geometry-prior effect at a
different injection site. Paired contrast = full - no_geom on matched seeds;
`full` and `no_geom` have identical parameter counts. The GCA column repeats the 
main-paper result for the same cell. Negative DOAE means geometry helps.

## DOAE_CD (deg) -- headline metric
| cell | n | convbias run 1 delta | gpu-logged rerun delta | deterministic SELD rerun delta | GCA delta | reading |
| ---- | - | -------------------- | ---------------------- | ------------------------------- | --------- | ------- |
| **FOA + CRNN** | 3 | **-1.60** | not rerun | not rerun | -5.02 | weak help, same sign as GCA |
| **FOA + Conformer** | 3+3+3 | **+0.21** | **+0.31** | **+7.59** | +0.88 | changes from near-zero to harm under SELD selection |
| **FOA + Transformer** | 3+3+3 | **+9.55** | **-2.89** | **+4.81** | +2.57 | sign/checkpoint sensitive |
| **MIC + Transformer** | 3 | **-3.79** | not rerun | not rerun | +5.70 | sign reversal from GCA |

## F1 (%) and SELD score (paired delta = full - no_geom)
| cell | run-1 delta F1 | run-1 delta SELD | deterministic SELD rerun delta F1 | deterministic SELD rerun delta SELD | reading |
| ---- | -------------- | ---------------- | ---------------------------------- | ------------------------------------ | ------- |
| FOA + CRNN | -0.47 | +0.001 | not rerun | not rerun | direction benefit is not a broad SELD-score gain |
| FOA + Conformer | +0.47 | -0.054 | -0.91 | +0.032 | SELD-selection rerun reverses the aggregate gain |
| FOA + Transformer | -0.67 | -0.015 | -1.11 | +0.003 | small aggregate difference but direction remains checkpoint-sensitive |
| MIC + Transformer | +0.17 | -0.044 | not rerun | not rerun | sign reversal from GCA; injection-dependent |

**Verdict:** convbias is not confirmatory cross-injection evidence. The
first two FOA Conformer runs are near zero, but the strict deterministic rerun
with validation-SELD checkpoint selection moves the same contrast to harm. FOA
Transformer is also sign/checkpoint sensitive, and MIC+Transformer reverses
relative to GCA. The paper should treat convbias as a sensitivity/boundary
condition and keep the primary architecture claim anchored on the GCA factorial
grid and STARSS22 replication.
