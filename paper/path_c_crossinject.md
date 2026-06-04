# Cross-injection robustness: convbias geometry prior on the two extreme cells

A *second* geometry-injection mechanism (`convbias`: geometry as a learned 
per-filter conv-feature bias) re-measured on the recurrent (helps) and 
pure-attention (harms) cells. Paired contrast = full - no_geom on matched seeds; 
`full` and `no_geom` have identical parameter counts. The GCA column repeats the 
main-paper result for the same cell.

## DOAE_CD (deg) -- headline metric (negative = geometry helps)
| cell | n | convbias full | convbias no_geom | delta DOAE (convbias) | t (p) | d_z | 95% CI | GCA delta | sign matches? |
| ---- | - | ------------- | ---------------- | --------------------- | ----- | --- | ------ | --------- | ------------- |
| **FOA + CRNN** | 2 | 35.88 ± 5.01 | 37.55 ± 2.62 | **-1.67** | t=-0.99 (p=0.502) | **-0.70** | [-3.36, +0.01] | -5.02 | **yes** |
| **MIC + Transformer** | 2 | 42.94 ± 4.02 | 47.47 ± 5.73 | **-4.52** | t=-0.66 (p=0.630) | **-0.46** | [-11.42, +2.37] | +5.70 | **NO** |

## F1 (%) and SELD score (paired delta = full - no_geom)
| cell | delta F1 | t (p) | delta SELD | t (p) |
| ---- | -------- | ----- | ---------- | ----- |
| FOA + CRNN | -0.61 | t=-3.05 (p=0.202) | -0.004 | t=-1.00 (p=0.500) |
| MIC + Transformer | +0.12 | t=+4.60 (p=0.136) | -0.017 | t=-1.89 (p=0.310) |

**Verdict:** the ordering does NOT fully replicate under convbias; interpret the geometry-prior effect as injection-dependent.