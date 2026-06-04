# Cross-injection robustness: convbias geometry prior on the two extreme cells

A *second* geometry-injection mechanism (`convbias`: geometry as a learned 
per-filter conv-feature bias) re-measured on the recurrent (helps) and 
pure-attention (harms) cells. Paired contrast = full - no_geom on matched seeds; 
`full` and `no_geom` have identical parameter counts. The GCA column repeats the 
main-paper result for the same cell.

## DOAE_CD (deg) -- headline metric (negative = geometry helps)
| cell | n | convbias full | convbias no_geom | delta DOAE (convbias) | t (p) | d_z | 95% CI | GCA delta | sign matches? |
| ---- | - | ------------- | ---------------- | --------------------- | ----- | --- | ------ | --------- | ------------- |
| **FOA + CRNN** | 3 | 37.38 ± 4.39 | 38.98 ± 3.08 | **-1.60** | t=-1.64 (p=0.243) | **-0.95** | [-3.36, +0.01] | -5.02 | **yes** |
| **MIC + Transformer** | 3 | 41.40 ± 3.90 | 45.19 ± 5.65 | **-3.79** | t=-0.94 (p=0.448) | **-0.54** | [-11.42, +2.37] | +5.70 | **NO** |

## F1 (%) and SELD score (paired delta = full - no_geom)
| cell | delta F1 | t (p) | delta SELD | t (p) |
| ---- | -------- | ----- | ---------- | ----- |
| FOA + CRNN | -0.47 | t=-2.59 (p=0.122) | +0.001 | t=+0.18 (p=0.873) |
| MIC + Transformer | +0.17 | t=+2.99 (p=0.096) | -0.044 | t=-1.61 (p=0.249) |

**Verdict:** the ordering does NOT fully replicate under convbias; interpret the geometry-prior effect as injection-dependent.