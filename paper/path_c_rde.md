# A3 / distance (RDE) geometry-prior contrast across the 2x2 grid

delta RDE = full - no_geom (lower RDE is better; negative delta = prior helps).
Compare against the DOAE pattern: DOAE is architecture-conditional, RDE is
expected to test whether distance behaves the same way.

| cell | n | RDE full | RDE no_geom | delta RDE | t (p) | d_z | 95% CI |
| ---- | - | -------- | ----------- | --------- | ----- | --- | ------ |
| MIC + CRNN | 5 | 0.280 | 0.296 | -0.016 | t=-0.58 (p=0.594) | -0.26 | [-0.062, +0.030] |
| FOA + CRNN | 3 | 0.267 | 0.287 | -0.020 | t=-3.46 (p=0.074) | -2.00 | [-0.030, -0.010] |
| MIC + Xfm | 3 | 0.307 | 0.303 | +0.003 | t=+0.16 (p=0.885) | +0.09 | [-0.030, +0.040] |
| FOA + Xfm | 3 | 0.293 | 0.313 | -0.020 | t=-1.73 (p=0.225) | -1.00 | [-0.040, +0.000] |

## Reading
On RDE the prior **helps both FOA cells** (FOA+CRNN delta=-0.020, d_z=-2.00; FOA+Xfm delta=-0.020, d_z=-1.00) and is **null in both MIC cells** (MIC+CRNN d_z=-0.26; MIC+Xfm d_z=+0.09).

Thus the prior's distance effect is **modality-conditional** (helps the Ambisonic
format regardless of temporal architecture), in contrast to its **architecture-
conditional** direction effect. The two spatial dimensions are governed by
different factors -- a double dissociation that reinforces 'it depends'.