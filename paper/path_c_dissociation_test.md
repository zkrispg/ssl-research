# Path C / P1-3: direction-vs-distance double-dissociation test

Each cell is a within-seed paired SECOND DIFFERENCE on the geometry effect
delta = full - nogeom. The ARCH contrast (delta_CRNN - delta_Xfm, pooled over
modality) tests architecture x prior; the MODALITY contrast
(delta_FOA - delta_MIC, pooled over the three backbones) tests modality x prior.
A clean double dissociation predicts: DIRECTION significant under the ARCH
contrast only; DISTANCE significant under the MODALITY contrast only.

## Paired second-difference interactions

| metric | ARCH contrast (CRNN-Xfm) | MODALITY contrast (FOA-MIC) |
| ------ | ------------------------ | --------------------------- |
| DOAE_CD (direction) | mean=-7.087, t=-2.59, p=0.029, d_z=-0.82 (n=10) | mean=-1.520, t=-0.46, p=0.654, d_z=-0.12 (n=15) |
| RDE (distance) | mean=-0.009, t=-0.68, p=0.515, d_z=-0.21 (n=10) | mean=-0.009, t=-0.62, p=0.542, d_z=-0.16 (n=15) |

## Type-II factorial interaction F/p (3-level design, n_obs=60)

| metric | arch x prior | modality x prior |
| ------ | ------------ | ---------------- |
| DOAE_CD (direction) | F=1.94, p=0.155 | F=0.26, p=0.612 |
| RDE (distance) | F=0.20, p=0.816 | F=0.39, p=0.536 |
