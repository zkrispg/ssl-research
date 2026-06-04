# Path C / P0-1: trend test + mixed-model robustness (DV = DOAE_CD / LE)

Confirmatory analyses for the *graded* geometry-prior effect. Headline metric
is class-dependent localization error (LE = DOAE_CD; lower = better; the
geometry effect is delta = full - nogeom, negative = prior helps).

## 1. Focal 2-level factorial OLS (CRNN vs Transformer)
- n_obs = 40, R^2 = 0.252

| term | F | p |
| ---- | - | - |
| C(modality, Treatment('MIC')) | 4.41 | 0.0438 * |
| C(arch, Treatment('CRNN')) | 0.35 | 0.5587 |
| C(prior, Treatment('nogeom')) | 0.13 | 0.7226 |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')) | 0.03 | 0.8707 |
| C(modality, Treatment('MIC')):C(prior, Treatment('nogeom')) | 1.21 | 0.2786 |
| arch:prior (FOCAL) | 4.62 | 0.0393 * |
| C(modality, Treatment('MIC')):C(arch, Treatment('CRNN')):C(prior, Treatment('nogeom')) | 0.02 | 0.8788 |

## 2. Mixed model: random intercept per seed (CRNN vs Transformer)
Seeds 0-4 are matched across cells; modelling a per-seed random intercept
removes between-seed variance. The arch x prior interaction is tested by a
likelihood-ratio test (full vs model with the interaction removed).

- The per-seed random-intercept variance is estimated at the **0 boundary**
  (singular random-effects covariance), i.e. **ICC(seed) ~ 0** on DOAE_CD:
  seeds carry no extra between-group variance for directional error.
- The mixed model therefore **degenerates to the fixed-effects OLS** of
  Section 1, so the focal OLS interaction is the appropriate estimate, and
  the exact seed-matched within-design test is the paired second-difference
  in Section 4.

## 3. Ordered-architecture trend test (CRNN < Conformer < Transformer)
On the per-seed geometry effect delta = LE(full) - LE(nogeom); H1 = delta
increases monotonically as built-in locality is removed (helpful -> harmful).

### pooled over modality (n=10/arch)
- mean delta: CRNN -2.95, Conformer -0.48, Transformer +4.13 (deg)
- Jonckheere-Terpstra: J = 192.0, z = +1.60, **p(1-sided, increasing) = 0.0551**, p(2-sided) = 0.1103
- linear trend (delta ~ arch rank): slope = +3.54 deg/step, p = 0.0726
- Spearman rho = +0.28, p = 0.1297

### MIC only (n=5/arch)
- mean delta: CRNN -0.88, Conformer -1.83, Transformer +5.70 (deg)
- Jonckheere-Terpstra: J = 43.0, z = +0.58, **p(1-sided, increasing) = 0.2806**, p(2-sided) = 0.5612
- linear trend (delta ~ arch rank): slope = +3.29 deg/step, p = 0.3573
- Spearman rho = +0.13, p = 0.6384

### FOA only (n=5/arch)
- mean delta: CRNN -5.02, Conformer +0.88, Transformer +2.57 (deg)
- Jonckheere-Terpstra: J = 53.0, z = +1.64, **p(1-sided, increasing) = 0.0507**, p(2-sided) = 0.1015
- linear trend (delta ~ arch rank): slope = +3.80 deg/step, p = 0.0637
- Spearman rho = +0.42, p = 0.1232

## 4. Paired second-difference interaction (CRNN vs Transformer)
arch x prior as a within-seed paired contrast: per matched (modality, seed),
D = delta_CRNN - delta_Transformer; one-sample t-test that D != 0.

- n_pairs = 10, mean(D) = -7.09 deg, t = -2.59, **p = 0.0294**, d_z = -0.82
- MIC: mean(D) = -6.58, t = -1.30, p = 0.2625 (n=5)
- FOA: mean(D) = -7.59, t = -2.66, p = 0.0567 (n=5)
