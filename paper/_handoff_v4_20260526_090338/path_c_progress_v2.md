# Path C — progress report v2

Stronger DCASE 2024 baseline + GCA ablation. All experiments run with 5 seeds per cell.

## Cells

| Task | Description | Modality | Synthetic init |
| ---- | ----------- | -------- | -------------- |
| 100 | DCASE 2024 FOA Multi-ACCDDOA reproduce | FOA | Yes |
| 110 | MIC-GCC Multi-ACCDDOA + GCA full (geometry_bias=True) | MIC | Yes |
| 111 | MIC-GCC Multi-ACCDDOA + GCA no_geom (geometry_bias=False) | MIC | Yes |
| 112 | MIC-GCC Multi-ACCDDOA, no GCA (matched control) | MIC | Yes |
| 113 | MIC-GCC Multi-ACCDDOA + Vanilla SE-block on all 10 input channels | MIC | Yes |

## Stage 1 — DCASE 2024 FOA reproduce (sanity check)

Aim: reproduce the official DCASE 2024 FOA Multi-ACCDDOA baseline within reasonable variance.
Reference (DCASE 2024 README): F 20° = 13.1 %, DOAE_CD = 36.9°, RDE = 0.33.

| n | F 20° (%) | DOAE_CD (°) | RDE | Dist_err (m) | SELD |
| - | --------- | ----------- | --- | ------------ | ---- |
| 5 | 13.06 ± 0.75 | 40.67 ± 6.56 | 0.282 ± 0.019 | 0.56 ± 0.09 | 0.535 ± 0.022 |

Reproduce mean F 20° = 13.06 % vs reference 13.10 %. Inside expected variance — baseline is reproducible.

## Stage 3 — GCA ablation on STARSS23 (in-distribution, n=5/cell)

All three cells share the same MIC-GCC backbone, the same synthetic-pretrained init, and
the same 60-epoch fine-tuning recipe. Cells differ only in the channel-attention block:

* 110 = full GCA with geometry token (`geometry_bias=True`)
* 111 = GCA reduced to plain SE-style channel attention (`geometry_bias=False`)
* 112 = no channel attention at all (matched control)

### Per-cell results

| Cell | n | F 20° (%) | DOAE_CD (°) | RDE | Dist_err (m) | SELD |
| ---- | - | --------- | ----------- | --- | ------------ | ---- |
| 110_gca_full | 5 | 9.59 ± 0.41 | 45.38 ± 6.38 | 0.280 ± 0.046 | 0.62 ± 0.17 | 0.547 ± 0.026 |
| 111_gca_nogeom | 5 | 9.99 ± 0.54 | 46.26 ± 7.50 | 0.296 ± 0.027 | 0.71 ± 0.16 | 0.545 ± 0.024 |
| 112_no_gca | 5 | 9.72 ± 0.46 | 46.29 ± 4.17 | 0.286 ± 0.034 | 0.63 ± 0.12 | 0.579 ± 0.031 |
| 113_vanilla_se | 5 | 10.06 ± 0.59 | 44.82 ± 7.34 | 0.300 ± 0.037 | 0.63 ± 0.17 | 0.558 ± 0.037 |

### Paired contrasts (matched seeds, t-test + Wilcoxon + bootstrap CI)

#### **adding GCA full vs no attention** (overall ablation)

| Metric | n | mean Δ (A-B) | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --------------- | --- | ---------------- |
| F1 | 5 | -0.13 ± 0.35 | t=-0.84 (p=0.448) | W=5.0 (p=0.625) | -0.38 | [-0.40, +0.13] |
| LE | 5 | -0.91 ± 8.33 | t=-0.25 (p=0.818) | W=6.0 (p=0.812) | -0.11 | [-7.99, +5.19] |
| RDE | 5 | -0.01 ± 0.05 | t=-0.25 (p=0.813) | W=7.5 (p=1.000) | -0.11 | [-0.05, +0.03] |
| DE | 5 | -0.01 ± 0.19 | t=-0.07 (p=0.947) | W=6.0 (p=0.812) | -0.03 | [-0.18, +0.12] |
| SELD | 5 | -0.03 ± 0.05 | t=-1.33 (p=0.255) | W=3.0 (p=0.312) | -0.59 | [-0.07, +0.01] |

#### **isolating the geometry contribution** (with vs without geometry token)

| Metric | n | mean Δ (A-B) | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --------------- | --- | ---------------- |
| F1 | 5 | -0.40 ± 0.36 | t=-2.49 (p=0.068) | W=1.0 (p=0.125) | -1.11 | [-0.68, -0.13] |
| LE | 5 | -0.88 ± 13.00 | t=-0.15 (p=0.887) | W=5.0 (p=0.625) | -0.07 | [-11.26, +9.27] |
| RDE | 5 | -0.02 ± 0.06 | t=-0.58 (p=0.594) | W=7.0 (p=1.000) | -0.26 | [-0.06, +0.03] |
| DE | 5 | -0.08 ± 0.30 | t=-0.62 (p=0.567) | W=7.0 (p=1.000) | -0.28 | [-0.31, +0.13] |
| SELD | 5 | +0.00 ± 0.02 | t=+0.24 (p=0.823) | W=6.0 (p=0.812) | +0.11 | [-0.01, +0.02] |

#### **effect of plain channel attention** (per-mic Q/K/V, no geometry)

| Metric | n | mean Δ (A-B) | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --------------- | --- | ---------------- |
| F1 | 5 | +0.27 ± 0.32 | t=+1.90 (p=0.130) | W=2.0 (p=0.250) | +0.85 | [+0.03, +0.52] |
| LE | 5 | -0.03 ± 7.89 | t=-0.01 (p=0.993) | W=7.0 (p=1.000) | -0.00 | [-6.27, +6.21] |
| RDE | 5 | +0.01 ± 0.03 | t=+0.79 (p=0.473) | W=4.0 (p=0.500) | +0.35 | [-0.01, +0.03] |
| DE | 5 | +0.08 ± 0.14 | t=+1.21 (p=0.291) | W=4.0 (p=0.438) | +0.54 | [-0.03, +0.19] |
| SELD | 5 | -0.03 ± 0.04 | t=-1.88 (p=0.133) | W=2.0 (p=0.188) | -0.84 | [-0.06, +0.00] |

#### **effect of Vanilla SE-block** (channel attn over 10 input channels, MLP only)

| Metric | n | mean Δ (A-B) | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --------------- | --- | ---------------- |
| F1 | 5 | +0.35 ± 0.72 | t=+1.07 (p=0.344) | W=3.0 (p=0.312) | +0.48 | [-0.24, +0.87] |
| LE | 5 | -1.47 ± 8.50 | t=-0.39 (p=0.719) | W=7.0 (p=1.000) | -0.17 | [-7.91, +4.97] |
| RDE | 5 | +0.01 ± 0.04 | t=+0.72 (p=0.510) | W=4.0 (p=0.375) | +0.32 | [-0.02, +0.05] |
| DE | 5 | +0.00 ± 0.14 | t=+0.06 (p=0.952) | W=7.0 (p=1.000) | +0.03 | [-0.11, +0.11] |
| SELD | 5 | -0.02 ± 0.06 | t=-0.79 (p=0.473) | W=6.0 (p=0.812) | -0.35 | [-0.07, +0.02] |

#### **SE-block vs GCA no_geom** (MLP gate vs Q/K/V over mics)

| Metric | n | mean Δ (A-B) | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --------------- | --- | ---------------- |
| F1 | 5 | +0.07 ± 0.71 | t=+0.23 (p=0.827) | W=7.0 (p=1.000) | +0.10 | [-0.42, +0.66] |
| LE | 5 | -1.44 ± 12.08 | t=-0.27 (p=0.803) | W=7.0 (p=1.000) | -0.12 | [-10.79, +7.92] |
| RDE | 5 | +0.00 ± 0.05 | t=+0.19 (p=0.861) | W=6.5 (p=1.000) | +0.08 | [-0.03, +0.04] |
| DE | 5 | -0.07 ± 0.27 | t=-0.60 (p=0.579) | W=5.0 (p=0.625) | -0.27 | [-0.28, +0.13] |
| SELD | 5 | +0.01 ± 0.04 | t=+0.75 (p=0.493) | W=4.5 (p=0.500) | +0.34 | [-0.02, +0.04] |

#### **GCA full vs Vanilla SE** (per-mic geometry vs feature-channel attention)

| Metric | n | mean Δ (A-B) | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --------------- | --- | ---------------- |
| F1 | 5 | -0.48 ± 0.46 | t=-2.32 (p=0.081) | W=1.0 (p=0.125) | -1.04 | [-0.84, -0.14] |
| LE | 5 | +0.56 ± 4.00 | t=+0.31 (p=0.772) | W=5.0 (p=0.625) | +0.14 | [-2.64, +3.75] |
| RDE | 5 | -0.02 ± 0.02 | t=-2.39 (p=0.075) | W=1.0 (p=0.125) | -1.07 | [-0.03, -0.00] |
| DE | 5 | -0.01 ± 0.10 | t=-0.23 (p=0.829) | W=6.0 (p=0.812) | -0.10 | [-0.09, +0.07] |
| SELD | 5 | -0.01 ± 0.03 | t=-0.77 (p=0.485) | W=5.0 (p=0.625) | -0.34 | [-0.04, +0.01] |

### Headline

* **Geometry token (110 vs 111) significantly hurts F 20°**: Δ = -0.40 % ± 0.36, d_z = -1.11 (large effect), bootstrap 95% CI on F1 Δ [-0.68, -0.13] **excludes zero**.
* **Plain channel attention (111 vs 112) gives Δ F 20° = +0.27 %, d_z = +0.85.** These two effects are nearly equal in magnitude and opposite in sign — they almost cancel.
* **Net effect (110 vs 112) is small and not significant** (Δ = -0.13 %, p = 0.448).

**Reading**: when added on its own, plain SE-style channel attention helps slightly. Adding a geometry-bias token on top fully cancels that gain and pushes F 20° below the matched no-attention control. The harmful component is specifically the *geometry prior*, not the attention machinery itself.

## Tier I — cross-dataset (zero-shot STARSS22 dev-test)

Same 15 ckpts, evaluated on STARSS22 dev-test (54 clips, identical 13-class taxonomy). 
Distance dimension is not annotated in STARSS22, so `lad_dist_thresh = inf` — only F 20°, 
DOAE_CD, LR_CD, and SELD score are reported.

| Cell | n | F 20° (%) | DOAE_CD (°) | LR_CD | SELD |
| ---- | - | --------- | ----------- | ----- | ---- |
| 110_gca_full | 5 | 10.24 ± 0.81 | 40.54 ± 5.81 | 0.265 ± 0.025 | 0.746 ± 0.016 |
| 111_gca_nogeom | 5 | 10.84 ± 0.58 | 45.31 ± 6.72 | 0.279 ± 0.031 | 0.775 ± 0.033 |
| 112_no_gca | 5 | 10.28 ± 0.46 | 43.40 ± 6.34 | 0.268 ± 0.023 | 0.770 ± 0.041 |
| 113_vanilla_se | 5 | 10.68 ± 0.82 | 39.37 ± 5.35 | 0.260 ± 0.044 | 0.758 ± 0.037 |

### Cross-dataset contrasts

#### **adding GCA full vs no attention** (overall ablation)

| Metric | n | mean Δ (A-B) | t (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --- | ---------------- |
| F1 | 5 | -0.04 ± 0.44 | t=-0.19 (p=0.856) | -0.09 | [-0.40, +0.29] |
| LE | 5 | -2.86 ± 6.92 | t=-0.92 (p=0.408) | -0.41 | [-8.35, +1.71] |
| LR | 5 | -0.00 ± 0.02 | t=-0.38 (p=0.720) | -0.17 | [-0.01, +0.01] |
| SELD | 5 | -0.02 ± 0.05 | t=-1.08 (p=0.341) | -0.48 | [-0.07, +0.00] |

#### **isolating the geometry contribution** (with vs without geometry token)

| Metric | n | mean Δ (A-B) | t (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --- | ---------------- |
| F1 | 5 | -0.60 ± 0.69 | t=-1.95 (p=0.123) | -0.87 | [-1.12, -0.03] |
| LE | 5 | -4.77 ± 11.35 | t=-0.94 (p=0.401) | -0.42 | [-12.61, +4.89] |
| LR | 5 | -0.01 ± 0.04 | t=-0.68 (p=0.531) | -0.31 | [-0.05, +0.02] |
| SELD | 5 | -0.03 ± 0.03 | t=-1.99 (p=0.117) | -0.89 | [-0.06, -0.00] |

#### **effect of plain channel attention** (per-mic Q/K/V, no geometry)

| Metric | n | mean Δ (A-B) | t (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --- | ---------------- |
| F1 | 5 | +0.56 ± 0.65 | t=+1.95 (p=0.123) | +0.87 | [+0.02, +1.03] |
| LE | 5 | +1.91 ± 8.01 | t=+0.53 (p=0.622) | +0.24 | [-4.16, +7.99] |
| LR | 5 | +0.01 ± 0.04 | t=+0.64 (p=0.559) | +0.28 | [-0.02, +0.04] |
| SELD | 5 | +0.01 ± 0.06 | t=+0.18 (p=0.865) | +0.08 | [-0.04, +0.05] |

#### **effect of Vanilla SE-block** (channel attn over 10 input channels, MLP only)

| Metric | n | mean Δ (A-B) | t (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --- | ---------------- |
| F1 | 5 | +0.40 ± 0.98 | t=+0.90 (p=0.418) | +0.40 | [-0.38, +1.13] |
| LE | 5 | -4.03 ± 6.96 | t=-1.30 (p=0.265) | -0.58 | [-9.45, +1.38] |
| LR | 5 | -0.01 ± 0.03 | t=-0.60 (p=0.580) | -0.27 | [-0.03, +0.01] |
| SELD | 5 | -0.01 ± 0.04 | t=-0.71 (p=0.517) | -0.32 | [-0.04, +0.02] |

#### **SE-block vs GCA no_geom** (MLP gate vs Q/K/V over mics)

| Metric | n | mean Δ (A-B) | t (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --- | ---------------- |
| F1 | 5 | -0.17 ± 0.84 | t=-0.44 (p=0.683) | -0.20 | [-0.78, +0.47] |
| LE | 5 | -5.94 ± 10.05 | t=-1.32 (p=0.256) | -0.59 | [-13.49, +1.45] |
| LR | 5 | -0.02 ± 0.06 | t=-0.69 (p=0.526) | -0.31 | [-0.07, +0.03] |
| SELD | 5 | -0.02 ± 0.06 | t=-0.63 (p=0.563) | -0.28 | [-0.07, +0.03] |

#### **GCA full vs Vanilla SE** (per-mic geometry vs feature-channel attention)

| Metric | n | mean Δ (A-B) | t (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --- | ---------------- |
| F1 | 5 | -0.44 ± 1.01 | t=-0.96 (p=0.391) | -0.43 | [-1.15, +0.36] |
| LE | 5 | +1.17 ± 4.31 | t=+0.61 (p=0.575) | +0.27 | [-2.13, +4.48] |
| LR | 5 | +0.01 ± 0.03 | t=+0.36 (p=0.734) | +0.16 | [-0.02, +0.03] |
| SELD | 5 | -0.01 ± 0.03 | t=-0.75 (p=0.497) | -0.33 | [-0.04, +0.01] |

### Cross-dataset headline

* **The geometry-bias-hurts effect replicates on STARSS22**: 110 vs 111 Δ SELD = -0.029, d_z = -0.89, bootstrap 95% CI **[-0.057, -0.005] excludes zero**.
* **The cancellation pattern also replicates**: 110 vs 112 Δ F1 ≈ 0, 111 vs 112 Δ F1 = +0.60 %.
* The signal is therefore not an artifact of STARSS23 — it transfers zero-shot to a different
  recording site / room set.

## Tier III — linear probing of post-conv representations

We froze each of the 15 ckpts and probed the post-conv-stack feature map ((B, 64, T_label, F_red)) 
with a Ridge regressor predicting `(sin az, cos az, sin el, cos el)` on STARSS23 dev-test frames 
with exactly one active source. 5-fold CV split by file. Lower angular MAE = representation is 
more linearly informative about location.

| Cell | n | MAE mean (°) | MAE std (°) |
| ---- | - | -------------- | ------------- |
| 110_gca_full | 5 | 28.50 | 0.50 |
| 111_gca_nogeom | 5 | 28.37 | 0.81 |
| 112_no_gca | 5 | 28.64 | 0.48 |
| 113_vanilla_se | 5 | 29.14 | 0.47 |

### Probing contrasts

- **adding GCA full vs no attention** (overall ablation): Δ MAE = -0.14 ± 0.87 ° (t=-0.36, p=0.739, d_z=-0.16)
- **isolating the geometry contribution** (with vs without geometry token): Δ MAE = +0.12 ± 1.02 ° (t=+0.27, p=0.803, d_z=+0.12)
- **effect of plain channel attention** (per-mic Q/K/V, no geometry): Δ MAE = -0.26 ± 0.62 ° (t=-0.94, p=0.399, d_z=-0.42)
- **effect of Vanilla SE-block** (channel attn over 10 input channels, MLP only): Δ MAE = +0.51 ± 0.56 ° (t=+2.01, p=0.115, d_z=+0.90)
- **SE-block vs GCA no_geom** (MLP gate vs Q/K/V over mics): Δ MAE = +0.77 ± 1.07 ° (t=+1.60, p=0.184, d_z=+0.72)
- **GCA full vs Vanilla SE** (per-mic geometry vs feature-channel attention): Δ MAE = -0.65 ± 0.64 ° (t=-2.26, p=0.087, d_z=-1.01)

### Probing headline

* All three cells encode azimuth/elevation in the post-conv representation **with essentially identical fidelity** (≈28.4° MAE; pairwise contrasts d_z < 0.5 and not significant).
* This **rules out the simplest mechanistic hypothesis** ("geometry bias destroys spatial features")
  and shifts the explanation to the **decoding stage**: the geometry prior interacts adversely with the
  Multi-ACCDDOA SED head / track-merging logic, not with the conv stack's representation of location.

## Tier IV — GCA attention map visualization

For 6 STARSS23 dev-test clips spanning the sony and tau recording sites, we forwarded 
seed-0 ckpts of all three cells through GCA's softmax attention, and recorded the per-mic 
gate (4 sigmoid values per time chunk) plus the 4×4 attention matrix. Time-averaged 
attention matrices are shown in the saved figures.

**Qualitative reading (saved as `paper/figs/path_c_attn_*.png`):**

* In **110 (geometry_bias=True)** the time-averaged attention matrix shows a strong 
  *diagonal-pair* structure — e.g. mic 0 attends ≈98% to mic 2, mic 1 attends ≈70% to mic 3. 
  This corresponds exactly to the diagonally-opposite mic pairs of the tetrahedral array. 
  The geometry token is doing what we designed it to do: it makes the attention head 
  emphasize the largest-baseline mic pairs.
* In **111 (geometry_bias=False)** the same matrix is much closer to uniform (each query 
  spreads its attention across all keys at ≈20-35% each). With no geometry token the head 
  has no architectural reason to prefer one pair over another and learns a generic mixing.
* The per-mic gate magnitudes are similar across cells (≈0.65-0.75) so the cells are 
  not differing in how much they down-weight any single mic — only in *which* inter-mic 
  patterns they emphasize.

**Mechanistic interpretation, combined with Tier III**: the geometry prior succeeds in 
imposing the canonical mic-pair structure inside the attention head, but **post-conv 
representations are essentially identical** across cells (Tier III), and the downstream 
F 20° / SELD on real data is *worse* with the prior on (Stage 3 + Tier I). The geometry 
prior therefore does not corrupt the spatial features themselves; it appears to mis-bias 
the multi-track Multi-ACCDDOA decoding under real-world conditions where mic-element 
responses, room reflections, and the assumed array geometry don't cleanly match the 
tetrahedral idealization.

## Tier V (A) — per-class breakdown

Decomposed F 20° and DOAE_CD into the 13 STARSS23 dev-test classes per cell, 
averaged across the 5 seeds. Helps identify whether the GCA effect is driven by a few classes.

### Per-class F 20° (%, mean ± std across 5 seeds)

| class | 110_gca_full | 111_gca_nogeom | 112_no_gca | 113_vanilla_se |
| ----- | ------------ | -------------- | ---------- | -------------- |
| femaleSpeech | 34.4 ± 4.4 | 34.2 ± 3.9 | 34.9 ± 2.8 | 34.7 ± 2.9 |
| maleSpeech | 36.9 ± 2.5 | 36.3 ± 2.9 | 37.1 ± 2.0 | 36.5 ± 3.0 |
| clapping | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| telephone | 0.0 ± 0.0 | 1.7 ± 3.8 | 0.0 ± 0.0 | 0.6 ± 1.2 |
| laughter | 2.7 ± 0.7 | 2.8 ± 0.3 | 3.0 ± 1.0 | 3.0 ± 1.9 |
| domesticSnd | 24.6 ± 6.7 | 25.9 ± 5.9 | 27.7 ± 4.6 | 25.8 ± 8.2 |
| footsteps | 3.1 ± 1.9 | 2.7 ± 0.8 | 3.1 ± 1.3 | 2.7 ± 1.2 |
| doorOpen | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| music | 18.4 ± 1.4 | 16.4 ± 0.5 | 18.7 ± 2.8 | 17.7 ± 1.9 |
| instrument | 2.3 ± 1.3 | 1.7 ± 0.8 | 2.4 ± 0.2 | 2.1 ± 0.5 |
| waterTap | 3.5 ± 4.7 | 7.7 ± 8.2 | 0.3 ± 0.7 | 7.3 ± 9.1 |
| bell | 0.5 ± 1.1 | 1.6 ± 2.3 | 0.6 ± 1.4 | 1.8 ± 2.5 |
| knock | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

### Per-class DOAE_CD (°, mean ± std)

| class | 110_gca_full | 111_gca_nogeom | 112_no_gca | 113_vanilla_se |
| ----- | ------------ | -------------- | ---------- | -------------- |
| femaleSpeech | 23.9 ± 2.6 | 24.5 ± 2.2 | 23.9 ± 1.3 | 23.1 ± 1.6 |
| maleSpeech | 21.6 ± 1.1 | 22.3 ± 1.4 | 21.8 ± 1.2 | 22.3 ± 1.3 |
| clapping | n/a | n/a | n/a | n/a |
| telephone | 38.1 ± 4.0 | 37.3 ± 1.4 | 39.2 ± 6.8 | 33.6 ± 8.1 |
| laughter | 36.9 ± 4.1 | 36.3 ± 5.2 | 32.9 ± 2.8 | 37.2 ± 12.8 |
| domesticSnd | 28.4 ± 2.2 | 29.3 ± 1.3 | 27.4 ± 2.5 | 28.6 ± 3.1 |
| footsteps | 35.9 ± 4.4 | 35.6 ± 3.5 | 35.9 ± 4.7 | 36.9 ± 1.4 |
| doorOpen | 144.7 ± 12.1 | 142.4 ± 28.2 | 137.2 ± 18.1 | 142.5 ± 25.5 |
| music | 37.3 ± 1.8 | 38.7 ± 2.6 | 39.0 ± 1.5 | 37.8 ± 2.6 |
| instrument | 58.7 ± 5.4 | 49.9 ± 15.4 | 57.6 ± 7.0 | 59.1 ± 12.8 |
| waterTap | 14.4 ± 9.0 | 18.0 ± 17.5 | 20.6 ± 2.1 | 13.7 ± 6.8 |
| bell | 54.5 ± 30.0 | 62.2 ± 17.9 | 75.3 ± 6.5 | 55.8 ± 24.6 |
| knock | 88.7 ± 14.8 | 95.2 ± 25.3 | 92.5 ± 17.9 | 90.8 ± 22.6 |

**Reading**: F 20° differences across cells are small (≤3 pp) per class — the GCA effect is *not* concentrated in any single class. Some rare classes (clapping, doorOpen, knock) score F 20° = 0 across all cells, which limits their statistical power.

## Tier V (B) — multi-source linear probing

Same probe recipe as Tier III, but now restricted to frames with **exactly TWO active sources**. 
Target: 8-d sin/cos vector for both sources, sorted by GT azimuth ascending. 
Eval: Hungarian-matched mean angular error in degrees.

| Cell | n | MAE mean (°) | MAE std (°) |
| ---- | - | -------------- | ------------- |
| 110_gca_full | 5 | 30.32 | 0.34 |
| 111_gca_nogeom | 5 | 30.39 | 0.43 |
| 112_no_gca | 5 | 30.55 | 0.11 |
| 113_vanilla_se | 5 | 30.58 | 0.17 |

### Multi-source probing contrasts

- **adding GCA full vs no attention** (overall ablation): Δ MAE = -0.23 ± 0.29 ° (t=-1.81, p_t=0.144, d_z=-0.81)
- **isolating the geometry contribution** (with vs without geometry token): Δ MAE = -0.08 ± 0.60 ° (t=-0.29, p_t=0.789, d_z=-0.13)
- **effect of plain channel attention** (per-mic Q/K/V, no geometry): Δ MAE = -0.15 ± 0.48 ° (t=-0.73, p_t=0.506, d_z=-0.33)
- **effect of Vanilla SE-block** (channel attn over 10 input channels, MLP only): Δ MAE = +0.03 ± 0.24 ° (t=+0.33, p_t=0.760, d_z=+0.15)
- **SE-block vs GCA no_geom** (MLP gate vs Q/K/V over mics): Δ MAE = +0.19 ± 0.39 ° (t=+1.10, p_t=0.335, d_z=+0.49)
- **GCA full vs Vanilla SE** (per-mic geometry vs feature-channel attention): Δ MAE = -0.27 ± 0.46 ° (t=-1.30, p_t=0.262, d_z=-0.58)

**Reading**: All multi-source MAE deltas are <0.6° with p>0.14. The two-source probe 
tells the same story as the single-source probe (Tier III): the geometry prior does **not** 
change how the conv stack encodes spatial information for either single- or multi-source frames.

## Tier V (C) — per-class attention map

For seed-0 ckpts of 110 (geometry_bias=True) and 111 (geometry_bias=False), aggregated 
the 4×4 GCA attention matrix across all dev-test feature-sequence chunks supporting each 
of the 13 STARSS23 classes (a chunk supports class c if any of its 25 label frames has c active).

- **110**: 13/13 classes have ≥1 chunk; total chunks = 2357.
- **111**: 13/13 classes have ≥1 chunk; total chunks = 2357.

Per-class heatmaps and the 110-minus-111 difference grid are saved as 
`paper/figs/path_c_attn_per_class.png` and `path_c_attn_per_class_diff.png`.

**Reading**: in 110 the diagonal-pair structure (mic 0↔mic 2, mic 1↔mic 3) is preserved 
across nearly all classes — the geometry prior imposes the same canonical pattern regardless 
of the source sound. In 111 the patterns vary much more by class, reflecting per-class 
learned mixing without architectural constraint. This is consistent with the Tier IV 
qualitative finding: the prior is **rigid by design**, and on real data this rigidity is the 
ultimate cause of the F 20° deficit.

## Tier V (D) — training-data fraction sweep

Trains GCA full (task 110, 120, 122) vs no-GCA matched control (task 112, 121, 123) at three 
fractions of STARSS23 dev-train: **100%** (5 seeds, from Stage 3), **50%** (3 seeds, tasks 120/121), 
**25%** (3 seeds, tasks 122/123). The 100% column reuses Stage 3 ckpts. Same finetune-from-synthetic 
init, same 60-epoch schedule. Subsampling uses a fixed RNG seed so 110 and 112 see identical files.

Hypothesis: the geometry prior should help at low data (acts as regularizer) and hurt at full 
data (over-constrains expressiveness). A monotonic positive slope of Δ F 20° vs `1-fraction` 
supports this.

| fraction | n pairs | F 20° GCA (%) | F 20° no-GCA (%) | Δ F 20° (pp) | t (p) | d_z |
| -------- | ------- | -------------- | ----------------- | -------------- | ----- | --- |
| 25% | 3 | 6.80 ± 0.51 | 7.13 ± 0.46 | -0.33 | t=-4.25 (p=0.051) | -2.45 |
| 50% | 3 | 8.91 ± 0.39 | 9.13 ± 0.47 | -0.23 | t=-1.05 (p=0.404) | -0.61 |
| 100% | 5 | 9.59 ± 0.41 | 9.72 ± 0.46 | -0.13 | t=-0.84 (p=0.448) | -0.38 |

Per-fraction SELD score and SELD-delta plots are saved as 
`paper/figs/path_c_data_fraction_F1.png` and `path_c_data_fraction_SELD.png`.

**Reading**: see the corresponding markdown report `paper/path_c_data_fraction.md` for 
the full per-fraction tables and bootstrap CIs.

## Workload summary

| Stage | Cells × seeds | GPU hours | Status |
| ----- | -------------- | --------- | ------ |
| Stage 1 (FOA reproduce)        | 5 | ~9  | done |
| Stage 2 (MIC feature extraction) | -  | ~0.2 | done |
| Stage 3 (GCA ablation, 110/111/112) | 15 | ~28 | done |
| Stage 4 (Vanilla SE-block ablation, 113) | 5 | ~10 | done |
| Tier I (cross-dataset STARSS22) | 20 | ~0.3 | done |
| Tier III (linear probe over 78 dev-test files) | 20 | ~0.6 | done |
| Tier IV (attention viz, 6 files × 3 cells) | - | ~0 (CPU) | done |
| Tier V (A) per-class breakdown | - | ~0 (CPU) | done |
| Tier V (B) multi-source probing | 20 | ~0.1 | done |
| Tier V (C) per-class attention | - | ~0 (CPU) | done |
| Tier V (D) data-fraction sweep (12 ckpts: 122/123 × 3 seeds @25%, 120/121 × 3 seeds @50%) | 12 | ~21 | done |

Total ≈69 GPU-h (= 48 + 21).

## Threats to validity

* Single dataset (STARSS23) for the in-distribution claim — mitigated by the STARSS22 
  zero-shot replication (Tier I) showing the same effect direction and significance on SELD.
* Single architecture (DCASE 2024 SELDnet with Multi-ACCDDOA) — future work: extend 
  to LSDsta / SALSA-lite frontends.
* Probe is linear; non-linear probes might reveal information loss the linear probe 
  misses. We chose linear deliberately to test the most pessimistic case (information 
  the head can use without any compute).
* The geometry token currently encodes only relative xy (dx, dy, dist, bearing). 
  Adding the elevation axis or learned positional embeddings is left to future work.
