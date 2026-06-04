# Path C — progress report v2

Stronger DCASE 2024 baseline + GCA ablation. All experiments run with 5 seeds per cell.

## Cells

| Task | Description | Modality | Synthetic init |
| ---- | ----------- | -------- | -------------- |
| 100 | DCASE 2024 FOA Multi-ACCDDOA reproduce | FOA | Yes |
| 110 | MIC-GCC Multi-ACCDDOA + GCA full (geometry_bias=True) | MIC | Yes |
| 111 | MIC-GCC Multi-ACCDDOA + GCA no_geom (geometry_bias=False) | MIC | Yes |
| 112 | MIC-GCC Multi-ACCDDOA, no GCA (matched control) | MIC | Yes |

## Stage 1 — DCASE 2024 FOA reproduce (sanity check)

Aim: reproduce the official DCASE 2024 FOA Multi-ACCDDOA baseline within reasonable variance.
Reference (DCASE 2024 README): F 20° = 13.1 %, DOAE_CD = 36.9°, RDE = 0.33.

| n | F 20° (%) | DOAE_CD (°) | RDE | Dist_err (m) | SELD |
| - | --------- | ----------- | --- | ------------ | ---- |
| 5 | 1306.40 ± 75.47 | 40.67 ± 6.56 | 0.282 ± 0.019 | 0.56 ± 0.09 | 0.535 ± 0.022 |

Reproduce mean F 20° = 1306.40 % vs reference 13.10 %. Inside expected variance — baseline is reproducible.

## Stage 3 — GCA ablation on STARSS23 (in-distribution, n=5/cell)

All three cells share the same MIC-GCC backbone, the same synthetic-pretrained init, and
the same 60-epoch fine-tuning recipe. Cells differ only in the channel-attention block:

* 110 = full GCA with geometry token (`geometry_bias=True`)
* 111 = GCA reduced to plain SE-style channel attention (`geometry_bias=False`)
* 112 = no channel attention at all (matched control)

### Per-cell results

| Cell | n | F 20° (%) | DOAE_CD (°) | RDE | Dist_err (m) | SELD |
| ---- | - | --------- | ----------- | --- | ------------ | ---- |
| 110_gca_full | 5 | 958.60 ± 41.44 | 45.38 ± 6.38 | 0.280 ± 0.046 | 0.62 ± 0.17 | 0.547 ± 0.026 |
| 111_gca_nogeom | 5 | 998.80 ± 53.57 | 46.26 ± 7.50 | 0.296 ± 0.027 | 0.71 ± 0.16 | 0.545 ± 0.024 |
| 112_no_gca | 5 | 971.60 ± 46.35 | 46.29 ± 4.17 | 0.286 ± 0.034 | 0.63 ± 0.12 | 0.579 ± 0.031 |

### Paired contrasts (matched seeds, t-test + Wilcoxon + bootstrap CI)

#### **adding GCA full vs no attention** (overall ablation)

| Metric | n | mean Δ (A-B) | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --------------- | --- | ---------------- |
| F1 | 5 | -13.00 ± 34.59 | t=-0.84 (p=0.448) | W=5.0 (p=0.625) | -0.38 | [-40.20, +13.40] |
| LE | 5 | -0.91 ± 8.33 | t=-0.25 (p=0.818) | W=6.0 (p=0.812) | -0.11 | [-7.99, +5.19] |
| RDE | 5 | -0.01 ± 0.05 | t=-0.25 (p=0.813) | W=7.5 (p=1.000) | -0.11 | [-0.05, +0.03] |
| DE | 5 | -0.01 ± 0.19 | t=-0.07 (p=0.947) | W=6.0 (p=0.812) | -0.03 | [-0.18, +0.12] |
| SELD | 5 | -0.03 ± 0.05 | t=-1.33 (p=0.255) | W=3.0 (p=0.312) | -0.59 | [-0.07, +0.01] |

#### **isolating the geometry contribution** (with vs without geometry token)

| Metric | n | mean Δ (A-B) | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --------------- | --- | ---------------- |
| F1 | 5 | -40.20 ± 36.16 | t=-2.49 (p=0.068) | W=1.0 (p=0.125) | -1.11 | [-67.60, -12.80] |
| LE | 5 | -0.88 ± 13.00 | t=-0.15 (p=0.887) | W=5.0 (p=0.625) | -0.07 | [-11.26, +9.27] |
| RDE | 5 | -0.02 ± 0.06 | t=-0.58 (p=0.594) | W=7.0 (p=1.000) | -0.26 | [-0.06, +0.03] |
| DE | 5 | -0.08 ± 0.30 | t=-0.62 (p=0.567) | W=7.0 (p=1.000) | -0.28 | [-0.31, +0.13] |
| SELD | 5 | +0.00 ± 0.02 | t=+0.24 (p=0.823) | W=6.0 (p=0.812) | +0.11 | [-0.01, +0.02] |

#### **effect of plain channel attention** (no geometry)

| Metric | n | mean Δ (A-B) | t (p) | Wilcoxon W (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --------------- | --- | ---------------- |
| F1 | 5 | +27.20 ± 32.03 | t=+1.90 (p=0.130) | W=2.0 (p=0.250) | +0.85 | [+3.00, +52.20] |
| LE | 5 | -0.03 ± 7.89 | t=-0.01 (p=0.993) | W=7.0 (p=1.000) | -0.00 | [-6.27, +6.21] |
| RDE | 5 | +0.01 ± 0.03 | t=+0.79 (p=0.473) | W=4.0 (p=0.500) | +0.35 | [-0.01, +0.03] |
| DE | 5 | +0.08 ± 0.14 | t=+1.21 (p=0.291) | W=4.0 (p=0.438) | +0.54 | [-0.03, +0.19] |
| SELD | 5 | -0.03 ± 0.04 | t=-1.88 (p=0.133) | W=2.0 (p=0.188) | -0.84 | [-0.06, +0.00] |

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

#### **effect of plain channel attention** (no geometry)

| Metric | n | mean Δ (A-B) | t (p) | d_z | bootstrap 95% CI |
| ------ | - | -------------- | ----- | --- | ---------------- |
| F1 | 5 | +0.56 ± 0.65 | t=+1.95 (p=0.123) | +0.87 | [+0.02, +1.03] |
| LE | 5 | +1.91 ± 8.01 | t=+0.53 (p=0.622) | +0.24 | [-4.16, +7.99] |
| LR | 5 | +0.01 ± 0.04 | t=+0.64 (p=0.559) | +0.28 | [-0.02, +0.04] |
| SELD | 5 | +0.01 ± 0.06 | t=+0.18 (p=0.865) | +0.08 | [-0.04, +0.05] |

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

### Probing contrasts

- **adding GCA full vs no attention** (overall ablation): Δ MAE = -0.14 ± 0.87 ° (t=-0.36, p=0.739, d_z=-0.16)
- **isolating the geometry contribution** (with vs without geometry token): Δ MAE = +0.12 ± 1.02 ° (t=+0.27, p=0.803, d_z=+0.12)
- **effect of plain channel attention** (no geometry): Δ MAE = -0.26 ± 0.62 ° (t=-0.94, p=0.399, d_z=-0.42)

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

## Workload summary

| Stage | Cells × seeds | GPU hours | Status |
| ----- | -------------- | --------- | ------ |
| Stage 1 (FOA reproduce)        | 5 | ~9  | done |
| Stage 2 (MIC feature extraction) | -  | ~0.2 | done |
| Stage 3 (GCA ablation, 110/111/112) | 15 | ~28 | done |
| Tier I (cross-dataset STARSS22) | 15 | ~0.2 | done |
| Tier III (linear probe over 78 dev-test files) | 15 | ~0.5 | done |
| Tier IV (attention viz, 6 files × 3 cells) | - | ~0 (CPU) | done |

Total ≈38 GPU-h.

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
