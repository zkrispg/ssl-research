# [TASLP Draft v0] When Do Geometry Priors Help Sound Event Localization? A Modality x Architecture Dissociation

**Target**: IEEE/ACM TASLP (or downgrade path: INTERSPEECH 2026 / ICASSP 2027)

**Status**: scaffolding. All experiments DONE except optional n=5 boost for the 6 cross-slot cells.
Numbers below are pulled from the actual result JSONs (paths cited per table). Cells marked
`TODO` need a value or a decision.

**Honest venue read** (see review 2026-05-29): the mechanistic dissociation + cross-dataset
replication is the strongest part and is genuinely novel. The weak absolute performance
(F1 9-13%, due to 4 GB GPU + compact official baseline) and n=3 on cross-slot cells are the
two reviewer-facing liabilities. The paper's narrative must rest on (a) the ANOVA interaction
(p=0.029), (b) the probing mechanism (p=0.037), and (c) the STARSS22 cross-dataset replication
(+8.5 deg matches in-domain +8.15 deg) -- NOT on any single marginal contrast.

---

## Title candidates (pick 1)

1. **When Do Geometry Priors Help Sound Event Localization? A Modality x Architecture Dissociation** (lead -- names the finding, frames as a "when" question reviewers like)
2. Geometry Priors Are Not Universally Beneficial for SELD: A Controlled Dissociation Study
3. The Conditional Value of Array-Geometry Priors in Deep Sound Event Localization and Detection

---

## One-sentence thesis

The benefit (or harm) of injecting hand-crafted microphone-array geometry into a channel-attention
SELD model is **not monotone**: it is jointly modulated by input modality (MIC tetrahedral vs FOA
ambisonic) and temporal architecture (CRNN+MHSA vs Transformer-only), flipping sign across the
2x2 grid, and the harm in the MIC+Transformer cell is mechanistically explained by degraded linear
decodability of direction-of-arrival from the learned representation.

---

## Abstract (~220 words target) -- DRAFT

Hand-crafted microphone-array geometry priors are widely injected into deep sound event
localization and detection (SELD) systems via attention biases, positional encodings, or
geometry-derived features, yet whether they help is rarely tested in a controlled, paired
ablation. We study a single one-bit intervention -- a geometry bias toggled on/off inside an
otherwise identical Geometry-aware Channel Attention (GCA) block -- on the official DCASE 2024
SELD baseline, across a 2x2 grid of input modality (4-channel MIC tetrahedral array vs
first-order Ambisonics) and temporal architecture (CRNN+self-attention vs Transformer-only).
Using multi-seed paired tests on the real-world STARSS23 dataset, we find a clean
**dissociation**: the geometry prior *improves* class-coupled localization error (DOAE_CD) by
7.7 deg in the FOA+CRNN cell (d_z=-1.73) but *degrades* it by 8.2 deg in the MIC+Transformer
cell (d_z=+2.38), while the MIC+CRNN and FOA+Transformer cells are null. A three-way factorial
ANOVA confirms a significant architecture x prior interaction (F=5.54, p=0.029). Linear probing
of the intermediate representation reveals the mechanism: in the harmed MIC+Transformer cell the
geometry-biased model encodes direction *less* linearly (probe MAE +0.38 deg, d_z=+2.91,
p=0.037). The MIC+Transformer harm replicates zero-shot on the independent STARSS22 dataset
(+8.5 deg, matching the +8.2 deg in-domain effect). We additionally stress-test on synthetic
TAU-NIGENS-SSE-2021 recordings. Our results argue that geometry priors should be treated as a
**conditional**, architecture-dependent design choice rather than a universal good.

---

## 1. Introduction (~1 page)

### 1.1 Motivation
- SELD = building block for far-field ASR, robot audition, AR/VR, surveillance.
- A common but under-ablated design choice: inject mic-array geometry (bias terms, side inputs,
  positional encodings, geometry-derived features).
- Two camps in the literature:
  | Camp | Stance | Examples |
  |---|---|---|
  | Geometry-aware | more priors = better | Hammer 2021, Yang 2023, many DCASE submissions |
  | Geometry-invariant | implicit phase suffices | Baek 2025 (GI-DOAEnet, TASLP), Neural-SRP 2024 |
- The dichotomy is rarely tested in a **paired** ablation; most comparisons span architectures,
  conflating geometry with capacity/modality.

### 1.2 This work
- We isolate geometry with a **one-bit toggle** inside one attention block, holding everything
  else fixed, and sweep the (modality x architecture) grid.
- We find the effect is **not monotone** -- it dissociates by cell. This reframes the debate from
  "do priors help?" to "when do priors help?".

### 1.3 Contributions
1. A **2x2 (modality x architecture) dissociation** of a geometry-prior's effect on SELD,
   established by multi-seed paired ablation on real recordings (STARSS23).
2. A **formal interaction test** (3-way factorial ANOVA) confirming the prior's effect is
   architecture-dependent (arch x prior, F=5.54, p=0.029).
3. A **mechanistic explanation** via linear probing: the harmful cell encodes DOA less linearly
   (probe MAE +0.38 deg, d_z=+2.91, p=0.037) -- the prior actively distorts the representation.
4. **Cross-dataset replication** (STARSS22 zero-shot, +8.5 deg) and a **synthetic stress test**
   (TAU-NIGENS-SSE-2021).
5. An open, reproducible pipeline (78 checkpoints, multi-seed, on commodity 4 GB GPU).

---

## 2. Method (~1 page)

### 2.1 Backbone (DCASE 2024 official baseline)
- Multi-ACCDDOA output (13 classes x 3 tracks x (activity + 3-axis DOA + distance)).
- MIC variant: 4 log-mel + 6 GCC-PHAT channels. FOA variant: 4 log-mel + 3 intensity-vector channels.
- Conv stack (t_pool=[5,1,1]) -> temporal model -> per-frame Multi-ACCDDOA heads.
- Synthetic-pretrained init (3_1 FOA / 6_1 MIC weights), 60-epoch fine-tune.

### 2.2 Geometry-aware Channel Attention (GCA)
- Single-head attention over the **channel/mic dimension** with an additive geometry bias
  `G_K[i,j]` derived from pairwise array geometry.
- **Toggle**: `geometry_bias=False` removes `G_K` -> plain SE-style channel attention (`no_geom`).
  This is the one-bit intervention. `use_gca=False` is the no-attention control.
- FOA case (Sec 2.4): geometry of ambisonic channels (W/X/Y/Z) defined by directional patterns
  (`foa_ambisonic_pair_geometry`), not physical mic coordinates.

### 2.3 Temporal architecture toggle
- `gru_mhsa`: BiGRU + 2-layer multi-head self-attention (default DCASE baseline).
- `transformer`: linear projection + 4-layer TransformerEncoder (norm_first). Same conv stack,
  same heads -- only the temporal model changes.

### 2.4 The 2x2 grid (12 task IDs)
| Cell | Modality | Arch | full | no_geom | no_gca control |
|---|---|---|---|---|---|
| MIC+CRNN | MIC | gru_mhsa | 110 | 111 | 112 (+113 vanilla-SE) |
| FOA+CRNN | FOA | gru_mhsa | 130 | 131 | 100 (no-GCA repro) |
| MIC+Xfm  | MIC | transformer | 141 | 142 | 140 |
| FOA+Xfm  | FOA | transformer | 151 | 152 | 150 |

### 2.5 Multi-seed paired-test protocol
- MIC+CRNN: n=5 seeds. Cross-slot cells (FOA+CRNN, MIC+Xfm, FOA+Xfm): n=3 (optionally n=5).
- Paired t-test + Wilcoxon + Cohen's d_z + bootstrap 95% CI on seed-matched deltas.
- Headline single test: 3-way factorial ANOVA (Type-II SS) on per-seed metrics.

---

## 3. Experiments

### 3.1 Datasets
- **STARSS23** (DCASE 2024 Task 3 dev): in-domain train + test. MIC + FOA, 13 classes.
- **STARSS22** (DCASE 2022 Task 3 dev-test, 54 clips): zero-shot cross-dataset (MIC; FOA audio
  unavailable -- see Limitations). Distance-blind matching (lad_dist_thresh=inf).
- **TAU-NIGENS-SSE-2021** (synthetic, same lab SRIRs, 12 classes): zero-shot synthetic stress
  test. 7 overlapping classes remapped to STARSS taxonomy; `kept_*` metrics restrict to those 7.

### 3.2 Metrics
DCASE 2024 Task 3: F_20deg, DOAE_CD (deg), LR_CD, RDE_CD, Dist_err, SELD score. Probe metric:
linear-ridge angular MAE (deg) decoding (sin/cos az, sin/cos el) from pooled post-conv features.

### 3.3 MAIN RESULT -- the 2x2 dissociation
**Table 1** [source: `paper/path_c_2x2.md`, `paper/path_c_2x2.json`]
Headline metric DOAE_CD, paired (GCA full - GCA no_geom), seed-matched:

| Cell | n | DOAE full | DOAE no_geom | delta DOAE | t (p) | d_z | bootstrap 95% CI | direction |
|---|---|---|---|---|---|---|---|---|
| MIC + CRNN | 5 | 45.38 | 46.26 | **-0.88** | -0.15 (0.887) | -0.07 | [-11.26, +9.27] | null |
| FOA + CRNN | 3 | 39.79 | 47.51 | **-7.72** | -3.00 (0.095) | -1.73 | [-10.40, -2.58] | prior HELPS |
| MIC + Xfm  | 3 | 50.17 | 42.02 | **+8.15** | +4.13 (0.054) | +2.38 | [+4.75, +11.59] | prior HURTS |
| FOA + Xfm  | 3 | 40.42 | 37.86 | **+2.55** | +0.61 (0.602) | +0.35 | [-4.08, +10.24] | null |

**Figure 1** [source: `paper/figs/path_c_2x2_dissociation.png`] -- the 2x2 dissociation plot.

### 3.4 Formal interaction test
**Table 2** [source: `paper/path_c_2x2_anova.md`] -- 3-way factorial ANOVA on DOAE_CD (n_obs=28):
- modality (MIC vs FOA): F=4.44, p=0.048 *
- **arch x prior interaction: F=5.54, p=0.029 *** <- the key statistic
- 3-way modality x arch x prior: F=0.02, p=0.881 (n.s. -- the dissociation is mostly a 2-way
  arch x prior effect, modulated by modality magnitude)
- On F1: strong modality (F=82.6 ***), arch (F=24.6 ***), modality x arch (F=11.8 **) main effects.

### 3.5 Mechanism -- linear probing
**Table 3** [source: `paper/path_c_probe.md`, `paper/path_c_probe.json`] -- angular probe MAE (deg):

| Cell | full MAE | no_geom MAE | delta (full-no_geom) | t (p) | d_z |
|---|---|---|---|---|---|
| MIC + CRNN | 28.50 | 28.37 | +0.12 | +0.27 (0.803) | +0.12 |
| FOA + CRNN | 20.96 | 20.98 | -0.02 | -0.06 (0.956) | -0.04 |
| **MIC + Xfm** | 26.54 | 26.16 | **+0.38** | **+5.05 (0.037)** | **+2.91** |
| FOA + Xfm | 18.05 | 18.21 | -0.16 | -0.37 (0.745) | -0.21 |

Reading: in the only cell where the prior HURTS SELD (MIC+Xfm), it ALSO makes DOA less linearly
decodable from the representation (significant, d_z=+2.91). The information-encoding delta and the
SELD delta point the same way -> the prior distorts, not merely fails to help. Other cells: probe
delta is null, consistent with their null/helpful SELD deltas.

### 3.6 Cross-dataset replication (STARSS22 zero-shot)
**Table 4** [source: `paper/path_c_cross_starss22.md`] -- MIC cells, GCA full - no_geom, DOAE:
- MIC + CRNN: delta = -4.77 deg (d_z=-0.42) -- prior mildly helps generalization (differs from
  in-domain null; discuss).
- **MIC + Xfm: delta = +8.53 deg (d_z=+2.11)** -- REPLICATES in-domain +8.15 deg almost exactly.

The MIC+Xfm harm is dataset-robust; this is the single most convincing external-validity result.

### 3.7 Synthetic stress test (TAU-NIGENS-SSE-2021)
**Table 5** [source: `paper/path_c_synth_nigens.md`] -- FOA cells, kept-class subset, zero-shot.
Mostly null on F1/LE (d_z 0.04-0.58); FOA+Xfm SELD trends toward no-GCA (d_z=-1.96 vs no_gca,
p=0.077). Frame as boundary condition: on a heavy synthetic domain shift the fine-grained
prior effect washes out, while the broad modality/arch effects persist. (MIC synthetic cells
pending mic_dev.z01 download -> add if finished.)

### 3.8 Per-class breakdown
**Table 6 / Figure 2** [source: `paper/path_c_per_class.md`, `paper/figs/path_c_per_class_*.png`]
13 cells x 13 classes. Speech/music/domestic classes carry most F1; rare classes (doorOpen,
clapping) near zero across all cells (detection-limited, not localization-limited). Use to argue
the dissociation is a localization-representation effect, not a detection artifact.

### 3.9 Supporting ablations (from progress doc)
- Vanilla SE-block (113) vs GCA no_geom (111): isolates Q/K/V-over-mics from MLP-gate attention.
- Data-fraction sweep (25/50/100%): [source: `paper/path_c_data_fraction.md`] -- prior effect
  vs training-data size; earlier d_z=-2.45 at 25% was sampling noise (corrected with n=5).
- Attention-map visualizations [source: `paper/figs/path_c_attn_*.png`].

---

## 4. Discussion (~3/4 page)

- **Why MIC+Transformer is harmed**: the Transformer's global token mixing already recovers
  inter-channel structure; an additive geometry bias over-constrains the channel attention toward
  a fixed pattern that mismatches real-room reverberation, measurably reducing DOA decodability.
- **Why FOA+CRNN is helped**: FOA channels carry directional info in a fixed analytic basis;
  the geometry bias aligns the recurrent model's limited mixing capacity with that basis.
- **Cross-dataset nuance**: MIC+CRNN prior helps generalization on STARSS22 (-4.77 deg) though
  null in-domain -- mild regularization effect under domain shift. MIC+Xfm harm is robust.
- **Takeaway for practitioners**: do not add geometry priors by default; their value is
  architecture- and modality-conditional. Ablate before shipping.

## 5. Limitations (write honestly -- pre-empts reviewers)
1. **Absolute performance is low** (F1 9-13%, DOAE 36-50 deg): compact official baseline on a
   4 GB GPU, not a SOTA system. We study a *controlled ablation*, not the performance frontier;
   the one-bit intervention holds capacity/data/recipe fixed, so relative effects are
   interpretable even at this scale. (State explicitly; cite baseline parity in Stage-1 repro:
   F_20deg 13.06% vs official 13.10%.)
2. **n=3 on cross-slot cells**: single contrasts are marginal (p=0.054-0.095); the claim rests
   on the ANOVA interaction + probing + cross-dataset replication, not individual t-tests.
   (Optional n=5 boost ~30 GPU-h closes this.)
3. **STARSS22 FOA unavailable** (audio deleted for disk); FOA external validity rests on
   synthetic NIGENS only. Asymmetric.
4. Single backbone family; results may not transfer to fundamentally different SELD paradigms
   (e.g. EINV2, CST-Former). We argue the *phenomenon* (conditional prior value) generalizes
   even if magnitudes differ.

## 6. Conclusion
Geometry priors are a conditional design choice. We provide the first paired 2x2 dissociation,
a formal interaction test, a probing-based mechanism, and cross-dataset replication.

---

## Figures/Tables -> artifact map (for assembling the camera-ready)

| Item | Artifact | Status |
|---|---|---|
| Tab 1 (2x2 dissociation) | `paper/path_c_2x2.{md,json}` | DONE |
| Fig 1 (2x2 plot) | `paper/figs/path_c_2x2_dissociation.png` | DONE |
| Tab 2 (ANOVA) | `paper/path_c_2x2_anova.{md,json}` | DONE |
| Tab 3 (probing) | `paper/path_c_probe.{md,json}` | DONE |
| Tab 4 (STARSS22 cross) | `paper/path_c_cross_starss22.{md,json}` | DONE (merged 2026-05-29) |
| Tab 5 (synth NIGENS) | `paper/path_c_synth_nigens.{md,json}` | DONE (FOA); MIC pending dl |
| Tab 6 / Fig 2 (per-class) | `paper/path_c_per_class.{md,json}`, `figs/path_c_per_class_*.png` | DONE |
| Fig 3 (attention maps) | `paper/figs/path_c_attn_*.png` | DONE |
| Supp (data fraction) | `paper/path_c_data_fraction.{md,json}` | DONE |
| Stage-1 repro parity | `paper/path_c_progress_v2.md` Stage 1 | DONE |

## Open decisions before submission
- [ ] Venue: TASLP (needs n=5 boost + ideally MIC synthetic) vs INTERSPEECH/ICASSP (current scope OK).
- [ ] Run optional n=5 boost on 6 cross-slot cells (~30 GPU-h)?
- [ ] Finish mic_dev.z01 download -> MIC synthetic cells in Table 5.
- [ ] Related-work positioning vs Baek 2025 (GI-DOAEnet) and DCASE 2024 baseline paper.
- [ ] Decide title (#1 lead).
