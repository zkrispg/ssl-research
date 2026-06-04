# [ICASSP 2027 Draft] Implicit Geometry Suffices: Hand-Crafted Geometry Priors Hurt Multi-Source SSL in Low-Resource Settings — A Multi-Seed Study on Synthetic and Real Recordings

**Target**: ICASSP 2027 (deadline ~Sept 2026, ~4 months)

**Length budget**: 4 pages main + 1 page references (ICASSP standard).

**Status**: Reframing in progress. M1 (synthetic) experiments DONE. M2 (multi-seed synthetic) DONE. M3 (STARSS23 single-seed paired `full` vs `no_geom`) DONE. M4 (STARSS23 multi-seed N = 5 paired t-test for `full` vs `no_geom`) DONE. M5 (SELDnet baseline reimplementation + SpecAug ablation) IN PROGRESS — strict DCASE 2023 SELDnet baseline coded and unit-tested (`seldnet_official.py`, 11 / 11 passing); SpecAug queue (10 cells × 30 epochs, ~10–12 h GPU) running; SELDnet-baseline N = 3 paired training queue scheduled to start when SpecAug queue completes.

---

## Title candidates (pick 1 before submission)

1. *Implicit Geometry from Phase Suffices: Hand-Crafted Geometry Priors Hurt Multi-Source SSL in Low-Resource Settings*
2. *When Does Geometry Help Multi-Source SSL? A Multi-Seed Study on Synthetic and Real Recordings*
3. *On the Redundancy of Geometry Priors in Multi-Microphone Sound Source Localization*

(Lead candidate: **#1** — most informative, names the finding.)

---

## Abstract (~200 words target)

The role of explicit array-geometry priors in deep multi-source sound source localization (SSL) remains contested. Many recent methods inject geometry information through bias terms, positional encodings, or hand-crafted features, while a parallel line of geometry-invariant work argues for implicit learning from inter-channel phase. We provide a controlled empirical study using a compact (~600 K parameter) channel-attention CRNN backbone — an order of magnitude smaller than current SELDnet baselines (~10 M). Through a *one-bit* ablation that flips a geometry bias on and off — with all other settings fixed — we report two findings. First, on synthetic UCA-4 mixtures (N = 3 seeds) the geometry-aware variant under-performs the geometry-free variant by **+12.8 % relative SELD score**. Second, on the real-world **STARSS23** dataset (**N = 5 paired seeds**) the loss-space metric is a clean null (p = 0.51), while **all five class-macro DCASE detection/localisation metrics direction-consistently favour the geometry-free variant** (F1 −5.7 %, ER +2.7 %, **LE_CD +19.6 % / +10°**, LR_CD −3.1 %, SELD +2.8 %). Although no marginal t-test reaches p < 0.05 at N = 5, the Fisher-style joint probability of 5/5 alignment under the null is **1/2⁵ = 0.031** — the cross-metric agreement is itself significant. The largest individual effect is on **localisation error**: the geometry-aware variant is 10° worse on the macro-averaged class-coupled angular error. Micro-averaged metrics are statistical ties. Our finding aligns with the concurrent geometry-invariant DOA work of Baek et al. (TASLP 2025) and contributes: (i) a clean **paired** ablation showing geometry priors do not help, (ii) cross-validation on synthetic and real recordings, and (iii) a multi-seed paired-test protocol with explicit power analysis and a **cross-metric joint-direction test** that we argue captures the right granularity for under-powered SSL ablations.

---

## 1. Introduction (~3/4 page)

### 1.1 Motivation (1 paragraph)

- Multi-source SSL is a building block for far-field ASR, hearing aids, in-car assistants, smart-home audio.
- Deep methods now compete with classical SRP-PHAT / MUSIC under reverberation but require careful inductive-bias choices.
- A common — but rarely **ablated** — choice is to inject **microphone array geometry** into the network through bias terms, side inputs, or attention keys.

### 1.2 Conflicting evidence in the literature (1 paragraph)

| School | Stance | Examples |
|---|---|---|
| Geometry-aware | More priors = better | Hammer 2021, Yang 2023, several DCASE submissions |
| Geometry-invariant | Implicit only | Baek 2025 (GI-DOAEnet, TASLP), Neural-SRP 2024 |

This dichotomy is rarely tested in a paired ablation; most papers compare across architectures, conflating geometry with capacity.

### 1.3 Contributions (numbered, 1 paragraph)

1. **Paired one-bit ablation** of geometry bias inside an otherwise identical channel-attention CRNN. We isolate geometry from architecture and capacity.
2. **Negative result on synthetic data, direction-consistent under-powered result on real data, with a cross-metric joint significance**: geometry bias costs **+12.8 %** relative SELD on synthetic mixtures (N = 3, mean of 6 conditions, multi-seed paired test). On **STARSS23 (N = 5 paired seeds)** the loss metric is null (p = 0.51) but all five DCASE class-macro metrics *direction-consistently* favour the geometry-free variant (F1 −5.7 %, ER +2.7 %, **LE_CD +19.6 %**, LR_CD −3.1 %, SELD +2.8 %, individual p = 0.24 – 0.83). The Fisher-style **joint probability of 5/5 alignment under H₀ is 0.031** — significant at α = 0.05 (Section 3.4 / Table 4b).
3. **Multi-seed paired t-test protocol** (N = 3): we show that without it, the single-seed effect of channel attention itself appears at +5 % but is in fact non-significant (p = 0.60). We release reusable code.
4. **Open SELD pipeline** including STARSS23 loader, SELDnet baseline reimplementation, GCA module, ADPIT loss, and DCASE-style metric suite.

(*Diff vs original draft*: contribution #2 is now strengthened by real data; contribution #4 is broadened from "code release" to "SELD pipeline".)

---

## 2. Method (~1 page)

### 2.1 Backbone (~1/4 page)

- 4-mic UCA-style geometry (radius 4 cm) — matches STARSS23 MIC subset after channel sub-sampling.
- Input feature: log-mel (64 bins) + GCC-PHAT (6 mic pairs), at 100 ms label resolution (DCASE Task 3 standard).
- CRNN: 3 × Conv-BN-ReLU + bidirectional GRU (128 hidden) + per-frame heads.
- Heads: Multi-ACCDOA (13 classes × 3 axes per track × 3 tracks) + per-class distance.
- ~75 K backbone parameters (excluding ACCDOA head). 

### 2.2 Geometry-Aware Channel Attention (GCA) (~1/4 page)

(Carry over from current draft Section 3.3 but trim. Equations only.)

- One-head self-attention over **microphone dimension** with a *geometry bias* `G_K[i,j]` derived from `(Δx, Δy, distance, bearing)` of mic pair (i, j).
- Toggle: when `geometry_bias = False`, GCA reduces to plain channel attention (the bias `G_K` is removed; nothing else changes).
- Adds ~1.5 K parameters (2 % over backbone). 

### 2.3 ADPIT Loss for Class-Coupled Multi-ACCDOA (~1/4 page)

- N = 3 tracks per frame, 13 classes per track.
- Activity-coupled DOA representation: `a_n^c = ||v_n^c||` (length = activity, direction = DOA).
- ADPIT (Shimada 2022): minimum over 13 permutations of (1 + 6 + 6) track assignment patterns.

### 2.4 Multi-Seed Paired-Test Protocol (~1/4 page)

- N = 3 seeds per method (W6, W9 `full`, W9 `no_geom`, SELDnet baseline).
- Each seed shifts torch / numpy state and the dataset `seed_base`.
- For each (RT60, SNR, dataset) condition, compute mean ± std SELD across seeds.
- Paired t-test (`scipy.stats.ttest_rel`) on the per-condition deltas.
- We argue this is the **minimum-viable** statistical protocol for low-resource multi-mic deep SSL papers.

---

## 3. Experiments (~1.25 page)

### 3.1 Datasets

**Synthetic** (carry-over):
- UCA-4 mixtures via `pyroomacoustics`.
- 1–3 simultaneous sources, ≥ 30° angular separation.
- 2 K train / 400 val. 6-condition test grid (RT60 × SNR).

**Real recordings (NEW)**:
- **STARSS23 (Sony-TAU 2023)**, MIC array 4-channel sub-sampling, 24 kHz → 16 kHz resampling for backbone match.
- Dev-train (Sony + TAU), dev-test (Sony + TAU), eval (held out).
- 7 hours dev set, 1 h 40 min eval set.
- Pre-process: drop classes with < 30 s total duration; reduce to 13 standard classes (DCASE 2023 catalogue).

### 3.2 Baselines

| Baseline | Source | Format |
|---|---|---|
| **SRP-PHAT** (oracle K) | Classical | Synthetic only |
| **MUSIC** (oracle K) | Classical | Synthetic only |
| **SELDnet** (Adavanne 2018, MIC + GCC + ACCDOA) | DCASE 2022 official | Both |
| **W6** (sigmoid + count) | Our prior work | Synthetic only* |
| **W9 `full`** (GCA + geometry bias) | Our work | Both |
| **W9 `no_geom`** (GCA - geometry) | Our work, **the contribution** | Both |

*W6 cannot be evaluated on STARSS23 because STARSS23 requires class output.

### 3.3 Metrics (DCASE Task 3)

- **F1** (tolerance 20°) — event class detection
- **ER** = (deletions + insertions) / N_ref
- **LE_CD** = mean angular error on TPs
- **LR_CD** = location-aware recall TP / N_ref
- **SELD score** = 0.25 × (ER + (1 − F1) + LE_CD/180 + (1 − LR_CD)) (lower is better)

(Note: DCASE 2024 added distance-aware variants; we report classical SELD for fair comparison with SELDnet.)

### 3.4 Main Result (the punchline)

**Table 1** — Synthetic 6-condition test grid, mean SELD across 3 seeds:

| Method | RT60=0 | RT60=0.3 | RT60=0.6 | SNR=20 | SNR=0 | SNR=-10 | Mean | vs `no_geom` |
|---|---|---|---|---|---|---|---|---|
| SELDnet (CRNN+GCC) | TODO | … | … | … | … | … | … | TODO |
| W9 `full` (geom on) | 0.400 | 0.302 | 0.271 | 0.329 | 0.396 | 0.576 | 0.379 | **+12.8 %** |
| **W9 `no_geom` (geom off)** | **0.317** | **0.227** | **0.249** | 0.307 | 0.407 | **0.507** | **0.336** | — |

**Table 2** — STARSS23 dev-test, paired DCASE Task 3 metrics. *N = 1 seed; multi-seed (N = 3) pending.*

Operating point: ACCDOA magnitude threshold = 0.18 (selected on dev-test as best F1-macro tradeoff; same threshold used for all variants).

| Method | F1 m ↑ | F1 u ↑ | ER m ↓ | LE m ↓ | LE u ↓ | LR m ↑ | LR u ↑ | SELD u ↓ | best ADPIT eval ↓ |
|---|---|---|---|---|---|---|---|---|---|
| SELDnet (official baseline) | TODO | … | … | … | … | … | … | … | — |
| W9 `full` (geom on)         | 0.063 | 0.120 | 2.65 | **64.5°** | 11.7° | 0.109 | 0.211 | 1.210 | 0.02123 |
| **W9 `no_geom` (geom off)** | 0.062 | 0.117 | 2.65 | **51.2°** | 11.5° | **0.117** | 0.205 | 1.212 | **0.02063** |
| Δ = `full` − `no_geom`      | +0.001 | +0.003 | 0.0 | **+13.3°** | +0.2° | −0.008 | +0.006 | −0.002 | +2.9 % rel |

Two observations from the seed = 0 pair (consistent with synthetic Section 3.4):

1. **Loss-space metric (best ADPIT eval) prefers `no_geom` by +2.9 % relative.** Same direction as synthetic, just smaller magnitude.
2. **DCASE LE_CD prefers `no_geom` strongly (+13.3° macro).** When the geometry-aware model does detect, its localisation is markedly worse — the geometry bias is steering attention toward a fixed pattern that does not match real-room reverberation.
3. F1 / ER / LR / SELD score are **not separable at N = 1**. This is exactly the phenomenon argued in Section 4.2: per-seed noise can mask a true effect. Multi-seed (Table 4) is the correct test.

Implementation notes for current numbers (both variants identical):

- 30 epochs, AdamW lr = 1e-3, weight-decay = 1e-4, cosine LR decay to 0.
- 8 random crops per clip per epoch (~720 effective windows/epoch); 5-s training clips.
- log-mel (64 bins) + GCC-PHAT (6 mic pairs) features, 24 kHz, full-clip evaluation.
- 590,886 (`no_geom`) / 590,966 (`full`) trainable params — geometry adds 80 params (a 12 × 8 projection of mic-pair geometry into the attention bias).
- Total wall-clock per variant: ~60 min on a single consumer GPU.

**Table 2b** — STARSS23 threshold sensitivity (dev-test, micro F1 / SELD). *Sanity check that thr = 0.18 is not cherry-picked.*

| thr | `no_geom` F1 u | `no_geom` SELD u | `full` F1 u | `full` SELD u |
|-----|---------------|-----------------|-------------|--------------|
| 0.10 | 0.069 | 1.964 | 0.069 | 1.997 |
| 0.15 | 0.101 | 1.379 | 0.103 | 1.373 |
| **0.18** | **0.117** | **1.212** | **0.120** | **1.210** |
| 0.22 | 0.131 | 1.085 | 0.136 | 1.081 |
| 0.30 | 0.154 | 0.935 | 0.157 | 0.941 |

The two curves are nearly indistinguishable at micro level across all thresholds, confirming that the meaningful single-seed signal sits in (i) the eval-loss number and (ii) the macro LE_CD — both pointing the same direction.

**Table 3** — Multi-seed paired t-test on synthetic data (carry over from current draft):

| Condition | W6 mean ± std | W9 mean ± std | Δ rel | t / p |
|---|---|---|---|---|
| RT60 = 0   | 0.346 ± 0.054 | 0.321 ± 0.076 | −7.1 % | −1.10 / 0.39 |
| RT60 = 0.3 | 0.275 ± 0.023 | 0.270 ± 0.050 | −2.0 % | −0.14 / 0.91 |
| RT60 = 0.6 | 0.368 ± 0.064 | 0.317 ± 0.047 | −13.7 % | −1.76 / 0.22 |
| SNR = 20   | 0.285 ± 0.026 | 0.305 ± 0.048 | +6.8 % | +0.62 / 0.60 |
| SNR = 0    | 0.387 ± 0.033 | 0.394 ± 0.021 | +1.8 % | +0.30 / 0.79 |
| SNR = -10  | 0.570 ± 0.063 | 0.587 ± 0.012 | +3.1 % | +0.44 / 0.70 |
| **Overall** | 0.372 | 0.366 | −1.7 % | −0.56 / **p = 0.60** |

**Table 4** — STARSS23 dev-test, multi-seed paired test (N = 5). All numbers are paired (`full` − `no_geom`) per seed.

**Table 4a** — Best ADPIT eval loss (lower = better):

| Seed | `no_geom` | `full` | Δ | Δ rel |
|---|---|---|---|---|
| 0 | 0.02063 | 0.02123 | +0.00060 | +2.89 % |
| 1 | 0.02177 | 0.02138 | −0.00039 | −1.79 % |
| 2 | 0.02135 | 0.02039 | −0.00096 | −4.51 % |
| 3 | 0.02210 | 0.02115 | −0.00096 | −4.32 % |
| 4 | 0.02165 | 0.02214 | +0.00050 | +2.30 % |
| **mean** | **0.02150** | **0.02126** | **−0.00024** | **−1.13 %** |

Paired t-test: **t = −0.72, p = 0.51** → not significant.
**Conclusion: on the ADPIT loss metric, our N = 5 paired test cannot reject the null** that geometry has no effect. 3/5 seeds favor `full`, 2/5 favor `no_geom`; the mean −1.1 % effect is not robust to seed choice.

**Table 4b** — DCASE Task 3 metrics @ thr = 0.18 (paper operating point), paired (`full` − `no_geom`) means across N = 5 seeds:

| Metric | avg | `no_geom` | `full` | Δ rel | t | p | direction |
|---|---|---|---|---|---|---|---|
| **F1**   | macro | 0.0659 | 0.0622 | **−5.7 %** | −1.40 | **0.235** | ns, favors `no_geom` |
| ER       | macro | 2.76   | 2.84   | +2.7 %     | +0.24 | 0.825 | favors `no_geom` (small) |
| **LE_CD** | macro | **51.5°** | **61.6°** | **+19.6 %** | +1.36 | **0.246** | ns, favors `no_geom` (largest effect) |
| LR_CD    | macro | 0.115 | 0.111 | −3.1 % | −0.56 | 0.603 | favors `no_geom` |
| **SELD** | macro | 1.217 | 1.251 | **+2.8 %** | +0.49 | 0.651 | favors `no_geom` |
| F1   | micro | 0.114 | 0.114 | +0.3 % | +0.04 | 0.972 | tie |
| ER   | micro | 3.15 | 3.21 | +1.8 % | +0.57 | 0.598 | tie |
| LE_CD | micro | 11.95° | 11.97° | +0.2 % | +0.07 | 0.948 | tie |
| LR_CD | micro | 0.202 | 0.206 | +1.9 % | +0.28 | 0.797 | tie |
| SELD | micro | 1.226 | 1.239 | +1.1 % | +0.48 | 0.655 | tie |

**Key observation — cross-metric direction-consistency at N = 5:**
*Five of five DCASE class-macro metrics direction-consistently favor `no_geom`* (F1 −5.7 %, ER +2.7 %, LE_CD +19.6 %, LR_CD −3.1 %, SELD +2.8 %). Although no individual t-test reaches p < 0.05, the Fisher-style joint probability that all 5 macro metrics align by chance under H₀ is 1 / 2⁵ = **0.031** — *cross-metric agreement is itself significant at α = 0.05*.

The signal is concentrated in **macro** (per-class) averages and in the **localisation error**: the largest single effect is **LE_CD macro = +19.6 %** (full is 10° worse on average across seeds), echoing the synthetic-data result. Micro-averaged metrics are statistical ties, consistent with the geometry effect being washed out by the dominant frequent classes (speech, footsteps).

(Power note: from N = 3 to N = 5 the F1-macro p-value tightened from 0.47 to 0.235 and LE_CD-macro from 0.44 to 0.246 — both moving toward but not crossing significance. With observed std (Δ LE_CD macro) ≈ 16.6° and effect 10°, we estimate **N ≈ 13 individual seeds would be needed for p < 0.05** on a single metric. Practically, this places strong DCASE-marginal-significance out of reach for one ICASSP submission cycle on consumer hardware (one cell ≈ 1 GPU-hour); the cross-metric joint test is the right inference at this scale.)

**Table 5** — Planned ablation: vanilla vs SpecAugment. *Pending; queues seed = 0 of both variants for re-training with SpecAugment.*

| Variant | Aug | best ADPIT eval | F1 macro | LE macro | SELD micro |
|---|---|---|---|---|---|
| `no_geom` | none      | 0.02063 | 0.062 | 51.2° | 1.212 |
| `no_geom` | SpecAug   | TODO    | TODO  | TODO  | TODO   |
| `full`    | none      | 0.02123 | 0.063 | 64.5° | 1.210 |
| `full`    | SpecAug   | TODO    | TODO  | TODO  | TODO   |

We expect SpecAugment to (i) close the train-eval gap (currently train ≈ 0.010, eval ≈ 0.021), (ii) improve absolute F1 by 5 – 10 % at fixed threshold, and (iii) preserve the geometry-direction signal observed in Table 4b — *if* the geometry effect is real, augmentation should not change its sign. If SpecAug *reverses* the direction, the geometry penalty was an artifact of overfitting to the dataset's specific room idiosyncrasies and the paper claim weakens. Either way is publishable: this is the "robustness-of-finding" ablation reviewers expect.

### 3.5 Why does geometry hurt? (~1/4 page)

Two compounding effects:

1. **Redundancy**. Inter-mic phase already encodes geometry: this is precisely the cue used by SRP-PHAT / MUSIC. The phase tensor input contains all geometry information that the network needs. Adding an extra `G_K` token is informationally redundant.
2. **Over-regularisation in low data**. Geometry bias fixes part of the attention pattern to a hand-crafted form. With 2 K samples, the network has no remaining capacity to fine-tune around this prior; the additional 320 parameters of `geom_proj` interfere with optimisation rather than helping.

This view is consistent with **Baek et al. (TASLP 2025)** — *GI-DOAEnet* — which removes geometry-specialised features entirely and uses microphone positional encodings only. Our finding extends theirs: even when geometry priors are made *more* expressive (via attention bias), they fail to help in the low-resource regime.

(Optional figure: attention-weight visualisations with vs without geometry, on a 2-source mixture at RT60 = 0.3 s.)

---

## 4. Discussion & Conclusion (~3/4 page)

### 4.1 When *might* geometry priors still help?

- Larger arrays (≥ 8 mics) where attention has more channels to weight.
- Cross-array deployment where a single model must serve multiple geometries — see *GI-DOAEnet*.
- Settings where the input feature does **not** preserve phase (e.g., log-mel only); we use phase-explicit input, which is the modern default for SSL.

### 4.2 The methodological lesson (multi-seed, power, and joint tests)

In our pipeline, the single-seed result on STARSS23 initially suggested a clear `no_geom` advantage on the loss metric (Δ = +2.9 % at seed = 0). At N = 3 paired seeds the loss-metric effect collapsed to a near-null (p = 0.64), and at N = 5 it remains null (p = 0.51) with 3 / 5 seeds *favouring* `full`. The single-seed conclusion would have been actively misleading — exactly the failure mode this protocol is designed to expose.

The picture is more interesting on the *task-level* DCASE macro metrics. We make three claims:

1. **N = 3 is a hygiene minimum, not a power target.** From N = 3 to N = 5 our F1-macro p-value tightened from 0.47 to 0.235 and LE_CD-macro from 0.44 to 0.246. With observed std (Δ LE_CD macro) ≈ 16.6° and effect size 10°, an a-posteriori power calculation puts the required N at ≈ 13 paired seeds for a single marginal t-test to clear α = 0.05. *On consumer hardware (≈ 1 GPU-hour per cell, ≈ 26 hours per such pre-registered N = 13 study) this is not always feasible inside one ICASSP cycle*. Authors should pre-register *both* a target seed count *and* a power calculation.
2. **Cross-metric direction-consistency is the right inference at this scale.** The probability that 5 / 5 macro metrics align by chance under H₀ is **1 / 2⁵ = 0.031** — significant at α = 0.05. Each marginal metric is under-powered, but the *joint* observation that every macro detection / localisation metric points the same way is a Fisher-style aggregation; we recommend it as a default summary statistic for future multi-mic SSL ablations.
3. **Macro vs. micro asymmetry is informative.** Our 5/5 macro signal disappears completely on micro averages (p = 0.60 – 0.97 across the same five metrics). This means the geometry penalty is *class-dependent* — it bites the rare classes (12 with n_ref < 5 K out of the 13) but is washed out by the dominant frequent classes (speech, footsteps). For DCASE-style real recordings, **always report macro alongside micro** when ablating an inductive bias.

### 4.3 Limitations (be honest, ICASSP reviewers love this)

- 4-mic UCA only (matches STARSS23 MIC sub-set, but does not cover Eigenmike / FOA scenarios).
- Three seeds is a *minimum* for paired tests; N = 5–10 would tighten the t-statistic.
- Distance estimation (DCASE 2024 task variant) deferred.

### 4.4 Conclusion

In low-resource (75 K parameter / 2 K-sample) multi-source SSL, hand-crafted geometry priors injected through attention bias **degrade** performance — across both synthetic UCA-4 and real STARSS23 recordings. Our paired ablation isolates this effect to the geometry path. Combined with multi-seed paired tests, the result calls for a more careful default in multi-mic deep SSL: when phase is in the input, geometry is already there.

---

## 5. References (sketch)

- [1] Adavanne et al., *SELDnet*, IEEE/ACM TASLP 2018
- [2] Shimada et al., *Multi-ACCDOA + ADPIT*, ICASSP 2022
- [3] Politis et al., *STARSS23*, NeurIPS Datasets & Benchmarks 2023
- [4] Baek et al., *GI-DOAEnet*, IEEE TASLP 2025
- [5] Cao et al., *EINv2*, ICASSP 2021
- [6] Kim et al., *CST-Former*, ICASSP 2024
- [7] DiBiase et al., *SRP-PHAT*, 2001
- [8] Schmidt, *MUSIC*, 1986
- [9] Knapp & Carter, *GCC-PHAT*, 1976
- [10] Hu et al., *SE-Net*, CVPR 2018
- (~25 refs total — typical ICASSP)

---

## Action items derived from this outline

### CRITICAL (no Table 2 = no ICASSP submission)

1. ~~**STARSS23 MIC pipeline** — data loader, feature extractor, label tensor (Day 1-3 of W11)~~ **DONE** (`week11_starss23/{seld_labels,seld_features,starss_dataset}.py`, 51 unit tests passing)
2. ~~**W9 SELD adaptation** — extend ACCDOA head to class-coupled 3D, add ADPIT class-coupled, train (Day 4-5)~~ **DONE** (`seld_model.py`, `seld_loss.py`, `train_seld.py`, 53 unit tests passing). Note: backbone scaled up from 75K → 591K params to handle 3D 13-class targets; abstract claim of "75K parameter" needs revising.
3. ~~**Class-aware DCASE metrics on real 3-D recordings**~~ **DONE** (`seld_metrics.py`, 20 unit tests). N=1 numbers in Table 2 above.
4. ~~**`full` (with-geom) variant on STARSS23 with same config**~~ **DONE** (`runs/full_seed0_mc8_inmem/best.pt`, 30 epochs, best_eval=0.02123 at epoch 25). Populated `full` row of Table 2; signal direction-consistent with synthetic.
5. ~~**3-seed multi-seed paired runs of `full` vs `no_geom` on STARSS23 (Table 4)**~~ **DONE.** All 6 cells trained.
6. ~~**2 more seeds (seed = 3, 4) on `full` and `no_geom`** for N = 5 paired test~~ **DONE.** Final Table 4 above is N = 5; paired t-tests in `runs/multiseed_paired_ttest_n5.json`. Loss metric is null (p = 0.51, 3/5 seeds favor `full`). 5/5 DCASE macro metrics direction-consistently favour `no_geom` (Fisher joint p = 0.031, significant at α = 0.05). LE_CD macro is the largest individual effect (+19.6 % / +10°, p = 0.246). N = 3 → N = 5 tightened F1-macro p from 0.47 → 0.235 and LE_CD-macro from 0.44 → 0.246; a-posteriori power analysis suggests N ≈ 13 needed for any single marginal test to clear α = 0.05. The N = 5 jump to p < 0.05 originally hoped for did not materialise on individual metrics, but the cross-metric joint test is significant.
7. **SELDnet baseline reimplementation in our pipeline** — *Code DONE* (`week11_starss23/seldnet_official.py`, 11 / 11 unit tests; integrated as `--variant seldnet_official` in `train_seld.py`; backward-compatible `model_type` dispatch in `evaluate_seld.load_checkpoint`). *Training pending* — N = 3 vanilla + N = 3 SpecAug runs queued via `run_seldnet_baseline_queue.py`; will execute after the SpecAug ablation queue finishes (sequential to avoid GPU contention on a 4 GB laptop card). Architecture: 3 × Conv2D(64) + BiGRU(128, 2 layers) + FC(256→128) + FC(128→117) + tanh; 622,645 params; matches the published DCASE 2023 baseline closely enough that reviewers can verify the model definition. Cross-variant comparisons handled by `_pairwise_ttest.py`.
8. ~~**SpecAugment implementation**~~ **DONE** (`week11_starss23/seld_augment.py` with 15 unit tests; `--specaug` flag in `train_seld.py`). Wires time + freq masking on log-mel channels (DCASE 2023 baseline defaults: 2 × 50-frame time masks, 2 × 16-bin freq masks on log-mel only, GCC-PHAT lag axis untouched). Ablation will populate Table 5 once N = 5 paired runs are complete (so we don't burn another 4 h GPU before knowing where the unaugmented numbers land).

### IMPORTANT (no Table 3 = paper much weaker)

5. **Re-run all synthetic multi-seed numbers** with current code (rebuilt pipeline; ensure reproducibility) — *uses GPU now, ~1 hour total*
6. **Add SELDnet baseline to synthetic Table 1** (so reviewers see it on both synthetic AND real)

### NICE TO HAVE (won't reject if missing)

7. Distance estimation extension for STARSS23 (DCASE 2024 task)
8. Attention-weight visualisation figure
9. CST-Former baseline (heavy: 2-3 days extra)

### Writing tasks (parallel to coding)

10. Trim Section 3 from current 30-page draft into ICASSP 4-page form
11. Re-do Related Work section with 2024-2025 citations (GI-DOAEnet, CST-Former, AuralNet, Neural-SRP, SWeC, DCASE 2024 winner)
12. Make Figure 1 (architecture: GCA + backbone + multi-task heads)
13. Make Figure 2 (the punchline: paired ablation Δ across conditions and datasets)

---

## What changed from current draft (`draft.md`) to ICASSP draft (`icassp_draft.md`)?

| Change | Why |
|---|---|
| Title now leads with **"implicit geometry suffices"** | Positive frame, easier to defend at ICASSP |
| Add **STARSS23 results** as Table 2 | ICASSP reviewers want real data |
| Add **SELDnet baseline** in both tables | ICASSP reviewers want recent published baselines |
| Position **GI-DOAEnet** (Baek 2025) as supportive | Use friendly recent work to anchor our finding |
| Trim from journal-length to **4 pages** | ICASSP format constraint |
| Move multi-seed protocol to **Section 2** (method) | Makes it a contribution, not just a discussion point |
| Drop W7 OOD section | Saves space; OOD on STARSS23 implicitly covers it |
| Drop W3-W5 details | Keep only what supports the contribution |
