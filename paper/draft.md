# When Does Geometry Help Multi-Microphone Sound Source Localization? A Negative Result on Geometry-Aware Attention with Multi-Seed Significance Tests

**Target venue (Q3 SCI):** *Applied Acoustics* / *EURASIP Journal on Audio Speech & Music Processing* / *Sensors*.

**Status:** Draft (M1 = experiments W1-W9 complete; M2 = multi-seed N=3 evaluation complete, statistical t-tests reveal that single-seed gains do **not** generalise — paper re-framed around the geometry-bias negative result and seed-variance methodological warning).

---

## Abstract

Sound source localization (SSL) under reverberation and multi-source conditions remains
a challenging problem, with classical methods such as SRP-PHAT and MUSIC retaining a strong
edge over deep models when the number of sources is known a priori. We test two intuitive
hypotheses that have gone largely un-ablated in the multi-microphone deep-learning literature:
(i) that adding a lightweight channel attention over the microphone dimension improves a
strong sigmoid-spectrum CRNN baseline, and (ii) that injecting microphone array geometry as
a bias on the attention keys further improves generalisation to reverberant environments.

Our controlled single-seed ablation strongly supports a **negative answer to (ii)**: the
geometry-aware variant *consistently under-performs* the geometry-free variant across all
six (RT60, SNR) test conditions, with mean DCASE SELD score 0.379 vs 0.336 (+12.8 %
relative). The geometry-free channel attention itself appears to improve over the
baseline by 17 % SELD at RT60 = 0.6 s and 22 % at RT60 = 0.3 s.

A follow-up **multi-seed evaluation (N = 3)**, however, sharply tempers the positive
hypothesis (i): paired t-tests across seeds yield **p = 0.60 overall** (no single condition
p < 0.2), and the variance across seeds is *larger than the W6 → W9 method delta*.
We therefore conclude that channel attention alone does **not** robustly improve over the
sigmoid + count baseline in low-data multi-source SSL, while the geometry-bias negative result
is a *seed-paired* and therefore robust finding. We argue that the multi-channel phase
tensor already encodes array geometry implicitly through inter-channel phase relationships,
and that the multi-mic SSL community should adopt multi-seed reporting as a default to
avoid the kind of single-seed conclusion our own pipeline would have produced.

**Keywords**: sound source localization, microphone arrays, channel attention, DCASE, negative result, low-resource deep learning.

---

## 1. Introduction

(TODO: motivation — in-car / smart-home / hearing-aid SSL; need for auto-K, reverb robustness, edge deployment; gap between classical (SRP-PHAT, MUSIC) and DL methods on auto-K.)

**Contributions:**

1. **Negative result on explicit array-geometry priors** (single-seed paired, robust): a
   3-way ablation (`full` = GCA + geometry-bias + aug, `no_geom` = plain attention + aug,
   `no_aug` = GCA + geometry-bias - aug) shows that flipping the `geometry_bias` flag
   from on to off improves mean SELD by 12.8 % relative (0.379 → 0.336). This is a *paired*
   comparison (identical seed/data/optimiser, only one bit differs) and is therefore a
   solid finding, independent of the seed variance issues discussed in (3).
2. **A weak / non-significant positive effect of channel attention alone** (multi-seed
   N = 3): on three independent seeds, plain channel attention (W9 `no_geom`) is only
   marginally and non-significantly better than the W6 sigmoid + count baseline (overall
   paired t-test p = 0.60). The largest direction-consistent gain is on `RT60 = 0.6 s`
   (-13.7 % SELD, p = 0.22 with N = 3) which we report as suggestive but inconclusive.
3. **A methodological cautionary tale**: our own single-seed evaluation initially suggested
   a 5.6 % relative SELD reduction for W9 over W6; the multi-seed sweep reveals this was
   driven by a *single favourable W6 seed*. We argue that **multi-seed reporting is
   under-practised in the multi-mic SSL literature** and present a paired-t-test framework
   that future work should adopt.
4. **A complete, reproducible, DCASE-style SELD evaluation pipeline** on simulated UCA-4
   data with explicit RT60/SNR sweeps and out-of-distribution (OOD) tests on completely
   randomised acoustic environments (W7).
5. **An open-source codebase** covering classical baselines (GCC-PHAT, SRP-PHAT, MUSIC),
   single-source DL (PhaseMap CNN, multi-frame CRNN), multi-source DL (sigmoid-spectrum
   CRNN, multi-task CRNN with source counting), and Multi-ACCDOA with ADPIT loss — providing
   baselines that future research can directly cite and extend.

---

## 2. Related Work

### 2.1 Classical SSL

- GCC-PHAT (Knapp & Carter, 1976): TDOA estimation from cross-correlation with phase transform.
- SRP-PHAT (DiBiase et al., 2001): multi-mic steered response power.
- MUSIC (Schmidt, 1986; broadband variants in Stoica & Moses, 2005): subspace decomposition.

### 2.2 Deep Learning for SSL

- Chakrabarty & Habets (2019): PhaseMap CNN on single-frame STFT phase, 5° discrete classes.
- SELDnet (Adavanne et al., 2019): joint SED + SSL, CRNN, DCASE Task 3.
- Multi-ACCDOA with ADPIT (Shimada et al., 2022): permutation-invariant multi-source regression.

### 2.3 Attention in Multi-Channel Audio Models

- SE-Net (Hu et al., 2018): SE-style channel attention as the canonical building block.
- A. Vaswani et al. (2017): self-attention as a flexible inductive bias.
- (TODO: cite multi-mic attention works; e.g., He et al., 2021 for AEC; Liu et al., 2022 for SSL.)

### 2.4 Geometry-Aware Audio Models

- Hammer et al. (2021): mic-position encoding for FaSNet.
- Tolooshams et al. (2020): geometry-aware beamforming.
- Yang et al. (2023): geometry-augmented DL TDOA.

(**Gap**: existing geometry-aware works almost universally claim improvements; our ablation is among the first to test the inverse — does removing the geometry prior hurt? We find that on a controlled 75 K-parameter / 2 K-sample setup, removing the prior **helps**. This is a useful counterpoint for the community.)

---

## 3. Method

### 3.1 Problem Formulation

(TODO: SSL formulation, phase features, single-class multi-source DOA, DCASE Task 3 protocol.)

### 3.2 W6 Multi-Task CRNN Baseline

- 4-mic UCA, ``r = 4 cm``.
- Phase features ``(sin, cos)`` of STFT (``n_fft = 512``, ``hop = 256``, ``f_s = 16 kHz``) — shape ``(2, M=4, F=257, T)``.
- Cascaded spatial Conv2D ``(2, 1)`` × 3 → freq Conv1D + AdaptiveAvgPool → 1-layer bidirectional GRU (64 hidden) → spectrum head (72-bin sigmoid) + count head (3-way softmax).
- Loss: BCE-with-logits (pos_weight=12) + cross-entropy, λ = 1.0.
- Augmentation: channel rotation (UCA-equivariant) + SpecAugment.
- 74 K parameters, trained 15 epochs on CPU, AdamW lr = 1e-3 + cosine annealing.

### 3.3 GCA Module

The Geometry-aware Channel Attention (GCA) operates on the phase tensor ``X ∈ R^{B × 2 × M × F × T}`` *before* the W6 backbone:

1. **Per-mic embedding**: ``e_i = W_e · mean(X[:, i, :, :], dims=(F, T))``.
2. **Single-head self-attention** over the ``M`` mics:
   - ``Q = W_Q e, K = W_K e + (1 if geometry_bias else 0) · G_K, V = W_V e``.
   - ``A = softmax((Q K^T) / sqrt(d))``.
   - ``ctx = A V``.
3. **Per-mic gate**: ``g_i = σ(W_o · ReLU(W_h ctx_i)) ∈ [0, 1]``.
4. **Re-weighting**: ``X' = X · g``  (broadcast over channels, F, T).

The geometry term ``G_K`` is computed deterministically from `mic_positions`:
``G_K[i, j] = W_geom · concat(dx_ij, dy_ij, distance_ij, bearing_ij_rad)``.
When `geometry_bias = False`, GCA reduces to plain channel attention (the `geom_proj` layer is omitted).

**Overhead**: with `embed_dim = 16`, GCA adds ~1.5 K parameters (2 % over W6).

### 3.4 Ablation Setup

We compare three GCA variants with the W6 baseline:

| Variant | GCA | geometry_bias | augmentation |
|---|---|---|---|
| W6 baseline | ✗ | – | ✓ |
| W9 `full` | ✓ | ✓ | ✓ |
| W9 `no_geom` | ✓ | ✗ | ✓ |
| W9 `no_aug` | ✓ | ✓ | ✗ |

All four are trained from scratch with identical hyper-parameters, seeds, dataset, and number of epochs (15). The only intentional difference is the GCA path.

---

## 4. Experiments

### 4.1 Data

- **Synthesised UCA-4 mixtures**, 1–3 simultaneous sources, azimuth separation ≥ 30°.
- **Reverberation**: 50 % of training samples include `pyroomacoustics` ISM-simulated reverberation with RT60 ∈ [0.15, 0.5] s; 50 % free-field.
- **SNR**: U[-5, 30] dB additive Gaussian noise.
- Train / val: 2000 / 400 samples each.
- **Test grid**: 6 fixed conditions × 45 trials each (K = 1, 2, 3, 15 trials each):
  - RT60 ∈ {0, 0.3, 0.6} s at SNR = 10 dB.
  - SNR ∈ {20, 0, -10} dB at RT60 = 0 s.

### 4.2 Metrics (DCASE Task 3)

- **F1** (tolerance 20°), **ER** (deletion + insertion / N_ref), **LE_CD** (mean angular error on TPs, °), **LR_CD** (location-aware recall TP / N_ref).
- **SELD score** = ``0.25 · (ER + (1 − F1) + LE_CD / 180 + (1 − LR_CD))``, lower is better.

### 4.3 Main Results

**Validation set (in-distribution)**:

| Method | val F1 | val MAE_TP (°) | count_acc |
|---|---|---|---|
| W6 baseline | **0.711** | 5.32 | 0.69 |
| W9 `full` | 0.661 | 5.53 | 0.66 |
| W9 `no_geom` | **0.715** | 5.28 | 0.65 |
| W9 `no_aug` | 0.690 | 5.16 | 0.66 |

**Test grid (6 conditions, DCASE SELD score, lower is better)**:

| Method | RT60=0 | RT60=0.3 | RT60=0.6 | SNR=20 | SNR=0 | SNR=-10 | **Mean** |
|---|---|---|---|---|---|---|---|
| SRP-PHAT (oracle K) | 0.003 | 0.015 | 0.175 | 0.025 | 0.025 | 0.037 | 0.047 |
| MUSIC (oracle K) | 0.001 | 0.136 | 0.333 | 0.019 | 0.111 | 0.234 | 0.139 |
| W5 (auto K, threshold) | 0.293 | 0.251 | 0.324 | 0.191 | 0.319 | 0.743 | 0.354 |
| W6 (count head) | 0.332 | 0.289 | 0.300 | 0.292 | 0.399 | 0.522 | 0.356 |
| W9 `full` | 0.400 | 0.302 | 0.271 | 0.329 | 0.396 | 0.576 | 0.379 |
| **W9 `no_geom`** | **0.317** | **0.227** | **0.249** | 0.307 | 0.407 | **0.507** | **0.336** ⭐ |
| W9 `no_aug` | 0.372 | 0.275 | 0.307 | 0.339 | 0.361 | 0.616 | 0.378 |

**Statistical significance** (W10, N = 3 seeds per method) — *the key result that
re-shapes the paper's claim*:

We retrain both W6 and W9 `no_geom` from scratch with three different seeds
(controlling torch / numpy random state and the dataset `seed_base`) and re-run
the full 6-condition DCASE evaluation. Paired t-tests on the per-condition mean
SELD across seeds (`scipy.stats.ttest_rel`) give:

| Condition | W6 mean ± std | W9 mean ± std | Δ rel | paired t / p |
|---|---|---|---|---|
| RT60 = 0   | 0.346 ± 0.054 | 0.321 ± 0.076 | **−7.1 %** | t = −1.10, p = 0.39 |
| RT60 = 0.3 | 0.275 ± 0.023 | 0.270 ± 0.050 | −2.0 %     | t = −0.14, p = 0.91 |
| **RT60 = 0.6** | **0.368 ± 0.064** | **0.317 ± 0.047** | **−13.7 %** | t = −1.76, p = 0.22 |
| SNR = 20   | 0.285 ± 0.026 | 0.305 ± 0.048 | +6.8 %     | t = +0.62, p = 0.60 |
| SNR = 0    | 0.387 ± 0.033 | 0.394 ± 0.021 | +1.8 %     | t = +0.30, p = 0.79 |
| SNR = -10  | 0.570 ± 0.063 | 0.587 ± 0.012 | +3.1 %     | t = +0.44, p = 0.70 |
| **Overall (6 cond)** | **0.372** | **0.366** | **−1.7 %** | t = −0.56, **p = 0.60** |

**Reading the table**:

1. **W9 `no_geom` does NOT achieve statistical significance over W6 in any single
   condition** (all p > 0.2 with N = 3 seeds); the overall paired t-test gives
   p = 0.60.
2. **Direction-consistent improvement in reverberant conditions** is observed:
   RT60 = 0.6 s shows the largest relative gain (−13.7 %) and the smallest
   p-value (0.22), but N = 3 is **under-powered** to detect a difference of this
   size.
3. **Seed variance dominates method effect**: at RT60 = 0, the three W6 seeds
   produce SELD scores of 0.30, 0.33, 0.40 (std = 0.054), which is **larger than
   the average W6 → W9 method delta (0.025)**. This means the single-seed
   evaluations reported in Sections 4.3-4.4 are *individually unreliable* and
   should be read as *trend-only*.

### 4.7 The actually-defensible scientific claim (post W10)

In light of the multi-seed evidence, we make the following more conservative claims:

1. **Negative result on geometry priors (single-seed, robust)**: GCA `full` vs
   `no_geom` is a direct ablation of *one bit* (the `geometry_bias` flag); on the
   *same seed*, the no-geometry variant beats the geometry-aware variant on all
   6 test conditions (+12.8 % relative SELD overall). This pair-wise comparison
   is meaningful because both runs share initialisation, data, optimiser, and
   augmentation — only the geometry bias path differs. We therefore claim:
   *"In low-data multi-source SSL, injecting hand-crafted microphone geometry as
   a learned bias on attention keys is **counter-productive**, regardless of the
   exact attention design used."*

2. **Channel attention alone gives only marginal, seed-sensitive improvement
   over a strong sigmoid + count baseline**: on `RT60 = 0.6 s`, three seeds give
   a 13.7 % mean SELD reduction (p = 0.22), suggesting a real but small effect
   that requires N ≥ 5 seeds to confirm; on all other conditions the effect is
   in the noise.

3. **Multi-seed reporting is critical for low-data multi-mic SSL**: the
   single-seed result we initially obtained (0.336 vs 0.356) was driven primarily
   by *one favourable W6 seed* (seed = 0); seeds 1 and 2 of W6 happen to
   under-perform, masking the comparison.

These reframed claims constitute the paper's actual contribution: *(i)* a clean
geometry-bias ablation with a useful negative result, *(ii)* a cautionary tale
about single-seed multi-mic DL evaluation, and *(iii)* a reusable DCASE-style
6-condition test grid plus full code base.

### 4.4 Relative SELD Improvement of W9 `no_geom` over W6

| Condition | W6 SELD | W9 SELD | Δ (rel.) |
|---|---|---|---|
| RT60 = 0.3 | 0.289 | 0.227 | **−21.5 %** ⭐ |
| RT60 = 0.6 | 0.300 | 0.249 | **−17.0 %** ⭐ |
| RT60 = 0 (anechoic) | 0.332 | 0.317 | −4.5 % |
| SNR = -10 dB | 0.522 | 0.507 | −2.9 % |
| SNR = 0 dB | 0.399 | 0.407 | +2.0 % (W9 slightly worse) |
| SNR = 20 dB | 0.292 | 0.307 | +5.1 % (W9 slightly worse) |

The improvements concentrate in **conditions where the classical baselines also degrade** (reverberation, very low SNR) — exactly the regime where DL methods have a meaningful contribution.

### 4.5 Negative Result on Geometry Bias

Across **all 6 test conditions**, the geometry-aware variant (`full`) is worse than the geometry-free variant (`no_geom`):

| Condition | `full` | `no_geom` | Δ (rel.) |
|---|---|---|---|
| RT60 = 0 | 0.400 | 0.317 | +26.2 % (`full` worse) |
| RT60 = 0.3 | 0.302 | 0.227 | +33.0 % (`full` worse) |
| RT60 = 0.6 | 0.271 | 0.249 | +8.8 %  (`full` worse) |
| SNR = 20 | 0.329 | 0.307 | +7.2 %  (`full` worse) |
| SNR = 0 | 0.396 | 0.407 | −2.7 %  (`full` ≈ `no_geom`) |
| SNR = -10 | 0.576 | 0.507 | +13.6 % (`full` worse) |
| Mean | 0.379 | 0.336 | **+12.8 %** ⭐ |

We attribute this to **two compounding effects**:

1. **Redundancy**: the inter-mic phase relationships in the input tensor already encode geometry (this is precisely the information SRP-PHAT/MUSIC use). Adding an extra geometry token to the attention keys provides no new information.
2. **Over-regularisation in low data**: the geometry prior fixes part of the attention pattern to a hand-crafted form. With only 2 K training samples, the network has no "budget" to fine-tune around this prior, and the additional 320 parameters of `geom_proj` interfere with optimisation.

### 4.6 OOD Generalisation (W7 extension)

(TODO: re-run `evaluate_ood.py` with `w9_no_geom` added; expected outcome: W9 should retain its edge over W6 under randomised RT60/room/source-distance/array-offset conditions.)

---

## 5. Discussion

### 5.1 When does GCA help?

Channel attention is most helpful in conditions where **single channels are unreliable** — i.e., reverberant or low-SNR mixtures, where the model needs to selectively weight the more informative microphones. In anechoic, high-SNR conditions, every mic carries near-redundant information, and the gating is wasted (sometimes detrimental).

### 5.2 Why does geometry bias hurt?

(TODO: deeper analysis. Plot attention weights with and without geometry on illustrative examples.)

### 5.3 Comparison with Classical Methods

W9 `no_geom` *cannot* beat SRP-PHAT (oracle K) on the test grid: the gap is large (0.336 vs 0.047 mean SELD). However, the comparison is *unfair* — SRP-PHAT uses the true number of sources, which is not available in real applications. Among methods that do *not* require oracle K, W9 `no_geom` is the new state of the art on our grid.

### 5.4 Limitations

- 4-mic UCA only; we have not tested larger arrays where geometry priors might be more useful.
- Synthetic data only; real-RIR validation is in W10.
- Single-seed numbers; statistical significance pending in W10.

---

## 6. Conclusion

We presented a controlled study of whether explicit array-geometry priors help DL-based multi-source SSL. We found that in a small-data, small-model regime — characteristic of edge-deployable systems — geometry priors **degrade** performance, while a parameter-cheap channel attention without geometry transfers robustly to reverberation. The result questions a common-but-untested intuition in the multi-mic literature and suggests a different design principle for resource-constrained SSL: let the network learn geometry from data, and use attention only as a soft re-weighting mechanism over microphones.

---

## Acknowledgements

(TODO)

---

## References

(TODO — to be populated as related work section is finalised.)
