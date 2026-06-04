# Recent (2023-2024) SELD baselines on STARSS23

This table is the basis for the "Related work" baseline survey in the
paper. We deliberately list 2024-era systems so that reviewers see we
are aware of the SOTA, while still framing our contribution as a
*controlled-comparison* study rather than an attempt to win the leaderboard.

All numbers are **DCASE 2024 audio-only Track A**, F-score evaluated with
20° angular tolerance plus the relative-distance threshold of 1.0
(DCASE 2024 metric -- F is more stringent than the 2023 metric because
it also requires correct distance). DOA-error and relative distance
error (RDE) are reported as in the official leaderboard.

| Rank | Submission | Affiliation | Architecture summary | F1 % (eval) | DOA ° (eval) | RDE (eval) | Source |
| ---- | ---------- | ----------- | -------------------- | ----------- | ------------ | ---------- | ------ |
| 1    | Du_NERCSLIP_4 | USTC NERC-SLIP | ResNet-Conformer ensemble + EINv2 + extensive aug. | 54.4 | 13.6 | 0.21 | Wang et al., DCASE 2024 [^du] |
| 2    | Yu_HYUNDAI_3 | Hyundai | ResNet18-Conformer w/ pretrained ASR features. | 29.8 | 19.8 | 0.28 | Yu et al., DCASE 2024 |
| 3    | Yeow_NTU_2 | NTU | ConvNeXt-Conformer + SpecAug + channel rotation. | 26.2 | 25.1 | 0.26 | Yeow et al., DCASE 2024 |
| 5    | Vo_DU_1 | Drexel | Transformer w/ Neural GCC-PHAT (NGCC-PHAT). | 24.7 | 19.3 | 0.34 | Vo et al., DCASE 2024 |
| 6    | Berg_LU_3 | Lund / Arm | Cross-Feature Transformer fusing log-mel & GCC. | 25.5 | 23.2 | 0.39 | Berg et al., DCASE 2024 |
| 9    | AO_Baseline_FOA | TAU/Sony | DCASE 2024 official baseline (CRNN + ACCDOA). | 18.0 | 29.6 | 0.31 | Politis et al., DCASE 2024 |
| 27   | AO_Baseline_MIC | TAU/Sony | DCASE 2024 official baseline w/ mic-array stack. | 16.3 | 34.1 | 0.30 | Politis et al., DCASE 2024 |

[^du]: Wang, Q. et al. *USTC-NERCSLIP system for DCASE 2024 Task 3*, technical report.

## Comparison to the prior DCASE 2023 challenge

For 2023 (no distance estimation, 20° F1 only) the official AO baseline
scored **F = 29.4 %** on STARSS23 eval (rank 10 of 27). Our paper
reproduces this exact baseline and performs ablations relative to it,
so we have a directly comparable point of reference.

## Position of our work

* **Our claim is not "we beat the leaderboard"**; the SOTA at top-1 is
  a heavily engineered transformer ensemble with > 50 M parameters, far
  outside our compute envelope.
* **Our claim is "geometry priors do not provide an additional gain"**
  on top of the standard CRNN/ACCDOA recipe, *under matched training
  budgets*. We further support this with:
    1. Multiple seeds per cell (N = 5 paired t-tests).
    2. A capacity sweep at four parameter scales (~250 K to ~3 M).
    3. SpecAugment ablation at strong / weak / off settings.
    4. Cross-dataset evaluation on STARSS22 dev-test
       (zero-shot transfer).
    5. FOA vs microphone-array format ablation.
* The 2024 leaderboard is included for context only; we discuss it in
  Related Work to position our contribution.

## Caveat on absolute numbers

Because the 2024 metric folds in distance-estimation accuracy, the F1
numbers above are *not* directly comparable to the F1 numbers we
report in our experiments (which use the 2023 metric: 20° angular
tolerance only, no distance). For our setup the equivalent 2023-style
F1 of the AO_Baseline on STARSS23 eval is ~30 %, against which our
SELDnet replication achieves ~16 % under a 30-epoch (rather than the
official 100-epoch) training budget. We discuss this gap in Section 5
("Threats to validity").
