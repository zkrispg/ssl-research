# Stronger baseline -- DCASE 2024 SELDnet (Multi-ACCDDOA)

## Why we replaced the W6 mini-baseline

The Path A / Path B comparison was made against an in-house CRNN that we
trained from scratch on STARSS23 dev-train. Reviewers of recent SELD work
(e.g. ICASSP 2025 Track 3 audio) consistently raise the concern that any
"X hurts Y" claim against an in-house baseline must be reproduced on a
*recognised, official* baseline before it can be trusted. We therefore
re-ran the full GCA ablation against the official **DCASE 2024 SELDnet
Multi-ACCDDOA** baseline released with the DCASE 2024 challenge, using
the official synthetic-pretrained checkpoint as init and the official
60-epoch fine-tune recipe.

## Reproducing the DCASE 2024 numbers

We first verified that our environment can reproduce the published
DCASE 2024 FOA result (F 20 deg = 13.1 percent in the README).  Across
five seeds we obtain F 20 deg = 13.06 +/- 0.75 percent and DOAE_CD =
40.7 +/- 6.6 deg (vs reference 36.9 deg). The baseline therefore behaves
as expected on our hardware/software stack and our results below are
not contaminated by an unintentional ablation.

## GCA ports cleanly to the DCASE baseline

The GCA module is a drop-in replacement for "no channel attention" inside
the conv stack. The DCASE 2024 SELDnet has three Conv2d blocks
(64 filters each) before a stack of MHSA+RNN blocks; we insert GCA on
the per-mic logmel input *before* the first conv. The geometry buffer
(tetrahedral mic array of the Eigenmike used in STARSS23) is registered
as a non-persistent buffer so model checkpoints stay compatible with
non-GCA controls. Smoke tests: forward / backward pass shape match,
parameter count delta = 1.5 K (geometry projection layer only),
non-strict load of the synthetic MIC checkpoint cleanly reports the GCA
parameters as "missing" while the rest of the backbone loads.
