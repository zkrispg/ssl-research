# Experiment upgrade plan

Last updated: 2026-06-18

This plan strengthens the three main review risks without turning the paper into
a new leaderboard project.

## Goal

Improve reviewer confidence on:

1. small seed count (`n=5`);
2. weak absolute baseline;
3. weak Conformer operating point.

## Stage 1: key extremes to n=10

Run on the training machine:

```powershell
cd D:\ssl-research\dcase2024_baseline
.\_run_gca_extremes_n10_local.ps1
```

This adds seeds `5..9` for the load-bearing extremes:

- `130/131`: FOA CRNN full/no_geom;
- `141/142`: MIC Transformer full/no_geom.

Expected cost on one RTX-class GPU: about 20 train/test runs, roughly 2--3
continuous days.

Output summary:

```text
runs/gca_extremes_n10_summary.md
runs/gca_extremes_n10_summary.csv
runs/gca_extremes_n10_summary.json
```

Decision rule:

- If FOA CRNN remains negative and MIC Transformer remains positive with tighter
  confidence intervals, update the manuscript as an `n=10` robustness check.
- If either flips, keep the current `n=5` manuscript result but discuss the
  instability openly before submitting.

## Stage 2: Conformer operating-point pilot

Run after Stage 1, or in parallel on a second GPU:

```powershell
cd D:\ssl-research\dcase2024_baseline
.\_run_conformer_tuning_sweep_local.ps1
```

Pilot grid:

- tasks `171/172` only (FOA Conformer full/no_geom);
- seeds `0,1`;
- best checkpoint metrics `seld` and `doae`;
- learning rates `1e-3`, `5e-4`, `3e-4`;
- dropout fixed at `0.05`.

Expected cost: 24 train/test runs, roughly 3--5 continuous days.

Output summaries:

```text
runs/conformer_tune_seld_lr1em3_summary.md
runs/conformer_tune_seld_lr5em4_summary.md
runs/conformer_tune_seld_lr3em4_summary.md
runs/conformer_tune_doae_lr1em3_summary.md
runs/conformer_tune_doae_lr5em4_summary.md
runs/conformer_tune_doae_lr3em4_summary.md
```

Decision rule:

- Choose a setting that improves Conformer absolute SELD/F20 without turning the
  geometry effect into a new extreme.
- If no setting improves the operating point, leave Conformer as a limitation
  rather than spending more compute.

## Stage 3: manuscript integration

After syncing new summaries back to this machine:

```bash
cd paper/applied-acoustics
make compile
make check
```

Expected manuscript changes:

- add a short robustness paragraph for the `n=10` extremes;
- optionally add a Conformer tuning note in Limitations or Appendix;
- keep the main claim scoped as a controlled GCA study, not a leaderboard result.

## Not recommended now

Do not attempt a full DCASE winning-system port before first submission. It would
likely take weeks and may blur the one-bit intervention design.
