# Cross-dataset zero-shot evaluation -- STARSS22 dev-test

To rule out the possibility that the GCA-hurts-on-real-data result is
specific to STARSS23, we re-evaluate every Stage 3 checkpoint
(110/111/112 x five seeds, fifteen models in total) on the **STARSS22
dev-test split** (54 clips, 13-class taxonomy identical to STARSS23,
recorded with a different Eigenmike unit in different rooms in different
recording sessions). The model is **never fine-tuned on STARSS22**, so
this is a strict zero-shot domain transfer test.

We harmonise the metadata (STARSS22 has no distance annotation; we set
`lad_dist_thresh = inf` so distance never disqualifies a match, and we
report only F 20 deg, DOAE_CD, LR_CD, and SELD score). Features are
extracted with the *same* MIC GCC pipeline as Stage 3 and normalised
with the STARSS23 train-set scaler (`mic_wts`) to keep the transfer
strictly zero-shot.

The full table is in path_c_progress_v2 ; the headline is:

* The geometry-bias-hurts effect **replicates on STARSS22**:
  110 (full GCA) vs 111 (no_geom GCA) gives delta SELD = -0.029 with
  d_z = -0.89 and bootstrap 95 percent CI [-0.057, -0.005] -- which
  excludes zero.
* The cancellation pattern also replicates: full GCA vs no-GCA control
  gives delta F 20 deg ~= 0.
* Effect sign and magnitude on SELD are essentially identical to the
  in-distribution STARSS23 contrast.

The signal is therefore **not an artefact of STARSS23**. It transfers,
zero-shot, to a recording campaign that the model has never seen.
