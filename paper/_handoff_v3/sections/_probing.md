# Linear probing -- ruling out a representation-quality story

A natural reading of the geometry-bias-hurts result would be that the
geometry token *destroys* spatial information inside the conv stack.
We test this directly with a linear probe.

For each of the fifteen Stage 3 checkpoints we freeze the network,
forward STARSS23 dev-test through the conv stack, and capture the
post-conv feature map (B, 64, T_label, F_red). Per frame we pool over
F_red with [mean ; max] (resulting in 128-dim vectors) and train a
Ridge regressor to predict (sin az, cos az, sin el, cos el) on frames
with exactly one active source. Five-fold cross-validation, splits
made at the file level so no single recording leaks into both folds.

| Cell | n | MAE mean (deg) | MAE std (deg) |
| ---- | - | -------------- | ------------- |
| 110_gca_full     | 5 | 28.50 | 0.50 |
| 111_gca_nogeom   | 5 | 28.37 | 0.81 |
| 112_no_gca       | 5 | 28.64 | 0.48 |

Pairwise contrasts:

* GCA full vs no-GCA: delta MAE = -0.14 +/- 0.87 deg, t = -0.36, p = 0.74, d_z = -0.16
* GCA full vs no_geom: delta MAE = +0.12 +/- 1.02 deg, t = +0.27, p = 0.80, d_z = +0.12
* no_geom vs no-GCA:  delta MAE = -0.26 +/- 0.62 deg, t = -0.94, p = 0.40, d_z = -0.42

All three cells encode location in the post-conv representation with
**essentially identical fidelity**. This rules out the simplest
mechanistic hypothesis (the geometry token corrupts the spatial features)
and shifts the locus of the harm to the **multi-track Multi-ACCDDOA
decoding stage** -- the conv stack still knows where the source is, but
the geometry-biased attention head changes how that information is
routed into the three ACCDDOA tracks, and that downstream interaction
is what we observe as a -0.4 percent F 20 deg loss.
