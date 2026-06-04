# What the geometry token actually does (Tier IV)

For six STARSS23 dev-test clips (covering the sony and tau recording
sites and rooms 23/24/10/8/16/2) we forward seed-0 checkpoints of all
three cells and capture the GCA softmax attention weights (4x4) and
the per-mic sigmoid gate (4 values per chunk). Time-averaged matrices
(see paper/figs/path_c_attn_*.png) tell a clean story:

* In **110 (geometry_bias=True)** the attention matrix has a strong,
  highly asymmetric *diagonal-pair* structure: typically mic 0 attends
  ~98 percent to mic 2, mic 1 attends ~70 percent to mic 3, while the
  opposite-direction reads have roughly half that mass. That is exactly
  the pattern of largest-baseline pairs in the tetrahedral array; the
  geometry token is doing *exactly* what we designed it to do.
* In **111 (geometry_bias=False)** the same matrix is much more uniform
  (each query spreads attention across 20-35 percent on every key) --
  with no geometry term the head has no architectural reason to prefer
  any mic pair and learns a generic mixing.
* Per-mic gate magnitudes are similar across cells (~0.65 - 0.75) so
  the cells differ in *which inter-mic patterns* they emphasise, not in
  how much any single mic is down-weighted.

Combined with the probing result, this is a sharper story than the
ablation alone would have given us: the geometry prior **succeeds in
imposing the canonical mic-pair structure inside the attention head**;
the conv stack learns location with comparable fidelity in all three
cells; and yet on real-world data the prior is detrimental on the
final SED metric. Our reading: the prior assumes a tetrahedral array
with idealised mic responses and minimal reflections; when those
assumptions fail (real rooms, microphone capsule mismatch, high
diffuse reverberation) the head over-commits to the canonical pair
correlations and the multi-track decoder pays the price.
