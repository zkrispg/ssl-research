# Cover letter draft

Dear Editor,

We are pleased to submit our manuscript, "When Do Microphone-Array Geometry Priors Help Sound Event Localization and Detection? The Role of Temporal Inductive Bias", for consideration in *Applied Acoustics*.

This work studies a practical question in spatial audio learning: when should a deep SELD system be given an explicit microphone-array geometry prior? Such priors are often assumed to be broadly helpful, but existing comparisons frequently change the architecture, input representation, and parameter count at the same time. We instead use a controlled one-bit intervention in the official DCASE 2024 SELD baseline, toggling a geometry bias inside an otherwise matched channel-attention block across MIC/FOA inputs and CRNN, Conformer, and Transformer temporal backbones.

Our main finding is that geometry priors are conditional design choices rather than universal improvements. In the primary GCA intervention, the effect on directional error is graded by temporal inductive bias: the prior helps recurrent models, remains near-neutral to weakly helpful in Conformers, and can harm pure-attention Transformers. We support this claim with seed-matched paired tests, a factorial architecture-by-prior interaction test, zero-shot validation on STARSS22, and a controlled representation probe. We also include a second injection mechanism, `convbias`, which acts as a sensitivity check and shows that the conclusion should not be overstated as mechanism-invariant.

We believe the manuscript is well suited to *Applied Acoustics* because it connects microphone-array geometry, spatial audio representation, and modern deep temporal architectures in a reproducible experimental framework. Rather than proposing another larger SELD model, the paper provides evidence about when a physically motivated prior should or should not be injected into a practical acoustic learning system.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission. The work uses publicly available datasets and a reproducible pipeline; code and checkpoints will be released upon publication.

Thank you for considering our submission.

Sincerely,

Anonymous Author(s)
