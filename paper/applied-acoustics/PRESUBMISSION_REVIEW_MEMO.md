# Pre-submission review memo

Last updated: 2026-06-17

## One-line paper claim

A matched one-bit geometry-prior ablation shows that explicit array geometry is
not universally useful for SELD: it helps most clearly in a recurrent FOA model,
is near-neutral to weakly helpful in a Conformer, and can hurt a pure-attention
Transformer; a second injection site bounds rather than generalizes the claim.

## What a reviewer is likely to ask

### Is the comparison controlled?

Current answer:
The manuscript emphasizes the one-bit GCA toggle, matched backbone/loss/data,
matched training recipe, and negligible parameter difference. The method table
lists full/no-geom/no-GCA task IDs.

Current check:
The Conformer rows now use the synced deterministic final files rather than the
older `path_c_2x2` placeholder values.

### Are the per-cell effects significant enough?

Current answer:
The text no longer leans on isolated per-cell significance. It frames the main
evidence as the recurrent-versus-Transformer interaction, the seed-matched
second difference, the ordered trend, and the STARSS22 FOA+CRNN replication.

Remaining check:
Keep the abstract and conclusion from overstating Transformer harm as fully
significant on its own. Current wording says it reproduces directionally, which
is the right level.

### Is the Conformer result overclaimed?

Current answer:
The manuscript treats Conformer as a near-neutral / weak-help middle point and
explicitly says it is not load-bearing evidence. The discussion notes its weak
operating point as a caveat.

Current check:
The stale FOA Conformer `+0.88` entries have been replaced with the final
`-0.57` DOAE contrast.

### Does `convbias` confirm the mechanism?

Current answer:
No. The manuscript now frames `convbias` as a sensitivity and boundary-condition
check, not a cross-mechanism replication. This protects the paper from the
unstable Transformer and checkpoint-selection results.

Remaining check:
Do not strengthen this section unless a larger deterministic convbias grid is
run later.

### Is the probe result useful despite being null?

Current answer:
Yes. The text explains that the null linear probe constrains the mechanism: the
behavioural effect is not a simple change in linearly decodable direction in the
post-convolutional representation.

Remaining check:
Make sure the probe table remains secondary and does not compete with the main
GCA result.

### Is the paper suitable for Applied Acoustics?

Current answer:
The topic is microphone-array signal processing, spatial audio, SELD, and
experimental validation on real recorded acoustic scenes. The cover letter and
metadata draft now frame it as a practical design rule rather than a leaderboard
model.

Remaining check:
Perform one manual visual pass over the final PDF before portal submission.

## Current hard blockers

- None reported by `make check`.

## Current warnings

- Bundled Tectonic is not on `PATH`; BasicTeX is the verified build toolchain.

## Ready pieces

- Narrative aligned around GCA as primary evidence.
- Final GCA Conformer result files are synced and table values are updated.
- `main.pdf` is freshly compiled with BasicTeX.
- Highlights are under Elsevier's 85-character limit.
- Cover letter draft exists.
- Submission metadata draft exists.
- Graphical abstract PNG is regenerated from SVG at `1328 x 531`.
- Checker validates inputs, figures, citations, labels, references, side files,
  graphical abstract dimensions, and known blockers.
