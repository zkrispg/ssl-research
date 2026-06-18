# Applied Acoustics paper workspace

This directory is the working submission package for the Applied Acoustics
manuscript.

## Current state

The manuscript narrative has been revised around the controlled GCA factorial
grid:

- CRNN: geometry priors help.
- Conformer: geometry priors are near-neutral to weakly helpful.
- Transformer: geometry priors can harm.
- `convbias`: sensitivity / boundary-condition evidence, not a clean
  mechanism-invariant replication.

The final deterministic GCA Conformer result files are now present in
`../../runs/`, and the manuscript tables have been updated from them.

Required files:

- `../../runs/gca_conformer_det_seld_final.md`
- `../../runs/gca_conformer_det_seld_final.csv`
- `../../runs/gca_conformer_det_seld_final.json`

## Daily commands

Run these from this directory:

```bash
make compile
make check
make plan-conformer
```

`make check` reports manuscript blockers. The expected current state is zero
blockers after `make compile`; a warning about bundled Tectonic not being on
`PATH` is acceptable because BasicTeX is the verified build toolchain.

`make plan-conformer` is read-only. After the final result files are synced, it
prints the final per-task means, paired contrasts, and manuscript table targets.

`make audit-progress` is retained as a read-only guard for the old progress
file. It is no longer needed for final packaging because the final files are now
available.

Submission-facing metadata lives in:

- `cover_letter_draft.md`
- `submission_metadata.md`
- `SUBMISSION_CHECKLIST.md`
- `PRESUBMISSION_REVIEW_MEMO.md`
- `BUILD_DIAGNOSTICS.md`
- `OVERLEAF_BUILD.md`

The current graphical abstract submission image is generated from
`graphical_abstract.svg`:

```bash
sips -s format png graphical_abstract.svg --out graphical_abstract.png
```

`graphical_abstract.tex` is retained as a legacy TikZ source, but the SVG/PNG
pair is the version aligned with the current manuscript wording.

## Sync final Conformer results

If the final files are on a mounted local directory:

```bash
make sync-final SOURCE=/path/to/experiment/runs
```

If the final files are on the training machine over SSH:

```bash
make sync-final SOURCE=user@host:/path/to/ssl-research/runs
```

The sync step copies only the three `gca_conformer_det_seld_final.*` files into
`../../runs/`, then runs the table-update planner and submission checker.

## Final build

The project uses bibliography tooling, so the final PDF should be built with
TeX Live or MacTeX:

```bash
make compile
```

If only bundled Tectonic is available, use Overleaf or install a local TeX
distribution before treating the PDF as submission-ready.

Current local build diagnostics are recorded in `BUILD_DIAGNOSTICS.md`.
Overleaf remains available as an independent layout check; see
`OVERLEAF_BUILD.md`.

## Package for submission

Preview the package contents:

```bash
make package-dry-run
```

Create the package only after `make check` passes:

```bash
make package
```

The package script refuses to create a final package while blockers remain. Use
`./package_submission.sh --force` only for an explicitly marked source-only
draft package. Forced draft packages intentionally omit stale PDFs.
