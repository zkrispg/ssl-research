# Overleaf build handoff

Use this when local TeX Live / MacTeX is unavailable. This path verifies layout
and bibliography on Overleaf, then brings the fresh PDF back into this local
paper workspace.

## Current status

Local BasicTeX is available and has produced the current `main.pdf`. The final
Conformer files have also been synced and the stale table entries have been
updated:

- `../../runs/gca_conformer_det_seld_final.md`
- `../../runs/gca_conformer_det_seld_final.csv`
- `../../runs/gca_conformer_det_seld_final.json`

Use Overleaf only as an optional independent layout check.

## Upload set

Upload these files/directories to a new Overleaf project:

- `main.tex`
- `refs.bib`
- `sections/`
- `../figs/locality_spectrum.png`
- `../figs/path_c_2x2_dissociation.png`
- `../figs/forest_gradient.png`

Optional side files for the submission portal, not required for compiling
`main.tex`:

- `highlights.tex`
- `graphical_abstract.svg`
- `graphical_abstract.png`
- `cover_letter_draft.md`
- `submission_metadata.md`
- `SUBMISSION_CHECKLIST.md`
- `PRESUBMISSION_REVIEW_MEMO.md`
- `BUILD_DIAGNOSTICS.md`

## Overleaf settings

- Main document: `main.tex`
- Compiler: pdfLaTeX
- Bibliography: BibTeX / standard LaTeX bibliography
- TeX Live version: latest available

The manuscript uses Elsevier's `elsarticle` class and `elsarticle-num`
bibliography style. If Overleaf reports either one missing, switch the project
to a recent TeX Live image or add the class/style file explicitly.

## After compile succeeds

1. Download the compiled `main.pdf`.
2. Replace local `paper/applied-acoustics/main.pdf` with the downloaded PDF.
3. Run:

   ```bash
   make check
   ```

4. Visually inspect the PDF for:
   - title page and anonymous metadata;
   - table widths;
   - figure placement;
   - references rendered and numbered;
   - appendix tables not overflowing;
   - no question marks for unresolved citations or references.

5. Only after `make check` passes and visual inspection is clean, run:

   ```bash
   make package
   ```

## Expected local state before final packaging

- `make check` reports zero blockers.
- `main.pdf` is newer than `.tex`, `.bib`, and `.bst` sources.
- The final Conformer files have been synced.
- No stale `+0.88` Conformer placeholder remains in the manuscript tables.
