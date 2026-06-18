# Applied Acoustics submission checklist

Last updated: 2026-06-17

## Manuscript narrative

- [x] Main claim is framed around the controlled GCA factorial grid.
- [x] Conformer is described consistently as near-neutral / weakly helpful.
- [x] `convbias` is framed as a sensitivity / boundary-condition check, not as cross-mechanism confirmation.
- [x] Related work now frames geometry priors by injection site: input features, positional codes, attention bias, feature-map bias, and structural regularization.
- [x] Graphical abstract, highlights, method hypothesis, results, discussion, and conclusion use the same `helps -> near-neutral -> harms` wording.

## Files already revised

- [x] `sections/00_abstract.tex`
- [x] `sections/01_introduction.tex`
- [x] `sections/02_related.tex`
- [x] `sections/03_method.tex`
- [x] `sections/04_experiments.tex`
- [x] `sections/05_discussion.tex`
- [x] `sections/06_conclusion.tex`
- [x] `highlights.tex`
- [x] `graphical_abstract.tex`
- [x] `refs.bib`

## Final Conformer data sync

- [x] Sync `runs/gca_conformer_det_seld_final.md`
- [x] Sync `runs/gca_conformer_det_seld_final.csv`
- [x] Sync `runs/gca_conformer_det_seld_final.json`

The final files were synced from `origin/main` commit `c9746b4` and the GCA
Conformer manuscript tables have been updated:

- [x] `sections/04_experiments.tex`, Table `tab:dissociation`
- [x] `sections/04_experiments.tex`, Table `tab:cross`, in-domain column
- [x] `sections/04_experiments.tex`, Table `tab:convbias`, GCA column
- [x] `sections/07_appendix.tex`, complete per-cell Conformer rows

The local knowledge base records the final summary as:

| Cell | Final local summary |
|---|---:|
| MIC Conformer `161_full - 162_no_geom` | `Delta DOAE = -1.83 deg`, `Delta SELD = -0.0266`, `Delta F20 = +1.03%` |
| FOA Conformer `171_full - 172_no_geom` | `Delta DOAE = -0.57 deg`, `Delta SELD = -0.0510`, `Delta F20 = +1.50%` |

The manuscript now uses the final `.csv` / `.json` values for Conformer paired
contrasts and per-cell mean/std rows.

## Build and submission package

- [x] Draft cover letter exists.
- [x] Draft submission metadata exists.
- [x] Pre-submission reviewer-risk memo exists.
- [x] Local build diagnostics are documented.
- [x] Overleaf build handoff exists.
- [x] Compile `main.tex` with TeX Live or MacTeX/BasicTeX.
- [x] Confirm `main.pdf` reflects the latest source, not the old checked-in PDF.
- [x] Check line/page length after bibliography is regenerated.
- [ ] Optional: export `graphical_abstract.pdf` if the submission portal asks for a PDF version.
- [ ] Prepare non-anonymous metadata for final submission if not double-blind.

## Current verification

- [x] `git diff --check` passes.
- [x] Highlights are under Elsevier's 85-character limit.
- [x] No `through neutral` / `helpful-to-neutral` wording remains in the manuscript source.
- [x] `check_submission_state.sh` records missing final data and TeX tools as explicit blockers if they regress.
- [x] `check_submission_state.sh` checks included `.tex` files, figure assets, and cited bibliography keys.
- [x] `check_submission_state.sh` checks LaTeX labels/references and submission side files.
- [x] `graphical_abstract.png` is at least 1328 x 531 px.
- [x] Submission package script exists and supports dry-run.
- [x] Forced draft packages omit stale PDFs and include `DRAFT_BLOCKED.txt`.
- [x] Progress audit script exists and prevents promoting incomplete progress files.
- [x] Full PDF build is verified on this machine with BasicTeX.

## One-command status check

Run from `paper/applied-acoustics/`:

```bash
make check
```

The checker should currently return zero blockers after `make compile`. A
warning about bundled Tectonic not being on `PATH` is acceptable because BasicTeX
is the verified build toolchain.

## Final Conformer table update helper

Run from the repository root when you want to re-audit the final Conformer
numbers:

```bash
make -C paper/applied-acoustics plan-conformer
```

This script is read-only. It computes the final Conformer means and paired
contrasts and lists the manuscript table locations that were updated.

## Final result sync helper

From `paper/applied-acoustics/`, copy the final files from a local path:

```bash
make sync-final SOURCE=/path/to/experiment/runs
```

or from the training machine over SSH:

```bash
make sync-final SOURCE=user@host:/path/to/ssl-research/runs
```

This copies only `gca_conformer_det_seld_final.md/.csv/.json`, then runs the
read-only table-update planner and the submission-state checker.
