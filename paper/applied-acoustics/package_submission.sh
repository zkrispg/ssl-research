#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"
PACKAGE_ROOT="$PAPER_DIR/submission_package"
PACKAGE_DIR="$PACKAGE_ROOT/applied-acoustics-submission"

dry_run=0
force=0

usage() {
  cat <<'EOF'
Usage:
  ./package_submission.sh --dry-run
  ./package_submission.sh [--force]

By default, packaging first runs ./check_submission_state.sh and refuses to
create a final package while blockers remain. A verified final package includes
the current PDFs. A forced draft package intentionally omits PDFs because they
may be stale.

Options:
  --dry-run   List package contents without copying files.
  --force     Create a source-only draft package even if blockers remain.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) dry_run=1 ;;
    --force) force=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
  shift
done

files=(
  "main.tex"
  "refs.bib"
  "highlights.tex"
  "graphical_abstract.svg"
  "graphical_abstract.png"
  "cover_letter_draft.md"
  "submission_metadata.md"
  "SUBMISSION_CHECKLIST.md"
  "PRESUBMISSION_REVIEW_MEMO.md"
  "EXPERIMENT_UPGRADE_PLAN.md"
  "BUILD_DIAGNOSTICS.md"
  "OVERLEAF_BUILD.md"
  "audit_conformer_progress.py"
  "plan_conformer_table_update.py"
)

dirs=(
  "sections"
)

figure_files=()
while IFS= read -r graphics_cmd; do
  figure_path="$(printf '%s\n' "$graphics_cmd" | sed 's/.*{//; s/}//')"
  candidates=(
    "$PAPER_DIR/$figure_path"
    "$PAPER_DIR/figs/$figure_path"
    "$PAPER_DIR/../figs/$figure_path"
  )
  if [[ "$figure_path" != *.* ]]; then
    candidates+=(
      "$PAPER_DIR/$figure_path.pdf" "$PAPER_DIR/$figure_path.png" "$PAPER_DIR/$figure_path.jpg"
      "$PAPER_DIR/figs/$figure_path.pdf" "$PAPER_DIR/figs/$figure_path.png" "$PAPER_DIR/figs/$figure_path.jpg"
      "$PAPER_DIR/../figs/$figure_path.pdf" "$PAPER_DIR/../figs/$figure_path.png" "$PAPER_DIR/../figs/$figure_path.jpg"
    )
  fi
  for candidate in "${candidates[@]}"; do
    if [ -f "$candidate" ]; then
      figure_files+=("$candidate")
      break
    fi
  done
done < <(rg -o '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$PAPER_DIR/main.tex" "$PAPER_DIR/sections" || true)

say_plan() {
  echo "Submission package plan"
  echo "Paper dir: $PAPER_DIR"
  echo ""
  echo "Top-level files:"
  for file in "${files[@]}"; do
    echo "  - $file"
  done
  echo ""
  echo "Directories:"
  for dir in "${dirs[@]}"; do
    echo "  - $dir/"
  done
  echo ""
  echo "Figure assets:"
  for figure in "${figure_files[@]}"; do
    echo "  - ${figure#$REPO_ROOT/}"
  done
  echo ""
  echo "PDF files:"
  if [ "$force" -eq 1 ]; then
    echo "  - omitted in --force draft packages"
  else
    echo "  - main.pdf"
    echo "  - graphical_abstract.pdf"
  fi
  echo ""
  echo "Output directory:"
  echo "  - $PACKAGE_DIR"
}

if [ "$dry_run" -eq 1 ]; then
  say_plan
  exit 0
fi

if [ "$force" -eq 0 ]; then
  "$PAPER_DIR/check_submission_state.sh"
else
  echo "WARNING: creating a draft package despite checker blockers."
fi

rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

for file in "${files[@]}"; do
  if [ -f "$PAPER_DIR/$file" ]; then
    cp "$PAPER_DIR/$file" "$PACKAGE_DIR/$file"
  else
    echo "WARNING: missing $file"
  fi
done

for dir in "${dirs[@]}"; do
  cp -R "$PAPER_DIR/$dir" "$PACKAGE_DIR/$dir"
done

mkdir -p "$PACKAGE_DIR/figs"
for figure in "${figure_files[@]}"; do
  cp "$figure" "$PACKAGE_DIR/figs/$(basename "$figure")"
done

if [ "$force" -eq 0 ]; then
  if [ -f "$PAPER_DIR/main.pdf" ]; then
    cp "$PAPER_DIR/main.pdf" "$PACKAGE_DIR/main.pdf"
  fi
  if [ -f "$PAPER_DIR/graphical_abstract.pdf" ]; then
    cp "$PAPER_DIR/graphical_abstract.pdf" "$PACKAGE_DIR/graphical_abstract.pdf"
  fi
fi

if [ "$force" -eq 1 ]; then
  cat > "$PACKAGE_DIR/DRAFT_BLOCKED.txt" <<'EOF'
This is a forced draft package, not a submission-ready package.

Known blockers at package time:
- Final gca_conformer_det_seld_final.md/.csv/.json files were not verified.
- Final Conformer manuscript table statistics may still be stale.
- Fresh TeX Live / MacTeX PDF build was not verified.
- main.pdf and graphical_abstract.pdf were intentionally omitted from this draft
  package to avoid sharing stale PDFs.

Run ./check_submission_state.sh and create the package without --force before
submitting.
EOF
fi

manifest="$PACKAGE_DIR/MANIFEST.txt"
{
  echo "Applied Acoustics submission package"
  echo "Generated: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "Source repository: $REPO_ROOT"
  echo ""
  find "$PACKAGE_DIR" -type f | sort | sed "s#^$PACKAGE_DIR/##"
} > "$manifest"

echo "Package created: $PACKAGE_DIR"
echo "Manifest: $manifest"
