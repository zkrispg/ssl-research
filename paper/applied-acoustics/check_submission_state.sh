#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"
BASICTEX_BIN="$HOME/usr/local/texlive/2026basic/bin/universal-darwin"

if [ -x "$BASICTEX_BIN/pdflatex" ]; then
  export PATH="$BASICTEX_BIN:$PATH"
fi

blockers=0
warnings=0

say() {
  printf '%s\n' "$*"
}

blocker() {
  blockers=$((blockers + 1))
  say "BLOCKER: $*"
}

warning() {
  warnings=$((warnings + 1))
  say "WARNING: $*"
}

ok() {
  say "OK: $*"
}

say "Applied Acoustics submission state"
say "Paper dir: $PAPER_DIR"
say ""

if ! command -v rg >/dev/null 2>&1; then
  blocker "ripgrep (rg) is required for source checks."
else
  ok "rg is available."
fi

say ""
say "1) Required final GCA Conformer result files"
for file in \
  "$REPO_ROOT/runs/gca_conformer_det_seld_final.md" \
  "$REPO_ROOT/runs/gca_conformer_det_seld_final.csv" \
  "$REPO_ROOT/runs/gca_conformer_det_seld_final.json"; do
  if [ -f "$file" ]; then
    ok "found ${file#$REPO_ROOT/}"
  else
    blocker "missing ${file#$REPO_ROOT/}; do not update final Conformer table statistics yet."
  fi
done

say ""
say "2) Known stale Conformer numeric placeholders"
if command -v rg >/dev/null 2>&1; then
  stale_hits="$(rg -n '\+0\.88' "$PAPER_DIR/sections/04_experiments.tex" "$PAPER_DIR/sections/07_appendix.tex" || true)"
  if [ -n "$stale_hits" ]; then
    warning "old FOA Conformer +0.88 entries are still present pending final result files:"
    printf '%s\n' "$stale_hits"
  else
    ok "no old +0.88 Conformer contrast found in experiment/appendix sources."
  fi
fi

say ""
say "3) Narrative wording consistency"
if command -v rg >/dev/null 2>&1; then
  manuscript_sources=(
    "$PAPER_DIR/main.tex"
    "$PAPER_DIR/highlights.tex"
    "$PAPER_DIR/graphical_abstract.tex"
    "$PAPER_DIR/sections"
  )
  bad_wording="$(rg -n 'through neutral|helpful-to-neutral|is neutral in' "${manuscript_sources[@]}" || true)"
  if [ -n "$bad_wording" ]; then
    blocker "old neutral wording remains:"
    printf '%s\n' "$bad_wording"
  else
    ok "no stale neutral wording patterns found."
  fi
fi

say ""
say "4) Elsevier highlight length"
if [ -f "$PAPER_DIR/highlights.tex" ]; then
  long_highlights="$(awk '/\\item/{line=$0; sub(/^.*\\item /,"",line); if (length(line) > 85) print length(line) ": " line}' "$PAPER_DIR/highlights.tex")"
  if [ -n "$long_highlights" ]; then
    blocker "highlight entries exceed 85 characters:"
    printf '%s\n' "$long_highlights"
  else
    ok "all highlights are <= 85 characters."
  fi
else
  blocker "missing highlights.tex"
fi

say ""
say "5) LaTeX source references"
if command -v rg >/dev/null 2>&1; then
  missing_inputs=""
  while IFS= read -r input_cmd; do
    input_path="$(printf '%s\n' "$input_cmd" | sed 's/.*{//; s/}//')"
    case "$input_path" in
      *.tex) input_file="$PAPER_DIR/$input_path" ;;
      *) input_file="$PAPER_DIR/$input_path.tex" ;;
    esac
    if [ ! -f "$input_file" ]; then
      missing_inputs="${missing_inputs}${input_path}"$'\n'
    fi
  done < <(rg -o '\\input\{[^}]+\}' "$PAPER_DIR/main.tex" || true)

  if [ -n "$missing_inputs" ]; then
    blocker "missing files referenced by \\input:"
    printf '%s' "$missing_inputs"
  else
    ok "all \\input files exist."
  fi

  missing_figures=""
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
    found=0
    for candidate in "${candidates[@]}"; do
      if [ -f "$candidate" ]; then
        found=1
        break
      fi
    done
    if [ "$found" -eq 0 ]; then
      missing_figures="${missing_figures}${figure_path}"$'\n'
    fi
  done < <(rg -o '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "$PAPER_DIR/main.tex" "$PAPER_DIR/sections" "$PAPER_DIR/graphical_abstract.tex" || true)

  if [ -n "$missing_figures" ]; then
    blocker "missing files referenced by \\includegraphics:"
    printf '%s' "$missing_figures"
  else
    ok "all \\includegraphics assets were found."
  fi

  if [ -f "$PAPER_DIR/refs.bib" ]; then
    tmp_dir="$(mktemp -d)"
    rg -o '\\cite\{[^}]+\}' "$PAPER_DIR/main.tex" "$PAPER_DIR/sections" \
      | sed 's/.*\\cite{//; s/}//' \
      | tr ',' '\n' \
      | sed 's/^ *//; s/ *$//' \
      | sed '/^$/d' \
      | sort -u > "$tmp_dir/cited.txt"
    sed -n 's/^@[^{}]*{\([^,]*\),.*/\1/p' "$PAPER_DIR/refs.bib" \
      | sort -u > "$tmp_dir/bibkeys.txt"
    missing_cites="$(comm -23 "$tmp_dir/cited.txt" "$tmp_dir/bibkeys.txt" || true)"
    rm -rf "$tmp_dir"
    if [ -n "$missing_cites" ]; then
      blocker "missing bibliography entries for cited keys:"
      printf '%s\n' "$missing_cites"
    else
      ok "all cited bibliography keys exist in refs.bib."
    fi
  else
    blocker "missing refs.bib"
  fi

  tmp_dir="$(mktemp -d)"
  rg -o '\\label\{[^}]+\}' "$PAPER_DIR/main.tex" "$PAPER_DIR/sections" "$PAPER_DIR/graphical_abstract.tex" \
    | sed 's/.*\\label{//; s/}//' \
    | sort > "$tmp_dir/labels.txt" || true
  rg -o '\\(eq)?ref\{[^}]+\}' "$PAPER_DIR/main.tex" "$PAPER_DIR/sections" \
    | sed 's/.*{//; s/}//' \
    | sort -u > "$tmp_dir/refs.txt" || true
  duplicate_labels="$(uniq -d "$tmp_dir/labels.txt" || true)"
  missing_refs="$(comm -23 "$tmp_dir/refs.txt" "$tmp_dir/labels.txt" || true)"
  rm -rf "$tmp_dir"

  if [ -n "$duplicate_labels" ]; then
    blocker "duplicate LaTeX labels:"
    printf '%s\n' "$duplicate_labels"
  else
    ok "no duplicate LaTeX labels found."
  fi

  if [ -n "$missing_refs" ]; then
    blocker "missing LaTeX labels referenced by \\ref/\\eqref:"
    printf '%s\n' "$missing_refs"
  else
    ok "all \\ref/\\eqref labels exist."
  fi
fi

say ""
say "6) Submission side files"
for file in \
  "$PAPER_DIR/cover_letter_draft.md" \
  "$PAPER_DIR/submission_metadata.md" \
  "$PAPER_DIR/PRESUBMISSION_REVIEW_MEMO.md" \
  "$PAPER_DIR/EXPERIMENT_UPGRADE_PLAN.md" \
  "$PAPER_DIR/BUILD_DIAGNOSTICS.md" \
  "$PAPER_DIR/OVERLEAF_BUILD.md" \
  "$PAPER_DIR/audit_conformer_progress.py" \
  "$PAPER_DIR/plan_conformer_table_update.py" \
  "$PAPER_DIR/SUBMISSION_CHECKLIST.md"; do
  if [ -f "$file" ]; then
    ok "found ${file#$PAPER_DIR/}"
  else
    warning "missing optional submission helper ${file#$PAPER_DIR/}"
  fi
done

if [ -f "$PAPER_DIR/graphical_abstract.png" ]; then
  dimensions="$(file "$PAPER_DIR/graphical_abstract.png" | sed -n 's/.*PNG image data, \([0-9][0-9]*\) x \([0-9][0-9]*\).*/\1 \2/p')"
  if [ -n "$dimensions" ]; then
    read -r width height <<< "$dimensions"
    if [ "$width" -lt 1328 ] || [ "$height" -lt 531 ]; then
      warning "graphical_abstract.png is ${width}x${height}; Elsevier's general guidance recommends at least 1328x531 px."
    else
      ok "graphical_abstract.png is ${width}x${height}."
    fi
  else
    warning "could not read graphical_abstract.png dimensions."
  fi
else
  warning "missing graphical_abstract.png"
fi

say ""
say "7) TeX toolchain and PDF freshness"
if command -v latexmk >/dev/null 2>&1 && command -v pdflatex >/dev/null 2>&1; then
  ok "TeX Live/MacTeX tools detected."
else
  blocker "latexmk/pdflatex not detected; final PDF build is not verified on this machine."
fi

TECTONIC_BUNDLED="/Users/zkr/.codex/plugins/cache/openai-bundled/latex/0.2.2/bin/tectonic"
if command -v tectonic >/dev/null 2>&1; then
  ok "tectonic is available on PATH."
elif [ -x "$TECTONIC_BUNDLED" ]; then
  warning "bundled tectonic exists but is not on PATH; see BUILD_DIAGNOSTICS.md before using it as a build verifier."
else
  warning "tectonic is not available."
fi

if [ -f "$PAPER_DIR/BUILD_DIAGNOSTICS.md" ]; then
  ok "found BUILD_DIAGNOSTICS.md"
else
  warning "missing BUILD_DIAGNOSTICS.md"
fi

if [ -f "$PAPER_DIR/main.pdf" ]; then
  newer_sources="$(find "$PAPER_DIR" \
    -path "$PAPER_DIR/submission_package" -prune -o \
    \( -name '*.tex' -o -name '*.bib' -o -name '*.bst' \) -newer "$PAPER_DIR/main.pdf" -print)"
  if [ -n "$newer_sources" ]; then
    warning "main.pdf is older than current source files:"
    printf '%s\n' "$newer_sources" | sed "s#^$PAPER_DIR/##"
  else
    ok "main.pdf is newer than TeX/Bib sources."
  fi
else
  blocker "main.pdf is missing."
fi

say ""
say "Summary: $blockers blocker(s), $warnings warning(s)."
if [ "$blockers" -gt 0 ]; then
  exit 1
fi
