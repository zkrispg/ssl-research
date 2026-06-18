#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"
RUNS_DIR="$REPO_ROOT/runs"

required_files=(
  "gca_conformer_det_seld_final.md"
  "gca_conformer_det_seld_final.csv"
  "gca_conformer_det_seld_final.json"
)

usage() {
  cat <<'EOF'
Usage:
  ./sync_final_conformer_files.sh /path/to/source/runs
  ./sync_final_conformer_files.sh user@host:/path/to/ssl-research/runs

Copies only:
  gca_conformer_det_seld_final.md
  gca_conformer_det_seld_final.csv
  gca_conformer_det_seld_final.json
EOF
}

if [ "$#" -ne 1 ]; then
  usage
  exit 2
fi

source_dir="${1%/}"
mkdir -p "$RUNS_DIR"

copy_local() {
  local src="$1"
  for file in "${required_files[@]}"; do
    if [ ! -f "$src/$file" ]; then
      echo "BLOCKER: missing $src/$file"
      exit 1
    fi
  done

  for file in "${required_files[@]}"; do
    cp "$src/$file" "$RUNS_DIR/$file"
    echo "copied $file"
  done
}

copy_remote() {
  local src="$1"
  if ! command -v scp >/dev/null 2>&1; then
    echo "BLOCKER: scp is required for remote sync."
    exit 1
  fi

  for file in "${required_files[@]}"; do
    scp "$src/$file" "$RUNS_DIR/$file"
    echo "copied $file"
  done
}

case "$source_dir" in
  *:*)
    copy_remote "$source_dir"
    ;;
  *)
    copy_local "$source_dir"
    ;;
esac

echo ""
echo "Planning manuscript table updates..."
python3 "$PAPER_DIR/plan_conformer_table_update.py"

echo ""
echo "Checking submission state..."
set +e
"$PAPER_DIR/check_submission_state.sh"
status=$?
set -e

if [ "$status" -ne 0 ]; then
  echo ""
  echo "Submission checker still reports blockers. Review the messages above."
fi

exit "$status"
