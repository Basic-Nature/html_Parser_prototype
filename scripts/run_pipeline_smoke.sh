#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

log() { echo "[pipeline_smoke] $*"; }

cd "$ROOT_DIR"

SMOKE_TARGETS=(
  "webapp/tests/test_context_coordinator.py"
  "webapp/tests/test_detect.py"
  "webapp/tests/test_librarian.py"
)

log "running targeted pytest smoke"
"$ROOT_DIR/scripts/run_tests.sh" "${SMOKE_TARGETS[@]}" -q --maxfail=1

maybe_run() {
  local label=$1
  shift
  if [[ -f $1 ]]; then
    log "${label}"
    "$PYTHON_BIN" "$@"
  else
    log "skip ${label} (missing $1)"
  fi
}

maybe_run "validate_tests.py" "validate_tests.py"
maybe_run "run_statement_test.py --dry-run" "run_statement_test.py" --dry-run

log "smoke checks complete"
