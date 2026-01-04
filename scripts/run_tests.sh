#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

log() {
  echo "[run_tests] $*"
}

cd "$ROOT_DIR"

if [[ -z "${SKIP_RUFF:-}" ]]; then
  if command -v ruff >/dev/null 2>&1; then
    log "ruff check webapp"
    ruff check webapp
  else
    log "ruff not installed; set SKIP_RUFF=1 to skip"
    exit 1
  fi
else
  log "SKIP_RUFF set; skipping ruff"
fi

if [[ -z "${SKIP_MYPY:-}" ]]; then
  if command -v mypy >/dev/null 2>&1; then
    log "mypy (formats + tests)"
    mypy webapp/parser/handlers/formats webapp/tests
  else
    log "mypy not installed; set SKIP_MYPY=1 to skip"
    exit 1
  fi
else
  log "SKIP_MYPY set; skipping mypy"
fi

log "pytest $*"
"$PYTHON_BIN" -m pytest "$@"
