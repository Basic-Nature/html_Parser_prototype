#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-$ROOT_DIR/artifacts}"

log() { echo "[ci_verify] $*"; }

mkdir -p "$ARTIFACTS_DIR"
cd "$ROOT_DIR"

if command -v ruff >/dev/null 2>&1; then
  log "ruff check webapp"
  ruff check webapp
else
  log "ruff not installed; install ruff or set SKIP_RUFF=1"
  [[ -n "${SKIP_RUFF:-}" ]] || exit 1
fi

if command -v mypy >/dev/null 2>&1; then
  log "mypy (formats + tests)"
  mypy webapp/parser/handlers/formats webapp/tests
else
  log "mypy not installed; install mypy or set SKIP_MYPY=1"
  [[ -n "${SKIP_MYPY:-}" ]] || exit 1
fi

log "pytest with coverage + junit"
"$PYTHON_BIN" -m pytest webapp/tests \
  --maxfail=1 \
  --junitxml="$ARTIFACTS_DIR/junit.xml" \
  --cov=webapp/parser \
  --cov-report=xml:"$ARTIFACTS_DIR/coverage.xml" \
  --cov-report=term

log "done"
