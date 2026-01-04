#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
REQUIRED_VARS=(FLASK_SECRET_KEY POSTGRES_HOST POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD)

log() { echo "[run_webapp] $*"; }
fatal() { echo "[run_webapp][fatal] $*" >&2; exit 1; }

cd "$ROOT_DIR"

if [[ -f "$ENV_FILE" ]]; then
  log "loading env from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  log "env file not found at $ENV_FILE (set ENV_FILE to override)"
fi

missing=()
for var in "${REQUIRED_VARS[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    missing+=("$var")
  fi
done

if (( ${#missing[@]} )); then
  fatal "Missing required env vars: ${missing[*]}. Set them in $ENV_FILE or environment."
fi

# Ensure runtime directories exist
mkdir -p input output uploads log

if [[ -z "${EMBEDDING_CACHE_DB_MODE:-}" ]]; then
  env_lower="${DEPLOY_ENV:-local}"; env_lower="${env_lower,,}"
  if [[ -z "$env_lower" || "$env_lower" == "local" || "$env_lower" == "dev" || "$env_lower" == "development" || "$env_lower" == "test" ]]; then
    export EMBEDDING_CACHE_DB_MODE=off
    log "EMBEDDING_CACHE_DB_MODE defaulting to off for $env_lower (override to rw/ro as needed)"
  fi
fi

log "starting webapp via python -m webapp.Smart_Elections_Parser_Webapp"
exec "$PYTHON_BIN" -m webapp.Smart_Elections_Parser_Webapp "$@"
