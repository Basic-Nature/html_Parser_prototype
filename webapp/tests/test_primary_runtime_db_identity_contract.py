from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "main_ballotlens.yml"
CONFIG = ROOT / "webapp" / "parser" / "config.py"

RUNTIME_USER = "electionpulse_app_runtime"
RUNTIME_PASSWORD_REFERENCE = (
    "@Microsoft.KeyVault("
    "VaultName=ballotlens-guardian;"
    "SecretName=electionpulse-runtime-postgres-password"
    ")"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _config_text() -> str:
    return CONFIG.read_text(encoding="utf-8")


def test_main_deploy_uses_dedicated_runtime_role_and_key_vault_reference() -> None:
    text = _workflow_text()

    assert 'PRODUCTION_RUNTIME_DB_USER: "electionpulse_app_runtime"' in text
    assert (
        'PRODUCTION_RUNTIME_DB_PASSWORD_REFERENCE: '
        '"@Microsoft.KeyVault(VaultName=ballotlens-guardian;'
        'SecretName=electionpulse-runtime-postgres-password)"'
    ) in text

    assert 'POSTGRES_USER="${{ env.PRODUCTION_RUNTIME_DB_USER }}"' in text
    assert (
        'POSTGRES_PASSWORD="${{ env.PRODUCTION_RUNTIME_DB_PASSWORD_REFERENCE }}"'
        in text
    )

    assert 'POSTGRES_USER="${{ secrets.POSTGRES_USER }}"' not in text
    assert 'POSTGRES_PASSWORD="${{ secrets.POSTGRES_PASSWORD }}"' not in text
    assert "SecretVersion=" not in RUNTIME_PASSWORD_REFERENCE


def test_runtime_db_identity_guard_precedes_appsettings_mutation() -> None:
    text = _workflow_text()

    guard = "Guard production runtime database identity contract"
    settings = "Set Azure App Settings (Environment Variables)"

    assert guard in text
    assert settings in text
    assert text.index(guard) < text.index(settings)


def test_key_vault_reference_resolution_gate_precedes_explicit_restart() -> None:
    text = _workflow_text()

    verify = "Verify runtime database Key Vault reference resolved"
    restart = "Restart Web App"

    assert verify in text
    assert restart in text
    assert text.index(verify) < text.index(restart)

    assert "/config/configreferences/appsettings/POSTGRES_PASSWORD" in text
    assert "api-version=2026-03-15" in text
    assert "--query properties.status" in text
    assert '[ "$STATUS" = "Resolved" ]' in text
    assert "--query properties.vaultName" in text
    assert "--query properties.secretName" in text
    assert "--query properties.activeVersion" not in text
    assert 'if [ -z "$ACTIVE_VERSION" ]; then' not in text
    assert "Key Vault reference has no active version." not in text
    assert (
        "Reference contract is intentionally unversioned; "
        "activeVersion metadata is not required."
    ) in text
    assert '[ "$VAULT_NAME" != "ballotlens-guardian" ]' in text
    assert '[ "$SECRET_NAME" != "electionpulse-runtime-postgres-password" ]' in text
    assert "SecretVersion=" not in RUNTIME_PASSWORD_REFERENCE


def test_application_code_consumes_standard_password_env_contract() -> None:
    text = _config_text()

    assert 'POSTGRES_USER_RAW = os.environ.get("POSTGRES_USER", "")' in text
    assert 'POSTGRES_PASSWORD_RAW = os.environ.get("POSTGRES_PASSWORD", "")' in text
    assert 'POSTGRES_AUTH = os.environ.get("POSTGRES_AUTH", "password").lower()' in text
    assert "POSTGRES_URL" in text
    assert "return create_engine(" in text


def test_primary_runtime_patch_does_not_touch_governed_migration_identity() -> None:
    text = _workflow_text()

    assert "BallotLens-Migration" not in text
    assert "electionpulse_migration" not in text
    assert "alembic upgrade" not in text
    assert "governed_schema_migration.py" not in text
