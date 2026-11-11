"""Contract for the existing governed W24 OIDC schema lane."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = 'f6a8b22c490e'
PREV = 'd2e33e73c6d2'
ROLE = 'electionpulse_oidc_session'
MIGRATION = ROOT / 'alembic/versions/f6a8b22c490e_oidc_sessions_postgres.py'
REGISTRY = ROOT / 'scripts/production/schema_migration_registry.json'
WORKFLOW = ROOT / '.github/workflows/production_schema_migration.yml'
CONTROLLER = ROOT / 'scripts/production/governed_schema_migration.py'


def test_allowlist_and_source_integrity():
    data = MIGRATION.read_bytes()
    ast.parse(data.decode('utf-8'))
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
    assert registry['schema_version'] == 'electionpulse_production_schema_migration_registry_v1'
    spec = registry['migrations'][REV]
    assert spec['migration_sha256'] == hashlib.sha256(data).hexdigest()
    assert spec['migration_path'] == MIGRATION.relative_to(ROOT).as_posix()
    assert spec['target_revision'] == REV
    assert spec['from_revision'] == PREV
    assert spec['expected_new_tables'] == ['electionpulse_oidc_sessions']
    assert spec['expect_new_tables_empty'] is True
    assert spec['required_session_role'] == ROLE
    assert spec['preserve_canonical_publication_metrics'] is True
    assert len(registry['migrations']) == 5


def test_manual_only_and_existing_controller():
    wf = WORKFLOW.read_text(encoding='utf-8')
    assert wf.count('          - ' + REV) == 1
    assert '  workflow_dispatch:' in wf
    assert '\n  push:' not in wf
    assert 'MIGRATION_EXECUTOR_APP_NAME: BallotLens-Migration' in wf
    controller = CONTROLLER.read_text(encoding='utf-8')
    assert 'OIDC_SESSION_ROLE_NOT_SAFE' in controller
    assert 'OIDC_SESSION_ROLE_PRIVILEGES_INCOMPLETE' in controller
    assert 'OIDC_SESSION_ROLE' not in controller or 'electionpulse_oidc_session' in controller
    assert 'redis' not in wf.lower()
    assert 'redis' not in MIGRATION.read_text(encoding='utf-8').lower()


def test_migration_only_bounded_oidc_table_and_no_automatic_downgrade():
    source = MIGRATION.read_text(encoding='utf-8')
    assert 'revision: str = "' + REV + '"' in source
    assert 'down_revision: str | None = "' + PREV + '"' in source
    for requirement in ('key_sha256', 'value_json', 'expires_at', 'created_at',
                        'octet_length(value_json) BETWEEN 1 AND 2048',
                        'GRANT SELECT, INSERT, DELETE', 'GRANT USAGE ON SCHEMA public'):
        assert requirement in source
    assert 'CREATE ROLE' not in source.upper()
    assert 'raise RuntimeError("OIDC session schema rollback' in source


def test_production_flask_oidc_cookie_requirements():
    source=(ROOT/'webapp/Smart_Elections_Parser_Webapp.py').read_text(encoding='utf-8')
    assert 'app.config["SESSION_COOKIE_HTTPONLY"] = True' in source
    assert 'app.config["SESSION_COOKIE_SAMESITE"] = "Lax"' in source
    assert 'app.config["SESSION_COOKIE_SECURE"]' in source
