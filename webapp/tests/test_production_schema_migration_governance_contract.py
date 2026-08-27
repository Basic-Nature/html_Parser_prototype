from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAIN_WORKFLOW = ROOT / ".github" / "workflows" / "main_ballotlens.yml"
MIGRATION_WORKFLOW = (
    ROOT / ".github" / "workflows" / "production_schema_migration.yml"
)
RUNNER = ROOT / "scripts" / "production" / "governed_schema_migration.py"
REGISTRY = ROOT / "scripts" / "production" / "schema_migration_registry.json"

TARGET = "e7b2c4d91f60"
FROM = "c2a3f7e91b4d"
MIGRATION_PATH = (
    "alembic/versions/"
    "e7b2c4d91f60_governed_workflow_schema_foundation.py"
)
MIGRATION_SHA256 = (
    "a097a6547da1753f8d7c56eb400cb5b2708ebb68fb5ea97b4a371a3e03f2e520"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "governed_schema_migration_contract_target",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_schema_workflow_is_manual_only() -> None:
    text = MIGRATION_WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "\n  push:" not in text
    assert "environment: production-schema-migration" in text
    assert "cancel-in-progress: false" in text


def test_schema_workflow_uses_oidc_and_entra_kudu_not_basic_auth() -> None:
    text = MIGRATION_WORKFLOW.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "id-token: write" in text
    assert "azure/login@v3" in text
    assert "Microsoft Entra bearer token" in text
    assert 'Authorization": f"Bearer {token}"' in runner

    forbidden = (
        "list-publishing-profiles",
        "publishUser",
        "userPWD",
        "SCM Basic Auth",
    )
    for marker in forbidden:
        assert marker not in text
        assert marker not in runner


def test_normal_deploy_never_runs_alembic_and_tracks_migration_source() -> None:
    text = MAIN_WORKFLOW.read_text(encoding="utf-8")
    assert "- 'alembic/**'" in text
    assert "alembic upgrade" not in text
    assert "governed_schema_migration.py" not in text


def test_deploy_uses_current_container_cli_flags() -> None:
    text = MAIN_WORKFLOW.read_text(encoding="utf-8")

    required = (
        "--container-image-name",
        "--container-registry-url",
        "--container-registry-user",
        "--container-registry-password",
    )
    forbidden = (
        "--docker-custom-image-name",
        "--docker-registry-server-url",
        "--docker-registry-server-user",
        "--docker-registry-server-password",
    )
    for marker in required:
        assert marker in text
    for marker in forbidden:
        assert marker not in text


def test_e7b2_transition_is_exactly_allow_listed() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert (
        payload["schema_version"]
        == "electionpulse_production_schema_migration_registry_v1"
    )
    spec = payload["migrations"][TARGET]
    assert spec["target_revision"] == TARGET
    assert spec["from_revision"] == FROM
    assert spec["migration_path"] == MIGRATION_PATH
    assert spec["migration_sha256"] == MIGRATION_SHA256
    assert spec["preserve_canonical_publication_metrics"] is True
    assert spec["expect_new_tables_empty"] is True
    assert set(spec["expected_new_tables"]) == {
        "workflow_items",
        "workflow_passes",
        "workflow_comparisons",
        "workflow_discrepancies",
        "workflow_reviews",
        "workflow_artifact_links",
        "workflow_events",
    }


def test_allow_list_sha_matches_migration_bytes() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    spec = payload["migrations"][TARGET]
    migration = ROOT / spec["migration_path"]

    import hashlib

    assert hashlib.sha256(migration.read_bytes()).hexdigest() == MIGRATION_SHA256


def test_runner_uses_exact_target_not_upgrade_head() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '"upgrade",' in text
    assert 'str(spec["target_revision"])' in text
    assert '"upgrade", "head"' not in text
    assert "'upgrade', 'head'" not in text
    assert "alembic upgrade head" not in text


def test_runner_requires_exact_apply_confirmation() -> None:
    module = _load_runner()
    registry = module.load_registry(REGISTRY)
    spec = module.get_spec(registry, TARGET)
    assert module.expected_confirmation(spec) == f"APPLY:{FROM}:{TARGET}"


def test_runner_preserves_accepted_tls_contract_and_deployed_sha_gate() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'state["client_cert_enabled"] is not True' in text
    assert 'state["client_cert_mode"] != "OptionalInteractiveUser"' in text
    assert "require_deployed_commit" in text
    assert 'os.environ.get("GITHUB_SHA"' in text


def test_runner_derives_kudu_scm_authority_from_azure_metadata() -> None:
    module = _load_runner()
    unique_scm = (
        "ballotlens-cubrcudretaebca9.scm."
        "westus3-01.azurewebsites.net"
    )
    site = {
        "enabledHostNames": [
            "ballotlens-cubrcudretaebca9.westus3-01.azurewebsites.net",
            unique_scm,
        ],
        "hostNameSslStates": [
            {
                "name": (
                    "ballotlens-cubrcudretaebca9."
                    "westus3-01.azurewebsites.net"
                ),
                "hostType": "Standard",
            },
            {
                "name": unique_scm,
                "hostType": "Repository",
            },
        ],
    }

    assert module.resolve_scm_base(site) == f"https://{unique_scm}"

    legacy_projection = {
        "enabledHostNames": [
            "ballotlens.azurewebsites.net",
            "ballotlens.scm.azurewebsites.net",
        ]
    }
    assert (
        module.resolve_scm_base(legacy_projection)
        == "https://ballotlens.scm.azurewebsites.net"
    )

    runner_text = RUNNER.read_text(encoding="utf-8")
    assert 'SCM_BASE = "https://ballotlens.scm.azurewebsites.net"' not in runner_text
    assert '"scm_base": resolve_scm_base(site)' in runner_text
    assert "scm_base=scm_base" in runner_text


def test_runner_keeps_canonical_metrics_as_migration_invariants() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "canonical_result_count" in text
    assert "canonical_race_count" in text
    assert "canonical_total_votes_sum" in text
    assert "Canonical publication metrics changed during migration" in text


def test_data_framework_warehouse_status_retires_legacy_workflow_contests() -> None:
    import ast

    path = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "api_data_framework_warehouse_status"
    ]
    assert len(nodes) == 1
    function = ast.get_source_segment(text, nodes[0]) or ""

    assert "workflow.contests" not in function
    assert "workflow_items" in function
    assert "canonical_race_id" in function
    assert "workflow_schema_not_provisioned" in function
    assert "e7b2c4d91f60" in function
    assert '"legacy_workflow_contests": "retired"' in function
    assert "ensure_db_tables()" not in function
    assert "SET TRANSACTION READ ONLY" in function
    assert '"unavailable_counts": "null"' in function
    assert '"identity_fields": "not_exposed"' in function


def test_data_framework_neighbor_handlers_survive_warehouse_status_patch() -> None:
    import ast

    path = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert names.count("api_data_framework_warehouse_status") == 1
    assert names.count("api_data_framework_exports") == 1
    assert names.count("api_data_framework_canonical_facets") == 1

    warehouse_idx = names.index("api_data_framework_warehouse_status")
    exports_idx = names.index("api_data_framework_exports")
    canonical_idx = names.index("api_data_framework_canonical_facets")

    assert warehouse_idx < exports_idx < canonical_idx
