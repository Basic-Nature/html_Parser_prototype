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


def test_runner_uses_unique_temporary_webjob_name_per_github_attempt() -> None:
    module = _load_runner()

    assert module.WEBJOB_BASE_NAME == "ElectionPulseGovernedSchemaMigration"
    assert module.build_governed_webjob_name("33041848008", "1") == (
        "ElectionPulseGovernedSchemaMigration-33041848008-1"
    )
    assert module.build_governed_webjob_name("33041848008", "2") == (
        "ElectionPulseGovernedSchemaMigration-33041848008-2"
    )
    assert module.build_governed_webjob_name("", "") == module.WEBJOB_BASE_NAME

    for run_id, attempt in (("not-a-run", "1"), ("33041848008", "x")):
        try:
            module.build_governed_webjob_name(run_id, attempt)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Malformed GitHub run identity must fail closed.")

    runner_text = RUNNER.read_text(encoding="utf-8")
    assert 'WEBJOB_NAME = build_governed_webjob_name()' in runner_text
    assert '"temporary_webjob_name": WEBJOB_NAME' in runner_text
    assert 'WEBJOB_NAME = "ElectionPulseGovernedSchemaMigration"' not in runner_text


def test_runner_uploads_webjob_zip_with_required_filename_header() -> None:
    module = _load_runner()
    captured = {}

    def fake_kudu_request(
        token,
        method,
        path_or_url,
        *,
        scm_base,
        body=None,
        content_type=None,
        content_disposition=None,
        timeout=60,
    ):
        captured.update(
            {
                "token": token,
                "method": method,
                "path_or_url": path_or_url,
                "scm_base": scm_base,
                "body": body,
                "content_type": content_type,
                "content_disposition": content_disposition,
                "timeout": timeout,
            }
        )
        return 200, {}, b""

    module.kudu_request = fake_kudu_request
    module.upload_job(
        "entra-token",
        b"zip-bytes",
        scm_base="https://example.scm.azurewebsites.net",
    )

    assert captured["method"] == "PUT"
    assert captured["content_type"] == "application/zip"
    assert captured["content_disposition"] == (
        f"attachment; filename={module.WEBJOB_NAME}.zip"
    )
    assert captured["body"] == b"zip-bytes"
    assert captured["timeout"] == 90


def test_runner_waits_for_uploaded_webjob_registration() -> None:
    module = _load_runner()
    calls = []
    responses = [
        module.KuduHttpError(
            status=404,
            method="GET",
            path=f"/api/triggeredwebjobs/{module.WEBJOB_NAME}",
            body="not found yet",
        ),
        (
            200,
            {},
            json.dumps(
                {
                    "name": module.WEBJOB_NAME,
                    "type": "triggered",
                    "runCommand": "python run.py",
                }
            ).encode("utf-8"),
        ),
    ]

    def fake_kudu_request(*args, **kwargs):
        calls.append((args, kwargs))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    original_sleep = module.time.sleep
    module.kudu_request = fake_kudu_request
    module.time.sleep = lambda _: None
    try:
        payload = module.wait_for_uploaded_job_registration(
            "entra-token",
            scm_base="https://example.scm.azurewebsites.net",
            timeout_seconds=30,
        )
    finally:
        module.time.sleep = original_sleep

    assert payload["name"] == module.WEBJOB_NAME
    assert payload["type"] == "triggered"
    assert payload["runCommand"] == "python run.py"
    assert len(calls) == 2
    assert calls[0][0][1] == "GET"


def test_runner_retries_only_exact_kudu_trigger_state_conflict() -> None:
    module = _load_runner()
    calls = []

    exact_conflict = module.KuduHttpError(
        status=500,
        method="POST",
        path=f"/api/triggeredwebjobs/{module.WEBJOB_NAME}/run",
        body=(
            '{"code":"Kudu.Core.Hooks.ConflictException",'
            '"message":"Operation is not valid due to the current state '
            'of the object."}'
        ),
    )

    responses = [
        exact_conflict,
        (
            202,
            {"location": "https://example/history/123"},
            b"",
        ),
    ]

    def fake_kudu_request(*args, **kwargs):
        calls.append((args, kwargs))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    original_sleep = module.time.sleep
    module.kudu_request = fake_kudu_request
    module.time.sleep = lambda _: None
    try:
        location = module.trigger_job(
            "entra-token",
            scm_base="https://example.scm.azurewebsites.net",
            mode="preflight",
            target="e7b2c4d91f60",
            confirmation="",
        )
    finally:
        module.time.sleep = original_sleep

    assert location == "https://example/history/123"
    assert len(calls) == 2

    unrelated = module.KuduHttpError(
        status=500,
        method="POST",
        path=f"/api/triggeredwebjobs/{module.WEBJOB_NAME}/run",
        body='{"code":"OtherFailure","message":"different server error"}',
    )

    module.kudu_request = lambda *args, **kwargs: (_ for _ in ()).throw(unrelated)
    try:
        module.trigger_job(
            "entra-token",
            scm_base="https://example.scm.azurewebsites.net",
            mode="preflight",
            target="e7b2c4d91f60",
            confirmation="",
        )
    except module.KuduHttpError as exc:
        assert exc is unrelated
    else:
        raise AssertionError("Unrelated Kudu HTTP 500 must not be retried/swallowed.")


def test_runner_parses_kudu_prefixed_governed_worker_marker() -> None:
    module = _load_runner()
    payload = {
        "schema": "electionpulse_governed_schema_migration_worker_v1",
        "mode": "preflight",
        "result": "PASS",
        "database_mutation": "NONE",
    }
    output = (
        "[08/27/2026 04:35:00 > abc123: SYS INFO] Status changed to Running\n"
        "[08/27/2026 04:35:01 > abc123: INFO] "
        + module.RESULT_MARKER
        + json.dumps(payload, sort_keys=True)
        + "\n"
        "[08/27/2026 04:35:01 > abc123: SYS INFO] Status changed to Success\n"
    )

    assert module.parse_worker_result(output) == payload


def test_runner_preserves_webjob_stdout_and_stderr_before_marker_parse() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'error_url = payload.get("error_url")' in text
    assert 'evidence["webjob_status"] = run_payload.get("status")' in text
    assert 'evidence["webjob_output_tail"] = safe_output[-12000:]' in text
    assert 'evidence["webjob_error_tail"] = safe_error_output[-12000:]' in text
    assert text.index('evidence["webjob_output_tail"]') < text.index(
        'worker = parse_worker_result(output)'
    )


def test_runner_redacts_database_password_from_preserved_webjob_logs() -> None:
    module = _load_runner()
    original = module.os.environ.get("POSTGRES_PASSWORD")
    module.os.environ["POSTGRES_PASSWORD"] = "super-secret-db-password"
    try:
        sanitized = module.sanitize_webjob_log(
            "failure password=super-secret-db-password"
        )
    finally:
        if original is None:
            module.os.environ.pop("POSTGRES_PASSWORD", None)
        else:
            module.os.environ["POSTGRES_PASSWORD"] = original

    assert "super-secret-db-password" not in sanitized
    assert "<REDACTED_PASSWORD>" in sanitized


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
