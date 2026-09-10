from __future__ import annotations

import ast
from pathlib import Path

import webapp.parser.socket_ballot_lens_orchestration as orchestration


SOURCE_PATH = Path("webapp/parser/socket_ballot_lens_orchestration.py")


def test_workflow_marker_routes_before_legacy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        orchestration,
        "_handle_workflow_ballot_lens_authority_split",
        lambda payload, hooks: calls.append(("workflow", dict(payload))),
    )
    monkeypatch.setattr(
        orchestration,
        "_initialize_session_and_auth",
        lambda payload, hooks: (_ for _ in ()).throw(
            AssertionError("legacy trusted initialization must not run")
        ),
    )

    payload = {
        "workflow_item_id": "11111111-1111-1111-1111-111111111111",
        "workflow_pass_id": "22222222-2222-2222-2222-222222222222",
        "expected_row_version": 7,
        "direct_urls": ["https://attacker.invalid"],
    }
    orchestration.run_ballot_lens_socket_handler(payload, hooks={})
    assert calls == [("workflow", payload)]


def test_public_marker_remains_first_and_never_falls_into_workflow(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        orchestration,
        "_handle_public_registry_authority_split",
        lambda payload, hooks: calls.append(("public", dict(payload))),
    )
    monkeypatch.setattr(
        orchestration,
        "_handle_workflow_ballot_lens_authority_split",
        lambda payload, hooks: (_ for _ in ()).throw(
            AssertionError("mixed public intent must not reach workflow")
        ),
    )

    payload = {
        "registry_source_id": "blsrc_v1_" + ("a" * 64),
        "workflow_item_id": "11111111-1111-1111-1111-111111111111",
        "workflow_pass_id": "22222222-2222-2222-2222-222222222222",
        "expected_row_version": 7,
    }
    orchestration.run_ballot_lens_socket_handler(payload, hooks={})
    assert calls == [("public", payload)]


def test_workflow_dispatch_uses_only_server_prepared_run_config(monkeypatch):
    calls = []
    internal_url = "https://results.example.gov/server-resolved"
    monkeypatch.setattr(
        orchestration,
        "_initialize_workflow_ballot_lens_authority",
        lambda payload, hooks: {
            "session_id": "sess_workflow_123",
            "principal": "cert:contributor",
            "principal_source": "trusted_session",
            "run_cfg": {
                "requested_source": "input",
                "requested_origin": "server",
                "force_parse_input_file": None,
                "force_parse_format": None,
                "manual_upload_rel": None,
                "warehouse_override_url": "",
                "direct_urls": [internal_url],
                "url_reference_hints": [],
                "trusted_run_mode": "worklist",
            },
        },
    )
    monkeypatch.setattr(
        orchestration,
        "_configure_logging_and_prompt",
        lambda session_id, hooks: calls.append(("logging", session_id)),
    )
    monkeypatch.setattr(
        orchestration,
        "_start_pipeline_worker",
        lambda session_id, principal, principal_source, bypass, run_cfg, hooks:
        calls.append(
            (
                "worker",
                session_id,
                principal,
                bypass,
                tuple(run_cfg["direct_urls"]),
                run_cfg["trusted_run_mode"],
            )
        ),
    )

    browser = {
        "workflow_item_id": "11111111-1111-1111-1111-111111111111",
        "workflow_pass_id": "22222222-2222-2222-2222-222222222222",
        "expected_row_version": 7,
    }
    orchestration._handle_workflow_ballot_lens_authority_split(
        browser,
        {},
    )

    assert calls == [
        ("logging", "sess_workflow_123"),
        (
            "worker",
            "sess_workflow_123",
            "cert:contributor",
            False,
            (internal_url,),
            "worklist",
        ),
    ]
    assert internal_url not in repr(browser)


def test_workflow_initializer_is_provider_neutral_and_skips_legacy_prepare():
    source = SOURCE_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_initialize_workflow_ballot_lens_authority"
    )
    text = ast.get_source_segment(source, helper) or ""

    assert "assert_workflow_runtime_capability" in text
    assert "CAP_BALLOT_LENS_EXECUTE" in text
    assert "build_workflow_ballot_lens_server_context" in text
    assert "authorize_workflow_ballot_lens_execution" in text
    assert "safe_validate_external_url" in text
    assert "allowlist_bypass=False" in text
    assert "_prepare_run_inputs" not in text
    assert "require_cert_for_socket_action" not in text


def test_handler_order_is_public_then_workflow_then_legacy():
    source = SOURCE_PATH.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    handler = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_ballot_lens_socket_handler"
    )
    text = ast.get_source_segment(source, handler) or ""

    public_at = text.index("if _is_public_registry_intent(payload):")
    workflow_at = text.index("if _is_workflow_ballot_lens_intent(payload):")
    legacy_at = text.index("_initialize_session_and_auth(payload, hooks)")
    assert public_at < workflow_at < legacy_at
