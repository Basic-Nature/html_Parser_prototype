from __future__ import annotations

import ast
from pathlib import Path

import pytest

from webapp.parser.auth.capability_policy import (
    Capability,
    CapabilityPolicyError,
    PUBLIC_READ_SURFACES,
    assert_public_read_surface,
)


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
QA_JS = ROOT / "webapp" / "static" / "js" / "quality_assurance_panel.js"
QA_ENDPOINTS = ROOT / "webapp" / "parser" / "quality_assurance" / "qa_endpoints.py"


def _function_source(path: Path, name: str, *, decorators: bool = False) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1, (path, name, len(matches))
    node = matches[0]
    start = node.lineno
    if decorators and node.decorator_list:
        start = min(dec.lineno for dec in node.decorator_list)
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(lines[start - 1 : end])


@pytest.mark.parametrize("surface", sorted(PUBLIC_READ_SURFACES))
def test_public_read_capability_is_read_method_only(surface):
    assert assert_public_read_surface(surface, "GET") is Capability.PUBLIC_READ
    assert assert_public_read_surface(surface, "HEAD") is Capability.PUBLIC_READ
    with pytest.raises(CapabilityPolicyError):
        assert_public_read_surface(surface, "POST")


def test_unknown_public_surface_fails_closed():
    with pytest.raises(CapabilityPolicyError):
        assert_public_read_surface("raw_database_admin", "GET")


@pytest.mark.parametrize(
    ("function_name", "surface"),
    [
        ("api_ballotlens_database", "ballotlens_canonical"),
        ("api_data_framework_scaffold", "data_framework_scaffold"),
        ("api_data_framework_scaffold_csv", "data_framework_scaffold_csv"),
        ("api_data_framework_curated", "data_framework_curated"),
        ("api_data_framework_warehouse_status", "data_framework_warehouse_status"),
        ("api_data_framework_canonical_facets", "data_framework_canonical_facets"),
        ("api_election_data_states_counties", "election_data_states_counties"),
        ("api_election_data_stats", "election_data_stats"),
    ],
)
def test_reviewed_public_handlers_use_capability_seam(function_name, surface):
    body = _function_source(MAIN, function_name)
    assert f'assert_public_read_surface("{surface}", request.method)' in body
    assert "if not principal and not ALLOW_DEV_NO_PRINCIPAL" not in body


def test_data_framework_preview_remains_protected_and_stateful():
    body = _function_source(MAIN, "api_data_framework_preview")
    assert "get_request_principal()" in body
    assert "if not principal and not ALLOW_DEV_NO_PRINCIPAL" in body
    assert "session.commit()" in body
    assert "DataFrameworkPreviewCache" in body


def test_workflow_detail_private_aggregates_public():
    helper = _function_source(MAIN, "_workflow_v1_read")
    items = _function_source(MAIN, "api_workflow_v1_items")
    detail = _function_source(MAIN, "api_workflow_v1_item_detail")
    facets = _function_source(MAIN, "api_workflow_v1_facets")
    stats = _function_source(MAIN, "api_workflow_v1_stats")
    assert "public_surface: str | None = None" in helper
    assert "get_request_principal()" in helper
    assert "if public_surface:" in helper
    assert "public_surface=" not in items
    assert "public_surface=" not in detail
    assert 'public_surface="workflow_v1_facets"' in facets
    assert 'public_surface="workflow_v1_stats"' in stats


def test_worklist_public_projection_redacts_assignments():
    body = _function_source(MAIN, "api_election_data_worklist")
    assert '"election_data_worklist_public_projection"' in body
    assert "public_projection = not bool(principal)" in body
    assert "def _public_worklist_record" in body
    for field in (
        "dl1_assigned_to",
        "dl2_assigned_to",
        "qc1_assigned_to",
        "qc2_assigned_to",
        "qc1_selected_dl",
    ):
        assert f'"{field}"' in body
    assert 'public_record["visibility"] = "public_projection"' in body


def test_worklist_overview_remains_principal_guarded():
    body = _function_source(MAIN, "api_election_data_worklist_overview")
    assert "get_request_principal()" in body
    assert "if not principal and not ALLOW_DEV_NO_PRINCIPAL" in body


def test_qa_reviewer_queues_not_polled_anonymously():
    source = QA_JS.read_text(encoding="utf-8")
    assert "async function hasTrustedSession()" in source
    assert source.count("if (!(await hasTrustedSession()))") >= 2
    assert "restricted: true" in source


def test_classification_and_promotion_remain_reviewer_guarded():
    classify = _function_source(QA_ENDPOINTS, "parse_and_classify", decorators=True)
    promote = _function_source(QA_ENDPOINTS, "verify_and_promote", decorators=True)
    assert '@qa_bp.route("/parse-and-classify", methods=["POST"])' in classify
    assert "@_require_reviewer" in classify
    assert '@qa_bp.route("/verify-and-promote", methods=["POST"])' in promote
    assert "@_require_reviewer" in promote
    assert '@_require_reviewer_tier("admin_reviewer")' in promote
