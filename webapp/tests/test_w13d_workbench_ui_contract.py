from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(".")
MAIN = ROOT / "webapp/Smart_Elections_Parser_Webapp.py"
BLUEPRINT = ROOT / "webapp/parser/routes/workflow_contributor_blueprint.py"
TEMPLATE = ROOT / "webapp/templates/worklist.html"
JS = ROOT / "webapp/static/js/workflow_public.js"
CSS = ROOT / "webapp/static/css/workflow_public.css"
TRUSTED = ROOT / "webapp/frontend/ballot-lens/services/trustedExecution.ts"
HANDOFF = ROOT / "webapp/frontend/ballot-lens/services/workflowHandoff.ts"
SOURCE_BROWSER = (
    ROOT / "webapp/frontend/ballot-lens/components/source/TrustedSourceBrowser.tsx"
)
APP_SHELL = ROOT / "webapp/frontend/ballot-lens/app/AppShell.tsx"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _function(path: Path, name: str) -> str:
    source = _read(path)
    tree = ast.parse(source, filename=str(path))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


def test_ballot_lens_trusted_ui_is_server_capability_projected():
    body = _function(MAIN, "ballot_lens")

    assert "get_request_principal()" in body
    compact = "".join(body.split())
    assert (
        "ballot_lens_operator_access="
        "_workflow_operator_access_projection(principal)"
        in compact
    )
    assert '"can_execute_ballot_lens"' in body
    assert "ballot_lens_trusted_controls = bool(" in body
    assert "ballot_lens_trusted_controls = bool(principal)" not in body
    assert body.index("ballot_lens_trusted_controls") < body.index(
        "get_all_file_lists()"
    )


def test_worklist_server_projects_operator_state_without_identity():
    route = _function(MAIN, "worklist")
    helper = _function(MAIN, "_workflow_operator_access_projection")
    template = _read(TEMPLATE)

    route_compact = "".join(route.split())
    assert "get_request_principal()" in route
    assert "_workflow_operator_access_projection(principal)" in route_compact
    assert "workflow_operator_access=workflow_operator_access" in route_compact

    assert "resolve_workflow_roles_for_principal" in helper
    assert "project_workflow_operator_access" in helper
    assert '"principal_disclosed": False' in helper
    assert "WORKFLOW_OPERATOR_ACCESS_CONTRACT" in helper

    assert "data-workflow-operator-access=" in template
    assert "data-ballot-lens-url=" in template
    assert "Operator Workbench" in template
    assert "Operator Access" in template
    for forbidden in (
        "assigned_principal",
        "reviewer_principal",
        "created_by_principal",
    ):
        assert forbidden not in template


def test_handoff_route_is_authenticated_get_only_and_nonmutating():
    blueprint = _read(BLUEPRINT)
    main = _read(MAIN)

    assert (
        '"/api/workflow/v1/contributor/items/<uuid:item_id>/'
        'ballot-lens-handoff"'
        in blueprint
    )
    assert 'methods=["GET"]' in blueprint
    assert "api_workflow_v1_ballot_lens_handoff" in blueprint

    handler = _function(MAIN, "api_workflow_v1_ballot_lens_handoff")
    handler_compact = "".join(handler.split())
    assert (
        "_workflow_contributor_authority(CAP_BALLOT_LENS_EXECUTE)"
        in handler_compact
    )
    assert "resolve_workflow_roles_for_principal(principal)" in handler
    assert "project_workflow_ballot_lens_handoff(" in handler
    service = _read(
        ROOT / "webapp/parser/services/workflow_ballot_lens_handoff.py"
    )
    assert 'safe.get("authorized")' not in service
    assert 'safe.get("principal_disclosed")' not in service
    assert 'safe.get("execution_mode") != "workflow"' in service
    assert "authority.resolved_source_url in repr(safe)" in service
    assert "db_session.commit(" not in handler
    assert "request.get_json" not in handler
    assert "source_url" not in handler


def test_workbench_browser_handoff_never_constructs_url_authority():
    source = _read(JS)

    assert (
        "/api/workflow/v1/contributor/items/"
        in source
    )
    assert "/ballot-lens-handoff" in source
    assert "workflow_item_id" in source
    assert "workflow_pass_id" in source
    assert "expected_row_version" in source
    assert "window.location.assign" in source
    assert "method: 'GET'" in source

    for forbidden in (
        "direct_urls",
        "assigned_principal",
        "reviewer_principal",
        "created_by_principal",
    ):
        assert forbidden not in source

    assert "method: 'POST'" not in source
    assert "method: 'PUT'" not in source
    assert "method: 'DELETE'" not in source


def test_ballot_lens_worklist_mode_emits_w13c_exact_id_payload():
    trusted = _read(TRUSTED)
    handoff = _read(HANDOFF)
    browser = _read(SOURCE_BROWSER)
    shell = _read(APP_SHELL)

    assert "workflowItemId" in trusted
    assert "workflowPassId" in trusted
    assert "expectedRowVersion" in trusted
    assert "workflow_item_id" in trusted
    assert "workflow_pass_id" in trusted
    assert "expected_row_version" in trusted
    assert "if(s.runMode==='worklist')" in "".join(trusted.split())

    assert "WORKFLOW_HANDOFF_QUERY_KEYS" in handoff
    assert "readWorkflowHandoffQueryIntent" in handoff
    assert "clearWorkflowHandoffQuery" in handoff

    assert "fetchTrustedWorklist" not in browser
    assert "Governed Workflow task ready" in browser
    assert 'href="/worklist"' in browser

    assert "readWorkflowHandoffQueryIntent" in shell
    assert "initialWorkflowHandoffIntentRef" in shell
    assert "setActiveMode('worklist')" in shell
    assert "clearWorkflowHandoffQuery" in shell


def test_public_surface_copy_and_accessibility_contracts_remain_present():
    template = _read(TEMPLATE)
    css = _read(CSS)

    assert "Public verification workflow" in template
    assert "identity-safe public projection" in template
    assert "Public visibility now; contributor actions next" in template
    assert 'aria-live="polite"' in template
    assert "workflow-empty-state" in template

    assert "W1 PUBLIC WORKFLOW PARTICIPATION FOUNDATION" in css
    assert ".workflow-operator" in css
    assert ".workflow-action-button" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion: reduce" in css
    assert css.count("{") == css.count("}")
