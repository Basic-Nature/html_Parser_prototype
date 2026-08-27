from __future__ import annotations

from pathlib import Path

from webapp.parser.auth.capability_policy import (
    Capability,
    PUBLIC_READ_SURFACES,
    assert_public_read_surface,
)
from webapp.parser.services.public_read_runtime import (
    _project_public_workflow_item,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = (
    REPO_ROOT / "webapp" / "parser" / "services" / "public_read_runtime.py"
)
BLUEPRINT = (
    REPO_ROOT / "webapp" / "parser" / "routes" / "workflow_blueprint.py"
)
TEMPLATE = REPO_ROOT / "webapp" / "templates" / "worklist.html"
PUBLIC_JS = REPO_ROOT / "webapp" / "static" / "js" / "workflow_public.js"


def _keys(value):
    found = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            found.add(str(key))
            found.update(_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_keys(nested))
    return found


def test_public_workflow_projection_strictly_omits_identity_and_internal_metadata():
    raw = {
        "id": "workflow-123",
        "authority": "operational_workflow",
        "canonical_authority": False,
        "lifecycle_state": "active",
        "current_stage": "independent_acquisition",
        "stage_condition": "in_progress",
        "priority": 7,
        "scope": {
            "election_year": 2024,
            "election_date": None,
            "state": "Arizona",
            "jurisdiction_name": "Pima",
            "jurisdiction_type": "county",
            "contest": "President",
            "office_basic": "President",
            "election_type": "general",
            "source_race_id": "AZ-2024-PRES",
        },
        "source_url": "https://secret.example/source",
        "canonical_reference": {
            "race_id": "internal-canonical-id",
            "linked": True,
            "lineage_inferred": False,
        },
        "blocker": {
            "reason_code": "source_missing",
            "detail": "internal reviewer note",
        },
        "created_by_principal": "real-person@example.com",
        "workflow_metadata": {
            "reviewer": "real-person@example.com",
        },
        "row_version": 4,
        "created_at": "2026-08-27T00:00:00+00:00",
        "updated_at": "2026-08-27T01:00:00+00:00",
    }

    public = _project_public_workflow_item(raw)

    assert public["id"] == "workflow-123"
    assert public["scope"]["jurisdiction_name"] == "Pima"
    assert public["scope"]["jurisdiction_type"] == "county"
    assert public["canonical_reference"] == {
        "linked": True,
        "lineage_inferred": False,
    }
    assert public["blocker"] == {"reason_code": "source_missing"}
    assert public["visibility"] == "public_projection"
    assert public["contribution"]["actions_enabled"] is False

    forbidden = {
        "created_by_principal",
        "assigned_principal",
        "reviewer_principal",
        "resolved_by_principal",
        "actor_principal",
        "workflow_metadata",
        "source_url",
        "detail",
        "race_id",
        "row_version",
    }
    assert _keys(public).isdisjoint(forbidden)


def test_public_workflow_items_is_explicit_get_only_capability():
    assert "workflow_v1_public_items" in PUBLIC_READ_SURFACES
    assert (
        assert_public_read_surface("workflow_v1_public_items", "GET")
        is Capability.PUBLIC_READ
    )


def test_raw_workflow_items_remain_delegated_private_and_projection_is_separate():
    source = BLUEPRINT.read_text(encoding="utf-8")

    assert '"/api/workflow/v1/items"' in source
    assert 'return _call_handler("api_workflow_v1_items")' in source

    assert '"/api/workflow/v1/public/items"' in source
    assert '"workflow_v1_public_items"' in source
    assert "read_public_workflow_items" in source

    raw_route_at = source.index('"/api/workflow/v1/items"')
    raw_handler_at = source.index(
        'return _call_handler("api_workflow_v1_items")',
        raw_route_at,
    )
    public_route_at = source.index('"/api/workflow/v1/public/items"')

    assert raw_route_at < raw_handler_at < public_route_at


def test_public_workflow_runtime_has_explicit_redaction_semantics():
    source = RUNTIME.read_text(encoding="utf-8")

    assert '"contract": "workflow_public_projection_v1"' in source
    assert '"identity_fields": "redacted"' in source
    assert '"internal_metadata": "omitted"' in source
    assert '"raw_source_url": "omitted"' in source
    assert '"canonical_internal_id": "omitted"' in source
    assert '"null": "preserved_null"' in source
    assert '"zero": "numeric_zero_only"' in source


def test_public_workflow_page_does_not_present_privileged_actions():
    template = TEMPLATE.read_text(encoding="utf-8")
    source = PUBLIC_JS.read_text(encoding="utf-8")

    for privileged_ui in (
        "Assign DL Owner",
        "Save DL1",
        "Save DL2",
        "Run Pre-QC Check Now",
        "Proceed to QC1",
        "Export to Production",
    ):
        assert privileged_ui not in template

    for identity_token in (
        "created_by_principal",
        "assigned_principal",
        "reviewer_principal",
        "resolved_by_principal",
        "actor_principal",
    ):
        assert identity_token not in template
        assert identity_token not in source

    assert "Public visibility now; contributor actions next" in template
