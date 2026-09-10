from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4

from flask import Flask

from webapp.parser.routes.workflow_publication_blueprint import (
    create_workflow_publication_blueprint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
SERVICE_PATH = (
    REPO_ROOT / "webapp" / "parser" / "services" / "workflow_publication.py"
)
ROUTE_PATH = (
    REPO_ROOT / "webapp" / "parser" / "routes" / "workflow_publication_blueprint.py"
)


def test_publication_blueprint_dispatches_post_and_rejects_get():
    app = Flask(__name__)
    item_id = uuid4()
    app.config["_WORKFLOW_PUBLICATION_ROUTE_HANDLERS"] = {
        "api_workflow_v1_publication_handoff": lambda item_id: (
            {"success": True, "item_id": str(item_id)},
            200,
        ),
    }
    app.register_blueprint(create_workflow_publication_blueprint())
    client = app.test_client()
    url = f"/api/workflow/v1/publication/items/{item_id}/handoff"
    response = client.post(url)
    assert response.status_code == 200
    assert response.get_json()["item_id"] == str(item_id)
    assert client.get(url).status_code == 405


def test_composition_root_publication_boundary_is_dedicated_default_off_and_exact_body():
    source = APP_PATH.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", source)

    for token in (
        "create_workflow_publication_blueprint",
        "WorkflowPublicationError",
        "WorkflowPublicationWriterFailure",
        "WorkflowPublicationLinkFailure",
        "publish_workflow_item",
        "build_workflow_canonical_writer",
        "CAP_PUBLICATION_HANDOFF",
        "WORKFLOW_PUBLICATION_MUTATIONS_ENABLED",
        "WORKFLOW_PUBLICATION_OPERATOR_PRINCIPALS",
        "_WORKFLOW_PUBLICATION_REQUEST_KEYS",
        "_WORKFLOW_PUBLICATION_ROUTE_HANDLERS",
        "_WORKFLOW_PUBLICATION_ARTIFACT_LOADER",
        "_WORKFLOW_PUBLICATION_PAYLOAD_ADAPTER",
    ):
        assert token in source

    assert (
        'os.environ.get("WORKFLOW_PUBLICATION_MUTATIONS_ENABLED", "false")'
        in source
    )
    assert '_WORKFLOW_PUBLICATION_REQUEST_KEYS=frozenset({"expected_row_version"})' in compact
    assert "_workflow_publication_authority(CAP_PUBLICATION_HANDOFF)" in compact
    assert "body_keys!=_WORKFLOW_PUBLICATION_REQUEST_KEYS" in compact
    assert 'canonical_writer=build_workflow_canonical_writer(SessionLocal)' in compact
    assert "workflow_session_factory=SessionLocal" in compact
    assert 'normalized_artifact_loader=artifact_loader' in compact
    assert 'comparison_payload_adapter=payload_adapter' in compact
    assert '"api_workflow_v1_publication_handoff"' in source

    for forbidden in (
        "client_selected_pass_id",
        "client_selected_staging_batch_id",
        "client_qc1_review_id",
        "client_qc2_review_id",
        "client_comparison_id",
        "client_idempotency_key",
        "client_canonical_race_id",
    ):
        assert forbidden not in source


def test_publication_route_is_orchestrator_only_and_no_public_writer_endpoint():
    route_source = ROUTE_PATH.read_text(encoding="utf-8")
    service_source = SERVICE_PATH.read_text(encoding="utf-8")
    app_source = APP_PATH.read_text(encoding="utf-8")

    assert route_source.count("@bp.route(") == 1
    assert 'methods=["POST"]' in route_source
    assert "/api/workflow/v1/publication/items/<uuid:item_id>/handoff" in route_source
    assert "canonical_writer(request)" not in route_source
    assert "write_workflow_canonical_publication(" not in route_source
    assert "@bp.route" not in service_source
    assert "from flask" not in service_source
    assert "session.commit(" not in service_source
    assert "session.rollback(" not in service_source
    assert "/canonical-writer" not in route_source
    assert "/canonical_writer" not in route_source
    assert "WORKFLOW_REVIEWER_MUTATIONS_ENABLED" in app_source
    assert "WORKFLOW_PUBLICATION_MUTATIONS_ENABLED" in app_source
