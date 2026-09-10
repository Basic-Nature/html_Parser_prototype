from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4

from flask import Flask

from webapp.parser.routes.workflow_reviewer_blueprint import (
    create_workflow_reviewer_blueprint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
SERVICE_PATH = (
    REPO_ROOT / "webapp" / "parser" / "services" / "workflow_reviews.py"
)


def test_qc_reviewer_blueprint_dispatch_and_get_guards():
    app = Flask(__name__)
    item_id = uuid4()
    app.config["_WORKFLOW_REVIEWER_ROUTE_HANDLERS"] = {
        "api_workflow_v1_resolve_discrepancies":
            lambda item_id, comparison_id: ({"success": True}, 200),
        "api_workflow_v1_submit_qc1_review":
            lambda item_id: (
                {
                    "success": True,
                    "item_id": str(item_id),
                    "review_stage": "qc1",
                },
                200,
            ),
        "api_workflow_v1_submit_qc2_review":
            lambda item_id: (
                {
                    "success": True,
                    "item_id": str(item_id),
                    "review_stage": "qc2",
                },
                200,
            ),
    }
    app.register_blueprint(create_workflow_reviewer_blueprint())
    client = app.test_client()

    qc1 = client.post(
        f"/api/workflow/v1/reviewer/items/{item_id}/reviews/qc1"
    )
    assert qc1.status_code == 200
    assert qc1.get_json()["review_stage"] == "qc1"
    assert (
        client.get(
            f"/api/workflow/v1/reviewer/items/{item_id}/reviews/qc1"
        ).status_code
        == 405
    )

    qc2 = client.post(
        f"/api/workflow/v1/reviewer/items/{item_id}/reviews/qc2"
    )
    assert qc2.status_code == 200
    assert qc2.get_json()["review_stage"] == "qc2"
    assert (
        client.get(
            f"/api/workflow/v1/reviewer/items/{item_id}/reviews/qc2"
        ).status_code
        == 405
    )


def test_composition_root_qc1_qc2_authority_contract():
    source = APP_PATH.read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", source)

    for token in (
        "WorkflowQCReviewError",
        "record_workflow_qc_review",
        "CAP_QC1_REVIEW",
        "CAP_QC2_REVIEW",
        "_WORKFLOW_REVIEWER_QC1_REQUEST_KEYS",
        "_WORKFLOW_REVIEWER_QC2_REQUEST_KEYS",
        "WORKFLOW_REVIEWER_MUTATIONS_ENABLED",
    ):
        assert token in source

    for key in (
        "expected_row_version",
        "decision",
        "checklist_version",
        "checklist_result",
        "reason_codes",
        "notes",
    ):
        assert f'"{key}"' in source

    assert (
        "body_keys!=_WORKFLOW_REVIEWER_QC1_REQUEST_KEYS" in compact
    )
    assert (
        "body_keys!=_WORKFLOW_REVIEWER_QC2_REQUEST_KEYS" in compact
    )
    assert "_workflow_reviewer_authority(CAP_QC1_REVIEW)" in compact
    assert "_workflow_reviewer_authority(CAP_QC2_REVIEW)" in compact
    assert 'review_stage="qc1"' in compact
    assert 'review_stage="qc2"' in compact
    assert '"api_workflow_v1_submit_qc1_review"' in source
    assert '"api_workflow_v1_submit_qc2_review"' in source


def test_qc_review_service_boundary_is_commit_free_and_server_selected():
    source = SERVICE_PATH.read_text(encoding="utf-8")

    assert (
        'WORKFLOW_QC_REVIEW_CONTRACT = "workflow_qc_review_v1"'
        in source
    )
    assert "def record_workflow_qc_review(" in source
    assert "stage not in REVIEW_STAGES" in source
    assert "def _load_approved_qc1_authority(" in source
    assert "approved_qc1_review_inherited" in source
    assert (
        "QC2 reviewer must differ from DL1, DL2, and QC1 principals."
        in source
    )

    assert "session.commit(" not in source
    assert "session.rollback(" not in source
    assert "canonical_writer(" not in source
    assert "client_selected_pass_id" not in source
    assert "client_selected_staging_batch_id" not in source
    assert "strict_equal_dl1" in source
    assert "DISCREPANCY_RESOLUTION_SELECTION_CODES" in source
    assert '"canonical_writer_invoked": False' in source
    assert '"mixed_side_value_merge": False' in source
    assert '"direct_value_edit": False' in source
    assert "WorkflowReview(" in source
    assert "WorkflowEvent(" in source
