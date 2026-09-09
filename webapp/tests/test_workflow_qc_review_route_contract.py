from __future__ import annotations
from pathlib import Path
import re
from uuid import uuid4
from flask import Flask
from webapp.parser.routes.workflow_reviewer_blueprint import create_workflow_reviewer_blueprint
REPO_ROOT=Path(__file__).resolve().parents[2]
APP_PATH=REPO_ROOT/"webapp"/"Smart_Elections_Parser_Webapp.py"
SERVICE_PATH=REPO_ROOT/"webapp"/"parser"/"services"/"workflow_reviews.py"

def test_qc1_reviewer_blueprint_dispatch_and_get_guard():
    app=Flask(__name__); item_id=uuid4()
    app.config["_WORKFLOW_REVIEWER_ROUTE_HANDLERS"]={"api_workflow_v1_resolve_discrepancies":lambda item_id,comparison_id:({"success":True},200),"api_workflow_v1_submit_qc1_review":lambda item_id:({"success":True,"item_id":str(item_id),"review_stage":"qc1"},200)}
    app.register_blueprint(create_workflow_reviewer_blueprint()); client=app.test_client()
    r=client.post(f"/api/workflow/v1/reviewer/items/{item_id}/reviews/qc1"); assert r.status_code==200 and r.get_json()["review_stage"]=="qc1"
    assert client.get(f"/api/workflow/v1/reviewer/items/{item_id}/reviews/qc1").status_code==405

def test_composition_root_qc1_authority_contract():
    source=APP_PATH.read_text(encoding="utf-8"); compact=re.sub(r"\s+","",source)
    assert "WorkflowQCReviewError" in source and "record_workflow_qc_review" in source and "CAP_QC1_REVIEW" in source and "_WORKFLOW_REVIEWER_QC1_REQUEST_KEYS" in source
    for key in ("expected_row_version","decision","checklist_version","checklist_result","reason_codes","notes"): assert f'"{key}"' in source
    assert "body_keys!=_WORKFLOW_REVIEWER_QC1_REQUEST_KEYS" in compact and "_workflow_reviewer_authority(CAP_QC1_REVIEW)" in compact and "WORKFLOW_REVIEWER_MUTATIONS_ENABLED" in source and 'review_stage="qc1"' in compact and '"api_workflow_v1_submit_qc1_review"' in source

def test_qc1_service_boundary_contract_is_commit_free_and_server_selected():
    source=SERVICE_PATH.read_text(encoding="utf-8")
    assert 'WORKFLOW_QC_REVIEW_CONTRACT = "workflow_qc_review_v1"' in source and "def record_workflow_qc_review(" in source and 'stage != "qc1"' in source
    assert "session.commit(" not in source and "session.rollback(" not in source and "canonical_writer(" not in source
    assert "client_selected_pass_id" not in source and "client_selected_staging_batch_id" not in source and "strict_equal_dl1" in source and "DISCREPANCY_RESOLUTION_SELECTION_CODES" in source
    assert "QC1 reviewer must differ from DL1 and DL2 principals." in source and '"canonical_writer_invoked": False' in source and '"mixed_side_value_merge": False' in source and '"direct_value_edit": False' in source
    assert "WorkflowReview(" in source and "WorkflowEvent(" in source
