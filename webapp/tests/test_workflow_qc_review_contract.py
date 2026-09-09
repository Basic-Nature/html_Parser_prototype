from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from webapp.parser.services.workflow_reviews import WorkflowQCReviewConflict, WorkflowQCReviewError, record_workflow_qc_review
from webapp.parser.utils.models import Base, WorkflowComparison, WorkflowDiscrepancy, WorkflowEvent, WorkflowItem, WorkflowPass, WorkflowReview

@pytest.fixture()
def db_session():
    engine=create_engine("sqlite:///:memory:",future=True)
    Session=sessionmaker(bind=engine,autoflush=False,autocommit=False,expire_on_commit=False)
    Base.metadata.create_all(engine); session=Session()
    try: yield session
    finally: session.rollback(); session.close(); engine.dispose()

def _seed_qc1(session, *, strict_equal=True, resolved_code="select_dl2", row_version=6):
    checked=datetime(2026,9,9,10,0,tzinfo=timezone.utc)
    item=WorkflowItem(id=uuid4(),lifecycle_state="active",current_stage="qc1_review",stage_condition="pending",priority=0,election_year=2024,election_date=None,state="Iowa",jurisdiction_name=None,jurisdiction_type=None,contest="President",office_basic="President",election_type=None,source_race_id="2024PRESIA",source_url="https://sos.example.gov/results.pdf",canonical_race_id=None,blocked_reason_code=None,blocker_detail=None,created_by_principal=None,workflow_metadata={},row_version=row_version)
    dl1=WorkflowPass(id=uuid4(),workflow_item_id=item.id,pass_number=1,pass_label="DL1",revision_number=1,is_current=True,status="submitted",assigned_principal="principal:dl1",source_evidence_ref="evidence:dl1",staging_batch_id=uuid4(),candidate_check_status="complete",candidate_check_result={},semantic_validation_status="complete",semantic_validation_result={},started_at=checked,submitted_at=checked,superseded_at=None,notes=None)
    dl2=WorkflowPass(id=uuid4(),workflow_item_id=item.id,pass_number=2,pass_label="DL2",revision_number=1,is_current=True,status="submitted",assigned_principal="principal:dl2",source_evidence_ref="evidence:dl2",staging_batch_id=uuid4(),candidate_check_status="complete",candidate_check_result={},semantic_validation_status="complete",semantic_validation_result={},started_at=checked,submitted_at=checked,superseded_at=None,notes=None)
    comparison=WorkflowComparison(id=uuid4(),workflow_item_id=item.id,left_pass_id=dl1.id,right_pass_id=dl2.id,comparison_version=1,status="complete",strict_equality_passed=bool(strict_equal),difference_count=0 if strict_equal else 2,difference_summary={"difference_count":0 if strict_equal else 2},checked_at=checked,checked_by_service_version="strict-comparison-test",reviewed_by_principal=None if strict_equal else "principal:resolver",reviewed_at=None if strict_equal else checked,created_at=checked)
    rows=[item,dl1,dl2,comparison]; discrepancies=[]
    if not strict_equal:
        for i in range(2):
            d=WorkflowDiscrepancy(id=uuid4(),comparison_id=comparison.id,workflow_item_id=item.id,category="value_mismatch",semantic_key=["records",f"Precinct {i+1}"],left_value=10,right_value=11,left_value_state="value",right_value_state="value",severity=None,resolution_status="resolved",resolution_code=resolved_code,resolution_notes="Resolved from official evidence.",resolved_by_principal="principal:resolver",resolved_at=checked,created_at=checked)
            discrepancies.append(d); rows.append(d)
    session.add_all(rows); session.flush(); return item,dl1,dl2,comparison,discrepancies

def _review(session,item,**overrides):
    kw={"review_stage":"qc1","principal":"principal:qc1","expected_row_version":item.row_version,"decision":"approved","checklist_version":"qc1-v1","checklist_result":{"source_matches_scope":True,"totals_reconcile":True},"reason_codes":[],"notes":""}; kw.update(overrides)
    return record_workflow_qc_review(session,item.id,**kw)

def test_equal_approved_selects_dl1_and_advances_qc2(db_session):
    item,dl1,dl2,comparison,_=_seed_qc1(db_session); payload=_review(db_session,item)
    assert payload["committed"] is False and payload["already_reviewed"] is False
    assert payload["selected_pass_id"]==str(dl1.id) and payload["selected_staging_batch_id"]==str(dl1.staging_batch_id)
    assert (item.lifecycle_state,item.current_stage,item.stage_condition,item.row_version)==("active","qc2_review","pending",7)
    reviews=db_session.query(WorkflowReview).all(); assert len(reviews)==1; review=reviews[0]
    assert review.review_stage=="qc1" and review.selected_pass_id==dl1.id and review.selected_staging_batch_id==dl1.staging_batch_id
    events=db_session.query(WorkflowEvent).filter(WorkflowEvent.event_type=="qc1_review_approved").all(); assert len(events)==1
    assert events[0].related_comparison_id==comparison.id and events[0].related_review_id==review.id and events[0].event_metadata["canonical_writer_invoked"] is False
    assert dl1.status=="submitted" and dl2.status=="submitted"

def test_mismatch_inherits_resolved_dl2_and_allows_same_resolver(db_session):
    item,_dl1,dl2,_c,_=_seed_qc1(db_session,strict_equal=False,resolved_code="select_dl2",row_version=9)
    p=_review(db_session,item,principal="principal:resolver",expected_row_version=9)
    assert p["selected_pass_id"]==str(dl2.id) and p["selected_pass_label"]=="DL2" and item.current_stage=="qc2_review" and item.row_version==10

def test_qc1_reviewer_must_differ_from_dl_principals(db_session):
    item,*_=_seed_qc1(db_session)
    with pytest.raises(WorkflowQCReviewConflict): _review(db_session,item,principal="principal:dl1")

def test_returned_stays_qc1_awaiting_dependency(db_session):
    item,dl1,*_=_seed_qc1(db_session)
    p=_review(db_session,item,decision="returned",checklist_result={"source_matches_scope":True,"totals_reconcile":False},reason_codes=["totals_mismatch"],notes="Return for governed reacquisition.")
    assert p["selected_pass_id"]==str(dl1.id)
    assert (item.lifecycle_state,item.current_stage,item.stage_condition,item.row_version)==("active","qc1_review","awaiting_dependency",7)
    assert item.blocked_reason_code is None and item.blocker_detail is None

def test_rejected_blocks_qc1(db_session):
    item,*_=_seed_qc1(db_session)
    _review(db_session,item,decision="rejected",checklist_result={"source_matches_scope":False,"totals_reconcile":False},reason_codes=["source_invalid"],notes="Source cannot support publication.")
    assert (item.lifecycle_state,item.current_stage,item.stage_condition,item.row_version)==("blocked","qc1_review","failed",7)
    assert item.blocked_reason_code=="qc1_rejected" and item.blocker_detail=="Source cannot support publication."

def test_approved_requires_all_checks_true_and_no_reasons(db_session):
    item,*_=_seed_qc1(db_session)
    with pytest.raises(WorkflowQCReviewError): _review(db_session,item,checklist_result={"source_matches_scope":True,"totals_reconcile":False})
    with pytest.raises(WorkflowQCReviewError): _review(db_session,item,reason_codes=["manual_override"])

def test_returned_and_rejected_require_reasons_and_notes(db_session):
    item,*_=_seed_qc1(db_session)
    with pytest.raises(WorkflowQCReviewError): _review(db_session,item,decision="returned",reason_codes=[],notes="Needs work.")
    with pytest.raises(WorkflowQCReviewError): _review(db_session,item,decision="rejected",reason_codes=["source_invalid"],notes="   ")

def test_exact_replay_is_idempotent(db_session):
    item,*_=_seed_qc1(db_session); first=_review(db_session,item); second=_review(db_session,item,expected_row_version=7)
    assert second["already_reviewed"] is True and second["review_id"]==first["review_id"] and second["row_version"]==7
    assert db_session.query(WorkflowReview).count()==1 and db_session.query(WorkflowEvent).filter(WorkflowEvent.event_type=="qc1_review_approved").count()==1

def test_nonexact_replay_conflicts(db_session):
    item,*_=_seed_qc1(db_session); _review(db_session,item)
    with pytest.raises(WorkflowQCReviewConflict): _review(db_session,item,expected_row_version=7,principal="principal:other-qc1")

def test_requires_exactly_one_completed_current_pair_comparison(db_session):
    item,dl1,dl2,c,_=_seed_qc1(db_session)
    db_session.add(WorkflowComparison(id=uuid4(),workflow_item_id=item.id,left_pass_id=dl1.id,right_pass_id=dl2.id,comparison_version=2,status="complete",strict_equality_passed=True,difference_count=0,difference_summary={"difference_count":0},checked_at=c.checked_at,checked_by_service_version="duplicate",reviewed_by_principal=None,reviewed_at=None,created_at=c.created_at)); db_session.flush()
    with pytest.raises(WorkflowQCReviewConflict): _review(db_session,item)

def test_mismatch_requires_complete_uniform_w9_resolution(db_session):
    item,_dl1,_dl2,_c,ds=_seed_qc1(db_session,strict_equal=False); ds[0].resolution_code="select_dl1"; db_session.flush()
    with pytest.raises(WorkflowQCReviewConflict): _review(db_session,item)
