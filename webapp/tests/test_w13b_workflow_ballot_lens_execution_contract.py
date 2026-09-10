from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from webapp.parser.contracts.workflow_authorization import (
    ROLE_CONTRIBUTOR,
    ROLE_REVIEWER,
)
from webapp.parser.services.trusted_parser_source_policy import (
    PROVENANCE_WORKFLOW_QC,
)
from webapp.parser.services.workflow_ballot_lens_execution import (
    WORKFLOW_EXECUTION_REQUEST_KEYS,
    WorkflowBallotLensExecutionDenied,
    WorkflowBallotLensServerContext,
    authorize_workflow_ballot_lens_execution,
    validate_workflow_execution_request,
)

URL = "https://sos.example.gov/results.pdf"


def build():
    item_id = str(uuid4())
    pass_id = str(uuid4())
    payload = {
        "workflow_item_id": item_id,
        "workflow_pass_id": pass_id,
        "expected_row_version": 7,
    }
    context = WorkflowBallotLensServerContext(
        workflow_item_id=item_id,
        workflow_pass_id=pass_id,
        row_version=7,
        assigned_principal="cert:operator",
        pass_is_current=True,
        pass_status="in_progress",
        source_url=URL,
        qc_evidence={
            "source_url": URL,
            "qc_backed": True,
            "provenance_class": PROVENANCE_WORKFLOW_QC,
        },
        registry_state={
            "source_url": URL,
            "exact_registry_identity": True,
            "registry_category": "curated",
            "review_status": "approved",
            "parser_eligible": True,
            "quarantined": False,
            "deprecated": False,
        },
    )
    return payload, context


def test_browser_contract_is_ids_and_version_only():
    assert WORKFLOW_EXECUTION_REQUEST_KEYS == {
        "workflow_item_id",
        "workflow_pass_id",
        "expected_row_version",
    }
    payload, _ = build()
    validate_workflow_execution_request(payload)
    for forbidden in (
        "source_url",
        "direct_urls",
        "registry_source_id",
        "execution_source_id",
        "role",
        "capability",
    ):
        invalid = dict(payload)
        invalid[forbidden] = "attacker-controlled"
        with pytest.raises(WorkflowBallotLensExecutionDenied):
            validate_workflow_execution_request(invalid)


def test_authorized_execution_keeps_url_internal():
    payload, context = build()
    authority = authorize_workflow_ballot_lens_execution(
        payload,
        principal="cert:operator",
        internal_roles=[ROLE_CONTRIBUTOR],
        server_context=context,
    )
    assert authority.resolved_source_url == URL
    safe = authority.safe_projection()
    assert safe["execution_mode"] == "workflow"
    assert safe["source_url_disclosed"] is False
    assert URL not in repr(safe)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_principal",
        "wrong_role",
        "stale_version",
        "wrong_item",
        "wrong_pass",
        "not_current",
        "not_in_progress",
        "untrusted_registry",
        "untrusted_qc",
    ],
)
def test_execution_revalidates_server_authority(mutation):
    payload, context = build()
    principal = "cert:operator"
    roles = [ROLE_CONTRIBUTOR]

    if mutation == "wrong_principal":
        principal = "cert:other"
    elif mutation == "wrong_role":
        roles = [ROLE_REVIEWER]
    elif mutation == "stale_version":
        payload = dict(payload)
        payload["expected_row_version"] = 6
    elif mutation == "wrong_item":
        payload = dict(payload)
        payload["workflow_item_id"] = str(uuid4())
    elif mutation == "wrong_pass":
        payload = dict(payload)
        payload["workflow_pass_id"] = str(uuid4())
    elif mutation == "not_current":
        context = replace(context, pass_is_current=False)
    elif mutation == "not_in_progress":
        context = replace(context, pass_status="submitted")
    elif mutation == "untrusted_registry":
        reg = dict(context.registry_state)
        reg["registry_category"] = "backlog"
        context = replace(context, registry_state=reg)
    elif mutation == "untrusted_qc":
        qc = dict(context.qc_evidence)
        qc["qc_backed"] = False
        context = replace(context, qc_evidence=qc)

    with pytest.raises(WorkflowBallotLensExecutionDenied):
        authorize_workflow_ballot_lens_execution(
            payload,
            principal=principal,
            internal_roles=roles,
            server_context=context,
        )
