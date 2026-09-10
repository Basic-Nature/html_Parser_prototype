from __future__ import annotations

import pytest

from webapp.parser.contracts.workflow_authorization import (
    CAP_BALLOT_LENS_EXECUTE,
    ROLE_AUDITOR,
    ROLE_CONTRIBUTOR,
    ROLE_PUBLICATION_OPERATOR,
    ROLE_REVIEWER,
)
from webapp.parser.services.workflow_operator_access import (
    WorkflowOperatorAccessError,
    project_workflow_operator_access,
)


def test_execute_capability_is_contributor_only():
    contributor = project_workflow_operator_access(
        "cert:contributor", [ROLE_CONTRIBUTOR]
    )
    assert contributor["can_execute_ballot_lens"] is True
    assert CAP_BALLOT_LENS_EXECUTE in contributor["capabilities"]
    assert contributor["principal_disclosed"] is False
    assert "cert:contributor" not in repr(contributor)

    for role in (
        ROLE_REVIEWER,
        ROLE_PUBLICATION_OPERATOR,
        ROLE_AUDITOR,
    ):
        projected = project_workflow_operator_access(
            "cert:other", [role]
        )
        assert projected["can_execute_ballot_lens"] is False


def test_operator_projection_requires_principal():
    with pytest.raises(WorkflowOperatorAccessError):
        project_workflow_operator_access("", [ROLE_CONTRIBUTOR])
