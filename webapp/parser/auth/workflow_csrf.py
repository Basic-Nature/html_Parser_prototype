from __future__ import annotations

import hmac
import secrets

from flask import session


WORKFLOW_CSRF_CONTRACT = "workflow_mutation_csrf_v1"
_WORKFLOW_CSRF_SESSION_KEY = "_workflow_csrf_token_v1"


class WorkflowCsrfError(PermissionError):
    status_code = 403
    code = "workflow_csrf_invalid"


def issue_workflow_csrf_token() -> str:
    current = str(session.get(_WORKFLOW_CSRF_SESSION_KEY) or "").strip()
    if current:
        return current
    token = secrets.token_urlsafe(32)
    session[_WORKFLOW_CSRF_SESSION_KEY] = token
    session.modified = True
    return token


def assert_workflow_csrf_token(value: object) -> None:
    expected = str(session.get(_WORKFLOW_CSRF_SESSION_KEY) or "").strip()
    token = str(value or "").strip()
    if not expected or not token:
        raise WorkflowCsrfError("Workflow mutation requires session-bound CSRF token.")
    if not hmac.compare_digest(expected, token):
        raise WorkflowCsrfError("Workflow CSRF validation failed.")
