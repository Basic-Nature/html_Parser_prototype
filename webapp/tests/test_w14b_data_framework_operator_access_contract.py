from __future__ import annotations

from webapp.parser.services.data_framework_operator_access import (
    DATA_FRAMEWORK_OPERATOR_ACCESS_CONTRACT,
    project_data_framework_operator_access,
)


def _authority(state: str, *, authenticated: bool) -> dict[str, object]:
    return {
        "state": state,
        "authenticated": authenticated,
        "fresh_proof": state == "fresh_certificate",
    }


def test_anonymous_projection_is_identity_free_and_non_mutating():
    result = project_data_framework_operator_access(
        _authority("anonymous", authenticated=False), 0
    )
    assert result == {
        "contract": DATA_FRAMEWORK_OPERATOR_ACCESS_CONTRACT,
        "authenticated": False,
        "can_upload_input": False,
        "mutation_controls_disclosed": False,
        "principal_disclosed": False,
    }


def test_fresh_certificate_can_project_trusted_upload_control():
    result = project_data_framework_operator_access(
        _authority("fresh_certificate", authenticated=True), 0
    )
    assert result["can_upload_input"] is True
    assert result["mutation_controls_disclosed"] is True
    assert result["principal_disclosed"] is False


def test_certificate_session_can_project_trusted_upload_control():
    result = project_data_framework_operator_access(
        _authority("certificate_session", authenticated=True), 0
    )
    assert result["can_upload_input"] is True
    assert result["principal_disclosed"] is False


def test_other_authenticated_authority_does_not_gain_upload_by_authentication_alone():
    result = project_data_framework_operator_access(
        _authority("authenticated_other", authenticated=True), 3
    )
    assert result["authenticated"] is True
    assert result["can_upload_input"] is False
    assert result["mutation_controls_disclosed"] is False


def test_missing_tier_fails_closed():
    result = project_data_framework_operator_access(
        _authority("fresh_certificate", authenticated=True), None
    )
    assert result["can_upload_input"] is False


def test_projection_never_contains_identity_fields():
    result = project_data_framework_operator_access(
        _authority("fresh_certificate", authenticated=True), 0
    )
    forbidden = {"principal", "principal_source", "roles", "fingerprint", "serial_number"}
    assert not (forbidden & set(result))
