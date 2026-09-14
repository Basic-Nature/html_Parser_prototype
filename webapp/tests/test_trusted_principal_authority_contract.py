from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

from webapp.parser.auth.trusted_identity_repository import (
    TrustedIdentityRepositoryError,
)
from webapp.parser.auth.trusted_principal_authority import (
    CANONICAL_PRINCIPAL_PREFIX,
    PROVIDER_ELECTIONPULSE_MTLS,
    TRUSTED_PRINCIPAL_AUTHORITY_CONTRACT,
    resolve_enrolled_mtls_principal,
)


FINGERPRINT = "ab" * 32
PRINCIPAL_ID = UUID("11111111-1111-4111-8111-111111111111")
CREDENTIAL_ID = UUID("22222222-2222-4222-8222-222222222222")
NOW = datetime(2026, 9, 14, 16, 0, tzinfo=timezone.utc)


class FakeRepository:
    def __init__(
        self,
        *,
        credential=None,
        principal=None,
        roles=frozenset(),
        capabilities=frozenset(),
        fail_at: str | None = None,
    ):
        self.credential = credential
        self.principal = principal
        self.roles = frozenset(roles)
        self.capabilities = frozenset(capabilities)
        self.fail_at = fail_at
        self.calls: list[str] = []

    def resolve_active_certificate(self, fingerprint_sha256: str):
        self.calls.append("resolve_active_certificate")
        if self.fail_at == "credential":
            raise TrustedIdentityRepositoryError("lookup failed")
        return self.credential

    def require_principal(self, principal_id: UUID):
        self.calls.append("require_principal")
        if self.fail_at == "principal":
            raise TrustedIdentityRepositoryError("lookup failed")
        return self.principal

    def current_role_names(self, principal_id: UUID, *, at=None):
        self.calls.append("current_role_names")
        if self.fail_at == "authorization":
            raise TrustedIdentityRepositoryError("lookup failed")
        return self.roles

    def capabilities_for_principal(self, principal_id: UUID):
        self.calls.append("capabilities_for_principal")
        if self.fail_at == "authorization":
            raise TrustedIdentityRepositoryError("lookup failed")
        return self.capabilities


def _credential(*, not_before=None, not_after=None):
    return SimpleNamespace(
        id=CREDENTIAL_ID,
        principal_id=PRINCIPAL_ID,
        state="active",
        not_before=not_before,
        not_after=not_after,
    )


def _principal(state="trusted"):
    return SimpleNamespace(
        id=PRINCIPAL_ID,
        principal_type="human",
        state=state,
    )


def test_invalid_fingerprint_fails_closed_without_repository_lookup():
    repo = FakeRepository()
    decision = resolve_enrolled_mtls_principal(repo, "not-a-fingerprint", at=NOW)

    assert decision.resolved is False
    assert decision.protected_operation_eligible is False
    assert decision.reason_code == "invalid_certificate_fingerprint"
    assert decision.canonical_principal is None
    assert repo.calls == []


def test_unknown_certificate_does_not_auto_enroll():
    repo = FakeRepository(credential=None)
    decision = resolve_enrolled_mtls_principal(repo, FINGERPRINT, at=NOW)

    assert decision.resolved is False
    assert decision.reason_code == "credential_not_enrolled"
    assert decision.compatibility_principal == f"cert:{FINGERPRINT}"
    assert repo.calls == ["resolve_active_certificate"]


def test_trusted_principal_resolves_to_opaque_canonical_identity():
    repo = FakeRepository(
        credential=_credential(
            not_before=NOW - timedelta(days=1),
            not_after=NOW + timedelta(days=1),
        ),
        principal=_principal("trusted"),
        roles={"workflow_auditor"},
        capabilities={"workflow.audit.read"},
    )

    decision = resolve_enrolled_mtls_principal(repo, FINGERPRINT, at=NOW)

    assert decision.contract_version == TRUSTED_PRINCIPAL_AUTHORITY_CONTRACT
    assert decision.provider == PROVIDER_ELECTIONPULSE_MTLS
    assert decision.resolved is True
    assert decision.protected_operation_eligible is True
    assert decision.reason_code == "trusted_principal_resolved"
    assert decision.canonical_principal == f"{CANONICAL_PRINCIPAL_PREFIX}{PRINCIPAL_ID}"
    assert decision.compatibility_principal == f"cert:{FINGERPRINT}"
    assert decision.role_names == ("workflow_auditor",)
    assert decision.capabilities == ("workflow.audit.read",)


def test_restricted_principal_is_resolved_but_not_protected_authority():
    repo = FakeRepository(
        credential=_credential(),
        principal=_principal("restricted"),
        roles={"workflow_auditor"},
        capabilities={"workflow.audit.read"},
    )

    decision = resolve_enrolled_mtls_principal(repo, FINGERPRINT, at=NOW)

    assert decision.resolved is True
    assert decision.protected_operation_eligible is False
    assert decision.reason_code == "principal_restricted"
    assert decision.canonical_principal == f"{CANONICAL_PRINCIPAL_PREFIX}{PRINCIPAL_ID}"
    assert decision.role_names == ()
    assert decision.capabilities == ()


def test_pending_suspended_and_revoked_principals_fail_closed():
    for state in ("pending", "suspended", "revoked"):
        repo = FakeRepository(
            credential=_credential(),
            principal=_principal(state),
        )
        decision = resolve_enrolled_mtls_principal(repo, FINGERPRINT, at=NOW)
        assert decision.resolved is False
        assert decision.protected_operation_eligible is False
        assert decision.reason_code == f"principal_state_{state}"


def test_certificate_time_bounds_fail_closed():
    not_yet = FakeRepository(
        credential=_credential(not_before=NOW + timedelta(seconds=1)),
    )
    expired = FakeRepository(
        credential=_credential(not_after=NOW),
    )

    assert (
        resolve_enrolled_mtls_principal(not_yet, FINGERPRINT, at=NOW).reason_code
        == "credential_not_yet_valid"
    )
    assert (
        resolve_enrolled_mtls_principal(expired, FINGERPRINT, at=NOW).reason_code
        == "credential_expired"
    )


def test_repository_failures_fail_closed():
    for fail_at, reason in (
        ("credential", "repository_lookup_failed"),
        ("principal", "principal_lookup_failed"),
        ("authorization", "authorization_lookup_failed"),
    ):
        repo = FakeRepository(
            credential=_credential(),
            principal=_principal("trusted"),
            fail_at=fail_at,
        )
        decision = resolve_enrolled_mtls_principal(repo, FINGERPRINT, at=NOW)
        assert decision.resolved is False
        assert decision.protected_operation_eligible is False
        assert decision.reason_code == reason


def test_decision_serialization_is_json_safe_and_contains_no_raw_certificate():
    repo = FakeRepository(
        credential=_credential(),
        principal=_principal("trusted"),
    )
    payload = resolve_enrolled_mtls_principal(repo, FINGERPRINT, at=NOW).to_dict()

    assert payload["canonical_principal"] == f"{CANONICAL_PRINCIPAL_PREFIX}{PRINCIPAL_ID}"
    assert payload["compatibility_principal"] == f"cert:{FINGERPRINT}"
    assert "certificate_bytes" not in payload
    assert "certificate_pem" not in payload
