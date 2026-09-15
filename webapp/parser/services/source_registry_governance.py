"""Fail-closed Source Registry governance foundation.

W20D deploys this module inert. It contains validation and authority-boundary
helpers only; no persistence mutation entrypoint is activated while
SOURCE_REGISTRY_MUTATIONS_ENABLED is false.
"""
from __future__ import annotations

from collections.abc import Mapping

from webapp.parser.contracts.source_registry_authorization import (
    SourceRegistryAuthorizationError,
    assert_normal_publish_separation,
)
from webapp.parser.utils.url_registry import source_registry_mutations_enabled

SOURCE_REGISTRY_GOVERNANCE_CONTRACT = "source_registry_governance_v1"

PROPOSAL_OPERATIONS = frozenset({
    "create",
    "revise_url",
    "revise_metadata",
    "change_eligibility",
    "quarantine",
    "deprecate",
    "restore",
})


class SourceRegistryGovernanceError(RuntimeError):
    pass


class SourceRegistryMutationDisabled(SourceRegistryGovernanceError):
    pass


def assert_mutation_feature_enabled() -> None:
    if not source_registry_mutations_enabled():
        raise SourceRegistryMutationDisabled(
            "Source Registry mutations are disabled."
        )


def assert_expected_row_version(
    *,
    expected: int | None,
    actual: int | None,
    allow_create: bool = False,
) -> None:
    if allow_create and expected is None and actual is None:
        return
    if not isinstance(expected, int) or expected < 1:
        raise SourceRegistryGovernanceError(
            "expected_target_row_version must be a positive integer"
        )
    if expected != actual:
        raise SourceRegistryGovernanceError(
            "Source Registry optimistic-concurrency version mismatch"
        )


def assert_publish_separation(
    *,
    proposer_principal: str,
    reviewer_principal: str,
    publisher_principal: str,
) -> None:
    try:
        assert_normal_publish_separation(
            proposer_principal=proposer_principal,
            reviewer_principal=reviewer_principal,
            publisher_principal=publisher_principal,
        )
    except SourceRegistryAuthorizationError as exc:
        raise SourceRegistryGovernanceError(str(exc)) from exc


def assert_reduce_only_quarantine(
    *,
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> None:
    for field in ("parser_eligible", "public_eligible", "workflow_eligible"):
        if after.get(field) is True:
            raise SourceRegistryGovernanceError(
                "Emergency quarantine may not enable eligibility."
            )
    if str(after.get("review_state") or "").strip().lower() != "quarantined":
        raise SourceRegistryGovernanceError(
            "Emergency quarantine must set review_state=quarantined."
        )
