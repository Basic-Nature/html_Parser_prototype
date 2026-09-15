from __future__ import annotations

import uuid
import pytest

from webapp.parser.contracts.source_registry_authorization import (
    CAP_SOURCE_REGISTRY_PROPOSE,
    CAP_SOURCE_REGISTRY_PUBLISH,
    SourceRegistryAuthorizationError,
    assert_capability,
    assert_normal_publish_separation,
)
from webapp.parser.services.source_registry_governance import (
    SourceRegistryMutationDisabled,
    assert_mutation_feature_enabled,
    assert_reduce_only_quarantine,
)
from webapp.parser.utils.models import Base, SourceRegistryBase
from webapp.parser.utils.url_registry import (
    source_registry_authority_mode,
    source_registry_mutations_enabled,
    stable_public_registry_v2_alias,
)

EXPECTED_TABLES = {
    "source_registry_sources",
    "source_registry_revisions",
    "source_registry_bindings",
    "source_registry_aliases",
    "source_registry_proposals",
    "source_registry_reviews",
    "source_registry_events",
}


def test_source_registry_metadata_is_isolated_from_application_base() -> None:
    assert not (EXPECTED_TABLES & set(Base.metadata.tables))
    assert EXPECTED_TABLES == set(SourceRegistryBase.metadata.tables)


def test_default_authority_and_mutation_boundary_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("SOURCE_REGISTRY_AUTHORITY_MODE", raising=False)
    monkeypatch.delenv("SOURCE_REGISTRY_MUTATIONS_ENABLED", raising=False)
    assert source_registry_authority_mode() == "legacy_file"
    assert source_registry_mutations_enabled() is False
    with pytest.raises(SourceRegistryMutationDisabled):
        assert_mutation_feature_enabled()


def test_role_capabilities_and_three_principal_publish_separation() -> None:
    assert_capability(["workflow_contributor"], CAP_SOURCE_REGISTRY_PROPOSE)
    assert_capability(["workflow_publication_operator"], CAP_SOURCE_REGISTRY_PUBLISH)
    assert_normal_publish_separation(
        proposer_principal="principal:a",
        reviewer_principal="principal:b",
        publisher_principal="principal:c",
    )
    with pytest.raises(SourceRegistryAuthorizationError):
        assert_normal_publish_separation(
            proposer_principal="principal:a",
            reviewer_principal="principal:b",
            publisher_principal="principal:a",
        )


def test_emergency_quarantine_can_only_reduce_authority() -> None:
    before = {
        "review_state": "approved",
        "parser_eligible": True,
        "public_eligible": True,
        "workflow_eligible": True,
    }
    after = {
        "review_state": "quarantined",
        "parser_eligible": False,
        "public_eligible": False,
        "workflow_eligible": False,
    }
    assert_reduce_only_quarantine(before=before, after=after)
    with pytest.raises(Exception):
        assert_reduce_only_quarantine(
            before=before,
            after={**after, "public_eligible": True},
        )


def test_stable_v2_alias_is_binding_identity_based() -> None:
    binding = str(uuid.uuid4())
    first = stable_public_registry_v2_alias(binding)
    second = stable_public_registry_v2_alias(binding)
    assert first == second
    assert first.startswith("blsrc_v2_")
    assert len(first) == len("blsrc_v2_") + 64
def test_source_registry_isolated_metadata_resolves_and_migration_retains_external_trust_fks() -> None:
    from pathlib import Path

    sorted_tables = SourceRegistryBase.metadata.sorted_tables
    assert {table.name for table in sorted_tables} == EXPECTED_TABLES
    assert not (EXPECTED_TABLES & set(Base.metadata.tables))

    target_names = {
        fk.target_fullname
        for table in sorted_tables
        for fk in table.foreign_keys
    }
    assert not any(name.startswith("trusted_principals.") for name in target_names)
    assert not any(name.startswith("trusted_credentials.") for name in target_names)

    root = Path(__file__).resolve().parents[2]
    migration_path = (
        root
        / "alembic"
        / "versions"
        / "d2e33e73c6d2_source_registry_control_plane_foundation.py"
    )
    migration_text = migration_path.read_text(encoding="utf-8")
    assert migration_text.count('"trusted_principals.id"') == 6
    assert migration_text.count('"trusted_credentials.id"') == 1
