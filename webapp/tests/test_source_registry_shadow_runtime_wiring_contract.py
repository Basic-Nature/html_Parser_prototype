from __future__ import annotations

import ast
from pathlib import Path

import pytest

from webapp.parser.services.source_registry_read_model import (
    SourceRegistryReadModel,
    SourceRegistryShadowMismatch,
)
from webapp.parser.services import source_registry_runtime as runtime_module
from webapp.parser.services.source_registry_runtime import (
    build_source_registry_read_model,
    list_exact_registry_entries,
    list_public_registry_identity_sources,
    load_url_registry,
    project_public_registry_sources,
)
from webapp.parser.utils.url_registry import (
    list_public_registry_sources as legacy_list_public_registry_sources,
    load_url_registry as legacy_load_url_registry,
    project_public_registry_sources as legacy_project_public_registry_sources,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "webapp/parser/urls.txt"


class FakeDb:
    def __init__(self, public, identity=None, exact=None):
        self.public = list(public)
        self.identity = list(identity if identity is not None else public)
        self.exact = list(exact or [])

    def list_public_sources(self):
        return list(self.public)

    def list_public_identity_sources(self):
        return list(self.identity)

    def resolve_public_source_alias(self, alias):
        return next(
            (item for item in self.public if item["registry_source_id"] == alias),
            None,
        )

    def list_trusted_exact_sources(self, source_url):
        return [
            dict(item)
            for item in self.exact
            if str(item.get("url") or "") == str(source_url or "")
        ]

    def resolve_trusted_exact_source(self, source_url):
        return None

    def resolve_workflow_binding(self, source_url):
        return None

    def list_trusted_registry_entries(self):
        return []

    def resolve_legacy_url_compatibility(self, source_url):
        return {
            "matched": False,
            "parser_eligible": False,
            "review_statuses": [],
            "normalized_urls": [],
        }


def test_runtime_legacy_mode_is_semantically_identical(monkeypatch) -> None:
    monkeypatch.setenv("SOURCE_REGISTRY_AUTHORITY_MODE", "legacy_file")
    assert project_public_registry_sources(REGISTRY) == (
        legacy_project_public_registry_sources(REGISTRY)
    )
    runtime_entries, runtime_diag = load_url_registry(REGISTRY)
    legacy_entries, legacy_diag = legacy_load_url_registry(REGISTRY)
    assert runtime_entries == legacy_entries
    assert runtime_diag == legacy_diag

    assert list_public_registry_identity_sources(REGISTRY) == (
        legacy_list_public_registry_sources(REGISTRY)
    )


def test_identity_projection_shadow_returns_legacy_and_fails_on_drift() -> None:
    legacy_model = SourceRegistryReadModel(
        REGISTRY,
        mode="legacy_file",
    )
    legacy = legacy_model.list_public_identity_sources()

    model = build_source_registry_read_model(
        REGISTRY,
        mode="shadow_db",
        db_reader=FakeDb([], identity=legacy),
    )
    assert model.list_public_identity_sources() == legacy

    drift = list(legacy)
    drift[0] = {**drift[0], "url": "https://example.invalid/drift"}
    with pytest.raises(SourceRegistryShadowMismatch):
        build_source_registry_read_model(
            REGISTRY,
            mode="shadow_db",
            db_reader=FakeDb([], identity=drift),
        ).list_public_identity_sources()


def test_central_builder_shadow_returns_legacy_and_fails_on_drift() -> None:
    legacy = SourceRegistryReadModel(
        REGISTRY,
        mode="legacy_file",
    ).list_public_sources()
    model = build_source_registry_read_model(
        REGISTRY,
        mode="shadow_db",
        db_reader=FakeDb(legacy),
    )
    assert model.list_public_sources() == legacy

    drift = list(legacy)
    drift[0] = {**drift[0], "contest": "DRIFT"}
    with pytest.raises(SourceRegistryShadowMismatch):
        build_source_registry_read_model(
            REGISTRY,
            mode="shadow_db",
            db_reader=FakeDb(drift),
        ).list_public_sources()


def test_projected_callsites_use_runtime_seam() -> None:
    expected = {
        "webapp/Smart_Elections_Parser_Webapp.py": {
            "list_public_registry_identity_sources",
            "load_url_registry",
            "project_public_registry_sources",
        },
        "webapp/parser/socket_ballot_lens_orchestration.py": {
            "is_parser_eligible_url",
            "resolve_public_registry_source",
        },
        "webapp/parser/services/workflow_actions.py": {
            "lookup_exact_registry_entry",
        },
        "webapp/parser/services/workflow_staging_binding.py": {
            "lookup_exact_registry_entry",
        },
        "webapp/parser/services/workflow_ballot_lens_runtime_context.py": {
            "list_exact_registry_entries",
        },
    }
    for rel, names in expected.items():
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        runtime = set()
        legacy = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            imported = {alias.name for alias in node.names}
            if node.module == "webapp.parser.services.source_registry_runtime":
                runtime |= imported
            if node.module == "webapp.parser.utils.url_registry":
                legacy |= imported
        assert names <= runtime
        assert not (names & legacy)


def test_durable_exact_entry_list_never_reads_legacy_file(
    monkeypatch,
    tmp_path,
) -> None:
    source_url = "https://example.invalid/results"
    durable = [{
        "year": "2024",
        "contest": "President",
        "state": "NY",
        "scope": "Rockland",
        "format": "Enhanced Voting",
        "notes": "durable",
        "url": source_url,
        "county": "Rockland",
        "registry_category": "curated",
        "review_status": "approved",
        "parser_eligible": True,
        "normalized_url": source_url,
    }]
    monkeypatch.setenv("SOURCE_REGISTRY_AUTHORITY_MODE", "durable_db")

    def legacy_read_forbidden(*_args, **_kwargs):
        raise AssertionError("durable exact-entry lookup touched legacy file")

    monkeypatch.setattr(
        runtime_module._legacy_registry,
        "load_url_registry",
        legacy_read_forbidden,
    )

    assert list_exact_registry_entries(
        tmp_path / "missing-legacy-registry.txt",
        source_url,
        db_reader=FakeDb([], exact=durable),
    ) == durable


def test_shadow_exact_entry_list_compares_semantic_projection(
    monkeypatch,
    tmp_path,
) -> None:
    source_url = "https://example.invalid/results"
    registry = tmp_path / "urls.txt"
    registry.write_text(
        "# === Curated | test ===\n"
        "2024\tPresident\tNY\tRockland\tEnhanced Voting\tdurable\t"
        + source_url
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOURCE_REGISTRY_AUTHORITY_MODE", "shadow_db")
    legacy, _ = runtime_module._legacy_registry.load_url_registry(registry)
    exact = [
        runtime_module._semantic_legacy_entry(dict(item))
        for item in legacy
        if str(item.get("url") or "").strip() == source_url
    ]

    assert list_exact_registry_entries(
        registry,
        source_url,
        db_reader=FakeDb([], exact=exact),
    ) == [
        dict(item)
        for item in legacy
        if str(item.get("url") or "").strip() == source_url
    ]


def test_concrete_reader_is_read_only_by_contract() -> None:
    source = (
        ROOT / "webapp/parser/services/source_registry_db_reader.py"
    ).read_text(encoding="utf-8")
    assert "SET TRANSACTION READ ONLY" in source
    assert "session.rollback()" in source
    assert "session.close()" in source
    forbidden = (
        "session.commit(",
        "session.add(",
        "session.delete(",
        "session.flush(",
    )
    assert not any(token in source for token in forbidden)
