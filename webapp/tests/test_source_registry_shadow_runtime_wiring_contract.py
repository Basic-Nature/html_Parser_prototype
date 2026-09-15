from __future__ import annotations

import ast
from pathlib import Path

import pytest

from webapp.parser.services.source_registry_read_model import (
    SourceRegistryReadModel,
    SourceRegistryShadowMismatch,
)
from webapp.parser.services.source_registry_runtime import (
    build_source_registry_read_model,
    load_url_registry,
    project_public_registry_sources,
)
from webapp.parser.utils.url_registry import (
    load_url_registry as legacy_load_url_registry,
    project_public_registry_sources as legacy_project_public_registry_sources,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "webapp/parser/urls.txt"


class FakeDb:
    def __init__(self, public):
        self.public = list(public)

    def list_public_sources(self):
        return list(self.public)

    def resolve_public_source_alias(self, alias):
        return next(
            (item for item in self.public if item["registry_source_id"] == alias),
            None,
        )

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
            "load_url_registry",
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
