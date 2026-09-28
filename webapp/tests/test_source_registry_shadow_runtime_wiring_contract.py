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
    load_trusted_url_library_view,
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

    def resolve_public_execution_source(self, alias):
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
    assert load_trusted_url_library_view(REGISTRY) == (
        legacy_entries,
        legacy_diag,
    )

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
            "load_trusted_url_library_view",
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


def test_durable_public_resolver_never_reads_legacy_file(
    monkeypatch,
    tmp_path,
) -> None:
    source_id = "blsrc_test_public"
    source_url = "https://example.invalid/public-results"
    durable = {
        "registry_source_id": source_id,
        "year": "2024",
        "contest": "President",
        "state": "NY",
        "scope": "Rockland",
        "format": "Enhanced Voting",
        "registry_category": "curated",
        "url": source_url,
    }
    fake = FakeDb([durable])
    monkeypatch.setenv("SOURCE_REGISTRY_AUTHORITY_MODE", "durable_db")

    def legacy_read_forbidden(*_args, **_kwargs):
        raise AssertionError("durable public resolver touched legacy file")

    monkeypatch.setattr(
        runtime_module._legacy_registry,
        "resolve_public_registry_source",
        legacy_read_forbidden,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_source_registry_read_model",
        lambda path, *, mode=None, db_reader=None, mismatch_recorder=None: (
            build_source_registry_read_model(
                path,
                mode=mode,
                db_reader=fake,
                mismatch_recorder=mismatch_recorder,
            )
        ),
    )

    resolved = runtime_module.resolve_public_registry_source(
        tmp_path / "missing-legacy-registry.txt",
        source_id,
    )
    assert resolved is not None
    assert resolved.registry_source_id == source_id
    assert resolved.url == source_url


def test_durable_exact_lookup_never_reads_legacy_file(
    monkeypatch,
    tmp_path,
) -> None:
    source_url = "https://example.invalid/exact-results"
    durable = {
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
    }
    fake = FakeDb([], exact=[durable])
    monkeypatch.setenv("SOURCE_REGISTRY_AUTHORITY_MODE", "durable_db")

    def legacy_read_forbidden(*_args, **_kwargs):
        raise AssertionError("durable exact lookup touched legacy file")

    monkeypatch.setattr(
        runtime_module._legacy_registry,
        "lookup_exact_registry_entry",
        legacy_read_forbidden,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_source_registry_read_model",
        lambda path, *, mode=None, db_reader=None, mismatch_recorder=None: (
            build_source_registry_read_model(
                path,
                mode=mode,
                db_reader=fake,
                mismatch_recorder=mismatch_recorder,
            )
        ),
    )

    resolved = runtime_module.lookup_exact_registry_entry(
        source_url,
        path=tmp_path / "missing-legacy-registry.txt",
    )
    assert resolved is not None
    assert resolved.url == source_url
    assert resolved.registry_category == "curated"


def test_durable_resolver_legacy_calls_are_shadow_only() -> None:
    source = (
        ROOT / "webapp/parser/services/source_registry_runtime.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "resolve_public_registry_source",
            "lookup_exact_registry_entry",
        }
    }
    assert set(functions) == {
        "resolve_public_registry_source",
        "lookup_exact_registry_entry",
    }

    lines = source.splitlines()
    for function_name, legacy_symbol in (
        (
            "resolve_public_registry_source",
            "_legacy_registry.resolve_public_registry_source",
        ),
        (
            "lookup_exact_registry_entry",
            "_legacy_registry.lookup_exact_registry_entry",
        ),
    ):
        node = functions[function_name]
        segment = "\n".join(lines[node.lineno - 1:node.end_lineno])
        legacy_mode = segment.index('if mode == "legacy_file":')
        shadow_mode = segment.index('if mode == "shadow_db":')
        positions = []
        start = 0
        while True:
            index = segment.find(legacy_symbol, start)
            if index < 0:
                break
            positions.append(index)
            start = index + 1
        assert len(positions) == 2
        assert legacy_mode < positions[0] < shadow_mode < positions[1]


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


def test_trusted_url_library_view_durable_never_reads_legacy_file(
    monkeypatch,
    tmp_path,
) -> None:
    durable_entries = [
        {
            "year": "2024",
            "contest": "President",
            "state": "NY",
            "scope": "Rockland",
            "format": "Enhanced Voting",
            "notes": "approved",
            "url": "https://example.invalid/approved",
            "county": "Rockland",
            "registry_category": "curated",
            "review_status": "approved",
            "parser_eligible": True,
            "normalized_url": "https://example.invalid/approved",
        },
        {
            "year": "2024",
            "contest": "President",
            "state": "NY",
            "scope": "Rockland",
            "format": "Enhanced Voting",
            "notes": "quarantined",
            "url": "https://example.invalid/quarantined",
            "county": "Rockland",
            "registry_category": "quarantine",
            "review_status": "quarantined",
            "parser_eligible": False,
            "normalized_url": "https://example.invalid/quarantined",
        },
    ]
    fake = FakeDb([])
    monkeypatch.setattr(
        fake,
        "list_trusted_registry_entries",
        lambda: [dict(item) for item in durable_entries],
    )
    monkeypatch.setenv("SOURCE_REGISTRY_AUTHORITY_MODE", "durable_db")

    def legacy_read_forbidden(*_args, **_kwargs):
        raise AssertionError("durable trusted URL library touched legacy file")

    monkeypatch.setattr(
        runtime_module._legacy_registry,
        "load_url_registry",
        legacy_read_forbidden,
    )

    entries, diagnostics = load_trusted_url_library_view(
        tmp_path / "missing-legacy-registry.txt",
        db_reader=fake,
    )
    assert entries == durable_entries
    assert diagnostics == {
        "contract": "trusted_url_library_view_v1",
        "diagnostics_source": "durable_structured_registry",
        "file_diagnostics_available": False,
        "row_count": 2,
        "malformed_row_count": 0,
        "quarantine_row_count": 1,
        "parser_eligible_count": 1,
    }


def test_raw_registry_loader_remains_fail_closed_for_durable_diagnostics() -> None:
    source = (
        ROOT / "webapp/parser/services/source_registry_runtime.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "load_url_registry",
            "load_trusted_url_library_view",
        }
    }
    assert set(functions) == {
        "load_url_registry",
        "load_trusted_url_library_view",
    }
    lines = source.splitlines()

    raw = functions["load_url_registry"]
    raw_segment = "\n".join(lines[raw.lineno - 1:raw.end_lineno])
    assert "_legacy_registry.load_url_registry(path)" in raw_segment
    assert "durable_db raw registry diagnostics are not yet an accepted" in raw_segment

    view = functions["load_trusted_url_library_view"]
    view_segment = "\n".join(lines[view.lineno - 1:view.end_lineno])
    assert 'if mode != "durable_db":' in view_segment
    assert "return load_url_registry(path)" in view_segment
    assert "_legacy_registry.load_url_registry" not in view_segment
    assert '"file_diagnostics_available": False' in view_segment
