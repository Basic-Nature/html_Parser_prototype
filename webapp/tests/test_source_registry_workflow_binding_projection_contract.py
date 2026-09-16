from __future__ import annotations

from types import SimpleNamespace

from webapp.parser.services import source_registry_db_reader as module


def test_workflow_projection_matches_legacy_contributor_contract() -> None:
    binding = SimpleNamespace(
        year="2024",
        contest="President",
        state="NY",
        scope="Rockland",
        format="Enhanced Voting",
        notes="example",
        review_state="approved",
        parser_eligible=True,
    )
    revision = SimpleNamespace(
        exact_url="https://example.invalid/results",
        normalized_url="https://example.invalid/results",
    )

    expected = {
        "year": "2024",
        "contest": "President",
        "state": "NY",
        "scope": "Rockland",
        "format": "Enhanced Voting",
        "notes": "example",
        "url": "https://example.invalid/results",
        "registry_category": "curated",
    }
    assert module._workflow_projection(binding, revision) == expected

    trusted = module._trusted_projection(binding, revision)
    assert trusted == {
        **expected,
        "county": "Rockland",
        "review_status": "approved",
        "parser_eligible": True,
        "normalized_url": "https://example.invalid/results",
    }


def test_workflow_resolver_returns_workflow_projection_not_rich_trusted_projection() -> None:
    import inspect

    source = inspect.getsource(
        module.SqlAlchemySourceRegistryDbReader.resolve_workflow_binding
    )
    assert "return _workflow_projection(*rows[0])" in source
    assert "return _trusted_projection(*rows[0])" not in source
