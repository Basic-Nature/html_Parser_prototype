from __future__ import annotations

from pathlib import Path
import pytest

from webapp.parser.services.source_registry_read_model import (
    SourceRegistryReadModel,
    SourceRegistryReadModelError,
    SourceRegistryShadowMismatch,
)

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "webapp/parser/urls.txt"


class FakeDb:
    def __init__(self, public):
        self.public = public

    def list_public_sources(self):
        return self.public

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


def test_legacy_mode_preserves_64_public_sources() -> None:
    model = SourceRegistryReadModel(REGISTRY, mode="legacy_file")
    sources = model.list_public_sources()
    assert len(sources) == 64
    assert all(item["registry_category"] == "curated" for item in sources)
    assert all("url" not in item for item in sources)


def test_shadow_mode_returns_legacy_authority_and_requires_exact_parity() -> None:
    legacy = SourceRegistryReadModel(REGISTRY, mode="legacy_file").list_public_sources()
    shadow = SourceRegistryReadModel(
        REGISTRY,
        mode="shadow_db",
        db_reader=FakeDb(legacy),
    )
    assert shadow.list_public_sources() == legacy

    mismatched = list(legacy)
    mismatched[0] = {**mismatched[0], "contest": "DRIFT"}
    with pytest.raises(SourceRegistryShadowMismatch):
        SourceRegistryReadModel(
            REGISTRY,
            mode="shadow_db",
            db_reader=FakeDb(mismatched),
        ).list_public_sources()


def test_durable_mode_never_silently_falls_back_to_file() -> None:
    model = SourceRegistryReadModel(REGISTRY, mode="durable_db")
    with pytest.raises(SourceRegistryReadModelError):
        model.list_public_sources()
