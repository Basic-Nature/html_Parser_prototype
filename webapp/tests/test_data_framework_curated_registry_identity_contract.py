from __future__ import annotations

from webapp.parser.services.data_framework_source_identity import (
    resolve_curated_registry_source_id,
)
from webapp.parser.utils.url_registry import PublicRegistrySource


def _source(
    source_id: str,
    *,
    url: str = "https://example.test/results",
    year: str = "2024",
    contest: str = "President",
    state: str = "Arizona",
    scope: str = "statewide",
) -> PublicRegistrySource:
    return PublicRegistrySource(
        registry_source_id=source_id,
        year=year,
        contest=contest,
        state=state,
        registry_scope=scope,
        registry_format="HTML",
        registry_category="curated",
        url=url,
    )


def test_unique_exact_public_url_projects_existing_registry_id():
    source = _source("blsrc_v1_" + "a" * 64)
    assert resolve_curated_registry_source_id(
        {"source_url": source.url, "state": "different"},
        [source],
        year="1999",
    ) == source.registry_source_id


def test_multiple_exact_url_rows_use_metadata_only_to_disambiguate():
    first = _source(
        "blsrc_v1_" + "a" * 64,
        contest="President",
        state="Arizona",
        scope="statewide",
    )
    second = _source(
        "blsrc_v1_" + "b" * 64,
        contest="Senate",
        state="Arizona",
        scope="Maricopa",
    )
    assert resolve_curated_registry_source_id(
        {
            "source_url": first.url,
            "state": "Arizona",
            "contest": "Senate",
            "county": "Maricopa",
        },
        [first, second],
        year="2024",
    ) == second.registry_source_id


def test_ambiguous_exact_url_fails_closed_without_synthetic_identity():
    first = _source("blsrc_v1_" + "a" * 64)
    second = _source("blsrc_v1_" + "b" * 64)
    assert resolve_curated_registry_source_id(
        {"source_url": first.url},
        [first, second],
    ) is None


def test_normalized_but_not_exact_url_does_not_inherit_registry_authority():
    source = _source(
        "blsrc_v1_" + "a" * 64,
        url="https://example.test/results",
    )
    assert resolve_curated_registry_source_id(
        {"source_url": "HTTPS://EXAMPLE.TEST/results#fragment"},
        [source],
        year="2024",
    ) is None


def test_missing_or_unknown_source_url_projects_null_identity():
    source = _source("blsrc_v1_" + "a" * 64)
    assert resolve_curated_registry_source_id({}, [source]) is None
    assert resolve_curated_registry_source_id(
        {"source_url": "https://unknown.test/results"},
        [source],
    ) is None
