from __future__ import annotations

from pathlib import Path
import pytest

from webapp.parser.utils.url_registry import (
    PublicRegistryResolutionError,
    is_parser_eligible_url,
    list_public_registry_sources,
    project_public_registry_sources,
    public_registry_source_id_for_entry,
    resolve_public_registry_source,
)


def _write_registry(path: Path) -> dict[str, str]:
    urls = {
        "curated": "https://example.gov/curated-results",
        "backlog": "https://example.gov/backlog-results",
        "quarantine": "https://example.gov/quarantine-results",
    }
    path.write_text(
        "\n".join([
            "# === Curated | 2024 General ===",
            "2024\tPresident\tExample\tstatewide\tHTML\tCertified\t" + urls["curated"],
            "# === Legacy / unsorted backlog ===",
            "2024\tGeneral Election\tExample\tstatewide\tHTML\tBacklog\t" + urls["backlog"],
            "# === Quarantine / third-party hosts ===",
            "2024\tPresident\tExample\tstatewide\tHTML\tReview\t" + urls["quarantine"],
            "legacy malformed line",
        ]) + "\n",
        encoding="utf-8",
    )
    return urls


def test_public_projection_contains_only_curated_parser_eligible_rows(tmp_path):
    registry = tmp_path / "urls.txt"
    urls = _write_registry(registry)
    assert is_parser_eligible_url(registry, urls["backlog"]) == (
        True,
        "approved_registry",
    )
    projected = project_public_registry_sources(registry)
    assert len(projected) == 1
    row = projected[0]
    assert row["registry_category"] == "curated"
    assert row["year"] == "2024"
    assert row["contest"] == "President"
    assert row["state"] == "Example"
    assert row["scope"] == "statewide"
    assert row["format"] == "HTML"
    assert row["registry_source_id"].startswith("blsrc_v1_")
    assert "url" not in row
    assert "notes" not in row
    assert urls["curated"] not in repr(row)
    assert urls["backlog"] not in repr(projected)


def test_public_registry_id_is_deterministic_and_exact_row_bound(tmp_path):
    registry = tmp_path / "urls.txt"
    _write_registry(registry)
    first = list_public_registry_sources(registry)
    second = list_public_registry_sources(registry)
    assert len(first) == 1
    assert first[0].registry_source_id == second[0].registry_source_id
    source_id = first[0].registry_source_id
    resolved = resolve_public_registry_source(registry, source_id)
    assert resolved is not None
    assert resolved.registry_source_id == source_id
    assert resolved.registry_category == "curated"
    assert resolved.url == "https://example.gov/curated-results"

    changed = {
        "year": "2024",
        "contest": "President",
        "state": "Example",
        "scope": "statewide",
        "format": "HTML",
        "notes": "Certified UPDATED",
        "url": "https://example.gov/curated-results",
        "section": "Curated | 2024 General",
        "review_status": "approved",
        "parser_eligible": True,
    }
    changed_id = public_registry_source_id_for_entry(changed)
    assert changed_id is not None
    assert changed_id != source_id


def test_unknown_or_non_id_inputs_do_not_resolve_publicly(tmp_path):
    registry = tmp_path / "urls.txt"
    _write_registry(registry)
    assert resolve_public_registry_source(
        registry, "blsrc_v1_" + ("0" * 64)
    ) is None
    assert resolve_public_registry_source(
        registry, "https://example.gov/curated-results"
    ) is None


def test_duplicate_exact_public_rows_fail_closed(tmp_path):
    registry = tmp_path / "urls.txt"
    row = (
        "2024\tPresident\tExample\tstatewide\tHTML\tCertified\t"
        "https://example.gov/curated-results"
    )
    registry.write_text(
        "# === Curated | 2024 General ===\n" + row + "\n" + row + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PublicRegistryResolutionError):
        list_public_registry_sources(registry)
