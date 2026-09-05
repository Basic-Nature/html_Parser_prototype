from __future__ import annotations

import asyncio

import pytest

from tools.production_acceptance_observer import (
    RENDERED_TEXT_AUTHORITY,
    SEMANTIC_TEXT_AUTHORITY,
    api_response_headers,
    assert_semantic_tokens,
    missing_semantic_tokens,
    read_text_authority,
)


class FakeAPIResponse:
    def __init__(self) -> None:
        self.headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-Observer": "safe",
        }

    async def all_headers(self):  # pragma: no cover - must never be called
        raise AssertionError("observer must not call APIResponse.all_headers()")


class BadHeadersResponse:
    headers = [("content-type", "application/json")]


class FakeLocator:
    async def text_content(self) -> str:
        return (
            "Canonical comparison "
            "READING canonical_read_in_progress "
            "canonical_production GET only"
        )

    async def inner_text(self) -> str:
        # Matches the R4 presentation mismatch caused by CSS text-transform.
        return (
            "CANONICAL COMPARISON\n"
            "READING canonical_read_in_progress canonical_production GET only"
        )


class EmptySemanticLocator:
    async def text_content(self):
        return None

    async def inner_text(self) -> str:
        return "PRESENTATION ONLY"


def test_api_response_headers_uses_property_not_all_headers_method() -> None:
    headers = api_response_headers(FakeAPIResponse())

    assert headers == {
        "content-type": "application/json; charset=utf-8",
        "x-observer": "safe",
    }


def test_api_response_headers_rejects_non_mapping_contract() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        api_response_headers(BadHeadersResponse())


def test_semantic_tokens_ignore_css_text_transform_presentation() -> None:
    authority = asyncio.run(read_text_authority(FakeLocator()))

    assert "Canonical comparison" in authority.semantic_text
    assert "CANONICAL COMPARISON" in authority.rendered_text
    assert missing_semantic_tokens(
        authority,
        (
            "Canonical comparison",
            "READING",
            "canonical_read_in_progress",
            "canonical_production",
            "GET only",
        ),
    ) == ()


def test_assert_semantic_tokens_fails_on_dom_semantic_absence() -> None:
    authority = asyncio.run(read_text_authority(FakeLocator()))

    with pytest.raises(
        AssertionError,
        match="DOM semantic text missing tokens",
    ):
        assert_semantic_tokens(
            authority,
            ("Canonical comparison", "MISSING_TOKEN"),
            context="Validation",
        )


def test_text_authority_evidence_separates_semantic_and_rendered_sources() -> None:
    authority = asyncio.run(read_text_authority(FakeLocator()))
    evidence = authority.evidence()

    assert evidence["semantic_source"] == SEMANTIC_TEXT_AUTHORITY
    assert evidence["rendered_source"] == RENDERED_TEXT_AUTHORITY
    assert evidence["semantic_sha256"] != evidence["rendered_sha256"]
    assert evidence["css_text_transform_is_not_semantic_authority"] is True


def test_none_text_content_normalizes_to_empty_semantic_text() -> None:
    authority = asyncio.run(read_text_authority(EmptySemanticLocator()))

    assert authority.semantic_text == ""
    assert authority.rendered_text == "PRESENTATION ONLY"
