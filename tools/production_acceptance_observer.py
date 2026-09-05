"""Reusable observer primitives for ElectionPulse production acceptance.

This module intentionally has no Playwright import-time dependency. It works
against the small response/locator protocols required by acceptance observers.

Authority rules:
- Playwright APIRequestContext responses expose ``headers`` as a property.
- Semantic DOM assertions use ``text_content()``.
- ``inner_text()`` is retained only as presentation evidence because CSS
  (for example ``text-transform``) may alter rendered text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol


SEMANTIC_TEXT_AUTHORITY = "DOM_textContent"
RENDERED_TEXT_AUTHORITY = "informational_innerText"


class APIResponseLike(Protocol):
    @property
    def headers(self) -> Mapping[str, str]:
        """Return response headers without an async method call."""


class LocatorLike(Protocol):
    async def text_content(self) -> str | None:
        """Return DOM text content, unaffected by CSS text-transform."""

    async def inner_text(self) -> str:
        """Return rendered text for presentation evidence."""


@dataclass(frozen=True, slots=True)
class TextAuthority:
    """Semantic and rendered text captured from one locator."""

    semantic_text: str
    rendered_text: str

    @property
    def semantic_sha256(self) -> str:
        return sha256(self.semantic_text.encode("utf-8")).hexdigest()

    @property
    def rendered_sha256(self) -> str:
        return sha256(self.rendered_text.encode("utf-8")).hexdigest()

    def evidence(self) -> dict[str, object]:
        return {
            "semantic_source": SEMANTIC_TEXT_AUTHORITY,
            "rendered_source": RENDERED_TEXT_AUTHORITY,
            "semantic_sha256": self.semantic_sha256,
            "rendered_sha256": self.rendered_sha256,
            "css_text_transform_is_not_semantic_authority": True,
        }


def api_response_headers(response: APIResponseLike) -> dict[str, str]:
    """Copy APIResponse headers using the Playwright Python property contract.

    Header keys are normalized to lower case for stable evidence and lookup.
    No ``all_headers()`` method is used or expected.
    """

    headers = response.headers
    if not isinstance(headers, Mapping):
        raise TypeError("APIResponse.headers must be a mapping")

    normalized: dict[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("APIResponse.headers must contain string pairs")
        normalized[key.lower()] = value
    return normalized


async def read_text_authority(locator: LocatorLike) -> TextAuthority:
    """Capture semantic DOM text and rendered presentation text separately."""

    semantic_text = (await locator.text_content()) or ""
    rendered_text = await locator.inner_text()

    if not isinstance(rendered_text, str):
        raise TypeError("Locator.inner_text() must return str")

    return TextAuthority(
        semantic_text=semantic_text,
        rendered_text=rendered_text,
    )


def missing_semantic_tokens(
    authority: TextAuthority,
    required_tokens: Sequence[str],
) -> tuple[str, ...]:
    """Return required tokens absent from DOM semantic text."""

    return tuple(
        token
        for token in required_tokens
        if token not in authority.semantic_text
    )


def assert_semantic_tokens(
    authority: TextAuthority,
    required_tokens: Sequence[str],
    *,
    context: str,
) -> None:
    """Assert tokens against DOM textContent, never rendered innerText."""

    missing = missing_semantic_tokens(authority, required_tokens)
    if missing:
        raise AssertionError(
            f"{context} DOM semantic text missing tokens: {list(missing)}"
        )
