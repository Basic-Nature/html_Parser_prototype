from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..contracts.browser_challenge import (
    ChallengeHandoff,
    ChallengeObservation,
    ChallengeStatus,
    build_human_challenge_handoff,
)


LEGACY_CHALLENGE_ADAPTER_VERSION = "legacy_challenge_observation_adapter_v1"


def _normalize_optional_hint(
    value: str | None,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be None or a nonblank string")
    return value.strip()


def _normalize_strings(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ValueError(f"{field_name} entries must be nonblank strings")
    return tuple(value.strip() for value in normalized)


@dataclass(frozen=True)
class LegacyChallengeAdaptation:
    """Typed result of adapting bounded legacy detection facts.

    This wrapper carries only the typed noncanonical observation and optional
    human handoff policy. It owns no live browser state and performs no browser
    operation.
    """

    observation: ChallengeObservation
    handoff: ChallengeHandoff | None


def adapt_legacy_challenge_detection(
    detected: bool,
    *,
    session_id: str,
    browser_context_ref: str,
    vendor_hint: str | None = None,
    challenge_type_hint: str | None = None,
    browser_engine_hint: str | None = None,
    indicators: Iterable[str] = (),
    evidence_refs: Iterable[str] = (),
    human_intervention_required: bool | None = None,
    human_intervention_completed: bool | None = None,
    observed_at: str | None = None,
    create_human_handoff: bool = False,
) -> LegacyChallengeAdaptation | None:
    """Adapt an already-computed legacy challenge detection into typed evidence.

    ``detected=False`` returns ``None`` so this seam does not fabricate a
    challenge. The adapter does not inspect a browser, HTML, cookies, storage,
    headers, or User-Agent values.

    Vendor/type/browser hints are descriptive metadata only. They never select
    a browser implementation or a resolution strategy.
    """

    if not isinstance(detected, bool):
        raise TypeError("detected must be bool")

    if not detected:
        return None

    normalized_vendor = _normalize_optional_hint(
        vendor_hint,
        "vendor_hint",
    )
    normalized_type = _normalize_optional_hint(
        challenge_type_hint,
        "challenge_type_hint",
    )
    normalized_engine = _normalize_optional_hint(
        browser_engine_hint,
        "browser_engine_hint",
    )
    normalized_indicators = _normalize_strings(
        indicators,
        "indicators",
    )
    normalized_evidence_refs = _normalize_strings(
        evidence_refs,
        "evidence_refs",
    )

    observation = ChallengeObservation(
        session_id=session_id,
        browser_context_ref=browser_context_ref,
        status=ChallengeStatus.DETECTED,
        vendor_hint=normalized_vendor,
        challenge_type_hint=normalized_type,
        browser_engine_hint=normalized_engine,
        indicators=normalized_indicators,
        human_intervention_required=human_intervention_required,
        human_intervention_completed=human_intervention_completed,
        automated_resolution_attempted=False,
        evidence_refs=normalized_evidence_refs,
        observed_at=observed_at,
        canonical_authority=False,
    )

    handoff = (
        build_human_challenge_handoff(observation)
        if create_human_handoff
        else None
    )

    return LegacyChallengeAdaptation(
        observation=observation,
        handoff=handoff,
    )