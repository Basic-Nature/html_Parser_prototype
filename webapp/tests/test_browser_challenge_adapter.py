from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from webapp.parser.contracts.browser_challenge import (
    ChallengePresentation,
    ChallengeStatus,
)
from webapp.parser.services.browser_challenge_adapter import (
    LEGACY_CHALLENGE_ADAPTER_VERSION,
    LegacyChallengeAdaptation,
    adapt_legacy_challenge_detection,
)


def _adapt(**overrides):
    values = {
        "detected": True,
        "session_id": "session-286",
        "browser_context_ref": "context-opaque-286",
    }
    values.update(overrides)
    return adapt_legacy_challenge_detection(**values)


def test_adapter_version_is_explicit():
    assert (
        LEGACY_CHALLENGE_ADAPTER_VERSION
        == "legacy_challenge_observation_adapter_v1"
    )


def test_false_detection_returns_none_without_fabricating_observation():
    result = _adapt(detected=False)
    assert result is None


def test_detected_must_be_actual_bool():
    with pytest.raises(TypeError, match="detected must be bool"):
        _adapt(detected=1)


def test_positive_detection_creates_noncanonical_detected_observation():
    result = _adapt()

    assert isinstance(result, LegacyChallengeAdaptation)
    assert result.observation.status is ChallengeStatus.DETECTED
    assert result.observation.session_id == "session-286"
    assert result.observation.browser_context_ref == "context-opaque-286"
    assert result.observation.canonical_authority is False
    assert result.observation.automated_resolution_attempted is False
    assert result.handoff is None


def test_unknown_vendor_and_type_remain_unknown():
    result = _adapt(
        vendor_hint=None,
        challenge_type_hint=None,
    )

    assert result.observation.vendor_hint is None
    assert result.observation.challenge_type_hint is None


def test_current_or_future_vendor_is_metadata_only():
    first = _adapt(
        vendor_hint="cloudflare",
        challenge_type_hint="managed-challenge",
    )
    future = _adapt(
        vendor_hint="future-provider-v99",
        challenge_type_hint="interactive-generation-42",
    )

    assert first.observation.status is ChallengeStatus.DETECTED
    assert future.observation.status is ChallengeStatus.DETECTED
    assert first.handoff is None
    assert future.handoff is None
    assert first.observation.vendor_hint == "cloudflare"
    assert future.observation.vendor_hint == "future-provider-v99"


def test_indicators_and_evidence_are_bounded_string_copies():
    indicators = ["turnstile-widget", "challenge-page"]
    evidence = ["evidence://challenge/286"]

    result = _adapt(
        indicators=indicators,
        evidence_refs=evidence,
    )

    indicators.append("later-mutation")
    evidence.append("evidence://later")

    assert result.observation.indicators == (
        "turnstile-widget",
        "challenge-page",
    )
    assert result.observation.evidence_refs == (
        "evidence://challenge/286",
    )


def test_whitespace_is_normalized_without_changing_semantics():
    result = _adapt(
        vendor_hint="  cloudflare  ",
        challenge_type_hint="  managed challenge ",
        browser_engine_hint="  chromium ",
        indicators=["  indicator-a  "],
        evidence_refs=["  evidence://a  "],
    )

    assert result.observation.vendor_hint == "cloudflare"
    assert result.observation.challenge_type_hint == "managed challenge"
    assert result.observation.browser_engine_hint == "chromium"
    assert result.observation.indicators == ("indicator-a",)
    assert result.observation.evidence_refs == ("evidence://a",)


def test_empty_optional_hints_fail_closed_instead_of_becoming_unknown():
    with pytest.raises(ValueError, match="vendor_hint"):
        _adapt(vendor_hint=" ")

    with pytest.raises(ValueError, match="challenge_type_hint"):
        _adapt(challenge_type_hint="")


def test_adapter_does_not_invent_human_completion_or_timestamp():
    result = _adapt()

    assert result.observation.human_intervention_required is None
    assert result.observation.human_intervention_completed is None
    assert result.observation.observed_at is None


def test_explicit_human_facts_are_preserved_without_inference():
    result = _adapt(
        human_intervention_required=True,
        human_intervention_completed=False,
        observed_at="evidence-time-ref-286",
    )

    assert result.observation.human_intervention_required is True
    assert result.observation.human_intervention_completed is False
    assert result.observation.observed_at == "evidence-time-ref-286"


def test_handoff_is_created_only_when_explicitly_requested():
    without_handoff = _adapt(
        human_intervention_required=True,
        create_human_handoff=False,
    )
    with_handoff = _adapt(
        human_intervention_required=True,
        create_human_handoff=True,
    )

    assert without_handoff.handoff is None
    assert with_handoff.handoff is not None
    assert (
        with_handoff.handoff.presentation
        is ChallengePresentation.CONTROLLED_BROWSER_CONTEXT
    )
    assert with_handoff.handoff.pause_automation is True
    assert with_handoff.handoff.human_action_authorized is True
    assert with_handoff.handoff.resume_same_context_required is True
    assert with_handoff.handoff.automated_bypass_authorized is False


def test_adapter_has_no_fields_for_raw_browser_or_identity_payloads():
    adaptation_fields = {
        item.name for item in fields(LegacyChallengeAdaptation)
    }

    forbidden = {
        "browser",
        "page",
        "driver",
        "html",
        "page_html",
        "cookies",
        "storage_state",
        "user_agent",
        "headers",
        "screenshot",
    }

    assert adaptation_fields.isdisjoint(forbidden)


def test_adapter_module_is_browser_framework_neutral_and_has_no_io():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "parser"
        / "services"
        / "browser_challenge_adapter.py"
    )
    text = source_path.read_text(encoding="utf-8-sig").lower()

    assert "playwright" not in text
    assert "selenium" not in text
    assert "puppeteer" not in text
    assert "requests" not in text
    assert "httpx" not in text
    assert "open(" not in text
    assert "write_text" not in text
    assert "write_bytes" not in text


def test_adapter_signature_cannot_accept_legacy_secret_payload_names():
    import inspect

    signature = inspect.signature(adapt_legacy_challenge_detection)
    names = set(signature.parameters)

    assert names.isdisjoint(
        {
            "page",
            "driver",
            "browser",
            "html",
            "page_html",
            "cookies",
            "storage_state",
            "user_agent",
            "headers",
            "screenshot",
        }
    )