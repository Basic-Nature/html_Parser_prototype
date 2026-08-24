from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from webapp.parser.contracts.browser_challenge import (
    CHALLENGE_CONTRACT_VERSION,
    ChallengeAuthority,
    ChallengeHandoff,
    ChallengeObservation,
    ChallengePresentation,
    ChallengeStatus,
    build_human_challenge_handoff,
)


def _observation(**overrides) -> ChallengeObservation:
    values = {
        "session_id": "session-123",
        "browser_context_ref": "context-opaque-123",
        "status": ChallengeStatus.DETECTED,
        "vendor_hint": None,
        "challenge_type_hint": None,
        "browser_engine_hint": None,
        "indicators": (),
        "human_intervention_required": True,
        "human_intervention_completed": False,
        "automated_resolution_attempted": False,
        "evidence_refs": (),
        "observed_at": None,
    }
    values.update(overrides)
    return ChallengeObservation(**values)


def test_contract_version_is_explicit_and_stable():
    assert CHALLENGE_CONTRACT_VERSION == "browser_challenge_v1"


def test_observation_is_vendor_agnostic_and_future_vendor_safe():
    observation = _observation(
        vendor_hint="future-provider-v17",
        challenge_type_hint="interactive-widget-generation-9",
        indicators=("opaque-widget-present",),
    )

    assert observation.vendor_hint == "future-provider-v17"
    assert observation.challenge_type_hint == "interactive-widget-generation-9"
    assert observation.indicators == ("opaque-widget-present",)


def test_vendor_hint_can_be_unknown_without_fabrication():
    observation = _observation(
        vendor_hint=None,
        challenge_type_hint=None,
    )

    assert observation.vendor_hint is None
    assert observation.challenge_type_hint is None


def test_observation_has_no_automatic_timestamp():
    observation = _observation()
    assert observation.observed_at is None


def test_observation_is_noncanonical_and_cannot_authorize_automatic_resolution():
    observation = _observation()

    assert observation.authority is ChallengeAuthority.NONCANONICAL_OBSERVATION
    assert observation.canonical_authority is False
    assert observation.automated_resolution_attempted is False

    with pytest.raises(ValueError, match="automated challenge resolution"):
        _observation(automated_resolution_attempted=True)

    with pytest.raises(ValueError, match="canonical authority"):
        _observation(canonical_authority=True)


def test_session_and_context_refs_are_required():
    with pytest.raises(ValueError, match="session_id"):
        _observation(session_id="")

    with pytest.raises(ValueError, match="browser_context_ref"):
        _observation(browser_context_ref="  ")


def test_list_inputs_are_copied_to_immutable_tuples():
    indicators = ["first-indicator"]
    evidence_refs = ["evidence://challenge/1"]

    observation = _observation(
        indicators=indicators,
        evidence_refs=evidence_refs,
    )

    indicators.append("later-mutation")
    evidence_refs.append("evidence://challenge/2")

    assert observation.indicators == ("first-indicator",)
    assert observation.evidence_refs == ("evidence://challenge/1",)


def test_handoff_requires_pause_human_action_and_same_context_policy():
    observation = _observation()
    handoff = build_human_challenge_handoff(observation)

    assert handoff.observation is observation
    assert (
        handoff.presentation
        is ChallengePresentation.CONTROLLED_BROWSER_CONTEXT
    )
    assert handoff.pause_automation is True
    assert handoff.human_action_authorized is True
    assert handoff.resume_same_context_required is True
    assert handoff.automated_bypass_authorized is False
    assert handoff.canonical_authority is False


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"pause_automation": False}, "pause automation"),
        ({"human_action_authorized": False}, "human interaction"),
        ({"resume_same_context_required": False}, "same-context"),
        ({"automated_bypass_authorized": True}, "bypass"),
        ({"canonical_authority": True}, "canonical authority"),
    ],
)
def test_handoff_fails_closed_when_authority_boundary_is_weakened(
    kwargs,
    match,
):
    with pytest.raises(ValueError, match=match):
        ChallengeHandoff(
            observation=_observation(),
            **kwargs,
        )


def test_contract_carries_no_raw_browser_secret_or_page_payload_fields():
    observation_fields = {item.name for item in fields(ChallengeObservation)}
    handoff_fields = {item.name for item in fields(ChallengeHandoff)}

    forbidden = {
        "browser",
        "page",
        "driver",
        "cookies",
        "cookie",
        "storage_state",
        "html",
        "page_html",
        "screenshot",
        "screenshot_bytes",
        "user_agent",
        "headers",
    }

    assert observation_fields.isdisjoint(forbidden)
    assert handoff_fields.isdisjoint(forbidden)


def test_contract_module_has_no_browser_framework_dependency():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "parser"
        / "contracts"
        / "browser_challenge.py"
    )
    text = source_path.read_text(encoding="utf-8-sig").lower()

    assert "playwright" not in text
    assert "selenium" not in text
    assert "puppeteer" not in text


def test_statuses_describe_observation_not_vendor_control_flow():
    assert {status.value for status in ChallengeStatus} == {
        "detected",
        "waiting_for_human",
        "human_interaction_active",
        "cleared",
        "timed_out",
        "aborted",
    }