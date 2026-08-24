from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


CHALLENGE_CONTRACT_VERSION = "browser_challenge_v1"


class ChallengeStatus(str, Enum):
    """Lifecycle state for a browser interaction challenge."""

    DETECTED = "detected"
    WAITING_FOR_HUMAN = "waiting_for_human"
    HUMAN_INTERACTION_ACTIVE = "human_interaction_active"
    CLEARED = "cleared"
    TIMED_OUT = "timed_out"
    ABORTED = "aborted"


class ChallengePresentation(str, Enum):
    """How an authorized human is expected to interact with the challenge."""

    CONTROLLED_BROWSER_CONTEXT = "controlled_browser_context"


class ChallengeAuthority(str, Enum):
    """Authority classification for challenge observations."""

    NONCANONICAL_OBSERVATION = "noncanonical_observation"


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


def _normalize_string_tuple(
    values: tuple[str, ...] | list[str],
    field_name: str,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ValueError(f"{field_name} entries must be nonblank strings")
    return normalized


@dataclass(frozen=True)
class ChallengeObservation:
    """Browser-neutral, noncanonical evidence that a challenge was observed.

    The observation deliberately carries no raw browser object, cookies, storage
    state, page HTML, screenshot bytes, or User-Agent value. Those belong to
    separately governed browser/session and evidence layers.

    ``vendor_hint`` and ``challenge_type_hint`` are optional metadata only.
    They do not select control flow and are intentionally free-form so future
    challenge types do not require contract changes.
    """

    session_id: str
    browser_context_ref: str
    status: ChallengeStatus
    vendor_hint: str | None = None
    challenge_type_hint: str | None = None
    browser_engine_hint: str | None = None
    indicators: tuple[str, ...] = field(default_factory=tuple)
    human_intervention_required: bool | None = None
    human_intervention_completed: bool | None = None
    automated_resolution_attempted: bool = False
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    observed_at: str | None = None
    authority: ChallengeAuthority = ChallengeAuthority.NONCANONICAL_OBSERVATION
    canonical_authority: bool = False

    def __post_init__(self) -> None:
        _require_nonblank(self.session_id, "session_id")
        _require_nonblank(self.browser_context_ref, "browser_context_ref")

        object.__setattr__(
            self,
            "indicators",
            _normalize_string_tuple(self.indicators, "indicators"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _normalize_string_tuple(self.evidence_refs, "evidence_refs"),
        )

        if self.vendor_hint is not None:
            _require_nonblank(self.vendor_hint, "vendor_hint")
        if self.challenge_type_hint is not None:
            _require_nonblank(self.challenge_type_hint, "challenge_type_hint")
        if self.browser_engine_hint is not None:
            _require_nonblank(self.browser_engine_hint, "browser_engine_hint")
        if self.observed_at is not None:
            _require_nonblank(self.observed_at, "observed_at")

        if self.automated_resolution_attempted:
            raise ValueError(
                "automated challenge resolution is outside this contract"
            )
        if self.canonical_authority:
            raise ValueError(
                "browser challenge observations cannot hold canonical authority"
            )
        if self.authority is not ChallengeAuthority.NONCANONICAL_OBSERVATION:
            raise ValueError("unsupported browser challenge authority")


@dataclass(frozen=True)
class ChallengeHandoff:
    """Policy contract for pausing automation and presenting the same context.

    This object does not perform presentation or browser control. It states the
    required authority boundary for a later runtime adapter.
    """

    observation: ChallengeObservation
    presentation: ChallengePresentation = (
        ChallengePresentation.CONTROLLED_BROWSER_CONTEXT
    )
    pause_automation: bool = True
    human_action_authorized: bool = True
    resume_same_context_required: bool = True
    automated_bypass_authorized: bool = False
    canonical_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ChallengeObservation):
            raise TypeError("observation must be ChallengeObservation")
        if self.presentation is not ChallengePresentation.CONTROLLED_BROWSER_CONTEXT:
            raise ValueError("unsupported challenge presentation")
        if not self.pause_automation:
            raise ValueError("challenge handoff must pause automation")
        if not self.human_action_authorized:
            raise ValueError("challenge handoff must authorize human interaction")
        if not self.resume_same_context_required:
            raise ValueError(
                "challenge handoff must require same-context continuation"
            )
        if self.automated_bypass_authorized:
            raise ValueError("automated CAPTCHA bypass cannot be authorized")
        if self.canonical_authority:
            raise ValueError(
                "browser challenge handoff cannot hold canonical authority"
            )


def build_human_challenge_handoff(
    observation: ChallengeObservation,
) -> ChallengeHandoff:
    """Return the fail-closed human handoff policy for an observation."""

    return ChallengeHandoff(observation=observation)