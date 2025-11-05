from __future__ import annotations

from enum import Enum
from typing import Dict


class SessionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_PROMPT = "waiting_prompt"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ERROR = "error"

    @classmethod
    def as_dict(cls) -> Dict[str, str]:
        return {member.name: member.value for member in cls}


class PipelinePhase(str, Enum):
    PREPARE = "prepare"
    SOURCE = "source"
    RUN = "run"
    RESOLVE = "resolve"
    REVIEW = "review"

    @classmethod
    def ordered(cls) -> list[str]:
        return [member.value for member in cls]


DEFAULT_PHASE_BY_STATE: Dict[str, str] = {
    SessionState.IDLE.value: PipelinePhase.PREPARE.value,
    SessionState.RUNNING.value: PipelinePhase.RUN.value,
    SessionState.WAITING_PROMPT.value: PipelinePhase.RESOLVE.value,
    SessionState.CANCELLING.value: PipelinePhase.RUN.value,
    SessionState.CANCELLED.value: PipelinePhase.PREPARE.value,
    SessionState.COMPLETED.value: PipelinePhase.REVIEW.value,
    SessionState.ERROR.value: PipelinePhase.REVIEW.value,
}


def export_session_enums() -> Dict[str, object]:
    """Return a JSON-serializable payload describing session states/phases."""
    return {
        "states": SessionState.as_dict(),
        "phases": {member.name: member.value for member in PipelinePhase},
        "phase_order": PipelinePhase.ordered(),
        "state_phase_map": dict(DEFAULT_PHASE_BY_STATE),
    }
