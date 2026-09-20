"""Structured evidence adapter for already-authorized trusted Ballot Lens runs."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .ballot_lens_checkpoint_runtime import ACTIONS, CHECKPOINTS, STATES

MODES = frozenset({"trusted_url", "manual_upload", "worklist"})


@dataclass
class TrustedBallotLensRuntime:
    run_mode: str
    session_id: str
    safe_emit: Callable[[dict[str, object]], None] | None = None
    _checkpoint_sequence: int = field(default=0, init=False)
    _workflow_output: dict[str, object] | None = field(default=None, init=False)
    _parser_observations: list[dict[str, object]] = field(default_factory=list, init=False)
    _workflow_completion: dict[str, object] | None = field(default=None, init=False)

    def __post_init__(self):
        self.run_mode = str(self.run_mode or "").strip()
        self.session_id = str(self.session_id or "").strip()
        if self.run_mode not in MODES or not self.session_id:
            raise ValueError("invalid trusted runtime")

    def _emit(self, payload):
        if self.safe_emit:
            self.safe_emit(
                {
                    **payload,
                    "session_id": self.session_id,
                    "run_mode": self.run_mode,
                }
            )

    def _text(self, value, maxlen, required=False):
        if value is None:
            if required:
                raise ValueError("required text")
            return None
        text = str(value).strip()
        if not text and required:
            raise ValueError("required text")
        return text[:maxlen] if text else None

    def emit_started(self):
        self._emit(
            {
                "level": "INFO",
                "type": "trusted_runtime",
                "reason_code": "trusted_runtime_started",
                "message": "Trusted parser runtime started.",
            }
        )

    def record_checkpoint(
        self,
        *,
        checkpoint_id,
        state,
        reason_code=None,
        summary=None,
        evidence_count=0,
        requires_action=False,
        action_type=None,
    ):
        if checkpoint_id not in CHECKPOINTS or state not in STATES:
            raise ValueError("invalid checkpoint")
        if (
            isinstance(evidence_count, bool)
            or not isinstance(evidence_count, int)
            or evidence_count < 0
        ):
            raise ValueError("invalid evidence count")
        if action_type is not None and action_type not in ACTIONS:
            raise ValueError("invalid action")
        if requires_action != (action_type is not None):
            raise ValueError("action fields disagree")
        self._checkpoint_sequence += 1
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "sequence": self._checkpoint_sequence,
            "state": state,
            "label": CHECKPOINTS[checkpoint_id],
            "reason_code": self._text(reason_code, 128),
            "summary": self._text(summary, 360),
            "evidence_count": evidence_count,
            "requires_action": requires_action,
            "action_type": action_type,
            "updated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        level = (
            "ERROR"
            if state == "error"
            else "WARNING"
            if state == "warning"
            else "INFO"
        )
        self._emit(
            {
                "level": level,
                "type": "trusted_runtime_checkpoint",
                "reason_code": "trusted_runtime_checkpoint_updated",
                "message": "Structured trusted parser checkpoint updated.",
                "checkpoint": dict(checkpoint),
            }
        )
        return checkpoint

    def record_action_required(
        self,
        *,
        prompt_id,
        checkpoint_id,
        action_type,
        summary,
    ):
        if checkpoint_id not in CHECKPOINTS or action_type not in ACTIONS:
            raise ValueError("invalid action")
        action = {
            "prompt_id": self._text(prompt_id, 128, True),
            "checkpoint_id": checkpoint_id,
            "action_type": action_type,
            "summary": self._text(summary, 360, True),
        }
        self._emit(
            {
                "level": "WARNING",
                "type": "trusted_runtime_action_required",
                "reason_code": "trusted_runtime_action_required",
                "message": "Structured trusted parser action is required.",
                "action": action,
            }
        )
        return action

    def record_action_resolved(self, *, prompt_id):
        self._emit(
            {
                "level": "INFO",
                "type": "trusted_runtime_action_resolved",
                "reason_code": "trusted_runtime_action_resolved",
                "message": "Structured trusted parser action resolved.",
                "prompt_id": self._text(prompt_id, 128, True),
            }
        )

    def record_result_checkpoints(self, *, headers: Sequence[str], contest):
        present = bool(str(contest or "").strip())
        self.record_checkpoint(
            checkpoint_id="contest.select",
            state="complete" if present else "warning",
            reason_code=(
                "trusted_contest_context_present"
                if present
                else "trusted_contest_context_missing"
            ),
            summary=(
                "Parser result returned contest context."
                if present
                else "Parser result did not expose contest context."
            ),
            evidence_count=1 if present else 0,
        )
        methods = [
            header
            for header in headers
            if isinstance(header, str)
            and " - " in header
            and not header.endswith(" - Total Votes")
            and not header.endswith(" - Total")
        ]
        self.record_checkpoint(
            checkpoint_id="vote_methods.detect",
            state="complete" if methods else "warning",
            reason_code=(
                "trusted_vote_method_columns_present"
                if methods
                else "trusted_vote_method_columns_not_observed"
            ),
            summary=(
                "Method-specific result columns were observed."
                if methods
                else "No method-specific result columns were observed."
            ),
            evidence_count=len(methods),
        )

    def capture_parser_observation(self, payload: Mapping[str, object]) -> None:
        if self.run_mode != "worklist":
            return
        if not isinstance(payload, Mapping):
            raise TypeError("parser observation payload must be an object")
        if payload.get("contract") != "parser_observation_bundle_v1":
            raise ValueError("unexpected parser observation contract")
        authority = payload.get("authority")
        if (
            not isinstance(authority, Mapping)
            or authority.get("canonical") is not False
            or payload.get("raw_rows_included") is not False
            or payload.get("raw_headers_included") is not False
            or payload.get("automatic_timestamp") is not False
        ):
            raise ValueError("parser observation violates W22 authority boundary")
        self._parser_observations.append(deepcopy(dict(payload)))

    def capture_finalized_output(
        self,
        *,
        headers: Sequence[str],
        rows: Sequence[Mapping[str, object]],
    ) -> None:
        if self.run_mode != "worklist":
            return
        if self._workflow_output is not None:
            raise RuntimeError(
                "one Workflow pass must produce exactly one finalized output"
            )
        self._workflow_output = {
            "headers": [str(header) for header in headers],
            "rows": [deepcopy(dict(row)) for row in rows],
            "csv_path": None,
            "metadata_path": None,
        }

    def capture_persisted_output(
        self,
        *,
        csv_path: str,
        metadata_path: str,
    ) -> None:
        if self.run_mode != "worklist":
            return
        if self._workflow_output is None:
            raise RuntimeError(
                "Workflow persisted output arrived before finalized output"
            )
        if self._workflow_output.get("csv_path") is not None:
            raise RuntimeError(
                "one Workflow pass must persist exactly one output"
            )
        self._workflow_output["csv_path"] = str(csv_path or "")
        self._workflow_output["metadata_path"] = str(metadata_path or "")

    def workflow_completion_capture(self) -> dict[str, object]:
        if self.run_mode != "worklist":
            raise RuntimeError("Workflow completion capture requires worklist mode")
        if self._workflow_output is None:
            raise RuntimeError("Workflow run did not capture finalized output")
        if not self._workflow_output.get("csv_path"):
            raise RuntimeError("Workflow run did not persist results.csv")
        if not self._parser_observations:
            raise RuntimeError("Workflow run did not emit parser observations")
        return {
            **deepcopy(self._workflow_output),
            "observations": deepcopy(self._parser_observations),
        }

    def record_workflow_completion(self, payload: Mapping[str, object]) -> None:
        if self.run_mode != "worklist":
            raise RuntimeError("Workflow completion requires worklist mode")
        if payload.get("contract") != "workflow_dl1_runtime_bridge_v1":
            raise ValueError("unexpected Workflow completion contract")
        self._workflow_completion = {
            "success": bool(payload.get("success")),
            "task_id": str(payload.get("task_id") or ""),
            "pass_id": str(payload.get("pass_id") or ""),
            "status": str(payload.get("status") or ""),
            "row_version": payload.get("row_version"),
            "current_stage": payload.get("current_stage"),
            "stage_condition": payload.get("stage_condition"),
            "committed": bool(payload.get("committed")),
        }
        self._emit(
            {
                "level": "INFO",
                "type": "workflow_dl1_completion",
                "reason_code": "workflow_dl1_submitted",
                "message": "Governed DL1 parser result was committed to Workflow staging and submitted.",
                "workflow": dict(self._workflow_completion),
            }
        )

    def persisted_outputs(self, paths: Sequence[str]):
        outputs = []
        for index, raw in enumerate(paths):
            rel = str(raw or "").replace("\\", "/").strip("/")
            if rel:
                outputs.append(
                    {
                        "output_id": f"{self.session_id}:persisted:{index + 1}",
                        "label": os.path.basename(rel) or rel,
                        "persistence": "persisted",
                        "download_available": True,
                    }
                )
        return outputs

    def result_payload(self, *, terminal_status, terminal_reason_code, outputs):
        if terminal_status not in {
            "success",
            "completed_with_errors",
            "failed",
            "cancelled",
        }:
            raise ValueError("invalid terminal status")
        payload = {
            "contract": "ballot_lens_trusted_runtime_result_v1",
            "session_id": self.session_id,
            "run_mode": self.run_mode,
            "terminal_status": terminal_status,
            "terminal_reason_code": self._text(terminal_reason_code, 128),
            "status_counts": {terminal_status: 1},
            "outputs": list(outputs),
        }
        if self._workflow_completion is not None:
            payload["workflow_completion"] = dict(self._workflow_completion)
        return payload
