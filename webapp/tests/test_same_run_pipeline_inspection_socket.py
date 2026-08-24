from __future__ import annotations

from webapp.parser.Context_Integration.context_write_policy import ContextWriteKind
from webapp.parser.contracts.table_pipeline import (
    SourceProvenance,
    TablePipelineResult,
    TableStage,
    TransformationRecord,
)
from webapp.parser.handlers.formats import csv_handler
from webapp.parser.socket_ballot_lens_orchestration import (
    _make_pipeline_inspection_emitter,
)


class FakeSocketIO:
    def __init__(self):
        self.calls = []

    def emit(self, event, payload, room=None):
        self.calls.append((event, payload, room))


def _result():
    return TablePipelineResult.from_sequences(
        stage=TableStage.INTERPRETED,
        headers=("Precinct", "Election Day"),
        rows=(
            {"Precinct": "P-1", "Election Day": None},
            {"Precinct": "P-2", "Election Day": 0},
            {"Precinct": "P-3", "Election Day": -4},
        ),
        source_provenance=SourceProvenance(
            source_type="csv",
            source_sha256="a" * 64,
            evidence_ref="fixture://c2g25",
        ),
        transformations=(
            TransformationRecord(
                sequence=0,
                from_stage=TableStage.INTERPRETED,
                to_stage=TableStage.INTERPRETED,
                operation="vote_method_header_canonicalization",
                details={
                    "unknown_example": None,
                    "confirmed_zero_example": 0,
                    "signed_example": -4,
                },
            ),
        ),
        write_kind=ContextWriteKind.NONE,
    )


def _projection():
    captured = []
    assert csv_handler._emit_pipeline_inspection_if_requested(
        _result(),
        inspection_emit_func=captured.append,
    )
    assert len(captured) == 1
    return captured[0]


def test_csv_emit_sink_is_optional_and_safe():
    assert (
        csv_handler._emit_pipeline_inspection_if_requested(
            _result(),
            inspection_emit_func=None,
        )
        is False
    )

    payload = _projection()
    assert payload["contract"] == "pipeline_inspection_v1"
    assert payload["authority"]["canonical"] is False
    assert payload["rows_included"] is False
    assert payload["headers_included"] is False

    details = payload["transformations"][0]["details"]
    assert details["unknown_example"] is None
    assert details["confirmed_zero_example"] == 0
    assert details["signed_example"] == -4


def test_socket_emitter_is_session_scoped_and_hides_principal():
    socketio = FakeSocketIO()
    emit_inspection = _make_pipeline_inspection_emitter(
        "session-25",
        "principal-25",
        {"socketio": socketio},
    )

    emit_inspection(_projection())

    assert len(socketio.calls) == 1
    event, envelope, room = socketio.calls[0]

    assert event == "pipeline_inspection"
    assert room == "session-25"
    assert envelope["contract"] == "pipeline_inspection_socket_v1"
    assert envelope["authority"] == {
        "canonical": False,
        "transport": "same_run_socket",
    }
    assert envelope["session_id"] == "session-25"
    assert envelope["inspection"]["contract"] == "pipeline_inspection_v1"
    assert "principal-25" not in repr(envelope)


def test_socket_emitter_rejects_missing_ownership():
    socketio = FakeSocketIO()

    for session_id, principal in (
        ("", "principal-25"),
        ("session-25", None),
        ("session-25", ""),
    ):
        try:
            _make_pipeline_inspection_emitter(
                session_id,
                principal,
                {"socketio": socketio},
            )
        except ValueError:
            continue
        raise AssertionError("missing ownership must fail closed")


def test_socket_emitter_rejects_canonical_payload():
    socketio = FakeSocketIO()
    emit_inspection = _make_pipeline_inspection_emitter(
        "session-25",
        "principal-25",
        {"socketio": socketio},
    )

    payload = _projection()
    payload["authority"]["canonical"] = True

    try:
        emit_inspection(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("canonical payload must be rejected")

    assert socketio.calls == []


def test_store_semantics_remain_independent_of_socket_sink():
    assert (
        csv_handler._store_pipeline_inspection_if_requested(
            _result(),
            inspection_store=None,
            session_id="session-25",
            principal=None,
        )
        is False
    )