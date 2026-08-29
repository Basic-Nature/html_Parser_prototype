from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "webapp" / "parser" / "parse_trace.py"

def _load_module():
    name = "electionpulse_parse_trace_contract_test_module"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def test_public_runtime_redacts_source_ref():
    trace = _load_module()
    trace.clear_recent_parse_traces()
    with trace.parse_run_scope(session_id="sess_test", source_ref="https://should-not-survive.example/results", source_scope="public_registry", public_runtime=True):
        trace.record_parse_observation(kind="jurisdiction_hint", value_summary={"state": "Utah"}, provenance="DETERMINISTIC")
        trace.observe_terminal_outcome(status="success", metadata={"state": "Utah"})
    row = trace.get_recent_parse_traces(limit=1)[0]
    assert row["source_ref"] is None
    assert row["source_ref_redacted"] is True
    assert "should-not-survive.example" not in repr(row)
    assert row["outcome"]["status"] == "SUCCESS"

def test_nested_scope_reuses_single_trace():
    trace = _load_module()
    trace.clear_recent_parse_traces()
    with trace.parse_run_scope(session_id="sess_nested", source_ref="internal://fixture", source_scope="internal", public_runtime=False) as outer:
        outer_id = outer["trace_id"]
        with trace.parse_run_scope(session_id="sess_nested", source_ref="internal://fixture", source_scope="internal", public_runtime=False) as inner:
            assert inner["trace_id"] == outer_id
            trace.record_parse_attempt(stage="acquisition_attempt", strategy="playwright", selection_reason="fixture")
    rows = trace.get_recent_parse_traces(limit=5)
    assert len(rows) == 1
    assert rows[0]["trace_id"] == outer_id
    assert len(rows[0]["attempts"]) == 1

def test_terminal_status_normalization_preserves_raw_status():
    trace = _load_module()
    trace.clear_recent_parse_traces()
    with trace.parse_run_scope(session_id=None, source_ref="internal://fixture", source_scope="internal", public_runtime=False):
        trace.observe_terminal_outcome(status="fail", reason_code="fixture_failure")
    outcome = trace.get_recent_parse_traces(limit=1)[0]["outcome"]
    assert outcome["status"] == "FAILED"
    assert outcome["raw_status"] == "fail"
    assert outcome["reason_code"] == "fixture_failure"

def test_trace_calls_without_active_scope_are_noops():
    trace = _load_module()
    trace.clear_recent_parse_traces()
    assert trace.record_parse_attempt(stage="x", strategy="y", selection_reason="z") is False
    assert trace.record_parse_observation(kind="x", value_summary={"a": 1}) is False
    assert trace.observe_terminal_outcome(status="success") is False
    assert trace.get_recent_parse_traces(limit=10) == []

def test_ml_nlp_provenance_is_explicit():
    trace = _load_module()
    trace.clear_recent_parse_traces()
    with trace.parse_run_scope(session_id="sess_ml", source_ref="internal://fixture", source_scope="internal", public_runtime=False):
        trace.record_parse_observation(kind="candidate_context", value_summary={"office": "President"}, provenance="ML_NLP_PROPOSED", confidence=0.72)
    obs = trace.get_recent_parse_traces(limit=1)[0]["observations"][0]
    assert obs["provenance"] == "ML_NLP_PROPOSED"
    assert obs["confidence"] == 0.72

def test_parser_body_exception_propagates_unchanged():
    trace = _load_module()
    trace.clear_recent_parse_traces()

    class ParserBodySentinel(RuntimeError):
        pass

    caught = None
    try:
        with trace.parse_run_scope(
            session_id="sess_exception",
            source_ref="internal://fixture",
            source_scope="internal",
            public_runtime=False,
        ):
            raise ParserBodySentinel("body-failure")
    except Exception as exc:
        caught = exc

    assert isinstance(caught, ParserBodySentinel)
    assert str(caught) == "body-failure"
    rows = trace.get_recent_parse_traces(limit=1)
    assert len(rows) == 1
    assert rows[0]["completed_at"] is not None

def test_public_runtime_redacts_urls_in_nested_trace_values():
    trace = _load_module()
    trace.clear_recent_parse_traces()

    raw_url = "https://official.example/elections/results?contest=president"
    with trace.parse_run_scope(
        session_id="sess_public_nested",
        source_ref=raw_url,
        source_scope="public_registry",
        public_runtime=True,
    ):
        trace.record_parse_observation(
            kind="failure_context",
            value_summary={
                "message": f"Failed while reading {raw_url}",
                "nested": [{"redirect": raw_url}],
            },
            provenance="OBSERVED",
            source_location=f"source={raw_url}",
        )
        trace.record_parse_attempt(
            stage="download_discovery",
            strategy=f"fetch:{raw_url}",
            selection_reason=f"redirected_from={raw_url}",
            details={"artifact": raw_url},
        )
        trace.observe_terminal_outcome(
            status="error",
            reason_code=f"failed_at:{raw_url}",
            metadata={"handler": f"handler_for:{raw_url}"},
        )

    row = trace.get_recent_parse_traces(limit=1)[0]
    rendered = repr(row)
    assert "official.example" not in rendered
    assert raw_url not in rendered
    assert "<redacted-url>" in rendered


def test_trace_value_stringification_failure_is_fail_open():
    trace = _load_module()
    trace.clear_recent_parse_traces()

    class ExplosiveString:
        def __str__(self):
            raise RuntimeError("stringification must not escape trace boundary")

    with trace.parse_run_scope(
        session_id="sess_bad_value",
        source_ref="internal://fixture",
        source_scope="internal",
        public_runtime=False,
    ):
        assert trace.record_parse_observation(
            kind="bad_value",
            value_summary=ExplosiveString(),
        ) is True
        assert trace.record_parse_attempt(
            stage="test",
            strategy=ExplosiveString(),
            selection_reason="adversarial-value",
        ) is True

    row = trace.get_recent_parse_traces(limit=1)[0]
    assert row["observations"][0]["value_summary"] == "<trace-value-unavailable>"
    assert row["attempts"][0]["strategy"] == "<trace-value-unavailable>"


def test_trace_storage_is_bounded():
    trace = _load_module()
    trace.clear_recent_parse_traces()

    for index in range(trace.MAX_RECENT_TRACES + 5):
        with trace.parse_run_scope(
            session_id=f"sess_{index}",
            source_ref=f"internal://fixture/{index}",
            source_scope="internal",
            public_runtime=False,
        ):
            pass

    rows = trace.get_recent_parse_traces(limit=trace.MAX_RECENT_TRACES + 50)
    assert len(rows) == trace.MAX_RECENT_TRACES
    assert rows[-1]["session_id"] == f"sess_{trace.MAX_RECENT_TRACES + 4}"

def test_zero_and_negative_trace_limits_return_empty():
    trace = _load_module()
    trace.clear_recent_parse_traces()

    with trace.parse_run_scope(
        session_id="sess_limit",
        source_ref="internal://fixture",
        source_scope="internal",
        public_runtime=False,
    ):
        pass

    assert trace.get_recent_parse_traces(limit=0) == []
    assert trace.get_recent_parse_traces(limit=-1) == []
    assert len(trace.get_recent_parse_traces(limit=1)) == 1


def test_trace_item_truncation_is_explicit():
    trace = _load_module()
    trace.clear_recent_parse_traces()
    original_limit = trace.MAX_ITEMS_PER_TRACE
    trace.MAX_ITEMS_PER_TRACE = 2
    try:
        with trace.parse_run_scope(
            session_id="sess_truncation",
            source_ref="internal://fixture",
            source_scope="internal",
            public_runtime=False,
        ):
            for index in range(3):
                trace.record_parse_attempt(
                    stage="test",
                    strategy=f"attempt-{index}",
                    selection_reason="fixture",
                )
            for index in range(3):
                trace.record_parse_observation(
                    kind=f"observation-{index}",
                    value_summary=index,
                )

        row = trace.get_recent_parse_traces(limit=1)[0]
        assert len(row["attempts"]) == 2
        assert len(row["observations"]) == 2
        assert row["limits"]["max_items_per_collection"] == 2
        assert row["limits"]["dropped_attempts"] == 1
        assert row["limits"]["dropped_observations"] == 1
    finally:
        trace.MAX_ITEMS_PER_TRACE = original_limit


def test_public_top_level_trace_fields_share_url_redaction_boundary():
    trace = _load_module()
    trace.clear_recent_parse_traces()

    raw_url = "https://top-level.example/private"
    with trace.parse_run_scope(
        session_id=f"sess:{raw_url}",
        source_ref=raw_url,
        source_scope=f"public_registry:{raw_url}",
        public_runtime=True,
    ):
        pass

    row = trace.get_recent_parse_traces(limit=1)[0]
    rendered = repr(row)
    assert "top-level.example" not in rendered
    assert raw_url not in rendered
    assert "<redacted-url>" in rendered

