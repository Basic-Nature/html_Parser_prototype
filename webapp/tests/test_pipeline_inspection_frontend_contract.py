from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONSUMER = REPO_ROOT / "webapp/static/js/pipeline_inspection_consumer.js"
MODERN = REPO_ROOT / "webapp/static/js/ballot_lens_modern.js"
TEMPLATE = REPO_ROOT / "webapp/templates/ballot_lens.html"


def test_consumer_is_loaded_before_ballot_lens_modern() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    consumer_idx = text.index("js/pipeline_inspection_consumer.js")
    modern_idx = text.index("js/ballot_lens_modern.js")

    assert consumer_idx < modern_idx


def test_modern_frontend_attaches_existing_socket_without_rendering_panel() -> None:
    text = MODERN.read_text(encoding="utf-8")

    assert "window.PipelineInspectionConsumer" in text
    assert "consumer.attach(" in text
    assert "new CustomEvent('pipeline:inspection'" in text
    assert "pipeline_inspection" in text
    assert "pipelineInspectionPanel" not in text
    assert "pipeline-inspection-panel" not in text


def test_consumer_has_no_persistence_network_or_canonical_action_surface() -> None:
    text = CONSUMER.read_text(encoding="utf-8")

    assert "localStorage" not in text
    assert "sessionStorage" not in text
    assert "fetch(" not in text
    assert ".emit(" not in text
    assert "verify-and-promote" not in text
    assert "canonical_action" not in text


def test_consumer_rejects_raw_election_payload_keys() -> None:
    text = CONSUMER.read_text(encoding="utf-8")

    for key in ("rows", "headers", "source_uri", "source_metadata"):
        assert repr(key) in text or f"'{key}'" in text

    assert "rows_included !== false" in text
    assert "headers_included !== false" in text
    assert "authority.canonical !== false" in text


def test_consumer_is_bounded_and_copying() -> None:
    text = CONSUMER.read_text(encoding="utf-8")

    assert "DEFAULT_MAX_PER_SESSION = 20" in text
    assert "DEFAULT_MAX_SESSIONS = 25" in text
    assert "cloneJson" in text
    assert "entries.shift()" in text
    assert "bySession.delete(evicted)" in text