from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BALLOT_JS = REPO_ROOT / "webapp" / "static" / "js" / "ballot_lens_modern.js"


def test_pipeline_inspection_attaches_only_after_socket_initialization():
    source = BALLOT_JS.read_text(encoding="utf-8")

    socket_marker = "const socket = io({"
    pipeline_marker = "const PipelineInspectionFrontend = (() => {"

    assert source.count(socket_marker) == 1
    assert source.count(pipeline_marker) == 1
    assert source.index(socket_marker) < source.index(pipeline_marker)


def test_pipeline_inspection_consumer_attach_has_initialized_socket_available():
    source = BALLOT_JS.read_text(encoding="utf-8")

    socket_index = source.index("const socket = io({")
    attach_index = source.index("consumer.attach(")

    assert socket_index < attach_index
    assert "Cannot access 'socket' before initialization" not in source
