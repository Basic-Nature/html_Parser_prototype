from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL = REPO_ROOT / "webapp/static/js/pipeline_inspection_panel.js"
CONSUMER = REPO_ROOT / "webapp/static/js/pipeline_inspection_consumer.js"
MODERN = REPO_ROOT / "webapp/static/js/ballot_lens_modern.js"
TEMPLATE = REPO_ROOT / "webapp/templates/ballot_lens.html"


def test_why_panel_script_loads_between_consumer_and_modern() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")

    consumer_idx = text.index("js/pipeline_inspection_consumer.js")
    panel_idx = text.index("js/pipeline_inspection_panel.js")
    modern_idx = text.index("js/ballot_lens_modern.js")

    assert consumer_idx < panel_idx < modern_idx


def test_why_panel_is_a_read_only_artifact_card() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")

    start = text.index('id="pipelineInspectionPanel"')
    end = text.index("</details>", start)
    panel = text[start:end]

    assert "Why this interpretation?" in panel
    assert "noncanonical" in panel
    assert "Read-only parser explanation" in panel
    assert "<button" not in panel
    assert "approve" in panel.lower()
    assert "promote" in panel.lower()


def test_renderer_uses_allowlisted_dom_construction_only() -> None:
    text = PANEL.read_text(encoding="utf-8")

    assert "pipeline:inspection" in text
    assert "buildViewModel" in text
    assert "textContent" in text
    assert "createElement" in text
    assert "replaceChildren" in text

    assert "innerHTML" not in text
    assert "JSON.stringify" not in text
    assert "localStorage" not in text
    assert "sessionStorage" not in text
    assert "fetch(" not in text
    assert ".emit(" not in text


def test_renderer_has_no_canonical_or_mutating_action_surface() -> None:
    text = PANEL.read_text(encoding="utf-8")

    assert "verify-and-promote" not in text
    assert "canonical_action" not in text
    assert "approve_result" not in text
    assert "promotion" not in text.lower()
    assert "authorityLabel: 'NONE'" in text


def test_c2g26_consumer_and_modern_attachment_are_not_reimplemented() -> None:
    panel = PANEL.read_text(encoding="utf-8")
    consumer = CONSUMER.read_text(encoding="utf-8")
    modern = MODERN.read_text(encoding="utf-8")

    assert "PipelineInspectionConsumer.attach(" not in panel
    assert "socket.on('pipeline_inspection'" not in panel
    assert "pipeline_inspection_socket_v1" in consumer
    assert "new CustomEvent('pipeline:inspection'" in modern


def test_panel_does_not_render_raw_rows_headers_or_source_location() -> None:
    text = PANEL.read_text(encoding="utf-8")

    assert "rows_included !== false" in text
    assert "headers_included !== false" in text
    assert "source_uri_included !== false" in text
    assert "source_metadata_included !== false" in text

    assert "source_sha256" not in text
    assert "location" not in text