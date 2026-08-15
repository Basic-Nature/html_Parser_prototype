from __future__ import annotations

from webapp.parser.handlers.formats import html_dynamic_fallback
from webapp.parser.utils import ml_telemetry


def test_dynamic_fallback_with_html_text(monkeypatch):
    captured_ml_events = []

    def capture_ml_event(*args, **kwargs):
        captured_ml_events.append((args, kwargs))

    # log_extraction_quality imports record_ml_event at call time. Patch the
    # concrete persistence sink so the real fallback/generic-result path runs
    # without appending to the tracked ml_usage_telemetry.jsonl file.
    monkeypatch.setattr(
        ml_telemetry,
        "record_ml_event",
        capture_ml_event,
    )

    html = """<html><body>
    <p>Candidate A: 123</p>
    <p>Candidate B: 456</p>
    </body></html>"""
    ctx = {"html_text": html, "file_name": "test_html"}

    res = html_dynamic_fallback.parse(
        page=None,
        coordinator=None,
        context=ctx,
        session_id="test",
    )

    # Preserve the recovered smoke contract.
    assert res is None or (isinstance(res, tuple) and len(res) == 4)

    # When generic extraction succeeds, extraction-quality telemetry should
    # reach the patched sink rather than disk. A None fallback result may
    # legitimately return before quality telemetry is emitted.
    if res is not None:
        assert captured_ml_events
        assert any(
            "extraction_quality" in repr(args)
            or "extraction_quality" in repr(kwargs)
            for args, kwargs in captured_ml_events
        )