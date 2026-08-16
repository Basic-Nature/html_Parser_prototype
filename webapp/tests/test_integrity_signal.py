#!/usr/bin/env python3
"""Quick test to verify integrity_signal emission in pipeline.

Simulates a parse session and checks that:
1. Context digest is written
2. Rolling trend file is updated
3. Integrity signal can be computed

Run from repo root:
    python tests/test_integrity_signal.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Ensure we're running from repo root
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from webapp.parser.utils import html_scanner


class _FakePage:
    # Minimal page object matching the scanner's current page.content() contract.
    def __init__(self, html: str, url: str):
        self._html = html
        self.url = url

    def content(self) -> str:
        return self._html


class _IsolatedCoordinator:
    # Scanner collaborator that never persists runtime/evidence context.
    def get_contests(self):
        return []

    def extract_field(self, *args, **kwargs):
        return []

    def get_known_county_to_PRECINCTS_map(self):
        return {}

    def extract_entities(self, *args, **kwargs):
        return []

    def extract_locations(self, *args, **kwargs):
        return []

    def extract_dates(self, *args, **kwargs):
        return []

    def ingest_ocr_text(self, *args, **kwargs):
        return {}

    def organize_and_enrich(self, *args, **kwargs):
        return {}


def _run_integrity_signal_emission(work_dir: Path) -> None:
    test_html = (
        "<html><body>"
        '<div class="contest-panel">'
        "<h2>President</h2>"
        '<div class="candidate">John Doe - 1,234 votes</div>'
        '<div class="candidate">Jane Smith - 5,678 votes</div>'
        "</div>"
        '<div class="precinct-selector">'
        "<button>Precinct 1</button>"
        "<button>Precinct 2</button>"
        "</div>"
        "</body></html>"
    )

    test_url = "https://example.gov/results"
    test_session_id = "integrity_test_001"
    emitted_events = []

    def mock_emit(payload: dict):
        emitted_events.append(payload)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    page = _FakePage(test_html, test_url)
    coordinator = _IsolatedCoordinator()
    original_cwd = Path.cwd()

    try:
        os.chdir(work_dir)
        with (
            patch.object(
                html_scanner,
                "_load_context_resources",
                return_value=({}, [], None),
            ),
            patch.object(
                html_scanner.ModelRegistry,
                "get_sentence_transformer",
                return_value=None,
            ),
            patch.object(
                html_scanner,
                "append_pattern_kb",
                return_value=None,
            ),
            patch.object(
                html_scanner,
                "append_feedback_log",
                return_value=None,
            ),
            patch.object(
                html_scanner,
                "save_context_cache_to_disk",
                return_value=None,
            ),
        ):
            result = html_scanner.scan_html_for_context(
                target_url=test_url,
                page=page,
                coordinator=coordinator,
                session_id=test_session_id,
                context_cache={},
                emit_func=mock_emit,
            )
    finally:
        os.chdir(original_cwd)

    assert "context_digest" in result, "context_digest missing from fresh-scan result"
    digest = result["context_digest"]
    assert digest["schema_version"] == "1.1"
    assert "model_signals" in digest
    assert "generated_at" in digest

    digest_dir = work_dir / "tools" / "debug_headless_output"
    digest_file = digest_dir / f"context_digest_{test_session_id}.json"
    assert digest_file.exists(), f"Digest file not found: {digest_file}"

    trend_file = digest_dir / "context_digest_trends.json"
    assert trend_file.exists(), f"Trend file not found: {trend_file}"

    trends = json.loads(trend_file.read_text(encoding="utf-8"))
    assert isinstance(trends, list)
    assert trends

    latest = trends[-1]
    assert latest.get("session_id") == test_session_id
    assert "confidence" in latest
    assert "unknown_ratio" in latest
    assert "segment_count" in latest

    context_digest_event = next(
        (event for event in emitted_events if event.get("type") == "context_digest"),
        None,
    )
    assert context_digest_event is not None
    assert context_digest_event["session_id"] == test_session_id

    integrity_signal_event = next(
        (event for event in emitted_events if event.get("type") == "integrity_signal"),
        None,
    )
    assert integrity_signal_event is not None
    assert integrity_signal_event["session_id"] == test_session_id

    signal = integrity_signal_event["signal"]
    assert signal["status"] in ["ok", "alert", "insufficient_data", "error"]

    if signal["status"] in {"ok", "alert"}:
        assert "baseline" in signal
        assert "recent" in signal
        assert "deltas" in signal
        assert "alerts" in signal

        for alert in signal["alerts"]:
            assert "type" in alert
            assert "severity" in alert
            assert "message" in alert
            assert alert["type"] in [
                "confidence_drop",
                "unknown_spike",
                "review_spike",
            ]
            assert alert["severity"] == "warning"

    print("[PASS] All integrity signal checks passed")
    print(f"[INFO] Emitted {len(emitted_events)} events:")
    for event in emitted_events:
        print(f"   - {event['type']}")


def test_integrity_signal_emission(tmp_path):
    _run_integrity_signal_emission(tmp_path)


if __name__ == "__main__":
    try:
        with tempfile.TemporaryDirectory(
            prefix="electionpulse_integrity_signal_"
        ) as temp_dir:
            _run_integrity_signal_emission(Path(temp_dir))
        sys.exit(0)
    except Exception as exc:
        print(f"[FAIL] Test failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
