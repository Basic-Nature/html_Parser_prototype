from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from webapp.parser.contracts.artifact_identity import ArtifactIdentityHandoff
from webapp.parser.navigator import dom_snapshot
from webapp.parser.services.dom_browser_evidence import (
    DOM_BROWSER_METADATA_CONTRACT,
    DOM_BROWSER_OBSERVATION_CONTRACT,
    DOM_BROWSER_SERIALIZATION_SEMANTICS,
    finalize_dom_browser_evidence,
)


class _Page:
    def __init__(self, html: str) -> None:
        self.html = html
        self.waited: list[tuple[str, int, str]] = []

    def content(self) -> str:
        return self.html

    def wait_for_selector(
        self,
        selector: str,
        *,
        timeout: int,
        state: str,
    ) -> None:
        self.waited.append((selector, timeout, state))


def test_exact_utf8_original_is_sealed_with_safe_sidecars(tmp_path: Path) -> None:
    html = "<html><body>SECRET_DOM_TEXT_é🙂</body></html>"
    raw = html.encode("utf-8")
    path = tmp_path / "SECRET_PATH_capture.html"

    result = finalize_dom_browser_evidence(
        html,
        path,
        capture_role="dom_snapshot",
    )

    assert path.read_bytes() == raw
    assert result["contract"] == DOM_BROWSER_METADATA_CONTRACT
    assert result["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["artifact_identity"] is None
    assert result["serialization_semantics"] == DOM_BROWSER_SERIALIZATION_SEMANTICS
    assert result["sealed_original_persisted"] is True
    assert result["raw_original_mutated"] is False
    assert result["raw_dom_payload_included"] is False
    assert result["filesystem_path_included"] is False
    assert result["dom_content_summary_included"] is False
    assert result["automatic_timestamp"] is False
    assert result["observation_emitted"] is False

    metadata_path = path.with_suffix(".html.metadata.json")
    seal_path = path.with_suffix(".html.seal.json")
    metadata_raw = metadata_path.read_bytes()
    seal_raw = seal_path.read_bytes()
    metadata = json.loads(metadata_raw.decode("utf-8"))
    seal = json.loads(seal_raw.decode("utf-8"))

    assert metadata["artifact_sha256"] == result["artifact_sha256"]
    assert seal["artifact_sha256"] == result["artifact_sha256"]
    assert seal["metadata_derivative_sha256"] == hashlib.sha256(metadata_raw).hexdigest()
    assert b"SECRET_DOM_TEXT" not in metadata_raw
    assert b"SECRET_DOM_TEXT" not in seal_raw
    assert b"SECRET_PATH" not in metadata_raw
    assert b"SECRET_PATH" not in seal_raw


def test_explicit_callback_receives_noncanonical_summary_only(tmp_path: Path) -> None:
    html = "<html><body>PRIVATE_DOM_CONTENT</body></html>"
    path = tmp_path / "SECRET_PATH_capture.html"
    captured: list[dict] = []

    result = finalize_dom_browser_evidence(
        html,
        path,
        capture_role="dom_snapshot",
        observation_emit_func=captured.append,
    )

    assert result["observation_emitted"] is True
    assert len(captured) == 1
    payload = captured[0]
    assert payload["contract"] == DOM_BROWSER_OBSERVATION_CONTRACT
    assert payload["authority"] == "NONCANONICAL_OBSERVATION"
    assert payload["canonical"] is False
    assert payload["artifact_identity"] is None
    assert payload["artifact_sha256"] == hashlib.sha256(html.encode("utf-8")).hexdigest()
    assert payload["raw_dom_bytes_included"] is False
    assert payload["raw_dom_text_included"] is False
    assert payload["raw_dom_path_included"] is False
    assert payload["dom_content_summary_included"] is False
    assert payload["automatic_timestamp"] is False
    encoded = json.dumps(payload, sort_keys=True)
    assert "PRIVATE_DOM_CONTENT" not in encoded
    assert "SECRET_PATH" not in encoded
    assert str(path) not in encoded


def test_dom_specific_identity_must_match_exact_utf8_bytes(tmp_path: Path) -> None:
    html = "<html><body>identity</body></html>"
    exact = hashlib.sha256(html.encode("utf-8")).hexdigest()

    ok_path = tmp_path / "exact.html"
    result = finalize_dom_browser_evidence(
        html,
        ok_path,
        capture_role="dom_snapshot",
        artifact_identity=ArtifactIdentityHandoff(document_sha256=exact),
    )
    assert result["artifact_identity"]["document_sha256"] == exact

    bad_path = tmp_path / "bad.html"
    with pytest.raises(
        ValueError,
        match="must match exact immutable DOM serialization bytes",
    ):
        finalize_dom_browser_evidence(
            html,
            bad_path,
            capture_role="dom_snapshot",
            artifact_identity=ArtifactIdentityHandoff(document_sha256="0" * 64),
        )
    assert not bad_path.exists()
    assert not bad_path.with_suffix(".html.metadata.json").exists()
    assert not bad_path.with_suffix(".html.seal.json").exists()


def test_exclusive_create_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "capture.html"
    original = b"DO_NOT_OVERWRITE"
    path.write_bytes(original)

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        finalize_dom_browser_evidence(
            "<html>new</html>",
            path,
            capture_role="dom_snapshot",
        )

    assert path.read_bytes() == original
    assert not path.with_suffix(".html.metadata.json").exists()
    assert not path.with_suffix(".html.seal.json").exists()


def test_capture_dom_snapshot_is_dormant_without_path_and_returns_same_string(
    tmp_path: Path,
) -> None:
    html = "<html><body>same-string-é🙂</body></html>"
    page = _Page(html)

    returned = dom_snapshot.capture_dom_snapshot(page)
    assert returned == html

    path = tmp_path / "snapshot.html"
    captured: list[dict] = []
    returned_with_evidence = dom_snapshot.capture_dom_snapshot(
        page,
        dom_evidence_path=path,
        dom_observation_emit_func=captured.append,
    )
    assert returned_with_evidence == html
    assert path.read_bytes() == html.encode("utf-8")
    assert len(captured) == 1

    with pytest.raises(
        ValueError,
        match="requires explicit dom_evidence_path",
    ):
        dom_snapshot.capture_dom_snapshot(
            page,
            dom_observation_emit_func=captured.append,
        )


def test_snapshot_pipeline_table_extraction_receives_exact_captured_string(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    html = "<html><table><tr><td>EXACT_DOM</td></tr></table></html>"
    page = _Page(html)
    received: list[str] = []

    def _extract(
        html_content: str,
        context=None,
        session_id=None,
    ):
        received.append(html_content)
        return ["Precinct"], [{"Precinct": "P1"}]

    monkeypatch.setattr(dom_snapshot, "extract_tables_from_snapshot", _extract)

    path = tmp_path / "snapshot.html"
    headers, rows, contest, metadata = dom_snapshot.snapshot_mode_pipeline(
        page,
        context={"contest": "Contest"},
        session_id="session",
        dom_evidence_path=path,
    )

    assert received == [html]
    assert path.read_bytes() == html.encode("utf-8")
    assert headers == ["Precinct"]
    assert rows == [{"Precinct": "P1"}]
    assert contest == "Contest"
    assert metadata["content_size"] == len(html)


def test_adjacent_dom_seams_and_html_parser_identity_remain_unwired() -> None:
    html_parser = Path("webapp/parser/html_election_parser.py").read_text(
        encoding="utf-8"
    )
    assert "dom_evidence_path" not in html_parser
    assert "dom_artifact_identity" not in html_parser
    assert "dom_observation_emit_func" not in html_parser
    assert "dom_browser_evidence" not in html_parser

    for rel in (
        "webapp/parser/Context_Integration/context_coordinator.py",
        "webapp/parser/navigator/navigation_runner.py",
        "webapp/parser/utils/browser_utils.py",
        "webapp/parser/utils/html_scanner.py",
        "webapp/parser/utils/retry_utils.py",
        "webapp/parser/utils/url_glimpse.py",
    ):
        text = Path(rel).read_text(encoding="utf-8")
        assert "dom_browser_evidence" not in text
        assert "dom_evidence_path" not in text


def test_dom_service_remains_separate_from_adjacent_contracts() -> None:
    service = Path("webapp/parser/services/dom_browser_evidence.py").read_text(
        encoding="utf-8"
    )
    assert "TablePipelineResult" not in service
    assert "finalize_election_output" not in service
    assert "parser_observation_bundle_v1" not in service
    assert "network_capture_evidence" not in service
    assert "screenshot_image_evidence" not in service
    assert "origin HTTP response body" in service
