from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from webapp.parser.contracts.artifact_identity import ArtifactIdentityHandoff
from webapp.parser.services.screenshot_image_evidence import (
    SCREENSHOT_IMAGE_METADATA_CONTRACT,
    SCREENSHOT_IMAGE_OBSERVATION_CONTRACT,
    finalize_screenshot_image_evidence,
)


def _png(width: int = 17, height: int = 9) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )


def test_finalize_preserves_raw_and_writes_safe_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "SECRET_PATH_capture.png"
    raw = _png()
    path.write_bytes(raw)

    result = finalize_screenshot_image_evidence(
        path,
        capture_role="runtime_diagnostic",
    )

    assert path.read_bytes() == raw
    assert result["contract"] == SCREENSHOT_IMAGE_METADATA_CONTRACT
    assert result["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["artifact_identity"] is None
    assert result["pixel_width"] == 17
    assert result["pixel_height"] == 9
    assert result["pixel_payload_included"] is False
    assert result["filesystem_path_included"] is False
    assert result["automatic_timestamp"] is False
    assert result["observation_emitted"] is False

    metadata_path = path.with_suffix(".png.metadata.json")
    seal_path = path.with_suffix(".png.seal.json")
    metadata_raw = metadata_path.read_bytes()
    seal_raw = seal_path.read_bytes()
    metadata = json.loads(metadata_raw.decode("utf-8"))
    seal = json.loads(seal_raw.decode("utf-8"))

    assert metadata["artifact_sha256"] == result["artifact_sha256"]
    assert seal["artifact_sha256"] == result["artifact_sha256"]
    assert seal["metadata_derivative_sha256"] == hashlib.sha256(metadata_raw).hexdigest()
    assert b"SECRET_PATH" not in metadata_raw
    assert b"SECRET_PATH" not in seal_raw
    assert raw not in metadata_raw
    assert raw not in seal_raw


def test_explicit_callback_receives_noncanonical_summary_only(tmp_path: Path) -> None:
    path = tmp_path / "capture.png"
    raw = _png(21, 11)
    path.write_bytes(raw)
    captured: list[dict] = []

    result = finalize_screenshot_image_evidence(
        path,
        capture_role="url_glimpse",
        observation_emit_func=captured.append,
    )

    assert result["observation_emitted"] is True
    assert len(captured) == 1
    payload = captured[0]
    assert payload["contract"] == SCREENSHOT_IMAGE_OBSERVATION_CONTRACT
    assert payload["authority"] == "NONCANONICAL_OBSERVATION"
    assert payload["canonical"] is False
    assert payload["artifact_identity"] is None
    assert payload["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert payload["raw_screenshot_bytes_included"] is False
    assert payload["raw_screenshot_path_included"] is False
    assert payload["ocr_text_included"] is False
    assert payload["visual_content_summary_included"] is False
    assert payload["automatic_timestamp"] is False
    assert str(path) not in json.dumps(payload, sort_keys=True)


def test_supplied_identity_must_match_exact_png_bytes(tmp_path: Path) -> None:
    path = tmp_path / "capture.png"
    raw = _png()
    path.write_bytes(raw)
    exact = hashlib.sha256(raw).hexdigest()

    result = finalize_screenshot_image_evidence(
        path,
        capture_role="runtime_diagnostic",
        artifact_identity=ArtifactIdentityHandoff(document_sha256=exact),
    )
    assert result["artifact_identity"]["document_sha256"] == exact

    with pytest.raises(ValueError, match="must match exact immutable screenshot bytes"):
        finalize_screenshot_image_evidence(
            path,
            capture_role="runtime_diagnostic",
            artifact_identity=ArtifactIdentityHandoff(document_sha256="0" * 64),
        )


def test_invalid_png_is_rejected_without_mutating_original(tmp_path: Path) -> None:
    path = tmp_path / "capture.png"
    raw = b"not-a-png"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="must be PNG bytes"):
        finalize_screenshot_image_evidence(
            path,
            capture_role="runtime_diagnostic",
        )
    assert path.read_bytes() == raw
    assert not path.with_suffix(".png.metadata.json").exists()
    assert not path.with_suffix(".png.seal.json").exists()


def test_runtime_producers_finalize_only_after_screenshot_write() -> None:
    browser = Path("webapp/parser/utils/browser_utils.py").read_text(encoding="utf-8")
    glimpse = Path("webapp/parser/utils/url_glimpse.py").read_text(encoding="utf-8")

    browser_shot = browser.index("page.screenshot(path=png_path, full_page=True)")
    browser_finalize = browser.index("finalize_screenshot_image_evidence(", browser_shot)
    assert browser_shot < browser_finalize

    glimpse_shot = glimpse.index(
        "page.screenshot(path=str(screenshot_path), full_page=True)"
    )
    glimpse_finalize = glimpse.index(
        "finalize_screenshot_image_evidence(", glimpse_shot
    )
    assert glimpse_shot < glimpse_finalize


def test_no_tooling_producer_or_adjacent_contract_wiring() -> None:
    service = Path("webapp/parser/services/screenshot_image_evidence.py").read_text(
        encoding="utf-8"
    )
    assert "TablePipelineResult" not in service
    assert "finalize_election_output" not in service
    assert "parser_observation_callback" not in service
    assert "network_capture_evidence" not in service

    for rel in (
        "tools/capture_har_smoke.py",
        "tools/capture_har_smoke_v2.py",
        "tools/capture_screenshots.py",
        "tools/check_layout.py",
        "tools/headless_check.py",
        "tools/headless_sequence.py",
        "tools/smoke_headless_debug.py",
        "tools/ui_prompt_smoke_puppeteer.js",
        "tools/ui_robust_check.py",
    ):
        assert "screenshot_image_evidence" not in Path(rel).read_text(
            encoding="utf-8"
        )
