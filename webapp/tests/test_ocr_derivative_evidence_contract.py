from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from webapp.parser.services.ocr_derivative_evidence import (
    observe_ocr_derivative,
    ocr_derivative_observer,
    ocr_derivative_source_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = REPO_ROOT / "webapp/parser/handlers/formats/pdf_handler.py"
BROWSER_PATH = REPO_ROOT / "webapp/parser/utils/browser_utils.py"

EXCLUDED = [
    REPO_ROOT / "tools/ocr_debug_artifacts.py",
    REPO_ROOT / "webapp/Smart_Elections_Parser_Webapp.py",
    REPO_ROOT / "webapp/parser/utils/shared_logic.py",
]


class BombSource:
    def __getattribute__(self, name):
        raise AssertionError(f"default-none path fingerprinted source unexpectedly: {name}")


def _capture():
    items = []

    def observer(payload):
        items.append(payload)

    return items, observer


def test_default_none_returns_exact_result_without_source_or_result_fingerprinting():
    result = object()
    assert observe_ocr_derivative(
        result,
        source_input=BombSource(),
        method="image_to_string",
        producer="unit",
    ) is result


def test_active_observer_emits_no_raw_ocr_text_or_raw_image_bytes():
    items, observer = _capture()
    secret = "SECRET OCR TEXT 123"
    with ocr_derivative_observer(observer):
        result = observe_ocr_derivative(
            secret,
            source_input=b"\x00\x01\x02SUPER_SECRET_PIXELS",
            method="image_to_string",
            producer="unit",
            context={"purpose_code": "unit", "raw_text": secret, "path": r"C:\secret"},
        )
    assert result is secret
    payload_text = json.dumps(items[0], sort_keys=True)
    assert secret not in payload_text
    assert "SUPER_SECRET_PIXELS" not in payload_text
    assert r"C:\secret" not in payload_text


def test_active_observer_computes_immediate_source_fingerprint():
    items, observer = _capture()
    raw = b"exact-source-bytes"
    with ocr_derivative_observer(observer):
        observe_ocr_derivative(
            "x",
            source_input=raw,
            method="image_to_string",
            producer="unit",
        )
    source = items[0]["source"]
    assert source["fingerprint_scheme"] == "OCR_INPUT_BYTES_V1"
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()
    assert source["byte_count"] == len(raw)


def test_root_source_identity_exact_when_supplied_and_unknown_when_missing():
    items, observer = _capture()
    identity = {"artifact_id": "abc", "sha256": "123"}
    with ocr_derivative_observer(observer):
        observe_ocr_derivative("x", source_input=b"a", method="image_to_string", producer="unit")
        with ocr_derivative_source_identity(identity):
            observe_ocr_derivative("y", source_input=b"b", method="image_to_string", producer="unit")
    assert items[0]["root_source_artifact_identity"] is None
    assert items[1]["root_source_artifact_identity"] is identity


def test_str_result_fingerprint_exact_utf8():
    items, observer = _capture()
    text = "precinct α\n"
    raw = text.encode("utf-8")
    with ocr_derivative_observer(observer):
        returned = observe_ocr_derivative(
            text,
            source_input=b"a",
            method="image_to_string",
            producer="unit",
        )
    assert returned is text
    derivative = items[0]["derivative"]
    assert derivative["fingerprint_scheme"] == "OCR_RESULT_UTF8_V1"
    assert derivative["sha256"] == hashlib.sha256(raw).hexdigest()
    assert derivative["byte_count"] == len(raw)


def test_dict_result_fingerprint_canonical_json():
    items, observer = _capture()
    value = {"b": [2, 1], "a": "x"}
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    with ocr_derivative_observer(observer):
        returned = observe_ocr_derivative(
            value,
            source_input=b"a",
            method="image_to_data",
            producer="unit",
        )
    assert returned is value
    derivative = items[0]["derivative"]
    assert derivative["fingerprint_scheme"] == "OCR_RESULT_CANONICAL_JSON_V1"
    assert derivative["sha256"] == hashlib.sha256(canonical).hexdigest()


def test_dataframe_result_fingerprint_canonical_csv_without_result_mutation():
    items, observer = _capture()
    df = pd.DataFrame([{"text": "A", "conf": 99}, {"text": "B", "conf": 98}])
    before = df.copy(deep=True)
    canonical = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with ocr_derivative_observer(observer):
        returned = observe_ocr_derivative(
            df,
            source_input=b"a",
            method="image_to_data",
            producer="unit",
        )
    assert returned is df
    pd.testing.assert_frame_equal(df, before)
    derivative = items[0]["derivative"]
    assert derivative["fingerprint_scheme"] == "OCR_RESULT_DATAFRAME_CSV_V1"
    assert derivative["sha256"] == hashlib.sha256(canonical).hexdigest()
    assert derivative["row_count"] == 2
    assert derivative["column_count"] == 2


def test_safe_context_allowlist_drops_paths_urls_sessions_and_unknown_keys():
    items, observer = _capture()
    with ocr_derivative_observer(observer):
        observe_ocr_derivative(
            "x",
            source_input=b"a",
            method="image_to_string",
            producer="unit",
            context={
                "purpose_code": "contest_probe",
                "page_index": 2,
                "session_id": "secret",
                "url": "https://secret.invalid",
                "path": r"C:\secret",
                "unknown": "drop-me",
            },
        )
    assert items[0]["context"] == {"page_index": 2, "purpose_code": "contest_probe"}


def test_callback_exception_propagates_when_explicitly_enabled():
    def explode(_payload):
        raise RuntimeError("observer failure")

    with ocr_derivative_observer(explode):
        with pytest.raises(RuntimeError, match="observer failure"):
            observe_ocr_derivative(
                "x",
                source_input=b"a",
                method="image_to_string",
                producer="unit",
            )


def test_pdf_mode_creates_no_files(tmp_path, monkeypatch):
    items, observer = _capture()
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.iterdir())
    with ocr_derivative_observer(observer):
        observe_ocr_derivative(
            "x",
            source_input=b"a",
            method="image_to_string",
            producer="pdf",
            context={"purpose_code": "pdf"},
        )
    assert sorted(tmp_path.iterdir()) == before
    assert items[0]["persistence"]["logical_persistence_mode"] == "EPHEMERAL_DERIVATIVE_OBSERVATION"


def test_existing_diagnostic_binding_hashes_existing_file_without_writing_it(tmp_path):
    path = tmp_path / "existing.txt"
    raw = b"already written diagnostic"
    path.write_bytes(raw)
    before = path.read_bytes()
    items, observer = _capture()
    with ocr_derivative_observer(observer):
        observe_ocr_derivative(
            "already written diagnostic",
            source_input=b"a",
            method="image_to_string",
            producer="save_diagnostics",
            persisted_path=path,
        )
    assert path.read_bytes() == before
    persistence = items[0]["persistence"]
    assert persistence["logical_persistence_mode"] == "EXISTING_DIAGNOSTIC_PERSISTENCE_BINDING"
    assert persistence["persisted_sha256"] == hashlib.sha256(raw).hexdigest()
    assert persistence["persisted_byte_count"] == len(raw)
    assert "path" not in persistence


def _count_calls(path: Path, leaf: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == leaf
    )


def _count_pytesseract_calls(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in {"image_to_string", "image_to_data"}
            and isinstance(func.value, ast.Name)
            and func.value.id == "pytesseract"
        ):
            count += 1
    return count


def test_all_eight_runtime_pytesseract_calls_are_instrumented_exactly_once():
    assert _count_pytesseract_calls(PDF_PATH) == 7
    assert _count_pytesseract_calls(BROWSER_PATH) == 1
    assert _count_calls(PDF_PATH, "observe_ocr_derivative") == 7
    assert _count_calls(BROWSER_PATH, "observe_ocr_derivative") == 1


def test_offline_tool_webapp_and_shared_logic_remain_out_of_scope():
    needle = "ocr_derivative_evidence"
    for path in EXCLUDED:
        assert needle not in path.read_text(encoding="utf-8")
