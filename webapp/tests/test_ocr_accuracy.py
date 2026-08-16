import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webapp.parser.handlers.formats.pdf_handler import adaptive_ocr_pipeline


def normalize_text(text: str) -> str:
    return "\n".join(
        line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines() if line.strip()
    )


def similarity_ratio(expected: str, actual: str) -> float:
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    return SequenceMatcher(None, expected_norm, actual_norm).ratio()


@pytest.mark.integration
def test_ocr_accuracy_against_reference():
    pdf_path = os.environ.get("OCR_TEST_PDF_PATH")
    if not pdf_path:
        pytest.skip("OCR_TEST_PDF_PATH not set")

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        pytest.skip(f"OCR test PDF not found: {pdf_file}")

    expected_path = os.environ.get("OCR_EXPECTED_TEXT_PATH")
    expected_text = None
    if expected_path:
        expected_file = Path(expected_path)
        if expected_file.exists():
            expected_text = expected_file.read_text(encoding="utf-8")
        else:
            pytest.skip(f"OCR expected text file not found: {expected_file}")

    text, avg_conf, runs_summary, best_params = adaptive_ocr_pipeline(
        str(pdf_file),
        session_id="pytest_ocr_accuracy",
        max_runs=10,
        max_seconds=90,
        stream_time_budget=90,
    )

    assert text is not None
    assert text.strip() != "", "OCR extraction should produce text"
    assert avg_conf >= 0.0
    assert isinstance(runs_summary, list)

    if expected_text is not None:
        ratio = similarity_ratio(expected_text, text)
        assert ratio > 0.1, f"OCR similarity ratio is unexpectedly low: {ratio:.3f}"
        print(f"OCR similarity ratio: {ratio:.3f}")
