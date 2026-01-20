import importlib
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore[import]

    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DEFAULT_PDF = BASE_DIR / "uploads" / "2016 General Election Official Results.PDF"
PDF_PATH_DEFAULT = Path(os.environ.get("PDF_PATH", DEFAULT_PDF)).expanduser()

# Comma-separated contest list env override, e.g. CONTESTS="Contest A,Contest B"
ENV_CONTESTS = [c.strip() for c in os.environ.get("CONTESTS", "").split(",") if c.strip()]


def run_for(pdf_path: Path, contest_title: str, session_id: str) -> None:
    module = importlib.reload(importlib.import_module("webapp.parser.handlers.formats.pdf_handler"))
    original_selector = module.select_contest_auto_first
    try:
        module.select_contest_auto_first = lambda **kwargs: [{"title": contest_title}]
        try:
            headers, rows, contest, metadata = module.parse_pdf_election_results(
                str(pdf_path),
                session_id=session_id,
            )
        except Exception as exc:
            import traceback
            print(f"[ERROR] parse_pdf_election_results failed for contest='{contest_title}' session_id='{session_id}': {exc}")
            print(traceback.format_exc())
            print()
            return
        columnar = metadata.get("columnar_reconstruction", {})
        csv_path = metadata.get("csv_path")
        metadata_path = metadata.get("metadata_path")

        def _rel(path: str | None) -> str | None:
            if not path:
                return None
            candidate = Path(path).resolve()
            try:
                return str(candidate.relative_to(BASE_DIR))
            except ValueError:
                return str(candidate)

        csv_rel = _rel(csv_path)
        meta_rel = _rel(metadata_path)
        print(f"== {contest} ==")
        print("header_count", len(headers))
        print("row_count", len(rows))
        print("csv_path", metadata.get("csv_path"))
        if csv_rel:
            print(f"[output] {csv_rel}")
        print("smart_standard", columnar.get("smart_standard_applied"))
        print("candidate_columns", len(columnar.get("candidate_columns", [])))
        if rows:
            print("sample_row", rows[0])
        if meta_rel:
            print(f"[output] {meta_rel}")
        # Surface OCR text paths when available
        for key in ("ocr_raw_text_path", "ocr_clean_text_path"):
            if metadata.get(key):
                print(f"{key}", _rel(metadata.get(key)))
        print()
    finally:
        module.select_contest_auto_first = original_selector


def main() -> None:
    pdf_path = PDF_PATH_DEFAULT
    # Contest sources: CLI args (after PDF) > CONTESTS env > defaults
    args = sys.argv[1:]
    contests: list[str] = []

    if args:
        first = Path(args[0]).expanduser()
        if first.exists():
            pdf_path = first
            contests = args[1:]
        else:
            contests = args

    if not pdf_path.exists():
        raise SystemExit(f"Missing PDF at {pdf_path}")

    if not contests:
        contests = ENV_CONTESTS or ["US Senator", "3rd District"]

    print(f"[HARNESS] Starting run for PDF: {pdf_path}")
    print(f"[HARNESS] Contests: {', '.join(contests)}")
    for idx, contest in enumerate(contests):
        sid = f"cli_{idx}_{contest.lower().replace(' ', '_')[:24]}"
        run_for(pdf_path, contest, sid)


if __name__ == "__main__":
    main()
