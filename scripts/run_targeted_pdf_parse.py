import importlib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PDF_PATH = Path("c:/Users/edu-loaner/html_Parser_prototype/uploads/2016 General Election Official Results.PDF")


def run_for(contest_title: str, session_id: str) -> None:
    module = importlib.reload(importlib.import_module("webapp.parser.handlers.formats.pdf_handler"))
    original_selector = module.select_contest_auto_first
    try:
        module.select_contest_auto_first = lambda **kwargs: [{"title": contest_title}]
        headers, rows, contest, metadata = module.parse_pdf_election_results(
            str(PDF_PATH),
            session_id=session_id,
        )
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
        print()
    finally:
        module.select_contest_auto_first = original_selector


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(f"Missing PDF at {PDF_PATH}")
    run_for("US Senator", "cli_us_senator")
    run_for("3rd District", "cli_third_district")


if __name__ == "__main__":
    main()
