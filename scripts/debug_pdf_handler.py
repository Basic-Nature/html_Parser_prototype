from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator  # noqa: E402
from webapp.parser.handlers.formats import pdf_handler  # noqa: E402


def _summarize(headers: list[str] | None, rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    headers = headers or []
    rows = rows or []
    sample_rows = [{k: v for k, v in row.items()} for row in rows[:3]]
    return {
        "header_count": len(headers),
        "headers_preview": headers[:10],
        "row_count": len(rows),
        "rows_preview": sample_rows,
    }


def _build_context_stub(pdf_path: Path, contest: str | None, headers: list[str] | None, rows: list[dict[str, Any]] | None, metadata: dict | None) -> dict:
    headers = headers or []
    rows = rows or []
    metadata = metadata or {}
    sample_rows = rows[:5]
    table_text_lines: list[str] = []
    if headers:
        table_text_lines.append(" | ".join(headers))
        for row in sample_rows:
            table_text_lines.append(" | ".join(str(row.get(col, "")) for col in headers))
    table_text = "\n".join(table_text_lines)
    return {
        "source": "pdf",
        "format": "pdf",
        "session_id": metadata.get("session_id"),
        "state": metadata.get("state"),
        "county": metadata.get("county"),
        "contests": [
            {"title": contest or "", "state": metadata.get("state"), "county": metadata.get("county"), "type_": metadata.get("contest_type"), "year": metadata.get("year")}
        ],
        "tables": [
            {"headers": headers, "rows": rows, "table_text": table_text, "segment_hash": "debug_table"}
        ],
        "line_records": metadata.get("line_records"),
        "pdf_context": {"path": str(pdf_path)},
        "metadata": metadata,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Universal PDF handler debug runner")
    parser.add_argument("--file", "-f", required=True, help="Path to PDF file to debug")
    parser.add_argument("--session-id", "-s", default="debug_pdf", help="Session id")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR path")
    parser.add_argument("--disable-camelot", action="store_true", help="Disable Camelot table extraction")
    parser.add_argument("--no-coordinator", action="store_true", help="Skip ContextCoordinator enrichment")
    args = parser.parse_args(argv)

    pdf_path = Path(args.file)
    if not pdf_path.exists():
        print(f"Missing PDF: {pdf_path}")
        return 2

    print(f"[debug] Running pdf_handler.parse for {pdf_path}")
    coordinator = None
    if not args.no_coordinator:
        coordinator = ContextCoordinator(use_library=True, enable_ml=True, alert_monitor=False, debug=False)

    html_context = {"source": "pdf", "format": "pdf", "manual_file": str(pdf_path), "session_id": args.session_id}

    # Apply runtime flags
    if args.disable_camelot:
        try:
            # Disable Camelot at both handler and utility levels to avoid any camelot I/O or pdfminer hangs
            pdf_handler._CAMELOT_AVAILABLE = False
            try:
                from webapp.parser.utils import camelot_utils
                camelot_utils._CAMELOT_AVAILABLE = False
            except Exception:
                pass
            print("[debug] Camelot disabled for this run.")
        except Exception:
            pass

    if args.force_ocr:
        os.environ.setdefault("ENABLE_OCR_FORCE", "true")
        print("[debug] OCR forcing enabled for this run.")

    try:
        headers, rows, contest, metadata = pdf_handler.parse(
            manual_file=str(pdf_path),
            coordinator=coordinator,
            html_context=html_context,
            session_id=args.session_id,
        )
    except Exception as exc:
        print(f"[error] parse failed: {exc}")
        return 1

    summary = _summarize(headers, rows)
    diag_payload = {"contest": contest, "summary": summary, "metadata": metadata}

    # Try enrichment if coordinator present
    if coordinator and isinstance(metadata, dict):
        try:
            context_stub = _build_context_stub(pdf_path, contest, headers, rows, metadata)
            organized = coordinator.organize_and_enrich(context_stub, suppress_dom_errors=True)
            diag_payload["enrichment"] = organized.get("metadata") if isinstance(organized, dict) else None
        except Exception as exc:
            diag_payload["enrichment_error"] = str(exc)

    out_dir = ROOT / "output" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (pdf_path.stem + "_debug.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(diag_payload, fh, indent=2, default=str)

    print(json.dumps({"contest": contest, "summary": summary}, indent=2, default=str))
    print(f"[debug] Detailed diagnostics written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
