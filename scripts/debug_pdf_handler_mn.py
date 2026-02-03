from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator  # noqa: E402
from webapp.parser.handlers.formats import pdf_handler  # noqa: E402

SESSION_ID = "mn_pdf_diagnostic"


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


def _build_context_stub(
    pdf_path: Path,
    contest: str | None,
    headers: list[str] | None,
    rows: list[dict[str, Any]] | None,
    metadata: dict | None,
) -> dict:
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
        "session_id": SESSION_ID,
        "state": metadata.get("state"),
        "county": metadata.get("county"),
        "contests": [
            {
                "title": contest or "",
                "state": metadata.get("state"),
                "county": metadata.get("county"),
                "type_": metadata.get("contest_type"),
                "year": metadata.get("year"),
            }
        ],
        "tables": [
            {
                "headers": headers,
                "rows": rows,
                "table_text": table_text,
                "segment_hash": "mn_debug_table",
            }
        ],
        "line_records": metadata.get("line_records"),
        "pdf_context": {"path": str(pdf_path)},
        "metadata": metadata,
    }


def main() -> None:
    pdf_path = ROOT / "uploads" / "2016generalelectionsMN.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF at {pdf_path}")

    print(f"[debug] Running pdf_handler.parse for {pdf_path}")
    coordinator = ContextCoordinator(use_library=True, enable_ml=True, alert_monitor=False, debug=False)
    html_context = {
        "source": "pdf",
        "format": "pdf",
        "manual_file": str(pdf_path),
        "session_id": SESSION_ID,
    }
    headers, rows, contest, metadata = pdf_handler.parse(
        manual_file=str(pdf_path),
        coordinator=coordinator,
        html_context=html_context,
        session_id=SESSION_ID,
    )

    summary = _summarize(headers, rows)
    diag_payload = {
        "contest": contest,
        "summary": summary,
        "metadata_keys": sorted(metadata.keys()) if isinstance(metadata, dict) else [],
        "metadata": metadata,
    }
    metadata_insights = {
        "route_summary": metadata.get("route_summary") if isinstance(metadata, dict) else None,
        "enrichment_plan": metadata.get("enrichment_plan") if isinstance(metadata, dict) else None,
        "dense_line_normalization": metadata.get("dense_line_normalization") if isinstance(metadata, dict) else None,
        "columnar_reconstruction": metadata.get("columnar_reconstruction") if isinstance(metadata, dict) else None,
        "reconstruction_debug_events": metadata.get("reconstruction_debug_events") if isinstance(metadata, dict) else None,
        "columnar_reconstruction_attempts": metadata.get("columnar_reconstruction_attempts") if isinstance(metadata, dict) else None,
        "columnar_reconstruction_failure": metadata.get("columnar_reconstruction_failure") if isinstance(metadata, dict) else None,
    }

    context_stub = _build_context_stub(pdf_path, contest, headers, rows, metadata if isinstance(metadata, dict) else {})
    try:
        organized = coordinator.organize_and_enrich(context_stub, suppress_dom_errors=True)
        plan_meta = organized.get("metadata", {}) if isinstance(organized, dict) else {}
        if isinstance(metadata, dict):
            metadata.setdefault("enrichment_plan", plan_meta.get("enrichment_plan"))
            metadata.setdefault("route_summary", plan_meta.get("route_summary"))
        metadata_insights["route_summary"] = metadata.get("route_summary")
        metadata_insights["enrichment_plan"] = metadata.get("enrichment_plan")
    except Exception as exc:
        print(f"[warn] Failed to organize context for diagnostics: {exc}")

    diag_payload["metadata_insights"] = metadata_insights

    diagnostics_dir = ROOT / "output" / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diagnostics_dir / "mn_pdf_handler_result.json"
    with diag_path.open("w", encoding="utf-8") as handle:
        json.dump(diag_payload, handle, indent=2, default=str)

    print("[debug] Parse finished.")
    preview = {k: diag_payload[k] for k in ("contest", "summary")}
    preview["metadata_insights"] = metadata_insights
    print(json.dumps(preview, indent=2, default=str))
    print(f"[debug] Detailed metadata saved to {diag_path}")


if __name__ == "__main__":
    main()
