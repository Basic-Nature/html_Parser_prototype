from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from ...Context_Integration.Context_Library.constants import (
    CONTEST_KEYWORDS,
    CONTEST_TITLE_SKIP_PHRASES,
)
from ...utils.contest_selector import select_contest_auto_first
from ...utils.location_helpers import (
    attach_precinct_column,
    collect_location_headers,
)
from ...utils.logger_singleton import logger
from ...utils.output_utils import finalize_election_output
from ...utils.pivot import expand_single_rawjson_row
from ...utils.shared_logic import safe_get, safe_slug
from ...utils.table_builder import build_table_noninteractive
from ...utils.table_core import robust_table_extraction

_HANDLER_NAME = "xlsx_handler"

try:
    import pandas as _pd_module
except ImportError:  # pragma: no cover - handled at runtime during parsing
    _pd_module = None

pd = cast(Any, _pd_module)


def _build_contest_regex(keywords: List[str] | set[str] | tuple[str, ...] | None) -> re.Pattern[str]:
    parts: List[str] = []
    for phrase in (keywords or []):
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        tokens = re.split(r"\s+", phrase.strip().lower())
        normalized: List[str] = []
        for tok in tokens:
            esc = re.escape(tok)
            esc = esc.replace(r"\.", r"\.?")
            esc = esc.replace(r"\-", r"[-\s]?")
            normalized.append(esc)
        parts.append(r"(?:[\s\-_\/]*?)".join(normalized))
    return re.compile("|".join(parts), re.I) if parts else re.compile(r"(?!x)x", re.I)


_CONTEST_RX = _build_contest_regex(CONTEST_KEYWORDS)


def _dataframe_to_records(df: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
    assert pd is not None, "pandas must be available to process dataframes"
    headers = [str(h).strip() for h in df.columns]
    records: List[Dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for key, value in record.items():
            k = str(key).strip()
            if isinstance(value, float) and pd.notna(value):
                clean[k] = value
            else:
                clean[k] = "" if value is None or (isinstance(value, float) and pd.isna(value)) else value
        if any(str(v).strip() for v in clean.values()):
            records.append(clean)
    return headers, records


def parse_xlsx_election_results(
    xlsx_path: str,
    session_id: Optional[str] = None,
    coordinator: Any = None,
    sheet: str | int | None = None,
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    if pd is None:
        logger.error({
            "level": "ERROR",
            "type": "input",
            "message": f"[{_HANDLER_NAME}] pandas is required to parse Excel files. Install 'pandas'.",
            "session_id": session_id,
        })
        return [], [], "", {"error": "pandas not available"}

    sheet_name: str | int | None = sheet
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name if sheet_name is not None else 0, dtype=object)
    except Exception as exc:
        logger.error({
            "level": "ERROR",
            "type": "input",
            "message": f"[{_HANDLER_NAME}] Failed to read Excel file: {exc}",
            "session_id": session_id,
        })
        return [], [], "", {"error": f"Excel read error: {exc}"}

    if isinstance(df, dict):  # pandas returns dict when sheet_name=None and multiple sheets requested
        # Select the first sheet deterministically
        first_sheet = next(iter(df.keys()))
        df = df[first_sheet]
        sheet_name = first_sheet

    if df is None or df.empty:
        logger.error({
            "level": "ERROR",
            "type": "input",
            "message": f"[{_HANDLER_NAME}] Excel sheet is empty.",
            "session_id": session_id,
        })
        return [], [], "", {"error": "Empty Excel sheet"}

    headers, data = _dataframe_to_records(df)
    contest_column = None
    possible_contest_cols = [col for col in headers if _CONTEST_RX.search((col or "").lower())]
    if possible_contest_cols:
        possible_contest_cols.sort(key=lambda c: len(c or ""), reverse=True)
        contest_column = possible_contest_cols[0]

    contest_names: List[str] = []
    if contest_column:
        contest_names = sorted({(row.get(contest_column, "") or "").strip() for row in data if row.get(contest_column)})
        contest_names = [c for c in contest_names if c]
    if not contest_names:
        contest_names = [os.path.basename(xlsx_path).replace(".xlsx", "").replace(".xls", "")]

    fname = os.path.basename(xlsx_path).lower()
    state = "Unknown"
    county = "Unknown"
    for part in re.split(r"[_\-]", fname.replace(".xlsx", "").replace(".xls", "")):
        if "county" in part:
            county = part.replace("county", "").strip().title() + " County"
        if len(part) == 2 and part.isalpha():
            state = part.upper()

    selection_context = {
        "selector_data": {
            "contests": [{"title": name} for name in contest_names],
            "noisy_patterns": [s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())],
        },
        "input_file": os.path.basename(xlsx_path),
    }
    if len(contest_names) == 1:
        contest = contest_names[0]
    else:
        auto_pick = select_contest_auto_first(
            coordinator=coordinator,
            context=selection_context,
            session_id=session_id,
            allow_multiple=False,
            force_interactive=False,
        )
        if not auto_pick:
            logger.error({
                "level": "ERROR",
                "type": "input",
                "message": f"[{_HANDLER_NAME}] No contest selected.",
                "session_id": session_id,
            })
            return [], [], "", {"error": "No contest selected"}
        contest = safe_get(auto_pick[0], "title") or contest_names[0]

    if contest_column:
        data = [row for row in data if (row.get(contest_column, "") or "").strip() == contest]

    location_headers = collect_location_headers(headers)
    headers, data, precinct_attached = attach_precinct_column(
        headers,
        data,
        location_headers=location_headers,
    )
    location_diagnostics = {
        "detected_location_headers": location_headers,
        "precinct_attached": precinct_attached,
    }

    domain = safe_slug(os.path.basename(xlsx_path))
    m = re.search(r"(19|20)\d{2}", fname)
    year = int(m.group(0)) if m else None
    context = {
        "contest": contest,
        "state": state,
        "county": county,
        "year": year,
        "session_id": session_id,
        "handler": _HANDLER_NAME,
        "source_slug": domain,
        "sheet_name": sheet_name,
        "location_headers": location_headers,
        "precinct_attached": precinct_attached,
        "location_diagnostics": location_diagnostics,
    }
    headers, data = expand_single_rawjson_row(headers, data, context=context)

    headers_final, data_final, _entity_info = build_table_noninteractive(
        domain=domain,
        headers=headers,
        data=data,
        coordinator=coordinator,
        context=context,
        pivot_to_wide=True,
        debug=False,
    )

    result = finalize_election_output(
        headers=headers_final,
        data=data_final,
        coordinator=coordinator,
        contest=contest,
        state=state,
        county=county,
        context={
            "handler": _HANDLER_NAME,
            "input_file": os.path.basename(xlsx_path),
            "session_id": session_id,
            "race": contest,
            "sheet_name": sheet_name,
            "location_headers": location_headers,
            "precinct_attached": precinct_attached,
            "location_diagnostics": location_diagnostics,
        },
        enable_user_feedback=False,
        session_id=session_id,
    )

    metadata = {
        "race": contest,
        "input_file": os.path.basename(xlsx_path),
        "output_file": os.path.basename(result.get("csv_path", "")),
        "headers": headers_final,
        "row_count": len(data_final),
        "handler": _HANDLER_NAME,
        "state": state,
        "county": county,
        "year": year,
        "sheet_name": sheet_name,
        "csv_path": result.get("csv_path"),
        "metadata_path": result.get("metadata_path"),
        "location_headers_detected": location_headers,
        "precinct_attached": precinct_attached,
        "location_diagnostics": location_diagnostics,
    }

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[{_HANDLER_NAME}] Wrote {len(data_final)} rows to: {result.get('csv_path')}",
        "session_id": session_id,
    })
    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[{_HANDLER_NAME}] Metadata written to: {result.get('metadata_path')}",
        "session_id": session_id,
    })

    return headers_final, data_final, contest, metadata


def parse(
    page: Any | None = None,
    coordinator: Any | None = None,
    html_context: Dict[str, Any] | None = None,
    manual_file: str | None = None,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[List[str] | None, List[Dict[str, Any]] | None, str | None, Dict[str, Any]]:
    html_context = html_context or {}
    provided_tables = html_context.get("provided_tables")
    if isinstance(provided_tables, list) and provided_tables:
        ctx = dict(html_context)
        ctx.update({
            "session_id": session_id,
            "coordinator": coordinator,
        })
        merged_headers, merged_rows = robust_table_extraction(page=None, extraction_context=ctx)

        contest = html_context.get("contest") or "Provided Tables"
        state = html_context.get("state") or "Unknown"
        county = html_context.get("county") or "Unknown"
        year = html_context.get("year")
        domain = html_context.get("source_slug") or safe_slug(contest)

        headers_final, data_final, _entity_info = build_table_noninteractive(
            domain=domain,
            headers=merged_headers,
            data=merged_rows,
            coordinator=coordinator,
            context={
                **ctx,
                "contest": contest,
                "state": state,
                "county": county,
                "year": year,
                "handler": _HANDLER_NAME,
            },
            pivot_to_wide=not bool(html_context.get("skip_pivot")),
            debug=False,
        )

        finalized = finalize_election_output(
            headers=headers_final,
            data=data_final,
            coordinator=coordinator,
            contest=contest,
            state=state,
            county=county,
            context={
                "handler": _HANDLER_NAME,
                "session_id": session_id,
                "race": contest,
                "provided_tables": True,
                "skip_pivot": bool(html_context.get("skip_pivot")),
            },
            enable_user_feedback=False,
            session_id=session_id,
        )

        metadata = {
            "race": contest,
            "input_file": html_context.get("input_file") or "<provided>",
            "output_file": os.path.basename(finalized.get("csv_path", "")),
            "headers": headers_final,
            "row_count": len(data_final),
            "handler": _HANDLER_NAME,
            "state": state,
            "county": county,
            "year": year,
            "csv_path": finalized.get("csv_path"),
            "metadata_path": finalized.get("metadata_path"),
        }
        return headers_final, data_final, contest, metadata

    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": f"[{_HANDLER_NAME}] Parsing intentionally skipped via context flag.",
            "session_id": session_id,
        })
        return None, None, None, {"skipped": True}

    if not manual_file or not os.path.isfile(manual_file):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": f"[{_HANDLER_NAME}] No Excel file provided to parse().",
            "session_id": session_id,
        })
        return None, None, None, {"skipped": True}

    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": f"[{_HANDLER_NAME}] Using Excel file: {manual_file}",
        "session_id": session_id,
    })

    sheet_hint = html_context.get("excel_sheet") or html_context.get("sheet_name")
    sheet: str | int | None
    if isinstance(sheet_hint, int):
        sheet = sheet_hint
    elif isinstance(sheet_hint, str) and sheet_hint.isdigit():
        sheet = int(sheet_hint)
    else:
        sheet = sheet_hint if isinstance(sheet_hint, str) else None

    result = parse_xlsx_election_results(
        manual_file,
        session_id=session_id,
        coordinator=coordinator,
        sheet=sheet,
    )
    result_any = cast(Any, result)
    if not (isinstance(result_any, tuple) and len(result_any) == 4):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": f"[{_HANDLER_NAME}] Invalid result from parse_xlsx_election_results (expected 4-tuple).",
            "session_id": session_id,
            "got_type": type(result).__name__,
        })
        return None, None, None, {"error": "Invalid parse result"}
    return cast(Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]], result_any)
