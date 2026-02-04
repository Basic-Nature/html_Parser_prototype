from __future__ import annotations

import csv
import os
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from ...config import ENABLE_PARALLEL  # type: ignore[attr-defined]
from ...Context_Integration.Context_Library.constants import (
    CONTEST_TITLE_SKIP_PHRASES,
)
from ...utils.contest_detection import (
    CONTEST_PATTERN as _CONTEST_RX,
)
from ...utils.contest_detection import (
    detect_contest_titles_from_text,
    gather_lines_for_contest_detection,
)
from ...utils.contest_selector import select_contest_auto_first
from ...utils.location_helpers import (
    attach_precinct_column,
    collect_location_headers,
)
from ...utils.logger_singleton import logger
from ...utils.output_utils import finalize_election_output
from ...utils.pivot import expand_single_rawjson_row
from ...utils.shared_logic import (
    derive_candidate_party_metadata,
    derive_state_county_from_table,
    safe_get,
    safe_slug,
)
from ...utils.table_builder import build_table_noninteractive
from ...utils.table_core import robust_table_extraction

_HANDLER_NAME = "txt_handler"
# Allow flexible delimiters commonly used in exported text tables
_DELIMITER_CANDIDATES = ",\t;|:"


# Use shared contest regex + detection helpers


def _read_delimited_file(txt_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Read a delimited text file with dialect sniffing and conservative cleanup."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(txt_path, mode="r", encoding=encoding, newline="") as handle:
                sample = handle.read(4096)
                handle.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample or "", delimiters=_DELIMITER_CANDIDATES)
                except Exception:
                    dialect = csv.excel
                reader = csv.DictReader(handle, dialect=dialect)
                headers = [str(h).strip() for h in (reader.fieldnames or []) if h]
                rows: List[Dict[str, Any]] = []
                for raw in reader:
                    if not raw:
                        continue
                    clean: Dict[str, Any] = {
                        (k or "").strip(): (v if v is not None else "")
                        for k, v in raw.items()
                    }
                    if any(str(val).strip() for val in clean.values()):
                        rows.append(clean)
                return headers, rows
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            break
    return [], []


def parse_txt_election_results(
    txt_path: str,
    session_id: Optional[str] = None,
    coordinator: Any = None,
    html_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]]:
    html_context = dict(html_context or {})
    headers, data = _read_delimited_file(txt_path)
    if not headers and not data:
        logger.error({
            "level": "ERROR",
            "type": "input",
            "message": f"[{_HANDLER_NAME}] Unable to read TXT file or file is empty.",
            "session_id": session_id,
        })
        return [], [], "", {"error": "Unparseable TXT file"}

    contest_column = None
    possible_contest_cols = [col for col in headers if _CONTEST_RX.search((col or "").lower())]
    if possible_contest_cols:
        possible_contest_cols.sort(key=lambda c: len(c or ""), reverse=True)
        contest_column = possible_contest_cols[0]

    contest_detection_diag: Dict[str, Any] = {}
    detection_lines = gather_lines_for_contest_detection(headers, data)
    detected_by_text = detect_contest_titles_from_text(
        detection_lines,
        txt_path,
        diagnostics=contest_detection_diag,
    )
    detected_by_text = list(dict.fromkeys(detected_by_text))
    if contest_detection_diag.get("raw_candidates"):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": f"[{_HANDLER_NAME}] Contest detection diagnostics available.",
            "session_id": session_id,
            "contest_detection": contest_detection_diag,
        })

    contest_names: List[str] = []
    if contest_column:
        contest_names = sorted({(row.get(contest_column, "") or "").strip() for row in data if row.get(contest_column)})
        contest_names = [c for c in contest_names if c]
    if not contest_names and detected_by_text:
        contest_names = detected_by_text
    elif (contest_column is None) and detected_by_text:
        for title in detected_by_text:
            if title not in contest_names:
                contest_names.append(title)
    if not contest_names:
        contest_names = [os.path.basename(txt_path).replace(".txt", "")]

    fname = os.path.basename(txt_path).lower()

    selection_context = {
        "selector_data": {
            "contests": [{"title": name} for name in contest_names],
            "noisy_patterns": [s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())],
        },
        "input_file": os.path.basename(txt_path),
        "handler": _HANDLER_NAME,
    }
    if contest_detection_diag:
        selection_context["contest_detection"] = contest_detection_diag
    force_contest_prompt = bool(os.environ.get("SMART_ELECTIONS_FORCE_CONTEST_PROMPT"))
    allow_parallel_auto = ENABLE_PARALLEL and not force_contest_prompt
    single_contest_detected = len(contest_names) == 1
    contest_selection_mode = "single_detected"
    if single_contest_detected and not force_contest_prompt:
        contest = contest_names[0]
    else:
        auto_pick = select_contest_auto_first(
            coordinator=coordinator,
            context=selection_context,
            session_id=session_id,
            allow_multiple=False,
            force_interactive=(not allow_parallel_auto) or force_contest_prompt,
        )
        contest_selection_mode = "auto" if (allow_parallel_auto and auto_pick) else "prompt"
        if not auto_pick:
            logger.error({
                "level": "ERROR",
                "type": "input",
                "message": f"[{_HANDLER_NAME}] No contest selected.",
                "session_id": session_id,
            })
            return [], [], "", {"error": "No contest selected"}
        contest = safe_get(auto_pick[0], "title") or contest_names[0]
    if single_contest_detected and force_contest_prompt:
        contest_selection_mode = "prompt"

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

    detection_context: Dict[str, Any] = {
        "contest": contest,
        "contests": [{"title": contest}],
        "session_id": session_id,
    }
    for key in ("state", "county", "url", "source_url", "page_url"):
        if key in html_context and html_context.get(key):
            detection_context[key] = html_context.get(key)

    state_display, county_display, state_county_diag = derive_state_county_from_table(
        headers,
        data,
        context=detection_context,
        filename=txt_path,
    )
    state = state_display or "Unknown"
    county = county_display or "Unknown"
    state_normalized = state_county_diag.get("state_normalized")
    county_normalized = state_county_diag.get("county_normalized")

    candidate_label_map, candidate_metadata, party_diag = derive_candidate_party_metadata(headers, data)

    location_diagnostics["state_county_detection"] = state_county_diag
    location_diagnostics["candidate_party_detection"] = party_diag

    domain = safe_slug(os.path.basename(txt_path))
    m = re.search(r"(19|20)\d{2}", fname)
    year = int(m.group(0)) if m else None
    if year is None:
        try:
            year_raw = html_context.get("year")
            if year_raw is not None:
                year_candidate = int(year_raw)
                if 1800 <= year_candidate <= 2100:
                    year = year_candidate
        except (TypeError, ValueError):
            pass
    context = {
        "contest": contest,
        "state": state,
        "county": county,
        "year": year,
        "session_id": session_id,
        "handler": _HANDLER_NAME,
        "source_slug": domain,
        "location_headers": location_headers,
        "precinct_attached": precinct_attached,
        "location_diagnostics": location_diagnostics,
        "state_normalized": state_normalized,
        "county_normalized": county_normalized,
        "state_county_detection": state_county_diag,
        "contest_selection_mode": contest_selection_mode,
    }
    if contest_detection_diag:
        context["contest_detection"] = contest_detection_diag
    if candidate_label_map:
        context["candidate_label_map"] = candidate_label_map
    if candidate_metadata:
        context["candidate_metadata"] = candidate_metadata
    if party_diag.get("candidate_count"):
        context["candidate_party_detection"] = party_diag
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

    finalize_context = {
        "handler": _HANDLER_NAME,
        "input_file": os.path.basename(txt_path),
        "session_id": session_id,
        "race": contest,
        "location_headers": location_headers,
        "precinct_attached": precinct_attached,
        "location_diagnostics": location_diagnostics,
        "state": state,
        "county": county,
        "state_normalized": state_normalized,
        "county_normalized": county_normalized,
        "state_county_detection": state_county_diag,
        "candidate_party_detection": party_diag,
        "contest_selection_mode": contest_selection_mode,
    }
    if contest_detection_diag:
        finalize_context["contest_detection"] = contest_detection_diag

    result = finalize_election_output(
        headers=headers_final,
        data=data_final,
        coordinator=coordinator,
        contest=contest,
        state=state,
        county=county,
        context=finalize_context,
        enable_user_feedback=False,
        session_id=session_id,
    )

    metadata = {
        "race": contest,
        "input_file": os.path.basename(txt_path),
        "output_file": os.path.basename(result.get("csv_path", "")),
        "headers": headers_final,
        "row_count": len(data_final),
        "handler": _HANDLER_NAME,
        "state": state,
        "county": county,
        "year": year,
        "csv_path": result.get("csv_path"),
        "metadata_path": result.get("metadata_path"),
        "location_headers_detected": location_headers,
        "precinct_attached": precinct_attached,
        "location_diagnostics": location_diagnostics,
        "state_normalized": state_normalized,
        "county_normalized": county_normalized,
        "state_county_detection": state_county_diag,
        "candidate_label_map": candidate_label_map,
        "candidate_metadata": candidate_metadata,
        "candidate_party_detection": party_diag,
        "contest_selection_mode": contest_selection_mode,
    }
    if contest_detection_diag:
        metadata["contest_detection"] = contest_detection_diag

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
            "message": f"[{_HANDLER_NAME}] No TXT file provided to parse().",
            "session_id": session_id,
        })
        return None, None, None, {"skipped": True}

    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": f"[{_HANDLER_NAME}] Using TXT file: {manual_file}",
        "session_id": session_id,
    })

    result = parse_txt_election_results(
        manual_file,
        session_id=session_id,
        coordinator=coordinator,
        html_context=html_context,
    )
    result_any = cast(Any, result)
    if not (isinstance(result_any, tuple) and len(result_any) == 4):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": f"[{_HANDLER_NAME}] Invalid result from parse_txt_election_results (expected 4-tuple).",
            "session_id": session_id,
            "got_type": type(result).__name__,
        })
        return None, None, None, {"error": "Invalid parse result"}
    return cast(Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]], result_any)
