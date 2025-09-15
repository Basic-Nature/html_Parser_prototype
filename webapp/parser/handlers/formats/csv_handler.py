from __future__ import annotations
# ==============================================================
# 🗳️ Smart Elections: Universal CSV Election Results Parser
# ==============================================================
import csv
import os
import orjson
import time
from ...config import (
    OUTPUT_DIR
)
from ...utils.logger_singleton import logger, prompt
from ...Context_Integration.Context_Library.constants import (
    LOCATION_KEYWORDS, CANDIDATE_KEYWORDS, BALLOT_TYPES, PARTY_KEYWORDS, TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS, CONTEST_KEYWORDS, CONTEST_TITLE_SKIP_PHRASES
)
from ...utils.table_core import harmonize_headers_and_data
from ...utils.contest_selector import select_contest
from ...utils.table_builder import build_table_noninteractive
from ...utils.output_utils import finalize_election_output
from ...utils.shared_logic import safe_slug
import re

def _build_contest_regex(keywords) -> re.Pattern:
    parts = []
    for phrase in (keywords or []):
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        toks = re.split(r"\s+", phrase.strip().lower())
        xtoks = []
        for t in toks:
            t = re.escape(t)
            t = t.replace(r"\.", r"\.?")
            t = t.replace(r"\-", r"[-\s]?")
            xtoks.append(t)
        pat = r"(?:[\s\-_\/]*?)".join(xtoks)
        pat = rf"(?<![A-Za-z0-9]){pat}(?![A-Za-z0-9])"
        parts.append(pat)
    return re.compile("|".join(parts), re.I) if parts else re.compile(r"(?!x)x", re.I)

_CONTEST_RX = _build_contest_regex(CONTEST_KEYWORDS)

def parse_csv_election_results(csv_path, session_id=None, coordinator=None):
    data = []
    headers = []
    contest_column = None

    # Robust file open with encoding fallback
    try:
        f = open(csv_path, newline='', encoding='utf-8')
    except Exception:
        f = open(csv_path, newline='', encoding='latin-1')

    with f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in (reader.fieldnames or [])]

        # Dynamic contest column detection (regex-tolerant)
        possible_contest_cols = [col for col in headers if _CONTEST_RX.search((col or "").lower())]
        if possible_contest_cols:
            # prefer the most specific (longest) column name
            possible_contest_cols.sort(key=lambda c: len(c or ""), reverse=True)
            contest_column = possible_contest_cols[0]

        for row in reader:
            row = { (k or "").strip(): (v if v is not None else "") for k, v in (row.items() if row else []) }
            if any((val or "").strip() for val in row.values()):
                data.append(row)

    # Build contest candidates
    contest_names = []
    if contest_column:
        contest_names = sorted({(row.get(contest_column, "") or "").strip() for row in data if row.get(contest_column)})
        contest_names = [c for c in contest_names if c]  # drop blanks
    if not contest_names:
        contest_names = [os.path.basename(csv_path).replace(".csv", "")]

    # Light context from filename
    fname = os.path.basename(csv_path).lower()
    state = "Unknown"
    county = "Unknown"
    for part in fname.replace(".csv", "").split("_"):
        if "county" in part:
            county = part.replace("county", "").strip().title() + " County"
        if len(part) == 2 and part.isalpha():
            state = part.upper()

    # Fast-path if only one contest
    if len(contest_names) == 1:
        contest = contest_names[0]
    else:
        selector_data = {
            "contests": [{"title": name} for name in contest_names],
            "noisy_patterns": [s.lower() for s in (CONTEST_TITLE_SKIP_PHRASES or set())]
        }
        selected = select_contest(
            coordinator=coordinator,
            state=state, county=county, year=None,
            session_id=session_id,
            context={"selector_data": selector_data},
            allow_multiple=False,
            prompt_message="[PROMPT] Select contest (index, text, or 'cancel'): ",
            force_interactive=True,
            disable_ml_verify=False
        )
        if not selected:
            logger.error({
                "level": "ERROR",
                "type": "input",
                "message": "No contest selected.",
                "session_id": session_id
            })
            return None, None, None, {"error": "No contest selected"}
        contest = (selected[0] or {}).get("title") or contest_names[0]

    # Filter rows by selected contest if we have a contest column
    if contest_column:
        data = [row for row in data if (row.get(contest_column, "") or "").strip() == contest]

    # Build table via non-interactive builder
    m = re.search(r"(19|20)\d{2}", fname)
    year = int(m.group(0)) if m else None
    domain = safe_slug(os.path.basename(csv_path))
    context = {
        "contest": contest,
        "state": state,
        "county": county,
        "year": year,
        "session_id": session_id,
        "handler": "csv_handler",
        # include slugs in context so downstream naming is stable
        "source_slug": domain
    }
    headers_final, data_final, _entity_info = build_table_noninteractive(
        domain=domain,
        headers=headers,
        data=data,
        coordinator=coordinator,
        context=context,
        pivot_to_wide=True,
        debug=False
    )

    result = finalize_election_output(
        headers=headers_final,
        data=data_final,
        coordinator=coordinator,
        contest=contest,
        state=state,
        county=county,
        context={
            "handler": "csv_handler",
            "input_file": os.path.basename(csv_path),
            "session_id": session_id,
            "race": contest
        },
        enable_user_feedback=False,
        session_id=session_id
    )

    metadata = {
        "race": contest,
        "input_file": os.path.basename(csv_path),
        "output_file": os.path.basename(result.get("csv_path", "")),
        "headers": headers_final,
        "row_count": len(data_final),
        "handler": "csv_handler",
        "state": state,
        "county": county,
        "year": year,
        "csv_path": result.get("csv_path"),
        "metadata_path": result.get("metadata_path")
    }

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[OUTPUT] Wrote {len(data_final)} rows to: {result.get('csv_path')}",
        "session_id": session_id
    })
    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[OUTPUT] Metadata written to: {result.get('metadata_path')}",
        "session_id": session_id
    })

    return headers_final, data_final, contest, metadata

def parse(page=None, coordinator=None, html_context=None, manual_file=None, session_id=None, **kwargs):
    """
    Universal pipeline entry: Accepts a CSV file path (manual_file) from the format router.
    Returns: headers, data, contest, metadata
    """
    html_context = html_context or {}
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[SKIP] CSV parsing intentionally skipped via context flag.",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    if not manual_file or not os.path.isfile(manual_file):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] No CSV file provided to parse().",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": f"[INFO] Using CSV file: {manual_file}",
        "session_id": session_id
    })

    result = parse_csv_election_results(manual_file, session_id=session_id, coordinator=coordinator)

    # Defensive: always return a 4-tuple, never a bool
    if not (isinstance(result, tuple) and len(result) == 4):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] Invalid result from parse_csv_election_results (expected 4-tuple).",
            "session_id": session_id,
            "got_type": type(result).__name__
        })
        return None, None, None, {"error": "Invalid parse result"}
    return result