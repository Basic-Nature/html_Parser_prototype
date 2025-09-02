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
    MISC_FOOTER_KEYWORDS, CONTEST_KEYWORDS
)
from ...utils.table_core import harmonize_headers_and_data

def parse_csv_election_results(csv_path, session_id=None):
    data = []
    headers = []
    contest_column = None

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames or []]

        # Dynamic contest column detection
        possible_contest_cols = [col for col in headers if any(k in col.lower() for k in CONTEST_KEYWORDS)]
        if possible_contest_cols:
            contest_column = possible_contest_cols[0]

        for row in reader:
            row = {k.strip(): v for k, v in row.items()}
            if any(val.strip() for val in row.values() if val):
                data.append(row)

        contest = None
        if contest_column:
            contests = sorted({row[contest_column].strip() for row in data if row.get(contest_column)})
            if len(contests) > 1:
                logger.info({
                    "level": "INFO",
                    "type": "input",
                    "message": "\nMultiple contests detected:",
                    "session_id": session_id
                })
                for i, name in enumerate(contests, 1):
                    logger.info({
                        "level": "INFO",
                        "type": "input",
                        "message": f" {i:2d}. {name}",
                        "session_id": session_id
                    })
                prompt_message = "\nEnter the contest name (exactly as shown), or type its number: "
                def validator(x):
                    x = str(x).strip()
                    if x.isdigit():
                        idx = int(x)
                        return 1 <= idx <= len(contests)
                    return x in contests
                user_input = prompt.prompt_input(
                    prompt_message,
                    validator=validator,
                    session_id=session_id,
                    context={"contests": contests}
                )
                if user_input is None:
                    logger.error({
                        "level": "ERROR",
                        "type": "input",
                        "message": "No contest selected.",
                        "session_id": session_id
                    })
                    return None, None, None, {"error": "No contest selected"}
                if str(user_input).isdigit():
                    idx = int(user_input)
                    try:
                        contest = contests[idx - 1]
                    except IndexError:
                        logger.error({
                            "level": "ERROR",
                            "type": "input",
                            "message": "Invalid contest number.",
                            "session_id": session_id
                        })
                        return None, None, None, {"error": "Invalid contest number"}
                else:
                    if user_input not in contests:
                        logger.error({
                            "level": "ERROR",
                            "type": "input",
                            "message": f"[ERROR] Contest name '{user_input}' not found.",
                            "session_id": session_id
                        })
                        return None, None, None, {"error": "Contest name not found"}
                    contest = user_input
                data = [row for row in data if row.get(contest_column, "").strip() == contest]
            elif contests:
                contest = contests[0]
        else:
            contest = os.path.basename(csv_path).replace(".csv", "")

    candidate_cols = [col for col in headers if any(k in col.lower() for k in CANDIDATE_KEYWORDS)]
    precinct_cols = [col for col in headers if any(k in col.lower() for k in LOCATION_KEYWORDS)]
    method_keys = set(BALLOT_TYPES) | set(TOTAL_KEYWORDS) | set(MISC_FOOTER_KEYWORDS)
    method_cols = [col for col in headers if any(m in col.lower() for m in method_keys)]

    wide_data = []
    reporting_unit_col = precinct_cols[0] if precinct_cols else headers[0]
    for row in data:
        wide_row = {reporting_unit_col: row.get(reporting_unit_col, "")}
        for cand_col in candidate_cols:
            candidate = row.get(cand_col, "")
            for method_col in method_cols:
                val = row.get(method_col, "")
                col_name = f"{candidate} - {method_col}"
                wide_row[col_name] = val
        if not candidate_cols:
            for method_col in method_cols:
                wide_row[method_col] = row.get(method_col, "")
        for col in headers:
            if col not in candidate_cols + method_cols + [reporting_unit_col]:
                wide_row[col] = row.get(col, "")
        wide_data.append(wide_row)

    all_keys = set()
    for row in wide_data:
        all_keys.update(row.keys())
    headers = [reporting_unit_col] + sorted([k for k in all_keys if k != reporting_unit_col])
    headers, wide_data = harmonize_headers_and_data(headers, wide_data)

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in contest).replace(" ", "_")
    output_csv = os.path.join(OUTPUT_DIR, f"{safe_title}_parsed.csv")
    output_meta = os.path.join(OUTPUT_DIR, f"{safe_title}_metadata.json")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in wide_data:
            writer.writerow(row)

    metadata = {
        "race": contest,
        "input_file": os.path.basename(csv_path),
        "output_file": os.path.basename(output_csv),
        "headers": headers,
        "row_count": len(wide_data),
        "handler": "csv_handler"
    }
    with open(output_meta, "w") as jf:
        jf.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[OUTPUT] Wrote {len(wide_data)} rows to: {output_csv}",
        "session_id": session_id
    })
    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"[OUTPUT] Metadata written to: {output_meta}",
        "session_id": session_id
    })

    return headers, wide_data, contest, metadata

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
    
    return parse_csv_election_results(manual_file, session_id=session_id)