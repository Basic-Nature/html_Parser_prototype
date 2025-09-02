from __future__ import annotations
# ============================================================
# 🗳️ Smart Elections: Universal JSON Election Results Parser
# ============================================================
import orjson
import os
import csv
import time
from collections import defaultdict
from ...config import (
    OUTPUT_DIR
)
from ...Context_Integration.Context_Library.constants import (
    GROUP_RENAME_MAP, LOCATION_KEYWORDS, CANDIDATE_KEYWORDS, BALLOT_TYPES, PARTY_KEYWORDS, TOTAL_KEYWORDS,
    MISC_FOOTER_KEYWORDS, CONTEST_KEYWORDS
)
from ...utils.logger_singleton import logger, console, prompt
from ...utils.table_core import harmonize_headers_and_data

def find_key_by_keywords(obj, keywords):
    """Find the first key in obj that matches any keyword (case-insensitive, partial match allowed)."""
    for key in obj.keys():
        for kw in keywords:
            if kw.lower() in key.lower():
                return key
    return None

def parse_json_election_results(json_path, session_id=None):
    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())

    # --- Dynamic contest extraction ---
    ballot_items_key = find_key_by_keywords(
        data.get("results", {}),
        set(CONTEST_KEYWORDS) | {"ballotitem", "ballotitems"}
    )
    ballot_items = data.get("results", {}).get(ballot_items_key, []) if ballot_items_key else []

    contests = set()
    contest_name_key = (
        find_key_by_keywords(ballot_items[0], set(CONTEST_KEYWORDS) | {"name"})
        if ballot_items else "name"
    )
    for item in ballot_items:
        name = item.get(contest_name_key, "").strip()
        if name:
            contests.add(name)

    logger.info({
        "level": "INFO",
        "type": "input",
        "message": "\nAvailable contests:",
        "session_id": session_id
    })
    for i, name in enumerate(sorted(contests), 1):
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

    try:
        user_input = prompt.prompt_input(
            prompt_message,
            validator=validator,
            session_id=session_id,
            context={"contests": sorted(contests)}
        )
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "input",
            "message": f"Exception during contest selection: {e}",
            "session_id": session_id
        })
        return None, None, None, {"error": "Contest selection failed"}

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
            target_contest = sorted(contests)[idx - 1]
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
                "message": "Contest name not found.",
                "session_id": session_id
            })
            return None, None, None, {"error": "Contest name not found"}
        target_contest = user_input

    logger.info({
        "level": "INFO",
        "type": "input",
        "message": f"\n🔍 Parsing contest: {target_contest}\n",
        "session_id": session_id
    })

    # --- Dynamic candidate/party/ballot type/precinct key detection ---
    contest_item = next((item for item in ballot_items if item.get(contest_name_key, "").strip() == target_contest), None)
    if not contest_item:
        logger.error({
            "level": "ERROR",
            "type": "input",
            "message": "Selected contest not found in data.",
            "session_id": session_id
        })
        return None, None, None, {"error": "Selected contest not found"}

    ballot_options_key = find_key_by_keywords(contest_item, set(CANDIDATE_KEYWORDS) | {"ballotoption", "ballotoptions"})
    ballot_options = contest_item.get(ballot_options_key, []) if ballot_options_key else []

    candidate_name_key = (
        find_key_by_keywords(ballot_options[0], set(CANDIDATE_KEYWORDS) | {"name"})
        if ballot_options else "name"
    )
    party_key = (
        find_key_by_keywords(ballot_options[0], set(PARTY_KEYWORDS) | {"politicalparty"})
        if ballot_options else "politicalParty"
    )
    precinct_results_key = find_key_by_keywords(
        ballot_options[0],
        set(LOCATION_KEYWORDS) | {"precinctresult", "precinctresults"}
    )
    group_results_key = find_key_by_keywords(
        ballot_options[0],
        set(BALLOT_TYPES) | {"groupresult", "groupresults"}
    )

    # --- Build normalization map for candidates/parties ---
    raw_candidates = {}
    for opt in ballot_options:
        raw = opt.get(candidate_name_key, "").strip()
        party = opt.get(party_key, "Unknown")
        label = f"{raw} ({party})"
        raw_candidates[raw] = label

    normalization_map = {k: v for k, v in raw_candidates.items()}

    # --- Build results ---
    results_nested = defaultdict(lambda: defaultdict(dict))
    for opt in ballot_options:
        raw_label = opt.get(candidate_name_key, "").strip()
        precinct_results = opt.get(precinct_results_key, []) if precinct_results_key else []
        for precinct in precinct_results:
            precinct_name_key = find_key_by_keywords(precinct, set(LOCATION_KEYWORDS) | {"name"})
            p = precinct.get(precinct_name_key, "").strip()
            results_nested[p][raw_label]["Total"] = precinct.get("voteCount")
            group_results = precinct.get(group_results_key, []) if group_results_key else []
            for grp in group_results:
                group_name_key = find_key_by_keywords(grp, set(BALLOT_TYPES) | {"groupname"})
                g = grp.get(group_name_key, "").strip()
                norm_g = GROUP_RENAME_MAP.get(g.lower(), g) if g else g
                results_nested[p][raw_label][norm_g] = grp.get("voteCount")

    # --- Build rows and headers dynamically ---
    rows = []
    all_keys = set()
    for precinct, cands in results_nested.items():
        row = {"Precinct": precinct}
        for raw_label, methods in cands.items():
            norm_label = normalization_map.get(raw_label, raw_label)
            for method, count in methods.items():
                col_name = f"{norm_label} - {method}"
                row[col_name] = count
                all_keys.add(col_name)
        rows.append(row)

    headers = ["Precinct"] + sorted(all_keys)
    headers, rows = harmonize_headers_and_data(headers, rows)

    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in target_contest).replace(" ", "_")
    output_csv = os.path.join(OUTPUT_DIR, f"{safe_title}_parsed.csv")
    output_meta = os.path.join(OUTPUT_DIR, f"{safe_title}_metadata.json")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    metadata = {
        "race": target_contest,
        "input_file": os.path.basename(json_path),
        "output_file": os.path.basename(output_csv),
        "headers": headers,
        "row_count": len(rows),
        "handler": "json_handler"
    }
    with open(output_meta, "w") as jf:
        jf.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))

    logger.info({
        "level": "INFO",
        "type": "output",
        "message": f"✅ Completed! Output CSV: {output_csv}, Metadata: {output_meta}",
        "session_id": session_id
    })

    return headers, rows, target_contest, metadata

def parse(page=None, coordinator=None, html_context=None, manual_file=None, session_id=None, **kwargs):
    """
    Universal pipeline entry: Accepts a JSON file path (manual_file) from the format router.
    Returns: headers, data, contest, metadata
    """
    html_context = html_context or {}
    if html_context.get("skip_format") or html_context.get("manual_skip"):
        logger.info({
            "level": "INFO",
            "type": "handler",
            "message": "[SKIP] JSON parsing intentionally skipped via context flag.",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    if not manual_file or not os.path.isfile(manual_file):
        logger.error({
            "level": "ERROR",
            "type": "handler",
            "message": "[ERROR] No JSON file provided to parse().",
            "session_id": session_id
        })
        return None, None, None, {"skipped": True}

    logger.info({
        "level": "INFO",
        "type": "handler",
        "message": f"[INFO] Using JSON file: {manual_file}",
        "session_id": session_id
    })

    return parse_json_election_results(manual_file, session_id=session_id)