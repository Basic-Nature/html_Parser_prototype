"""Exercise the export-2012NovGen fast-path twice to ensure table output stays stable."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from unittest.mock import patch

from webapp.parser.utils import logger_singleton
from webapp.parser.handlers.formats import json_handler

logger_singleton.set_log_level("ERROR")
logger_singleton.logger.suppress(True)

EXPORT_PATH = Path("uploads/export-2012NovGen.json")
if not EXPORT_PATH.exists():
    raise SystemExit(f"Missing fixture: {EXPORT_PATH}")


def _pick_us_house(contests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the first contest entry that references a U.S. Representative race."""

    for entry in contests:
        title = (entry.get("title") or "").lower()
        meta = entry.get("metadata") or {}
        primary = (meta.get("primary_title") or "").lower()
        if "u.s. representative" in title or "u.s. representative" in primary:
            return [entry]
    return contests[:1]


captured_runs: List[Dict[str, Any]] = []


def _capture_finalize(headers, data, coordinator, contest, state, county, context, **kwargs):
    captured_runs.append(
        {
            "contest": contest,
            "state": state,
            "county": county,
            "headers": len(headers or []),
            "rows": len(data or []),
            "sample_headers": headers[:8],
        }
    )
    return {"csv_path": "<mock>", "metadata_path": "<mock>"}


def _fake_selector(*, context: dict | None = None, **kwargs):
    selector_data = (context or {}).get("selector_data", {})
    contests = selector_data.get("contests", [])
    return _pick_us_house(contests)


with patch("webapp.parser.handlers.formats.json_handler.select_contest_auto_first", _fake_selector), patch(
    "webapp.parser.handlers.formats.json_handler.finalize_election_output", _capture_finalize
):
    for run_idx in range(2):
        json_handler.parse_json_election_results(
            str(EXPORT_PATH),
            session_id=f"export-2012NovGen-run-{run_idx}",
            coordinator=None,
        )

if not captured_runs:
    raise SystemExit("Fast-path did not emit any table output")

first_contest = captured_runs[0]["contest"]

print(
    {
        "contest": first_contest,
        "runs": captured_runs,
    }
)
