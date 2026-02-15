#!/usr/bin/env python3
"""Smoke test for learned navigation recipe replay.

Writes a temporary navigation learning log entry and verifies it is converted
into a learned recipe and matched for the same state/county.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from webapp.parser.navigator.navigation_recipes import NavigationRecipeStore


def _write_log_entry(path: Path) -> None:
    entry = {
        "timestamp": "2026-02-11T00:00:00Z",
        "script_id": "learned_demo",
        "success": True,
        "context_before": {
            "state": "new_york",
            "county": "rockland",
            "url": "https://example.gov/results",
        },
        "context_after": {
            "state": "new_york",
            "county": "rockland",
            "url": "https://example.gov/results",
        },
        "telemetry": [
            {
                "action": "click",
                "status": "ok",
                "details": {
                    "selector": "a#results",
                    "wait_after_ms": 500,
                },
            }
        ],
        "metadata": {
            "page_url": "https://example.gov/results",
            "url_domain": "example.gov",
        },
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "navigation_learning_log.jsonl"
        _write_log_entry(log_path)

        store = NavigationRecipeStore(
            learned_log_path=log_path,
            learned_enabled=True,
            learned_min_actions=1,
            learned_min_ok_ratio=1.0,
        )

        learned = store.load_learned()
        if not learned:
            raise SystemExit("No learned recipes were generated.")

        context = {"state": "new_york", "county": "rockland"}
        matches = store.candidates_for(context)
        learned_matches = [m for m in matches if str(m.get("id", "")).startswith("learned::")]
        if not learned_matches:
            raise SystemExit("Learned recipe did not match expected context.")

        steps = learned_matches[0].get("steps") or []
        if not steps or steps[0].get("action") != "click":
            raise SystemExit("Learned recipe did not capture expected steps.")

        print("PASS: learned navigation recipe converted and matched.")


if __name__ == "__main__":
    main()
