from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from webapp.parser.config import CONTEXT_LIBRARY_PATH
from webapp.parser.Context_Integration.Integrity_check import print_integrity_summary
from webapp.parser.Context_Integration.librarian import load_context_library
from webapp.parser.utils.logger_singleton import logger


def load_contests(context_path: Path) -> list[dict[str, Any]]:
    """Return the contest entries stored in the context library."""
    library = load_context_library(str(context_path)) or {}
    contests = library.get("contests") if isinstance(library, dict) else []
    if not isinstance(contests, list):
        logger.warning("[INTEGRITY] Context library at %s is missing contest data", context_path)
        return []
    return contests


def run_integrity_summary(
    context_path: Path | None = None,
    expected_year: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Load contests and stream the Integrity_check summary to stdout."""
    target_path = Path(context_path or CONTEXT_LIBRARY_PATH)
    contests = load_contests(target_path)
    if limit is not None and limit > 0:
        contests = contests[:limit]
    total = len(contests)
    logger.info("[INTEGRITY] Starting summary for %s contests (context=%s)", total, target_path)
    if not contests:
        print("[INTEGRITY] No contest entries found; skipping summary.")
        return {"context_path": str(target_path), "contest_count": 0}
    print_integrity_summary(contests, expected_year=expected_year)
    print(
        f"[INTEGRITY] Completed summary for {total} contest(s) using {target_path}"
    )
    return {"context_path": str(target_path), "contest_count": total}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Integrity_check summary over the current context library.",
    )
    parser.add_argument(
        "--context",
        type=Path,
        help="Path to context_library.json (defaults to configured CONTEXT_LIBRARY_PATH).",
    )
    parser.add_argument(
        "--expected-year",
        type=int,
        dest="expected_year",
        help="Optional election year hint for anomaly detection.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only run over the first N contests (useful for smoke tests).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        summary = run_integrity_summary(
            context_path=args.context,
            expected_year=args.expected_year,
            limit=args.limit,
        )
    except FileNotFoundError as exc:
        logger.error("[INTEGRITY] Context library not found: %s", exc)
        raise SystemExit(1) from exc
    print(f"[INTEGRITY] Summary metadata: {summary}")


if __name__ == "__main__":
    main()
