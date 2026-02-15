"""Deduplicate list fields in the context library JSON."""
from __future__ import annotations

import argparse
from pathlib import Path

from webapp.parser.config import CONTEXT_LIBRARY_PATH
from webapp.parser.Context_Integration import librarian


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate context library fields.")
    parser.add_argument(
        "--path",
        default=str(CONTEXT_LIBRARY_PATH),
        help="Path to context_library.json (defaults to configured path).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = Path(args.path)
    librarian.dedupe_context_library_fields(path)
    print(f"Deduplicated context library at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
