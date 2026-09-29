"""Deduplicate an explicitly selected noncanonical URL-list file by URL.

Usage:
  python scripts/dedupe_urls.py --path PATH

The tracked canonical webapp/parser/urls.txt artifact is preserved and cannot
be selected by this utility. Generic explicit noncanonical file maintenance is
retained for operator-owned files. Comments and blank lines are preserved and
the first occurrence of each URL is kept.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def dedupe_urls_file(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out_lines: List[str] = []
    seen = set()
    kept = 0
    removed = 0

    for ln in lines:
        if not ln.strip() or ln.lstrip().startswith("#"):
            out_lines.append(ln)
            continue
        parts = ln.split("\t")
        url = parts[-1].strip() if parts else ln.strip()
        if not url:
            out_lines.append(ln)
            continue
        if url in seen:
            removed += 1
            continue
        seen.add(url)
        out_lines.append(ln)
        kept += 1

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=None,
        help="Explicit noncanonical URL-list file to deduplicate.",
    )
    args = parser.parse_args()

    if args.path is None:
        print(
            "ERROR: --path is required. The tracked canonical urls.txt artifact "
            "is not a default maintenance authority."
        )
        raise SystemExit(2)

    p = Path(args.path)
    canonical_urls_path = Path("webapp/parser/urls.txt")
    if p.resolve() == canonical_urls_path.resolve():
        print(
            "ERROR: mutation of tracked canonical urls.txt is prohibited. "
            "Use the governed Source Registry persistence plane."
        )
        raise SystemExit(2)

    if not p.exists():
        print(f"ERROR: {p} not found")
        raise SystemExit(2)

    # Backup (use timezone-aware UTC datetime)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = p.with_suffix(p.suffix + f".bak.{ts}")
    try:
        bak.write_bytes(p.read_bytes())
        print(f"Backup written to: {bak}")
    except Exception as e:
        print(f"WARNING: could not write backup: {e}")

    kept, removed = dedupe_urls_file(p)
    print(f"Deduplication complete. Kept: {kept}, Removed duplicates: {removed}")


if __name__ == "__main__":
    main()
