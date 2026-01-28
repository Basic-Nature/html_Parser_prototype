"""Deduplicate webapp/parser/urls.txt by URL (last tab-delimited field).

Usage:
  python scripts/dedupe_urls.py [--path PATH]

By default this writes a backup `urls.txt.bak.TIMESTAMP` and rewrites
the original file. It preserves comments and blank lines and keeps
the first occurrence of each URL.
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
    parser.add_argument("--path", default="webapp/parser/urls.txt", help="Path to urls.txt")
    args = parser.parse_args()

    p = Path(args.path)
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
