from __future__ import annotations

import os
from pathlib import Path

import orjson

from webapp.parser.config import OUTPUT_DIR, PROCESSED_URLS_FILE
from webapp.parser.utils.misc_utils import load_processed_urls


def _resolve_metadata_path(entry: dict) -> str | None:
    meta_path = entry.get("metadata_path")
    if isinstance(meta_path, str) and meta_path:
        return meta_path

    output_dir = entry.get("output_dir")
    if not output_dir and isinstance(entry.get("output_file"), str):
        output_dir = os.path.dirname(entry.get("output_file"))

    if isinstance(output_dir, str) and output_dir:
        candidate = os.path.join(output_dir, "results.metadata.json")
        if os.path.exists(candidate):
            return candidate

    return None


def _load_metadata(meta_path: str) -> dict | None:
    try:
        if not meta_path or not os.path.exists(meta_path):
            return None
        with open(meta_path, "rb") as fh:
            data = orjson.loads(fh.read())
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_metadata(meta_path: str, data: dict) -> bool:
    try:
        with open(meta_path, "wb") as fh:
            fh.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        return True
    except Exception:
        return False


def backfill_source_url() -> int:
    processed = load_processed_urls()
    if not processed:
        print("No processed URL cache found.")
        return 0

    updated = 0
    skipped = 0

    for url, entry in processed.items():
        if not isinstance(entry, dict):
            skipped += 1
            continue
        meta_path = _resolve_metadata_path(entry)
        if not meta_path:
            skipped += 1
            continue
        meta = _load_metadata(meta_path)
        if not meta:
            skipped += 1
            continue
        if meta.get("source_url"):
            skipped += 1
            continue
        meta["source_url"] = url
        if _write_metadata(meta_path, meta):
            updated += 1
        else:
            skipped += 1

    print(f"Processed cache: {Path(PROCESSED_URLS_FILE).resolve()}")
    print(f"Output root: {Path(OUTPUT_DIR).resolve()}")
    print(f"Updated metadata files: {updated}")
    print(f"Skipped entries: {skipped}")
    return updated


if __name__ == "__main__":
    backfill_source_url()
