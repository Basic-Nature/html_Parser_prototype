#!/usr/bin/env python3
"""
Pipeline regression checker for Smart Elections Parser.

Validates outputs referenced by .processed_urls:
- results.csv exists and is readable
- results.metadata.json is valid
- basic schema checks (headers, row counts)
- optional DL1 hash matching when available
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import orjson
except Exception:  # pragma: no cover - optional
    orjson = None

from webapp.parser.config import OUTPUT_DIR, PROCESSED_URLS_FILE, VERIFICATION_LOG_DIR
from webapp.parser.verification.local_dl_sync import LocalStorageSync


def _load_json(path: Path) -> Any:
    raw = path.read_bytes()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw.decode("utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if orjson is not None:
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_statuses(raw: str) -> List[str]:
    return [s.strip().lower() for s in (raw or "").split(",") if s.strip()]


def _parse_entry_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    return None


def _safe_path(val: Any) -> Path | None:
    if not val:
        return None
    try:
        return Path(str(val)).expanduser().resolve()
    except Exception:
        return None


def _collect_entries(processed_path: Path) -> List[Dict[str, Any]]:
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed URLs file not found: {processed_path}")
    payload = _load_json(processed_path)
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _build_dl1_hash_index(dl1_dir: Path) -> Dict[str, Path]:
    hash_index: Dict[str, Path] = {}
    for csv_path in sorted(dl1_dir.glob("*.csv")):
        try:
            file_hash = LocalStorageSync.compute_file_hash(csv_path)
            hash_index[file_hash] = csv_path
        except Exception:
            continue
    return hash_index


def _validate_csv(csv_path: Path) -> Tuple[List[str], int, List[str]]:
    warnings: List[str] = []
    headers: List[str] = []
    row_count = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not headers:
            warnings.append("csv_missing_headers")
        else:
            lower_seen = set()
            for h in headers:
                if not h or not str(h).strip():
                    warnings.append("csv_blank_header")
                    continue
                key = str(h).strip().lower()
                if key in lower_seen:
                    warnings.append("csv_duplicate_header")
                    break
                lower_seen.add(key)
        for _ in reader:
            row_count += 1
    if row_count == 0:
        warnings.append("csv_no_rows")
    return headers, row_count, warnings


def _validate_metadata(meta_path: Path) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    meta = _load_json(meta_path)
    if not isinstance(meta, dict):
        return {}, ["metadata_not_object"]
    for key in ("row_count", "headers", "csv_path"):
        if key not in meta:
            warnings.append(f"metadata_missing_{key}")
    if not meta.get("source_url"):
        warnings.append("metadata_missing_source_url")
    return meta, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Check parser outputs for regression issues.")
    parser.add_argument("--statuses", default="success,partial", help="Comma-separated statuses to validate")
    parser.add_argument("--max-entries", type=int, default=0, help="Max number of entries to check (0 = all)")
    parser.add_argument("--soft", action="store_true", help="Do not exit non-zero on failures")
    parser.add_argument("--require-dl1-match", action="store_true", help="Fail if no DL1 hash match found")
    parser.add_argument(
        "--missing-output-policy",
        choices=("fail", "warn", "skip"),
        default="warn",
        help="How to treat entries with missing output files",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=14,
        help="Treat missing outputs as stale warnings when entry older than this many days",
    )

    args = parser.parse_args()

    processed_path = Path(PROCESSED_URLS_FILE)
    statuses = set(_normalize_statuses(args.statuses))

    report: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "processed_file": str(processed_path),
        "statuses": sorted(statuses),
        "total_entries": 0,
        "checked_entries": 0,
        "failures": [],
        "warnings": [],
        "dl1_matches": 0,
        "dl1_checked": 0,
        "dl1_available": False,
    }

    try:
        entries = _collect_entries(processed_path)
    except Exception as exc:
        report["failures"].append({"type": "processed_urls_error", "error": str(exc)})
        _write_json(Path(OUTPUT_DIR) / "reports" / "pipeline_check_latest.json", report)
        return 1

    report["total_entries"] = len(entries)

    verification_dir = Path(VERIFICATION_LOG_DIR).parent
    dl1_dir = verification_dir / "dl1"
    dl1_hash_index: Dict[str, Path] = {}
    if dl1_dir.exists():
        dl1_hash_index = _build_dl1_hash_index(dl1_dir)
        report["dl1_available"] = bool(dl1_hash_index)

    for entry in entries:
        status = str(entry.get("status", "")).lower()
        if statuses and status not in statuses:
            continue
        report["checked_entries"] += 1
        if args.max_entries and report["checked_entries"] > args.max_entries:
            break

        source_url = entry.get("url") or entry.get("source_url")
        context = {
            "url": source_url,
            "status": status,
        }

        entry_ts = _parse_entry_timestamp(entry.get("timestamp"))
        stale_cutoff = datetime.now() - timedelta(days=max(args.stale_days, 0))
        is_stale = bool(entry_ts and entry_ts < stale_cutoff)

        output_dir = _safe_path(entry.get("output_dir"))
        csv_path = _safe_path(entry.get("output_file") or entry.get("csv_path"))
        meta_path = _safe_path(entry.get("metadata_path"))

        if csv_path is None and output_dir is not None:
            candidate = output_dir / "results.csv"
            if candidate.exists():
                csv_path = candidate
        if meta_path is None and output_dir is not None:
            candidate = output_dir / "results.metadata.json"
            if candidate.exists():
                meta_path = candidate

        if csv_path is None or not csv_path.exists():
            payload = {"type": "missing_csv", **context, "path": str(csv_path), "stale": is_stale}
            if args.missing_output_policy == "skip":
                report["warnings"].append(payload)
                continue
            if args.missing_output_policy == "warn" or is_stale:
                report["warnings"].append(payload)
                continue
            report["failures"].append(payload)
            continue
        if meta_path is None or not meta_path.exists():
            payload = {"type": "missing_metadata", **context, "path": str(meta_path), "stale": is_stale}
            if args.missing_output_policy == "skip":
                report["warnings"].append(payload)
                continue
            if args.missing_output_policy == "warn" or is_stale:
                report["warnings"].append(payload)
                continue
            report["failures"].append(payload)
            continue

        try:
            headers, row_count, csv_warnings = _validate_csv(csv_path)
            for warn in csv_warnings:
                report["warnings"].append({"type": warn, **context, "path": str(csv_path)})
        except Exception as exc:
            report["failures"].append({"type": "csv_read_error", **context, "error": str(exc), "path": str(csv_path)})
            continue

        try:
            metadata, meta_warnings = _validate_metadata(meta_path)
            for warn in meta_warnings:
                report["warnings"].append({"type": warn, **context, "path": str(meta_path)})
        except Exception as exc:
            report["failures"].append({"type": "metadata_read_error", **context, "error": str(exc), "path": str(meta_path)})
            continue

        meta_row_count = metadata.get("row_count")
        if isinstance(meta_row_count, int) and meta_row_count != row_count:
            report["warnings"].append({
                "type": "row_count_mismatch",
                **context,
                "csv_rows": row_count,
                "metadata_rows": meta_row_count,
                "path": str(csv_path),
            })

        meta_headers = metadata.get("headers")
        if isinstance(meta_headers, list) and headers and meta_headers != headers:
            report["warnings"].append({
                "type": "header_mismatch",
                **context,
                "path": str(csv_path),
            })

        if dl1_hash_index:
            report["dl1_checked"] += 1
            try:
                output_hash = LocalStorageSync.compute_file_hash(csv_path)
            except Exception:
                output_hash = ""
            if output_hash and output_hash in dl1_hash_index:
                report["dl1_matches"] += 1
            elif args.require_dl1_match:
                report["failures"].append({
                    "type": "dl1_no_match",
                    **context,
                    "path": str(csv_path),
                })

    report_path = Path(OUTPUT_DIR) / "reports" / "pipeline_check_latest.json"
    _write_json(report_path, report)

    if report["failures"] and not args.soft:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
