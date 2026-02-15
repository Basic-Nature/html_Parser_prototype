"""
Import SMART Elections Database-Lite into warehouse_election_results.

Usage:
  python scripts/import_database_lite.py --sheet-id <ID> [--worksheet "Sheet1"] [--limit N] [--dry-run]

Defaults:
  - sheet-id from SE_DB_LITE_SHEET_ID, or GOOGLE_SHEETS_WORKBOOK_ID
  - worksheet: first sheet
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import gspread
from google.oauth2.service_account import Credentials
from sqlalchemy import MetaData, Table, inspect
from webapp.parser.Context_Integration.librarian import clean_for_json
from webapp.parser.utils.db_utils import SessionLocal, get_engine, update_batch_metadata
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.models import BatchMetadata

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

STATE_HINTS = ("state",)
COUNTY_HINTS = ("county",)
CONTEST_HINTS = ("contest", "race", "office")
CANDIDATE_HINTS = ("candidate", "ballot candidate", "choice", "nominee")
PARTY_HINTS = ("party", "ballot party")
DATE_HINTS = ("election date", "date", "election_day")
YEAR_HINTS = ("year",)
SOURCE_URL_HINTS = ("source", "source url", "source link", "download")

TOTAL_VOTE_HINTS = (
    "total votes",
    "votes",
    "vote total",
    "vote count",
    "calculated total votes",
    "uncategorized votes",
)

VOTE_COMPONENT_HINTS = (
    "early",
    "election day",
    "mail",
    "absentee",
    "provisional",
    "write-in",
)


def _normalize_header(header: str) -> str:
    return header.strip().lower()


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _pick_value(row: Dict[str, Any], headers: Iterable[str]) -> Optional[str]:
    for header in headers:
        if header in row and _has_value(row[header]):
            return str(row[header]).strip()
    return None


def _find_header_by_hints(headers: Iterable[str], hints: Iterable[str]) -> List[str]:
    normalized = [(h, _normalize_header(h)) for h in headers if isinstance(h, str)]
    matches = []
    for raw, norm in normalized:
        if any(hint in norm for hint in hints):
            matches.append(raw)
    return matches


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"na", "n/a", "null", "none", "--", "-"}:
        return None
    try:
        if "." in text:
            return int(round(float(text)))
        return int(text)
    except ValueError:
        return None


def _parse_date(value: Any, year_value: Any) -> Optional[datetime]:
    if value:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        text = str(value).strip()
        if text:
            text = text.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(text)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    if year_value:
        try:
            year_int = int(str(year_value).strip())
            return datetime(year_int, 1, 1, tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _extract_votes(row: Dict[str, Any], headers: List[str]) -> Tuple[Optional[int], Dict[str, int]]:
    totals = []
    components: Dict[str, int] = {}

    total_headers = _find_header_by_hints(headers, TOTAL_VOTE_HINTS)
    for header in total_headers:
        value = _coerce_int(row.get(header))
        if value is not None:
            totals.append(value)

    if totals:
        return totals[0], components

    component_headers = _find_header_by_hints(headers, VOTE_COMPONENT_HINTS)
    for header in component_headers:
        value = _coerce_int(row.get(header))
        if value is not None:
            components[header] = value

    if components:
        return sum(components.values()), components

    return None, components


def _load_sheet(sheet_id: str, worksheet_name: Optional[str]) -> Tuple[List[str], List[List[str]], str]:
    creds_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_PATH is not set")
    credentials = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    client = gspread.authorize(credentials)
    workbook = client.open_by_key(sheet_id)
    if worksheet_name:
        worksheet = workbook.worksheet(worksheet_name)
    else:
        worksheet = workbook.get_worksheet(0)
    rows = worksheet.get_all_values()
    if not rows:
        return [], [], worksheet.title
    headers = rows[0]
    data = rows[1:]
    return headers, data, worksheet.title


def build_records(headers: List[str], data: List[List[str]], *, limit: Optional[int]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "rows": 0,
        "skipped": 0,
        "missing_votes": 0,
        "missing_candidate": 0,
    }
    records: List[Dict[str, Any]] = []
    if not headers:
        return records, stats

    header_map = {h: h for h in headers if h}

    state_headers = _find_header_by_hints(headers, STATE_HINTS)
    county_headers = _find_header_by_hints(headers, COUNTY_HINTS)
    contest_headers = _find_header_by_hints(headers, CONTEST_HINTS)
    candidate_headers = _find_header_by_hints(headers, CANDIDATE_HINTS)
    party_headers = _find_header_by_hints(headers, PARTY_HINTS)
    date_headers = _find_header_by_hints(headers, DATE_HINTS)
    year_headers = _find_header_by_hints(headers, YEAR_HINTS)
    source_headers = _find_header_by_hints(headers, SOURCE_URL_HINTS)

    for idx, row_values in enumerate(data, start=2):
        if limit and len(records) >= limit:
            break
        row = {headers[i]: row_values[i] if i < len(row_values) else "" for i in range(len(headers))}
        stats["rows"] += 1

        state = _pick_value(row, state_headers)
        county = _pick_value(row, county_headers)
        contest = _pick_value(row, contest_headers)
        candidate = _pick_value(row, candidate_headers)
        party = _pick_value(row, party_headers)
        date_value = _pick_value(row, date_headers)
        year_value = _pick_value(row, year_headers)
        source_url = _pick_value(row, source_headers)

        votes, components = _extract_votes(row, headers)
        if votes is None:
            stats["skipped"] += 1
            stats["missing_votes"] += 1
            continue
        if not candidate:
            stats["skipped"] += 1
            stats["missing_candidate"] += 1
            continue

        election_date = _parse_date(date_value, year_value)

        record = {
            "state": state or "Unknown",
            "county": county or "Unknown",
            "contest": contest or "Unknown Contest",
            "candidate": candidate,
            "party": party,
            "votes": votes,
            "precinct": "All Precincts",
            "election_date": election_date,
            "source_url": source_url,
            "metastats": {
                "source_row_index": idx,
                "source_headers": list(header_map.keys()),
                "vote_components": components,
                "source_row": clean_for_json(row),
            },
        }
        records.append(record)

    return records, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Database-Lite Google Sheet into warehouse_election_results")
    parser.add_argument("--sheet-id", default=os.getenv("SE_DB_LITE_SHEET_ID") or os.getenv("GOOGLE_SHEETS_WORKBOOK_ID"))
    parser.add_argument("--worksheet", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verification-status", default="verified")
    parser.add_argument("--source-principal", default="database_lite_sheet")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    if not args.sheet_id:
        raise SystemExit("Missing --sheet-id or SE_DB_LITE_SHEET_ID")

    headers, data, sheet_title = _load_sheet(args.sheet_id, args.worksheet)
    records, stats = build_records(headers, data, limit=args.limit)

    engine = get_engine()
    inspector = inspect(engine)
    available_cols = {col["name"] for col in inspector.get_columns("warehouse_election_results")}
    warehouse_table = Table("warehouse_election_results", MetaData(), autoload_with=engine)

    print(f"Sheet: {sheet_title} ({args.sheet_id})")
    print(f"Rows processed: {stats['rows']}")
    print(f"Records ready: {len(records)}")
    if stats["skipped"]:
        print(f"Skipped: {stats['skipped']} (missing votes: {stats['missing_votes']}, missing candidate: {stats['missing_candidate']})")

    if args.dry_run:
        if records:
            sample = records[0].copy()
            sample["metastats"] = {"source_row_index": sample["metastats"]["source_row_index"]}
            print("Sample record:")
            print(sample)
        return 0

    batch_id = None
    batch_session = SessionLocal()
    try:
        batch = BatchMetadata(source=f"database_lite:{args.sheet_id}", status="PENDING")
        batch_session.add(batch)
        batch_session.flush()
        batch_id = batch.batch_id
        batch_session.commit()
    except Exception as exc:
        batch_session.rollback()
        logger.error({
            "level": "ERROR",
            "type": "database",
            "message": f"Batch creation failed: {exc}",
            "session_id": None,
        })
        raise
    finally:
        batch_session.close()

    if not batch_id:
        raise RuntimeError("Failed to create batch record.")

    inserted = 0
    session = SessionLocal()
    try:
        batch_values = []
        for idx, record in enumerate(records, start=1):
            payload = dict(record)
            payload.update({
                "batch_id": batch_id,
                "verification_status": args.verification_status,
                "source_principal": args.source_principal,
            })
            filtered = {k: v for k, v in payload.items() if k in available_cols}
            batch_values.append(filtered)
            if idx % args.batch_size == 0:
                session.execute(warehouse_table.insert(), batch_values)
                session.commit()
                batch_values = []
            inserted += 1
        if batch_values:
            session.execute(warehouse_table.insert(), batch_values)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error({
            "level": "ERROR",
            "type": "database",
            "message": f"Database-Lite import failed: {exc}",
            "session_id": None,
        })
        update_batch_metadata(batch_id, status="ERROR")
        raise
    finally:
        session.close()

    update_batch_metadata(batch_id, status="COMPLETED")
    print(f"Inserted {inserted} rows into warehouse_election_results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
