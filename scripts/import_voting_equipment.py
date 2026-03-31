"""
Import county-level voting equipment data from Verified Voting's The Verifier API.

Source (no scraping — direct download endpoint):
    https://verifiedvoting.org/api/api_sandbox.php?advanced&year=YYYY&download=csv
    https://verifiedvoting.org/api/api_sandbox.php?advanced&year=YYYY&download=json

Available years: 2006 2008 2010 2012 2014 2016 2018 2020 2022 2024 2026

Loaded model: webapp/parser/models/election_data.CountyEquipment
Joins to ElectionResult / VoterDropoff / ValidationRecord via (year, state, county).

Usage:
    # Dry-run single year (prints sample row, no DB write)
    python scripts/import_voting_equipment.py --year 2024 --dry-run

    # Import all years covering the Database-Lite range
    python scripts/import_voting_equipment.py --years 2012 2014 2016 2018 2020 2022 2024

    # Import all available years (2006–2026)
    python scripts/import_voting_equipment.py --all-years

    # Truncate existing rows before import (full refresh)
    python scripts/import_voting_equipment.py --year 2024 --replace

TODO — Column mapping:
    The exact CSV column names returned by the Verified Voting API must be confirmed
    by inspecting a live response. Run with --dry-run --show-headers to print the
    raw header row before committing to the field map below.

    Known vendor/model categories from https://verifiedvoting.org/equipmentdb/:
        Vendors: Dominion Voting Systems, ES&S, Hart InterCivic, Clear Ballot,
                 Unisyn, SmartMatic, etc.
        Standard method vocab (std_voting_method column):
            hand_marked_paper, bmd_all_voters, bmd_accessible_only,
            dre_with_vvpat, dre_without_vvpat, hybrid_bmd_tabulator,
            hand_count, optical_scan

    Update _build_record() once the live header row is confirmed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen

from webapp.parser.utils.db_utils import SessionLocal
from webapp.parser.utils.logger_singleton import logger

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------

_API_BASE = "https://verifiedvoting.org/api/api_sandbox.php"
_ALL_YEARS = [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026]

# ---------------------------------------------------------------------------
# TODO: confirm these against a live --dry-run --show-headers response
# The values below are best-guess based on the equipmentdb page and Verifier UI.
# Update once real column names are verified.
# ---------------------------------------------------------------------------
_FIELD_MAP: Dict[str, str] = {
    # CSV column name     → CountyEquipment attribute
    "State":              "state",
    "State Abbr":         "state_abbr",
    "County":             "county",
    "Jurisdiction":       "jurisdiction",
    # Standard (election day) equipment
    "Vendor":             "std_vendor",
    "Model":              "std_model",
    "Type":               "std_voting_method",
    # Accessible equipment — update keys once headers are confirmed
    "Accessible Vendor":  "acc_vendor",
    "Accessible Model":   "acc_model",
    # Mail ballot tabulation — update keys once headers are confirmed
    "Mail Vendor":        "mail_vendor",
    "Mail Model":         "mail_model",
}

# Normalised voting-method vocabulary (maps raw API strings → our enum vocab)
_METHOD_NORMALISE: Dict[str, str] = {
    "hand marked paper ballots":                    "hand_marked_paper",
    "hand marked paper ballots and bmds":           "hand_marked_paper",
    "ballot marking devices for all voters":        "bmd_all_voters",
    "hybrid bmd/tabulator for all voters":          "hybrid_bmd_tabulator",
    "dres with vvpat for all voters":               "dre_with_vvpat",
    "dres without vvpat for all voters":            "dre_without_vvpat",
    "hand counted paper ballots":                   "hand_count",
    "optical scan":                                 "optical_scan",
}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _api_url(year: int, fmt: str = "csv") -> str:
    return f"{_API_BASE}?advanced&year={year}&download={fmt}"


def _fetch_csv(year: int) -> Tuple[List[str], List[Dict[str, str]], str]:
    """Download CSV for *year* from the Verifier API.

    Returns (headers, rows_as_dicts, source_url).
    """
    url = _api_url(year, "csv")
    logger.info({"level": "INFO", "type": "import", "message": f"[VotingEquipment] Fetching year={year} from {url}"})
    with urlopen(url, timeout=30) as resp:  # noqa: S310 — url is constructed internally
        raw = resp.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(raw))
    headers = reader.fieldnames or []
    rows = list(reader)
    return list(headers), rows, url


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _normalise_method(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return _METHOD_NORMALISE.get(raw.strip().lower(), raw.strip().lower())


def _build_record(row: Dict[str, str], year: int, source_url: str) -> Dict[str, Any]:
    """Map a raw CSV row to CountyEquipment column values."""
    mapped: Dict[str, Any] = {
        "year": year,
        "api_year": year,
        "source_url": source_url,
        "raw_api_row": json.dumps(row, ensure_ascii=False),
        "imported_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    for csv_col, attr in _FIELD_MAP.items():
        val = row.get(csv_col, "").strip() or None
        mapped[attr] = val

    # Normalise voting method vocabulary
    if "std_voting_method" in mapped:
        mapped["std_voting_method"] = _normalise_method(mapped["std_voting_method"])

    return mapped


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def _upsert_year(session: Any, records: List[Dict[str, Any]], *, replace: bool) -> int:
    """Insert or update records for a single year.

    When replace=True, existing rows for that year are deleted first (full refresh).
    Otherwise existing rows are skipped (idempotent append).
    """
    # Import here to avoid circular imports at module level
    from webapp.parser.models.election_data import CountyEquipment  # noqa: PLC0415

    year = records[0]["year"] if records else None

    if replace and year is not None:
        deleted = session.query(CountyEquipment).filter_by(year=year).delete()
        logger.info({"level": "INFO", "type": "import", "message": f"[VotingEquipment] Deleted {deleted} existing rows for year={year}"})

    inserted = 0
    for rec in records:
        if not replace:
            existing = (
                session.query(CountyEquipment)
                .filter_by(year=rec["year"], state=rec.get("state"), county=rec.get("county"))
                .first()
            )
            if existing:
                continue
        obj = CountyEquipment(**{k: v for k, v in rec.items() if k != "imported_at" or not replace})
        session.add(obj)
        inserted += 1

    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import Verified Voting county equipment data into county_equipment table"
    )
    year_group = parser.add_mutually_exclusive_group(required=True)
    year_group.add_argument("--year", type=int, help="Single election year to import")
    year_group.add_argument(
        "--years", type=int, nargs="+",
        metavar="YEAR",
        help="Space-separated list of years, e.g. --years 2012 2016 2020 2024",
    )
    year_group.add_argument(
        "--all-years", action="store_true",
        help=f"Import all available years: {_ALL_YEARS}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but do not write to DB")
    parser.add_argument("--show-headers", action="store_true", help="Print raw CSV headers then exit (use with --dry-run)")
    parser.add_argument("--replace", action="store_true", help="Delete existing rows for the year before inserting")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    if args.all_years:
        years = _ALL_YEARS
    elif args.years:
        years = args.years
    else:
        years = [args.year]

    total_inserted = 0

    for year in years:
        try:
            headers, rows, source_url = _fetch_csv(year)
        except Exception as exc:
            logger.error({"level": "ERROR", "type": "import", "message": f"[VotingEquipment] Fetch failed year={year}: {exc}"})
            continue

        if args.show_headers:
            print(f"\nyear={year} headers ({len(headers)}):")
            for h in headers:
                print(f"  {h!r}")
            if args.dry_run:
                return 0

        records = [_build_record(row, year, source_url) for row in rows]
        print(f"year={year}: {len(rows)} rows fetched → {len(records)} records built")

        if args.dry_run:
            if records:
                sample = dict(records[0])
                sample.pop("raw_api_row", None)
                print(f"  Sample record: {sample}")
            continue

        session = SessionLocal()
        try:
            inserted = _upsert_year(session, records, replace=args.replace)
            session.commit()
            total_inserted += inserted
            print(f"  Inserted {inserted} rows for year={year}")
        except Exception as exc:
            session.rollback()
            logger.error({"level": "ERROR", "type": "import", "message": f"[VotingEquipment] DB write failed year={year}: {exc}"})
        finally:
            session.close()

    if not args.dry_run:
        print(f"\nTotal inserted: {total_inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
