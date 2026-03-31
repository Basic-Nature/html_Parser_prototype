"""
Backfill missing midterm election cycles into SMART Elections Database-Lite.

Current data coverage gap (confirmed 2026-03-30 audit):
    Finalized Data sheet:            2012, 2016, 2020, 2024  ← presidential only
    Down-Ballot Calculations sheet:  2012, 2016, 2020, 2024  ← presidential only
    Missing midterm cycles:          2014, 2018, 2022

Why this matters:
    - Down-ballot dropoff analysis is only valid across presidential cycles without midterms.
    - Senate, Governor, and US House races appear in midterms, not presidential cycles.
    - Without 2014/2018/2022, aggregate dropoff percentages overstate presidential participation.

Recommended data source — MIT Election Data and Science Lab (MEDSL):
    County-level results (President, Senate, Governor, House):
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ

    Annual CSV files: countypres_20XX.csv
    Columns: year, state, state_po, county_name, county_fips, office, candidate,
             party, candidatevotes, totalvotes, mode (in-person/mail/etc.)

Alternative: OpenElections project (individual state parsers, less uniform):
    https://openelections.net/

Usage:
    # Step 1 — check current DB coverage
    python scripts/backfill_midterm_elections.py --check-coverage

    # Step 2 — dry-run a single year from a downloaded MEDSL CSV
    python scripts/backfill_midterm_elections.py --year 2022 --src path/to/countypres_2022.csv --dry-run

    # Step 3 — import (writes to staging_records table, same tier as import_database_lite.py)
    python scripts/backfill_midterm_elections.py --year 2022 --src path/to/countypres_2022.csv

    # Step 4 — import all three missing midterm years at once (requires all three files)
    python scripts/backfill_midterm_elections.py --backfill-all --src-dir path/to/medsl_csvs/

TODO:
    1. Confirm MEDSL column names against a downloaded file (run with --show-headers).
    2. Map MEDSL party codes to SMART Elections FEC party codes (DEM, REP, LIB, GRE, etc.).
    3. Build candidate-name standardisation from FEC ID lookup (see webapp/parser/fec_lookup.py).
    4. Decide whether to push backfilled rows to the Google Sheet or direct-insert to DB only.
    5. Add Down-Ballot Calculations rows from imported midterm office results
       (senate/governor midterm races won't have a presidential dropoff counterpart — needs
       a separate dropoff baseline, e.g. prior midterm or same-cycle total-vote baseline).
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from webapp.parser.utils.db_utils import SessionLocal
from webapp.parser.utils.logger_singleton import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MISSING_YEARS = [2014, 2018, 2022]

# TODO: confirm against actual MEDSL CSV download (--show-headers)
_MEDSL_FIELD_MAP: Dict[str, str] = {
    "year":             "year",
    "state":            "state",
    "state_po":         "state_abbr",
    "county_name":      "county",
    "county_fips":      "county_fips",
    "office":           "office",
    "candidate":        "ballot_candidate_name",
    "party":            "ballot_party",
    "candidatevotes":   "total_votes",
    "totalvotes":       "total_contest_votes",
    "mode":             "vote_mode",            # TOTAL | ELECTION DAY | MAIL
}

# FEC party-code normalisation — extend as needed
_PARTY_MAP: Dict[str, str] = {
    "DEMOCRAT":     "DEM",
    "DEMOCRATIC":   "DEM",
    "REPUBLICAN":   "REP",
    "LIBERTARIAN":  "LIB",
    "GREEN":        "GRE",
    "OTHER":        "OTH",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_party(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    return _PARTY_MAP.get(raw.strip().upper(), raw.strip().upper())


def _coerce_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _read_medsl_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


def _build_staging_record(row: Dict[str, str], year: int) -> Dict[str, Any]:
    """Map a MEDSL row to a StagingRecord-compatible dict."""
    mapped: Dict[str, Any] = {"year": year}
    for csv_col, attr in _MEDSL_FIELD_MAP.items():
        mapped[attr] = row.get(csv_col, "").strip() or None

    # Normalise party
    mapped["party"] = _normalise_party(mapped.get("ballot_party"))

    # Votes
    mapped["total_votes"] = str(_coerce_int(mapped.get("total_votes")) or 0)

    # Timestamps
    mapped["ingested_at"] = datetime.now(timezone.utc)
    mapped["is_processed"] = False
    mapped["source_data_url"] = "medsl_backfill"
    return mapped


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def _check_coverage(session: Any) -> None:
    """Print year coverage of staging_records and election_results tables."""
    from sqlalchemy import text  # noqa: PLC0415

    for table in ("staging_records", "election_results"):
        try:
            result = session.execute(
                text(f"SELECT year, COUNT(*) AS cnt FROM {table} GROUP BY year ORDER BY year")  # noqa: S608
            ).fetchall()
            print(f"\n{table} year coverage:")
            if result:
                for row in result:
                    print(f"  {row[0]}: {row[1]:,} rows")
            else:
                print("  (empty)")
        except Exception as exc:
            print(f"  {table}: could not query — {exc}")

    missing_in_db = MISSING_YEARS  # TODO: compute from actual coverage above
    print(f"\nMissing midterm years (expected): {missing_in_db}")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def _import_year(session: Any, rows: List[Dict[str, str]], year: int, *, dry_run: bool) -> int:
    from webapp.parser.models.election_data import StagingRecord  # noqa: PLC0415

    records = [_build_staging_record(r, year) for r in rows]

    # Filter rows that carry actual vote data (skip mode != TOTAL if present)
    mode_col = "vote_mode"
    records = [
        r for r in records
        if (r.get(mode_col) or "TOTAL").upper() in ("TOTAL", "")
    ]

    print(f"  year={year}: {len(rows)} raw rows → {len(records)} TOTAL-mode records to stage")

    if dry_run:
        if records:
            sample = {k: v for k, v in records[0].items() if k not in ("ingested_at",)}
            print(f"  Sample: {sample}")
        return len(records)

    inserted = 0
    for rec in records:
        # Use only columns present on StagingRecord
        valid_cols = {c.key for c in StagingRecord.__table__.columns}
        filtered = {k: v for k, v in rec.items() if k in valid_cols}
        session.add(StagingRecord(**filtered))
        inserted += 1

    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill missing midterm election cycles into staging_records")
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--src", metavar="FILE", help="Path to a downloaded MEDSL county CSV")
    src_group.add_argument("--src-dir", metavar="DIR", help="Directory containing MEDSL CSVs named countypres_YYYY.csv")
    parser.add_argument("--year", type=int, help="Election year to import (with --src)")
    parser.add_argument("--backfill-all", action="store_true", help=f"Import all missing years {MISSING_YEARS} from --src-dir")
    parser.add_argument("--check-coverage", action="store_true", help="Print current DB year coverage then exit")
    parser.add_argument("--show-headers", action="store_true", help="Print CSV headers from --src then exit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    session = SessionLocal()

    if args.check_coverage:
        _check_coverage(session)
        session.close()
        return 0

    # Resolve year → file pairs
    pairs: List[tuple[int, str]] = []
    if args.backfill_all:
        if not args.src_dir:
            raise SystemExit("--backfill-all requires --src-dir")
        for year in MISSING_YEARS:
            candidate = Path(args.src_dir) / f"countypres_{year}.csv"
            if candidate.exists():
                pairs.append((year, str(candidate)))
            else:
                print(f"  WARN: {candidate} not found — skipping year {year}")
    elif args.src:
        if not args.year:
            raise SystemExit("--src requires --year")
        pairs.append((args.year, args.src))
    else:
        raise SystemExit("Provide --src FILE --year YYYY  or  --backfill-all --src-dir DIR  or  --check-coverage")

    total = 0
    for year, path in pairs:
        headers, rows = _read_medsl_csv(path)
        print(f"\nyear={year}: {path} — {len(rows)} rows, {len(headers)} columns")

        if args.show_headers:
            print(f"  Headers: {headers}")
            continue

        count = _import_year(session, rows, year, dry_run=args.dry_run)
        total += count

    if not args.dry_run and pairs and not args.show_headers:
        try:
            session.commit()
            print(f"\nCommitted {total} staging records across {len(pairs)} year(s)")
        except Exception as exc:
            session.rollback()
            logger.error({"level": "ERROR", "type": "import", "message": f"[MidtermBackfill] Commit failed: {exc}"})
            raise
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
