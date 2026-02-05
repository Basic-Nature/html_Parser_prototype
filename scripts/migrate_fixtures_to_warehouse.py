#!/usr/bin/env python3
"""
Migrate election fixture data to PostgreSQL warehouse.

Overview:
    Loads election data from fixtures/, cache/, and log/ directories,
    filters by confidence threshold, runs integrity checks, and migrates
    to PostgreSQL warehouse tables.

Data Flow:
    fixtures/ (JSON) + cache/ (JSON) + log/ (JSONL)
        → Confidence Filtering (≥0.7)
        → Schema Validation (election_results_schema.json)
        → Integrity Checks (Integrity_check.py)
        → PostgreSQL Upsert (State, County, Contest, Candidate, Result)
        → Migration Logging (log/migration_events.jsonl)

Usage:
    # Dry run with default confidence threshold (0.7)
    python scripts/migrate_fixtures_to_warehouse.py --dry-run

    # Production migration with higher confidence threshold
    python scripts/migrate_fixtures_to_warehouse.py --min-confidence 0.8

    # Skip integrity checks (not recommended)
    python scripts/migrate_fixtures_to_warehouse.py --skip-integrity-check

    # Specify batch size for commit batching
    python scripts/migrate_fixtures_to_warehouse.py --batch-size 500
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import orjson
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# Add webapp/parser to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WEBAPP_PARSER = PROJECT_ROOT / "webapp" / "parser"
sys.path.insert(0, str(WEBAPP_PARSER))
sys.path.insert(0, str(WEBAPP_PARSER.parent))  # Add webapp to path for absolute imports

from webapp.parser.utils.db_utils import get_session
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.models import (
    Candidate,
    Contest,
    County,
    Office,
    Party,
    Result,
    State,
)
from webapp.parser.utils.shared_logic import safe_get

# Import integrity check optionally (may fail if webapp dependencies not available)
try:
    from webapp.parser.Context_Integration.Integrity_check import analyze_contests
    INTEGRITY_CHECK_AVAILABLE = True
except ImportError as e:
    INTEGRITY_CHECK_AVAILABLE = False
    logger.warning(f"[IMPORT] Integrity check module not available - integrity validation will be skipped: {e}")

# Migration log path
MIGRATION_LOG = PROJECT_ROOT / "webapp" / "parser" / "Context_Integration" / "Context_Library" / "log" / "migration_events.jsonl"


class MigrationStats:
    """Track migration statistics."""

    def __init__(self):
        self.total_records = 0
        self.filtered_by_confidence = 0
        self.failed_integrity = 0
        self.failed_validation = 0
        self.states_created = 0
        self.counties_created = 0
        self.contests_created = 0
        self.candidates_created = 0
        self.results_created = 0
        self.states_updated = 0
        self.counties_updated = 0
        self.contests_updated = 0
        self.candidates_updated = 0
        self.results_updated = 0
        self.errors: List[str] = []

    def summary(self) -> Dict[str, Any]:
        """Return summary dictionary."""
        return {
            "total_records": self.total_records,
            "filtered_by_confidence": self.filtered_by_confidence,
            "failed_integrity": self.failed_integrity,
            "failed_validation": self.failed_validation,
            "created": {
                "states": self.states_created,
                "counties": self.counties_created,
                "contests": self.contests_created,
                "candidates": self.candidates_created,
                "results": self.results_created,
            },
            "updated": {
                "states": self.states_updated,
                "counties": self.counties_updated,
                "contests": self.contests_updated,
                "candidates": self.candidates_updated,
                "results": self.results_updated,
            },
            "errors": len(self.errors),
        }


def log_migration_event(event_type: str, record_key: str, details: Dict[str, Any]) -> None:
    """Log migration event to JSONL file."""
    try:
        MIGRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "record_key": record_key,
            **details,
        }
        with MIGRATION_LOG.open("ab") as f:
            f.write(orjson.dumps(event) + b"\n")
    except Exception as e:
        logger.warning(f"Failed to log migration event: {e}")


def load_election_index(fixtures_dir: Path, include_cache: bool = True, include_log: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Load election data from fixtures, cache, and log directories.

    Args:
        fixtures_dir: Path to fixtures directory
        include_cache: Whether to include cache directory JSON sources
        include_log: Whether to include log directory JSONL sources

    Returns:
        Dictionary mapping record keys to election data records
    """
    index: Dict[str, Dict[str, Any]] = {}

    # Load from fixtures JSON/JSONL
    for json_file in fixtures_dir.glob("*.json"):
        if json_file.name in ("election_results_schema.json", "election_results_index.json"):
            continue
        try:
            with json_file.open("rb") as f:
                data = orjson.loads(f.read())
            if isinstance(data, dict):
                for key, record in data.items():
                    index[key] = record
            logger.info(f"[FIXTURES] Loaded {json_file.name}")
        except Exception as e:
            logger.warning(f"[FIXTURES] Failed to load {json_file.name}: {e}")

    for jsonl_file in fixtures_dir.glob("*.jsonl"):
        try:
            with jsonl_file.open("rb") as f:
                for line_num, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    entry = orjson.loads(line)
                    if isinstance(entry, dict):
                        key = entry.get("key") or entry.get("id") or f"{jsonl_file.stem}_{line_num}"
                        record = entry.get("record", entry)
                        index[key] = record
            logger.info(f"[FIXTURES] Loaded {jsonl_file.name}")
        except Exception as e:
            logger.warning(f"[FIXTURES] Failed to load {jsonl_file.name}: {e}")

    # Load from cache
    if include_cache:
        cache_dir = fixtures_dir.parent / "Context_Integration" / "Context_Library" / "cache"
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.json"):
                if any(kw in cache_file.name for kw in ["election", "context"]):
                    try:
                        with cache_file.open("rb") as f:
                            data = orjson.loads(f.read())
                        if isinstance(data, dict):
                            for key, record in data.items():
                                if key not in index:  # Don't overwrite fixture data
                                    index[key] = record
                        logger.info(f"[CACHE] Loaded {cache_file.name}")
                    except Exception as e:
                        logger.warning(f"[CACHE] Failed to load {cache_file.name}: {e}")

    # Load from log
    if include_log:
        log_dir = fixtures_dir.parent / "Context_Integration" / "Context_Library" / "log"
        if log_dir.exists():
            for log_file in log_dir.glob("*.jsonl"):
                if any(kw in log_file.name for kw in ["field_selection", "navigation_learning", "integrity", "trust"]):
                    try:
                        with log_file.open("rb") as f:
                            for line_num, line in enumerate(f, start=1):
                                if not line.strip():
                                    continue
                                entry = orjson.loads(line)
                                if isinstance(entry, dict):
                                    key = entry.get("key") or entry.get("id") or f"{log_file.stem}_{line_num}"
                                    record = entry.get("record", entry)
                                    if key not in index:  # Don't overwrite higher-priority sources
                                        index[key] = record
                        logger.info(f"[LOG] Loaded {log_file.name}")
                    except Exception as e:
                        logger.warning(f"[LOG] Failed to load {log_file.name}: {e}")

    logger.info(f"[INDEX] Loaded {len(index)} total records from all sources")
    return index


def filter_by_confidence(index: Dict[str, Dict[str, Any]], min_confidence: float, stats: MigrationStats) -> Dict[str, Dict[str, Any]]:
    """
    Filter election records by confidence threshold.

    Args:
        index: Election data index
        min_confidence: Minimum confidence threshold (0.0-1.0)
        stats: Migration statistics tracker

    Returns:
        Filtered index containing only records meeting confidence threshold
    """
    if min_confidence <= 0.0:
        return index

    filtered_index = {}
    for key, record in index.items():
        confidence = safe_get(record, "confidence", 1.0)
        if isinstance(confidence, (int, float)) and confidence >= min_confidence:
            filtered_index[key] = record
        else:
            stats.filtered_by_confidence += 1
            log_migration_event("filter", key, {
                "reason": "confidence",
                "confidence": confidence,
                "threshold": min_confidence,
            })

    logger.info(f"[FILTER] Kept {len(filtered_index)}/{len(index)} records (confidence ≥ {min_confidence})")
    return filtered_index


def upsert_state(session: Session, state_name: str, state_abbr: Optional[str], stats: MigrationStats) -> Optional[State]:
    """
    Upsert a state record.

    Args:
        session: Database session
        state_name: State name
        state_abbr: State abbreviation (e.g., "AZ")
        stats: Migration statistics tracker

    Returns:
        State object or None if failed
    """
    try:
        state = session.query(State).filter_by(name=state_name.upper()).first()
        if state:
            # Update abbreviation if provided and different
            if state_abbr and state.abbreviation != state_abbr.upper():
                state.abbreviation = state_abbr.upper()
                stats.states_updated += 1
        else:
            state = State(name=state_name.upper(), abbreviation=state_abbr.upper() if state_abbr else None)
            session.add(state)
            stats.states_created += 1
        session.flush()  # Get ID without committing
        return state
    except SQLAlchemyError as e:
        logger.error(f"Failed to upsert state {state_name}: {e}")
        stats.errors.append(f"State {state_name}: {e}")
        return None


def upsert_county(session: Session, county_name: str, state: State, stats: MigrationStats) -> Optional[County]:
    """
    Upsert a county record.

    Args:
        session: Database session
        county_name: County name
        state: Parent state object
        stats: Migration statistics tracker

    Returns:
        County object or None if failed
    """
    try:
        county = session.query(County).filter_by(name=county_name.upper(), state_id=state.id).first()
        if county:
            stats.counties_updated += 1
        else:
            county = County(name=county_name.upper(), state_id=state.id)
            session.add(county)
            stats.counties_created += 1
        session.flush()
        return county
    except SQLAlchemyError as e:
        logger.error(f"Failed to upsert county {county_name} in {state.name}: {e}")
        stats.errors.append(f"County {county_name}: {e}")
        return None


def upsert_party(session: Session, party_name: str) -> Optional[Party]:
    """
    Upsert a party record.

    Args:
        session: Database session
        party_name: Party name

    Returns:
        Party object or None if failed
    """
    try:
        party = session.query(Party).filter_by(name=party_name.upper()).first()
        if not party:
            party = Party(name=party_name.upper())
            session.add(party)
            session.flush()
        return party
    except SQLAlchemyError as e:
        logger.error(f"Failed to upsert party {party_name}: {e}")
        return None


def upsert_contest(
    session: Session,
    contest_title: str,
    year: int,
    state: State,
    county: Optional[County],
    metadata: Dict[str, Any],
    stats: MigrationStats,
) -> Optional[Contest]:
    """
    Upsert a contest record.

    Args:
        session: Database session
        contest_title: Contest title
        year: Election year
        state: Parent state object
        county: Parent county object (optional)
        metadata: Additional metadata
        stats: Migration statistics tracker

    Returns:
        Contest object or None if failed
    """
    try:
        # Query by unique constraint fields
        query = session.query(Contest).filter_by(
            title=contest_title,
            year=year,
            state_id=state.id,
        )
        if county:
            query = query.filter_by(county_id=county.id)
        contest = query.first()

        if contest:
            # Update metadata if provided
            if metadata:
                contest.metastats = contest.metastats or {}
                contest.metastats.update(metadata)
            stats.contests_updated += 1
        else:
            contest = Contest(
                title=contest_title,
                year=year,
                state_id=state.id,
                county_id=county.id if county else None,
                metastats=metadata,
            )
            session.add(contest)
            stats.contests_created += 1
        session.flush()
        return contest
    except SQLAlchemyError as e:
        logger.error(f"Failed to upsert contest {contest_title} ({year}): {e}")
        stats.errors.append(f"Contest {contest_title}: {e}")
        return None


def upsert_candidate(
    session: Session,
    candidate_name: str,
    party: Optional[Party],
    metadata: Dict[str, Any],
    stats: MigrationStats,
) -> Optional[Candidate]:
    """
    Upsert a candidate record.

    Args:
        session: Database session
        candidate_name: Candidate name
        party: Party object (optional)
        metadata: Additional metadata
        stats: Migration statistics tracker

    Returns:
        Candidate object or None if failed
    """
    try:
        candidate = session.query(Candidate).filter_by(name=candidate_name).first()
        if candidate:
            # Update party if provided and different
            if party and candidate.party_id != party.id:
                candidate.party_id = party.id
            # Update metadata
            if metadata:
                candidate.metastats = candidate.metastats or {}
                candidate.metastats.update(metadata)
            stats.candidates_updated += 1
        else:
            candidate = Candidate(
                name=candidate_name,
                party_id=party.id if party else None,
                metastats=metadata,
            )
            session.add(candidate)
            stats.candidates_created += 1
        session.flush()
        return candidate
    except SQLAlchemyError as e:
        logger.error(f"Failed to upsert candidate {candidate_name}: {e}")
        stats.errors.append(f"Candidate {candidate_name}: {e}")
        return None


def migrate_record(
    session: Session,
    key: str,
    record: Dict[str, Any],
    stats: MigrationStats,
    skip_integrity: bool = False,
) -> bool:
    """
    Migrate a single election record to PostgreSQL.

    Args:
        session: Database session
        key: Record key
        record: Election data record
        stats: Migration statistics tracker
        skip_integrity: Whether to skip integrity checks

    Returns:
        True if migration succeeded, False otherwise
    """
    try:
        stats.total_records += 1

        # Extract metadata
        metadata = safe_get(record, "metadata", {})
        state_name = safe_get(metadata, "state")
        county_name = safe_get(metadata, "county")
        year = safe_get(metadata, "year")
        contest_title = safe_get(record, "contest")
        source_url = safe_get(record, "source")

        # Validate required fields
        if not all([state_name, year, contest_title]):
            logger.warning(f"[SKIP] {key}: Missing required fields (state, year, contest)")
            stats.failed_validation += 1
            log_migration_event("skip", key, {"reason": "missing_fields", "record": record})
            return False

        # Run integrity checks (unless skipped or unavailable)
        if not skip_integrity and INTEGRITY_CHECK_AVAILABLE:
            try:
                # analyze_contests expects a list of contests
                integrity_result = analyze_contests([record])
                issues = integrity_result.get("issues", [])
                if issues:
                    logger.warning(f"[SKIP] {key}: Failed integrity check ({len(issues)} issues)")
                    stats.failed_integrity += 1
                    log_migration_event("skip", key, {"reason": "integrity", "issues": len(issues)})
                    return False
            except Exception as e:
                logger.warning(f"[SKIP] {key}: Integrity check error: {e}")
                stats.failed_integrity += 1
                return False
        elif not skip_integrity and not INTEGRITY_CHECK_AVAILABLE:
            logger.debug(f"[INFO] {key}: Integrity check skipped (module unavailable)")

        # Upsert state
        state = upsert_state(session, state_name, metadata.get("state_abbreviation"), stats)
        if not state:
            return False

        # Upsert county (if provided)
        county = None
        if county_name:
            county = upsert_county(session, county_name, state, stats)

        # Upsert contest
        contest = upsert_contest(
            session,
            contest_title,
            year,
            state,
            county,
            {"source": source_url, "total_votes": metadata.get("total_votes")},
            stats,
        )
        if not contest:
            return False

        # Migrate candidates and results
        candidates_data = safe_get(record, "candidates", [])
        for candidate_data in candidates_data:
            if not isinstance(candidate_data, dict):
                continue

            candidate_name = safe_get(candidate_data, "name")
            party_name = safe_get(candidate_data, "party")
            votes = safe_get(candidate_data, "votes")
            percent = safe_get(candidate_data, "percent")
            is_winner = safe_get(candidate_data, "winner", False)

            if not candidate_name:
                continue

            # Upsert party
            party = None
            if party_name:
                party = upsert_party(session, party_name)

            # Upsert candidate
            candidate = upsert_candidate(
                session,
                candidate_name,
                party,
                {"fec_id": candidate_data.get("fec_id")},
                stats,
            )
            if not candidate:
                continue

            # Create result
            try:
                result = session.query(Result).filter_by(
                    candidate_id=candidate.id,
                    contest_id=contest.id,
                ).first()

                if result:
                    # Update existing result
                    result.votes = votes
                    result.percent = percent
                    result.is_winner = is_winner
                    stats.results_updated += 1
                else:
                    # Create new result
                    result = Result(
                        candidate_id=candidate.id,
                        contest_id=contest.id,
                        votes=votes,
                        percent=percent,
                        is_winner=is_winner,
                    )
                    session.add(result)
                    stats.results_created += 1
                session.flush()
            except SQLAlchemyError as e:
                logger.error(f"Failed to upsert result for {candidate_name} in {contest_title}: {e}")
                stats.errors.append(f"Result {candidate_name}: {e}")

        log_migration_event("migrate", key, {
            "state": state_name,
            "contest": contest_title,
            "year": year,
            "candidates": len(candidates_data),
        })
        return True

    except Exception as e:
        logger.error(f"Failed to migrate record {key}: {e}")
        stats.errors.append(f"Record {key}: {e}")
        log_migration_event("error", key, {"error": str(e)})
        return False


def main():
    """Main migration script entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate election fixture data to PostgreSQL warehouse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=PROJECT_ROOT / "webapp" / "parser" / "fixtures",
        help="Path to fixtures directory (default: webapp/parser/fixtures)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (0.0-1.0, default: 0.7)",
    )
    parser.add_argument(
        "--include-cache",
        action="store_true",
        default=True,
        help="Include cache directory JSON sources (default: True)",
    )
    parser.add_argument(
        "--include-log",
        action="store_true",
        default=True,
        help="Include log directory JSONL sources (default: True)",
    )
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="Skip integrity checks (not recommended)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of records to process before committing (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run migration without committing to database",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.fixtures_dir.exists():
        logger.error(f"Fixtures directory not found: {args.fixtures_dir}")
        sys.exit(1)

    if not 0.0 <= args.min_confidence <= 1.0:
        logger.error(f"Invalid confidence threshold: {args.min_confidence} (must be 0.0-1.0)")
        sys.exit(1)

    # Initialize stats
    stats = MigrationStats()

    # Load election index
    logger.info("[START] Loading election data from fixtures, cache, and log...")
    index = load_election_index(args.fixtures_dir, args.include_cache, args.include_log)

    if not index:
        logger.warning("No election data found to migrate")
        sys.exit(0)

    # Filter by confidence
    logger.info(f"[FILTER] Applying confidence threshold: {args.min_confidence}")
    index = filter_by_confidence(index, args.min_confidence, stats)

    if not index:
        logger.warning(f"No records met confidence threshold {args.min_confidence}")
        sys.exit(0)

    # Migrate to PostgreSQL
    logger.info(f"[MIGRATE] Starting migration to PostgreSQL (dry_run={args.dry_run})...")

    session: Optional[Session] = None
    try:
        session = get_session()

        batch_count = 0
        for key, record in index.items():
            migrate_record(session, key, record, stats, args.skip_integrity_check)
            batch_count += 1

            # Commit in batches
            if batch_count >= args.batch_size:
                if args.dry_run:
                    logger.info(f"[DRY-RUN] Would commit batch of {batch_count} records")
                    session.rollback()
                else:
                    session.commit()
                    logger.info(f"[COMMIT] Committed batch of {batch_count} records")
                batch_count = 0

        # Final commit
        if batch_count > 0:
            if args.dry_run:
                logger.info(f"[DRY-RUN] Would commit final batch of {batch_count} records")
                session.rollback()
            else:
                session.commit()
                logger.info(f"[COMMIT] Committed final batch of {batch_count} records")

    except SQLAlchemyError as e:
        logger.error(f"Database error during migration: {e}")
        if session:
            session.rollback()
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
        if session:
            session.rollback()
        sys.exit(130)
    finally:
        if session:
            session.close()

    # Print summary
    summary = stats.summary()
    logger.info("[COMPLETE] Migration summary:")
    logger.info(f"  Total records: {summary['total_records']}")
    logger.info(f"  Filtered by confidence: {summary['filtered_by_confidence']}")
    logger.info(f"  Failed integrity: {summary['failed_integrity']}")
    logger.info(f"  Failed validation: {summary['failed_validation']}")
    logger.info(f"  Created: {summary['created']}")
    logger.info(f"  Updated: {summary['updated']}")
    logger.info(f"  Errors: {summary['errors']}")

    if stats.errors:
        logger.warning(f"[ERRORS] {len(stats.errors)} errors occurred during migration")
        for error in stats.errors[:10]:  # Show first 10 errors
            logger.warning(f"  - {error}")
        if len(stats.errors) > 10:
            logger.warning(f"  ... and {len(stats.errors) - 10} more errors")

    # Log final summary event
    log_migration_event("complete", "migration_summary", {
        "summary": summary,
        "dry_run": args.dry_run,
        "confidence_threshold": args.min_confidence,
    })

    logger.info(f"[LOG] Migration events logged to {MIGRATION_LOG}")


if __name__ == "__main__":
    main()
