from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import orjson
from webapp.parser.config import OUTPUT_DIR
from webapp.parser.Context_Integration.librarian import clean_for_json
from webapp.parser.utils.db_utils import (
    create_batch_metadata,
    create_warehouse_election_result,
    update_batch_metadata,
)
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.models import StatusEnum
from webapp.parser.health.promotion_helpers import check_exact_duplicate, get_url_verification_tier


PRECINCT_HINTS: tuple[str, ...] = (
    "precinct",
    "ward",
    "district",
    "division",
    "jurisdiction",
    "county",
)
CANDIDATE_HINTS: tuple[str, ...] = (
    "ballot candidate",
    "candidate",
    "choice",
    "nominee",
)
PARTY_HINTS: tuple[str, ...] = (
    "ballot party",
    "party",
    "affiliation",
)
VOTE_HINTS: tuple[str, ...] = (
    "calculated total votes",
    "total votes",
    "votes",
    "vote total",
    "vote count",
    "reported votes",
    "ballots",
    "total",
)
PREFERRED_HEADERS = {
    "precinct": ("Precinct", "Reporting Unit"),
    "candidate": ("Ballot Candidate Name", "Candidate", "Candidate Name"),
    "party": ("Ballot Party", "Party"),
    "votes": ("Calculated Total Votes", "Votes", "Total Votes"),
}
DATASET_METADATA = "results.metadata.json"
DATASET_CSV = "results.csv"

__all__ = [
    "discover_dataset_dirs",
    "resolve_dataset_path",
    "build_warehouse_records",
    "promote_dataset",
]


def discover_dataset_dirs(root: Path) -> list[Path]:
    """Return dataset directories that contain parser output."""
    if not root.exists():
        return []
    candidates: list[Path] = []
    for entry in root.iterdir():
        if entry.is_dir() and (entry / DATASET_METADATA).exists() and (entry / DATASET_CSV).exists():
            candidates.append(entry)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def resolve_dataset_path(dataset: str | Path | None, root: Path) -> Path:
    """Resolve a dataset folder either explicitly or by picking the newest output."""
    if dataset:
        candidate = Path(dataset)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {candidate}")
        return candidate
    discovered = discover_dataset_dirs(root)
    if not discovered:
        raise FileNotFoundError(f"No dataset folders with {DATASET_METADATA} under {root}")
    return discovered[0]


def _load_metadata(dataset_dir: Path) -> dict[str, Any]:
    metadata_path = dataset_dir / DATASET_METADATA
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file at {metadata_path}")
    return orjson.loads(metadata_path.read_bytes())


def _load_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    csv_path = dataset_dir / DATASET_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV output at {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _match_field(
    row: dict[str, Any],
    preferred: Sequence[str],
    hints: Iterable[str],
) -> Any:
    lower_map = {header.lower(): header for header in row.keys() if isinstance(header, str)}
    for exact in preferred:
        header = lower_map.get(exact.lower())
        if header and _has_value(row.get(header)):
            return row.get(header)
    for header, value in row.items():
        if not isinstance(header, str) or not _has_value(value):
            continue
        header_norm = header.strip().lower()
        if any(hint in header_norm for hint in hints):
            return value
    return None


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_votes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        return int(round(value))
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace(",", "").replace("_", "")
    lowered = normalized.lower()
    if lowered in {"na", "n/a", "null", "none", "--", "-", "n"}:
        return None
    try:
        if "." in normalized:
            return int(round(float(normalized)))
        return int(normalized)
    except ValueError:
        return None


def _resolve_election_date(metadata: dict[str, Any]) -> datetime | None:
    context = metadata.get("context") or {}
    candidates = [
        context.get("results_timestamp"),
        context.get("results_reported_at"),
        context.get("contest_date"),
        metadata.get("created_at"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if isinstance(candidate, datetime):
            return candidate if candidate.tzinfo else candidate.replace(tzinfo=timezone.utc)
        text = str(candidate).strip()
        if not text:
            continue
        text = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def build_warehouse_records(
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    dataset_label: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Transform CSV rows into WarehouseElectionResult payloads."""
    context = metadata.get("context") or {}
    contest = _coerce_text(metadata.get("contest")) or _coerce_text(context.get("contest")) or _coerce_text(context.get("race"))
    state = _coerce_text(metadata.get("state")) or _coerce_text(context.get("state")) or _coerce_text(context.get("state_original"))
    county = _coerce_text(metadata.get("county")) or _coerce_text(context.get("county")) or _coerce_text(context.get("county_original"))
    contest = contest or "Unknown Contest"
    state = state or "Unknown"
    county = county or "Unknown"
    dataset_name = dataset_label or metadata.get("output_base_name") or contest
    election_date = _resolve_election_date(metadata)
    records: list[dict[str, Any]] = []
    skipped = 0
    for idx, row in enumerate(rows):
        if limit is not None and limit > 0 and len(records) >= limit:
            break
        candidate_val = _match_field(row, PREFERRED_HEADERS["candidate"], CANDIDATE_HINTS)
        party_val = _match_field(row, PREFERRED_HEADERS["party"], PARTY_HINTS)
        precinct_val = _match_field(row, PREFERRED_HEADERS["precinct"], PRECINCT_HINTS)
        votes_val = _match_field(row, PREFERRED_HEADERS["votes"], VOTE_HINTS)
        votes = _coerce_votes(votes_val)
        if votes is None:
            skipped += 1
            continue
        record = {
            "state": state,
            "county": county,
            "contest": contest,
            "candidate": _coerce_text(candidate_val) or "Unknown Candidate",
            "party": _coerce_text(party_val),
            "votes": votes,
            "precinct": _coerce_text(precinct_val) or "All Precincts",
            "election_date": election_date,
            "metastats": {
                "dataset": dataset_name,
                "row_index": idx,
                "source_row": clean_for_json(row),
            },
        }
        records.append(record)
    return records, skipped


def promote_dataset(
    dataset_dir: Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Promote a dataset folder into warehouse_election_results."""
    metadata = _load_metadata(dataset_dir)
    rows = _load_rows(dataset_dir)
    payloads, skipped = build_warehouse_records(
        metadata,
        rows,
        dataset_label=metadata.get("output_base_name") or dataset_dir.name,
        limit=limit,
    )
    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "dataset_label": metadata.get("output_base_name") or dataset_dir.name,
        "records_prepared": len(payloads),
        "skipped_rows": skipped,
        "dry_run": dry_run,
    }
    if not payloads:
        print("[PROMOTE] No valid rows found; nothing to insert.")
        return summary
    if dry_run:
        print(f"[PROMOTE] Dry-run: would insert {len(payloads)} rows from {dataset_dir}.")
        if payloads:
            preview = payloads[0].copy()
            preview["metastats"] = {"dataset": preview["metastats"]["dataset"], "row_index": preview["metastats"]["row_index"]}
            print(f"[PROMOTE] Sample payload: {preview}")
        return summary
    batch = create_batch_metadata(
        source=f"dataset_promotion:{dataset_dir.name}",
        status=StatusEnum.PENDING,
    )
    inserted = 0
    duplicates_skipped = 0
    blocked_urls_skipped = 0
    
    # Get URL from metadata for verification tier
    source_url = metadata.get('source_url')
    url_tier = get_url_verification_tier(source_url) if source_url else 'pending'
    
    try:
        from webapp.parser.utils.db_utils import get_session
        session = get_session()
        
        for payload in payloads:
            # Set verification status based on URL tier
            if url_tier == 'blocked':
                blocked_urls_skipped += 1
                logger.warning(f"[PROMOTE] Skipping blocked URL: {source_url}")
                continue
            elif url_tier == 'trusted':
                payload['verification_status'] = 'verified'
                payload['verified_at'] = datetime.now(timezone.utc)
            else:  # pending
                payload['verification_status'] = 'pending'
            
            payload['source_url'] = source_url
            payload['source_principal'] = metadata.get('source_principal')
            
            # Check for exact duplicate
            if check_exact_duplicate(
                session,
                state=payload.get('state'),
                county=payload.get('county'),
                contest=payload.get('contest'),
                candidate=payload.get('candidate'),
                party=payload.get('party'),
                votes=payload.get('votes'),
                precinct=payload.get('precinct'),
                election_date=payload.get('election_date'),
            ):
                duplicates_skipped += 1
                continue
            
            create_warehouse_election_result(batch_id=batch.batch_id, **payload)
            inserted += 1
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("[PROMOTE] Failed to insert records: %s", exc)
        update_batch_metadata(batch.batch_id, status=StatusEnum.ERROR)
        raise
    update_batch_metadata(
        batch.batch_id,
        status=StatusEnum.COMPLETED,
        metastats={
            "dataset_dir": str(dataset_dir),
            "records_inserted": inserted,
            "duplicates_skipped": duplicates_skipped,
            "blocked_urls_skipped": blocked_urls_skipped,
            "skipped_rows": skipped,
            "url_tier": url_tier,
            "contest": metadata.get("contest"),
            "state": metadata.get("state"),
            "county": metadata.get("county"),
        },
    )
    summary["batch_id"] = str(batch.batch_id)
    summary["inserted_records"] = inserted
    summary["duplicates_skipped"] = duplicates_skipped
    summary["blocked_urls_skipped"] = blocked_urls_skipped
    summary["url_tier"] = url_tier
    print(f"[PROMOTE] Inserted {inserted} rows (duplicates_skipped={duplicates_skipped}, blocked={blocked_urls_skipped}) from {dataset_dir} (batch={batch.batch_id}).")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote parser outputs into warehouse_election_results.")
    parser.add_argument(
        "--dataset",
        help="Dataset directory to promote. Defaults to the newest folder under --root.",
    )
    parser.add_argument(
        "--root",
        default=str(OUTPUT_DIR),
        help="Root directory to scan for datasets (defaults to configured OUTPUT_DIR).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of rows inserted (useful for spot checks).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be inserted without touching the database.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        dataset_dir = resolve_dataset_path(args.dataset, root)
        summary = promote_dataset(dataset_dir, dry_run=args.dry_run, limit=args.limit)
    except FileNotFoundError as exc:
        logger.error("[PROMOTE] %s", exc)
        raise SystemExit(1) from exc
    print(f"[PROMOTE] Summary: {summary}")


if __name__ == "__main__":
    main()
