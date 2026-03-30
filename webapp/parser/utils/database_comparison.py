"""
Database Comparison Utility

Checks if finalized data already exists for a URL before launching the parser.
Supports:
- Google Sheets finalized data lookup
- Warehouse database (warehouse_election_results) queries
- verified_datasets table checks
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from .logger_singleton import logger


def check_existing_finalized_data(
    url: str,
    *,
    session_id: Optional[str] = None,
    state: Optional[str] = None,
    county: Optional[str] = None,
    contest: Optional[str] = None
) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Check if finalized data already exists for a URL.
    
    Checks in order:
    1. Google Sheets finalized data
    2. Warehouse database (warehouse_election_results)
    3. verified_datasets table
    
    Args:
        url: The URL to check
        session_id: Optional session ID for logging
        state: Optional state hint for filtering
        county: Optional county hint for filtering
        contest: Optional contest hint for filtering
    
    Returns:
        (data_exists, data_source, metadata) tuple where:
        - data_exists: True if finalized data found
        - data_source: "google_sheets", "warehouse", or None
        - metadata: Dict with details about the found data
    """
    url = (url or "").strip()
    if not url:
        return False, None, None
    
    logger.info({
        "level": "INFO",
        "type": "database",
        "message": f"[DatabaseComparison] Checking for existing finalized data: {url}",
        "session_id": session_id,
        "url": url
    })
    
    # --- 1. Google Sheets Check ---
    sheets_result = _check_google_sheets_finalized_data(
        url,
        session_id=session_id,
        state=state,
        county=county,
        contest=contest
    )
    if sheets_result[0]:
        logger.info({
            "level": "INFO",
            "type": "database",
            "message": f"[DatabaseComparison] Found existing data in Google Sheets for {url}",
            "session_id": session_id,
            "url": url,
            "data_source": "google_sheets"
        })
        return True, "google_sheets", sheets_result[1]
    
    # --- 2. Warehouse Database Check ---
    warehouse_result = _check_warehouse_database(
        url,
        session_id=session_id,
        state=state,
        county=county,
        contest=contest
    )
    if warehouse_result[0]:
        logger.info({
            "level": "INFO",
            "type": "database",
            "message": f"[DatabaseComparison] Found existing data in warehouse for {url}",
            "session_id": session_id,
            "url": url,
            "data_source": "warehouse"
        })
        return True, "warehouse", warehouse_result[1]
    
    # --- 3. verified_datasets Check ---
    verified_result = _check_verified_datasets(
        url,
        session_id=session_id,
        state=state,
        county=county,
        contest=contest
    )
    if verified_result[0]:
        logger.info({
            "level": "INFO",
            "type": "database",
            "message": f"[DatabaseComparison] Found existing data in verified_datasets for {url}",
            "session_id": session_id,
            "url": url,
            "data_source": "verified_datasets"
        })
        return True, "verified_datasets", verified_result[1]
    
    logger.info({
        "level": "INFO",
        "type": "database",
        "message": f"[DatabaseComparison] No existing finalized data found for {url}",
        "session_id": session_id,
        "url": url
    })
    return False, None, None


def _check_google_sheets_finalized_data(
    url: str,
    *,
    session_id: Optional[str] = None,
    state: Optional[str] = None,
    county: Optional[str] = None,
    contest: Optional[str] = None
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Check Google Sheets 'Finalized Data' tab for matching URL."""
    try:
        from ..data_standardization.google_sheets_client import GoogleSheetsElectionClient
        
        client = GoogleSheetsElectionClient()
        result = client.fetch_finalized_data()
        
        if not result.success or not result.records:
            return False, None
        
        # Look for matching URL in records
        for record in result.records:
            if not isinstance(record, dict):
                continue
            
            record_url = (record.get("source_url") or "").strip()
            if not record_url:
                continue
            
            # Exact match
            if record_url.lower() == url.lower():
                metadata = {
                    "state": record.get("state"),
                    "county": record.get("county"),
                    "contest": record.get("contest"),
                    "candidate_count": len([k for k in record.keys() if "candidate" in k.lower()]),
"source": "google_sheets_finalized_data",
                    "record": record
                }
                return True, metadata
            
            # URL normalization: strip trailing slash, query params, fragments
            from urllib.parse import urlparse
            parsed_record = urlparse(record_url)
            parsed_target = urlparse(url)
            
            # Compare normalized paths (scheme + netloc + path)
            if (parsed_record.scheme == parsed_target.scheme and
                parsed_record.netloc.lower() == parsed_target.netloc.lower() and
                parsed_record.path.rstrip("/") == parsed_target.path.rstrip("/")):
                
                metadata = {
                    "state": record.get("state"),
                    "county": record.get("county"),
                    "contest": record.get("contest"),
                    "candidate_count": len([k for k in record.keys() if "candidate" in k.lower()]),
                    "source": "google_sheets_finalized_data",
                    "record": record
                }
                return True, metadata
        
        return False, None
        
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "database",
            "message": f"[DatabaseComparison] Google Sheets check failed: {exc}",
            "session_id": session_id
        })
        return False, None


def _check_warehouse_database(
    url: str,
    *,
    session_id: Optional[str] = None,
    state: Optional[str] = None,
    county: Optional[str] = None,
    contest: Optional[str] = None
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Check warehouse_election_results table for matching URL."""
    try:
        from sqlalchemy import inspect, text

        from ..utils.db_utils import get_engine
        
        engine = get_engine()
        
        # Verify table exists and has source_url column
        inspector = inspect(engine)
        try:
            cols = inspector.get_columns("warehouse_election_results")
        except Exception:
            return False, None
        
        col_names = {col.get("name") for col in cols if col.get("name")}
        if "source_url" not in col_names:
            return False, None
        
        # Build query
        select_cols = ["source_url"]
        for col in ("state", "county", "contest"):
            if col in col_names:
                select_cols.append(col)
        
        aggregates = ["COUNT(*) AS row_count"]
        if "candidate" in col_names:
            aggregates.append("COUNT(DISTINCT candidate) AS candidate_count")
        
        select_sql = ", ".join(select_cols + aggregates)
        group_sql = ", ".join(select_cols)
        
        query = f"""
            SELECT {select_sql}
            FROM warehouse_election_results
            WHERE source_url = :url
            GROUP BY {group_sql}
            LIMIT 1
        """
        
        with engine.connect() as conn:
            rows = conn.execute(text(query), {"url": url}).mappings().all()
        
        if rows:
            row = rows[0]
            metadata = {
                "state": row.get("state"),
                "county": row.get("county"),
                "contest": row.get("contest"),
                "row_count": row.get("row_count", 0),
                "candidate_count": row.get("candidate_count", 0),
                "source": "warehouse_election_results"
            }
            return True, metadata
        
        return False, None
        
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "database",
            "message": f"[DatabaseComparison] Warehouse check failed: {exc}",
            "session_id": session_id
        })
        return False, None


def _check_verified_datasets(
    url: str,
    *,
    session_id: Optional[str] = None,
    state: Optional[str] = None,
    county: Optional[str] = None,
    contest: Optional[str] = None
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """Check verified_datasets table for matching URL with approved QA status."""
    try:
        from sqlalchemy import inspect, text

        from ..utils.db_utils import get_engine
        
        engine = get_engine()
        
        # Verify table exists
        inspector = inspect(engine)
        try:
            cols = inspector.get_columns("verified_datasets")
        except Exception:
            return False, None
        
        col_names = {col.get("name") for col in cols if col.get("name")}
        if "source_url" not in col_names or "qa_status" not in col_names:
            return False, None
        
        # Look for approved datasets with matching URL
        query = """
            SELECT 
                source_url,
                qa_status,
                state,
                county,
                contest,
                verified_at,
                row_count
            FROM verified_datasets
            WHERE source_url = :url
                AND qa_status IN ('approved', 'verified', 'finalized')
            ORDER BY verified_at DESC
            LIMIT 1
        """
        
        with engine.connect() as conn:
            rows = conn.execute(text(query), {"url": url}).mappings().all()
        
        if rows:
            row = rows[0]
            metadata = {
                "state": row.get("state"),
                "county": row.get("county"),
                "contest": row.get("contest"),
                "qa_status": row.get("qa_status"),
                "verified_at": row.get("verified_at"),
                "row_count": row.get("row_count", 0),
                "source": "verified_datasets"
            }
            return True, metadata
        
        return False, None
        
    except Exception as exc:
        logger.warning({
            "level": "WARNING",
            "type": "database",
            "message": f"[DatabaseComparison] verified_datasets check failed: {exc}",
            "session_id": session_id
        })
        return False, None


def fetch_database_reference_metadata(
    url: str,
    *,
    session_id: Optional[str] = None,
    state: Optional[str] = None,
    county: Optional[str] = None,
    contest: Optional[str] = None,
) -> tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Fetch best-available reference metadata from database tables for a URL.

    Unlike ``check_existing_finalized_data``, this helper does not query Google Sheets.
    It is intended for lightweight cross-checking during output finalization.

    Returns:
        (found, source, metadata)
        - source is ``verified_datasets`` or ``warehouse`` when found.
    """
    url = (url or "").strip()
    if not url:
        return False, None, None

    verified_ok, verified_meta = _check_verified_datasets(
        url,
        session_id=session_id,
        state=state,
        county=county,
        contest=contest,
    )
    warehouse_ok, warehouse_meta = _check_warehouse_database(
        url,
        session_id=session_id,
        state=state,
        county=county,
        contest=contest,
    )

    if verified_ok:
        merged = dict(verified_meta or {})
        # Backfill missing cardinality fields from warehouse when available.
        if warehouse_ok and isinstance(warehouse_meta, dict):
            for field in ("candidate_count", "row_count", "contest", "state", "county"):
                if merged.get(field) in (None, "") and warehouse_meta.get(field) not in (None, ""):
                    merged[field] = warehouse_meta.get(field)
        return True, "verified_datasets", merged

    if warehouse_ok:
        return True, "warehouse", warehouse_meta

    return False, None, None


def evaluate_url_processing_policy(
    url: str,
    *,
    session_id: Optional[str] = None,
    state: Optional[str] = None,
    county: Optional[str] = None,
    contest: Optional[str] = None,
    skip_database_check: bool = False,
    force_reparse: bool = False,
) -> Dict[str, Any]:
    """Return a single decision payload for URL skip/reparse logic.

    This centralizes behavior that was previously spread across parser entry points.
    """
    normalized_url = (url or "").strip()
    payload: Dict[str, Any] = {
        "url": normalized_url,
        "should_skip": False,
        "decision": "process",
        "data_source": None,
        "metadata": None,
        "checked": False,
    }

    if not normalized_url:
        payload["should_skip"] = True
        payload["decision"] = "invalid_url"
        return payload

    if force_reparse:
        payload["decision"] = "force_reparse"
        return payload

    if skip_database_check:
        payload["decision"] = "database_check_disabled"
        return payload

    payload["checked"] = True
    data_exists, data_source, metadata = check_existing_finalized_data(
        normalized_url,
        session_id=session_id,
        state=state,
        county=county,
        contest=contest,
    )
    if data_exists:
        payload["should_skip"] = True
        payload["decision"] = "skipped_data_exists"
        payload["data_source"] = data_source
        payload["metadata"] = metadata
    return payload


def cross_check_profile_for_source(reference_source: Optional[str]) -> Dict[str, Any]:
    """Return tolerance/severity profile for database reference sources."""

    def _bool_env(name: str, default: bool = False) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _int_env(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except Exception:
            return default

    def _float_env(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except Exception:
            return default

    source_key = re.sub(r"[^A-Za-z0-9]+", "_", (reference_source or "default").strip().upper())
    defaults = {
        "DEFAULT": {
            "row_delta_abs": 2,
            "row_delta_ratio": 0.10,
            "candidate_delta_abs": 0,
            "strict_labels": False,
        },
        "VERIFIED_DATASETS": {
            "row_delta_abs": 1,
            "row_delta_ratio": 0.05,
            "candidate_delta_abs": 0,
            "strict_labels": True,
        },
        "WAREHOUSE": {
            "row_delta_abs": 3,
            "row_delta_ratio": 0.15,
            "candidate_delta_abs": 1,
            "strict_labels": False,
        },
        "GOOGLE_SHEETS": {
            "row_delta_abs": 2,
            "row_delta_ratio": 0.10,
            "candidate_delta_abs": 0,
            "strict_labels": True,
        },
    }

    base = dict(defaults.get(source_key, defaults["DEFAULT"]))
    prefix = f"DB_CROSSCHECK_{source_key}"
    base["row_delta_abs"] = _int_env(f"{prefix}_ROW_DELTA_ABS", base["row_delta_abs"])
    base["row_delta_ratio"] = _float_env(f"{prefix}_ROW_DELTA_RATIO", base["row_delta_ratio"])
    base["candidate_delta_abs"] = _int_env(f"{prefix}_CANDIDATE_DELTA_ABS", base["candidate_delta_abs"])
    base["strict_labels"] = _bool_env(f"{prefix}_STRICT_LABELS", base["strict_labels"])
    base["source_key"] = source_key
    return base


def should_fail_database_cross_check(context: Dict[str, Any]) -> bool:
    """Decide whether a mismatch should block output finalization."""
    if "database_cross_check_fail_on_mismatch" in context:
        try:
            return bool(context.get("database_cross_check_fail_on_mismatch"))
        except Exception:
            return False
    raw = os.environ.get("DB_CROSSCHECK_FAIL_ON_MISMATCH")
    return bool(raw and raw.strip().lower() in {"1", "true", "yes", "on"})


def _compute_extracted_candidate_count(headers: List[str], rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    candidate_values: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_val = row.get("Candidate")
        if isinstance(candidate_val, str) and candidate_val.strip():
            candidate_values.add(candidate_val.strip().lower())

    if candidate_values:
        return len(candidate_values)

    header_candidates = {
        h.split(" - ", 1)[0].strip().lower()
        for h in (headers or [])
        if isinstance(h, str) and " - " in h and "candidate" not in h.lower()
    }
    return len({h for h in header_candidates if h})


def build_database_cross_check(
    *,
    source_url: str,
    headers: List[str],
    rows: List[Dict[str, Any]],
    contest: str,
    state: str,
    county: str,
    reference_source: Optional[str],
    reference_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a structured comparison between extracted output and reference metadata."""
    profile = cross_check_profile_for_source(reference_source)
    extracted = {
        "row_count": len(rows or []),
        "candidate_count": _compute_extracted_candidate_count(headers or [], rows or []),
        "contest": (contest or "").strip(),
        "state": (state or "").strip(),
        "county": (county or "").strip(),
    }

    reference = reference_metadata or {}
    result: Dict[str, Any] = {
        "status": "unavailable",
        "source_url": source_url,
        "reference_source": reference_source,
        "profile": profile,
        "reference": reference,
        "extracted": extracted,
        "mismatches": [],
    }
    if not reference:
        return result

    mismatches: List[Dict[str, Any]] = []

    def _norm(v: Any) -> str:
        return str(v or "").strip().lower()

    for field in ("contest", "state", "county"):
        ref_val = reference.get(field)
        ext_val = extracted.get(field)
        if ref_val and ext_val and _norm(ref_val) != _norm(ext_val):
            mismatches.append(
                {
                    "field": field,
                    "reference": ref_val,
                    "extracted": ext_val,
                    "severity": "error" if profile.get("strict_labels") else "warning",
                }
            )

    ref_row_count = reference.get("row_count")
    if isinstance(ref_row_count, int) and ref_row_count >= 0:
        ext_row_count = extracted["row_count"]
        if ref_row_count == 0 and ext_row_count > 0:
            mismatches.append(
                {
                    "field": "row_count",
                    "reference": ref_row_count,
                    "extracted": ext_row_count,
                    "severity": "warning",
                }
            )
        elif ref_row_count > 0:
            delta = abs(ext_row_count - ref_row_count)
            ratio = delta / max(ref_row_count, 1)
            if delta > int(profile.get("row_delta_abs", 2)) and ratio > float(profile.get("row_delta_ratio", 0.10)):
                mismatches.append(
                    {
                        "field": "row_count",
                        "reference": ref_row_count,
                        "extracted": ext_row_count,
                        "delta": delta,
                        "delta_ratio": round(ratio, 3),
                        "severity": "warning",
                    }
                )

    ref_candidate_count = reference.get("candidate_count")
    if isinstance(ref_candidate_count, int) and ref_candidate_count >= 0:
        ext_candidate_count = extracted["candidate_count"]
        delta = abs(ext_candidate_count - ref_candidate_count)
        if delta > int(profile.get("candidate_delta_abs", 0)):
            mismatches.append(
                {
                    "field": "candidate_count",
                    "reference": ref_candidate_count,
                    "extracted": ext_candidate_count,
                    "delta": delta,
                    "severity": "warning",
                }
            )

    result["mismatches"] = mismatches
    result["status"] = "match" if not mismatches else "mismatch"
    return result
