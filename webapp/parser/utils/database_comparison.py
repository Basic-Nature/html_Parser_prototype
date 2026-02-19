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
from typing import Any, Dict, Optional

from ..config import LOG_DIR
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
            from urllib.parse import urlparse, urlunparse
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
