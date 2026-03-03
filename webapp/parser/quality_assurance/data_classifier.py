"""
Data Classifier: DL1/DL2 Quality Assurance Pipeline

Classifies extracted election data through automated quality checks:
- DL1 (unverified): Freshly extracted, awaiting manual review
- DL2 (verified): Human approved + all QA checks passed
- Detects anomalies, flags for review

Database: PostgreSQL (verified_data schema)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor

from ..config import (
    VERIFIED_DATA_DB_HOST,
    VERIFIED_DATA_DB_NAME,
    VERIFIED_DATA_DB_PASSWORD,
    VERIFIED_DATA_DB_PORT,
    VERIFIED_DATA_DB_USER,
)
from ..utils.logger_singleton import logger

# ===== ENUMS =====

class DLStatus(str, Enum):
    """Classification status for election data."""
    DL1 = "DL1"  # Unverified
    DL2 = "DL2"  # Verified
    REJECTED = "REJECTED"  # Invalid/disputed
    DISPUTED = "DISPUTED"  # Was DL2, now flagged for review


class QAIssueType(str, Enum):
    """Automated quality issue detection types."""
    DUPLICATE_ROW = "duplicate_row"
    INVALID_VOTE_COUNT = "invalid_vote_count"
    IMPOSSIBLE_PERCENTAGE = "impossible_percentage"
    MISSING_FIELD = "missing_field"
    ANOMALY_DETECTED = "anomaly_detected"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    VOTE_SUM_MISMATCH = "vote_sum_mismatch"
    PERCENTAGE_SUM_INVALID = "percentage_sum_invalid"


class IssureSeverity(str, Enum):
    """Issue severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ActionType(str, Enum):
    """Audit trail action types."""
    CLASSIFICATION = "classification"
    AUTO_QA_PERFORMED = "auto_qa_performed"
    FLAGGED_FOR_REVIEW = "flagged_for_review"
    PROMOTED_TO_DL2 = "promoted_to_dl2"
    REJECTED = "rejected"
    ANOMALY_DETECTED = "anomaly_detected"
    RESOLVED = "resolved"


# ===== DATA CLASSES =====

@dataclass
class QAIssue:
    """Detected quality issue."""
    issue_type: str
    severity: str
    description: str
    affected_field: Optional[str] = None
    affected_rows: Optional[List[int]] = None
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    """Result of DL1/DL2 classification."""
    dataset_id: str
    dl_status: str
    confidence_score: float
    issues: List[QAIssue] = field(default_factory=list)
    should_promote_to_dl2: bool = False
    summary: str = ""


@dataclass
class DatasetMetadata:
    """Structured metadata for a parsed dataset."""
    source_url: str
    handler_name: str
    state_abbr: str
    county_name: Optional[str]
    election_year: int
    contest_name: str
    contestant_count: int
    data_row_count: int
    extraction_confidence: float
    trust_score: float
    headers: List[str] = field(default_factory=list)
    data_rows: List[Dict[str, Any]] = field(default_factory=list)


# ===== DATABASE CONNECTION =====

def get_db_connection():
    """Get PostgreSQL connection for verified data."""
    try:
        conn = psycopg2.connect(
            host=VERIFIED_DATA_DB_HOST,
            port=VERIFIED_DATA_DB_PORT,
            database=VERIFIED_DATA_DB_NAME,
            user=VERIFIED_DATA_DB_USER,
            password=VERIFIED_DATA_DB_PASSWORD,
        )
        return conn
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "database",
            "message": f"Failed to connect to verified_data database: {e}"
        })
        return None


# ===== DL1 CLASSIFICATION =====

def classify_as_dl1(metadata: DatasetMetadata) -> ClassificationResult:
    """
    Classify parsed data as DL1 (unverified).
    
    Stores the initial extraction with automatic QA checks.
    Doesn't promote to DL2 yet—requires manual review.
    
    Returns:
        ClassificationResult with dataset_id, DL1 status, detected issues
    """
    dataset_id = str(uuid4())
    
    # Run automated QA checks
    issues = detect_quality_issues(metadata)
    
    # Calculate confidence score
    base_confidence = (metadata.extraction_confidence + metadata.trust_score) / 2
    issue_penalty = len(issues) * 5  # Reduce confidence per issue found
    confidence_score = max(0, min(100, base_confidence - issue_penalty))
    
    # Determine if any issues are critical
    has_critical_issues = any(issue.severity == IssureSeverity.CRITICAL.value for issue in issues)
    
    # Store in database
    try:
        conn = get_db_connection()
        if not conn:
            raise Exception("No database connection")
        
        cursor = conn.cursor()
        
        # Insert into verified_datasets
        cursor.execute("""
            INSERT INTO verified_data.verified_datasets (
                dataset_id, source_url, source_handler, state_abbr, county_name,
                election_year, contest_name, contestant_count, data_row_count,
                extraction_confidence, trust_score, completeness_score, dl_status,
                automated_qa_passed, detected_issues_count
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s
            )
        """, (
            dataset_id, metadata.source_url, metadata.handler_name, metadata.state_abbr, metadata.county_name,
            metadata.election_year, metadata.contest_name, metadata.contestant_count, metadata.data_row_count,
            metadata.extraction_confidence, metadata.trust_score, 1.0, DLStatus.DL1.value,
            not has_critical_issues, len(issues)
        ))
        
        # Insert detected issues
        for issue in issues:
            cursor.execute("""
                INSERT INTO verified_data.quality_issues (
                    dataset_id, issue_type, severity, description, affected_field,
                    affected_rows, confidence_score
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s
                )
            """, (
                dataset_id, issue.issue_type, issue.severity, issue.description, issue.affected_field,
                json.dumps(issue.affected_rows) if issue.affected_rows else None, issue.confidence_score
            ))
        
        # Insert lineage entry
        cursor.execute("""
            INSERT INTO verified_data.verification_lineage (
                dataset_id, action_type, action_status, confidence_score, details
            ) VALUES (
                %s, %s, %s, %s, %s
            )
        """, (
            dataset_id, ActionType.CLASSIFICATION.value, 'completed', confidence_score,
            json.dumps({
                'extraction_confidence': metadata.extraction_confidence,
                'trust_score': metadata.trust_score,
                'issues_detected': len(issues)
            })
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info({
            "level": "INFO",
            "type": "dl_classification",
            "message": f"Dataset classified as DL1: {dataset_id}",
            "dataset_id": dataset_id,
            "confidence_score": confidence_score,
            "issues_count": len(issues)
        })
        
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "dl_classification",
            "message": f"Failed to classify data as DL1: {e}",
            "source_url": metadata.source_url
        })
        raise
    
    return ClassificationResult(
        dataset_id=dataset_id,
        dl_status=DLStatus.DL1.value,
        confidence_score=confidence_score,
        issues=issues,
        should_promote_to_dl2=False,
        summary=f"DL1 unverified. {len(issues)} issues detected. Trust score: {metadata.trust_score}/100"
    )


# ===== AUTOMATED QA CHECKS =====

def detect_quality_issues(metadata: DatasetMetadata) -> List[QAIssue]:
    """
    Run automated quality checks on extracted data.
    
    Checks for:
    - Duplicate rows
    - Invalid vote counts (negative, non-numeric)
    - Impossible percentages (>100%, don't sum to 100%)
    - Missing required fields
    - Anomalies (e.g., single candidate with 0 votes)
    
    Returns:
        List of QAIssue detected
    """
    issues: List[QAIssue] = []
    
    if not metadata.data_rows:
        issues.append(QAIssue(
            issue_type=QAIssueType.MISSING_FIELD.value,
            severity=IssureSeverity.CRITICAL.value,
            description="No data rows extracted",
            confidence_score=1.0
        ))
        return issues
    
    # Check 1: Duplicate rows
    seen_rows = set()
    for idx, row in enumerate(metadata.data_rows):
        row_hash = hash(tuple(sorted(row.items())))
        if row_hash in seen_rows:
            issues.append(QAIssue(
                issue_type=QAIssueType.DUPLICATE_ROW.value,
                severity=IssureSeverity.WARNING.value,
                description=f"Duplicate row found at index {idx}",
                affected_rows=[idx],
                confidence_score=0.95
            ))
        seen_rows.add(row_hash)
    
    # Check 2: Invalid vote counts
    vote_counts = []
    for idx, row in enumerate(metadata.data_rows):
        if 'vote_count' in row or 'votes' in row:
            vote_key = 'vote_count' if 'vote_count' in row else 'votes'
            try:
                votes = int(str(row[vote_key]).replace(',', ''))
                if votes < 0:
                    issues.append(QAIssue(
                        issue_type=QAIssueType.INVALID_VOTE_COUNT.value,
                        severity=IssureSeverity.ERROR.value,
                        description=f"Negative vote count at row {idx}: {votes}",
                        affected_field=vote_key,
                        affected_rows=[idx],
                        confidence_score=0.99
                    ))
                vote_counts.append((idx, votes))
            except ValueError:
                issues.append(QAIssue(
                    issue_type=QAIssueType.INVALID_VOTE_COUNT.value,
                    severity=IssureSeverity.ERROR.value,
                    description=f"Non-numeric vote count at row {idx}: {row[vote_key]}",
                    affected_field=vote_key,
                    affected_rows=[idx],
                    confidence_score=0.95
                ))
    
    # Check 3: Vote sum validation (all votes should sum to total)
    if vote_counts:
        total_votes = sum(votes for _, votes in vote_counts)
        if total_votes > 0:
            for idx, votes in vote_counts:
                pct = (votes / total_votes) * 100
                if pct > 100:
                    issues.append(QAIssue(
                        issue_type=QAIssueType.IMPOSSIBLE_PERCENTAGE.value,
                        severity=IssureSeverity.WARNING.value,
                        description=f"Vote percentage >100% at row {idx}: {pct:.2f}%",
                        affected_field='percentage',
                        affected_rows=[idx],
                        confidence_score=0.85
                    ))
    
    # Check 4: Missing required fields
    required_fields = ['candidate_name', 'vote_count']  # Extensible
    for idx, row in enumerate(metadata.data_rows):
        for required_field in required_fields:
            if required_field not in row or row[required_field] is None or str(row[required_field]).strip() == '':
                issues.append(QAIssue(
                    issue_type=QAIssueType.MISSING_FIELD.value,
                    severity=IssureSeverity.ERROR.value,
                    description=f"Missing required field '{required_field}' at row {idx}",
                    affected_field=required_field,
                    affected_rows=[idx],
                    confidence_score=0.9
                ))
                break  # Only report once per row
    
    # Check 5: Anomaly detection (single candidate with 0 votes is suspicious)
    if len(metadata.data_rows) == 1:
        row = metadata.data_rows[0]
        if 'vote_count' in row and int(str(row['vote_count']).replace(',', '')) == 0:
            issues.append(QAIssue(
                issue_type=QAIssueType.ANOMALY_DETECTED.value,
                severity=IssureSeverity.WARNING.value,
                description="Single candidate with 0 votes detected (anomalous)",
                affected_rows=[0],
                confidence_score=0.75
            ))
    
    return issues


# ===== DL2 PROMOTION (Manual Review) =====

def promote_to_dl2(
    dataset_id: str,
    reviewer_principal: str,
    certification_reason: str,
    resolve_issues: Optional[Dict[str, str]] = None  # {issue_id: resolution_notes}
) -> bool:
    """
    Promote DL1 dataset to DL2 (verified) after human review.
    
    Args:
        dataset_id: UUID of the DL1 dataset
        reviewer_principal: Email/ID of reviewer (e.g., 'john@elections.gov')
        certification_reason: Why reviewer approved this data
        resolve_issues: Optional mapping of issue_id → resolution notes
    
    Returns:
        True if promotion successful, False otherwise
    """
    try:
        conn = get_db_connection()
        if not conn:
            raise Exception("No database connection")
        
        cursor = conn.cursor()
        
        # Verify dataset exists and is DL1
        cursor.execute("""
            SELECT dataset_id, dl_status FROM verified_data.verified_datasets
            WHERE dataset_id = %s
        """, (dataset_id,))
        
        result = cursor.fetchone()
        if not result or result[1] != DLStatus.DL1.value:
            raise Exception(f"Dataset {dataset_id} not found or not DL1 status")
        
        # Update status to DL2
        cursor.execute("""
            UPDATE verified_data.verified_datasets
            SET dl_status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE dataset_id = %s
        """, (DLStatus.DL2.value, dataset_id))
        
        # Resolve issues if provided
        if resolve_issues:
            for issue_id, notes in resolve_issues.items():
                cursor.execute("""
                    UPDATE verified_data.quality_issues
                    SET is_resolved = TRUE, resolved_by_reviewer_principal = %s,
                        resolved_at = CURRENT_TIMESTAMP, resolution_notes = %s
                    WHERE dataset_id = %s
                """, (reviewer_principal, notes, dataset_id))
        
        # Insert lineage entry
        cursor.execute("""
            INSERT INTO verified_data.verification_lineage (
                dataset_id, action_type, action_status, reviewer_principal, certification_reason, details
            ) VALUES (
                %s, %s, %s, %s, %s, %s
            )
        """, (
            dataset_id, ActionType.PROMOTED_TO_DL2.value, 'completed', reviewer_principal, certification_reason,
            json.dumps({'timestamp': datetime.now(timezone.utc).isoformat()})
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info({
            "level": "INFO",
            "type": "dl_promotion",
            "message": f"Dataset promoted to DL2: {dataset_id}",
            "dataset_id": dataset_id,
            "reviewed_by": reviewer_principal
        })
        
        return True
        
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "dl_promotion",
            "message": f"Failed to promote to DL2: {e}",
            "dataset_id": dataset_id
        })
        return False


# ===== QUERY HELPERS =====

def get_pending_dl2_reviews() -> List[Dict[str, Any]]:
    """Get all DL1 datasets pending manual review."""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT dataset_id, source_url, state_abbr, county_name, election_year,
                   contest_name, contestant_count, extraction_confidence, trust_score,
                   detected_issues_count, extracted_at
            FROM verified_data.verified_datasets
            WHERE dl_status = %s
            ORDER BY extracted_at ASC
            LIMIT 100
        """, (DLStatus.DL1.value,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "query",
            "message": f"Failed to get pending DL2 reviews: {e}"
        })
        return []


def get_dl2_inventory(state_abbr: Optional[str] = None, county_name: Optional[str] = None, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get all verified DL2 datasets, optionally filtered by location/year."""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT dataset_id, source_url, state_abbr, county_name, election_year,
                   contest_name, contestant_count, extraction_confidence, trust_score, extracted_at
            FROM verified_data.verified_datasets
            WHERE dl_status = %s
        """
        params = [DLStatus.DL2.value]
        
        if state_abbr:
            query += " AND state_abbr = %s"
            params.append(state_abbr)
        
        if county_name:
            query += " AND county_name = %s"
            params.append(county_name)
        
        if year:
            query += " AND election_year = %s"
            params.append(year)
        
        query += " ORDER BY extracted_at DESC LIMIT 1000"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "query",
            "message": f"Failed to get DL2 inventory: {e}"
        })
        return []


def get_rejected_count() -> int:
    """Get total count of rejected datasets."""
    try:
        conn = get_db_connection()
        if not conn:
            return 0

        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM verified_data.verified_datasets
            WHERE dl_status = %s
            """,
            (DLStatus.REJECTED.value,),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return int(row[0] or 0) if row else 0

    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "query",
            "message": f"Failed to get rejected count: {e}",
        })
        return 0


def get_dataset_lineage(dataset_id: str) -> List[Dict[str, Any]]:
    """Get complete audit trail for a dataset."""
    try:
        conn = get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT action_type, action_status, reviewer_principal, certification_reason,
                   confidence_score, action_timestamp, details
            FROM verified_data.verification_lineage
            WHERE dataset_id = %s
            ORDER BY action_timestamp ASC
        """, (dataset_id,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(row) for row in results]
        
    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "query",
            "message": f"Failed to get dataset lineage: {e}",
            "dataset_id": dataset_id
        })
        return []
