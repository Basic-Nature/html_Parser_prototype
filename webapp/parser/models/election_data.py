"""
Election Data SQLAlchemy Models
Defines schema for election results with staging, validation, and production tiers.
"""

from datetime import datetime
from enum import Enum as PyEnum
from typing import Any

from sqlalchemy import Boolean as _Boolean
from sqlalchemy import Column, ForeignKey, Index
from sqlalchemy import DateTime as _DateTime
from sqlalchemy import Enum as _SQLEnumType
from sqlalchemy import Float as _Float
from sqlalchemy import Integer as _Integer
from sqlalchemy import String as _String
from sqlalchemy import Text as _Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def Integer(*args: Any, **kwargs: Any):
    return Column(_Integer, *args, **kwargs)


def String(length: int | None = None, *args: Any, **kwargs: Any):
    if length is None:
        return Column(_String, *args, **kwargs)
    return Column(_String(length), *args, **kwargs)


def Text(*args: Any, **kwargs: Any):
    return Column(_Text, *args, **kwargs)


def Boolean(*args: Any, **kwargs: Any):
    return Column(_Boolean, *args, **kwargs)


def DateTime(*args: Any, **kwargs: Any):
    return Column(_DateTime, *args, **kwargs)


def Float(*args: Any, **kwargs: Any):
    return Column(_Float, *args, **kwargs)


def SQLEnum(enum_class: Any, *args: Any, **kwargs: Any):
    return Column(_SQLEnumType(enum_class), *args, **kwargs)


class DataQualityTier(PyEnum):
    """Data quality progression through pipeline"""
    STAGING = "staging"  # Raw from Google Sheets
    VALIDATION = "validation"  # Standardized, flagged for review
    PRODUCTION = "production"  # Approved for official use


class ManualReviewStatus(PyEnum):
    """Manual review workflow states"""
    PENDING = "pending"  # Waiting for human review
    REVIEWED = "reviewed"  # Human reviewed
    APPROVED = "approved"  # Approved for production
    REJECTED = "rejected"  # Rejected or needs corrections
    CORRECTED = "corrected"  # Corrected after rejection


class DataQualityFlagType(PyEnum):
    """Quality flags requiring manual review"""
    MISSING_FEC_ID = "missing_fec_id"
    MISSING_PARTY = "missing_party_code"
    PARTY_MISMATCH = "party_ballot_vs_fec_mismatch"
    CANDIDATE_NAME_UNCLEAR = "candidate_name_unclear"
    VOTE_TYPE_AMBIGUOUS = "vote_type_ambiguous"
    WRITE_IN_UNCERTAIN = "write_in_uncertain"
    VOTE_TOTAL_MISMATCH = "vote_total_mismatch"
    DUPLICATE_CANDIDATE = "duplicate_candidate"


class ElectionResult(Base):
    """
    Main election results table (Finalized Data standardized)
    Public table for approved, production-ready records
    """
    __tablename__ = 'election_results'
    __table_args__ = (
        Index('ix_election_results_race_id', 'race_id'),
        Index('ix_election_results_county_state', 'county', 'state'),
        Index('ix_election_results_candidate_race', 'candidate_name', 'race_id'),
    )
    
    # Primary Key
    id = Integer(primary_key=True)
    
    # Elections & Races
    year = Integer(nullable=False)
    state = String(50, nullable=False)
    county = String(100, nullable=False)
    office = String(200, nullable=False)
    race_id = String(100, nullable=False)  # Unique identifier from Smart Elections DB
    
    # Candidate Information
    candidate_name = String(250, nullable=False)  # LASTNAME, FIRSTNAME format
    fec_id = String(50, nullable=True)  # FEC candidate ID (nullable for write-ins, overvotes)
    is_write_in = Boolean(nullable=False, default=False)
    
    # Party Information
    ballot_party = String(50, nullable=True)  # As reported on ballot
    fec_party = String(10, nullable=True)  # Standardized FEC code (DEM, REP, LIB, etc.)
    
    # Vote Counts (separate columns for each type)
    uncategorized_votes = Integer(nullable=True, default=0)
    early_votes = Integer(nullable=True, default=0)
    election_day_votes = Integer(nullable=True, default=0)
    mail_votes = Integer(nullable=True, default=0)
    provisional_votes = Integer(nullable=True, default=0)
    total_votes = Integer(nullable=False, default=0)
    vote_type_classification = String(50, nullable=True)  # UNCATEGORIZED, EARLY, etc.
    
    # Source Tracking
    source_data_url = Text(nullable=True)
    source_file_name = String(255, nullable=True)
    
    # Audit Trail
    created_at = DateTime(nullable=False, default=datetime.utcnow)
    created_by = String(100, nullable=True)  # User or system name
    updated_at = DateTime(nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = String(100, nullable=True)
    
    # Manual Review & Quality Control
    manual_review_status = SQLEnum(ManualReviewStatus, nullable=False, default=ManualReviewStatus.APPROVED)
    reviewed_by = String(100, nullable=True)  # Who approved this record
    reviewed_at = DateTime(nullable=True)
    review_notes = Text(nullable=True)
    
    # Reconciliation Fields
    original_ballot_name = String(250, nullable=True)  # Pre-standardization name
    original_ballot_party = String(50, nullable=True)  # Pre-standardization party
    
    # Relationships
    validation_record = relationship("ValidationRecord", back_populates="election_result", uselist=False)
    audit_entries = relationship("AuditLog", back_populates="election_result")


class ValidationRecord(Base):
    """
    Validation tier records - data with standardization applied and quality flags
    Intermediate table for manual review before promotion to production
    """
    __tablename__ = 'validation_records'
    __table_args__ = (
        Index('ix_validation_records_race_id', 'race_id'),
        Index('ix_validation_records_status', 'review_status'),
        Index('ix_validation_records_flags', 'has_flags'),
    )
    
    # Primary Key
    id = Integer(primary_key=True)
    
    # Link to approved record (nullable until approved)
    election_result_id = Integer(ForeignKey('election_results.id'), nullable=True)
    election_result = relationship("ElectionResult", back_populates="validation_record")
    
    # Elections & Races
    year = Integer(nullable=False)
    state = String(50, nullable=False)
    county = String(100, nullable=False)
    office = String(200, nullable=False)
    race_id = String(100, nullable=False)
    
    # Candidate Information
    ballot_candidate_name = String(250, nullable=False)  # Raw from Google Sheets
    standardized_candidate_name = String(250, nullable=False)  # After standardization
    fec_id = String(50, nullable=True)
    is_write_in = Boolean(nullable=False, default=False)
    
    # Party Information
    ballot_party = String(50, nullable=True)
    fec_party = String(10, nullable=True)
    
    # Vote Counts
    uncategorized_votes = Integer(nullable=True)
    early_votes = Integer(nullable=True)
    election_day_votes = Integer(nullable=True)
    mail_votes = Integer(nullable=True)
    provisional_votes = Integer(nullable=True)
    total_votes = Integer(nullable=False)
    
    # Quality Control
    has_flags = Boolean(nullable=False, default=False)
    quality_flags = Text(nullable=True)  # JSON array of flag objects
    warning_messages = Text(nullable=True)  # JSON array of warnings
    
    # Source Tracking
    source_url = Text(nullable=True)
    
    # Manual Review Workflow
    review_status = SQLEnum(ManualReviewStatus, nullable=False, default=ManualReviewStatus.PENDING)
    assigned_to = String(100, nullable=True)  # User assigned for review
    reviewed_by = String(100, nullable=True)
    reviewed_at = DateTime(nullable=True)
    review_notes = Text(nullable=True)
    
    # Timestamps
    created_at = DateTime(nullable=False, default=datetime.utcnow)
    updated_at = DateTime(nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    standardized_at = DateTime(nullable=True)
    
    # Metadata
    standardization_version = String(50, nullable=True)  # Version of standardizer used


class StagingRecord(Base):
    """
    Staging tier - raw data directly from Google Sheets
    No transformation applied; used for initial ingestion and debugging
    """
    __tablename__ = 'staging_records'
    __table_args__ = (
        Index('ix_staging_records_race_id', 'race_id'),
        Index('ix_staging_records_processed', 'is_processed'),
    )
    
    id = Integer(primary_key=True)
    
    # Raw columns from Google Sheets - Finalized Data tab
    year = Integer(nullable=True)
    county_district = String(100, nullable=True)
    ballot_candidate_name = String(250, nullable=True)
    ballot_party = String(50, nullable=True)
    uncategorized_votes = String(50, nullable=True)
    early_votes = String(50, nullable=True)
    election_day_votes = String(50, nullable=True)
    mail_in_votes = String(50, nullable=True)
    provisional_votes = String(50, nullable=True)
    is_write_in = String(20, nullable=True)
    candidate = String(250, nullable=True)
    office = String(200, nullable=True)
    state = String(50, nullable=True)
    party = String(50, nullable=True)
    fec_id = String(50, nullable=True)
    source_data_url = Text(nullable=True)
    race_id = String(100, nullable=True)
    total_votes = String(50, nullable=True)
    office_basic = String(200, nullable=True)
    
    # Processing metadata
    is_processed = Boolean(nullable=False, default=False)
    processing_error = Text(nullable=True)
    
    # Timestamps
    ingested_at = DateTime(nullable=False, default=datetime.utcnow)
    processed_at = DateTime(nullable=True)


class VoterDropoff(Base):
    """
    Down-ballot calculations - voter drop-off analysis by party/county
    Supplementary table from Down-Ballot Calculations workbook
    """
    __tablename__ = 'voter_dropoff'
    __table_args__ = (
        Index('ix_voter_dropoff_race_id', 'race_id'),
        Index('ix_voter_dropoff_county_party', 'county', 'party'),
    )
    
    id = Integer(primary_key=True)
    
    # Dimensions
    year = Integer(nullable=False)
    state = String(50, nullable=False)
    county = String(100, nullable=False)
    office = String(200, nullable=False)
    race_id = String(100, nullable=False)
    party = String(50, nullable=False)
    
    # Vote Metrics
    presidential_votes = Integer(nullable=False)
    down_ballot_votes = Integer(nullable=False)
    dropoff_percentage = Float(nullable=False)  # Down-ballot as % of presidential
    
    # Audit Trail
    created_at = DateTime(nullable=False, default=datetime.utcnow)
    updated_at = DateTime(nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class RaceMetadata(Base):
    """
    Race metadata tracking - catalog of all races in database
    From Races workbook
    """
    __tablename__ = 'race_metadata'
    __table_args__ = (
        Index('ix_race_metadata_race_id', 'race_id'),
    )
    
    id = Integer(primary_key=True)
    
    # Race Identifiers
    year = Integer(nullable=False)
    state = String(50, nullable=False)
    office = String(200, nullable=False)
    race_id = String(100, nullable=False, unique=True)
    
    # Statistics
    record_count = Integer(nullable=False, default=0)
    candidate_count = Integer(nullable=False, default=0)
    total_votes = Integer(nullable=False, default=0)
    
    # Quality Tracking
    flagged_record_count = Integer(nullable=False, default=0)
    pending_review_count = Integer(nullable=False, default=0)
    approved_count = Integer(nullable=False, default=0)
    
    # Chain of Custody
    chain_of_custody = Text(nullable=True)  # JSON tracking
    
    # Audit Trail
    created_at = DateTime(nullable=False, default=datetime.utcnow)
    updated_at = DateTime(nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    """
    Complete audit trail of all changes to election data
    Tracks what, who, when for regulatory compliance and traceability
    """
    __tablename__ = 'audit_log'
    __table_args__ = (
        Index('ix_audit_log_record_id', 'election_result_id'),
        Index('ix_audit_log_action_date', 'action_date'),
        Index('ix_audit_log_user', 'performed_by'),
    )
    
    id = Integer(primary_key=True)
    
    # Reference to affected record
    election_result_id = Integer(ForeignKey('election_results.id'), nullable=True)
    election_result = relationship("ElectionResult", back_populates="audit_entries")
    
    # Change Details
    action = String(100, nullable=False)  # CREATED, UPDATED, APPROVED, REJECTED, CORRECTED
    field_name = String(100, nullable=True)  # Which field changed
    old_value = Text(nullable=True)
    new_value = Text(nullable=True)
    
    # Context
    description = Text(nullable=True)
    
    # Accountability
    performed_by = String(100, nullable=False)
    action_date = DateTime(nullable=False, default=datetime.utcnow)
    
    # Source (for tracking mass updates)
    source_system = String(100, nullable=True)  # "Google Sheets mass update", "manual correction", etc.
    related_batch_id = String(100, nullable=True)  # For grouping related changes


class ManualReviewQueue(Base):
    """
    Queue of flagged records pending manual review
    Provides efficient filtering and assignment workflow
    """
    __tablename__ = 'manual_review_queue'
    __table_args__ = (
        Index('ix_review_queue_status', 'status'),
        Index('ix_review_queue_assigned', 'assigned_to'),
        Index('ix_review_queue_priority', 'priority_level'),
    )
    
    id = Integer(primary_key=True)
    
    # Reference to validation record
    validation_record_id = Integer(ForeignKey('validation_records.id'), nullable=False)
    
    # Priority & Assignment
    priority_level = Integer(nullable=False, default=0)  # 0=low, 1=medium, 2=high
    status = SQLEnum(ManualReviewStatus, nullable=False, default=ManualReviewStatus.PENDING)
    assigned_to = String(100, nullable=True)
    
    # Flag Details
    primary_flag = String(100, nullable=False)  # Main flag requiring review
    all_flags = Text(nullable=False)  # JSON array of all flags
    
    # Data for Review (denormalized for quick UI access)
    race_id = String(100, nullable=False)
    candidate_name = String(250, nullable=False)
    ballot_party = String(50, nullable=True)
    fec_party = String(10, nullable=True)
    fec_id = String(50, nullable=True)
    
    # Workflow
    review_notes = Text(nullable=True)
    correction_notes = Text(nullable=True)
    
    # Timestamps
    created_at = DateTime(nullable=False, default=datetime.utcnow)
    assigned_at = DateTime(nullable=True)
    completed_at = DateTime(nullable=True)
    
    # SLA tracking
    age_hours = Integer(nullable=True)  # For reporting


class GoogleSheetsSync(Base):
    """
    Metadata for Google Sheets synchronization
    Tracks which data has been imported and when
    """
    __tablename__ = 'google_sheets_sync'
    
    id = Integer(primary_key=True)
    
    # Sync Details
    workbook_name = String(100, nullable=False)  # e.g., "Finalized Data", "Down-Ballot Calculations"
    sheet_id = String(100, nullable=False)
    last_sync_time = DateTime(nullable=False, default=datetime.utcnow)
    records_imported = Integer(nullable=False, default=0)
    records_flagged = Integer(nullable=False, default=0)
    
    # Status
    sync_status = String(50, nullable=False)  # SUCCESS, PARTIAL, FAILED
    error_message = Text(nullable=True)
    
    # Sync Range Tracking
    last_row_processed = Integer(nullable=True)
    
    # Audit
    synced_by = String(100, nullable=False)  # System user or person who triggered sync
    next_sync_scheduled = DateTime(nullable=True)


# =====================================================================
# SMART ELECTIONS WORKFLOW MODELS
# DL1/DL2 parallel workflow with QC1/QC2 checkpoints
# =====================================================================

class DownloadRecord(Base):
    """
    Worklist tracking - single row per race through entire 4-step workflow
    Maps to Worklist columns A-R from SMART Elections process
    """
    __tablename__ = 'download_records'
    __table_args__ = (
        Index('ix_download_records_race_id', 'race_id'),
        Index('ix_download_records_workflow_status', 'workflow_status'),
    )
    
    id = Integer(primary_key=True)
    
    # Race Identifiers (Worklist Columns A-G)
    year = Integer(nullable=False)
    state = String(50, nullable=False)
    county = String(100, nullable=True)
    office = String(200, nullable=False)
    race_id = String(100, nullable=False, unique=True)
    data_format = String(100, nullable=True)  # Format from Step 1 form
    
    # Step 1 - Source URL Tracking (Column K)
    source_url = Text(nullable=True)
    source_url_added_by = String(100, nullable=True)
    source_url_added_at = DateTime(nullable=True)
    
    # Step 2a - DL1 Tracking (Columns L-O)
    dl1_assigned_to = String(100, nullable=True)
    dl1_status = String(50, nullable=False, default='pending')  # pending|in_progress|completed|ready_for_qc
    dl1_data_source = String(100, nullable=True)  # 'reformatted_sheet'|'source_url'|'manual_entry'
    dl1_candidate_check_completed = Boolean(nullable=False, default=False)
    dl1_candidates_reviewed = Boolean(nullable=False, default=False)
    dl1_completed_at = DateTime(nullable=True)
    
    # Step 2b - DL2 Tracking (Columns P-S)
    dl2_assigned_to = String(100, nullable=True)
    dl2_status = String(50, nullable=False, default='pending')  # pending|in_progress|completed|ready_for_qc
    dl2_data_source = String(100, nullable=True)  # 'google_sheets_enriched'|'reformatted_sheet'
    dl2_candidate_check_completed = Boolean(nullable=False, default=False)
    dl2_candidates_reviewed = Boolean(nullable=False, default=False)
    dl2_completed_at = DateTime(nullable=True)
    
    # Step 2 Pre-QC (Columns T-U)
    preqc_auto_check_completed = Boolean(nullable=False, default=False)
    preqc_result = String(50, nullable=True)  # passed|failed|review_needed
    preqc_strict_passed = Boolean(nullable=True)
    preqc_fuzzy_score = Float(nullable=True)  # 0.0-1.0 confidence
    preqc_discrepancy_count = Integer(nullable=False, default=0)
    preqc_checked_at = DateTime(nullable=True)
    
    # Step 3 - QC1 Tracking (Columns V-X)
    qc1_assigned_to = String(100, nullable=True)
    qc1_status = String(50, nullable=False, default='pending')  # pending|in_progress|completed
    qc1_selected_dl = String(10, nullable=True)  # DL1|DL2 - which one to upload
    qc1_completed_at = DateTime(nullable=True)
    qc1_data_inspection_result = String(50, nullable=True)  # pass|fail
    qc1_inspection_notes = Text(nullable=True)
    
    # Step 4 - QC2 Tracking (Columns Y-Z)
    qc2_assigned_to = String(100, nullable=True)
    qc2_status = String(50, nullable=False, default='pending')  # pending|in_progress|completed
    qc2_imported_dl = String(10, nullable=True)  # Which DL was actually imported
    qc2_imported_at = DateTime(nullable=True)
    qc2_final_approval = String(50, nullable=True)  # approved|rejected
    qc2_completed_at = DateTime(nullable=True)
    
    # Workflow Summary
    workflow_status = String(50, nullable=False, default='step_1')  # step_1|step_2|step_3|step_4|completed|failed
    
    # Timestamps
    created_at = DateTime(nullable=False, default=datetime.utcnow)
    updated_at = DateTime(nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ValidationRecord_DL1(Base):
    """
    DL1 Validation Record - Human-curated reference data
    Step 2a output: manually standardized data by DL1 owner
    Read-only after QC1 selects it
    """
    __tablename__ = 'validation_records_dl1'
    __table_args__ = (
        Index('ix_dl1_race_id_year', 'race_id', 'year'),
        Index('ix_dl1_review_status', 'review_status'),
    )
    
    id = Integer(primary_key=True)
    download_record_id = Integer(ForeignKey('download_records.id'), nullable=True)
    
    # Race & Metadata
    year = Integer(nullable=False)
    state = String(50, nullable=False)
    county = String(100, nullable=False)
    office = String(200, nullable=False)
    race_id = String(100, nullable=False)
    data_tier = String(10, nullable=False, default='DL1')
    source_type = String(50, nullable=False, default='manual_human_entry')
    
    # Candidate Information
    ballot_candidate_name = String(250, nullable=False)  # Raw from source
    standardized_candidate_name = String(250, nullable=False)  # After standardization
    fec_id = String(50, nullable=True)
    is_write_in = Boolean(nullable=False, default=False)
    
    # Party Information
    ballot_party = String(50, nullable=True)
    fec_party = String(10, nullable=True)
    
    # Vote Counts
    uncategorized_votes = Integer(nullable=True)
    early_votes = Integer(nullable=True)
    election_day_votes = Integer(nullable=True)
    mail_votes = Integer(nullable=True)
    provisional_votes = Integer(nullable=True)
    total_votes = Integer(nullable=False)
    
    # Quality Control
    has_flags = Boolean(nullable=False, default=False)
    quality_flags = Text(nullable=True)  # JSON array of flags
    warning_messages = Text(nullable=True)  # JSON array of warnings
    
    # Data Entry
    data_entry_by = String(100, nullable=True)  # DL1 owner who entered data
    entry_notes = Text(nullable=True)
    source_url = Text(nullable=True)
    
    # Review Status
    review_status = SQLEnum(ManualReviewStatus, nullable=False, default=ManualReviewStatus.PENDING)
    reviewed_by = String(100, nullable=True)
    reviewed_at = DateTime(nullable=True)
    review_notes = Text(nullable=True)
    
    # Timestamps
    created_at = DateTime(nullable=False, default=datetime.utcnow)
    updated_at = DateTime(nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    standardized_at = DateTime(nullable=True)


class ValidationRecord_DL2(Base):
    """
    DL2 Validation Record - Auto-enriched from Google Sheets
    Step 2b output: standardized data + enrichment from SMART Elections database
    Can be corrected by QC1/QC2, all changes logged
    """
    __tablename__ = 'validation_records_dl2'
    __table_args__ = (
        Index('ix_dl2_race_id_year', 'race_id', 'year'),
        Index('ix_dl2_review_status', 'review_status'),
    )
    
    id = Integer(primary_key=True)
    download_record_id = Integer(ForeignKey('download_records.id'), nullable=True)
    
    # Race & Metadata
    year = Integer(nullable=False)
    state = String(50, nullable=False)
    county = String(100, nullable=False)
    office = String(200, nullable=False)
    race_id = String(100, nullable=False)
    data_tier = String(10, nullable=False, default='DL2')
    source_type = String(50, nullable=False, default='google_sheets_enriched')
    
    # Candidate Information
    ballot_candidate_name = String(250, nullable=False)  # Raw from source
    standardized_candidate_name = String(250, nullable=False)  # After standardization or enrichment
    fec_id = String(50, nullable=True)
    is_write_in = Boolean(nullable=False, default=False)
    
    # Party Information
    ballot_party = String(50, nullable=True)
    fec_party = String(10, nullable=True)
    
    # Vote Counts
    uncategorized_votes = Integer(nullable=True)
    early_votes = Integer(nullable=True)
    election_day_votes = Integer(nullable=True)
    mail_votes = Integer(nullable=True)
    provisional_votes = Integer(nullable=True)
    total_votes = Integer(nullable=False)
    
    # Quality Control
    has_flags = Boolean(nullable=False, default=False)
    quality_flags = Text(nullable=True)  # JSON array of flags
    warning_messages = Text(nullable=True)  # JSON array of warnings
    auto_flags = Text(nullable=True)  # ML-detected potential issues (JSON)
    
    # Enrichment
    enriched_version = String(50, nullable=True)  # Version of enrichment applied
    enriched_from_row = Integer(nullable=True)  # Source row in Google Sheets
    
    # Data Source
    data_source = String(100, nullable=True)  # 'google_sheets_enriched'|'reformatted_sheet'
    
    # Review Status
    review_status = SQLEnum(ManualReviewStatus, nullable=False, default=ManualReviewStatus.PENDING)
    reviewed_by = String(100, nullable=True)
    reviewed_at = DateTime(nullable=True)
    review_notes = Text(nullable=True)
    
    # Timestamps
    created_at = DateTime(nullable=False, default=datetime.utcnow)
    updated_at = DateTime(nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    standardized_at = DateTime(nullable=True)


class PreQCComparison(Base):
    """
    Pre-QC Auto-check Results
    Strict equality + fuzzy matching comparison of DL1 vs DL2
    """
    __tablename__ = 'preqc_comparisons'
    __table_args__ = (
        Index('ix_preqc_race_id', 'race_id'),
        Index('ix_preqc_status', 'comparison_status'),
    )
    
    id = Integer(primary_key=True)
    download_record_id = Integer(ForeignKey('download_records.id'), nullable=True)
    
    # References
    race_id = String(100, nullable=False)
    dl1_record_id = Integer(ForeignKey('validation_records_dl1.id'), nullable=True)
    dl2_record_id = Integer(ForeignKey('validation_records_dl2.id'), nullable=True)
    
    # Comparison Results
    strict_equality_passed = Boolean(nullable=False, default=False)
    fuzzy_match_confidence = Float(nullable=True)  # 0.0-1.0 overall confidence
    fuzzy_candidate_confidence = Float(nullable=True)
    fuzzy_party_confidence = Float(nullable=True)
    fuzzy_fec_id_confidence = Float(nullable=True)
    
    # Discrepancies
    discrepancy_count = Integer(nullable=False, default=0)
    discrepancy_fields = Text(nullable=True)  # JSON: {field: {dl1, dl2, reason}}
    
    # Summary
    comparison_status = String(50, nullable=False)  # passed|failed|review_needed
    comparison_summary = Text(nullable=True)  # Human-readable summary
    
    # Audit
    checked_by = String(100, nullable=True)
    checked_at = DateTime(nullable=False, default=datetime.utcnow)


class QC1Checkpoint(Base):
    """
    QC1 Designee Review Checkpoint
    Step 3: Data inspection, checklist, and source selection
    """
    __tablename__ = 'qc1_checkpoints'
    __table_args__ = (
        Index('ix_qc1_download_record', 'download_record_id'),
        Index('ix_qc1_reviewer', 'reviewed_by'),
    )
    
    id = Integer(primary_key=True)
    download_record_id = Integer(ForeignKey('download_records.id'), nullable=True)
    preqc_comparison_id = Integer(ForeignKey('preqc_comparisons.id'), nullable=True)
    
    # QC1 Designee & Review
    reviewed_by = String(100, nullable=False)
    reviewed_at = DateTime(nullable=False, default=datetime.utcnow)
    
    # QC1 Checklist Results (from Data Standards workbook)
    qc1_checklist_results = Text(nullable=True)  # JSON: {question_id: answer, ...}
    
    # Data Inspection
    data_inspection_result = String(50, nullable=False)  # pass|fail
    data_inspection_notes = Text(nullable=True)
    
    # Housekeeping Issues (non-blocking)
    housekeeping_issues = Text(nullable=True)  # JSON: [{issue, recommendation}, ...]
    
    # Source Selection
    selected_dl_source = String(10, nullable=False)  # DL1|DL2 - which to upload
    selection_reason = Text(nullable=True)
    
    # Approval
    approved_at = DateTime(nullable=True)
    approval_status = String(50, nullable=False, default='pending')  # pending|approved|rejected


class QC2Checkpoint(Base):
    """
    QC2 Designee Final QC Checkpoint
    Step 4: Import DL file to QC database, ML flagging, final approval
    """
    __tablename__ = 'qc2_checkpoints'
    __table_args__ = (
        Index('ix_qc2_download_record', 'download_record_id'),
        Index('ix_qc2_reviewer', 'reviewed_by'),
    )
    
    id = Integer(primary_key=True)
    download_record_id = Integer(ForeignKey('download_records.id'), nullable=True)
    qc1_checkpoint_id = Integer(ForeignKey('qc1_checkpoints.id'), nullable=True)
    
    # QC2 Designee & Review
    reviewed_by = String(100, nullable=False)
    reviewed_at = DateTime(nullable=False, default=datetime.utcnow)
    
    # Import Details
    imported_dl_file = String(10, nullable=False)  # DL1|DL2 - which was imported
    imported_record_count = Integer(nullable=True)
    import_completed_at = DateTime(nullable=True)
    
    # Data Validation Results
    data_validation_result = Text(nullable=True)  # JSON: validation checks output
    validation_passed = Boolean(nullable=True)
    
    # ML-Detected Issues (auto-flagging for QC2 attention)
    ml_flagged_issues = Text(nullable=True)  # JSON: [{flag, record_id, suggested_action}, ...]
    ml_flags_addressed = Boolean(nullable=False, default=False)
    ml_flags_notes = Text(nullable=True)
    
    # Final Approval
    final_review_result = String(50, nullable=False)  # approved|rejected
    final_review_notes = Text(nullable=True)
    
    # Export to Production
    exported_to_production_at = DateTime(nullable=True)
    production_snapshot_id = String(100, nullable=True)  # Reference for audit trail


class ChainOfCustody(Base):
    """
    Complete Audit Trail
    Every change to election data is logged with who, when, what, and justification
    Ensures full traceability for regulatory compliance
    """
    __tablename__ = 'chain_of_custody'
    __table_args__ = (
        Index('ix_custody_record_id', 'record_id'),
        Index('ix_custody_action_date', 'action_date'),
        Index('ix_custody_performed_by', 'performed_by'),
        Index('ix_custody_action', 'action'),
    )
    
    id = Integer(primary_key=True)
    
    # Reference to Data Record
    record_id = String(100, nullable=True)  # race_id or other identifier
    election_result_id = Integer(ForeignKey('election_results.id'), nullable=True)
    
    # Action Details
    action = String(100, nullable=False)  # created|standardized|enriched|flagged|corrected|approved_qc1|approved_qc2|exported
    field_changed = String(100, nullable=True)  # Which field if applicable
    old_value = Text(nullable=True)
    new_value = Text(nullable=True)
    
    # Context
    description = Text(nullable=True)
    reason = Text(nullable=True)  # Why was this change made
    
    # Accountability
    performed_by = String(100, nullable=False)
    action_date = DateTime(nullable=False, default=datetime.utcnow)
    
    # Source & Classification
    source_table = String(100, nullable=True)  # validation_records_dl1|validation_records_dl2|qc1|qc2|system
    dl_source = String(10, nullable=True)  # DL1|DL2 classification
    
    # Batch Operations
    related_batch_id = String(100, nullable=True)  # For grouping related changes
    
    # Workflow Context
    workflow_step = String(50, nullable=True)  # step_1|step_2|step_3|step_4
