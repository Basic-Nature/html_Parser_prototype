-- ==============================================================================
-- Smart Elections Parser - Verified Data Schema
-- Phase 2: Data Quality Assurance & Verification Framework
-- ==============================================================================
-- 
-- This SQL script creates the PostgreSQL tables for the Data Framework (DL1/DL2).
-- Run this on your Azure PostgreSQL database to enable QA functionality.
--
-- Prerequisites:
--   - Azure Database for PostgreSQL (Flexible Server)
--   - Database created: verified_data
--   - User with CREATE TABLE privileges
--
-- Execution:
--   Azure Portal → PostgreSQL → Query Editor → Paste & Run
--   OR
--   psql -h <host> -U <user> -d verified_data -f create_verified_data_schema.sql
-- ==============================================================================

-- Enable UUID generation extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ==============================================================================
-- 1. VERIFIED_DATASETS (Master Table)
-- ==============================================================================
-- Stores metadata about each verified election dataset (DL1 or DL2)
-- Each row = one parsed election result (contest + office + metadata)

CREATE TABLE IF NOT EXISTS verified_datasets (
    -- Primary Key
    dataset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Source Metadata
    source_url TEXT NOT NULL,
    source_handler VARCHAR(255),  -- e.g., "California General", "Statement of Votes Cast"
    source_type VARCHAR(50),      -- 'HTML', 'PDF', 'API'
    source_hash VARCHAR(64) UNIQUE,  -- SHA-256 of source content (uniqueness check)
    
    -- Geographic Context
    state_abbr VARCHAR(2),        -- 'CA', 'NY', etc.
    county_name VARCHAR(255),
    jurisdiction VARCHAR(255),    -- Statewide, District 5, etc.
    
    -- Election Context
    election_year INT,
    election_date DATE,
    election_type VARCHAR(100),   -- 'General', 'Primary', 'Special'
    
    -- Contest Information
    contest_name VARCHAR(255),    -- "President", "U.S. Representative District 5"
    office_name VARCHAR(255),     -- Normalized office name
    party VARCHAR(100),           -- Party affiliation (if applicable)
    contestant_count INT,         -- Number of candidates/choices
    
    -- Extracted Data Summary
    data_row_count INT,           -- Number of data rows in parsed_results
    headers JSONB,                -- Column headers from parsed data
    metadata JSONB,               -- Flexible storage for handler-specific metadata
    
    -- Data Quality Metrics
    extraction_confidence DECIMAL(5, 2),  -- 0-100 (from parser ML model)
    trust_score DECIMAL(5, 2),            -- 0-100 (from trust scorer)
    completeness_score DECIMAL(5, 2),     -- 0-100 (% of expected fields populated)
    
    -- Data Lineage Status
    dl_status VARCHAR(20) NOT NULL DEFAULT 'DL1',  -- DL1, DL2, REJECTED, DISPUTED
    last_verified_at TIMESTAMP,   -- When it was promoted to DL2 (null if DL1)
    last_verified_by VARCHAR(255),  -- Reviewer principal who verified it
    
    -- Timestamps
    extracted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_datasets_status ON verified_datasets(dl_status);
CREATE INDEX IF NOT EXISTS idx_datasets_state_county ON verified_datasets(state_abbr, county_name);
CREATE INDEX IF NOT EXISTS idx_datasets_year ON verified_datasets(election_year);
CREATE INDEX IF NOT EXISTS idx_datasets_extracted_at ON verified_datasets(extracted_at);
CREATE INDEX IF NOT EXISTS idx_datasets_source_hash ON verified_datasets(source_hash);

-- ==============================================================================
-- 2. QUALITY_ISSUES (Issue Tracking)
-- ==============================================================================
-- Tracks automated QA checks and manual review flags for each dataset

CREATE TABLE IF NOT EXISTS quality_issues (
    -- Primary Key
    issue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    
    -- Issue Classification
    issue_type VARCHAR(100) NOT NULL,  -- 'MISSING_HEADERS', 'DUPLICATE_ROWS', 'IMPOSSIBLE_PERCENTAGE', etc.
    severity VARCHAR(20) NOT NULL,     -- 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    
    -- Issue Details
    description TEXT NOT NULL,
    details JSONB,  -- {affected_rows: [1, 2, 3], field_name: 'vote_count', expected_range: [0, 100000], ...}
    
    -- Resolution
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(255),  -- Reviewer principal who resolved it
    resolution_note TEXT,
    
    -- Timestamps
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (dataset_id) REFERENCES verified_datasets(dataset_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_issues_dataset ON quality_issues(dataset_id);
CREATE INDEX IF NOT EXISTS idx_issues_severity ON quality_issues(severity);
CREATE INDEX IF NOT EXISTS idx_issues_unresolved ON quality_issues(is_resolved) WHERE is_resolved = FALSE;

-- ==============================================================================
-- 3. VERIFICATION_LINEAGE (Audit Trail)
-- ==============================================================================
-- Immutable append-only log of all verification actions (promotions, rejections)

CREATE TABLE IF NOT EXISTS verification_lineage (
    -- Primary Key
    lineage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    
    -- Action Details
    action_type VARCHAR(50) NOT NULL,  -- 'PROMOTED_TO_DL2', 'DEMOTED_TO_DL1', 'REJECTED', 'DISPUTED'
    action_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Attribution
    reviewer_principal VARCHAR(255) NOT NULL,  -- 'user:admin', 'system:auto-classifier', 'cert:election_official'
    privilege_level VARCHAR(50),  -- 'TIER_1_COUNTY', 'TIER_2_STATE', 'TIER_3_FEDERAL'
    
    -- Justification
    certification_reason TEXT,  -- Human-entered reason for promotion/rejection
    
    -- Metadata
    metadata JSONB,  -- {client_ip: '1.2.3.4', user_agent: 'Chrome/...', session_id: 'abc123'}
    
    FOREIGN KEY (dataset_id) REFERENCES verified_datasets(dataset_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_lineage_dataset ON verification_lineage(dataset_id);
CREATE INDEX IF NOT EXISTS idx_lineage_timestamp ON verification_lineage(action_timestamp);
CREATE INDEX IF NOT EXISTS idx_lineage_reviewer ON verification_lineage(reviewer_principal);

-- ==============================================================================
-- 4. DATA_VERSIONS (Version Tracking)
-- ==============================================================================
-- Tracks changes to datasets over time (e.g., source URL updated, re-parsed)

CREATE TABLE IF NOT EXISTS data_versions (
    -- Primary Key
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    
    -- Version Info
    version_number INT NOT NULL,  -- 1, 2, 3, ...
    is_current BOOLEAN DEFAULT TRUE,
    
    -- Diff from Previous
    changes_from_previous JSONB,  -- {field_name: {old_value, new_value}, ...}
    change_summary TEXT,  -- Human-readable summary of what changed
    
    -- When This Version was Created
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (dataset_id) REFERENCES verified_datasets(dataset_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_versions_dataset_current ON data_versions(dataset_id, is_current);

-- ==============================================================================
-- 5. PARSED_RESULTS (Actual Election Data)
-- ==============================================================================
-- Stores the parsed election data (headers, rows, contest info)

CREATE TABLE IF NOT EXISTS parsed_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    
    -- Contest Info
    office_name VARCHAR(255),
    party VARCHAR(100),
    jurisdiction VARCHAR(255),  -- Statewide, District 5, etc.
    
    -- Candidate/Result Row
    candidate_name VARCHAR(255),
    party_affiliation VARCHAR(100),
    vote_count INT,
    percentage DECIMAL(7, 4),  -- 0-100 with decimals
    status VARCHAR(100),  -- 'Reported', 'Unreported', etc.
    
    -- Data Lineage
    source_row_index INT,  -- Position in original parsed data
    confidence_score DECIMAL(5, 2),  -- Confidence in this specific extraction
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (dataset_id) REFERENCES verified_datasets(dataset_id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_results_dataset ON parsed_results(dataset_id);
CREATE INDEX IF NOT EXISTS idx_results_candidate ON parsed_results(candidate_name);
CREATE INDEX IF NOT EXISTS idx_results_office ON parsed_results(office_name);

-- ==============================================================================
-- SCHEMA CREATION COMPLETE
-- ==============================================================================

-- Verify tables were created
SELECT 
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_schema = 'public'
  AND table_name IN (
    'verified_datasets',
    'quality_issues',
    'verification_lineage',
    'data_versions',
    'parsed_results'
  )
ORDER BY table_name;

-- Expected output: 5 tables with column counts
-- verified_datasets: ~30 columns
-- quality_issues: ~11 columns
-- verification_lineage: ~9 columns
-- data_versions: ~7 columns
-- parsed_results: ~12 columns
