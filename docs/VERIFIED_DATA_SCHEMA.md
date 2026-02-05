# PostgreSQL Schema: Verified Data Management (DL1/DL2 Classification)

**Purpose**: Store election data through quality assurance pipeline (DL1 unverified → DL2 verified), track lineage, audit all decisions.

---

## Core Tables

### 1. `verified_datasets` (Primary Records)

Stores each parsed election dataset with classification status.

```sql
CREATE TABLE verified_datasets (
    -- Primary Key
    dataset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Classification & Status
    dl_status VARCHAR(20) NOT NULL CHECK (dl_status IN ('DL1', 'DL2', 'REJECTED', 'DISPUTED')),
    -- DL1 = Unverified (auto-extracted, awaiting manual review)
    -- DL2 = Verified (human approved + QA passed)
    -- REJECTED = Invalid/disputed data, rejected by reviewer
    -- DISPUTED = DL2 data flagged as having anomalies, pending re-review
    
    -- Source Information
    source_url VARCHAR(2048) NOT NULL,
    source_handler VARCHAR(128) NOT NULL,  -- e.g., 'html_handler', 'pdf_handler'
    source_session_id VARCHAR(255),  -- Links to ballot_lens session if parsed via UI
    
    -- Location Identifiers
    state_abbr CHAR(2) NOT NULL,  -- 'CA', 'TX', etc.
    county_name VARCHAR(255),     -- NULL for statewide contests
    election_year INT NOT NULL,
    election_type VARCHAR(50),    -- 'General', 'Primary', 'Special', etc.
    
    -- Data Content Summary
    contest_name VARCHAR(255) NOT NULL,  -- 'President', 'Governor', etc.
    contestant_count INT,  -- Number of candidates/options
    data_row_count INT,    -- Number of records (rows)
    
    -- Quality Metrics
    extraction_confidence DECIMAL(5, 2),  -- 0-100 from parser
    trust_score DECIMAL(5, 2),  -- 0-100 from url_trust_scorer
    completeness_score DECIMAL(5, 2),  -- % of fields populated
    
    -- Automated QA Results
    automated_qa_passed BOOLEAN DEFAULT NULL,  -- NULL = not yet run, TRUE/FALSE = result
    detected_issues_count INT DEFAULT 0,
    
    -- Timestamps
    extracted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexing for efficient queries
    UNIQUE(source_url, extracted_at),
    INDEX idx_dl_status (dl_status),
    INDEX idx_state_county_year (state_abbr, county_name, election_year),
    INDEX idx_contest (contest_name),
    INDEX idx_classification_date (classified_at)
);
```

---

### 2. `quality_issues` (Automated QA Detection)

Stores results from automated quality checks.

```sql
CREATE TABLE quality_issues (
    issue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    
    -- Issue Classification
    issue_type VARCHAR(64) NOT NULL,  -- 'duplicate_row', 'invalid_vote_count', 'impossible_percentage', 'missing_field', 'anomaly_detected', etc.
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    
    -- Issue Details
    description TEXT,
    affected_field VARCHAR(255),  -- e.g., 'vote_count', 'percentage', 'candidate_name'
    affected_rows TEXT,  -- JSON array of row indices or candidate IDs affected
    
    -- Confidence Score
    confidence_score DECIMAL(5, 2),  -- 0-100: How confident is the system this is actually an issue?
    
    -- Remediation
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_by_reviewer_principal VARCHAR(255),  -- If manually cleared by reviewer
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    
    -- Metadata
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dataset (dataset_id),
    INDEX idx_issue_type (issue_type),
    INDEX idx_severity (severity),
    
    FOREIGN KEY (dataset_id) REFERENCES verified_datasets(dataset_id) ON DELETE CASCADE
);
```

---

### 3. `verification_lineage` (Audit Trail)

Immutable log of every decision/action on a dataset.

```sql
CREATE TABLE verification_lineage (
    lineage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    
    -- What Action Occurred
    action_type VARCHAR(64) NOT NULL,  -- 'classification', 'auto_qa_performed', 'flagged_for_review', 'promoted_to_dl2', 'rejected', 'anomaly_detected', 'resolved'
    action_status VARCHAR(32) NOT NULL,  -- 'pending', 'in_progress', 'completed', 'failed'
    
    -- Who Performed the Action (if human)
    reviewer_principal VARCHAR(255),  -- e.g., 'john.reviewer@elections.gov'
    reviewer_role VARCHAR(50),  -- 'REVIEWER', 'QA_OFFICER', 'ADMIN'
    certification_reason TEXT,  -- Why they made this decision
    
    -- Metadata About the Action
    confidence_score DECIMAL(5, 2),
    details JSONB,  -- Flexible payload: {dl1_confidence: 92, reasons: [...], flags: [...]}
    
    -- Timestamps (immutable)
    action_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dataset (dataset_id),
    INDEX idx_action_type (action_type),
    INDEX idx_reviewer (reviewer_principal),
    
    FOREIGN KEY (dataset_id) REFERENCES verified_datasets(dataset_id) ON DELETE CASCADE
);
```

---

### 4. `data_versions` (Historical Tracking)

If same election data is re-parsed, track updates and comparisons.

```sql
CREATE TABLE data_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID NOT NULL,
    
    -- Version Info
    version_number INT NOT NULL,  -- 1, 2, 3 for same election
    is_current BOOLEAN DEFAULT TRUE,
    
    -- Diff from Previous
    changes_from_previous JSONB,  -- {field_name: {old_value, new_value}, ...}
    change_summary TEXT,  -- Human-readable summary of what changed
    
    -- When This Version was Created
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_dataset_current (dataset_id, is_current),
    FOREIGN KEY (dataset_id) REFERENCES verified_datasets(dataset_id) ON DELETE CASCADE
);
```

---

### 5. `parsed_results` (Actual Election Data)

Stores the parsed election data (headers, rows, contest info).

```sql
CREATE TABLE parsed_results (
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
    INDEX idx_dataset (dataset_id),
    INDEX idx_candidate (candidate_name),
    INDEX idx_office (office_name),
    
    FOREIGN KEY (dataset_id) REFERENCES verified_datasets(dataset_id) ON DELETE CASCADE
);
```

---

## SQL Initialization Script

```sql
-- Create schema
CREATE SCHEMA IF NOT EXISTS verified_data;

-- Create tables (paste individual CREATE TABLE statements above)
-- Ensure indexes are created for performance

-- Sample migration: Transition from fixtures to PostgreSQL
INSERT INTO verified_datasets (
    source_url, source_handler, state_abbr, county_name, election_year,
    contest_name, contestant_count, data_row_count, extraction_confidence,
    trust_score, completeness_score, dl_status
)
SELECT 
    source_url, handler_name, state, county, year,
    contest, contestant_count, row_count, extraction_confidence,
    trust_score, 1.0, 'DL1'  -- Mark all as DL1 (unverified) initially
FROM fixtures.election_results_index
WHERE year >= 2024;  -- Load recent data only
```

---

## Key Design Decisions

1. **Immutable Lineage**: `verification_lineage` table is append-only. Every decision is permanently logged with timestamp + principal attribution.

2. **DL Classification States**:
   - **DL1 (Unverified)**: Freshly parsed, awaiting manual review
   - **DL2 (Verified)**: Human approved + all auto QA checks passed
   - **REJECTED**: Human marked as invalid/disputed
   - **DISPUTED**: Was DL2, but later anomalies flagged it for re-review

3. **Flexible Issue Tracking**: `quality_issues` captures both automated checks and manual flags (e.g., "duplicate record", "impossible percentage").

4. **JSONB Details Column**: Allows flexible schema evolution (different types of issues can store different metadata).

5. **Referential Integrity**: Foreign keys ensure orphaned records can't exist. ON DELETE CASCADE cleans up lineage + issues if dataset deleted.

6. **Indexing Strategy**: Indexes on frequently queried fields (status, state/county, dates) for performance on large datasets.

---

## Queries You'll Use

### Get all DL2 (verified) data for a state

```sql
SELECT ds.dataset_id, ds.contest_name, ds.county_name, ds.extracted_at,
       ds.extraction_confidence, ds.trust_score
FROM verified_datasets ds
WHERE ds.state_abbr = 'CA'
  AND ds.dl_status = 'DL2'
  AND ds.election_year = 2024
ORDER BY ds.extracted_at DESC;
```

### Get unresolved QA issues

```sql
SELECT qi.issue_type, COUNT(*) as count
FROM quality_issues qi
WHERE qi.dataset_id IN (SELECT dataset_id FROM verified_datasets WHERE dl_status = 'DL1')
  AND qi.is_resolved = FALSE
GROUP BY qi.issue_type
ORDER BY count DESC;
```

### Get full audit trail for a dataset

```sql
SELECT action_type, action_timestamp, reviewer_principal, certification_reason
FROM verification_lineage
WHERE dataset_id = $1
ORDER BY action_timestamp ASC;
```

### Check if dataset was updated (version tracking)

```sql
SELECT version_number, change_summary, created_at
FROM data_versions
WHERE dataset_id = $1
ORDER BY version_number ASC;
```

---

## Environment Configuration

Add to `.env`:

```bash
# PostgreSQL for Verified Data
VERIFIED_DATA_DB_HOST=ballotlens-server.postgres.database.azure.com
VERIFIED_DATA_DB_PORT=5432
VERIFIED_DATA_DB_NAME=verified_data
VERIFIED_DATA_DB_USER=verified_user
VERIFIED_DATA_DB_PASSWORD=<secure-password>

# Schema prefix
VERIFIED_DATA_SCHEMA=verified_data
```

---

## Migration Path (Fixtures → PostgreSQL)

1. **Phase 1** (Current): Use fixtures as local cache, write operations go to PostgreSQL
2. **Phase 2**: Sync fixture index → PostgreSQL on app startup
3. **Phase 3**: PostgreSQL as primary source of truth, fixtures optional backup only
4. **Phase 4**: Retire fixtures, use PostgreSQL exclusively

This allows gradual transition without losing historical data.
