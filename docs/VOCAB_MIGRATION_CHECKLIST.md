# Constants.py Migration Checklist: Risk Mitigation & Election Integrity

## Executive Summary

This checklist guides the migration from monolithic `webapp/parser/config.py` constants to compartmentalized, security-hardened **VocabLoader-backed** entity files. Every step includes **safety checkpoints** to ensure:

- ✅ No malicious modification of vocab files (integrity checks)
- ✅ Only verified entities accepted (trust scoring)
- ✅ Full audit trail for all vocab access (provenance tracking)
- ✅ Immutability during live election scanning (election_mode locking)
- ✅ Backward compatibility with existing handlers (migration shims)

---

## Phase 0: Pre-Migration Audit

### ☐ 0.1 Inventory Current Constants

**Objective**: Catalog all constants currently in `webapp/parser/config.py` and related files.

**Steps**:

1. Run grep to find all constant definitions:

   ```bash
   grep -r "^[A-Z_]*\s*=" webapp/parser/config.py webapp/parser/*.py | head -50
   ```

2. Categorize by domain:
   - **Jurisdictional**: State, county, district names
   - **Political**: Party names, office titles, candidate attributes
   - **Technical**: File extensions, MIME types, database configs
   - **Safety/Trust**: Allowlist hosts, verified source URLs, retry limits
3. **Document in**: `docs/CONSTANTS_INVENTORY.md`
4. **Output format**:

   ```markdown
   | Category | Constant Name | Current Value | Target Vocab File | Trust Level |
   |----------|---------------|---------------|-------------------|-------------|
   | Jurisdiction | STATE_NAMES | list(...) | entities/jurisdictions.txt | high |
   | Political | VALID_OFFICES | list(...) | entities/offices.txt | high |
   ```

**Safety Checkpoint**:

- [ ] Inventory complete (all constants identified)
- [ ] Categorization reviewed by domain expert
- [ ] Trust levels assigned (high/medium/low)
- [ ] No sensitive credentials in inventory (passwords, API keys must stay in config)

---

### ☐ 0.2 Extract Verified Source Information

**Objective**: Map each constant's source to official election authority or trusted registry.

**Steps**:

1. For each constant, determine authoritative source:
   - Official Secretary of State API? → trust_score = 0.99
   - Official county website? → trust_score = 0.95
   - Manual curation from election officials? → trust_score = 0.90
   - Legacy data or best-guess? → trust_score = 0.50 (quarantine)
2. Record URL + verification method in source tracking spreadsheet:

   ```txt
   Constant: STATE_ABBREVIATIONS
   Source URLs:
     - https://www.sos.ca.gov/elections/candidates/candidate-data/ (FIPS official)
     - https://www.sos.gov/fips/ (federal FIPS registry)
   Trust Score: 0.99
   Updated: 2026-01-15
   ```

3. **Document in**: `webapp/parser/Context_Integration/vocab/sources/verified_sources.txt` (template)

**Safety Checkpoint**:

- [ ] Every constant traced to authoritative source
- [ ] Trust scores assigned based on verification method
- [ ] High-trust sources (≥0.85) identified for election_mode
- [ ] Low-trust sources (< 0.50) flagged for review

---

### ☐ 0.3 Set Up Vocabulary Folder Structure

**Objective**: Create the 8-folder compartmentalized structure.

**Steps**:

1. Create directories:

   ```tree
   webapp/parser/Context_Integration/vocab/
   ├── entities/
   ├── validators/
   ├── sources/
   ├── scoring/
   ├── snapshots/
   ├── manifest.md
   └── README.md
   ```

2. Create `manifest.md` template (see VOCAB_LOADER_API_SPECIFICATION.md §14)
3. Create `README.md` with folder descriptions
4. Initialize git tracking:

   ```bash
   git add webapp/parser/Context_Integration/vocab/
   git commit -m "feat: Initialize vocabulary folder structure for ML/NLP advancement"
   ```

**Safety Checkpoint**:

- [ ] All 5 folders created with proper permissions (rwx for parser user)
- [ ] manifest.md template initialized (empty, awaiting file checksums)
- [ ] README.md documents each folder's purpose
- [ ] Git history clean (no sensitive data in commit)

---

## Phase 1: Populate Vocabulary Files

### ☐ 1.1 Populate entities/offices.txt

**File Purpose**: Canonical office names (no aliases, one per line).

**Steps**:

1. Extract from current config:

   ```python
   # OLD config.py
   VALID_OFFICES = [
       "President",
       "Vice President",
       "Senator",
       "Representative",
       "Governor",
       "State Senator",
       "State Representative",
       # ...
   ]
   ```

2. Convert to newline-separated format:

   ```text
   # Official office titles (US + State)
   # Source: Secretary of State registries, curated by Election Integrity Team
   President
   Vice President
   Senator
   House Representative
   Governor
   Lieutenant Governor
   Attorney General
   State Senator
   State Representative
   County Supervisor
   County Board Member
   ```

3. Add header comments with source metadata:

   ```text
   # Authority: Official Secretary of State registries + election codes
   # Updated: 2026-01-15
   # Trust Score: 0.99
   # Last Verified By: elections@ca.gov
   ```

4. Compute SHA-256 hash and record in manifest:

   ```bash
   sha256sum webapp/parser/Context_Integration/vocab/entities/offices.txt
   # Output: 3a5c7f2e... → Record in manifest.md
   ```

**Safety Checkpoint**:

- [ ] File contains only canonical forms (no "Pres.", "Rep.", etc.)
- [ ] No duplicates
- [ ] SHA-256 hash matches manifest
- [ ] Comments document source and trust level
- [ ] File has Unix line endings (LF, not CRLF)

---

### ☐ 1.2 Populate entities/parties.txt

**File Purpose**: Canonical party names.

**Steps**:

1. Extract from config:

   ```python
   # OLD config.py
   VALID_PARTIES = ["Democratic", "Republican", "Independent", "Green", "Libertarian"]
   ```

2. Convert to newline-separated (no aliases):

   ```text
   # Political parties recognized in official election results
   # Authority: State election codes + federal FEC registries
   # Updated: 2026-01-15
   # Trust Score: 0.98
   Democratic Party
   Republican Party
   Green Party
   Libertarian Party
   American Independent
   Peace and Freedom Party
   American Solidarity Party
   United States Pirate Party
   ```

3. Compute hash and record in manifest

**Safety Checkpoint**:

- [ ] Only canonical party names (not "Dem", "GOP", "Ind")
- [ ] Matches official state election commission registries
- [ ] No third-party data injection
- [ ] Hash verified against manifest

---

### ☐ 1.3 Populate entities/jurisdictions.txt

**File Purpose**: State and county canonical names.

**Steps**:

1. Extract and canonicalize:

   ```python
   # OLD config.py
   STATES = ["California", "New York", "Texas", ...]
   COUNTIES_CA = ["Los Angeles", "San Francisco", ...]
   ```

2. Create hierarchical format:

   ```text
   # US States (canonical names)
   Alabama
   Alaska
   Arizona
   Arkansas
   California
   Colorado
   # ... all 50 states
   
   # California Counties
   California/Alameda County
   California/Alpine County
   California/Amador County
   California/Butte County
   # ... all CA counties
   ```

3. Compute hash and record

**Safety Checkpoint**:

- [ ] All 50 states present (exact spelling, capitalization)
- [ ] All counties for target state(s) present
- [ ] Hierarchical format preserved (State/County)
- [ ] No abbreviations (CA → California)
- [ ] Hash verified

---

### ☐ 1.4 Populate entities/contest_types.txt

**File Purpose**: Election contest types.

**Steps**:

```text
# Contest type definitions
# Authority: State election codes
# Updated: 2026-01-15
# Trust Score: 0.99
General Election
Primary Election
Special Election
Runoff Election
Special Runoff Election
```

**Safety Checkpoint**:

- [ ] Matches state election code definitions
- [ ] No proprietary/made-up types
- [ ] Hash verified

---

### ☐ 1.5 Populate entities/result_terms.txt

**File Purpose**: Common result/vote terms for header matching.

**Steps**:

```text
# Common terms in election result tables
# Purpose: Vocabulary for header normalization
# Authority: Common usage in official SoS result exports
# Updated: 2026-01-15
# Trust Score: 0.90
Candidate
Candidate Name
Vote Count
Votes
Votes Received
Votes Cast
Total Votes
Percentage
Vote Percentage
Margin
Total Ballots
Ballots Cast
Valid Ballots
Registered Voters
Voter Turnout
```

**Safety Checkpoint**:

- [ ] Terms match official state result exports
- [ ] Hash verified

---

### ☐ 1.6 Populate validators/state_aliases.txt

**File Purpose**: State name variations → canonical form.

**Format**: `alias -> canonical`

**Steps**:

```text
# State name aliases and abbreviations
# Authority: FIPS standard + manual curation
# Updated: 2026-01-15
# Trust Score: 0.95
CA -> California
California -> California
Calif. -> California
CA. -> California
Calif -> California
CA/ -> California
NY -> New York
New York -> New York
N.Y. -> New York
TX -> Texas
Texas -> Texas
TX. -> Texas
# ... all 50 states with variants
```

**Safety Checkpoint**:

- [ ] All common abbreviations covered
- [ ] No conflicting mappings (CA → only California)
- [ ] Canonical form appears in entities/jurisdictions.txt
- [ ] Hash verified

---

### ☐ 1.7 Populate validators/county_aliases.txt

**File Purpose**: County name variations → canonical form.

**Format**: `alias -> canonical`

**Steps**:

```text
# California county name aliases
# Authority: California Government Code §26000+
# Updated: 2026-01-15
# Trust Score: 0.98
LA -> California/Los Angeles County
Los Angeles -> California/Los Angeles County
Los Angeles Co. -> California/Los Angeles County
Los Angeles County -> California/Los Angeles County
SF -> California/San Francisco County
San Francisco -> California/San Francisco County
San Francisco Co. -> California/San Francisco County
SF County -> California/San Francisco County
# ... all CA counties with variants
```

**Safety Checkpoint**:

- [ ] Aliases map to canonical jurisdictions (full path with state)
- [ ] No circular references (A → B → A)
- [ ] Common abbreviations + full names covered
- [ ] Hash verified

---

### ☐ 1.8 Populate sources/verified_sources.txt

**File Purpose**: Approved election result source URLs.

**Format**:

```txt
URL|Trust Score|Source Type|Jurisdiction|Last Verified
```

**Steps**:

```text
# Verified Election Result Sources
# Purpose: Trust scoring for incoming data URLs
# Updated: 2026-02-03
# Maintenance: Update weekly during election season
https://results.sos.ca.gov/2024/general|0.99|official_sos_api|California|2026-02-03T10:00:00Z
https://www.sos.ca.gov/elections/results/|0.98|official_sos_website|California|2026-02-03T10:00:00Z
https://elections.los-angeles-county.gov/results|0.96|official_county|California/Los Angeles County|2026-02-03T10:00:00Z
https://elections.sfgov.org/results|0.96|official_county|California/San Francisco County|2026-02-03T10:00:00Z
# ... one URL per line
# DO NOT ADD: cloud storage (Dropbox, Drive), personal websites, etc.
```

**Safety Checkpoint**:

- [ ] Only official election authority URLs
- [ ] No cloud storage (Dropbox, Google Drive, etc.)
- [ ] Trust scores justified (0.99 = official SoS API; 0.95 = official county website)
- [ ] Jurisdiction matched to URL scope
- [ ] Last verified timestamp updated weekly during election season
- [ ] Hash verified

---

### ☐ 1.9 Populate scoring/trust_signals.txt

**File Purpose**: Keyword signal → confidence contribution.

**Format**:

```txt
Signal Name|Confidence Delta|Context|Reason
```

**Steps**:

```text
# Trust signals for keyword scoring
# Purpose: Confidence calculation for header matching
# Authority: Election Integrity Team
# Updated: 2026-02-03
exact_match|+0.95|header_match|Exact word match in canonical headers
substring_match|+0.70|header_match|Keyword contains canonical term (e.g., "Total Votes" contains "Votes")
abbreviation_match|+0.60|header_match|Common abbreviation (e.g., "Cand." matches "Candidate")
fuzzy_match_90|+0.45|header_match|Levenshtein distance ≥ 90% (e.g., "Candidate" vs "Candidat")
fuzzy_match_80|+0.20|header_match|Levenshtein distance ≥ 80%
unmatched_keyword|-0.30|header_mismatch|Keyword not found in canonical set
suspicious_keyword|-0.50|anomaly|Keyword suggests data manipulation (e.g., "HiddenVotes")
verified_source_match|+0.20|source|URL matches verified source registry
unverified_source|-0.40|source|URL not in verified registry
```

**Safety Checkpoint**:

- [ ] All signals weighted reasonably (sum should be 0 if balanced)
- [ ] Negative scores for suspicious patterns
- [ ] Context documented (which anomaly reason each signal applies to)
- [ ] Hash verified

---

### ☐ 1.10 Populate scoring/anomaly_reasons.txt

**File Purpose**: Anomaly reason code definitions.

**Format**:

```txt
Reason Code|Severity|Quarantine Required|Eligible for Ingestion|Description
```

**Steps**:

```text
# Anomaly reason code definitions
# Purpose: Contextual logging and ingestion eligibility determination
# Authority: Election Integrity Team + state election code
# Updated: 2026-02-03
mismatched_totals|high|yes|no|Sum of candidate votes does not match reported total
missing_candidate|high|yes|no|Expected candidate from verified registry not found in results
suspicious_header|medium|yes|maybe|Header names don't match expected pattern
unrecognized_contest|medium|yes|maybe|Contest type not in canonical registry
empty_table|high|yes|no|Result table has no data rows
header_mismatch|low|no|yes|Header column names differ from expected (but semantically correct)
confidence_below_threshold|medium|yes|maybe|Overall extraction confidence < 0.70
unverified_source|low|no|yes|Result source URL not in verified registry
late_timestamp|low|no|yes|Result timestamp older than threshold
duplicate_entry|medium|yes|maybe|Result already processed (detected via content hash)
```

**Safety Checkpoint**:

- [ ] HIGH severity anomalies have quarantine_required=yes
- [ ] HIGH severity anomalies have eligible_for_ingestion=no
- [ ] MEDIUM severity anomalies reviewed individually before ingestion
- [ ] Description explains business logic (not just technical details)
- [ ] Hash verified

---

### ☐ 1.11 Populate snapshots/table_headers.txt

**File Purpose**: Common header column variations from real-world result exports.

**Steps**:

```text
# Common column header names in official election result tables
# Purpose: Header normalization and pattern matching
# Authority: Observed in real SoS result exports + official templates
# Updated: 2026-02-03
# Trust Score: 0.85
Candidate
Candidate Name
Full Name
Name
Office
Office Title
Race
Contest
Party
Party Affiliation
Votes
Votes Received
Votes Cast
Vote Count
Total
Total Votes
Percent
Percentage
%
Vote %
Vote Percentage
Margin
Blank Votes
Overvote
Invalid
Total Ballots
Ballots Cast
Registered Voters
Turnout
Precincts Reporting
Precincts Total
```

**Safety Checkpoint**:

- [ ] Headers reflect real-world data
- [ ] Normalized to common case (Title Case)
- [ ] Hash verified

---

## Phase 2: Implement VocabLoader Class

### ☐ 2.1 Create vocab_loader.py

**Objective**: Implement full VocabLoader class per API specification.

**Steps**:

1. Create `webapp/parser/Context_Integration/vocab_loader.py` with:
   - `__init__()` with security context (session_id, principal, trust_threshold, election_mode)
   - `load_vocab_set()` with hash verification and trust filtering
   - `resolve_alias()` with exact + fuzzy matching
   - `get_verified_source()` for trust scoring
   - `score_keyword_combination()` for anomaly confidence
   - `get_anomaly_reason_definition()` for ingestion eligibility
   - `lock_election_mode()` and `create_election_snapshot()` for election day scanning
   - Full audit logging to JSONL

2. Implement exception hierarchy:
   - `VocabLoaderError`
   - `VocabFileNotFound`
   - `VocabIntegrityError` (tampering detected)
   - `VocabSecurityError` (unverified entity in election_mode)
   - `VocabContextNotFound`

3. Add caching layer (in-memory with TTL):
   - Lazy loading on first access
   - 1-hour TTL (frozen during election_mode)
   - Memory cap: 50 MB with LRU eviction

**Safety Checkpoint**:

- [ ] All exception types defined
- [ ] Hash verification implemented (calls SHA-256)
- [ ] Audit logging implemented (JSONL persistence)
- [ ] Caching strategy correctly freezes during election_mode
- [ ] No hardcoded credentials or secrets in code
- [ ] Unit tests pass (file loading, hash verification, exception handling)

---

### ☐ 2.2 Create Unit Tests for VocabLoader

**Objective**: Comprehensive test coverage for all VocabLoader methods.

**Test Cases**:

```python
# tests/unit/test_vocab_loader.py

class TestVocabLoader:
    
    def test_load_vocab_set_success(self):
        """Load valid vocab file, verify contents"""
        loader = VocabLoader()
        offices = loader.load_vocab_set("entities/offices.txt")
        assert "President" in offices
        assert len(offices) > 0
    
    def test_load_vocab_set_hash_mismatch(self):
        """Detect tampered file (hash mismatch)"""
        # Modify file content
        # Call loader.load_vocab_set()
        # Assert raises VocabIntegrityError
        pass
    
    def test_load_vocab_set_file_not_found(self):
        """Raise VocabFileNotFound for missing file"""
        loader = VocabLoader()
        with pytest.raises(VocabFileNotFound):
            loader.load_vocab_set("entities/nonexistent.txt")
    
    def test_resolve_alias_exact_match(self):
        """Resolve state abbreviation to canonical form"""
        loader = VocabLoader()
        result = loader.resolve_alias("state", "CA")
        assert result == "California"
    
    def test_resolve_alias_fuzzy_match(self):
        """Fuzzy match with confidence penalty"""
        loader = VocabLoader()
        result = loader.resolve_alias("state", "Californa", exact_match=False)
        assert result == "California"
        # Note: fuzzy match has lower trust score in audit log
    
    def test_resolve_alias_unverified_in_election_mode(self):
        """Raise VocabSecurityError for unverified alias during election_mode"""
        loader = VocabLoader(election_mode=True)
        # Try to resolve unverified alias
        with pytest.raises(VocabSecurityError):
            loader.resolve_alias("state", "UnknownState")
    
    def test_get_verified_source_success(self):
        """Return trust metadata for verified source"""
        loader = VocabLoader()
        source = loader.get_verified_source("https://results.sos.ca.gov/2024/general")
        assert source["trust_score"] >= 0.85
        assert source["verified_source_type"] == "official_sos_api"
    
    def test_get_verified_source_not_found(self):
        """Return None for unknown source"""
        loader = VocabLoader()
        source = loader.get_verified_source("https://unknown.example.com")
        assert source is None
    
    def test_score_keyword_combination_exact_match(self):
        """High score for exact header match"""
        loader = VocabLoader()
        score = loader.score_keyword_combination(
            keywords=["Candidate", "Votes"],
            context="header_validation"
        )
        assert score["base_score"] > 0.85
        assert score["confidence"] == "high"
    
    def test_score_keyword_combination_suspicious(self):
        """Low score for unmatched keywords"""
        loader = VocabLoader()
        score = loader.score_keyword_combination(
            keywords=["HiddenVotes", "SecretData"],
            context="header_validation"
        )
        assert score["base_score"] < 0.50
        assert score["confidence"] == "low"
    
    def test_election_mode_locks_files(self):
        """Prevent modifications during election_mode"""
        loader = VocabLoader(election_mode=True)
        # Simulate file modification attempt
        with pytest.raises(VocabSecurityError):
            # Try to load modified file
            pass
    
    def test_create_election_snapshot(self):
        """Create point-in-time snapshot with immutable reference"""
        loader = VocabLoader(election_mode=True)
        snapshot = loader.create_election_snapshot(
            snapshot_time="2026-11-03T15:45:00Z",
            completion_percentage=87.45,
            source_url="https://results.sos.ca.gov/2024/general"
        )
        assert snapshot["snapshot_id"].startswith("snap_")
        assert snapshot["completion_percentage"] == 87.45
        assert snapshot["vocab_version_hash"] is not None
    
    def test_audit_logging(self):
        """Verify all operations logged to JSONL"""
        loader = VocabLoader(session_id="test_session")
        loader.load_vocab_set("entities/offices.txt")
        
        # Check audit log file exists
        audit_file = Path("logs/vocab_audit.jsonl")
        assert audit_file.exists()
        
        # Verify log entry structure
        with open(audit_file, 'r') as f:
            last_line = f.readlines()[-1]
            log_entry = json.loads(last_line)
            assert log_entry["session_id"] == "test_session"
            assert log_entry["operation"] == "load_vocab_set"
            assert log_entry["result"] == "success"
    
    def test_access_control_public_read_blocked(self):
        """Prevent public (unauthenticated) reads when configured"""
        loader = VocabLoader(
            require_principal=True,
            principal=None  # No principal
        )
        with pytest.raises(VocabSecurityError):
            loader.load_vocab_set("entities/offices.txt")
    
    def test_rate_limiting(self):
        """Enforce per-session rate limit on vocab loads"""
        loader = VocabLoader(session_id="rate_test")
        
        # Load up to rate limit (100 loads/min)
        for i in range(100):
            loader.load_vocab_set("entities/offices.txt")
        
        # 101st load should raise RateLimitError
        with pytest.raises(RateLimitError):
            loader.load_vocab_set("entities/offices.txt")
```

**Safety Checkpoint**:

- [ ] All 15+ test cases pass
- [ ] Coverage > 90% for VocabLoader class
- [ ] Hash verification tests confirm tampering detection
- [ ] Election mode locking tested
- [ ] Audit logging verified
- [ ] Access control tested
- [ ] Rate limiting tested

---

### ☐ 2.3 Compute File Hashes and Update Manifest

**Objective**: Generate SHA-256 hashes for all vocab files and record in manifest.

**Steps**:

```bash
# For each vocab file, compute hash and record in manifest.md
cd webapp/parser/Context_Integration/vocab/

# entities/offices.txt
sha256sum entities/offices.txt
# 3a5c7f2e9b1d4c6a8f2e5b7c1a9d3f6e9b1a4c7 → Record in manifest

# All other files...
sha256sum entities/*.txt validators/*.txt sources/*.txt scoring/*.txt snapshots/*.txt

# Update manifest.md with table of file paths and hashes
```

1. Create `manifest.md` with table:

```markdown
| File | SHA-256 | Size | Updated | Source |
|------|---------|------|---------|--------|
| entities/offices.txt | 3a5c7f2e... | 1.2 KB | 2026-02-03 | official_sos |
| entities/parties.txt | 7b2f9c1a... | 0.8 KB | 2026-02-03 | official_sos |
| ... | ... | ... | ... | ... |
```

1. Commit to git:

```bash
git add webapp/parser/Context_Integration/vocab/manifest.md
git commit -m "feat: Add vocab manifest with SHA-256 hashes for integrity verification"
```

**Safety Checkpoint**:

- [ ] All vocab files have computed hashes
- [ ] Hashes recorded in manifest.md
- [ ] manifest.md committed to git (immutable record)
- [ ] VocabLoader can verify hashes on load

---

## Phase 3: Refactor constants.py to Helper Module

### ☐ 3.1 Create Backward Compatibility Shim

**Objective**: Allow existing handlers to import from config.py without modification (during transition).

**Steps**:

1. Rename current `config.py` to `config_legacy.py` (for reference during transition)
2. Create new `config.py` that loads from VocabLoader:

```python
# webapp/parser/config.py (refactored)

from webapp.parser.Context_Integration.vocab_loader import VocabLoader
import logging

logger = logging.getLogger(__name__)

# Initialize loader once at module import time
_loader = VocabLoader(
    vocab_root="webapp/parser/Context_Integration/vocab",
    trust_threshold=0.75,  # Reasonable default
    enable_audit_logging=True
)

# OLD API (deprecated, but still works)
# Handlers import: from webapp.parser.config import VALID_OFFICES
# This now loads from vocab files instead of static list

VALID_OFFICES = _loader.load_vocab_set("entities/offices.txt")
VALID_PARTIES = _loader.load_vocab_set("entities/parties.txt")
VALID_STATES = _loader.load_vocab_set("entities/jurisdictions.txt")  # States only
# ... etc for all constants

# NEW API (preferred going forward)
# Handlers import: from webapp.parser.config import get_vocab_loader
def get_vocab_loader(session_id=None, trust_threshold=0.75, election_mode=False):
    """Return a VocabLoader instance with optional session context."""
    return VocabLoader(
        session_id=session_id,
        trust_threshold=trust_threshold,
        election_mode=election_mode,
        enable_audit_logging=True
    )

# Utility functions for common queries
def get_office_by_name(office_name: str, context: str | None = None) -> str | None:
    """Alias resolution wrapper."""
    loader = get_vocab_loader()
    return loader.resolve_alias("office", office_name, exact_match=True)

def get_state_canonical(state_alias: str) -> str | None:
    """State alias resolution wrapper."""
    loader = get_vocab_loader()
    return loader.resolve_alias("state", state_alias)

def verify_result_source(source_url: str) -> dict | None:
    """Verify source URL against trusted registry."""
    loader = get_vocab_loader()
    return loader.get_verified_source(source_url)

# ... etc
```

1. Test backward compatibility:

   ```python
   # Old import still works
   from webapp.parser.config import VALID_OFFICES
   assert "President" in VALID_OFFICES
   
   # New import available
   from webapp.parser.config import get_vocab_loader
   loader = get_vocab_loader()
   assert "President" in loader.load_vocab_set("entities/offices.txt")
   ```

**Safety Checkpoint**:

- [ ] Old imports work unchanged (backward compatible)
- [ ] New imports available (preferred path forward)
- [ ] No circular dependencies
- [ ] VocabLoader initialized once (singleton pattern)
- [ ] Audit logging enabled for all vocab access

---

### ☐ 3.2 Update Handler Imports (Phased)

**Objective**: Gradually migrate handler imports from static constants to dynamic vocab loading.

**Handlers to Update** (priority order):

1. HTML handler (most reliant on entity lookup)
2. XLSX handler (header validation)
3. CSV handler (header validation)
4. PDF handler (text extraction + entity matching)
5. JSON handler (schema validation)
6. TXT handler (last priority)

**Steps for Each Handler**:

**Before (OLD)**:

```python
# handlers/html_handler.py
from webapp.parser.config import VALID_OFFICES, VALID_STATES

def parse_html(page, context):
    offices = VALID_OFFICES
    states = VALID_STATES
    # ... use static lists
```

**After (NEW)**:

```python
# handlers/html_handler.py
from webapp.parser.config import get_vocab_loader

def parse_html(page, context, session_id=None):
    loader = get_vocab_loader(
        session_id=session_id,
        trust_threshold=0.85,  # Handler-specific threshold
        election_mode=context.get("election_mode", False)
    )
    
    try:
        offices = loader.load_vocab_set("entities/offices.txt")
        states = loader.load_vocab_set("entities/jurisdictions.txt")
    except VocabSecurityError as e:
        # Log anomaly for manual review
        logger.error(f"Unverified entity in election mode: {e}", extra={"session_id": session_id})
        return None, None, None, {"error": "Entity verification failed"}
    
    # ... use vocab lists with trust verification
```

**Verification Steps**:

- [ ] Handler imports updated
- [ ] Session context passed to loader
- [ ] Trust threshold set appropriately
- [ ] Error handling implemented (VocabSecurityError catch)
- [ ] Audit logging shows handler name in logs
- [ ] Unit tests pass for handler

**Migration Timeline**:

- Week 1-2: HTML handler
- Week 2-3: XLSX + CSV handlers
- Week 3-4: PDF handler
- Week 4: JSON + TXT handlers

**Safety Checkpoint**:

- [ ] All handlers updated in priority order
- [ ] Session context propagated
- [ ] Error handling catches VocabSecurityError
- [ ] Audit logs show which handler accessed which vocab files
- [ ] No regressions in parsing accuracy

---

### ☐ 3.3 Remove Static Constant Definitions

**Objective**: Once all handlers migrated, remove old static lists from config.py.

**Steps**:

1. Verify all handlers use new VocabLoader API
2. Remove lines like:

   ```python
   # REMOVE THESE (now loaded dynamically from vocab files)
   VALID_OFFICES = ["President", "Senator", ...]
   VALID_STATES = ["California", "New York", ...]
   VALID_PARTIES = ["Democratic", "Republican", ...]
   ```

3. Keep backward compatibility shim for a grace period (1-2 weeks):

   ```python
   # Deprecated (for backward compatibility during transition)
   VALID_OFFICES = _loader.load_vocab_set("entities/offices.txt")
   # ^^ Will be removed after all handlers migrated
   ```

4. After grace period, remove entirely
5. Commit to git:

   ```bash
   git commit -m "refactor: Remove static constants from config.py (now loaded from vocab)"
   ```

**Safety Checkpoint**:

- [ ] All handlers verified using new API
- [ ] Grace period announced to team (1-2 weeks)
- [ ] Old constants removed
- [ ] No broken imports
- [ ] VocabLoader used for all entity lookups

---

## Phase 4: Enable Audit Logging & Provenance Tracking

### ☐ 4.1 Create Audit Log Storage

**Objective**: Set up persistent JSONL audit trail for all vocab operations.

**Steps**:

1. Create log directory:

   ```bash
   mkdir -p webapp/parser/logs
   echo "# Audit logs (git-ignored)" > webapp/parser/logs/.gitkeep
   git add webapp/parser/logs/.gitkeep
   ```

2. Create `.gitignore` entry:

   ```txt
   webapp/parser/logs/vocab_audit.jsonl
   webapp/parser/logs/vocab_audit.jsonl.*
   ```

3. VocabLoader logs all operations to `webapp/parser/logs/vocab_audit.jsonl`:

   ```jsonl
   {"timestamp": "2026-02-03T14:32:45Z", "session_id": "sess_...", "operation": "load_vocab_set", ...}
   {"timestamp": "2026-02-03T14:32:46Z", "session_id": "sess_...", "operation": "alias_resolve", ...}
   ```

4. Rotation policy:
   - Daily rotation (vocab_audit.2026-02-03.jsonl)
   - Retention: 90 days
   - Compression: gzip after 7 days

**Safety Checkpoint**:

- [ ] Audit log file created
- [ ] VocabLoader appends operations to JSONL
- [ ] Log structure matches specification (§5.1)
- [ ] Rotation working (daily files created)
- [ ] Compression working (old files gzipped)
- [ ] Retention policy enforced (90-day cleanup)

---

### ☐ 4.2 Create Audit Log Query Tool

**Objective**: Enable forensic analysis of vocab access patterns.

**Steps**:

1. Create `scripts/query_vocab_audit.py`:

   ```python
   #!/usr/bin/env python3
   """
   Query vocab audit logs for forensic analysis.
   
   Usage:
       python scripts/query_vocab_audit.py --session-id sess_2026_02_03_abc123
       python scripts/query_vocab_audit.py --principal user@elections.ca.gov --date 2026-02-03
       python scripts/query_vocab_audit.py --anomalies  # Show failed operations
   """
   
   def query_by_session(session_id):
       # Read vocab_audit.jsonl
       # Filter by session_id
       # Print summary + details
       pass
   
   def query_by_principal(principal, date=None):
       # Filter by principal + optional date
       pass
   
   def query_anomalies():
       # Show all failed operations
       pass
   
   def generate_report():
       # Daily report: access counts, top principals, anomalies
       pass
   ```

2. Make executable:

   ```bash
   chmod +x scripts/query_vocab_audit.py
   ```

**Safety Checkpoint**:

- [ ] Query tool created and tested
- [ ] Supports session, principal, date, anomaly filtering
- [ ] Report generation working
- [ ] Can detect suspicious patterns (e.g., repeated failures, after-hours access)

---

## Phase 5: Implement Trust Verification & Verified Sources

### ☐ 5.1 Populate Verified Sources Registry

**Objective**: Build authoritative list of trusted election result sources.

**Steps**:

1. Gather URLs from state election authorities:
   - Secretary of State websites
   - County clerk result pages
   - Official election authority APIs

2. Assign trust scores:

   ```table
   Source Type                              | Trust Score
   ----------------------------------------|---------------
   Official SoS API (HTTPS, validated cert) | 0.99
   Official SoS website (HTTPS, validated) | 0.97
   Official county website (HTTPS)         | 0.95
   Official county clerk email             | 0.90
   Third-party aggregator (Reuters, AP)    | 0.85
   News organization analysis              | 0.70
   Social media (Twitter/X, Reddit)        | 0.30
   Unverified link                         | 0.00 (rejected)
   ```

3. Create `webapp/parser/Context_Integration/vocab/sources/verified_sources.txt`:

   ```text
   https://results.sos.ca.gov/2024/general|0.99|official_sos_api|California|2026-02-03T10:00:00Z
   https://www.sos.ca.gov/elections/results|0.97|official_sos_website|California|2026-02-03T10:00:00Z
   https://elections.los-angeles-county.gov/results|0.95|official_county|California/Los Angeles|2026-02-03T10:00:00Z
   # ... one per line
   ```

4. Compute hash and update manifest

**Safety Checkpoint**:

- [ ] All state/county election authority URLs listed
- [ ] Trust scores assigned correctly
- [ ] No suspicious URLs (cloud storage, personal sites)
- [ ] Updated weekly during election season
- [ ] Hash verified in manifest

---

### ☐ 5.2 Integrate Trust Scoring with Parsing

**Objective**: Verify source URL before parsing results.

**Steps**:

1. Update handler entrypoint to check trust:

   ```python
   # handlers/html_handler.py
   def parse_html(page, context, session_id=None):
       source_url = context.get("url")
       
       # Check trust BEFORE parsing
       loader = get_vocab_loader(session_id=session_id)
       source_info = loader.get_verified_source(source_url)
       
       if not source_info or source_info["trust_score"] < 0.85:
           logger.warning(f"Unverified source: {source_url}", extra={"session_id": session_id})
           # Log anomaly for manual review
           return None, None, None, {
               "error": "Source not verified",
               "trust_score": source_info["trust_score"] if source_info else 0
           }
       
       # Proceed with parsing
       # ...
   ```

2. Add to anomaly schema:

   ```json
   {
     "event_type": "anomaly",
     "reason_code": "unverified_source",
     "severity": "low",
     "trust_score": 0.30,
     "source_url": "https://unknown.example.com"
   }
   ```

**Safety Checkpoint**:

- [ ] All handlers check trust before parsing
- [ ] Anomalies logged for unverified sources
- [ ] Trust score < 0.85 triggers quarantine
- [ ] Session audit logs show source verification

---

## Phase 6: Election Mode & Live Scanning

### ☐ 6.1 Enable Election Mode

**Objective**: Lock vocab files and track timestamps during live election day scanning.

**Steps**:

1. Create election mode configuration:

   ```python
   # webapp/parser/config.py
   
   ELECTION_MODE_CONFIG = {
       "enabled": False,  # Set True during live scanning
       "election_date": "2026-11-03",  # ISO 8601
       "state": "CA",
       "jurisdiction": None,  # State-wide, or specific county
       "lock_start": "2026-11-03T08:00:00Z",
       "lock_end": "2026-11-03T21:00:00Z",
       "enforce_immutability": True,  # Block any vocab modifications
       "snapshot_frequency_minutes": 15  # Create snapshot every 15 min
   }
   ```

2. Enable during elections:

   ```python
   # On election day morning
   from webapp.parser.config import get_vocab_loader
   
   loader = get_vocab_loader(election_mode=True)
   # All file modifications now blocked
   # All reads logged with strict audit trail
   ```

3. Test election mode:

   ```python
   def test_election_mode_blocks_modifications():
       loader = VocabLoader(election_mode=True)
       
       # Try to modify vocab file
       with pytest.raises(VocabSecurityError):
           # Simulate file write
           pass
   ```

**Safety Checkpoint**:

- [ ] Election mode can be enabled/disabled
- [ ] File modifications blocked during election_mode
- [ ] All operations logged
- [ ] Snapshots created at regular intervals
- [ ] Can be reversed (unlock after elections close)

---

### ☐ 6.2 Create Election Day Snapshots

**Objective**: Capture point-in-time vocabulary state linked to result completion percentage.

**Steps**:

1. Implement snapshot creation:

   ```python
   # During live scanning
   loader = get_vocab_loader(election_mode=True)
   
   snapshot = loader.create_election_snapshot(
       snapshot_time="2026-11-03T15:45:00Z",
       completion_percentage=87.45,
       source_url="https://results.sos.ca.gov/2024/general"
   )
   # snapshot_id: "snap_ca_2024_110315_8745"
   ```

2. Store snapshots with results:

   ```json
   {
     "snapshot_id": "snap_ca_2024_110315_8745",
     "timestamp": "2026-11-03T15:45:00Z",
     "completion_percentage": 87.45,
     "vocab_version_hash": "7b2f9c1a...",
     "verified_entities": {
       "offices": ["President", "Senator", ...],
       "parties": ["Democratic", "Republican", ...],
       "jurisdictions": ["California", "California/Los Angeles", ...]
     }
   }
   ```

3. Link anomalies to snapshots:

   ```json
   {
     "event_type": "anomaly",
     "snapshot_id": "snap_ca_2024_110315_8745",
     "reason_code": "mismatched_totals",
     "message": "Flagged at 87.45% reporting; may be data lag from slow precinct"
   }
   ```

**Safety Checkpoint**:

- [ ] Snapshots created every 15 minutes during elections
- [ ] Snapshot IDs immutable + traceable
- [ ] Vocab hash captured for integrity
- [ ] Anomalies correlated with snapshots
- [ ] Enables forensic analysis (e.g., "data changed at 3:45 PM, affecting X precincts")

---

## Phase 7: Validation & Testing

### ☐ 7.1 Integration Testing

**Objective**: Verify vocab system works end-to-end with handlers + UI.

**Steps**:

1. Run full test suite:

   ```bash
   python -m pytest tests/unit/test_vocab_loader.py -v
   python -m pytest tests/integration/test_vocab_handler_integration.py -v
   python -m pytest webapp/tests/ -v
   ```

2. Test critical paths:
   - [ ] Parse HTML result from verified source
   - [ ] Detect anomaly in unverified source
   - [ ] Alias resolution in all contexts (state, county, party)
   - [ ] Election mode locking
   - [ ] Snapshot creation + correlation
   - [ ] Audit logging + querying

3. Regression testing:
   - [ ] All existing handlers produce same results
   - [ ] No performance degradation (< 10ms per vocab load)
   - [ ] Backward compatibility maintained

**Safety Checkpoint**:

- [ ] 100+ integration tests pass
- [ ] No regressions
- [ ] Performance acceptable

---

### ☐ 7.2 Security & Compliance Review

**Objective**: Verify security controls and audit trail.

**Review Checklist**:

- [ ] File integrity: Hashes verified on every load
- [ ] Access control: Principal-based audit logging
- [ ] Immutability: Election mode prevents modifications
- [ ] Provenance: All sources traced to authoritative registry
- [ ] Tampering: Any file modification detected immediately
- [ ] Audit trail: All operations logged (success + failure)
- [ ] Rate limiting: Per-session limits enforced
- [ ] Quarantine: Unverified sources marked for review

**Compliance Verification**:

- [ ] Meets state election integrity requirements
- [ ] Follows federal voting systems standards (VVSG)
- [ ] No unverified data accepted during elections
- [ ] Full forensic audit available (90-day retention)

---

## Phase 8: Production Deployment

### ☐ 8.1 Pre-Deployment Checklist

**Before Deploying to Production:**

- [ ] All tests pass (unit + integration)
- [ ] Security review completed and approved
- [ ] Audit logs functional and rotated
- [ ] Backward compatibility verified
- [ ] Rollback plan documented
- [ ] Performance metrics acceptable
- [ ] Documentation complete

---

### ☐ 8.2 Deployment Steps

**Steps**:

1. **Staging Deployment**:

   ```bash
   git pull origin main
   python -m pytest tests/ -v
   # If all pass:
   docker build -t election-parser:staging .
   docker run -e ENVIRONMENT=staging election-parser:staging
   ```

2. **Validation**:
   - Parse sample election results
   - Verify audit logs created
   - Check for errors

3. **Production Deployment**:

   ```bash
   git tag vocab-migration-v1.0
   git push origin vocab-migration-v1.0
   # Deployment pipeline triggered (CI/CD)
   ```

4. **Post-Deployment Monitoring**:
   - Monitor audit logs for anomalies
   - Check handler error rates
   - Verify vocab loads work (sample queries)
   - Track performance metrics

**Safety Checkpoint**:

- [ ] Staging deployment successful
- [ ] Sample results parsed correctly
- [ ] Production deployment successful
- [ ] Monitoring active
- [ ] Rollback tested (can revert quickly if issues)

---

### ☐ 8.3 Rollback Plan

**If Issues Detected:**

1. Immediately revert to previous version:

   ```bash
   git revert HEAD
   git push origin main
   # CI/CD redeploys
   ```

2. Investigate issues:
   - Check logs for errors
   - Query audit trail for anomalies
   - Identify root cause

3. Fix and redeploy:
   - Apply hotfix
   - Test in staging
   - Deploy to production

**Safety Checkpoint**:

- [ ] Rollback tested and working
- [ ] Can revert in < 5 minutes if needed
- [ ] Monitoring active (alert on errors)

---

## Phase 9: Election Season Operations

### ☐ 9.1 Weekly Verification Update

**During Election Season:**

1. Every Monday (or as needed):

   ```bash
   # Verify all sources still valid
   python scripts/verify_vocab_sources.py
   
   # Output:
   # [✓] https://results.sos.ca.gov/2024/general → 0.99 (verified 1 hour ago)
   # [✓] https://elections.lac.gov/results → 0.95 (verified 2 days ago)
   # [!] https://elections.sfgov.org → Last verified 14 days ago (refresh)
   ```

2. Update last-verified timestamps:

   ```bash
   # Refresh outdated URLs
   python scripts/refresh_vocab_sources.py
   ```

3. Commit updates:

   ```bash
   git add webapp/parser/Context_Integration/vocab/sources/verified_sources.txt
   git commit -m "ops: Weekly vocab sources verification"
   ```

**Safety Checkpoint**:

- [ ] All sources verified weekly
- [ ] Last-verified timestamps current (< 1 week old)
- [ ] Any new sources added and trust-scored
- [ ] Blacklist updated (any compromised URLs removed)

---

### ☐ 9.2 Election Day Monitoring

**On Election Day:**

1. Enable election mode:

   ```python
   # Set in config
   ELECTION_MODE_CONFIG["enabled"] = True
   ```

2. Monitor audit logs in real-time:

   ```bash
   # Stream incoming vocab operations
   tail -f logs/vocab_audit.jsonl | jq '.[] | select(.result=="failure")'
   ```

3. Create snapshots every 15 minutes:
   - Automatic via scheduled job
   - Manually if needed: `python scripts/create_snapshot.py`

4. Watch for anomalies:
   - Failed source verifications
   - Unrecognized entities
   - Rate limiting triggers

**Safety Checkpoint**:

- [ ] Election mode enabled
- [ ] Audit logs streaming without errors
- [ ] Snapshots created on schedule
- [ ] No suspicious patterns in logs
- [ ] Team standing by for manual review of quarantined data

---

### ☐ 9.3 Post-Election Analysis

**After Elections Close:**

1. Disable election mode:

   ```python
   ELECTION_MODE_CONFIG["enabled"] = False
   ```

2. Generate audit report:

   ```bash
   python scripts/generate_audit_report.py --date 2026-11-03
   ```

3. Analyze patterns:
   - Which sources most frequently anomalous?
   - Which anomaly reasons most common?
   - False positive rate?
   - Any indicators of data tampering?

4. Publish findings:
   - Internal report to election integrity team
   - Recommend vocab adjustments for next election

**Safety Checkpoint**:

- [ ] Audit report complete
- [ ] Anomaly analysis done
- [ ] Recommendations documented
- [ ] Vocab updates planned for next cycle

---

## Success Criteria

**Migration Complete When:**

✅ **Code Level**:

- [ ] All handlers use VocabLoader API
- [ ] No static constants in config.py (or only backward compat shim)
- [ ] VocabLoader has 95%+ test coverage
- [ ] Zero security exceptions in staging/production
- [ ] Backward compatibility maintained (old imports still work)

✅ **Data Level**:

- [ ] All 13+ vocab files populated and hashed
- [ ] Manifest contains checksums for all files
- [ ] Verified sources registry complete (all state/county URLs)
- [ ] All constants migrated to vocab files
- [ ] File integrity verified on every load

✅ **Operations Level**:

- [ ] Audit logs flowing to JSONL (all operations logged)
- [ ] Election mode can be activated/deactivated
- [ ] Snapshots created on schedule
- [ ] Anomaly detection working
- [ ] Team trained on new system

✅ **Security Level**:

- [ ] No unverified sources accepted (during election_mode)
- [ ] All modifications blocked during elections
- [ ] Tampering detected (hash verification)
- [ ] 90-day audit retention
- [ ] Rate limiting enforced
- [ ] Access control audit logs complete

✅ **Election Integrity Level**:

- [ ] Results from verified sources only (during elections)
- [ ] Unverified sources quarantined
- [ ] Confidence scores calculated for all data
- [ ] Point-in-time snapshots for forensic analysis
- [ ] Full provenance chain (source → handler → output)

---

## Timeline Estimate

| Phase | Duration | Start | End |
| ------- | ---------- | ------- | ----- |
| 0: Pre-Migration Audit | 3-5 days | Week 1 | Week 1 |
| 1: Populate Vocab Files | 5-7 days | Week 1-2 | Week 2 |
| 2: Implement VocabLoader | 5-7 days | Week 2 | Week 2-3 |
| 3: Refactor constants.py | 3-5 days | Week 3 | Week 3 |
| 4: Audit Logging | 2-3 days | Week 3 | Week 4 |
| 5: Trust Verification | 3-5 days | Week 4 | Week 4 |
| 6: Election Mode | 2-3 days | Week 4 | Week 5 |
| 7: Validation & Testing | 3-5 days | Week 5 | Week 5 |
| 8: Production Deployment | 1-2 days | Week 6 | Week 6 |
| 9: Election Ops | Ongoing | Week 7+ | Ongoing |

***Total: 6-8 weeks to full production deployment***

---

## Risk Mitigation Strategy

| Risk | Mitigation |
| ------- | ----------- |
| Handler regression | Comprehensive integration tests + staging validation |
| Performance degradation | Caching layer + load testing before deployment |
| Audit log filling disk | Rotation policy + automatic cleanup |
| Compromised sources | Verify URLs weekly + manual review process |
| Election mode misconfig | Pre-election dry run + monitoring alerts |
| Unverified data injection | Quarantine policy + manual review workflow |
| Tampering detection lag | Hash verification on every load (immediate detection) |

---

## Appendix: Key Files Changed

**New Files Created:**

- `docs/VOCAB_LOADER_API_SPECIFICATION.md` (this document)
- `docs/VOCAB_MIGRATION_CHECKLIST.md` (detailed implementation plan)
- `docs/CONSTANTS_INVENTORY.md` (catalog of current constants)
- `webapp/parser/Context_Integration/vocab_loader.py` (main loader class)
- `tests/unit/test_vocab_loader.py` (comprehensive test suite)
- `tests/integration/test_vocab_handler_integration.py` (handler tests)
- `scripts/query_vocab_audit.py` (audit log analysis tool)
- `scripts/verify_vocab_sources.py` (source verification tool)

**Files Modified:**

- `webapp/parser/config.py` (refactored to use VocabLoader)
- `webapp/parser/handlers/html_handler.py` (updated imports)
- `webapp/parser/handlers/xlsx_handler.py` (updated imports)
- `webapp/parser/handlers/csv_handler.py` (updated imports)
- `webapp/parser/handlers/pdf_handler.py` (updated imports)
- `webapp/parser/handlers/json_handler.py` (updated imports)
- `webapp/parser/handlers/txt_handler.py` (updated imports)
- `.gitignore` (add vocab audit logs)

---

**Owner**: Election Integrity Team  
**Last Updated**: 2026-02-03  
**Status**: Ready for Implementation  
**Next Step**: Begin Phase 0 (Pre-Migration Audit)
