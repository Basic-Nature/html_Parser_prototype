# Vocabulary Loader API Specification

## Executive Summary

This document specifies the **VocabLoader** API—a security-first Python module that loads context-compartmentalized entity files from the `webapp/parser/Context_Integration/vocab/` folder while enforcing **trust verification, provenance tracking, and audit logging**.

The loader is designed specifically for **election result accuracy and data integrity**, with embedded safeguards against improper use, malicious modification, and unverified data injection.

---

## 1. Design Principles

### 1.1 Trust-First Architecture

- **Verified Source Gating**: All vocab entities carry optional `verified_source` metadata linking back to official election authority sources
- **Provenance Tracking**: Each entity load operation is logged with source URL, session ID, timestamp, and trust score
- **Immutability During Elections**: Vocab files are treated as read-only during active election periods (configurable via `ELECTION_MODE`)
- **Session-Level Audit Trail**: All vocab queries tagged with session context for forensic analysis

### 1.2 Risk Mitigation

- **Malicious Modification Detection**: File integrity checks (SHA-256 hashes) prevent tampering
- **Rate Limiting**: Per-session vocab load requests limited to prevent enumeration attacks
- **Canonical Form Enforcement**: Only verified canonical names accepted; aliases require explicit mapping
- **Negative Backscatter Prevention**: Unverified entities quarantined, not ingested into core vocab

### 1.3 Election Result Integrity

- **Live Result Timestamp Anchoring**: Vocab entities linked to election day snapshot versions (immutable point-in-time snapshots)
- **Percentage Completion Tracking**: Staged result sets tagged with snapshot time and completion percentage
- **Source Verification at Parse Time**: Parsed results validated against verified source registry before acceptance
- **Confidence Scoring**: Each entity carries confidence level based on verification method (official API, live scrape, manual entry)

---

## 2. Core API Specification

### 2.1 VocabLoader Class

```python
class VocabLoader:
    """
    Thread-safe vocabulary loader with embedded trust verification, provenance tracking,
    and election-specific safeguards.
    
    Usage:
        loader = VocabLoader(
            vocab_root="webapp/parser/Context_Integration/vocab",
            session_id="sess_2026_02_03_abc123",
            trust_threshold=0.75,  # Minimum trust score accepted
            election_mode=False     # Set True during live election scanning
        )
        
        offices = loader.load_vocab_set("entities/offices.txt")
        # Returns: ["President", "Senator", "Representative", ...]
        
        state_canonical = loader.resolve_alias("state", "CA")
        # Returns: "California" (with verified_source tracking)
    """
```

### 2.2 Initialization

```python
def __init__(
    self,
    vocab_root: str = "webapp/parser/Context_Integration/vocab",
    session_id: str | None = None,
    trust_threshold: float = 0.75,
    election_mode: bool = False,
    enable_audit_logging: bool = True,
    principal: str | None = None,  # Session owner (from client cert or SSO)
    principal_source: str | None = None  # "client_cert" | "sso" | "dev_bypass"
):
    """
    Initialize the VocabLoader with security context.
    
    Args:
        vocab_root: Root path to vocabulary folder structure
        session_id: Session ID for audit trail (from webapp session)
        trust_threshold: Minimum trust score (0.0-1.0) to accept entities
        election_mode: If True, vocab files treated as immutable
        enable_audit_logging: Enable persistent audit trail (JSONL)
        principal: Session owner identity for access control
        principal_source: Identity source (for multi-tenant scenarios)
    
    Raises:
        VocabLoaderError: If vocab_root missing or corrupted
        SecurityError: If integrity checks fail (tampering detected)
    """
```

**Initialization Checks:**

1. Verify `vocab_root` exists and is readable
2. Load `vocab_root/manifest.md` (schema documentation)
3. Compute and verify SHA-256 hashes for all TXT files (embedded in manifest)
4. Check for modification timestamps (warn if modified during `election_mode=True`)
5. Load verified sources registry to initialize trust scoring
6. Create session-scoped audit context

---

## 3. Core Methods

### 3.1 Load Vocab Set

```python
def load_vocab_set(
    self,
    file_path: str,  # e.g., "entities/offices.txt"
    allow_aliases: bool = True,
    verify_source: bool = True
) -> list[str]:
    """
    Load a vocabulary file, returning canonical entity names.
    
    Args:
        file_path: Relative path to vocab file (e.g., "entities/offices.txt")
        allow_aliases: If True, resolve `alias -> canonical` mappings
        verify_source: If True, only return entities with trust >= trust_threshold
    
    Returns:
        List of canonical entity names, sorted
        
    Raises:
        VocabFileNotFound: If file_path doesn't exist
        VocabIntegrityError: If file hash doesn't match manifest
        VocabSecurityError: If file modified during election_mode
    
    Example:
        # Load all valid offices for this jurisdiction
        offices = loader.load_vocab_set(
            "entities/offices.txt",
            allow_aliases=False,
            verify_source=True
        )
        # Returns: ["President", "Senator", "Representative", "State Legislator"]
        
        # Audit log entry:
        # {
        #   "type": "vocab_load",
        #   "session_id": "sess_...",
        #   "file_path": "entities/offices.txt",
        #   "count": 4,
        #   "timestamp": "2026-02-03T14:32:45Z",
        #   "trust_score": 0.95,
        #   "verified_source": "official_sos_api"
        # }
    """
```

**Internal Process:**

1. Resolve full path: `vocab_root / file_path`
2. Verify path is within `vocab_root` (prevent directory traversal)
3. Check file hash against manifest (detect tampering)
4. Check modification timestamp (warn/block if during `election_mode`)
5. Load file into memory
6. Parse lines (skip comments `#`, strip whitespace)
7. If `allow_aliases=True`, resolve `alias -> canonical` mappings
8. If `verify_source=True`, filter entities by trust score
9. Log operation with session context and trust metadata
10. Return sorted canonical list

### 3.2 Resolve Alias

```python
def resolve_alias(
    self,
    context: str,  # e.g., "state", "party", "county"
    alias: str,    # e.g., "CA", "Dem", "Los Angeles Co."
    exact_match: bool = True
) -> str | None:
    """
    Resolve an alias to its canonical form.
    
    Args:
        context: Vocab category (e.g., "state", "party", "county")
        alias: Alias string to resolve (case-insensitive)
        exact_match: If False, attempt fuzzy matching (with confidence penalty)
    
    Returns:
        Canonical form string, or None if not found
        
    Raises:
        VocabContextNotFound: If context doesn't exist
        VocabSecurityError: If resolving unverified alias in election_mode
    
    Example:
        # Resolve state abbreviation
        state = loader.resolve_alias("state", "CA")
        # Returns: "California" (trust_score=0.99)
        
        # Attempt fuzzy match (lower confidence)
        county = loader.resolve_alias("county", "Los Angles", exact_match=False)
        # Returns: "Los Angeles" (trust_score=0.72, fuzzy=True)
        
        # Audit log entry:
        # {
        #   "type": "alias_resolve",
        #   "session_id": "sess_...",
        #   "context": "state",
        #   "input": "CA",
        #   "output": "California",
        #   "exact_match": True,
        #   "trust_score": 0.99,
        #   "timestamp": "2026-02-03T14:32:46Z"
        # }
    """
```

**Internal Process:**

1. Load validators file for context (e.g., `validators/state_aliases.txt`)
2. Search for exact match (case-insensitive, strip whitespace)
3. If found, return canonical + log with trust metadata
4. If not found and `exact_match=False`, attempt fuzzy match (Levenshtein distance)
5. If fuzzy match succeeds but confidence < trust_threshold, raise VocabSecurityError
6. Audit log all attempts (success and failure)

### 3.3 Query Verified Sources

```python
def get_verified_source(
    self,
    source_url: str
) -> dict | None:
    """
    Query the verified sources registry for trust metadata.
    
    Args:
        source_url: URL to verify (e.g., "https://results.sos.ca.gov/2024")
    
    Returns:
        Dict with structure:
        {
            "url": str,
            "trust_score": float (0.0-1.0),
            "verified_source_type": "official_sos_api" | "official_county_website" | "official_election_authority" | "manual_entry",
            "jurisdiction": str,  # e.g., "California" or "California/Los Angeles"
            "updated_at": str,    # ISO 8601 timestamp of last verification
            "snapshot_time": str, # ISO 8601 of election day snapshot
            "completion_percentage": float,  # Progress on election day
            "confidence_level": "high" | "medium" | "low"
        }
        or None if not found
    
    Example:
        source_info = loader.get_verified_source(
            "https://results.sos.ca.gov/2024/general"
        )
        # Returns:
        # {
        #   "url": "https://results.sos.ca.gov/2024/general",
        #   "trust_score": 0.99,
        #   "verified_source_type": "official_sos_api",
        #   "jurisdiction": "California",
        #   "updated_at": "2026-02-03T14:32:00Z",
        #   "snapshot_time": "2026-02-03T14:30:00Z",
        #   "completion_percentage": 87.5,
        #   "confidence_level": "high"
        # }
    """
```

**Internal Process:**

1. Load verified sources file: `sources/verified_sources.txt`
2. Normalize URL (strip trailing slash, lowercase domain)
3. Search for exact match or parent domain match
4. Return trust metadata + snapshot info for election day correlation
5. Log query with session context

### 3.4 Score Keyword Combination

```python
def score_keyword_combination(
    self,
    keywords: list[str],
    context: str | None = None  # Optional: "header_mismatch", "missing_candidate", etc.
) -> dict:
    """
    Score a combination of keywords for anomaly confidence calculation.
    
    Args:
        keywords: List of observed keywords (e.g., ["Candidate", "Votes"])
        context: Optional anomaly reason code for contextual weighting
    
    Returns:
        {
            "base_score": float,  # 0.0-1.0, before context adjustment
            "contextual_score": float,  # 0.0-1.0, after context weighting
            "signal_breakdown": {
                "keyword1": float,
                "keyword2": float,
                ...
                "combination_multiplier": float  # e.g., 1.2 if common pairing
            },
            "matched_headers": list[str],  # Which canonical headers matched
            "unmatched_keywords": list[str],  # Keywords without match
            "confidence": "high" | "medium" | "low"
        }
    
    Example:
        score = loader.score_keyword_combination(
            keywords=["Candidate Name", "Vote Count"],
            context="header_mismatch"
        )
        # Returns:
        # {
        #   "base_score": 0.92,
        #   "contextual_score": 0.78,  # Lower due to mismatch context
        #   "signal_breakdown": {
        #       "Candidate Name": 0.95,
        #       "Vote Count": 0.88,
        #       "combination_multiplier": 1.05
        #   },
        #   "matched_headers": ["Candidate", "Votes"],
        #   "unmatched_keywords": [],
        #   "confidence": "high"
        # }
    """
```

**Internal Process:**

1. Load scoring file: `scoring/trust_signals.txt`
2. For each keyword, look up base signal score
3. Calculate combination score (product of individual scores with multiplier for common pairings)
4. If context provided, apply contextual weighting
5. Log keyword scoring attempt with session context
6. Return breakdown for debugging/audit

### 3.5 Get Anomaly Reason Code Definition

```python
def get_anomaly_reason_definition(
    self,
    reason_code: str
) -> dict | None:
    """
    Retrieve definition and severity for an anomaly reason code.
    
    Args:
        reason_code: Anomaly code (e.g., "mismatched_totals", "missing_candidate")
    
    Returns:
        {
            "reason_code": str,
            "description": str,
            "severity": "low" | "medium" | "high",
            "quarantine_required": bool,
            "requires_manual_review": bool,
            "eligible_for_ingestion": bool,  # If False, data cannot be accepted
            "related_headers": list[str]
        }
        or None if code not recognized
    
    Example:
        defn = loader.get_anomaly_reason_definition("mismatched_totals")
        # Returns:
        # {
        #   "reason_code": "mismatched_totals",
        #   "description": "Sum of candidate votes does not match reported total",
        #   "severity": "high",
        #   "quarantine_required": True,
        #   "requires_manual_review": True,
        #   "eligible_for_ingestion": False,
        #   "related_headers": ["Total Votes", "Candidate Votes"]
        # }
    """
```

**Internal Process:**

1. Load anomaly definitions: `scoring/anomaly_reasons.txt`
2. Parse reason code and severity mapping
3. Determine eligibility for ingestion (high severity = not eligible)
4. Return structured definition

---

## 4. Election Mode & Snapshots

### 4.1 Lock Vocab During Active Elections

```python
def lock_election_mode(
    self,
    election_date: str,  # ISO 8601 date (e.g., "2026-11-03")
    state: str,          # State code (e.g., "CA")
    jurisdiction: str | None = None  # County name (optional)
):
    """
    Activate election mode: make vocab files immutable until election closes.
    
    During election_mode=True:
    - All vocab files treated as read-only (modifications blocked)
    - All loads logged with strict audit trail
    - Load failures trigger immediate alerts
    - Snapshot versions created for point-in-time result verification
    
    Args:
        election_date: Election date in ISO 8601 format
        state: State FIPS code or abbreviation
        jurisdiction: Optional county/jurisdiction for sub-state elections
    """
```

### 4.2 Create Point-in-Time Snapshot

```python
def create_election_snapshot(
    self,
    snapshot_time: str,  # ISO 8601 timestamp (e.g., "2026-11-03T20:45:00Z")
    completion_percentage: float,  # Current result % complete (0.0-100.0)
    source_url: str | None = None  # URL of official results source
) -> dict:
    """
    Create an immutable snapshot of current vocab state linked to specific election moment.
    
    Returns:
        {
            "snapshot_id": str,  # e.g., "snap_ca_2026_110320_8745"
            "snapshot_time": str,
            "completion_percentage": float,
            "source_url": str,
            "vocab_version_hash": str,  # SHA-256 of all vocab files at snapshot time
            "verified_entities": {
                "offices": [...],
                "parties": [...],
                "jurisdictions": [...]
            }
        }
    """
```

---

## 5. Audit Logging & Provenance

### 5.1 Audit Log Structure

All vocab operations logged to `logs/vocab_audit.jsonl` with this schema:

```json
{
  "timestamp": "2026-02-03T14:32:45.123Z",
  "session_id": "sess_2026_02_03_abc123",
  "principal": "user@elections.ca.gov",
  "principal_source": "client_cert",
  "operation": "load_vocab_set|alias_resolve|verify_source|score_combination",
  "file_path": "entities/offices.txt",
  "parameters": {
    "allow_aliases": true,
    "verify_source": true
  },
  "result": "success|failure",
  "error_code": null,
  "result_count": 4,
  "trust_score": 0.95,
  "verified_source": "official_sos_api",
  "election_mode": false,
  "election_date": "2026-11-03"
}
```

### 5.2 Access Control Enforcement

```python
def set_access_control(
    self,
    allow_public_read: bool = False,  # Allow anonymous reads
    allow_manual_entry: bool = True,  # Allow unverified entries (if election_mode=False)
    require_principal: bool = True,   # Require authenticated session
    min_trust_threshold: float = 0.75
):
    """
    Configure access control policies for vocab operations.
    
    During election_mode=True, strictest settings enforced regardless.
    """
```

---

## 6. Migration Checklist for constants.py

See `VOCAB_MIGRATION_CHECKLIST.md` for detailed step-by-step process.

**High-Level Summary:**

1. ✅ **Extract Constants** → Populate vocab TXT files
2. ✅ **Implement VocabLoader** → Replace config.py with helper module
3. ✅ **Update Handler Imports** → Point to new loader API
4. ✅ **Add Trust Verification** → Enforce verified source checks
5. ✅ **Enable Audit Logging** → Track all vocab access
6. ✅ **Test Election Mode** → Verify immutability during live scanning
7. ✅ **Validate Snapshot Creation** → Ensure point-in-time result correlation
8. ✅ **Production Deployment** → Rollout with backward compatibility shim

---

## 7. Error Handling

### 7.1 Exception Hierarchy

```python
class VocabLoaderError(Exception):
    """Base exception for all VocabLoader errors"""
    pass

class VocabFileNotFound(VocabLoaderError):
    """Raised when vocab file doesn't exist"""
    pass

class VocabIntegrityError(VocabLoaderError):
    """Raised when file hash doesn't match manifest (tampering detected)"""
    pass

class VocabSecurityError(VocabLoaderError):
    """Raised when security policies violated (e.g., unverified entity in election mode)"""
    pass

class VocabContextNotFound(VocabLoaderError):
    """Raised when requested vocab context (e.g., "state", "party") doesn't exist"""
    pass
```

---

## 8. Implementation Roadmap

| Phase | Deliverable | Timeline | Dependencies |
| ------- | ------------- | ---------- | -------------- |
| 1 | VocabLoader class skeleton + file loading | Week 1 | Vocab file structure |
| 2 | Trust verification + verified sources lookup | Week 1-2 | Verified source registry populated |
| 3 | Alias resolution + keyword scoring | Week 2 | Signal definitions |
| 4 | Election mode locking + snapshots | Week 2-3 | Election calendar config |
| 5 | Audit logging + provenance tracking | Week 3 | JSONL infrastructure |
| 6 | Access control enforcement | Week 3 | Principal/session context |
| 7 | Error handling + exception hierarchy | Week 3 | Core classes complete |
| 8 | constants.py migration + tests | Week 4 | VocabLoader stable |
| 9 | Handler integration + validation | Week 4 | VocabLoader deployed |
| 10 | Production deployment + rollback | Week 5 | Full testing complete |

---

## 9. Security Considerations

### 9.1 File Integrity Protection

- **Manifest-Based Hashing**: All vocab file hashes stored in `vocab_root/manifest.md` (signed with SHA-256)
- **Tamper Detection**: Automatic hash verification on every load; raises exception if modified
- **Timestamped Modifications**: Track when files were last updated; block modifications during election_mode

### 9.2 Access Control

- **Principal-Based Auditing**: All operations tagged with authenticated principal (client cert or SSO)
- **Rate Limiting**: Per-session vocab load requests limited (default: 100 loads/minute)
- **Immutability During Elections**: Files locked during active elections; only reads allowed

### 9.3 Data Provenance

- **Verified Source Linking**: Each entity traces back to authoritative source (election authority API, official website, manual entry)
- **Snapshot Versioning**: Point-in-time snapshots created for election day result correlation
- **Confidence Scoring**: All entities carry trust metadata (0.0-1.0 confidence)

### 9.4 Negative Backscatter Prevention

- **Quarantine Unverified Data**: Entities from unverified sources not ingested into core vocab during election_mode
- **Contextual Matching**: Unverified entities only accepted if context confirms legitimacy (e.g., alias already in canonical form)
- **Audit Trail Requirements**: All ingestion attempts logged with decision rationale

---

## 10. Example Usage Patterns

### Pattern 1: Parse Election Results with Trust Verification

```python
from webapp.parser.Context_Integration.vocab_loader import VocabLoader

loader = VocabLoader(
    session_id="sess_2026_02_03_xyz789",
    trust_threshold=0.85,
    election_mode=True,
    principal="parser@elections.ca.gov"
)

# Verify source before parsing
source_info = loader.get_verified_source("https://results.sos.ca.gov/2024/general")
if not source_info or source_info["trust_score"] < 0.85:
    raise SecurityError(f"Source not trusted: {source_info}")

# Load expected headers for validation
expected_headers = loader.load_vocab_set("snapshots/table_headers.txt")

# Score observed headers against expected
observed_headers = ["Candidate", "Votes", "%"]
score = loader.score_keyword_combination(
    keywords=observed_headers,
    context="header_validation"
)

if score["contextual_score"] < 0.70:
    # Anomaly detected, log for manual review
    anomaly_reason = "suspicious_header"
    reason_def = loader.get_anomaly_reason_definition(anomaly_reason)
    if reason_def["quarantine_required"]:
        # Quarantine results, trigger manual review
        pass
```

### Pattern 2: Election Day Snapshot Creation

```python
# At 3:45 PM on election day (87.45% of precincts reporting)
snapshot = loader.create_election_snapshot(
    snapshot_time="2026-11-03T15:45:00Z",
    completion_percentage=87.45,
    source_url="https://results.sos.ca.gov/2024/general"
)

# snapshot_id now usable to correlate anomalies to specific moment in time
# e.g., "This anomaly flagged at 87.45% completion; likely data lag from slow precinct"
```

### Pattern 3: Alias Resolution with Confidence Tracking

```python
# Incoming data has "Dem" instead of "Democratic"
canonical_party = loader.resolve_alias("party", "Dem", exact_match=False)
# Returns: "Democratic" (with audit log noting fuzzy match)

# Audit log shows: confidence=0.89, fuzzy=True, allow_aliases=True
# Handler can decide whether to accept based on confidence threshold
```

---

## 11. Testing Strategy

**Unit Tests Required:**

- [ ] File loading (success + missing file error)
- [ ] Hash verification (tampering detection)
- [ ] Alias resolution (exact + fuzzy matching)
- [ ] Trust scoring (keyword combinations)
- [ ] Verified source lookup
- [ ] Election mode locking
- [ ] Snapshot creation
- [ ] Audit logging (structure + persistence)
- [ ] Access control enforcement
- [ ] Rate limiting
- [ ] Exception handling

**Integration Tests Required:**

- [ ] Handler integration (xlsx/csv/json/html/pdf)
- [ ] Session context propagation
- [ ] Multi-tenant isolation (different principals)
- [ ] Election mode + snapshot correlation
- [ ] Anomaly quarantine workflow
- [ ] Backward compatibility (old constants.py imports)

---

## 12. Backward Compatibility

During migration, maintain compatibility shim in `config.py`:

```python
# OLD (deprecated, but still works)
from webapp.parser.config import VALID_OFFICES

# NEW (preferred)
from webapp.parser.Context_Integration.vocab_loader import VocabLoader
loader = VocabLoader()
offices = loader.load_vocab_set("entities/offices.txt")

# SHIM (maps old import to new loader)
_loader = VocabLoader()
VALID_OFFICES = _loader.load_vocab_set("entities/offices.txt")
```

This allows gradual handler migration without breaking existing code.

---

## 13. Performance Considerations

### 13.1 Caching Strategy

- **Lazy Loading**: Files loaded on first access, cached in memory
- **TTL-Based Invalidation**: Cache expires every 1 hour (configurable) during normal operation
- **Election Mode Cache Freezing**: Cache locked during election_mode (no expiration until unlock)
- **Memory Cap**: Cache limited to 50 MB (configurable); LRU eviction if exceeded

### 13.2 I/O Optimization

- **Batch File Loading**: Load related files together (e.g., all validator files at once)
- **Parallel Hash Verification**: Verify multiple file hashes concurrently during initialization
- **Async Audit Logging**: Log operations asynchronously to JSONL (non-blocking)

---

## 14. Documentation Requirements

**Manifest File (vocab_root/manifest.md):**

```markdown
# Vocabulary Manifest

## File Checksums (SHA-256)

| File | SHA-256 | Size | Updated | Source |
|------|---------|------|---------|--------|
| entities/offices.txt | 3a5c... | 124 B | 2026-01-15 | official_sos |
| entities/parties.txt | 7b2f... | 256 B | 2025-12-20 | manual_entry |
| validators/state_aliases.txt | 8d9e... | 512 B | 2026-01-10 | official_sos_api |

## File Descriptions

### entities/offices.txt
- **Purpose**: Canonical office names (President, Senator, etc.)
- **Format**: One office per line, no aliases
- **Source**: Official Secretary of State registry
- **Trust Score**: 0.99
- **Last Verified**: 2026-01-15

### validators/state_aliases.txt
- **Purpose**: State abbreviations → canonical names
- **Format**: `alias -> canonical_form`
- **Source**: FIPS standard + manual curation
- **Trust Score**: 0.95
- **Last Verified**: 2026-01-10
```

---

## Next Steps

1. **Implement VocabLoader** class following this specification
2. **Populate vocab TXT files** with extracted constants (see migration checklist)
3. **Create verified sources registry** with election authority URLs
4. **Deploy with election_mode=False** (testing phase)
5. **Enable election_mode=True** during 2026 election cycle
6. **Monitor audit logs** for anomalies and improper access
7. **Iterate on trust thresholds** based on real-world accuracy metrics

---

**Owner**: Election Integrity Team  
**Last Updated**: 2026-02-03  
**Status**: Design Complete, Ready for Implementation
