# Handler Migration Guide: From Static Constants to VocabLoader

**Purpose**: Step-by-step refactoring of election result handlers to use VocabLoader API instead of hardcoded constants.

**Timeline**: 4-5 weeks (phased by handler importance)  
**Risk Level**: Medium (comprehensive testing required)  
**Effort per Handler**: 4-8 hours (including testing)

---

## Overview: Before & After

### Before (Current State)

```python
# handlers/html_handler.py (BEFORE)

from webapp.parser.config import (
    VALID_OFFICES,
    VALID_STATES,
    VALID_PARTIES,
    RESULT_COLUMN_HEADERS
)

def parse_html(page, context):
    """Parse election results from HTML page"""
    
    # Static constants used for validation
    offices = VALID_OFFICES
    states = VALID_STATES
    parties = VALID_PARTIES
    expected_headers = RESULT_COLUMN_HEADERS
    
    # Validation happens without trust context
    # No audit trail
    # No anomaly detection
    # No election mode protection
    
    if validate_headers(page, expected_headers):
        rows = extract_table(page)
        for row in rows:
            if row["office"] in offices:
                # Process...
                pass
    
    return headers, data, contest, metadata
```

### After (With VocabLoader)

```python
# handlers/html_handler.py (AFTER)

from webapp.parser.config import get_vocab_loader

def parse_html(page, context, session_id=None):
    """Parse election results from HTML page"""
    
    # VocabLoader with session context + trust verification
    loader = get_vocab_loader(
        session_id=session_id,
        trust_threshold=0.85,  # Handler-specific threshold
        election_mode=context.get("election_mode", False)
    )
    
    try:
        # Dynamically load vocab with trust verification
        offices = loader.load_vocab_set("entities/offices.txt")
        states = loader.load_vocab_set("entities/jurisdictions.txt")
        parties = loader.load_vocab_set("entities/parties.txt")
    except VocabSecurityError as e:
        # Unverified entities in election mode
        logger.error(f"Vocab security error: {e}", extra={"session_id": session_id})
        return None, None, None, {"error": "Entity verification failed"}
    
    # Verify source before parsing
    source_url = context.get("url")
    source_info = loader.get_verified_source(source_url)
    if not source_info or source_info["trust_score"] < 0.85:
        logger.warning(f"Unverified source: {source_url}", extra={"session_id": session_id})
        return None, None, None, {"error": "Source not verified"}
    
    # Score headers for anomaly detection
    observed_headers = extract_header_row(page)
    score = loader.score_keyword_combination(
        keywords=observed_headers,
        context="header_validation"
    )
    
    if score["confidence"] == "low":
        reason = loader.get_anomaly_reason_definition("suspicious_header")
        if reason and reason["quarantine_required"]:
            logger.warning(f"Suspicious headers detected: {observed_headers}",
                          extra={"session_id": session_id})
            return None, None, None, {
                "error": "Anomaly detected",
                "anomaly_reason": "suspicious_header",
                "confidence": score["confidence"]
            }
    
    # Extract with full audit trail
    if validate_headers(page, expected_headers):
        rows = extract_table(page)
        for row in rows:
            # Resolve office with alias handling
            office_canonical = loader.resolve_alias("office", row["office"])
            if office_canonical:
                # Process with trust metadata
                # Audit log shows which vocab was used
                pass
    
    return headers, data, contest, metadata
```

---

## Phase 1: HTML Handler (Priority 1)

**Importance**: Highest (most complex, foundational for other handlers)

### Step 1: Analyze Current Implementation

**File**: `webapp/parser/handlers/html_handler.py`

**Current Constants Used**:

- `VALID_OFFICES`
- `VALID_STATES`
- `VALID_PARTIES`
- `RESULT_COLUMN_HEADERS`
- `COMMON_HEADER_VARIATIONS`

**Current Validation Logic**:

- Header matching (exact + substring)
- Office canonicalization (limited)
- State/party lookups

### Step 2: Add Imports

```python
# At top of html_handler.py, replace:
from webapp.parser.config import (
    VALID_OFFICES,
    VALID_STATES,
    VALID_PARTIES,
    RESULT_COLUMN_HEADERS
)

# With:
from webapp.parser.config import get_vocab_loader
from webapp.parser.Context_Integration.vocab_loader import (
    VocabSecurityError,
    VocabFileNotFound
)
```

### Step 3: Update Function Signature

```python
# Before
def parse_html(page, context):
    """Parse election results from HTML"""
    pass

# After
def parse_html(page, context, session_id=None):
    """Parse election results from HTML"""
    pass
```

### Step 4: Initialize VocabLoader

```python
def parse_html(page, context, session_id=None):
    """Parse election results from HTML"""
    
    # Initialize loader with session context
    loader = get_vocab_loader(
        session_id=session_id,
        trust_threshold=0.85,
        election_mode=context.get("election_mode", False)
    )
    
    # Continue with implementation...
```

### Step 5: Add Source Verification

```python
# Before: No source verification
if is_table_present(page):
    rows = extract_table(page)

# After: Verify source before processing
source_url = context.get("url")
source_info = loader.get_verified_source(source_url)

if not source_info or source_info["trust_score"] < 0.85:
    logger.warning(
        f"Unverified source: {source_url}",
        extra={"session_id": session_id, "trust_score": source_info.get("trust_score") if source_info else 0}
    )
    return None, None, None, {
        "error": "Source not verified",
        "trust_score": source_info.get("trust_score") if source_info else 0
    }

if is_table_present(page):
    rows = extract_table(page)
```

### Step 6: Replace Static Constant Usage

```python
# Before: Static constants
offices = VALID_OFFICES
states = VALID_STATES
parties = VALID_PARTIES

# After: Dynamic loading with error handling
try:
    offices = loader.load_vocab_set("entities/offices.txt")
    states = loader.load_vocab_set("entities/jurisdictions.txt")
    parties = loader.load_vocab_set("entities/parties.txt")
except VocabSecurityError as e:
    logger.error(f"Vocab security error: {e}", extra={"session_id": session_id})
    return None, None, None, {"error": "Entity verification failed"}
except VocabFileNotFound as e:
    logger.error(f"Vocab file missing: {e}", extra={"session_id": session_id})
    return None, None, None, {"error": "Vocab file not found"}
```

### Step 7: Add Header Validation with Scoring

```python
# Before: Simple header check
if expected_headers_match(page):
    process_table(page)

# After: Score headers for anomaly detection
observed_headers = extract_header_row(page)
score = loader.score_keyword_combination(
    keywords=observed_headers,
    context="header_validation"
)

if score["confidence"] == "low":
    reason = loader.get_anomaly_reason_definition("suspicious_header")
    if reason and reason["quarantine_required"]:
        logger.warning(
            f"Suspicious headers: {observed_headers}",
            extra={"session_id": session_id, "confidence": score["confidence"]}
        )
        return None, None, None, {
            "error": "Anomaly detected",
            "anomaly_reason": "suspicious_header",
            "confidence": score["confidence"]
        }

if is_table_valid(page):
    process_table(page)
```

### Step 8: Add Alias Resolution

```python
# Before: Direct lookup (if matches static list)
if candidate_office in VALID_OFFICES:
    process_row(row)

# After: Resolve via alias + get canonical form
candidate_office = loader.resolve_alias(
    "office",
    row["office"],
    exact_match=True
)

if candidate_office:
    # Use canonical form
    process_row(row, office=candidate_office)
else:
    # Unknown office - log anomaly
    logger.warning(
        f"Unknown office: {row['office']}",
        extra={"session_id": session_id}
    )
    return None, None, None, {
        "error": "Unknown office in results",
        "office": row["office"]
    }
```

### Step 9: Integrate Anomaly Logging

```python
# When anomaly detected, log with full context
if anomaly_detected:
    anomaly_reason = "missing_candidate"  # or other reason
    reason_def = loader.get_anomaly_reason_definition(anomaly_reason)
    
    logger.warning(
        f"Anomaly: {reason_def['description']}",
        extra={
            "session_id": session_id,
            "anomaly_reason": anomaly_reason,
            "severity": reason_def.get("severity"),
            "quarantine_required": reason_def.get("quarantine_required")
        }
    )
    
    if reason_def.get("quarantine_required"):
        # Quarantine data for manual review
        return None, None, None, {
            "error": "Data quarantined due to anomaly",
            "anomaly_reason": anomaly_reason
        }
```

### Step 10: Testing

**Test Cases to Add**:

```python
# tests/integration/test_html_handler_vocab.py

def test_parse_html_with_verified_source(self):
    """Parse HTML from verified SoS source"""
    page = mock_page(url="https://results.sos.ca.gov/2024/general")
    result = parse_html(page, {"election_mode": True}, session_id="test")
    assert result[0] is not None  # Headers returned
    
def test_parse_html_unverified_source_rejected(self):
    """Reject results from unverified source"""
    page = mock_page(url="https://unknown.example.com/results")
    result = parse_html(page, {"election_mode": True}, session_id="test")
    assert result[0] is None  # Rejected
    assert "Source not verified" in result[3].get("error", "")

def test_parse_html_suspicious_headers_quarantined(self):
    """Quarantine results with suspicious header patterns"""
    page = mock_page(headers=["HiddenVotes", "SecretData"])
    result = parse_html(page, {}, session_id="test")
    assert result[0] is None
    assert result[3].get("anomaly_reason") == "suspicious_header"

def test_parse_html_alias_resolution(self):
    """Resolve office aliases to canonical form"""
    page = mock_page(offices=["Pres.", "Sen."])
    result = parse_html(page, {}, session_id="test")
    # Should resolve "Pres." → "President", "Sen." → "Senator"
    assert "President" in result[0]
```

---

## Phase 2: XLSX Handler (Priority 1)

**File**: `webapp/parser/handlers/xlsx_handler.py`

**Differences from HTML**:

- No need for DOM navigation
- More structured header row
- Simpler table extraction

**Migration Steps** (abbreviated):

1. Add VocabLoader imports
2. Add session_id parameter
3. Initialize loader
4. Verify source URL
5. Load vocab sets (offices, states, parties)
6. Score headers (likely to have cleaner format)
7. Resolve aliases for office/party columns
8. Extract rows + validate

---

## Phase 3: CSV Handler (Priority 2)

**File**: `webapp/parser/handlers/csv_handler.py`

**Notes**: Similar to XLSX, but handle:

- Various delimiter options (`,`, `;`, `\t`)
- Quoted headers
- UTF-8 encoding variations

---

## Phase 4: PDF Handler (Priority 2)

**File**: `webapp/parser/handlers/pdf_handler.py`

**Complexity**: Highest (text extraction + OCR + regex)

**Additional Considerations**:

- Text extraction may produce malformed headers
- Use fuzzy matching (`exact_match=False`) for aliases
- Higher anomaly thresholds (OCR errors)

---

## Phase 5: JSON Handler (Priority 3)

**File**: `webapp/parser/handlers/json_handler.py`

**Notes**: Validate JSON structure + entity names

---

## Phase 6: TXT Handler (Priority 3)

**File**: `webapp/parser/handlers/txt_handler.py`

---

## Testing Strategy

### Unit Tests (Per Handler)

```python
# tests/unit/test_html_handler_vocab_migration.py

class TestHTMLHandlerVocabMigration:
    
    def test_vocab_loader_initialization(self):
        """Loader initialized with correct parameters"""
        # Mock get_vocab_loader
        # Assert called with trust_threshold=0.85
        pass
    
    def test_source_verification_before_parse(self):
        """Source verified before parsing begins"""
        # Mock unverified URL
        # Assert parse returns error
        pass
    
    def test_backward_compatibility_static_constant_import(self):
        """Old imports still work during grace period"""
        from webapp.parser.config import VALID_OFFICES
        assert len(VALID_OFFICES) > 0
```

### Integration Tests

```python
# tests/integration/test_handlers_vocab_integration.py

class TestHandlersWithVocabLoader:
    
    def test_html_pdf_csv_parse_verified_source(self):
        """All handlers parse verified source correctly"""
        for handler in [parse_html, parse_pdf, parse_csv]:
            result = handler(page, {"election_mode": True}, session_id="test")
            assert result[0] is not None
    
    def test_handlers_reject_unverified_source(self):
        """All handlers reject unverified source"""
        for handler in [parse_html, parse_pdf, parse_csv]:
            result = handler(page, {"url": "https://unknown.com"}, session_id="test")
            assert result[0] is None
    
    def test_audit_logs_show_handler_name(self):
        """Audit logs identify which handler performed operation"""
        parse_html(page, {}, session_id="audit_test")
        
        audit_file = Path("logs/vocab_audit.jsonl")
        entries = [json.loads(line) for line in audit_file.readlines()]
        
        assert any(e["handler"] == "html_handler" for e in entries)
    
    def test_handlers_no_regression_accuracy(self):
        """Parsing accuracy unchanged after migration"""
        # Compare old (static) vs new (vocab) handler
        # Results should be identical (or better with trust scoring)
        pass
```

---

## Rollback Plan

If issues detected post-deployment:

### Option 1: Quick Revert (< 5 min)

```bash
git revert HEAD  # Reverts handler changes
git push origin main
# CD pipeline redeploys with old handlers
```

### Option 2: Disable VocabLoader (Fallback to Static)

```python
# In config.py, temporary override
ENABLE_VOCAB_LOADER = False

# Handlers check flag:
if ENABLE_VOCAB_LOADER:
    loader = get_vocab_loader(...)
else:
    # Fallback to static constants
    offices = VALID_OFFICES
    states = VALID_STATES
```

### Option 3: Per-Handler Disable

```python
VOCAB_HANDLER_CONFIG = {
    "html": True,      # Use VocabLoader
    "pdf": False,      # Fallback to static
    "csv": True,
}

# In each handler:
if VOCAB_HANDLER_CONFIG.get(handler_type):
    loader = get_vocab_loader(...)
```

---

## Success Criteria

### Per-Handler Migration

- ✅ VocabLoader imports added
- ✅ session_id parameter added to function signature
- ✅ Source URL verified before processing
- ✅ Vocab sets loaded dynamically (not static)
- ✅ Header scoring implemented
- ✅ Alias resolution integrated
- ✅ Anomaly logging added
- ✅ Unit tests pass (95%+ coverage)
- ✅ Integration tests pass
- ✅ No parsing accuracy regression
- ✅ Audit logs show handler context
- ✅ Backward compatibility maintained (during grace period)

### Full Migration

- ✅ All 6 handlers migrated
- ✅ 0 security exceptions
- ✅ Audit logs flowing
- ✅ Performance acceptable (< 10ms per vocab load)
- ✅ Team trained on new API

---

## Timeline

| Phase | Handler | Start | Duration | Dependencies |
| ------- | --------- | ------- | ---------- | -------------- |
| 1 | HTML | Week 2 | 1 week | VocabLoader implemented |
| 1 | XLSX | Week 2 | 1 week | HTML tested |
| 2 | CSV | Week 3 | 4 days | XLSX tested |
| 2 | PDF | Week 3 | 4 days | CSV tested |
| 3 | JSON | Week 4 | 2 days | PDF tested |
| 3 | TXT | Week 4 | 2 days | JSON tested |
| — | Testing & Validation | Week 4-5 | 5 days | All handlers ready |
| — | Production Deployment | Week 5 | 1 day | Staging validated |

---

**Owner**: Election Integrity Team  
**Last Updated**: 2026-02-03  
**Status**: Ready for Implementation
