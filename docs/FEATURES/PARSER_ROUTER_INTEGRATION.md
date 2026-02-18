# Parser Integration with State Router & Format Detection

## Overview

This document explains how the URL and Filename parsers integrate with the existing state router and format detection systems in the parsing pipeline.

## Pipeline Flow

```branch
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT SOURCE                              │
│  ┌──────────────┐                    ┌──────────────┐           │
│  │  URL Input   │                    │ File Upload  │           │
│  │              │                    │              │           │
│  └──────┬───────┘                    └──────┬───────┘           │
└─────────┼──────────────────────────────────┼──────────────────┘
          │                                    │
          ▼                                    ▼
  ┌───────────────┐                    ┌──────────────┐
  │  URL Parser   │                    │   Filename   │
  │               │                    │    Parser    │
  │ • State       │                    │ • State      │
  │ • County      │                    │ • County     │
  │ • Year        │                    │ • Year       │
  │ • Vendor      │                    │ • Contest    │
  │ • Keywords    │                    │ • Scope      │
  └───────┬───────┘                    └──────┬───────┘
          │                                    │
          └────────────┬──────────────────────┘
                       │
                       │   METADATA HINTS
                       │
                       ▼
          ┌────────────────────────┐
          │   State Router         │
          │                        │
          │ Uses state hint to     │
          │ select handler:        │
          │ • Alabama handler      │
          │ • Georgia handler      │
          │ • Generic handler      │
          └───────┬────────────────┘
                  │
                  ▼
          ┌────────────────────────┐
          │   Format Router        │
          │                        │
          │ Uses vendor/format     │
          │ hints to detect:       │
          │ • Clarity ENR          │
          │ • PDF Canvass          │
          │ • CSV Export           │
          └───────┬────────────────┘
                  │
                  ▼
          ┌────────────────────────┐
          │   Handler Execution    │
          │                        │
          │ Processes data with    │
          │ state & format context │
          └────────────────────────┘
```

## Integration Points

### 1. State Router Integration

The state router in `webapp/parser/utils/state_router.py` uses parsed metadata to select the appropriate handler.

**Before Parser Integration:**

```python
# Old approach: Manual state selection or URL pattern matching
def get_handler(url, manual_state=None):
    if manual_state:
        return state_handlers.get(manual_state)
    
    # Try to guess from URL
    if '.ga.gov' in url:
        return georgia_handler
    elif '.al.gov' in url:
        return alabama_handler
    
    return generic_handler
```

**After Parser Integration:**

```python
from webapp.parser.url_parser import parse_url_simple
from webapp.parser.filename_parser import parse_filename_simple

def get_handler_with_parser(source, manual_state=None):
    """
    Get appropriate handler using parser metadata.
    
    Args:
        source: URL string or filename
        manual_state: User-provided state override
    
    Returns:
        Handler instance for the detected state
    """
    # User override takes precedence
    if manual_state:
        return state_handlers.get(manual_state.upper())
    
    # Parse source for hints
    if source.startswith('http'):
        parsed = parse_url_simple(source)
        state_hint = parsed.get('state')
        county_hint = parsed.get('county')
    else:
        parsed = parse_filename_simple(source)
        state_hint = parsed.get('state')
        county_hint = parsed.get('county')
    
    # Use state hint to select handler
    if state_hint:
        state_normalized = normalize_state_code(state_hint)
        handler = state_handlers.get(state_normalized)
        
        if handler:
            logger.info(f"Selected {state_normalized} handler based on parsed metadata")
            return handler
    
    # Fallback to generic handler
    logger.info("Using generic handler (no state detected)")
    return generic_handler
```

### 2. Format Router Integration

The format router in `webapp/parser/utils/format_router.py` uses vendor hints and extension info to detect the format.

**Before Parser Integration:**

```python
def detect_format(url, content):
    if 'clarityelections' in url:
        return 'clarity_enr'
    elif url.endswith('.pdf'):
        return 'pdf'
    elif url.endswith('.csv'):
        return 'csv'
    
    return scan_html_for_format(content)
```

**After Parser Integration:**

```python
from webapp.parser.url_parser import parse_url_simple

def detect_format_with_parser(url, content, filename=None):
    """
    Detect format using parser metadata.
    
    Args:
        url: Source URL (if web)
        content: Page content
        filename: Filename (if upload)
    
    Returns:
        Detected format string
    """
    # Parse URL if available
    if url:
        parsed_url = parse_url_simple(url)
        vendor = parsed_url.get('vendor_hint')
        
        # Vendor-specific formats
        if vendor == 'clarity':
            return 'clarity_enr'
        elif vendor == 'voteworks':
            return 'voteworks'
        elif vendor == 'dominion':
            return 'dominion'
    
    # Parse filename if available
    if filename:
        parsed_file = parse_filename_simple(filename)
        ext = parsed_file.get('extension', '').lower()
        format_hint = parsed_file.get('format_hint')
        
        # Format-specific detection
        if ext == '.pdf':
            if format_hint == 'canvass':
                return 'pdf_canvass'
            elif format_hint == 'summary':
                return 'pdf_summary'
            return 'pdf_generic'
        
        elif ext == '.csv':
            return 'csv_export'
        
        elif ext in ['.xlsx', '.xls']:
            return 'excel'
    
    # Fallback to content scan
    return scan_html_for_format(content)
```

### 3. Session Integration (Upload Flow)

When a file is uploaded, parsed metadata is stored in the session for downstream use.

**Implementation in `Smart_Elections_Parser_Webapp.py`:**

```python
@app.route("/upload/uploads", methods=["POST"])
def upload_to_uploads():
    file = request.files.get("data_file") or request.files.get("file")
    
    # ... authentication/validation ...
    
    ok, saved_name, err_path = _save_uploaded_file(file, str(UPLOADS_DIR))
    
    if ok and saved_name:
        # Store filename and format
        session['FORCE_PARSE_INPUT_FILE'] = saved_name
        session['FORCE_PARSE_FORMAT'] = saved_name.rsplit('.', 1)[-1].lower()
        session['manual_source_pref'] = 'uploads'
        
        # Parse filename for metadata hints
        parsed_filename = parse_filename_simple(saved_name)
        
        # Store parsed metadata in session
        if parsed_filename.get('state'):
            session['PARSED_STATE_HINT'] = parsed_filename['state']
        if parsed_filename.get('county'):
            session['PARSED_COUNTY_HINT'] = parsed_filename['county']
        if parsed_filename.get('year'):
            session['PARSED_YEAR_HINT'] = parsed_filename['year']
        if parsed_filename.get('contest_type'):
            session['PARSED_CONTEST_HINT'] = parsed_filename['contest_type']
        
        flash(f"File uploaded. Detected: {parsed_filename.get('state', 'Unknown state')}", "success")
    
    return redirect(url_for("ballot_lens"))
```

**Using Session Metadata in Handlers:**

```python
def process_uploaded_file(session_data):
    """Handler can access parsed metadata from session"""
    
    filename = session_data.get('FORCE_PARSE_INPUT_FILE')
    state_hint = session_data.get('PARSED_STATE_HINT')
    county_hint = session_data.get('PARSED_COUNTY_HINT')
    year_hint = session_data.get('PARSED_YEAR_HINT')
    
    # Use hints to guide processing
    if state_hint:
        handler = get_state_handler(state_hint)
    else:
        handler = generic_handler
    
    # Pass hints to handler for validation
    result = handler.process(
        filename=filename,
        expected_state=state_hint,
        expected_county=county_hint,
        expected_year=year_hint
    )
    
    return result
```

### 4. Contest Selection Integration

Parsed metadata can guide contest selection in the UI.

**Frontend Integration (ballot_lens_modern.js):**

```javascript
// After parsing URL or uploading file
async function loadMetadataHints() {
  // Get parsed metadata from backend
  const response = await fetch('/api/session/metadata_hints');
  const hints = await response.json();
  
  // Use hints to filter contest dropdown
  if (hints.state) {
    stateDropdown.value = hints.state;
    await loadCounties(hints.state);
  }
  
  if (hints.county) {
    countyDropdown.value = hints.county;
  }
  
  if (hints.year) {
    yearFilter.value = hints.year;
  }
  
  if (hints.contest_type) {
    contestTypeFilter.value = hints.contest_type;
  }
  
  // Refresh contest list with filters applied
  await loadContestList();
}
```

### 5. Validation Against Dataset

The parsers can validate their output against known data from Google Sheets.

**Validation Flow:**

```python
def validate_parser_output(url, parsed_metadata):
    """
    Compare parser output with actual data from database.
    
    Returns: (is_valid, confidence_score, discrepancies)
    """
    # Fetch known data for this URL
    actual_data = fetch_from_google_sheets(url)
    
    if not actual_data:
        return (True, 0.5, ["No reference data available"])
    
    discrepancies = []
    matches = 0
    checks = 0
    
    # Compare state
    if actual_data.get('State'):
        checks += 1
        if normalize_state(parsed_metadata['state']) == normalize_state(actual_data['State']):
            matches += 1
        else:
            discrepancies.append(f"State: parsed={parsed_metadata['state']}, actual={actual_data['State']}")
    
    # Compare county
    if actual_data.get('County'):
        checks += 1
        if normalize_county(parsed_metadata['county']) == normalize_county(actual_data['County']):
            matches += 1
        else:
            discrepancies.append(f"County: parsed={parsed_metadata['county']}, actual={actual_data['County']}")
    
    # Calculate confidence
    confidence = matches / checks if checks > 0 else 0.5
    is_valid = confidence >= 0.8  # 80% threshold
    
    return (is_valid, confidence, discrepancies)
```

## Fallback Hierarchy

The system uses a fallback hierarchy for metadata resolution:

1. **User-provided metadata** (highest priority)
   - Manual state selection
   - Manual county input
   - User corrections

2. **Parsed metadata hints**
   - URL parser output
   - Filename parser output
   - Session stored hints

3. **Content-based detection**
   - HTML scanning
   - Table header analysis
   - Pattern matching in data

4. **Default/generic handling** (lowest priority)
   - Generic state handler
   - Format guesser
   - Manual prompts

**Implementation:**

```python
def resolve_state(user_input, parsed_hint, content_detected, default='UNKNOWN'):
    """Resolve state using fallback hierarchy"""
    
    if user_input and user_input.upper() in VALID_STATE_CODES:
        logger.info(f"Using user-provided state: {user_input}")
        return user_input.upper()
    
    if parsed_hint and parsed_hint.upper() in VALID_STATE_CODES:
        logger.info(f"Using parsed state hint: {parsed_hint}")
        return parsed_hint.upper()
    
    if content_detected and content_detected.upper() in VALID_STATE_CODES:
        logger.info(f"Using content-detected state: {content_detected}")
        return content_detected.upper()
    
    logger.warning(f"No state detected, using default: {default}")
    return default
```

## Testing Integration

Validate that parsers work correctly with state and format routers:

```python
def test_router_integration():
    """Test that parsers integrate correctly with routers"""
    
    # Test URL with known state
    url = "https://results.sos.ga.gov/results/2024/general"
    parsed = parse_url_simple(url)
    handler = get_handler_with_parser(url)
    
    assert parsed['state'] == 'GA'
    assert handler.__class__.__name__ == 'GeorgiaHandler'
    
    # Test filename with known state
    filename = "Alabama_Jefferson_County_2024.pdf"
    parsed = parse_filename_simple(filename)
    handler = get_handler_with_parser(filename)
    
    assert parsed['state'] == 'AL'
    assert handler.__class__.__name__ == 'AlabamaHandler'
    
    # Test format detection
    format = detect_format_with_parser(
        url="https://results.enr.clarityelections.com/GA/105430/",
        content=None
    )
    assert format == 'clarity_enr'
```

## Configuration

Add parser settings to `.env`:

```bash
# Parser Configuration
ENABLE_URL_PARSER=true
ENABLE_FILENAME_PARSER=true

# Parser behavior
PARSER_STATE_PRIORITY=user,parsed,content,default
PARSER_CONFIDENCE_THRESHOLD=0.8

# Store parsed training data
STORE_PARSED_URLS=true
STORE_PARSED_FILENAMES=true
```

## Monitoring

Log parser usage and accuracy:

```python
def log_parser_result(source, parsed, actual, handler_used):
    """Log parser results for monitoring"""
    
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "parsed_state": parsed.get('state'),
        "actual_state": actual.get('state'),
        "match": parsed.get('state') == actual.get('state'),
        "handler": handler_used,
        "confidence": parsed.get('confidence', 0.0)
    }
    
    with open(LOG_DIR / "parser_accuracy.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

## Summary

The URL and Filename parsers are now fully integrated with:

- ✅ State router for handler selection
- ✅ Format router for format detection
- ✅ Upload flow with session storage
- ✅ Contest selection in UI
- ✅ Validation against known data
- ✅ Fallback hierarchy for robustness
- ✅ Monitoring and logging

This integration ensures consistent metadata extraction across all input modalities (URLs and files) and robust routing to appropriate handlers.
