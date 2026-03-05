---
layout: default
---

# Technical Reference: API & Contract Specifications

**Version:** 1.0  
**Last Updated:** Current Session  
**Status:** Final Specification

---

## 1. Handler Registry API

**File:** `webapp/parser/handlers/registry.py`

### Function: `get_state_handler_module_path(state_abbr: str) -> str`

**Purpose:** Resolve state handler module path with registry fallback.

**Parameters:**

- `state_abbr` (str) – Two-letter state abbreviation (e.g., "NY", "CA")

**Returns:**

- str – Python module path (e.g., `"webapp.parser.handlers.states.new_york.new_york"`)
- Falls back to `DEFAULT_STATE_HANDLER` if no specific handler found

**Logic:**

1. Normalize state abbreviation (`normalize_state_name()`)
2. Check registry overrides dictionary
3. Attempt to find module via `importlib.util.find_spec()`
4. Return fallback if not found

**Example:**

```python
from webapp.parser.handlers.registry import get_state_handler_module_path

path = get_state_handler_module_path("NY")
# Returns: "webapp.parser.handlers.states.new_york.new_york"
# Falls back to: "webapp.parser.handlers.shared.state_scaffold"
```

---

### Function: `get_county_handler_module_path(state_abbr: str, county_name: str) -> str | None`

**Purpose:** Resolve county handler module path if override exists.

**Parameters:**

- `state_abbr` (str) – Two-letter state abbreviation
- `county_name` (str) – County name

**Returns:**

- str – Python module path if county override exists
- None – If no county-specific handler

**Logic:**

1. Check county registry overrides: `{(state, county): module_path}`
2. Normalize both state and county names
3. Return module path if found, None otherwise

**Example:**

```python
path = get_county_handler_module_path("NY", "Rockland")
# Returns: "webapp.parser.handlers.counties.new_york.rockland"
# or None if no override exists
```

---

## 2. Shared Scaffold API

**File:** `webapp/parser/handlers/shared/state_scaffold.py`

### Function: `parse(page, html_context, coordinator, session_id) → tuple[list, list, dict, dict] | None`

**Purpose:** Unified parsing entry point delegating to dynamic parser.

**Parameters:**

- `page` (Page) – Playwright page object
- `html_context` (dict) – Context dictionary with state/county/url

  ```python
  {
    "url": "https://...",
    "state": "NY",
    "county": "Rockland",
    "html": "<html>...",
    "detected_state": "NY",  # optional
    "detected_county": "Rockland"  # optional
  }
  ```

- `coordinator` (ContextCoordinator) – Coordination object for recording feedback
- `session_id` (str) – Session identifier for logging

**Returns:**

- tuple[list, list, dict, dict] – (headers, data_rows, contest, metadata) on success

  ```python
  (
    ["Candidate", "Party", "Votes"],  # headers
    [["John Doe", "Democratic", "1000"], ...],  # data_rows
    {"office": "President", "election_type": "general"},  # contest
    {"source": "ny.gov", "timestamp": "2025-01-22T14:30:00Z"}  # metadata
  )
  ```

- None – If parsing failed or no data found

**Logic:**

1. Import `parse as dynamic_parse` from `html_dynamic_fallback`
2. Normalize context (ensure state/county present)
3. Call `dynamic_parse(page, html_context, coordinator, session_id)`
4. Return result untouched

**Contract:**

- Preserves all parameters and return values from `html_dynamic_fallback.parse()`
- No transformation of input/output
- Coordinator is passed through for side-effect logging

**Example:**

```python
from webapp.parser.handlers.shared.state_scaffold import parse

headers, rows, contest, metadata = parse(
    page=browser_page,
    html_context={"state": "NY", "county": "Rockland", "url": "https://..."},
    coordinator=coord_obj,
    session_id="sess_12345"
)

if headers is not None:
    print(f"Found {len(rows)} rows")
else:
    print("Parsing failed")
```

---

## 3. Navigation Recipes API

**File:** `webapp/parser/navigator/navigation_recipes.py`

### Class: `NavigationRecipeStore`

**Purpose:** Recipe generation, storage, and replay from learned patterns.

**Constructor:**

```python
def __init__(
    self,
    enabled: bool = True,
    learned_log_path: str = DEFAULT_LEARNED_LOG,
    learned_min_ok_ratio: float = 0.8,
    learned_max_entries: int = 2000
) -> None:
```

**Parameters:**

- `enabled` (bool) – Enable learned recipe lookups (default: True)
- `learned_log_path` (str) – Path to JSONL learning log (default: `log/navigation_learning_log.jsonl`)
- `learned_min_ok_ratio` (float) – Minimum success ratio for recipe generation (default: 0.8)
- `learned_max_entries` (int) – Max entries to load from JSONL (default: 2000)

---

### Method: `get_recipes(state: str, county: str | None, page_url: str) → list[dict]`

**Purpose:** Retrieve ranked recipe candidates for a given context.

**Parameters:**

- `state` (str) – State abbreviation (e.g., "NY")
- `county` (str | None) – County name or None
- `page_url` (str) – Page URL for domain matching

**Returns:**

- list[dict] – Ranked recipes sorted by ok_ratio (highest first)

  ```python
  [
    {
      "id": "learned::ny_elections_v1",
      "ok_ratio": 0.95,
      "steps": [
        {"action": "click", "selector": ".button", "timeout": 5000},
        {"action": "wait", "timeout": 2000}
      ]
    },
    {
      "id": "hardcoded::ny_recipe",
      "ok_ratio": 0.85,
      "steps": [...]
    }
  ]
  ```

**Logic:**

1. Extract domain from page_url via `urlparse()`
2. Load hardcoded recipe candidates (if exist)
3. Load learned recipes from JSONL (filter: success=true, ok_ratio >= min_ratio, >= 2 actions)
4. Match recipes by state/county/domain filters
5. Merge and sort by ok_ratio descending
6. Return top candidates

---

### Method: `_build_learned_recipes() → dict[str, dict]`

**Purpose:** Convert JSONL log entries into replayable recipe objects.

**Returns:**

- dict – Dictionary of learned recipes keyed by "learned::script_id"

**Conversion Logic:**

```json
Input JSONL entry:
{
  "script_id": "ny_elections_v1",
  "success": true,
  "ok_ratio": 0.95,
  "telemetry": [
    {"action": "click", "selector": ".button", "status": "ok"},
    {"action": "wait", "timeout": 2000, "status": "ok"}
  ]
}

↓ Filter: success=true + ok_ratio >= 0.8 + len(telemetry) >= 2

↓ Convert telemetry to steps

Output recipe:
{
  "id": "learned::ny_elections_v1",
  "ok_ratio": 0.95,
  "steps": [
    {"action": "click", "selector": ".button", "timeout": 5000},
    {"action": "wait", "timeout": 2000}
  ]
}
```

---

### Method: `_telemetry_to_steps(telemetry: list[dict]) → list[dict]`

**Purpose:** Convert telemetry action trace to replayable step objects.

**Parameters:**

- `telemetry` (list[dict]) – List of telemetry actions with status

**Returns:**

- list[dict] – List of replayable step objects

**Mapping:**

```python
# Input telemetry action
{"action": "click", "selector": ".button", "status": "ok"}

# Output step
{"action": "click", "selector": ".button", "timeout": 5000}

# Input telemetry action
{"action": "wait", "timeout": 2000, "status": "ok"}

# Output step
{"action": "wait", "timeout": 2000}

# Input telemetry action
{"action": "fill", "selector": "input", "value": "search term", "status": "ok"}

# Output step
{"action": "fill", "selector": "input", "value": "search term", "timeout": 5000}
```

---

## 4. Context Coordinator API

**File:** `webapp/parser/Context_Integration/context_coordinator.py`

### Method: `record_navigation_feedback(navigation_script_id, success, context_before, context_after, telemetry_trace, metadata) → None`

**Purpose:** Record navigation telemetry and feedback to JSONL log for learning.

**Parameters:**

- `navigation_script_id` (str) – Recipe ID (e.g., "learned::ny_elections")
- `success` (bool) – True if navigation succeeded
- `context_before` (dict) – Context state before navigation

  ```python
  {"state": "NY", "county": "Rockland"}
  ```

- `context_after` (dict) – Context state after navigation

  ```python
  {"state": "NY", "county": "Rockland", "tables_found": 3}
  ```

- `telemetry_trace` (list[dict]) – Steps executed with status

  ```python
  [
    {"action": "click", "selector": ".btn", "status": "ok"},
    {"action": "wait", "timeout": 2000, "status": "ok"}
  ]
  ```

- `metadata` (dict) – Additional metadata

  ```python
  {
    "page_url": "https://elections.ny.gov/...",
    "source": "manual",
    "timestamp": "2025-01-22T14:30:00Z"
  }
  ```

**Side Effects:**

1. Enriches metadata with URL domain + hash
2. Constructs JSONL entry with all parameters
3. Appends entry to `log/navigation_learning_log.jsonl`

**Enrichment Logic:**

```python
meta = dict(metadata or {})
url = meta.get("page_url") or context_after.get("url")
if isinstance(url, str) and url:
    parsed = urlparse(url)
    meta.setdefault("url_domain", parsed.hostname)
    meta.setdefault("url_hash", hashlib.sha1(url.encode()).hexdigest()[:12])
```

**JSONL Entry Format:**

```json
{
  "timestamp": "2025-01-22T14:30:00Z",
  "script_id": "learned::ny_elections",
  "success": true,
  "context_before": {"state": "NY", "county": "Rockland"},
  "context_after": {"state": "NY", "county": "Rockland", "tables_found": 3},
  "telemetry": [
    {"action": "click", "selector": ".btn", "status": "ok"},
    {"action": "wait", "timeout": 2000, "status": "ok"}
  ],
  "metadata": {
    "page_url": "https://elections.ny.gov/...",
    "url_domain": "elections.ny.gov",
    "url_hash": "abc123def456",
    "source": "manual"
  }
}
```

---

## 5. State Router Integration

**File:** `webapp/parser/state_router.py`

### Integration Point: Handler Resolution

**Path:**

1. URL scanned for detectable state/county
2. Registry queried via `get_state_handler_module_path(state)`
3. Returned module imported and called
4. Module delegates to shared scaffold (either explicitly or via registry fallback)

**Fallback Chain:**

```branch
URL → detect state/county
  ↓
registry.get_state_handler_module_path(state)
  ↓
Check registry overrides (if any)
  ↓
Check filesystem (importlib.util.find_spec)
  ↓
Fall back to DEFAULT_STATE_HANDLER (shared scaffold)
  ↓
Import and call handler
```

**Contract:**

- Each imported handler module must export `parse()` function
- `parse()` signature: `(page, html_context, coordinator, session_id) → tuple | None`
- Shared scaffold delegates to `html_dynamic_fallback.parse()` without transformation

---

## 6. Learning JSONL Format

**File:** `log/navigation_learning_log.jsonl`

**Entry Schema:**

```json
{
  "timestamp": "ISO8601-string (e.g., 2025-01-22T14:30:00.123456Z)",
  "script_id": "string (e.g., learned::ny_elections_v1)",
  "success": "boolean (true if navigation succeeded)",
  "ok_ratio": "float (0.0 - 1.0, ratio of successful steps)",
  "context_before": {
    "state": "string (state abbreviation)",
    "county": "string (county name, optional)",
    "additional_keys": "any"
  },
  "context_after": {
    "state": "string",
    "county": "string",
    "tables_found": "integer (optional)",
    "additional_keys": "any"
  },
  "telemetry": [
    {
      "action": "string (click|wait|fill|autoscroll|scan_context)",
      "status": "string (ok|failed|timeout)",
      "selector": "string (CSS selector, for click/fill only)",
      "timeout": "integer (milliseconds, optional)",
      "value": "string (for fill only)",
      "details": {
        "additional_keys": "any"
      }
    }
  ],
  "metadata": {
    "page_url": "string (full URL)",
    "url_domain": "string (hostname, auto-added)",
    "url_hash": "string (SHA1 truncated to 12 chars, auto-added)",
    "source": "string (e.g., manual, automated)",
    "additional_keys": "any"
  }
}
```

**Entry Constraints:**

- Exactly one entry per navigation outcome (success or failure)
- Timestamp auto-added if missing
- URL domain + hash enriched by `record_navigation_feedback()`
- Only appended (never modified in place)

**Learned Recipe Filter Criteria:**

- `success == true`
- `ok_ratio >= 0.8` (configurable)
- `len(telemetry) >= 2`

---

## 7. Configuration Constants

**File Locations:**

- Registry defaults: `webapp/parser/handlers/registry.py`
- Navigation defaults: `webapp/parser/navigator/navigation_recipes.py`
- Coordinator settings: `webapp/parser/Context_Integration/context_coordinator.py`

**Key Constants:**

```python
# Registry
DEFAULT_STATE_HANDLER = "webapp.parser.handlers.shared.state_scaffold"
STATE_MODULE_MAP = {}  # State abbr → module path (optional overrides)
COUNTY_MODULE_MAP = {}  # (State, County) → module path (optional overrides)

# Navigation
DEFAULT_LEARNED_LOG = "log/navigation_learning_log.jsonl"
LEARNED_RECIPE_MIN_OK_RATIO = 0.8
LEARNED_RECIPE_MAX_ENTRIES = 2000
LEARNED_RECIPE_ID_PREFIX = "learned::"

# Coordinator
NAVIGATION_LOG_PATH = "log/navigation_learning_log.jsonl"
```

---

## 8. Error Handling

### Registry Errors

***Case: Module not found***

- Falls back to `DEFAULT_STATE_HANDLER`
- No exception raised

***Case: Invalid state abbreviation***

- Normalizes via `normalize_state_name()`
- Falls back to `DEFAULT_STATE_HANDLER` if normalization fails

### Recipe Generation Errors

***Case: JSONL missing or corrupted***

- Returns empty recipe list
- Falls back to hardcoded recipe candidates

***Case: Telemetry malformed***

- Skips entry with warning logged
- Continues processing remaining entries

### Coordinator Errors

***Case: JSONL append fails***

- Logs error
- Does not raise exception (non-blocking feedback recording)

---

## 9. Performance Characteristics

### Registry Lookup: O(1)

- Dictionary lookup for state/county
- Filesystem check via `importlib.util.find_spec()` (first call cached)

### Recipe Generation: O(n)

- n = number of entries in JSONL (max 2000)
- Single-pass filter + conversion

### Recipe Matching: O(m)

- m = number of recipe candidates (typically 2-5)
- Domain + state/county filter per recipe

**Overall:** Negligible overhead; learned recipe loading (~100ms for 2000 entries).

---

## 10. Testing Contracts

### Unit Test: Recipe Conversion

**Fixture:** Mock JSONL entry
**Assertion:** Converted recipe matches input telemetry action count
**Success criteria:** ID prefix == "learned::", steps generated, filters applied

### Integration Test: Navigation Smoke

**Fixture:** Real URLs from urls.txt
**Assertion:** Navigation runner executes without errors
**Success criteria:** Page loaded, context scanned, feedback recorded (if successful)

### End-to-End Test: Learning Loop

**Phase 1:** Record navigation feedback
**Phase 2:** Verify JSONL entry persisted
**Phase 3:** Regenerate learned recipes
**Success criteria:** Recipe recovered from JSONL, matches input pattern

---

## Summary

All APIs are documented with:

- ✓ Purpose statements
- ✓ Parameter specifications
- ✓ Return value contracts
- ✓ Error handling expectations
- ✓ Integration points
- ✓ Code examples

This reference enables:

- Safe API usage by external callers
- Clear contracts for future extensions
- Validated error handling
- Performance expectations
