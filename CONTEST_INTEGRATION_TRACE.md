# Contest Selection Integration - Detailed Trace & Test Plan

## Full Integration Flow Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BACKEND HANDLER LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Handler (json_handler.py, csv_handler.py, etc.)                            │
│      ├─ Detects multiple contests in data                                   │
│      ├─ Builds selection_context with:                                       │
│      │    ├─ "selector_data": {"contests": [...]}                           │
│      │    ├─ "handler": name of handler                                     │
│      │    ├─ "input_file": source filename                                  │
│      │    ├─ "state", "county", "year": metadata                            │
│      │    └─ "webapp": True (in json_handler only)                          │
│      │                                                                       │
│      └─> select_contest_auto_first(                                         │
│           coordinator=...,                                                   │
│           context=selection_context,                                        │
│           session_id=session_id,                                            │
│           force_interactive=True   # when >1 contest found                  │
│        )                                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│         CONTEST SELECTOR AUTO-SELECTION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  select_contest_auto_first() [line 1127]                                     │
│      │                                                                       │
│      ├─ Attempts non-interactive selection first                            │
│      │   └─> select_contest_noninteractive()                               │
│      │       └─ Returns high-confidence match if found                      │
│      │                                                                       │
│      ├─ If NO confident match AND force_interactive=True:                  │
│      │   └─> Calls select_contest() [INTERACTIVE]                          │
│      │                                                                       │
│      └─ Returns [list] of selected contests or None if cancelled            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓ (interactive path taken)
┌─────────────────────────────────────────────────────────────────────────────┐
│        INTERACTIVE CONTEST SELECTION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  select_contest() [line 1405]                                                │
│      │                                                                       │
│      ├─ PHASE 1: Candidate Preparation                                      │
│      │   ├─ Clusters contest titles by semantic similarity                  │
│      │   ├─ Scores each cluster with coordinator (if available)             │
│      │   ├─ Filters by requested year (if any)                             │
│      │   └─ Injects bundle records for variant grouping                     │
│      │                                                                       │
│      ├─ PHASE 2: Emission (NEW!)  ◄─── Integration Point                    │
│      │   │                                                                   │
│      │   └─> _emit_contest_options_to_webapp() [line 1629] ───┐             │
│      │       ├─ Checks: prompt.mode == "webapp"               │             │
│      │       ├─ Builds structured_options: [{                 │             │
│      │       │   "index": int,                                │             │
│      │       │   "label": str,                                │             │
│      │       │   "meta": str,                                 │             │
│      │       │   "metadata": dict                             │             │
│      │       │ }, ...]                                        │             │
│      │       ├─ Logs via: logger.info(                        │             │
│      │       │   type="contest_options",                      │             │
│      │       │   options=...,                                 │             │
│      │       │   context={state, county, handler, ...}        │             │
│      │       │ )                                              │             │
│      │       └─ Returns None (no blocking)                    │             │
│      │                                                         │             │
│      │                                                         ↓             │
│      ├─ PHASE 3: User Interaction (Awaits response)           ↓             │
│      │   └─> prompt.prompt_input(                             ↓             │
│      │       message="[PROMPT] ...",                          ↓             │
│      │       context={kind:"contest", ...},                   ↓             │
│      │       session_id=session_id                            ↓             │
│      │   )                                                     ↓             │
│      │       └─ Blocks until user submits via Socket.IO       ↓             │
│      │                                                         ↓             │
│      └─ PHASE 4: Response Processing                          ↓             │
│          ├─ Parses user indices/search                        ↓             │
│          ├─ Selects matching contests                         ↓             │
│          └─ Returns [selected_contests]                       ↓             │
│                                                               ↓             │
└───────────────────────────────────────────────────────────────┼─────────────┘
                                                                 ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              LOGGER/SOCKETIO INTEGRATION LAYER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Logger.info() emits the log entry                                           │
│      │                                                                       │
│      └─> socketio_emit_func(log_line) [Smart_Elections_Parser_Webapp.py]    │
│          │                                                                   │
│          ├─ Parse log line as JSON object                                   │
│          ├─ Normalize log object (add timestamp, etc.)                      │
│          ├─ Detect: obj.get("type") == "contest_options" ◄─── NEW!        │
│          │                                                                   │
│          ├─ IF contest_options:                                             │
│          │   ├─ Extract: options, context from log                          │
│          │   ├─ Build payload: {                                            │
│          │   │   "session_id": sid,                                         │
│          │   │   "context": {...},                                          │
│          │   │   "total_count": len(options),                               │
│          │   │   "options": [...]                                           │
│          │   │ }                                                             │
│          │   ├─ Store in session logs (audit trail)                         │
│          │   └─> socketio.emit('contest_options', payload, room=sid) ◄─── │
│          │                                                                   │
│          └─ ELSE (regular log):                                             │
│              ├─ Normalize and store                                         │
│              └─> socketio.emit('parser_output', obj, room=sid)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SOCKET.IO TRANSPORT                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Network: 'contest_options' event emitted to session room                    │
│  Payload: {                                                                  │
│    "session_id": "sess_...",                                                │
│    "context": {state, county, year, source, handler, url, input_file},     │
│    "total_count": N,                                                        │
│    "options": [                                                             │
│      {                                                                       │
│        "index": 0,                                                          │
│        "label": "2024 General Election",                                    │
│        "meta": "conf=0.95",                                                │
│        "metadata": {confidence: 0.95, year: 2024, ...}                      │
│      },                                                                      │
│      ...                                                                     │
│    ]                                                                         │
│  }                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FRONTEND (BROWSER)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  socket.on('contest_options', handler) [run_parser_modern.js:771]           │
│      │                                                                       │
│      └─> handleContestOptions(data)                                         │
│          ├─ Extract options from payload                                    │
│          ├─ Map to {index, label, meta, metadata} format                   │
│          ├─ Validate options array                                          │
│          │                                                                   │
│          └─> showPrompt({                                                    │
│              title: 'Select Contest',                                        │
│              message: 'context.message or default',                         │
│              options: [...],                                                │
│              placeholder: 'Search or click to choose'                       │
│          })                                                                  │
│              │                                                              │
│              ├─ Set modal title/message                                     │
│              ├─ Store options in activePromptOptions                        │
│              ├─ Call renderPromptOptions('') to display all                 │
│              │   └─ Create buttons for each option                          │
│              │   └─ Show bundle grouping if applicable                      │
│              └─ Show modal (remove 'hidden' class)                          │
│                                                                              │
│  User Interaction:                                                          │
│      ├─ Click option ──> selectOption(index) ──> selectedPromptOptions      │
│      ├─ Type search ──> renderPromptOptions(term) ──> filter display       │
│      ├─ Click Submit ──> submitPrompt()                                     │
│      │   └─> socket.emit('parser_prompt', {                                │
│      │       session_id: currentSessionId,                                  │
│      │       value: "0" or "0,2" (selected indices)                        │
│      │   })                                                                  │
│      │   └─> hidePrompt()                                                   │
│      └─ Click Cancel ──> submitPrompt('cancel')                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                   BACKEND RESPONSE HANDLER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  @socketio.on('parser_prompt')                                              │
│  def handle_parser_prompt(data):                                            │
│      ├─ Extract session_id, value (e.g., "0")                              │
│      ├─ Retrieve prompt_session from prompt.prompt_sessions                │
│      │                                                                       │
│      └─> prompt_session.set_response(value)                                 │
│          ├─ Store response in session.response                              │
│          └─ Signal event: session.event.set()  (unblocks wait)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│           BACKEND PROCESSING (select_contest continues)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  select_contest() while loop resumes:                                        │
│      │                                                                       │
│      ├─ Receives "0" from prompt.prompt_input()                            │
│      ├─ Parses as index or search term                                      │
│      ├─ Selects matching candidate: candidates[0]                          │
│      └─ Returns [selected_contest_dict]                                     │
│          │                                                                   │
│          └─> select_contest_auto_first() returns [...]                      │
│              │                                                               │
│              └─> json_handler processes selection:                          │
│                  ├─ Extracts selected contest's contest_ids                 │
│                  ├─ Filters data rows by contest_id                        │
│                  ├─ Builds output CSV/JSON                                  │
│                  └─ Returns (headers, data, contest, metadata)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Test Scenarios

### Scenario 1: Single Contest (Auto-selection)

- **Input**: Single contest in JSON file
- **Flow**: auto-selection succeeds → returns immediately → no modal shown
- **Verification**:
  - ✅ No contest_options event emitted
  - ✅ Parser produces output without user interaction

### Scenario 2: Multiple Contests (Interactive)

- **Input**: JSON file with 3 contests
- **Flow**: auto fails → interactive selection → user selects via modal
- **Verification**:
  - ✅ contest_options event received by frontend
  - ✅ Modal shows 3 options
  - ✅ Each option has index, label, metadata
  - ✅ User selection processed correctly
  - ✅ Selected contest extracted properly

### Scenario 3: CLI Mode (No Modal)

- **Input**: Multiple contests
- **Flow**: CLI mode (prompt.mode != "webapp")
- **Verification**:
  - ✅ No contest_options emission
  - ✅ CLI prompts user directly (text-based)
  - ✅ Works as before

### Scenario 4: Search/Filter

- **Input**: 10 contests
- **Flow**: User types search term in modal
- **Verification**:
  - ✅ Frontend filters display in real-time
  - ✅ No roundtrip to backend
  - ✅ User can select from filtered results

### Scenario 5: Multiple Selection

- **Input**: 3 contests
- **Flow**: User selects contest 0 and 2 (if allow_multiple=True)
- **Verification**:
  - ✅ Both contests extracted
  - ✅ Data combined correctly in output
  - ✅ Metadata includes both contest sources

### Scenario 6: Bundle Grouping

- **Input**: Contests with variants (e.g., different county levels)
- **Flow**: Frontend groups related contests
- **Verification**:
  - ✅ Bundles shown collapsed by default
  - ✅ User can expand to see variants
  - ✅ Selection of parent selects all children

### Scenario 7: Session Persistence

- **Input**: User starts contest selection, disconnects, reconnects
- **Flow**: Same session recovered
- **Verification**:
  - ✅ Contest options re-emitted on reconnect
  - ✅ Previous logs restored
  - ✅ User can continue selection

### Scenario 8: Timeout/Cancellation

- **Input**: User cancels prompt
- **Flow**: PromptCancelled exception → fallback to auto-select
- **Verification**:
  - ✅ select_contest returns None
  - ✅ Handler falls back to single top-ranked contest
  - ✅ Process completes with reasonable default

## Performance Considerations

1. **Emission**: Fast (no blocking) - logs asynchronously
2. **Frontend Display**: Instant rendering (<100ms for 100 options)
3. **Search/Filter**: Client-side, no roundtrip
4. **Bundle Rendering**: Efficient with collapsed-by-default
5. **Network**: Single Socket.IO event per contest discovery

## Debugging Checklist

### Backend

- [ ] Verify contest_selector.py loads without errors
- [ ] Check `prompt.mode` is set to "webapp" in test
- [ ] Verify _emit_contest_options_to_webapp() is called
- [ ] Check logger emits with type="contest_options"
- [ ] Verify socketio_emit_func intercepts it
- [ ] Check session room is valid

### Frontend

- [ ] Console: socket.on('contest_options') handler logs
- [ ] Console: handleContestOptions() receives full payload
- [ ] DOM: Modal element exists and is visible
- [ ] DOM: Options rendered with correct structure
- [ ] Interaction: Option clicks trigger selection
- [ ] Network: parser_prompt event sent with correct value

### End-to-End

- [ ] Backend receives and processes selection
- [ ] Output file contains selected contest data only
- [ ] Metadata includes selected contest info
- [ ] Logs stored for audit trail
- [ ] No errors in browser console
- [ ] No errors in server logs

## Implementation Files Changed

1. **webapp/parser/utils/contest_selector.py**
   - Added: `_emit_contest_options_to_webapp()` (lines ~1332-1407)
   - Modified: `select_contest()` - added emission call (lines ~1629-1636)

2. **webapp/Smart_Elections_Parser_Webapp.py**
   - Modified: `socketio_emit_func()` - added contest_options handling (lines ~862-878)

3. **No frontend changes** - Already had handlers

## Rollback Instructions

If needed, revert changes:

```bash
git diff HEAD webapp/parser/utils/contest_selector.py
git diff HEAD webapp/Smart_Elections_Parser_Webapp.py
git checkout -- <file>
```
