# Modern UI Advanced Features

## Overview

This document describes the advanced features integrated from the classic UI into the modern dashboard interface.

## Features Added

### 1. Direct URL Batch Processing

**Location:** Right sidebar → Controls section

**Functionality:**

- Radio button option to select "Direct URLs" as file source
- Textarea for entering multiple URLs (one per line)
- Real-time validation with visual feedback
- Maximum 20 URLs per batch
- URL security validation (no credentials, http/https only)
- Session-specific draft persistence

**UI Elements:**

```html
<textarea id="directUrlTextarea" rows="4" placeholder="https://example.com/results1&#10;https://example.com/results2"></textarea>
<small id="directUrlFeedback">Enter one URL per line.</small>
<button id="directUrlClearBtn">Clear</button>
```

**JavaScript Functions:**

- `parseDirectUrlField()` - Validates URLs and returns array
- `initDirectUrlControl()` - Sets up event listeners and visibility
- Draft saved per session in `AdvancedFeatures.directUrlDraftBySession`

### 2. Batch Mode Processing

**Location:** Right sidebar → Controls section

**Functionality:**

- Checkbox to enable automatic processing of multiple contests
- When enabled, parser will automatically process all detected contests without prompting
- Passed to backend via `batch_mode: true` payload

**UI Element:**

```html
<label class="checkbox-label">
  <input type="checkbox" id="batchMode"> Batch Mode
</label>
<small>Process multiple contests automatically</small>
```

### 3. Filter Presets

**Location:** Right sidebar → Filters section

**Functionality:**

- Save current filter settings (confidence, state, log level) as named presets
- Load previously saved presets from dropdown
- Delete unwanted presets
- Presets stored in localStorage for persistence across sessions

**UI Elements:**

```html
<select id="filterPresetSelect">
  <option value="">— Save new preset —</option>
  <!-- Dynamic options -->
</select>
<button id="saveFiltersBtn">Save</button>
<button id="deletePresetBtn">Delete</button>
```

**JavaScript Functions:**

- `AdvancedFeatures.loadPresets()` - Load from localStorage
- `AdvancedFeatures.savePresets()` - Save to localStorage
- `AdvancedFeatures.getCurrentFilters()` - Get current filter state
- `AdvancedFeatures.applyFilters(filters)` - Apply preset
- `initFilterPresets()` - Initialize controls

### 4. Session Management Actions

**Location:** Right sidebar → Session Actions section

**Functionality:**

- **Clone Session:** Duplicate current session with all settings and logs
- **Export Data:** Export session results as JSON file
- **Clear Logs:** Remove all logs for current session

**UI Elements:**

```html
<button id="btnCloneSession">🔄 Clone Session</button>
<button id="btnExportSession">📤 Export Data</button>
<button id="btnClearSession">🗑️ Clear Logs</button>
```

**Socket Events:**

- `clone_session` - Server creates new session copy
- `session_cloned` - Notification when clone completes

### 5. Keyboard Shortcuts

**Global shortcuts available anywhere in the interface:**

| Shortcut | Action |
| ---------- | -------- |
| `Ctrl+E` | Export session data (JSON) |
| `Ctrl+Shift+E` | Export as CSV |
| `Ctrl+L` | Clear logs |
| `Ctrl+Shift+C` | Clone current session |
| `Ctrl+Shift+P` | Open command palette |
| `Ctrl+/` | Show shortcuts help modal |
| `Escape` | Close modals |

**JavaScript Function:**

- `initKeyboardShortcuts()` - Sets up global keydown listener
- `showShortcutsHelp()` - Displays modal with shortcut reference

## Architecture

### State Management

```javascript
const AdvancedFeatures = (() => {
  const filterPresets = new Map();
  let currentSessionId = null;
  let directUrlDraftBySession = new Map();
  
  return {
    filterPresets,
    loadPresets,
    savePresets,
    getCurrentFilters,
    applyFilters,
    directUrlDraftBySession,
    currentSessionId
  };
})();
```

### Initialization Sequence

```javascript
document.addEventListener('DOMContentLoaded', () => {
  initDirectUrlControl();
  initFilterPresets();
  initSessionActions();
  initKeyboardShortcuts();
});
```

### Socket Integration

**Outbound Events:**

```javascript
socket.emit('run_parser', {
  session_id: currentSessionId,
  file_source: 'direct',
  direct_urls: ['http://...', 'http://...'],
  batch_mode: true
});

socket.emit('clone_session', { 
  session_id: currentSessionId 
});
```

**Inbound Events:**

```javascript
socket.on('session_cloned', (data) => {
  // data: { old_session, new_session }
});

socket.on('session_deleted', (data) => {
  // data: { session_id }
});
```

## CSS Styling

All advanced feature styles are in `run_parser_modern.css`:

- `.advanced-option` - Container for conditional features (Direct URLs)
- `.advanced-option.hidden` - Hide when not selected
- `.btn-group-sm` - Small button groups (Save/Delete)
- `.btn-group-vertical` - Vertical button stacks (Session Actions)
- `.btn-sm` - Small button style
- `#shortcutsModal` - Keyboard shortcuts help modal
- `#directUrlFeedback.text-success/danger/muted` - URL validation colors

## Backend Integration

### Flask Routes (no changes needed)

The existing Socket.IO handlers support all features:

- `handle_run_parser()` - Accepts `direct_urls` and `batch_mode` parameters
- `handle_clone_session()` - Emits `session_cloned` event
- `handle_delete_session()` - Emits `session_deleted` event

### Python Parser Integration

Direct URLs are processed via:

```python
def main(urls=None, ...):
    if urls:  # Direct URLs override
        selected_urls = urls
    else:
        selected_urls = load_urls()  # From urls.txt
```

## User Workflow Examples

### Batch URL Processing

1. Click "Direct URLs" radio button
2. Paste multiple URLs (one per line)
3. Enable "Batch Mode" checkbox
4. Click "Run Parser"
5. Parser automatically processes all URLs and contests

### Filter Preset Management

1. Adjust filters (confidence, state, level)
2. Click "Save" in Filter Presets section
3. Enter preset name
4. Later: select preset from dropdown to restore filters

### Session Management

1. Run parser and accumulate results
2. Click "Clone Session" to duplicate
3. Click "Export Data" to download JSON
4. Click "Clear Logs" to reset log drawer

## Testing Checklist

- [ ] Direct URL validation rejects invalid URLs
- [ ] Direct URL validation rejects URLs with credentials
- [ ] Direct URL validation limits to 20 URLs
- [ ] Direct URL textarea shows/hides based on radio selection
- [ ] Direct URL draft persists when switching sessions
- [ ] Batch mode checkbox value included in run_parser payload
- [ ] Filter presets save to localStorage
- [ ] Filter presets restore all three filters
- [ ] Clone session creates new session
- [ ] Export data downloads valid JSON
- [ ] Clear logs empties log drawer
- [ ] All keyboard shortcuts work
- [ ] Shortcuts help modal displays correctly
- [ ] Advanced option visibility toggles properly

## Future Enhancements

- CSV export option for session data
- Import filter presets from file
- Bulk session operations (delete multiple)
- URL history/favorites
- Advanced URL validation (domain whitelist/blacklist)
- Preset sharing between users
- Session templates
- Auto-save drafts on blur
