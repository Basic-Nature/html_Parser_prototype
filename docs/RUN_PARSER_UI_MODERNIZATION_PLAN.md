# Run Parser UI Modernization Plan

**Version:** 1.0  
**Date:** January 11, 2026  
**Objective:** Transform run_parser.html from CLI emulation to a modern, reactive web application

---

## 📋 Executive Summary

### Current Issues

- **Debug console dominates screen** (400+ lines of terminal output)
- **No default minimize** for logs
- **CLI-style approach** doesn't leverage web capabilities
- **Limited data preview** for parsed outputs
- **Poor mobile/responsive design**
- **No multi-session visualization**
- **Underutilized parallel processing capabilities**

### Vision

Transform into a **dashboard-first** interface where:

1. **Results are primary** - data preview, statistics, downloads front and center
2. **Logs are secondary** - minimized by default, expandable when needed
3. **Real-time updates** - live progress cards, streaming charts
4. **Multi-session support** - visual session switcher with parallel tracking
5. **Mobile-friendly** - responsive grid layout, touch-optimized
6. **Data-centric** - inline table/JSON/CSV preview without leaving page

---

## 🎯 Phase 1: Layout Restructure (High Priority)

### 1.1 Three-Column Dashboard Layout

**Replace current layout with:**

```text
┌─────────────────────────────────────────────────────────────┐
│  Navbar (unchanged)                                         │
├────────┬──────────────────────────────────────┬────────────┤
│ Side-  │  Main Dashboard Area                 │ Session    │
│ bar    │  ┌────────────────────────────────┐  │ Panel      │
│        │  │ Results Grid (Cards/Preview)   │  │            │
│ Files  │  │                                │  │ Active:    │
│ URLs   │  │ ┌──────┐ ┌──────┐ ┌──────┐    │  │ • sess_1   │
│        │  │ │ CSV  │ │ JSON │ │ Stats│    │  │ • sess_2   │
│ (collap│  │ └──────┘ └──────┘ └──────┘    │  │            │
│ sible) │  │                                │  │ + New      │
│        │  └────────────────────────────────┘  │            │
│        │  ┌────────────────────────────────┐  │ Controls:  │
│        │  │ Progress Tracker               │  │ [Run]      │
│        │  │ ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿░░░░░░░ 65%     │  │ [Cancel]   │
│        │  └────────────────────────────────┘  │ [Bypass]   │
│        │  ┌────────────────────────────────┐  │            │
│        │  │ 🪵 Debug Console (minimized)   │  │ Filters:   │
│        │  │ [Expand] 3 warnings, 12 info   │  │ Level: ▼   │
│        │  └────────────────────────────────┘  │ Type:  ▼   │
└────────┴──────────────────────────────────────┴────────────┘
```

**Key Changes:**

- **Left Sidebar** - Files/URLs (existing, kept but auto-collapse on mobile)
- **Center Main** - Results dashboard (NEW primary view)
- **Right Panel** - Session management + controls (NEW dedicated panel)
- **Console** - Bottom collapsible drawer, minimized by default

### 1.2 Responsive Breakpoints

```css
/* Desktop (> 1200px): Three columns */
.layout-dashboard {
  display: grid;
  grid-template-columns: 280px 1fr 300px;
}

/* Tablet (768px - 1200px): Two columns + drawer */
@media (max-width: 1200px) {
  .layout-dashboard {
    grid-template-columns: 1fr 320px;
  }
  .sidebar { position: absolute; /* slide-out */ }
}

/* Mobile (< 768px): Single column + bottom tabs */
@media (max-width: 768px) {
  .layout-dashboard {
    grid-template-columns: 1fr;
  }
  .session-panel { position: fixed; bottom: 0; }
}
```

---

## 🎨 Phase 2: Results-First Dashboard (High Priority)

### 2.1 Results Grid Component

**Replace terminal-first with cards grid:**

```html
<div class="results-dashboard">
  <div class="results-header">
    <h2>Parsed Results <span class="badge">3 items</span></h2>
    <div class="result-actions">
      <button class="btn-icon" title="Grid view">⊞</button>
      <button class="btn-icon" title="List view">☰</button>
      <button class="btn-icon" title="Download all">⬇</button>
    </div>
  </div>
  
  <div class="results-grid" id="resultsGrid">
    <!-- Dynamically populated result cards -->
  </div>
  
  <div class="empty-state" hidden>
    <img src="/static/img/empty-results.svg" alt="">
    <p>No results yet. Run the parser to get started.</p>
  </div>
</div>
```

**Result Card Template:**

```html
<div class="result-card" data-file="CA__Alameda_County__20251206.csv">
  <div class="card-header">
    <span class="file-icon">📊</span>
    <span class="file-name">Alameda County</span>
    <span class="file-type-badge">CSV</span>
  </div>
  
  <div class="card-stats">
    <div class="stat">
      <span class="label">Rows:</span>
      <span class="value">1,234</span>
    </div>
    <div class="stat">
      <span class="label">Confidence:</span>
      <span class="value confidence-high">94.5%</span>
    </div>
  </div>
  
  <div class="card-preview">
    <table class="mini-preview">
      <thead><tr><th>Candidate</th><th>Votes</th><th>%</th></tr></thead>
      <tbody>
        <tr><td>John Doe</td><td>45,234</td><td>52.3%</td></tr>
        <tr><td>Jane Smith</td><td>41,123</td><td>47.7%</td></tr>
      </tbody>
    </table>
  </div>
  
  <div class="card-actions">
    <button class="btn-sm" onclick="previewFile(...)">👁 Preview</button>
    <button class="btn-sm" onclick="downloadFile(...)">⬇ Download</button>
    <button class="btn-sm" onclick="shareFile(...)">🔗 Share</button>
  </div>
</div>
```

### 2.2 Inline File Preview Modal

**Full-screen overlay for CSV/JSON/XLSX:**

```html
<div id="filePreviewModal" class="preview-modal">
  <div class="preview-header">
    <h3 id="previewFileName">Alameda_County_Results.csv</h3>
    <div class="preview-tabs">
      <button class="tab active" data-view="table">Table</button>
      <button class="tab" data-view="json">JSON</button>
      <button class="tab" data-view="metadata">Metadata</button>
      <button class="tab" data-view="chart">Charts</button>
    </div>
    <button class="btn-close" onclick="closePreview()">×</button>
  </div>
  
  <div class="preview-body">
    <!-- Table View (default) -->
    <div class="preview-pane active" id="tableView">
      <div class="table-controls">
        <input type="search" placeholder="Filter rows...">
        <button>Export Selected</button>
      </div>
      <div class="table-wrapper" id="previewTable">
        <!-- SheetJS or AG-Grid integration -->
      </div>
    </div>
    
    <!-- JSON View -->
    <div class="preview-pane" id="jsonView">
      <pre><code class="language-json" id="jsonContent"></code></pre>
    </div>
    
    <!-- Metadata View -->
    <div class="preview-pane" id="metadataView">
      <dl class="metadata-list">
        <dt>Handler:</dt><dd>pdf_handler</dd>
        <dt>State:</dt><dd>CA</dd>
        <dt>County:</dt><dd>Alameda</dd>
        <dt>Contest:</dt><dd>County Attorney General</dd>
        <dt>Rows:</dt><dd>1,234</dd>
        <dt>Confidence:</dt><dd>94.5%</dd>
        <dt>Timestamp:</dt><dd>2025-12-06 14:23:15</dd>
      </dl>
    </div>
    
    <!-- Chart View (bonus) -->
    <div class="preview-pane" id="chartView">
      <canvas id="resultChart"></canvas>
    </div>
  </div>
</div>
```

**Technologies:**

- **SheetJS (xlsx.js)** - Client-side Excel parsing/rendering
- **Prism.js** - JSON syntax highlighting
- **Chart.js** - Already used in quality dashboard

---

## ⚡ Phase 3: Real-Time Progress Tracking (Medium Priority)

### 3.1 Live Progress Cards

**Replace static stepper with animated cards:**

```html
<div class="progress-tracker">
  <div class="progress-card" data-session="sess_abc123">
    <div class="card-header">
      <h4>Session: sess_abc123</h4>
      <span class="status-badge running">Running</span>
    </div>
    
    <div class="progress-stages">
      <div class="stage completed">✓ Prepare</div>
      <div class="stage completed">✓ Source</div>
      <div class="stage active">⟳ Run (3/5 URLs)</div>
      <div class="stage pending">○ Review</div>
    </div>
    
    <div class="progress-bar">
      <div class="progress-fill" style="width: 60%;"></div>
      <span class="progress-text">60% - Processing URL 3/5</span>
    </div>
    
    <div class="progress-stats">
      <div class="stat">
        <span class="icon">🕒</span>
        <span class="label">Elapsed:</span>
        <span class="value" id="elapsed-sess_abc123">00:02:34</span>
      </div>
      <div class="stat">
        <span class="icon">📊</span>
        <span class="label">Parsed:</span>
        <span class="value">3 files</span>
      </div>
      <div class="stat">
        <span class="icon">⚠️</span>
        <span class="label">Warnings:</span>
        <span class="value">2</span>
      </div>
    </div>
    
    <div class="card-actions">
      <button class="btn-sm" onclick="viewLogs('sess_abc123')">View Logs</button>
      <button class="btn-sm btn-danger" onclick="cancelSession('sess_abc123')">Cancel</button>
    </div>
  </div>
</div>
```

### 3.2 Real-Time Chart Updates

**Live streaming bar chart of parsing progress:**

```javascript
// Update on each 'parser_output' event
function updateProgressChart(sessionId, data) {
  const ctx = document.getElementById('progressChart').getContext('2d');
  progressCharts[sessionId].data.datasets[0].data.push({
    x: new Date(),
    y: data.parsed_count || 0
  });
  progressCharts[sessionId].update('none'); // Smooth animation
}
```

**Mini Chart Integration:**

- Small sparkline showing parse rate over time
- Color-coded: green (success), yellow (warnings), red (errors)
- Embedded in progress card

---

## 🎛️ Phase 4: Enhanced Session Management (Medium Priority)

### 4.1 Session Panel (Right Sidebar)

**Dedicated session control panel:**

```html
<aside class="session-panel">
  <div class="panel-header">
    <h3>Sessions</h3>
    <button class="btn-icon" id="addSessionBtn">+ New</button>
  </div>
  
  <div class="session-list" id="sessionList">
    <!-- Active session cards -->
    <div class="session-card active" data-session="sess_abc123">
      <div class="session-header">
        <span class="session-badge">1</span>
        <span class="session-id">sess_abc123</span>
        <button class="btn-icon" onclick="cloneSession('sess_abc123')">⎘</button>
        <button class="btn-icon" onclick="deleteSession('sess_abc123')">🗑</button>
      </div>
      <div class="session-status">
        <span class="status-dot running"></span>
        <span class="status-text">Running - 3/5 URLs</span>
      </div>
      <div class="session-actions">
        <button class="btn-sm" onclick="switchSession('sess_abc123')">Switch</button>
      </div>
    </div>
    
    <div class="session-card" data-session="sess_def456">
      <div class="session-header">
        <span class="session-badge">2</span>
        <span class="session-id">sess_def456</span>
        <button class="btn-icon" onclick="cloneSession('sess_def456')">⎘</button>
        <button class="btn-icon" onclick="deleteSession('sess_def456')">🗑</button>
      </div>
      <div class="session-status">
        <span class="status-dot completed"></span>
        <span class="status-text">Completed - 10 files</span>
      </div>
      <div class="session-actions">
        <button class="btn-sm" onclick="switchSession('sess_def456')">Switch</button>
      </div>
    </div>
  </div>
  
  <div class="panel-controls">
    <h4>Active Session Controls</h4>
    
    <div class="control-group">
      <label>File Source</label>
      <select id="fileSourceSelect" class="form-control-sm">
        <option value="input">Input Folder</option>
        <option value="uploads">Manual Uploads</option>
      </select>
    </div>
    
    <div class="control-group">
      <label>
        <input type="checkbox" id="outputBypassCheckbox">
        Bypass Output
      </label>
    </div>
    
    <div class="action-buttons">
      <button class="btn btn-primary btn-block" id="runParserBtn">
        ▶ Run Parser
      </button>
      <button class="btn btn-danger btn-block" id="cancelParserBtn">
        ⏹ Cancel
      </button>
    </div>
  </div>
  
  <div class="panel-filters">
    <h4>Log Filters</h4>
    <select class="form-control-sm" id="logFilterSelect">
      <option value="all">All Levels</option>
      <option value="ERROR">Errors</option>
      <option value="WARNING">Warnings</option>
      <option value="INFO">Info</option>
    </select>
    
    <select class="form-control-sm" id="logTypeFilterSelect">
      <option value="all">All Types</option>
      <option value="input">Input</option>
      <option value="output">Output</option>
      <option value="validation">Validation</option>
    </select>
  </div>
</aside>
```

### 4.2 Multi-Session Parallel Tracking

**Visual comparison view:**

```html
<div class="multi-session-view" hidden>
  <h3>Parallel Sessions Overview</h3>
  <div class="session-grid">
    <!-- Up to 4 sessions side-by-side -->
    <div class="mini-session" data-session="sess_1">
      <div class="mini-header">Session 1</div>
      <div class="mini-progress">60%</div>
      <div class="mini-stats">3/5 URLs</div>
    </div>
    <!-- ... -->
  </div>
</div>
```

---

## 📱 Phase 5: Mobile Optimization (Medium Priority)

### 5.1 Bottom Sheet Navigation

**Mobile-first drawer system:**

```html
<!-- Mobile: Bottom sheet instead of sidebar -->
<div class="bottom-sheet" id="mobileMenu">
  <div class="sheet-handle"></div>
  <div class="sheet-tabs">
    <button class="tab active" data-tab="results">Results</button>
    <button class="tab" data-tab="files">Files</button>
    <button class="tab" data-tab="sessions">Sessions</button>
    <button class="tab" data-tab="logs">Logs</button>
  </div>
  <div class="sheet-content">
    <!-- Tab panels -->
  </div>
</div>
```

### 5.2 Touch Gestures

- **Swipe down** - Collapse log drawer
- **Swipe up** - Expand results view
- **Pull to refresh** - Reload file lists
- **Long press** - Quick actions menu

### 5.3 Responsive Result Cards

```css
@media (max-width: 768px) {
  .results-grid {
    grid-template-columns: 1fr; /* Single column */
  }
  
  .result-card {
    flex-direction: column; /* Stack vertically */
  }
  
  .card-preview {
    max-height: 150px; /* Limit preview height */
    overflow: hidden;
  }
}
```

---

## 🪵 Phase 6: Minimized Debug Console (High Priority)

### 6.1 Collapsible Log Drawer

**Bottom drawer, minimized by default:**

```html
<div class="log-drawer minimized" id="logDrawer">
  <div class="drawer-handle" onclick="toggleLogDrawer()">
    <span class="handle-icon">⌃</span>
    <span class="drawer-summary">
      <span class="log-badge error">2</span>
      <span class="log-badge warning">5</span>
      <span class="log-badge info">127</span>
    </span>
    <span class="drawer-hint">Click to expand logs</span>
  </div>
  
  <div class="drawer-content">
    <div class="drawer-header">
      <h3>Debug Console</h3>
      <div class="drawer-actions">
        <button class="btn-sm" onclick="clearLogs()">Clear</button>
        <button class="btn-sm" onclick="exportLogs()">Export</button>
        <button class="btn-sm" onclick="toggleLogDrawer()">Minimize</button>
      </div>
    </div>
    
    <div class="drawer-filters">
      <!-- Existing level/type filters, more compact -->
    </div>
    
    <div class="drawer-body" id="terminal">
      <!-- Log output here -->
    </div>
  </div>
</div>
```

### 6.2 Log Density Options

```html
<select id="logDensitySelect">
  <option value="compact">Compact (errors only)</option>
  <option value="normal">Normal (warnings + errors)</option>
  <option value="verbose">Verbose (all logs)</option>
</select>
```

**Auto-expand triggers:**

- Error occurs (slide up to show error)
- User clicks "View Logs" from progress card
- Prompt requires input

---

## 🔧 Phase 7: Advanced Features (Low Priority)

### 7.1 Bulk Operations

```html
<div class="bulk-actions" hidden>
  <div class="selection-summary">
    <span>3 results selected</span>
    <button onclick="clearSelection()">Clear</button>
  </div>
  <div class="bulk-buttons">
    <button class="btn-sm">Download All (ZIP)</button>
    <button class="btn-sm">Delete Selected</button>
    <button class="btn-sm">Compare</button>
  </div>
</div>
```

### 7.2 Search & Filter Results

```html
<div class="results-toolbar">
  <input type="search" 
         placeholder="Search results by state, county, or contest..." 
         id="resultSearchInput">
  
  <div class="filter-chips">
    <span class="chip">State: CA <button>×</button></span>
    <span class="chip">Type: CSV <button>×</button></span>
    <button class="btn-sm">+ Add Filter</button>
  </div>
  
  <select id="resultSortSelect">
    <option value="recent">Most Recent</option>
    <option value="name">Name (A-Z)</option>
    <option value="size">Size</option>
    <option value="confidence">Confidence</option>
  </select>
</div>
```

### 7.3 Result Comparison View

```html
<div class="comparison-view">
  <div class="comparison-header">
    <h3>Compare Results</h3>
    <button onclick="closeComparison()">×</button>
  </div>
  
  <div class="comparison-grid">
    <div class="comparison-pane">
      <h4>CA - Alameda County</h4>
      <table><!-- Data --></table>
    </div>
    
    <div class="comparison-pane">
      <h4>CA - San Francisco County</h4>
      <table><!-- Data --></table>
    </div>
  </div>
  
  <div class="comparison-diff">
    <h4>Differences Detected</h4>
    <ul>
      <li>Column "Party" missing in San Francisco</li>
      <li>Row count mismatch: 1,234 vs 987</li>
    </ul>
  </div>
</div>
```

### 7.4 Drag & Drop Upload

```javascript
// Drop zone for entire dashboard
document.addEventListener('drop', (e) => {
  e.preventDefault();
  const files = e.dataTransfer.files;
  uploadFiles(files, 'uploads');
});
```

### 7.5 Keyboard Shortcuts

```text
Ctrl+R - Run parser (current session)
Ctrl+N - New session
Ctrl+L - Toggle log drawer
Ctrl+F - Focus search
Ctrl+Shift+C - Clear logs
Esc - Close modals
```

---

## 📊 Phase 8: Data Visualization Enhancements

### 8.1 Result Statistics Dashboard

```html
<div class="stats-overview">
  <div class="stat-card">
    <span class="stat-icon">📊</span>
    <span class="stat-value">47</span>
    <span class="stat-label">Total Results</span>
  </div>
  
  <div class="stat-card">
    <span class="stat-icon">✓</span>
    <span class="stat-value">94.3%</span>
    <span class="stat-label">Avg Confidence</span>
  </div>
  
  <div class="stat-card">
    <span class="stat-icon">🕒</span>
    <span class="stat-value">2m 34s</span>
    <span class="stat-label">Avg Parse Time</span>
  </div>
  
  <div class="stat-card">
    <span class="stat-icon">⚠️</span>
    <span class="stat-value">3</span>
    <span class="stat-label">Warnings</span>
  </div>
</div>
```

### 8.2 Inline Charts (Chart.js)

**Candidate vote distribution in preview:**

```javascript
function renderResultChart(containerId, data) {
  new Chart(document.getElementById(containerId), {
    type: 'bar',
    data: {
      labels: data.candidates,
      datasets: [{
        label: 'Votes',
        data: data.votes,
        backgroundColor: '#2563eb'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false
    }
  });
}
```

---

## 🎨 Phase 9: UI/UX Polish

### 9.1 Loading States

```html
<!-- Skeleton loaders for results -->
<div class="result-card skeleton">
  <div class="skeleton-header"></div>
  <div class="skeleton-stats"></div>
  <div class="skeleton-preview"></div>
</div>
```

### 9.2 Empty States

```html
<div class="empty-state">
  <img src="/static/img/empty-results.svg" alt="No results">
  <h3>No parsed results yet</h3>
  <p>Run the parser to extract election data from your files or URLs.</p>
  <button class="btn btn-primary" onclick="focusRunButton()">
    Get Started
  </button>
</div>
```

### 9.3 Success Animations

```css
@keyframes pulse-success {
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
}

.result-card.just-added {
  animation: pulse-success 0.5s ease-out;
  border: 2px solid #10b981;
}
```

### 9.4 Toast Notifications

```javascript
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  
  setTimeout(() => toast.classList.add('show'), 10);
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
```

---

## 🛠️ Technical Implementation Details

### 10.1 File Preview Technologies

## Option A: SheetJS (xlsx.js)

```javascript
import * as XLSX from 'xlsx';

async function previewCSV(filePath) {
  const response = await fetch(filePath);
  const arrayBuffer = await response.arrayBuffer();
  const workbook = XLSX.read(arrayBuffer);
  const worksheet = workbook.Sheets[workbook.SheetNames[0]];
  const html = XLSX.utils.sheet_to_html(worksheet);
  document.getElementById('tableView').innerHTML = html;
}
```

## Option B: AG-Grid (Enterprise, more features)

```javascript
import { Grid } from 'ag-grid-community';

const gridOptions = {
  columnDefs: [...],
  rowData: [...],
  pagination: true,
  paginationPageSize: 50
};

new Grid(document.getElementById('previewTable'), gridOptions);
```

### 10.2 Real-Time Updates Architecture

```javascript
// Socket.IO event handlers
socket.on('parser_output', (data) => {
  // Update progress card
  updateProgressCard(data.session_id, data);
  
  // Update result grid if new file
  if (data.type === 'output' && data.metadata?.output_file) {
    addResultCard(data.metadata);
  }
  
  // Append to log drawer (minimized)
  appendToLogDrawer(data);
  
  // Update live chart
  updateProgressChart(data.session_id, data);
});

socket.on('session_state', (data) => {
  updateSessionPanel(data.session_id, data.state, data.phase);
});
```

### 10.3 State Management

## Option A: Lightweight (Vanilla JS)

```javascript
const AppState = {
  sessions: new Map(),
  results: [],
  activeSession: null,
  
  addResult(result) {
    this.results.unshift(result);
    this.render();
  },
  
  switchSession(sessionId) {
    this.activeSession = sessionId;
    this.render();
  },
  
  render() {
    renderResults();
    renderSessionPanel();
    renderProgressCards();
  }
};
```

## Option B: Reactive (Alpine.js or Petite-Vue)

```html
<div x-data="appData">
  <div x-for="result in results" :key="result.id">
    <div class="result-card" @click="previewResult(result)">
      <h4 x-text="result.name"></h4>
    </div>
  </div>
</div>
```

### 10.4 Performance Optimizations

1. **Virtual Scrolling** - Only render visible result cards
2. **Lazy Loading** - Load file previews on-demand
3. **Debounced Search** - 300ms delay on search input
4. **Web Workers** - Parse large CSVs in background thread
5. **IndexedDB Caching** - Cache parsed results locally

---

## 📦 Dependencies to Add

### Frontend Libraries

```json
{
  "dependencies": {
    "xlsx": "^0.18.5",
    "chart.js": "^4.4.0",
    "prismjs": "^1.29.0",
    "fuse.js": "^7.0.0"
  },
  "devDependencies": {
    "terser": "^5.24.0",
    "postcss": "^8.4.32",
    "autoprefixer": "^10.4.16"
  }
}
```

## Or CDN alternatives: (no build step)

```html
<script src="https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js"></script>
```

---

## 🗓️ Implementation Roadmap

### Sprint 1 (Week 1-2): Core Layout Restructure

- [ ] Implement three-column grid layout
- [ ] Create collapsible log drawer (minimized by default)
- [ ] Move controls to session panel
- [ ] Basic responsive breakpoints

### Sprint 2 (Week 3-4): Results Dashboard

- [ ] Build result card component
- [ ] Implement results grid
- [ ] Add empty state
- [ ] Basic file download links

### Sprint 3 (Week 5-6): File Preview

- [ ] Integrate SheetJS for CSV/XLSX
- [ ] Build preview modal
- [ ] Add JSON/metadata tabs
- [ ] Implement table search/filter

### Sprint 4 (Week 7-8): Progress Tracking

- [ ] Create live progress cards
- [ ] Add session timer
- [ ] Implement streaming chart updates
- [ ] Build multi-session view

### Sprint 5 (Week 9-10): Mobile Optimization

- [ ] Implement bottom sheet
- [ ] Add touch gestures
- [ ] Optimize for small screens
- [ ] Test on real devices

### Sprint 6 (Week 11-12): Polish & Features

- [ ] Add keyboard shortcuts
- [ ] Implement toast notifications
- [ ] Build comparison view
- [ ] Performance optimization
- [ ] Accessibility audit

---

## 🎯 Success Metrics

### User Experience

- **Time to first result preview:** < 2 seconds
- **Log drawer default state:** Minimized (80%+ sessions)
- **Mobile usability score:** > 90 (Lighthouse)
- **Session switch time:** < 500ms

### Performance

- **First Contentful Paint:** < 1.5s
- **Time to Interactive:** < 3s
- **Result card render:** < 100ms each
- **Virtual scroll FPS:** 60fps

### Adoption

- **Preview usage:** > 60% of results previewed before download
- **Multi-session usage:** > 30% of users run parallel sessions
- **Mobile usage:** > 25% of sessions on mobile devices

---

## 🚧 Migration Strategy

### Phase A: Parallel Implementation (Recommended)

1. Keep existing `run_parser.html` as-is
2. Create `run_parser_v2.html` with new design
3. Add feature flag in Flask: `USE_NEW_PARSER_UI`
4. Gradual rollout to beta users

### Phase B: Incremental Refactor (Lower Risk)

1. Add log drawer collapse (preserve existing layout)
2. Add result cards above log panel
3. Move controls to sidebar
4. Full layout switch when ready

### Phase C: Big Bang (Fastest)

1. Build entire new UI in feature branch
2. Test thoroughly with QA team
3. Deploy all at once
4. Provide rollback option

**Recommendation:** **Phase A** for production safety.

---

## 🔐 Security Considerations

1. **File Preview XSS:** Sanitize CSV/JSON content before rendering
2. **CORS:** Ensure file downloads respect same-origin policy
3. **Path Traversal:** Validate file paths in preview API
4. **CSP:** Update nonces for dynamically loaded SheetJS
5. **Rate Limiting:** Prevent excessive preview requests

---

## ♿ Accessibility Checklist

- [ ] Keyboard navigation for all controls
- [ ] ARIA labels on all interactive elements
- [ ] Focus trap in modals
- [ ] Screen reader announcements for state changes
- [ ] High contrast mode support
- [ ] Reduced motion respect (disable animations)
- [ ] Skip links for main content
- [ ] Semantic HTML (proper heading hierarchy)

---

## 📚 Documentation Updates Required

1. **User Guide:** New UI walkthrough with screenshots
2. **API Docs:** Document new file preview endpoints
3. **Keyboard Shortcuts:** Cheat sheet page
4. **Mobile Guide:** Touch gesture reference
5. **Developer Guide:** State management architecture

---

## 🎓 Training Materials

### For End Users

- **Video Tutorial:** "New Parser UI Tour" (5 min)
- **Interactive Demo:** Guided walkthrough on first visit
- **Changelog:** "What's New" modal on login

### For Developers

- **Architecture Doc:** Component hierarchy diagram
- **Code Examples:** Common patterns (add result, switch session)
- **Migration Guide:** Converting old handlers to new format

---

## 🤝 Stakeholder Sign-Off

### Key Decisions Requiring Approval

1. **Layout:** Three-column vs. two-column?
2. **File Preview Lib:** SheetJS (free) vs. AG-Grid (paid)?
3. **Mobile Priority:** Bottom sheet vs. hamburger menu?
4. **Migration:** Parallel (v2) vs. incremental refactor?
5. **Timeline:** 12-week sprint vs. phased 6-month rollout?

---

## 📝 Open Questions

1. Should we keep URLs.txt sidebar or integrate into main area?
2. What's the max number of parallel sessions to support?
3. Do we need offline mode (PWA with service worker)?
4. Should result cards auto-refresh when files change?
5. Export format for bulk downloads (ZIP, tar.gz)?

---

## ✅ Next Steps

1. **Review this plan** with product owner and dev team
2. **Create wireframes** for key screens (Figma/Sketch)
3. **Set up feature flag** in Flask app
4. **Spike SheetJS integration** (1-day proof of concept)
5. **Design system audit** (ensure consistency with quality dashboard)
6. **Kick off Sprint 1** with layout restructure

---

**Document Owner:** Development Team  
**Last Updated:** January 11, 2026  
**Status:** 🟡 Draft - Awaiting Approval
