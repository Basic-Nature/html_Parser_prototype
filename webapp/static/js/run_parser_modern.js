/**
 * Smart Elections Parser - Modern UI JavaScript
 * Phase 1: Core Layout, SheetJS Integration, Component Interactions
 */

// ============================================
// Configuration & Constants
// ============================================

const CONFIG = {
  toastDuration: 4000,
  logBufferSize: 500,
  searchDebounceMs: 300,
  sessionRefreshMs: 1000,
  maxPreviewRows: 500,
};

const STATES = [
  'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
  'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
  'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
  'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
  'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
];

// ============================================
// Socket.IO Setup (existing integration)
// ============================================

const socket = io({
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: 5,
});

let currentSessionId = null;

socket.on('connect', () => {
  console.log('[Socket.IO] Connected:', socket.id);
});

socket.on('session_id', (data) => {
  currentSessionId = data.session_id;
  console.log('[Session] ID:', currentSessionId);
  updateSessionsList();
});

socket.on('parser_output', (data) => {
  addLog(data);
});

socket.on('session_state', (data) => {
  console.log('[Session State]', data);
  updateProgressCard(data);
  updateSessionsList();
});

socket.on('session_list', (data) => {
  updateSessionsList(data.sessions);
});

// ============================================
// State Management
// ============================================

const state = {
  results: [],
  sessions: [],
  logs: [],
  filters: {
    search: '',
    confidence: 0,
    state: '',
    level: '',
  },
  selectedResults: new Set(),
  currentFile: null,
  autoScroll: true,
};

// ============================================
// Utility Functions
// ============================================

function $(selector) {
  return document.querySelector(selector);
}

function $$(selector) {
  return document.querySelectorAll(selector);
}

function showToast(message, type = 'info', duration = CONFIG.toastDuration) {
  const toast = document.createElement('div');
  toast.className = `toast ${type} fade-in`;
  
  const icons = {
    info: 'ℹ️',
    success: '✓',
    warning: '⚠️',
    error: '✗',
  };
  
  toast.innerHTML = `
    <div class="toast-icon">${icons[type] || type}</div>
    <div class="toast-message">${escapeHtml(message)}</div>
    <button class="toast-close">×</button>
  `;
  
  $('#toastContainer').appendChild(toast);
  
  toast.querySelector('.toast-close').addEventListener('click', () => {
    toast.remove();
  });
  
  setTimeout(() => {
    toast.style.animation = 'slideOutRight 300ms ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(date) {
  const d = new Date(date);
  return d.toLocaleString();
}

function parseConfidenceClass(confidence) {
  if (confidence >= 90) return 'high-confidence';
  if (confidence >= 70) return 'medium-confidence';
  return 'low-confidence';
}

// ============================================
// Log Management
// ============================================

function addLog(logObj) {
  const normalized = {
    timestamp: logObj.timestamp || Date.now(),
    level: logObj.level || 'INFO',
    type: logObj.type || 'info',
    message: logObj.message || '',
    sessionId: logObj.session_id || currentSessionId,
  };
  
  state.logs.push(normalized);
  
  // Keep buffer size manageable
  if (state.logs.length > CONFIG.logBufferSize) {
    state.logs.shift();
  }
  
  // Update counts
  updateLogCounts();
  
  // Apply filter and render
  renderLogs();
  
  // Auto-scroll if enabled
  if (state.autoScroll) {
    const logOutput = $('#logOutput');
    logOutput.scrollTop = logOutput.scrollHeight;
  }
}

function updateLogCounts() {
  const errors = state.logs.filter(l => l.level === 'ERROR').length;
  const warnings = state.logs.filter(l => l.level === 'WARNING').length;
  const infos = state.logs.filter(l => l.level === 'INFO').length;
  
  $('#errorCount').textContent = errors;
  $('#warningCount').textContent = warnings;
  $('#infoCount').textContent = infos;
}

function renderLogs() {
  const filtered = state.logs.filter(log => {
    if (state.filters.level && log.level !== state.filters.level) return false;
    return true;
  });
  
  const logOutput = $('#logOutput');
  logOutput.innerHTML = filtered.map(log => `
    <div class="log-line">
      <span class="log-timestamp">${new Date(log.timestamp).toLocaleTimeString()}</span>
      <span class="log-level ${log.level.toLowerCase()}">${log.level}</span>
      <div class="log-message">${escapeHtml(log.message)}</div>
    </div>
  `).join('');
}

// ============================================
// Results Management (SheetJS Integration)
// ============================================

function createResultCard(result) {
  const confClass = parseConfidenceClass(result.confidence || 0);
  
  return `
    <div class="result-card" data-result-id="${result.id}">
      <div class="card-header">
        <div class="card-icon">📊</div>
        <div class="card-title">
          <div class="card-name">${escapeHtml(result.name)}</div>
          <span class="card-type-badge">${result.type.toUpperCase()}</span>
        </div>
      </div>
      
      <div class="card-stats">
        <div class="card-stat">
          <span class="stat-label">Rows</span>
          <span class="stat-value">${(result.rows || 0).toLocaleString()}</span>
        </div>
        <div class="card-stat">
          <span class="stat-label">Confidence</span>
          <span class="stat-value ${confClass}">${(result.confidence || 0).toFixed(1)}%</span>
        </div>
      </div>
      
      <div class="card-preview">${result.preview || 'No preview available'}</div>
      
      <div class="card-actions">
        <button class="btn-sm" onclick="previewFile('${result.id}')">👁 Preview</button>
        <button class="btn-sm" onclick="downloadFile('${result.id}')">⬇ Download</button>
        <input type="checkbox" style="margin-left: auto;" onchange="toggleSelectResult('${result.id}')">
      </div>
    </div>
  `;
}

function renderResults() {
  const filtered = state.results.filter(r => {
    if (state.filters.search && !r.name.toLowerCase().includes(state.filters.search.toLowerCase())) {
      return false;
    }
    if (state.filters.confidence && (r.confidence || 0) < state.filters.confidence) {
      return false;
    }
    if (state.filters.state && !r.state?.includes(state.filters.state)) {
      return false;
    }
    return true;
  });
  
  const grid = $('#resultsGrid');
  if (filtered.length === 0) {
    grid.style.display = 'none';
    $('#emptyState').style.display = 'flex';
  } else {
    grid.style.display = 'grid';
    grid.innerHTML = filtered.map(r => createResultCard(r)).join('');
    $('#emptyState').style.display = 'none';
  }
}

// ============================================
// File Preview Modal with SheetJS
// ============================================

function previewFile(resultId) {
  const result = state.results.find(r => r.id === resultId);
  if (!result) return;
  
  state.currentFile = result;
  
  // Update modal header
  $('#previewTitle').textContent = `Preview: ${result.name}`;
  
  // Load and parse file based on type
  loadFilePreview(result);
  
  // Show modal
  $('#previewModal').style.display = 'flex';
}

function loadFilePreview(result) {
  // In real implementation, fetch the actual file
  // For now, show sample data
  
  if (result.type === 'csv' || result.type === 'xlsx') {
    displayTablePreview(result);
  } else if (result.type === 'json') {
    displayJsonPreview(result);
  }
  
  displayFileInfo(result);
}

function displayTablePreview(result) {
  // Simulated data - in production, load actual file
  const sampleData = [
    ['Candidate', 'Votes', 'Percentage', 'Party'],
    ['Alice Johnson', '45234', '52.3%', 'Democratic'],
    ['Bob Smith', '41123', '47.7%', 'Republican'],
  ];
  
  const table = $('#previewTable');
  table.innerHTML = '';
  
  // Headers
  const thead = document.createElement('thead');
  thead.innerHTML = `
    <tr>
      ${sampleData[0].map(h => `<th>${escapeHtml(h)}</th>`).join('')}
    </tr>
  `;
  table.appendChild(thead);
  
  // Body
  const tbody = document.createElement('tbody');
  sampleData.slice(1, CONFIG.maxPreviewRows + 1).forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = row.map(cell => `<td>${escapeHtml(cell)}</td>`).join('');
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function displayJsonPreview(result) {
  const tabContent = $('#tabPreview');
  const sampleJson = {
    contest: 'County Attorney',
    candidates: [
      { name: 'Alice Brown', votes: 45234 },
      { name: 'Bob Smith', votes: 41123 },
    ],
  };
  
  const pre = document.createElement('pre');
  pre.textContent = JSON.stringify(sampleJson, null, 2);
  pre.style.background = 'var(--bg-primary)';
  pre.style.padding = 'var(--spacing-lg)';
  pre.style.borderRadius = 'var(--radius-md)';
  pre.style.overflow = 'auto';
  
  tabContent.innerHTML = '';
  tabContent.appendChild(pre);
}

function displayFileInfo(result) {
  $('#infoFileName').textContent = result.name;
  $('#infoRows').textContent = (result.rows || 0).toLocaleString();
  $('#infoColumns').textContent = result.columns || 'N/A';
  $('#infoConfidence').textContent = (result.confidence || 0).toFixed(1) + '%';
  $('#infoHandler').textContent = result.handler || 'unknown';
  $('#infoTimestamp').textContent = formatDate(result.timestamp || Date.now());
}

// ============================================
// Session Management
// ============================================

function updateSessionsList(sessions = state.sessions) {
  state.sessions = sessions || state.sessions;
  
  const list = $('#sessionsList');
  
  if (!state.sessions.length) {
    list.innerHTML = '<p style="color: var(--text-muted); font-size: 0.875rem;">No sessions</p>';
    return;
  }
  
  list.innerHTML = state.sessions.map(session => `
    <div class="session-card ${session.id === currentSessionId ? 'active' : ''}">
      <div class="session-id">${session.id}</div>
      <div class="session-progress">
        <span class="session-status ${session.status || 'pending'}"></span>
        ${session.progress || 'Initializing...'}
      </div>
    </div>
  `).join('');
  
  $('#sessionCount').textContent = state.sessions.length;
}

function updateProgressCard(sessionData) {
  if (!sessionData || sessionData.state === 'IDLE') {
    $('#progressCard').style.display = 'none';
    return;
  }
  
  $('#progressCard').style.display = 'block';
  $('#progressSessionId').textContent = sessionData.session_id;
  $('#progressStatus').textContent = sessionData.state;
  
  // Update phases
  const phases = ['PREPARE', 'SOURCE', 'RUN', 'REVIEW'];
  const stagesHtml = phases.map(phase => {
    let className = '';
    if (phase === sessionData.phase) className = 'active';
    else if (phases.indexOf(phase) < phases.indexOf(sessionData.phase)) className = 'completed';
    return `<div class="stage ${className}">${phase}</div>`;
  }).join('');
  
  $('#progressStages').innerHTML = stagesHtml;
}

// ============================================
// Event Listeners: Sidebar Controls
// ============================================

// File Source Toggle
$$('input[name="fileSource"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    socket.emit('set_manual_source', {
      session_id: currentSessionId,
      file_source: e.target.value,
    });
  });
});

// Output Bypass Toggle
$('#outputBypass').addEventListener('change', () => {
  socket.emit('toggle_output_bypass', {
    session_id: currentSessionId,
  });
});

// Run Parser Button
$$('#btnRunParser, #btnRunParser2').forEach(btn => {
  btn.addEventListener('click', () => {
    socket.emit('run_parser', {
      session_id: currentSessionId,
      file_source: $('input[name="fileSource"]:checked').value,
    });
    $('#btnRunParser2').disabled = true;
    $('#btnCancel').disabled = false;
  });
});

// Cancel Button
$('#btnCancel').addEventListener('click', () => {
  socket.emit('cancel_parser', {
    session_id: currentSessionId,
  });
  $('#btnRunParser2').disabled = false;
  $('#btnCancel').disabled = true;
});

// ============================================
// Event Listeners: Filters
// ============================================

$('#searchResults').addEventListener('input', (e) => {
  state.filters.search = e.target.value;
  renderResults();
});

$('#filterConfidence').addEventListener('input', (e) => {
  state.filters.confidence = parseInt(e.target.value);
  $('#filterConfidenceValue').textContent = e.target.value + '%+';
  renderResults();
});

$('#filterState').addEventListener('change', (e) => {
  state.filters.state = e.target.value;
  renderResults();
});

$('#filterLevel').addEventListener('change', (e) => {
  state.filters.level = e.target.value;
  renderLogs();
});

// ============================================
// Event Listeners: Log Drawer
// ============================================

$('#drawerHandle').addEventListener('click', () => {
  $('#logDrawer').classList.toggle('minimized');
  $('#logDrawer').classList.toggle('expanded');
});

$('#btnClearLogs').addEventListener('click', () => {
  state.logs = [];
  renderLogs();
  updateLogCounts();
  showToast('Logs cleared', 'info');
});

$('#btnExportLogs').addEventListener('click', () => {
  const csv = state.logs.map(l => 
    `${new Date(l.timestamp).toISOString()},${l.level},${l.type},"${l.message.replace(/"/g, '""')}"`
  ).join('\n');
  
  const blob = new Blob(
    ['timestamp,level,type,message\n' + csv],
    { type: 'text/csv' }
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `parser_logs_${Date.now()}.csv`;
  a.click();
  showToast('Logs exported', 'success');
});

$('#btnToggleScroll').addEventListener('click', () => {
  state.autoScroll = !state.autoScroll;
  $('#btnToggleScroll').textContent = state.autoScroll ? 'Pin' : 'Unpin';
  showToast(`Auto-scroll ${state.autoScroll ? 'enabled' : 'disabled'}`, 'info');
});

// ============================================
// Event Listeners: Modal
// ============================================

$('#btnClosePreview').addEventListener('click', () => {
  $('#previewModal').style.display = 'none';
});

$('#btnClosePreviewAlt').addEventListener('click', () => {
  $('#previewModal').style.display = 'none';
});

// Tab switching in modal
$$('.tab-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    // Remove active from all
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-content').forEach(c => c.classList.remove('active'));
    
    // Add active to clicked
    e.target.classList.add('active');
    const tabName = e.target.getAttribute('data-tab');
    $(`#tab${tabName.charAt(0).toUpperCase() + tabName.slice(1)}`).classList.add('active');
  });
});

$('#btnDownloadPreview').addEventListener('click', () => {
  if (state.currentFile) {
    downloadFile(state.currentFile.id);
  }
});

// ============================================
// Event Listeners: Results
// ============================================

function toggleSelectResult(resultId) {
  if (state.selectedResults.has(resultId)) {
    state.selectedResults.delete(resultId);
  } else {
    state.selectedResults.add(resultId);
  }
  
  // Update button state
  $('#btnBulkExport').disabled = state.selectedResults.size === 0;
}

$('#btnBulkExport').addEventListener('click', () => {
  if (state.selectedResults.size === 0) {
    showToast('No results selected', 'warning');
    return;
  }
  
  const selected = Array.from(state.selectedResults).map(id => 
    state.results.find(r => r.id === id)
  ).filter(Boolean);
  
  showToast(`Exporting ${selected.length} file(s)...`, 'info');
  
  // In production, fetch files and create ZIP
  setTimeout(() => {
    showToast(`Successfully exported ${selected.length} file(s)`, 'success');
  }, 1000);
});

$('#btnRefreshResults').addEventListener('click', () => {
  // In production, fetch updated results from API
  showToast('Results refreshed', 'success');
});

// ============================================
// File Operations (Stubs for Production)
// ============================================

function downloadFile(resultId) {
  const result = state.results.find(r => r.id === resultId);
  if (!result) return;
  
  showToast(`Downloading ${result.name}...`, 'info');
  
  // In production: fetch actual file
  setTimeout(() => {
    showToast(`${result.name} downloaded`, 'success');
  }, 1000);
}

// ============================================
// Command Palette
// ============================================

const commands = [
  { title: 'Run Parser', description: 'Start parsing', shortcut: 'Ctrl+Enter', action: () => $('#btnRunParser2').click() },
  { title: 'Cancel Parser', description: 'Stop parsing', shortcut: 'Ctrl+Shift+C', action: () => $('#btnCancel').click() },
  { title: 'Clear Logs', description: 'Clear debug console', shortcut: 'Ctrl+K', action: () => $('#btnClearLogs').click() },
  { title: 'Toggle Theme', description: 'Switch dark/light mode', shortcut: 'Ctrl+Shift+T', action: () => toggleTheme() },
  { title: 'Export Logs', description: 'Download debug logs', shortcut: 'Ctrl+Shift+E', action: () => $('#btnExportLogs').click() },
];

$('#btnCommandPalette').addEventListener('click', () => {
  $('#commandPalette').style.display = 'flex';
  $('#commandInput').focus();
});

$('#commandInput').addEventListener('input', (e) => {
  const query = e.target.value.toLowerCase();
  const results = commands.filter(c => 
    c.title.toLowerCase().includes(query) || 
    c.description.toLowerCase().includes(query)
  );
  
  $('#commandResults').innerHTML = results.map((cmd, idx) => `
    <div class="command-item" onclick="executeCommand(${idx})">
      <div class="command-text">
        <div class="command-title">${cmd.title}</div>
        <div class="command-description">${cmd.description}</div>
      </div>
      <div class="command-shortcut">${cmd.shortcut}</div>
    </div>
  `).join('');
});

function executeCommand(index) {
  commands[index].action();
  $('#commandPalette').style.display = 'none';
}

// Close command palette on ESC
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    $('#commandPalette').style.display = 'none';
    if ($('#previewModal').style.display !== 'none') {
      $('#previewModal').style.display = 'none';
    }
  }
  
  // Open command palette with Ctrl+Shift+P
  if (e.ctrlKey && e.shiftKey && e.key === 'P') {
    e.preventDefault();
    $('#btnCommandPalette').click();
  }
});

// ============================================
// Theme Management
// ============================================

function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', isDark ? 'light' : 'dark');
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
  showToast(`Theme switched to ${isDark ? 'light' : 'dark'} mode`, 'success');
}

// Load theme from localStorage
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);

// ============================================
// Populate Filter States
// ============================================

STATES.forEach(state => {
  const option = document.createElement('option');
  option.value = state;
  option.textContent = state;
  $('#filterState').appendChild(option);
});

// ============================================
// Data Loading: Real API + Fallback Sample Data
// ============================================

/**
 * Fetch results from warehouse API and transform to UI format.
 * Gracefully falls back to sample data if API unavailable.
 */
async function loadRealData() {
  try {
    console.log('[API] Fetching results from warehouse...');
    const response = await fetch('/api/warehouse_election_results?limit=50', {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      credentials: 'same-origin',
    });

    if (!response.ok) {
      throw new Error(`API returned ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    const items = Array.isArray(data.items) ? data.items : [];

    if (items.length === 0) {
      console.warn('[API] No results found in warehouse, using sample data');
      loadSampleData();
      return;
    }

    // Transform warehouse schema to UI results format
    state.results = items.map((item, idx) => ({
      id: String(item.id || idx + 1),
      name: item.contest || item.county || `Result #${idx + 1}`,
      type: (item.format || 'csv').toLowerCase(),
      rows: item.row_count || 0,
      columns: item.column_count || 0,
      confidence: item.confidence_score ? parseFloat(item.confidence_score) * 100 : 85.0,
      state: item.state || 'N/A',
      county: item.county || '',
      handler: item.handler_name || 'unknown',
      timestamp: item.created_at ? new Date(item.created_at).getTime() : Date.now(),
      source_url: item.source_url || '',
      preview: item.preview_html || item.preview_text || '(No preview available)',
    }));

    console.log(`[API] Loaded ${state.results.length} results from warehouse`);
    renderResults();
  } catch (error) {
    console.error('[API] Failed to load real data:', error);
    showToast(`Failed to load results: ${error.message}. Using sample data.`, 'warning');
    loadSampleData();
  }
}

/**
 * Fallback: Sample data for development & testing
 */
function loadSampleData() {
  console.log('[Sample Data] Loading development fixtures...');
  state.results = [
    {
      id: '1',
      name: 'Alameda County - General 2026',
      type: 'csv',
      rows: 1234,
      columns: 5,
      confidence: 94.5,
      state: 'CA',
      county: 'Alameda',
      handler: 'ca_handler',
      timestamp: Date.now() - 3600000,
      preview: 'Candidate | Votes\nJohn Doe | 45,234\nJane Smith | 41,123',
    },
    {
      id: '2',
      name: 'San Francisco County Results',
      type: 'json',
      rows: 987,
      columns: 4,
      confidence: 91.2,
      state: 'CA',
      county: 'San Francisco',
      handler: 'generic_json',
      timestamp: Date.now() - 7200000,
      preview: '{ "contest": "County Attorney", "candidates": [...] }',
    },
    {
      id: '3',
      name: 'Santa Clara County Export',
      type: 'xlsx',
      rows: 2156,
      columns: 6,
      confidence: 89.7,
      state: 'CA',
      county: 'Santa Clara',
      handler: 'xlsx_handler',
      timestamp: Date.now() - 10800000,
      preview: 'Sheet 1: Statewide Results\nSheet 2: County Breakdown\nSheet 3: Precincts',
    },
  ];
  
  console.log('[Sample Data] Loaded 3 fixture results');
  renderResults();
}

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  console.log('[Parser UI] Initializing modern interface...');
  
  // Load real data from warehouse API (with fallback to sample data)
  loadRealData();
  
  // Initialize state filters
  $$('#filterState option').forEach(opt => {
    if (!STATES.includes(opt.value)) {
      STATES.forEach(state => {
        if (!Array.from($$('#filterState option')).some(o => o.value === state)) {
          const option = document.createElement('option');
          option.value = state;
          option.textContent = state;
          $('#filterState').appendChild(option);
        }
      });
    }
  });
  
  // Request initial session ID
  socket.emit('join', {
    username: localStorage.getItem('username') || 'anonymous',
  });
  
  updateSessionsList();
  
  console.log('[Parser UI] Initialization complete');
});
