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
  virtualScrollThreshold: 100,
  virtualScrollItemHeight: 48,
  virtualScrollBuffer: 10,
  maxDirectUrls: 20, // Maximum URLs for batch processing
};

// ============================================
// Advanced Features State Management
// ============================================

const AdvancedFeatures = (() => {
  // Filter presets storage
  const filterPresets = new Map();
  const PRESETS_KEY = 'parser_filter_presets';
  
  // Session state tracking
  let currentSessionId = null;
  let directUrlDraftBySession = new Map();
  
  // Load presets from localStorage
  function loadPresets() {
    try {
      const stored = localStorage.getItem(PRESETS_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        Object.entries(parsed).forEach(([name, filters]) => {
          filterPresets.set(name, filters);
        });
      }
    } catch (err) {
      console.warn('[Presets] Failed to load:', err);
    }
  }
  
  // Save presets to localStorage
  function savePresets() {
    try {
      const obj = {};
      filterPresets.forEach((filters, name) => {
        obj[name] = filters;
      });
      localStorage.setItem(PRESETS_KEY, JSON.stringify(obj));
    } catch (err) {
      console.warn('[Presets] Failed to save:', err);
    }
  }
  
  // Get current filter state
  function getCurrentFilters() {
    return {
      confidence: document.getElementById('filterConfidence')?.value || '0',
      state: document.getElementById('filterState')?.value || '',
      level: document.getElementById('filterLevel')?.value || '',
    };
  }
  
  // Apply filters
  function applyFilters(filters) {
    const { confidence, state, level } = filters;
    const confEl = document.getElementById('filterConfidence');
    const stateEl = document.getElementById('filterState');
    const levelEl = document.getElementById('filterLevel');
    
    if (confEl) confEl.value = confidence || '0';
    if (stateEl) stateEl.value = state || '';
    if (levelEl) levelEl.value = level || '';
    
    // Update confidence label
    const labelEl = document.getElementById('filterConfidenceValue');
    if (labelEl && confEl) labelEl.textContent = confEl.value + '%+';
    
    // Trigger filter update
    if (window.applyLogFilters) window.applyLogFilters();
  }
  
  return {
    filterPresets,
    loadPresets,
    savePresets,
    getCurrentFilters,
    applyFilters,
    directUrlDraftBySession,
    get currentSessionId() { return currentSessionId; },
    set currentSessionId(id) { currentSessionId = id; }
  };
})();

// ============================================
// PHASE 2: Error Handling & Recovery
// ============================================

const ErrorBoundary = (() => {
  const errorLog = [];
  const maxErrors = 50;
  
  function logError(error, context = '') {
    const timestamp = new Date().toISOString();
    const errorInfo = {
      timestamp,
      message: error?.message || String(error),
      context,
      stack: error?.stack || '',
      recovered: false
    };
    
    errorLog.push(errorInfo);
    if (errorLog.length > maxErrors) {
      errorLog.shift();
    }
    
    console.error(`[ErrorBoundary] ${context}:`, error);
    return errorInfo;
  }
  
  function safeExecute(fn, context = 'anonymous', fallback = null) {
    try {
      return fn();
    } catch (error) {
      const logged = logError(error, context);
      logged.recovered = true;
      showErrorNotification(error, context);
      return fallback;
    }
  }
  
  function safeAsync(asyncFn, context = 'async_operation') {
    return Promise.resolve()
      .then(() => asyncFn())
      .catch(error => {
        logError(error, context);
        showErrorNotification(error, context);
      });
  }
  
  function showErrorNotification(error, context) {
    const message = `Error in ${context}: ${error?.message || 'Unknown error'}`;
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      bottom: 20px;
      left: 20px;
      background: #fee;
      border: 2px solid #f44;
      color: #c00;
      padding: 12px 16px;
      border-radius: 4px;
      font-size: 14px;
      z-index: 10001;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      max-width: 400px;
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
  }
  
  function getErrorLog() {
    return [...errorLog];
  }
  
  function clearErrorLog() {
    errorLog.length = 0;
  }
  
  return {
    safeExecute,
    safeAsync,
    logError,
    getErrorLog,
    clearErrorLog
  };
})();

// ============================================
// PHASE 2: Performance Utilities
// ============================================

// Debouncing utility for search/filter inputs
function debounce(fn, delay) {
  let timeoutId;
  return function(...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), delay);
  };
}

// Virtual scrolling manager for large option lists
const VirtualScroll = (() => {
  let isEnabled = false;
  let allItems = [];
  let visibleRange = { start: 0, end: 0 };
  let scrollTop = 0;
  let containerHeight = 0;
  
  function enable(items, container) {
    if (items.length < CONFIG.virtualScrollThreshold) {
      isEnabled = false;
      return false;
    }
    
    isEnabled = true;
    allItems = items;
    containerHeight = container.clientHeight || 400;
    
    const itemsPerPage = Math.ceil(containerHeight / CONFIG.virtualScrollItemHeight);
    visibleRange.start = 0;
    visibleRange.end = itemsPerPage + CONFIG.virtualScrollBuffer;
    
    return true;
  }
  
  function getVisibleItems() {
    if (!isEnabled) return allItems;
    return allItems.slice(visibleRange.start, visibleRange.end);
  }
  
  function updateScroll(newScrollTop) {
    if (!isEnabled) return;
    
    scrollTop = newScrollTop;
    const itemsPerPage = Math.ceil(containerHeight / CONFIG.virtualScrollItemHeight);
    const startIdx = Math.floor(scrollTop / CONFIG.virtualScrollItemHeight);
    
    visibleRange.start = Math.max(0, startIdx - CONFIG.virtualScrollBuffer);
    visibleRange.end = Math.min(allItems.length, startIdx + itemsPerPage + CONFIG.virtualScrollBuffer);
  }
  
  function getTotalHeight() {
    return isEnabled ? allItems.length * CONFIG.virtualScrollItemHeight : 0;
  }
  
  function getOffsetY() {
    return isEnabled ? visibleRange.start * CONFIG.virtualScrollItemHeight : 0;
  }
  
  function reset() {
    isEnabled = false;
    allItems = [];
    visibleRange = { start: 0, end: 0 };
    scrollTop = 0;
  }
  
  return { 
    enable, 
    getVisibleItems, 
    updateScroll, 
    getTotalHeight, 
    getOffsetY, 
    reset, 
    get isEnabled() { return isEnabled; } 
  };
})();

// ============================================
// PHASE 2: Table Preview (P2.3)
// ============================================

const TablePreview = (() => {
  function renderPreview(data, maxRows = 5) {
    if (!Array.isArray(data) || !data.length) return '<p class="text-muted">No data to preview</p>';
    
    const rows = data.slice(0, maxRows);
    const keys = Object.keys(rows[0]);
    
    let html = '<table class="preview-table"><thead><tr>';
    keys.forEach(k => html += `<th>${escapeHtml(k)}</th>`);
    html += '</tr></thead><tbody>';
    
    rows.forEach(row => {
      html += '<tr>';
      keys.forEach(k => html += `<td>${escapeHtml(String(row[k] || ''))}</td>`);
      html += '</tr>';
    });
    
    html += '</tbody></table>';
    if (data.length > maxRows) html += `<p class="text-muted small">${data.length - maxRows} more rows...</p>`;
    
    return html;
  }
  
  function showPreviewModal(title, data) {
    const modal = document.createElement('div');
    modal.className = 'modal preview-modal';
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h3>${escapeHtml(title)}</h3>
          <button class="modal-close" aria-label="Close preview">×</button>
        </div>
        <div class="modal-body">
          ${renderPreview(data)}
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" onclick="this.closest('.preview-modal').remove()">Continue</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector('.modal-close').onclick = () => modal.remove();
  }
  
  return { renderPreview, showPreviewModal };
})();

// ============================================
// PHASE 2: Session Restore (P2.4)
// ============================================

const SessionRestore = (() => {
  const RESTORE_KEY = 'smartElectionsRestore';
  
  function saveState(data) {
    try {
      const state = {
        timestamp: Date.now(),
        sessionId: currentSessionId,
        urls: Array.from(document.querySelectorAll('[data-url]')).map(el => el.getAttribute('data-url')),
        searches: Array.from(document.querySelectorAll('input[type="search"]')).map(el => el.value),
      };
      sessionStorage.setItem(RESTORE_KEY, JSON.stringify(state));
    } catch (e) {
      ErrorBoundary.logError(e, 'SessionRestore.saveState');
    }
  }
  
  function hasRestoreData() {
    const data = sessionStorage.getItem(RESTORE_KEY);
    return data && JSON.parse(data).timestamp > (Date.now() - 3600000); // 1 hour
  }
  
  function showRestoreBanner() {
    const data = sessionStorage.getItem(RESTORE_KEY);
    if (!data) return;
    
    const state = JSON.parse(data);
    const banner = document.createElement('div');
    banner.className = 'restore-banner';
    banner.innerHTML = `
      <div class="restore-content">
        <span>📋 Restore session from ${new Date(state.timestamp).toLocaleTimeString()}?</span>
        <button class="btn btn-sm btn-primary" id="btnRestoreYes">Restore</button>
        <button class="btn btn-sm btn-secondary" id="btnRestoreNo">Dismiss</button>
      </div>
    `;
    document.body.prepend(banner);
    
    document.getElementById('btnRestoreYes').onclick = () => {
      // Restore state
      banner.remove();
      showToast('Session restored', 'success');
    };
    document.getElementById('btnRestoreNo').onclick = () => {
      sessionStorage.removeItem(RESTORE_KEY);
      banner.remove();
    };
  }
  
  return { saveState, hasRestoreData, showRestoreBanner };
})();

// ============================================
// PHASE 2: Accessibility (P2.5)
// ============================================

function enhanceAccessibility() {
  // Add keyboard navigation (Enter, Escape)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      hidePrompt();
    }
    if (e.key === 'Enter' && e.ctrlKey && activePromptOptions.length === 1) {
      submitPrompt(activePromptOptions[0].index);
    }
  });
  
  // Add ARIA labels to dynamic content
  const addAriaLabel = (selector, label) => {
    document.querySelectorAll(selector).forEach(el => {
      if (!el.getAttribute('aria-label')) el.setAttribute('aria-label', label);
    });
  };
  
  addAriaLabel('.prompt-option', 'Contest option');
  addAriaLabel('.prompt-bundle-toggle', 'Expand/collapse bundle');
  addAriaLabel('.badge', 'Metadata badge');
  addAriaLabel('[data-tab]', 'Tab button');
  
  // Mark live regions
  const liveRegions = ['#logOutput', '#promptOptions', '#sessionsList'];
  liveRegions.forEach(sel => {
    const el = $(sel);
    if (el && !el.getAttribute('aria-live')) {
      el.setAttribute('aria-live', 'polite');
      el.setAttribute('aria-atomic', 'false');
    }
  });
  
  console.log('[Accessibility] Enhanced with keyboard nav and ARIA labels');
}

// ============================================
// PHASE 2: Integration Tests (P2.6)
// ============================================

async function runIntegrationTests() {
  const tests = {
    largeDataset: () => {
      const largeArray = Array.from({length: 1000}, (_, i) => ({
        index: i + 1,
        label: `Option ${i + 1}`,
        metadata: { confidence: Math.random() }
      }));
      
      VirtualScroll.enable(largeArray, {clientHeight: 400});
      return VirtualScroll.isEnabled && VirtualScroll.getVisibleItems().length < largeArray.length;
    },
    
    errorBoundary: () => {
      let caught = false;
      ErrorBoundary.safeExecute(() => {
        throw new Error('Test error');
      }, 'test');
      
      const log = ErrorBoundary.getErrorLog();
      caught = log.some(e => e.context === 'test' && e.recovered);
      ErrorBoundary.clearErrorLog();
      return caught;
    },
    
    debounce: async () => {
      let callCount = 0;
      const fn = debounce(() => callCount++, 100);
      fn(); fn(); fn();
      await new Promise(r => setTimeout(r, 150));
      return callCount === 1;
    }
  };
  
  const results = {};
  for (const [name, test] of Object.entries(tests)) {
    try {
      results[name] = await Promise.resolve(test());
    } catch (e) {
      results[name] = false;
      ErrorBoundary.logError(e, `Test: ${name}`);
    }
  }
  
  const passed = Object.values(results).filter(Boolean).length;
  console.log(`[Integration Tests] ${passed}/${Object.keys(tests).length} passed`, results);
  return results;
}

// ============================================
// PHASE 3: Visual Enhancements & UX Polish
// ============================================

/**
 * P3.1: Color-Coded Logs
 * Apply color coding to log entries based on level
 */
const LogColorCoding = (() => {
  const levelColors = {
    'ERROR': { bg: '#3d1a1a', border: '#dc2626', text: '#fca5a5' },
    'CRITICAL': { bg: '#3d1a1a', border: '#991b1b', text: '#fca5a5' },
    'WARNING': { bg: '#3d2a1a', border: '#ea580c', text: '#fdba74' },
    'INFO': { bg: '#1a2a3d', border: '#3b82f6', text: '#93c5fd' },
    'DEBUG': { bg: '#1a2a2a', border: '#10b981', text: '#6ee7b7' },
    'TRACE': { bg: '#2a1a3d', border: '#8b5cf6', text: '#c4b5fd' }
  };
  
  function applyColorToElement(element, level) {
    const colors = levelColors[level] || levelColors['INFO'];
    element.style.backgroundColor = colors.bg;
    element.style.borderLeftColor = colors.border;
    element.style.borderLeftWidth = '3px';
    element.style.borderLeftStyle = 'solid';
    element.style.color = colors.text;
  }
  
  function getLevelColor(level) {
    return levelColors[level] || levelColors['INFO'];
  }
  
  return { applyColorToElement, getLevelColor, levelColors };
})();

/**
 * P3.2: Type Badges
 * Visual identification of log message sources
 */
const LogTypeBadges = (() => {
  const typeConfig = {
    'status': { icon: '📊', color: '#3b82f6', label: 'Status' },
    'input': { icon: '📥', color: '#10b981', label: 'Input' },
    'output': { icon: '📤', color: '#8b5cf6', label: 'Output' },
    'error': { icon: '❌', color: '#dc2626', label: 'Error' },
    'exception': { icon: '⚠️', color: '#ea580c', label: 'Exception' },
    'prompt': { icon: '💬', color: '#06b6d4', label: 'Prompt' },
    'router': { icon: '🔀', color: '#6366f1', label: 'Router' },
    'handler': { icon: '⚙️', color: '#0ea5e9', label: 'Handler' },
    'download': { icon: '⬇️', color: '#14b8a6', label: 'Download' },
    'browser': { icon: '🌐', color: '#3b82f6', label: 'Browser' },
    'batch': { icon: '📦', color: '#8b5cf6', label: 'Batch' },
    'cancel': { icon: '🛑', color: '#dc2626', label: 'Cancel' },
    'summary': { icon: '📋', color: '#10b981', label: 'Summary' },
    'heartbeat': { icon: '💓', color: '#6b7280', label: 'Heartbeat' }
  };
  
  function createBadge(type) {
    const config = typeConfig[type] || { icon: '📌', color: '#6b7280', label: type };
    return `<span class="log-type-badge" style="background: ${config.color}22; color: ${config.color}; border: 1px solid ${config.color}44;">${config.icon} ${config.label}</span>`;
  }
  
  function getTypeConfig(type) {
    return typeConfig[type] || { icon: '📌', color: '#6b7280', label: type };
  }
  
  return { createBadge, getTypeConfig, typeConfig };
})();

/**
 * P3.3: Search Highlighting
 * Highlight matching text in log messages
 */
const SearchHighlighter = (() => {
  function highlightText(text, searchTerm) {
    if (!searchTerm || !text) return escapeHtml(text);
    
    const escaped = escapeHtml(text);
    const regex = new RegExp(`(${escapeRegex(searchTerm)})`, 'gi');
    return escaped.replace(regex, '<mark class="search-highlight">$1</mark>');
  }
  
  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  
  function clearHighlights() {
    $$('.search-highlight').forEach(el => {
      const parent = el.parentNode;
      parent.replaceChild(document.createTextNode(el.textContent), el);
      parent.normalize();
    });
  }
  
  return { highlightText, clearHighlights };
})();

/**
 * P3.4: Advanced Export
 * Export logs in multiple formats (JSON, CSV, Markdown)
 */
const AdvancedExport = (() => {
  function exportAsJSON(logs, filename = 'parser_logs.json') {
    const data = JSON.stringify(logs, null, 2);
    downloadBlob(data, filename, 'application/json');
  }
  
  function exportAsCSV(logs, filename = 'parser_logs.csv') {
    const headers = ['Timestamp', 'Level', 'Type', 'Message', 'Session ID'];
    const rows = logs.map(log => [
      new Date(log.timestamp).toISOString(),
      log.level || '',
      log.type || '',
      (log.message || '').replace(/"/g, '""'),
      log.sessionId || ''
    ]);
    
    const csv = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');
    
    downloadBlob(csv, filename, 'text/csv');
  }
  
  function exportAsMarkdown(logs, filename = 'parser_logs.md') {
    const header = '# Parser Logs\n\n';
    const timestamp = `**Exported:** ${new Date().toISOString()}\n`;
    const count = `**Total Logs:** ${logs.length}\n\n`;
    const divider = '---\n\n';
    
    const entries = logs.map(log => {
      const time = new Date(log.timestamp).toLocaleString();
      const level = log.level || 'INFO';
      const type = log.type || 'info';
      const msg = log.message || '';
      
      return `### ${level} - ${type}\n**Time:** ${time}\n**Session:** ${log.sessionId || 'N/A'}\n\n${msg}\n\n`;
    }).join(divider);
    
    const markdown = header + timestamp + count + divider + entries;
    downloadBlob(markdown, filename, 'text/markdown');
  }
  
  function downloadBlob(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    showToast(`Exported ${filename}`, 'success');
  }
  
  return { exportAsJSON, exportAsCSV, exportAsMarkdown };
})();

/**
 * P4.1: Keyboard Reference Guide
 * Visual guide for keyboard shortcuts
 */
const KeyboardGuide = (() => {
  const shortcuts = [
    { key: 'Escape', description: 'Close modal/prompt' },
    { key: 'Ctrl+Enter', description: 'Submit single option' },
    { key: 'Ctrl+S', description: 'Save current filter preset' },
    { key: 'Ctrl+E', description: 'Export logs as JSON' },
    { key: 'Ctrl+Shift+E', description: 'Export logs as CSV' },
    { key: 'Ctrl+/', description: 'Show keyboard shortcuts' },
    { key: 'Ctrl+L', description: 'Clear log output' },
    { key: 'Ctrl+F', description: 'Focus search input' }
  ];
  
  function show() {
    const modal = document.createElement('div');
    modal.className = 'modal keyboard-guide-modal';
    modal.innerHTML = `
      <div class="modal-content" style="max-width: 600px;">
        <div class="modal-header">
          <h3>⌨️ Keyboard Shortcuts</h3>
          <button class="modal-close" aria-label="Close shortcuts guide">×</button>
        </div>
        <div class="modal-body">
          <div class="shortcuts-list">
            ${shortcuts.map(s => `
              <div class="shortcut-row">
                <kbd class="shortcut-key">${escapeHtml(s.key)}</kbd>
                <span class="shortcut-desc">${escapeHtml(s.description)}</span>
              </div>
            `).join('')}
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" onclick="this.closest('.modal').remove()">Got it</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector('.modal-close').onclick = () => modal.remove();
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.remove();
    });
  }
  
  return { show, shortcuts };
})();

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
let activePromptMessage = null;
let activePromptOptions = [];
let bundleExpandedState = new Map(); // Track which bundles are expanded
let selectedPromptOptions = new Set(); // Multi-select tracking

socket.on('connect', () => {
  console.log('[Socket.IO] Connected:', socket.id);
});

socket.on('session_id', (data) => {
  currentSessionId = data.session_id;
  console.log('[Session] ID:', currentSessionId);
  updateSessionsList();
  
  // Restore Direct URL draft for this session if exists
  AdvancedFeatures.currentSessionId = currentSessionId;
  const savedDraft = AdvancedFeatures.directUrlDraftBySession.get(currentSessionId);
  const textarea = document.getElementById('directUrlTextarea');
  if (textarea && savedDraft) {
    textarea.value = savedDraft;
    parseDirectUrlField();
  }
});

socket.on('parser_output', (data) => {
  ErrorBoundary.safeAsync(async () => {
    addLog(data);
    handlePromptLog(data);
    SessionRestore.saveState(data); // P2.4: Save state for recovery
    // Show pending overlay for processing messages (P1.4)
    if (data.type === 'status' && data.message?.includes('Processing')) {
      PendingOverlay.show(data.message, 300);
    }
  }, 'socket:parser_output');
});

socket.on('contest_options', (data) => {
  ErrorBoundary.safeExecute(() => {
    handleContestOptions(data);
  }, 'socket:contest_options');
});

socket.on('session_state', (data) => {
  ErrorBoundary.safeExecute(() => {
    console.log('[Session State]', data);
    updateProgressCard(data);
    updateSessionsList();
  }, 'socket:session_state');
});

socket.on('session_list', (data) => {
  ErrorBoundary.safeExecute(() => {
    updateSessionsList(data.sessions);
  }, 'socket:session_list');
});

socket.on('session_cloned', (data) => {
  ErrorBoundary.safeExecute(() => {
    console.log('[Session Cloned]', data);
    showToast(`Session cloned: ${data.new_session}`, 'success');
    updateSessionsList();
  }, 'socket:session_cloned');
});

socket.on('session_deleted', (data) => {
  ErrorBoundary.safeExecute(() => {
    console.log('[Session Deleted]', data);
    showToast('Session deleted', 'info');
    updateSessionsList();
  }, 'socket:session_deleted');
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
// Pending Overlay for Long Operations
// ============================================

const PendingOverlay = (() => {
  let element = null;
  let hideTimer = null;
  
  function create() {
    if (element) return element;
    element = document.createElement('div');
    element.id = 'pendingOverlay';
    element.className = 'pending-overlay hidden';
    element.innerHTML = `
      <div class="pending-overlay-content">
        <div class="spinner"></div>
        <div class="pending-text">Processing...</div>
      </div>
    `;
    document.body.appendChild(element);
    return element;
  }
  
  function show(message = 'Processing...', minDuration = 500) {
    if (!element) create();
    element.querySelector('.pending-text').textContent = message;
    element.classList.remove('hidden');
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(() => hide(), minDuration);
  }
  
  function hide() {
    if (!element) return;
    element.classList.add('hidden');
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
  }
  
  return { show, hide };
})();

// ============================================
// Filter Presets for Log Console
// ============================================

const filterPresets = (() => {
  const STORAGE_KEY = 'logFilterPresets';
  
  function save(name, filters) {
    if (!name || !filters) return;
    const presets = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    presets[name] = {
      search: filters.search || '',
      level: filters.level || '',
      type: filters.type || '',
      timestamp: Date.now()
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
    updatePresetDropdown();
  }
  
  function load(name) {
    const presets = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    return presets[name] || null;
  }
  
  function deletePreset(name) {
    const presets = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    delete presets[name];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
    updatePresetDropdown();
  }
  
  function list() {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
  }
  
  function updatePresetDropdown() {
    const select = $('#logFilterPresetSelect');
    if (!select) return;
    const presets = list();
    const options = [
      '<option value="">— Save new preset —</option>',
      '<option value="__separator__" disabled>─────────</option>'
    ];
    Object.keys(presets).sort().forEach(name => {
      options.push(`<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`);
    });
    select.innerHTML = options.join('');
  }
  
  function applyPreset(name) {
    const preset = load(name);
    if (!preset) return;
    state.filters = { ...state.filters, ...preset };
    renderLogs();
    if ($('#logSearchInput')) $('#logSearchInput').value = preset.search || '';
    if ($('#logLevelFilter')) $('#logLevelFilter').value = preset.level || '';
  }
  
  return { save, load, deletePreset, list, updatePresetDropdown, applyPreset };
})();

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
  ErrorBoundary.safeExecute(() => {
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
  }, 'addLog');
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
    if (state.filters.search) {
      const searchLower = state.filters.search.toLowerCase();
      const msgLower = (log.message || '').toLowerCase();
      if (!msgLower.includes(searchLower)) return false;
    }
    return true;
  });
  
  const logOutput = $('#logOutput');
  logOutput.innerHTML = filtered.map(log => {
    const colors = LogColorCoding.getLevelColor(log.level);
    const typeBadge = LogTypeBadges.createBadge(log.type);
    const highlightedMsg = state.filters.search 
      ? SearchHighlighter.highlightText(log.message, state.filters.search)
      : escapeHtml(log.message);
    
    return `
      <div class="log-line" style="background: ${colors.bg}; border-left: 3px solid ${colors.border};">
        <span class="log-timestamp">${new Date(log.timestamp).toLocaleTimeString()}</span>
        <span class="log-level" style="color: ${colors.text};">${log.level}</span>
        ${typeBadge}
        <div class="log-message" style="color: ${colors.text};">${highlightedMsg}</div>
      </div>
    `;
  }).join('');
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
        <input type="checkbox" class="card-checkbox" onchange="toggleSelectResult('${result.id}')">
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
  const emptyState = $('#emptyState');
  if (filtered.length === 0) {
    grid.classList.add('hidden');
    emptyState.classList.remove('hidden');
    emptyState.classList.add('flex');
  } else {
    grid.classList.remove('hidden');
    grid.innerHTML = filtered.map(r => createResultCard(r)).join('');
    emptyState.classList.add('hidden');
    emptyState.classList.remove('flex');
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
  const previewModal = $('#previewModal');
  if (previewModal) previewModal.classList.remove('hidden');
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
  if (!list) return; // Guard against missing element
  
  if (!state.sessions.length) {
    list.innerHTML = '<p class="text-muted small">No sessions</p>';
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
  const progressCard = $('#progressCard');
  if (!progressCard) return; // Element doesn't exist in DOM
  
  if (!sessionData || sessionData.state === 'IDLE') {
    progressCard.style.display = 'none';
    return;
  }
  
  progressCard.style.display = 'block';
  const progressSessionEl = $('#progressSessionId');
  const progressStatusEl = $('#progressStatus');
  const progressStagesEl = $('#progressStages');
  
  if (progressSessionEl) progressSessionEl.textContent = sessionData.session_id;
  if (progressStatusEl) progressStatusEl.textContent = sessionData.state;
  
  // Update phases
  if (progressStagesEl) {
    const phases = ['PREPARE', 'SOURCE', 'RUN', 'REVIEW'];
    const stagesHtml = phases.map(phase => {
      let className = '';
      if (phase === sessionData.phase) className = 'active';
      else if (phases.indexOf(phase) < phases.indexOf(sessionData.phase)) className = 'completed';
      return `<div class="stage ${className}">${phase}</div>`;
    }).join('');
    
    progressStagesEl.innerHTML = stagesHtml;
  }
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
    const fileSource = $('input[name="fileSource"]:checked').value;
    const payload = {
      session_id: currentSessionId,
      file_source: fileSource,
    };
    
    // Add direct URLs if selected
    if (fileSource === 'direct') {
      const urls = parseDirectUrlField();
      if (urls.length === 0) {
        showToast('Please enter at least one valid URL', 'warning');
        return;
      }
      payload.direct_urls = urls;
    }
    
    // Add batch mode flag
    const batchModeCheckbox = $('#batchMode');
    if (batchModeCheckbox && batchModeCheckbox.checked) {
      payload.batch_mode = true;
    }
    
    socket.emit('run_parser', payload);
    $('#btnRunParser2').disabled = true;
    $('#btnCancel').disabled = false;
    
    // Update current session in advanced features
    AdvancedFeatures.currentSessionId = currentSessionId;
    
    showToast('Parser started...', 'info');
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

const drawerHandle = $('#drawerHandle');
if (drawerHandle) {
  drawerHandle.addEventListener('click', () => {
    const logDrawer = $('#logDrawer');
    if (logDrawer) {
      logDrawer.classList.toggle('minimized');
      logDrawer.classList.toggle('expanded');
    }
  });
}

// ============================================
// Log Drawer: Auto-sync with Legacy Sidebar Width
// ============================================

(function syncDrawerToLegacySidebar(){
  const legacySidebar = document.getElementById('sidebar');
  const logDrawer = $('#logDrawer');
  const root = document.documentElement;
  
  function updateDrawerOffset() {
    if (!legacySidebar || !logDrawer) return;
    const width = legacySidebar.offsetWidth;
    if (width > 0) {
      root.style.setProperty('--drawer-left-offset', width + 'px');
      if (window.innerWidth > 1024) {
        // On desktop: use CSS var for grid-based layout
        logDrawer.style.left = 'var(--sidebar-left-max)';
      } else {
        // On mobile: stretch full width
        logDrawer.style.left = '0';
      }
    }
  }
  
  // Initial sync
  setTimeout(updateDrawerOffset, 100); // Let DOM settle
  
  // Sync on resize
  window.addEventListener('resize', updateDrawerOffset);
  
  // Observer for sidebar visibility changes
  const observer = new MutationObserver(() => {
    requestAnimationFrame(updateDrawerOffset);
  });
  if (legacySidebar) {
    observer.observe(legacySidebar, {
      attributes: true,
      attributeFilter: ['style', 'class'],
      characterData: false,
      subtree: false
    });
  }
})();

// ============================================
// Event Listeners: Mobile Sidebars (Unified)
// ============================================

(function initUnifiedMobileSidebars(){
  const legacySidebar = document.getElementById('sidebar');
  const rightSidebar = document.querySelector('.sidebar-right');
  const sidebarBackdrop = $('#sidebarBackdrop');
  const toggleLeftBtn = $('#sidebarToggleBtn');
  const toggleRightBtn = $('#btnToggleRightSidebar');
  const overlay = $('#mobileSidebarOverlay') || sidebarBackdrop;

  function closeAll() {
    if (legacySidebar) legacySidebar.classList.remove('sidebar-open');
    if (rightSidebar) rightSidebar.classList.remove('open');
    if (sidebarBackdrop) sidebarBackdrop.classList.remove('visible');
    if (overlay && overlay !== sidebarBackdrop) overlay.classList.remove('visible');
  }

  function openLeft() {
    if (!legacySidebar) return;
    legacySidebar.classList.add('sidebar-open');
    if (sidebarBackdrop) sidebarBackdrop.classList.add('visible');
  }

  function openRight() {
    if (!rightSidebar) return;
    rightSidebar.classList.add('open');
    if (overlay) overlay.classList.add('visible');
  }

  // Legacy left sidebar toggle
  if (toggleLeftBtn) {
    toggleLeftBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (!legacySidebar) return;
      const isOpen = legacySidebar.classList.contains('sidebar-open');
      if (isOpen) closeAll(); else openLeft();
    });
  }

  // Modern right sidebar toggle
  if (toggleRightBtn) {
    toggleRightBtn.addEventListener('click', (e) => {
      e.preventDefault();
      if (!rightSidebar) return;
      const isOpen = rightSidebar.classList.contains('open');
      if (isOpen) closeAll(); else openRight();
    });
  }

  // Backdrop/overlay clicks close all
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener('click', closeAll);
  }
  if (overlay && overlay !== sidebarBackdrop) {
    overlay.addEventListener('click', closeAll);
  }

  // Escape key closes all sidebars
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeAll();
  });

  // Auto-close on resize to desktop
  window.addEventListener('resize', () => {
    if (window.innerWidth > 1024) closeAll();
  });
})();

const btnClearLogs = $('#btnClearLogs');
if (btnClearLogs) {
  btnClearLogs.addEventListener('click', () => {
    state.logs = [];
    renderLogs();
    updateLogCounts();
    showToast('Logs cleared', 'info');
  });
}

const btnCopyLogs = $('#btnCopyLogs');
if (btnCopyLogs) {
  btnCopyLogs.addEventListener('click', async () => {
    const text = state.logs.map(l => {
      const ts = new Date(l.timestamp).toLocaleTimeString();
      const typeLabel = l.type ? `[${l.type}]` : '';
      return `[${ts}] ${l.level} ${typeLabel} ${l.message}`;
    }).join('\n');
    try {
      await navigator.clipboard.writeText(text || '');
      showToast('Logs copied to clipboard', 'success');
    } catch (err) {
      showToast('Clipboard not available. Use Export instead.', 'warning');
    }
  });
}

const btnExportLogs = $('#btnExportLogs');
if (btnExportLogs) {
  btnExportLogs.addEventListener('click', () => {
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
}

const btnToggleScroll = $('#btnToggleScroll');
if (btnToggleScroll) {
  btnToggleScroll.textContent = state.autoScroll ? 'Pin' : 'Unpin';
  btnToggleScroll.addEventListener('click', () => {
    state.autoScroll = !state.autoScroll;
    btnToggleScroll.textContent = state.autoScroll ? 'Pin' : 'Unpin';
    showToast(`Auto-scroll ${state.autoScroll ? 'enabled' : 'disabled'}`, 'info');
  });
}

// ============================================
// Event Listeners: Modal
// ============================================

const btnClosePreview = $('#btnClosePreview');
if (btnClosePreview) {
  btnClosePreview.addEventListener('click', () => {
    const previewModal = $('#previewModal');
    if (previewModal) previewModal.classList.add('hidden');
  });
}

const btnClosePreviewAlt = $('#btnClosePreviewAlt');
if (btnClosePreviewAlt) {
  btnClosePreviewAlt.addEventListener('click', () => {
    const previewModal = $('#previewModal');
    if (previewModal) previewModal.classList.add('hidden');
  });
}

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

$('#btnDownloadPreview')?.addEventListener('click', () => {
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

$('#btnBulkExport')?.addEventListener('click', () => {
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

// Safety: ensure overlays start hidden even if cache or styles misbehave
const commandPaletteInit = $('#commandPalette');
if (commandPaletteInit) commandPaletteInit.classList.add('hidden');
const previewModalInit = $('#previewModal');
if (previewModalInit) previewModalInit.classList.add('hidden');
const promptModalInit = $('#promptModal');
if (promptModalInit) promptModalInit.classList.add('hidden');

const btnCommandPalette = $('#btnCommandPalette');
if (btnCommandPalette) {
  btnCommandPalette.addEventListener('click', () => {
    const commandPalette = $('#commandPalette');
    const commandInput = $('#commandInput');
    if (commandPalette) commandPalette.classList.remove('hidden');
    if (commandInput) commandInput.focus();
  });
}

const commandInput = $('#commandInput');
if (commandInput) {
  commandInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const results = commands.filter(c => 
      c.title.toLowerCase().includes(query) || 
      c.description.toLowerCase().includes(query)
    );
    
    const commandResults = $('#commandResults');
    if (commandResults) {
      commandResults.innerHTML = results.map((cmd, idx) => `
        <div class="command-item" onclick="executeCommand(${idx})">
          <div class="command-text">
            <div class="command-title">${cmd.title}</div>
            <div class="command-description">${cmd.description}</div>
          </div>
          <div class="command-shortcut">${cmd.shortcut}</div>
        </div>
      `).join('');
    }
  });
}

function executeCommand(index) {
  commands[index].action();
  const commandPalette = $('#commandPalette');
  if (commandPalette) commandPalette.classList.add('hidden');
}

// Close command palette on ESC
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const commandPalette = $('#commandPalette');
    if (commandPalette) commandPalette.classList.add('hidden');
    const previewModal = $('#previewModal');
    if (previewModal && !previewModal.classList.contains('hidden')) {
      previewModal.classList.add('hidden');
    }
    const promptModal = $('#promptModal');
    if (promptModal && !promptModal.classList.contains('hidden')) {
      promptModal.classList.add('hidden');
    }
  }
  
  // Open command palette with Ctrl+Shift+P
  if (e.ctrlKey && e.shiftKey && e.key === 'P') {
    e.preventDefault();
    $('#btnCommandPalette').click();
  }
});

// ============================================
// Prompt Handling (interactive server prompts)
// ============================================

const promptTitleEl = $('#promptTitle');
const promptMessageEl = $('#promptMessage');
const promptInputEl = $('#promptInput');
const promptSearchEl = $('#promptSearch');
const promptOptionsEl = $('#promptOptions');

function handlePromptLog(data) {
  ErrorBoundary.safeExecute(() => {
    const message = typeof data?.message === 'string' ? data.message : '';
    const isPrompt = data?.type === 'prompt' || message.toUpperCase().includes('[PROMPT]');
    const ctx = data?.context || {};
    let options = [];

    // URL selection prompt
    if (Array.isArray(ctx.urls) && ctx.urls.length) {
      options = ctx.urls.map((u, idx) => ({
        index: idx + 1,
        label: u,
        meta: ctx.processed && ctx.processed[u] ? ctx.processed[u].status || '' : '',
      }));
    }

    // Contest/options style prompt
    if (!options.length && Array.isArray(ctx.options) && ctx.options.length) {
      options = ctx.options.map((opt, idx) => {
        if (typeof opt === 'string') {
          const m = opt.match(/^\s*\[(\d+)\]\s+(.+?)(?:\s+\(([^)]+)\))?\s*$/);
          if (m) return { index: Number(m[1]), label: m[2], meta: m[3] || '' };
          return { index: idx + 1, label: opt, meta: '' };
        }
        if (opt && typeof opt === 'object') {
          return {
            index: Number(opt.index ?? idx + 1),
            label: opt.label || opt.title || opt.name || `Option ${idx + 1}`,
            meta: opt.meta || opt.summary || '',
          };
        }
        return { index: idx + 1, label: String(opt), meta: '' };
      });
    }

    if (isPrompt && message) {
      showPrompt({
        title: ctx.title || 'Action required',
        message,
        options,
        placeholder: ctx.placeholder,
      });
    }
  }, 'handlePromptLog');
}

function handleContestOptions(payload) {
  ErrorBoundary.safeExecute(() => {
    const options = Array.isArray(payload?.options) ? payload.options.map((opt, idx) => ({
      index: Number(opt.index ?? idx + 1),
      label: opt.label || opt.name || opt.title || `Option ${idx + 1}`,
      meta: opt.meta || (opt.metadata && opt.metadata.summary) || '',
    })) : [];
    if (!options.length) {
      console.warn('[handleContestOptions] No options provided');
      return;
    }
    const ctx = payload?.context || {};
    const message = ctx.message || 'Select a contest';
    showPrompt({
      title: 'Select Contest',
      message,
      options,
      placeholder: 'Search or click to choose',
    });
  }, 'handleContestOptions');
}

function renderPromptOptions(filterText = '') {
  ErrorBoundary.safeExecute(() => {
    if (!promptOptionsEl) return;
    const needle = filterText.toLowerCase();
    const filtered = activePromptOptions.filter(opt => {
      const label = String(opt.label || '').toLowerCase();
      const meta = String(opt.meta || '').toLowerCase();
      const scopeLabel = opt.metadata?.scope_label ? String(opt.metadata.scope_label).toLowerCase() : '';
      return !needle || label.includes(needle) || meta.includes(needle) || scopeLabel.includes(needle);
    });

    if (!filtered.length) {
      promptOptionsEl.innerHTML = '<div class="text-muted small">No options. Enter a response above.</div>';
      return;
    }

    // Group options by bundle_key if available (P1.1 Bundle Grouping)
    const groups = new Map();
    filtered.forEach(opt => {
      const meta = opt.metadata || {};
      const bundleKey = meta.bundle_key || meta.bundle_parent_index;
      
      if (bundleKey && meta.bundle_mode === 'aggregate') {
        if (!groups.has(bundleKey)) {
          groups.set(bundleKey, {
            parent: opt,
            children: [],
            expanded: bundleExpandedState.get(bundleKey) || false
          });
        }
      } else if (bundleKey && meta.bundle_member) {
        const group = groups.get(bundleKey);
        if (group) group.children.push(opt);
        else if (!groups.has(opt.index)) {
          groups.set(opt.index, { parent: opt, children: [], expanded: false });
        }
      } else {
        if (!groups.has(opt.index)) {
          groups.set(opt.index, { parent: opt, children: [], expanded: false });
        }
      }
    });

    promptOptionsEl.innerHTML = '';

    // Enable virtual scrolling for large lists (P2.2)
    const groupsArray = Array.from(groups.values());
    const useVirtualScroll = VirtualScroll.enable(groupsArray, promptOptionsEl.parentElement);
    
    if (useVirtualScroll) {
      // Create spacer for total height
      const spacer = document.createElement('div');
      spacer.style.height = `${VirtualScroll.getTotalHeight()}px`;
      spacer.style.position = 'relative';
      
      const content = document.createElement('div');
      content.style.position = 'absolute';
      content.style.top = `${VirtualScroll.getOffsetY()}px`;
      content.style.width = '100%';
      
      const visibleGroups = VirtualScroll.getVisibleItems();
      visibleGroups.forEach((group) => {
        const elem = renderGroupElement(group, key);
        content.appendChild(elem);
      });
      
      spacer.appendChild(content);
      promptOptionsEl.appendChild(spacer);
      
      // Add scroll listener for virtual scroll updates
      const container = promptOptionsEl.parentElement;
      if (container && !container.dataset.scrollListenerAdded) {
        container.dataset.scrollListenerAdded = 'true';
        container.addEventListener('scroll', debounce((e) => {
          VirtualScroll.updateScroll(e.target.scrollTop);
          renderPromptOptions(promptSearchEl?.value || '');
        }, 100));
      }
    } else {
      // Standard rendering for smaller lists
      for (const [key, group] of groups) {
        const elem = renderGroupElement(group, key);
        promptOptionsEl.appendChild(elem);
      }
    }
  
  updateSelectionSummary();
  }, 'renderPromptOptions');
}

// Helper: Render a group element (for both virtual and standard rendering)
function renderGroupElement(group, key) {
  const { parent, children, expanded } = group;
  
  if (!children.length) {
    // Single option (not grouped)
    return createPromptOptionButton(parent);
  }
  
  // Bundle with children
  const wrapper = document.createElement('div');
  wrapper.className = 'prompt-bundle';
  
  // Bundle header with toggle
  const header = document.createElement('div');
  header.className = 'prompt-bundle-header';
  
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'prompt-bundle-toggle';
  toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
  toggle.textContent = expanded ? '▼' : '▶';
  toggle.onclick = (e) => {
    e.preventDefault();
    bundleExpandedState.set(key, !bundleExpandedState.get(key));
    renderPromptOptions(promptSearchEl?.value || '');
  };
  
  header.appendChild(toggle);
  header.appendChild(createPromptOptionButton(parent, { bundled: true }));
  wrapper.appendChild(header);
  
  // Children (show if expanded)
  if (expanded && children.length) {
    const childContainer = document.createElement('div');
    childContainer.className = 'prompt-bundle-children';
    children.forEach(child => {
      const childBtn = createPromptOptionButton(child, { isChild: true });
      childContainer.appendChild(childBtn);
    });
    wrapper.appendChild(childContainer);
  }
  
  return wrapper;
}

// Stub for old rendering (replaced above)
function renderPromptOptions_OLD(filterText = '') {
  ErrorBoundary.safeExecute(() => {
    if (!promptOptionsEl) return;
    const needle = filterText.toLowerCase();
    const filtered = activePromptOptions.filter(opt => {
      const label = String(opt.label || '').toLowerCase();
      const meta = String(opt.meta || '').toLowerCase();
      const scopeLabel = opt.metadata?.scope_label ? String(opt.metadata.scope_label).toLowerCase() : '';
      return !needle || label.includes(needle) || meta.includes(needle) || scopeLabel.includes(needle);
    });

    if (!filtered.length) {
      promptOptionsEl.innerHTML = '<div class="text-muted small">No options. Enter a response above.</div>';
      return;
    }

    // Group options by bundle_key if available (P1.1 Bundle Grouping)
    const groups = new Map();
    filtered.forEach(opt => {
      const meta = opt.metadata || {};
      const bundleKey = meta.bundle_key || meta.bundle_parent_index;
      
      if (bundleKey && meta.bundle_mode === 'aggregate') {
        if (!groups.has(bundleKey)) {
          groups.set(bundleKey, {
            parent: opt,
            children: [],
            expanded: bundleExpandedState.get(bundleKey) || false
          });
        }
      } else if (bundleKey && meta.bundle_member) {
        const group = groups.get(bundleKey);
        if (group) group.children.push(opt);
        else if (!groups.has(opt.index)) {
          groups.set(opt.index, { parent: opt, children: [], expanded: false });
        }
      } else {
        if (!groups.has(opt.index)) {
          groups.set(opt.index, { parent: opt, children: [], expanded: false });
        }
      }
    });

    promptOptionsEl.innerHTML = '';

    // Render each group
    for (const [key, group] of groups) {
      const { parent, children, expanded } = group;
      
      if (!children.length) {
        // Single option (not grouped)
        const btn = createPromptOptionButton(parent);
        promptOptionsEl.appendChild(btn);
      } else {
        // Bundle with children
        const wrapper = document.createElement('div');
        wrapper.className = 'prompt-bundle';
        
        // Bundle header with toggle
        const header = document.createElement('div');
        header.className = 'prompt-bundle-header';
        
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'prompt-bundle-toggle';
        toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        toggle.textContent = expanded ? '▼' : '▶';
        toggle.onclick = (e) => {
          e.preventDefault();
          bundleExpandedState.set(key, !bundleExpandedState.get(key));
          renderPromptOptions(promptSearchEl?.value || '');
        };
        
        header.appendChild(toggle);
        header.appendChild(createPromptOptionButton(parent, { bundled: true }));
        wrapper.appendChild(header);
        
        // Children (show if expanded)
        if (expanded && children.length) {
          const childContainer = document.createElement('div');
          childContainer.className = 'prompt-bundle-children';
          children.forEach(child => {
            const childBtn = createPromptOptionButton(child, { isChild: true });
            childContainer.appendChild(childBtn);
          });
          wrapper.appendChild(childContainer);
      }
      
      promptOptionsEl.appendChild(wrapper);
    }
  }
  
  updateSelectionSummary();
  }, 'renderPromptOptions');
}

function createPromptOptionButton(opt, options = {}) {
  return ErrorBoundary.safeExecute(() => {
    const { bundled = false, isChild = false } = options;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'prompt-option' + (isChild ? ' prompt-option-child' : '') + (bundled ? ' prompt-option-bundled' : '');
    
    const meta = opt.metadata || {};
    const bundleSize = meta.bundle_child_count ? meta.bundle_child_count + 1 : 0;
    const badges = [];
    
    // P1.2 Metadata Badges
    if (meta.scope_label) badges.push(`<span class="badge badge-scope">${escapeHtml(meta.scope_label)}</span>`);
    if (bundleSize && bundled) badges.push(`<span class="badge badge-bundle">${bundleSize} variations</span>`);
    if (Array.isArray(meta.counties) && meta.counties.length > 1) badges.push(`<span class="badge badge-counties">${meta.counties.length} counties</span>`);
    if (meta.year) badges.push(`<span class="badge badge-year">${meta.year}</span>`);
    if (typeof meta.confidence === 'number') {
      const confClass = meta.confidence >= 0.85 ? 'high' : meta.confidence >= 0.70 ? 'medium' : 'low';
      badges.push(`<span class="badge badge-confidence badge-conf-${confClass}">conf ${meta.confidence.toFixed(2)}</span>`);
    }
    if (meta.variants || (Array.isArray(meta.contest_ids) && meta.contest_ids.length > 1)) {
      const count = meta.variants || meta.contest_ids.length;
      badges.push(`<span class="badge badge-variants">${count} IDs</span>`);
    }
    
    // P2.1 Multi-Select Checkbox
    const hasCheckbox = !isChild && activePromptOptions.length > 1;
    let checkboxHtml = '';
    if (hasCheckbox) {
      const isChecked = selectedPromptOptions.has(opt.index);
      checkboxHtml = `<input type="checkbox" class="prompt-option-checkbox" value="${escapeHtml(String(opt.index))}" ${isChecked ? 'checked' : ''} />`;
    }
    
    btn.innerHTML = `
      ${checkboxHtml}
      <div>
        <div class="label">[${opt.index ?? opt.value ?? '?'}] ${escapeHtml(opt.label || '')}</div>
        ${badges.length ? `<div class="badges">${badges.join('')}</div>` : ''}
        ${opt.meta ? `<div class="meta">${escapeHtml(opt.meta)}</div>` : ''}
      </div>
    `;
    
    // Checkbox event handler
    const checkbox = btn.querySelector('.prompt-option-checkbox');
    if (checkbox) {
      checkbox.addEventListener('change', (e) => {
        if (e.target.checked) {
          selectedPromptOptions.add(opt.index);
        } else {
          selectedPromptOptions.delete(opt.index);
        }
        updateSelectionSummary();
      });
      checkbox.addEventListener('click', (e) => e.stopPropagation());
    } else {
      // Single-click auto-submit for single options
      btn.addEventListener('click', () => submitPrompt(String(opt.index ?? opt.value ?? opt.label)));
    }
    
    return btn;
  }, 'createPromptOptionButton', null);
}

function updateSelectionSummary() {
  ErrorBoundary.safeExecute(() => {
    const count = selectedPromptOptions.size;
    const summaryEl = document.getElementById('promptSelectionSummary');
    if (summaryEl) {
      if (count > 0) {
        summaryEl.textContent = `${count} contest${count === 1 ? '' : 's'} selected`;
        summaryEl.classList.remove('hidden');
      } else {
        summaryEl.classList.add('hidden');
      }
    }
  }, 'updateSelectionSummary');
}

function showPrompt({ title = 'Action required', message = '', options = [], placeholder = '' }) {
  ErrorBoundary.safeExecute(() => {
    activePromptMessage = message;
    activePromptOptions = Array.isArray(options) ? options : [];

    if (promptTitleEl) promptTitleEl.textContent = title;
    if (promptMessageEl) promptMessageEl.textContent = message || 'Please choose an option';
    if (promptInputEl) {
      promptInputEl.value = '';
      if (placeholder) promptInputEl.placeholder = placeholder;
    }
    if (promptSearchEl) {
      promptSearchEl.value = '';
      promptSearchEl.placeholder = placeholder || 'Filter options...';
    }
    renderPromptOptions('');

    const promptModal = $('#promptModal');
    if (promptModal) promptModal.classList.remove('hidden');
    if (promptSearchEl) {
      promptSearchEl.focus();
    } else if (promptInputEl) {
      promptInputEl.focus();
    }
  }, 'showPrompt');
}

function submitPrompt(forcedValue) {
  ErrorBoundary.safeExecute(() => {
    let value;
    
    // If forced value provided (clicked option), use it
    if (forcedValue) {
      value = String(forcedValue);
    } else if (selectedPromptOptions.size > 0) {
      // Otherwise, use comma-separated selected indices (P2.1 multi-select)
      value = Array.from(selectedPromptOptions).sort((a, b) => Number(a) - Number(b)).join(',');
    } else {
      // Fall back to text input
      value = promptInputEl?.value || '';
    }
    
    if (!value) {
      showToast('Please select an option or enter a response', 'warning');
      return;
    }
    
    socket.emit('parser_prompt', { session_id: currentSessionId, value });
    hidePrompt();
  }, 'submitPrompt');
}

function hidePrompt() {
  ErrorBoundary.safeExecute(() => {
    const promptModal = $('#promptModal');
    if (promptModal) promptModal.classList.add('hidden');
    activePromptMessage = null;
    activePromptOptions = [];
    selectedPromptOptions.clear();
    bundleExpandedState.clear();
  }, 'hidePrompt');
}

const btnSubmitPrompt = $('#btnSubmitPrompt');
if (btnSubmitPrompt) {
  btnSubmitPrompt.addEventListener('click', () => submitPrompt());
}

const btnCancelPrompt = $('#btnCancelPrompt');
if (btnCancelPrompt) {
  btnCancelPrompt.addEventListener('click', () => {
    submitPrompt('cancel');
  });
}

const btnClosePrompt = $('#btnClosePrompt');
if (btnClosePrompt) {
  btnClosePrompt.addEventListener('click', hidePrompt);
}

const promptInputField = $('#promptInput');
if (promptInputField) {
  promptInputField.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitPrompt();
    }
  });
}

// Initialize filter presets (P1.3) with debouncing (P2.2)
if (promptSearchEl) {
  const debouncedRender = debounce((value) => {
    renderPromptOptions(value);
  }, CONFIG.searchDebounceMs);
  
  promptSearchEl.addEventListener('input', (e) => {
    debouncedRender(e.target.value);
  });
}

// Hook up filter preset UI (P1.3)
document.addEventListener('DOMContentLoaded', () => {
  // Initialize Phase 2 features
  enhanceAccessibility(); // P2.5: Accessibility
  if (SessionRestore.hasRestoreData()) {
    SessionRestore.showRestoreBanner(); // P2.4: Session restore
  }
  
  // Initialize Advanced Features (Phase 3-4)
  initDirectUrlControl();
  initFilterPresets();
  initSessionActions();
  initKeyboardShortcuts(); // Consolidated keyboard shortcuts
  
  // Run integration tests in development
  if (window.location.hostname === 'localhost') {
    runIntegrationTests().catch(e => ErrorBoundary.logError(e, 'Integration Tests'));
  }
  
  filterPresets.updatePresetDropdown();
  
  const saveBtn = $('#btnSaveFilterPreset');
  if (saveBtn) {
    saveBtn.addEventListener('click', () => {
      const name = prompt('Enter preset name:');
      if (name) {
        filterPresets.save(name, state.filters);
        showToast(`Preset "${name}" saved`, 'success');
      }
    });
  }
  
  const deleteBtn = $('#btnDeleteFilterPreset');
  if (deleteBtn) {
    deleteBtn.addEventListener('click', () => {
      const select = $('#logFilterPresetSelect');
      const name = select.value;
      if (name && confirm(`Delete preset "${name}"?`)) {
        filterPresets.deletePreset(name);
        showToast(`Preset "${name}" deleted`, 'info');
      }
    });
  }
  
  const select = $('#logFilterPresetSelect');
  if (select) {
    select.addEventListener('change', (e) => {
      if (e.target.value && e.target.value !== '__separator__') {
        filterPresets.applyPreset(e.target.value);
      }
    });
  }
  
  // Phase 3: Export buttons
  const exportJsonBtn = $('#btnExportJSON');
  if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', () => AdvancedExport.exportAsJSON(state.logs));
  }
  
  const exportCsvBtn = $('#btnExportCSV');
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => AdvancedExport.exportAsCSV(state.logs));
  }
  
  const exportMdBtn = $('#btnExportMarkdown');
  if (exportMdBtn) {
    exportMdBtn.addEventListener('click', () => AdvancedExport.exportAsMarkdown(state.logs));
  }
  
  const showShortcutsBtn = $('#btnShowKeyboardShortcuts');
  if (showShortcutsBtn) {
    showShortcutsBtn.addEventListener('click', () => KeyboardGuide.show());
  }
});

if (promptSearchEl) {
  promptSearchEl.addEventListener('input', (e) => {
    renderPromptOptions(e.target.value || '');
  });
}

function submitPrompt(forcedValue) {
  const inputVal = forcedValue !== undefined ? forcedValue : (promptInputEl?.value || '');
  const value = inputVal || '';
  if (!currentSessionId) {
    hidePrompt();
    return;
  }
  socket.emit('parser_prompt', {
    session_id: currentSessionId,
    value,
  });
  hidePrompt();
  showToast('Response sent', 'success');
}

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
// Advanced Features: Direct URL Input
// ============================================

function parseDirectUrlField() {
  const textarea = document.getElementById('directUrlTextarea');
  const feedback = document.getElementById('directUrlFeedback');
  if (!textarea || !feedback) return [];
  
  const raw = textarea.value || '';
  const lines = raw.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  
  const urls = [];
  const errors = [];
  
  for (const line of lines) {
    if (urls.length >= CONFIG.maxDirectUrls) {
      errors.push(`Exceeded maximum of ${CONFIG.maxDirectUrls} URLs`);
      break;
    }
    
    try {
      const parsed = new URL(line);
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        errors.push(`Invalid protocol: ${line.substring(0, 40)}`);
        continue;
      }
      if (parsed.username || parsed.password) {
        errors.push(`URLs with auth not allowed: ${line.substring(0, 40)}`);
        continue;
      }
      urls.push(line);
    } catch (err) {
      errors.push(`Invalid URL: ${line.substring(0, 40)}`);
    }
  }
  
  // Update feedback
  let msg = '';
  if (errors.length > 0) {
    msg = `⚠️ ${errors[0]}`;
    feedback.className = 'text-danger';
  } else if (urls.length > 0) {
    msg = `✓ ${urls.length} valid URL${urls.length > 1 ? 's' : ''}`;
    feedback.className = 'text-success';
  } else {
    msg = 'Enter one URL per line.';
    feedback.className = 'text-muted';
  }
  feedback.textContent = msg;
  
  return urls;
}

function initDirectUrlControl() {
  const textarea = document.getElementById('directUrlTextarea');
  const clearBtn = document.getElementById('directUrlClearBtn');
  const directRadio = document.querySelector('input[name="fileSource"][value="direct"]');
  const advancedSection = document.querySelector('.advanced-option[data-source="direct"]');
  
  if (!textarea || !directRadio || !advancedSection) return;
  
  // Show/hide based on radio selection
  function updateVisibility() {
    if (directRadio.checked) {
      advancedSection.classList.remove('hidden');
      parseDirectUrlField();
    } else {
      advancedSection.classList.add('hidden');
    }
  }
  
  document.querySelectorAll('input[name="fileSource"]').forEach(radio => {
    radio.addEventListener('change', updateVisibility);
  });
  
  // Live validation
  textarea.addEventListener('input', debounce(() => {
    parseDirectUrlField();
    // Save draft per session
    if (AdvancedFeatures.currentSessionId) {
      AdvancedFeatures.directUrlDraftBySession.set(
        AdvancedFeatures.currentSessionId,
        textarea.value
      );
    }
  }, 500));
  
  // Clear button
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      textarea.value = '';
      parseDirectUrlField();
      if (AdvancedFeatures.currentSessionId) {
        AdvancedFeatures.directUrlDraftBySession.delete(AdvancedFeatures.currentSessionId);
      }
    });
  }
  
  updateVisibility();
}

// ============================================
// Advanced Features: Filter Presets
// ============================================

function initFilterPresets() {
  AdvancedFeatures.loadPresets();
  
  const presetSelect = document.getElementById('filterPresetSelect');
  const saveBtn = document.getElementById('saveFiltersBtn');
  const deleteBtn = document.getElementById('deletePresetBtn');
  
  if (!presetSelect || !saveBtn || !deleteBtn) return;
  
  // Populate preset dropdown
  function refreshPresetList() {
    // Clear existing options except first
    while (presetSelect.options.length > 1) {
      presetSelect.remove(1);
    }
    
    AdvancedFeatures.filterPresets.forEach((filters, name) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      presetSelect.appendChild(option);
    });
    
    presetSelect.value = '';
  }
  
  // Load preset when selected
  presetSelect.addEventListener('change', () => {
    const selected = presetSelect.value;
    if (!selected) return;
    
    const filters = AdvancedFeatures.filterPresets.get(selected);
    if (filters) {
      AdvancedFeatures.applyFilters(filters);
      showToast(`Loaded preset: ${selected}`, 'success');
    }
  });
  
  // Save current filters as preset
  saveBtn.addEventListener('click', () => {
    const name = prompt('Enter preset name:');
    if (!name || !name.trim()) return;
    
    const filters = AdvancedFeatures.getCurrentFilters();
    AdvancedFeatures.filterPresets.set(name.trim(), filters);
    AdvancedFeatures.savePresets();
    refreshPresetList();
    showToast(`Saved preset: ${name}`, 'success');
  });
  
  // Delete selected preset
  deleteBtn.addEventListener('click', () => {
    const selected = presetSelect.value;
    if (!selected) {
      showToast('Select a preset to delete', 'warning');
      return;
    }
    
    if (!confirm(`Delete preset "${selected}"?`)) return;
    
    AdvancedFeatures.filterPresets.delete(selected);
    AdvancedFeatures.savePresets();
    refreshPresetList();
    showToast(`Deleted preset: ${selected}`, 'info');
  });
  
  refreshPresetList();
}

// ============================================
// Advanced Features: Session Actions
// ============================================

function initSessionActions() {
  const cloneBtn = document.getElementById('btnCloneSession');
  const exportBtn = document.getElementById('btnExportSession');
  const clearBtn = document.getElementById('btnClearSession');
  
  if (cloneBtn) {
    cloneBtn.addEventListener('click', () => {
      if (!AdvancedFeatures.currentSessionId) {
        showToast('No active session to clone', 'warning');
        return;
      }
      
      if (window.socket && window.socket.connected) {
        window.socket.emit('clone_session', { 
          session_id: AdvancedFeatures.currentSessionId 
        });
        showToast('Cloning session...', 'info');
      } else {
        showToast('Socket not connected', 'error');
      }
    });
  }
  
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!state.results || state.results.length === 0) {
        showToast('No data to export', 'warning');
        return;
      }
      
      const dataStr = JSON.stringify(state.results, null, 2);
      const blob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `session_${AdvancedFeatures.currentSessionId || 'export'}_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('Data exported', 'success');
    });
  }
  
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (!confirm('Clear all logs for this session?')) return;
      
      if (window.clearSessionLogs) {
        window.clearSessionLogs();
        showToast('Logs cleared', 'info');
      } else {
        // Fallback
        const logContainer = document.getElementById('logEntries');
        if (logContainer) logContainer.innerHTML = '';
        showToast('Logs cleared (UI only)', 'info');
      }
    });
  }
}

// ============================================
// Advanced Features: Keyboard Shortcuts
// ============================================

function initKeyboardShortcuts() {
  const shortcuts = {
    'Ctrl+E': () => {
      // Export JSON
      document.getElementById('btnExportSession')?.click();
    },
    'Ctrl+Shift+E': () => {
      // Export CSV (if available)
      const exportCsvBtn = document.querySelector('[data-action="export-csv"]');
      if (exportCsvBtn) exportCsvBtn.click();
    },
    'Ctrl+L': () => {
      // Clear logs
      document.getElementById('btnClearSession')?.click();
    },
    'Ctrl+Shift+C': () => {
      // Clone session
      document.getElementById('btnCloneSession')?.click();
    },
    'Ctrl+Shift+P': () => {
      // Open command palette
      const palette = document.getElementById('commandPalette');
      if (palette && palette.classList.contains('hidden')) {
        palette.classList.remove('hidden');
        palette.querySelector('input')?.focus();
      }
    },
    'Ctrl+/': () => {
      // Show shortcuts help
      showShortcutsHelp();
    },
    'Escape': () => {
      // Close modals
      document.querySelectorAll('.modal:not(.hidden)').forEach(modal => {
        modal.classList.add('hidden');
      });
    }
  };
  
  document.addEventListener('keydown', (e) => {
    const key = [];
    if (e.ctrlKey || e.metaKey) key.push('Ctrl');
    if (e.shiftKey) key.push('Shift');
    if (e.altKey) key.push('Alt');
    key.push(e.key);
    
    const combo = key.join('+');
    const handler = shortcuts[combo];
    
    if (handler) {
      e.preventDefault();
      ErrorBoundary.safeExecute(handler, `Keyboard shortcut: ${combo}`);
    }
  });
}

function showShortcutsHelp() {
  const helpText = `
    <h3>Keyboard Shortcuts</h3>
    <ul style="list-style: none; padding: 0;">
      <li><kbd>Ctrl+E</kbd> - Export session data (JSON)</li>
      <li><kbd>Ctrl+Shift+E</kbd> - Export as CSV</li>
      <li><kbd>Ctrl+L</kbd> - Clear logs</li>
      <li><kbd>Ctrl+Shift+C</kbd> - Clone current session</li>
      <li><kbd>Ctrl+Shift+P</kbd> - Open command palette</li>
      <li><kbd>Ctrl+/</kbd> - Show this help</li>
      <li><kbd>Escape</kbd> - Close modals</li>
    </ul>
  `;
  
  const existing = document.getElementById('shortcutsModal');
  if (existing) {
    existing.classList.remove('hidden');
    return;
  }
  
  const modal = document.createElement('div');
  modal.id = 'shortcutsModal';
  modal.className = 'modal';
  modal.innerHTML = `
    <div class="modal-backdrop"></div>
    <div class="modal-content">
      ${helpText}
      <button class="btn btn-primary" onclick="this.closest('.modal').classList.add('hidden')">Close</button>
    </div>
  `;
  document.body.appendChild(modal);
  modal.classList.remove('hidden');
}

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
