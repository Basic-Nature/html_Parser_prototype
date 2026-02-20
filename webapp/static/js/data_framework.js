/**
 * Data Framework UI (refactored with defensive guards & client-side hardening)
 * NOTE: Real SQL injection mitigation must occur server-side via parameterized queries.
 * This client enforces:
 *  - Column name allowlisting (alphanumeric + underscore)
 *  - Sanitized search term (length + control char stripping)
 *  - Strict sort direction cycling / no arbitrary query fragments
 *  - Safe CSV generation (quotes + CR/LF normalized)
 */
document.addEventListener('DOMContentLoaded', () => {
  // ---------- Bootstrap activation ----------
  if ((/** @type {any} */ (window)).bootstrap) {
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
      .forEach(el => bootstrap.Tooltip.getOrCreateInstance(el));
    document.querySelectorAll('[data-bs-toggle="popover"]')
      .forEach(el => bootstrap.Popover.getOrCreateInstance(el));
  }

  // ---------- Config hydration ----------
  const cfgEl = document.getElementById('dataFrameworkConfig');
  const hydratedUrl = cfgEl?.dataset?.apiUrl;
  const apiUrl =
    hydratedUrl ||  // server now injects absolute path via url_for
    ((/** @type {any} */ (window)).__DATA_FRAMEWORK__ && (/** @type {any} */ (window)).__DATA_FRAMEWORK__.apiUrl) ||
    '/api/warehouse_election_results';
  // Additional injectable endpoints
  const uploadUrl = cfgEl?.dataset?.uploadUrl || '/upload/input';
  const scaffoldJsonUrl = cfgEl?.dataset?.scaffoldJsonUrl || '/api/data_framework/scaffold';
  const scaffoldCsvUrl = cfgEl?.dataset?.scaffoldCsvUrl || '/api/data_framework/scaffold.csv';
  const curatedUrl = cfgEl?.dataset?.curatedUrl || '/api/data_framework/curated';
  const priorityUrl = cfgEl?.dataset?.priorityUrl || '/api/data_framework/warehouse_status';
  const previewUrl = cfgEl?.dataset?.previewUrl || '/api/data_framework/preview';
  const dbLiteFinalizedUrl = cfgEl?.dataset?.dbliteFinalizedUrl || '/api/election_data/db_lite/finalized?limit=2000';
  const dbLiteDownBallotUrl = cfgEl?.dataset?.dbliteDownballotUrl || '/api/election_data/db_lite/down_ballot?limit=2000';
  const worklistOverviewUrl = '/api/election_data/worklist/overview';
  const csrfToken = cfgEl?.dataset?.csrfToken || null;

  // ---------- Elements ----------
  const el = {
    theadRow: document.getElementById('table-header'),
    tbody: document.getElementById('table-body'),
    status: document.getElementById('tableStatus'),
    search: document.getElementById('globalSearch'),
    pageSize: document.getElementById('pageSizeSelect'),
    first: document.getElementById('firstPageBtn'),
    prev: document.getElementById('prevPageBtn'),
    next: document.getElementById('nextPageBtn'),
    last: document.getElementById('lastPageBtn'),
    pageInfo: document.getElementById('pageInfo'),
    exportCsv: document.getElementById('exportCsvBtn'),
    scaffoldJson: document.getElementById('scaffoldJsonBtn'),
    scaffoldCsv: document.getElementById('scaffoldCsvBtn'),
    compactToggle: document.getElementById('compactToggleBtn'),
    warehouseTable: document.getElementById('warehouseTableSection'),
    resetFilters: document.getElementById('resetFiltersBtn'),
    refresh: document.getElementById('refreshBtn'),
    colBtn: document.getElementById('columnChooserBtn'),
    colMenu: document.getElementById('columnChooserMenu'),
    copyVisibleCsv: document.getElementById('copyVisibleCsv'),
    uploadForm: document.getElementById('uploadForm'),
    uploadStatus: document.getElementById('uploadStatus'),
    warehousePriorityStatus: document.getElementById('warehousePriorityStatus'),
    warehousePriorityMeta: document.getElementById('warehousePriorityMeta'),
    priorityStateSelect: document.getElementById('priorityStateSelect'),
    priorityYearSelect: document.getElementById('priorityYearSelect'),
    curatedSearch: document.getElementById('curatedSearch'),
    curatedState: document.getElementById('curatedStateFilter'),
    curatedCounty: document.getElementById('curatedCountyFilter'),
    curatedList: document.getElementById('curatedDatasetList'),
    curatedDetail: document.getElementById('curatedDatasetDetail'),
    curatedRows: document.getElementById('curatedRows'),
    curatedColumns: document.getElementById('curatedColumns'),
    curatedUpdated: document.getElementById('curatedUpdated'),
    curatedMeta: document.getElementById('curatedMeta'),
    curatedLinks: document.getElementById('curatedLinks'),
    curatedStatus: document.getElementById('curatedStatus'),
    curatedRefresh: document.getElementById('curatedRefreshBtn'),
    vizChart: document.getElementById('warehouseVizChart'),
    vizTable: document.getElementById('warehouseVizTable'),
    vizPanel: document.getElementById('vizPanel'),
    vizFilters: document.querySelector('.viz-filters'),
    vizDataset: document.getElementById('vizDatasetSelect'),
    vizPreviewStatus: document.getElementById('vizPreviewStatus'),
    dropoffDrawer: document.getElementById('dropoffDrawer'),
    dropoffDrawerToggle: document.getElementById('dropoffDrawerToggle'),
    dropoffDrawerOverlay: document.getElementById('dropoffDrawerOverlay'),
    vizYear: document.getElementById('vizYearSelect'),
    vizState: document.getElementById('vizStateSelect'),
    vizCounty: document.getElementById('vizCountySelect'),
    vizContest: document.getElementById('vizContestSelect'),
    vizTopRace: document.getElementById('vizTopRaceSelect'),
    vizTopRaceCount: document.getElementById('vizTopRaceCountSelect'),
    vizParty: document.getElementById('vizPartySelect'),
    vizPrevStateBtn: document.getElementById('vizPrevStateBtn'),
    vizAutoToggleBtn: document.getElementById('vizAutoToggleBtn'),
    vizNextStateBtn: document.getElementById('vizNextStateBtn'),
    vizDropoffOverlayToggle: document.getElementById('vizDropoffOverlayToggle'),
    vizHint: document.getElementById('vizAutoHint'),
    dropoffModeDropoff: document.getElementById('dropoffModeDropoff'),
    dropoffModeTotals: document.getElementById('dropoffModeTotals'),
    dropoffState: document.getElementById('dropoffStateInput'),
    dropoffContest: document.getElementById('dropoffContestInput'),
    dropoffCounty: document.getElementById('dropoffCountySelect'),
    dropoffYear: document.getElementById('dropoffYearSelect'),
    dropoffParty: document.getElementById('dropoffPartySelect'),
    dropoffScale: document.getElementById('dropoffScaleSelect'),
    dropoffCountyLimit: document.getElementById('dropoffCountyLimitSelect'),
    dropoffOrder: document.getElementById('dropoffOrderSelect'),
    dropoffMetricPercent: document.getElementById('dropoffMetricPercent'),
    dropoffMetricRaw: document.getElementById('dropoffMetricRaw'),
    dropoffChart: document.getElementById('dropoffChart'),
    dropoffChartVotes: document.getElementById('dropoffChartVotes'),
    dropoffChartPercent: document.getElementById('dropoffChartPercent'),
    dropoffSummary: document.getElementById('dropoffSummary'),
    dropoffControls: document.getElementById('dropoffControls'),
    dropoffPanel: document.getElementById('dropoffPanel'),
    ghostPanel: document.getElementById('ghostPanel'),
    ghostPanelToggle: document.getElementById('ghostPanelToggle'),
    ghostPanelBody: document.getElementById('ghostPanelBody'),
    pipelineSteps: document.getElementById('uploadPipelineSteps'),
    pipelineDetail: document.getElementById('uploadPipelineDetail')
  };
  const colWrap = el.colBtn?.parentElement;

  // ---------- Toast helpers ----------
  function toast(id, msg) {
    const t = document.getElementById(id);
    if (!t) return;
    const body = t.querySelector('.toast-body');
    if (body && msg) body.textContent = msg;
    bootstrap?.Toast.getOrCreateInstance(t).show();
  }
  const showInfoToast = m => toast('toastInfo', m);
  const showErrorToast = m => toast('toastError', m);

  // ---------- State ----------
  let rawData = [];
  let visibleColumns = [];
  let allowedColumns = new Set();       // allowlist derived + validated
  let sortBy = null;
  let sortDir = 'none';                 // 'ascending' | 'descending' | 'none'
  let searchTerm = '';
  let page = 1;
  let priorityState = '';
  let priorityYear = '';
  let lastPriorityPayload = null;
  let worklistOverviewRecords = [];
  let worklistOverviewMeta = null;
  let curatedItems = [];
  let curatedSelection = null;
  let curatedSearch = '';
  let curatedState = '';
  let curatedCounty = '';
  let vizRows = [];
  let dbLiteFinalizedRows = [];
  let dbLiteDownBallotRows = [];
  let vizDataset = 'finalized';
  let vizYear = '';
  let vizState = '';
  let vizCounty = '';
  let vizContest = '';
  let vizParty = '';
  let vizAutoTimer = null;
  let vizTopRaces = [];
  let vizAutoIndex = 0;
  let vizAutoLocked = false;
  let vizTopRaceCount = 5;
  let vizAutoOrder = [];
  let vizAutoPaused = false;
  let vizOverlayEnabled = false;
  let previewActive = false;
  let previewMode = 'idle';
  let previewTimer = null;
  let compactPreferenceSet = false;
  let dropoffDrawerOpen = false;
  let dropoffData = [];
  let dropoffMetric = 'percent';
  let dropoffYear = '';
  let dropoffScaleMode = 'absolute';
  let dropoffCountyLimit = 50;
  let dropoffOrderStrategy = 'turnout_weighted';
  // Safely read pageSize from select/input if present
  let pageSize = 25;
  try {
    const psEl = el.pageSize;
    if (psEl instanceof HTMLSelectElement || psEl instanceof HTMLInputElement) {
      pageSize = Math.max(1, parseInt(psEl.value || '25', 10) || 25);
    }
  } catch (err) {
    pageSize = 25;
  }

  // ---------- Constants / Policies ----------
  const COL_NAME_RX = /^[A-Za-z0-9_]{1,64}$/;
  const MAX_SEARCH_LEN = 1200;
  const MAX_VISIBLE_COLS = 200;
  const MAX_ROWS_EXPORT = 200000; // safeguard client memory for CSV
  const SKELETON_ROWS = 6;
  const VIZ_TOP_COUNT_KEY = 'df_viz_top_count';
  const COMPACT_TABLE_KEY = 'df_table_compact';
  const COMPACT_AUTO_THRESHOLD = 300;
  const PRIORITY_REFRESH_MS = 60000;
  const DROPOFF_DRAWER_KEY = 'df_dropoff_drawer';
  const VIZ_OVERLAY_KEY = 'df_viz_overlay_enabled';
  const VIZ_PLAYBACK_PAUSED_KEY = 'df_viz_playback_paused';
  const DROPOFF_ORDER_KEY = 'df_dropoff_order_strategy';
  const DROPOFF_SCALE_KEY = 'df_dropoff_scale_mode';
  const DROPOFF_COUNTY_LIMIT_KEY = 'df_dropoff_county_limit';
  const VIZ_DATASET_FINALIZED = 'finalized';
  const VIZ_DATASET_DOWN_BALLOT = 'down_ballot';
  const DEFAULT_VISIBLE_COLUMNS = ['state', 'county', 'contest', 'candidate', 'party', 'votes'];

  // ---------- Utilities ----------
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
  const safeGet = v => (v == null ? '' : String(v));
  function setStatus(target, type, text) {
    if (!target) return;
    target.className = 'status ' + (type === 'ok' ? 'status-ok' : type === 'error' ? 'status-error' : 'status-info');
    target.textContent = text || '';
  }
  function sanitizeSearch(raw) {
    if (!raw) return '';
    let s = raw.slice(0, MAX_SEARCH_LEN);
    // Strip control chars except basic whitespace
    s = s.replace(/[\x00-\x08\x0B-\x1F\x7F]/g, '');
    return s.trim();
  }
  function parseNumeric(value) {
    if (value == null || value === '') return 0;
    if (typeof value === 'number') return value;
    const cleaned = String(value).replace(/,/g, '').replace(/[^\d.-]/g, '').trim();
    const num = Number(cleaned);
    return Number.isFinite(num) ? num : 0;
  }
  function parsePercent(value) {
    if (value == null || value === '') return 0;
    if (typeof value === 'number') return Number.isFinite(value) ? value : 0;
    const cleaned = String(value).replace('%', '').trim();
    const num = Number(cleaned);
    return Number.isFinite(num) ? num : 0;
  }
  function extractYearFromValue(value) {
    if (!value) return '';
    const match = String(value).match(/(20\d{2})/);
    return match ? match[1] : '';
  }
  function getRowYear(row) {
    return extractYearFromValue(row.year || row.election_date || row.date || row.timestamp || '');
  }
  function isAllowedColumn(name) {
    return COL_NAME_RX.test(name);
  }
  function normalizeColumns(keys) {
    const filtered = [];
    for (const k of keys) {
      if (isAllowedColumn(k)) {
        filtered.push(k);
        allowedColumns.add(k);
      }
    }
    return filtered.slice(0, MAX_VISIBLE_COLS);
  }
  function cycleSortDirection(current) {
    if (current === 'none') return 'ascending';
    if (current === 'ascending') return 'descending';
    return 'none';
  }

  function setStatusText(target, text) {
    if (!target) return;
    target.textContent = text || '';
  }

  function setPriorityStatus(text, tone = 'info') {
    if (!el.warehousePriorityStatus) return;
    el.warehousePriorityStatus.className = `warehouse-status-strip status status-${tone}`;
    el.warehousePriorityStatus.textContent = text || '';
  }

  function setPriorityMeta(text) {
    if (!el.warehousePriorityMeta) return;
    el.warehousePriorityMeta.textContent = text || '';
  }

  function hydratePriorityStates(payload) {
    if (!(el.priorityStateSelect instanceof HTMLSelectElement)) return;
    const states = Array.isArray(payload?.states) ? payload.states : [];
    if (!states.length) return;
    const existing = Array.from(el.priorityStateSelect.options).map(opt => opt.value);
    if (existing.length > 1) return;
    el.priorityStateSelect.innerHTML = '';
    const allOpt = document.createElement('option');
    allOpt.value = '';
    allOpt.textContent = 'All states';
    el.priorityStateSelect.appendChild(allOpt);
    states.forEach(state => {
      const opt = document.createElement('option');
      opt.value = state;
      opt.textContent = state;
      el.priorityStateSelect.appendChild(opt);
    });
    if (priorityState) el.priorityStateSelect.value = priorityState;
  }

  function hydratePriorityYears(payload) {
    if (!(el.priorityYearSelect instanceof HTMLSelectElement)) return;
    const years = Array.isArray(payload?.available_years) ? payload.available_years : [];
    el.priorityYearSelect.innerHTML = '';
    const allOpt = document.createElement('option');
    allOpt.value = '';
    allOpt.textContent = 'All years';
    el.priorityYearSelect.appendChild(allOpt);
    years.forEach(year => {
      const opt = document.createElement('option');
      opt.value = String(year);
      opt.textContent = String(year);
      el.priorityYearSelect.appendChild(opt);
    });
    if (priorityYear && years.length && !years.map(String).includes(String(priorityYear))) {
      priorityYear = '';
    }
    const selectedYear = payload?.selected_year ? String(payload.selected_year) : '';
    if (!priorityYear && selectedYear) {
      priorityYear = selectedYear;
    }
    if (priorityYear) {
      el.priorityYearSelect.value = priorityYear;
    }
  }

  function getSelectedPriorityYear(payload) {
    if (priorityYear) return String(priorityYear);
    const selected = payload?.selected_year;
    if (selected) return String(selected);
    const years = Array.isArray(payload?.available_years) ? payload.available_years : [];
    return years.length ? String(years[0]) : '';
  }

  function formatDivisionSummary(payload) {
    const selectedYear = getSelectedPriorityYear(payload);
    const perYear = Array.isArray(payload?.division_summary_by_year)
      ? payload.division_summary_by_year
      : [];
    let divisions = [];
    if (selectedYear && perYear.length) {
      divisions = perYear.filter(row => String(row.year) === String(selectedYear));
    }
    if (!divisions.length) {
      divisions = Array.isArray(payload?.division_summary) ? payload.division_summary : [];
    }
    if (!divisions.length) return 'Divisions: —';
    const divText = divisions.map(row => `${row.type} ${row.rows || 0}`).join(' • ');
    return selectedYear ? `Divisions (${selectedYear}): ${divText}` : `Divisions: ${divText}`;
  }

  function normalizeWorklistKey(value) {
    return String(value || '')
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '');
  }

  function getWorklistRecordValue(record, keys) {
    if (!record || typeof record !== 'object') return '';
    const lookup = Object.keys(record).reduce((acc, key) => {
      acc[normalizeWorklistKey(key)] = record[key];
      return acc;
    }, {});
    for (const key of keys) {
      const normalized = normalizeWorklistKey(key);
      if (normalized in lookup) return lookup[normalized];
    }
    return '';
  }

  function matchesPriorityState(recordState, filterState) {
    if (!filterState) return true;
    const recordValue = String(recordState || '').trim().toLowerCase();
    const filterValue = String(filterState || '').trim().toLowerCase();
    if (!recordValue || !filterValue) return false;
    if (recordValue === filterValue) return true;
    if (filterValue.length === 2 && recordValue.startsWith(filterValue)) return true;
    if (recordValue.length === 2 && filterValue.startsWith(recordValue)) return true;
    return false;
  }

  function buildWorklistSummary(records, payload) {
    if (!Array.isArray(records) || !records.length) return '';
    const targetYear = getSelectedPriorityYear(payload);
    const filtered = records.filter(record => {
      const stateValue = getWorklistRecordValue(record, ['state', 'state_code', 'st']);
      if (!matchesPriorityState(stateValue, priorityState)) return false;
      if (!targetYear) return true;
      const yearValue = getWorklistRecordValue(record, ['year', 'election_year', 'cycle', 'election_date']);
      const recordYear = extractYearFromValue(yearValue);
      return recordYear ? String(recordYear) === String(targetYear) : false;
    });
    const total = filtered.length;
    if (!total) return 'Worklist: 0 rows';
    const statusCounts = new Map();
    const priorityCounts = new Map();
    filtered.forEach(record => {
      const statusValue = getWorklistRecordValue(record, ['status', 'workflow_status', 'workflow', 'step', 'stage', 'phase']);
      const priorityValue = getWorklistRecordValue(record, ['priority', 'tier', 'importance']);
      if (statusValue) {
        const key = String(statusValue).trim();
        statusCounts.set(key, (statusCounts.get(key) || 0) + 1);
      }
      if (priorityValue) {
        const key = String(priorityValue).trim();
        priorityCounts.set(key, (priorityCounts.get(key) || 0) + 1);
      }
    });
    const topStatus = Array.from(statusCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([label, count]) => `${label} ${count}`);
    const topPriority = Array.from(priorityCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([label, count]) => `${label} ${count}`);
    const parts = [];
    if (topStatus.length) parts.push(`Status: ${topStatus.join(' • ')}`);
    if (topPriority.length) parts.push(`Priority: ${topPriority.join(' • ')}`);
    return `Worklist: ${total} rows${parts.length ? ` | ${parts.join(' | ')}` : ''}`;
  }

  function formatPriorityMeta(payload) {
    const years = Array.isArray(payload?.available_years) ? payload.available_years : [];
    const yearText = years.length ? `Years: ${years.join(', ')}` : 'Years: —';
    const divText = formatDivisionSummary(payload);
    const worklistText = buildWorklistSummary(worklistOverviewRecords, payload);
    return [yearText, divText, worklistText].filter(Boolean).join(' | ');
  }

  function formatPrioritySummary(payload) {
    const expected = Number(payload?.expected_total || 0);
    const missing = Number(payload?.missing_total || 0);
    const parts = [`Missing ${missing} / ${expected} expected`];
    const byPriority = Array.isArray(payload?.by_priority) ? payload.by_priority : [];
    const top = byPriority
      .filter(row => row?.priority)
      .sort((a, b) => (b.missing || 0) - (a.missing || 0))
      .slice(0, 3)
      .map(row => `${row.priority}: ${row.missing || 0}`);
    if (top.length) parts.push(top.join(' • '));
    return parts.join(' | ');
  }

  async function fetchPriorityStatus() {
    if (!priorityUrl) return;
    try {
      const url = new URL(priorityUrl, window.location.origin);
      if (priorityState) url.searchParams.set('state', priorityState);
      if (priorityYear) url.searchParams.set('year', priorityYear);
      const response = await fetch(url.toString(), { headers: { 'Accept': 'application/json' } });
      if (!response.ok) {
        setPriorityStatus('Priority tracker unavailable.', 'error');
        setPriorityMeta('');
        return;
      }
      const payload = await response.json().catch(() => null);
      if (!payload) {
        setPriorityStatus('Priority tracker unavailable.', 'error');
        setPriorityMeta('');
        return;
      }
      if (payload.error || payload.available === false) {
        const msg = payload.error || 'Priority tracker unavailable.';
        setPriorityStatus(msg, 'error');
        setPriorityMeta('');
        return;
      }
      hydratePriorityStates(payload);
      hydratePriorityYears(payload);
      lastPriorityPayload = payload;
      const summary = formatPrioritySummary(payload);
      setPriorityStatus(summary || 'Priority tracker ready.', payload.missing_total ? 'info' : 'ok');
      setPriorityMeta(formatPriorityMeta(payload));
    } catch (err) {
      setPriorityStatus('Priority tracker unavailable.', 'error');
      setPriorityMeta('');
    }
  }

  async function fetchWorklistOverview() {
    if (!worklistOverviewUrl) return;
    try {
      const url = new URL(worklistOverviewUrl, window.location.origin);
      url.searchParams.set('limit', '200');
      const response = await fetch(url.toString(), { headers: { 'Accept': 'application/json' } });
      if (!response.ok) return;
      const payload = await response.json().catch(() => null);
      if (!payload || payload.success === false) return;
      worklistOverviewRecords = Array.isArray(payload.records) ? payload.records : [];
      worklistOverviewMeta = {
        sheet: payload.sheet_name || '',
        rowCount: payload.row_count || 0
      };
      if (lastPriorityPayload) {
        setPriorityMeta(formatPriorityMeta(lastPriorityPayload));
      }
    } catch (err) {
      // Best-effort only.
    }
  }

  function setCompactTable(active) {
    if (el.warehouseTable) {
      el.warehouseTable.classList.toggle('is-compact', !!active);
    }
    if (el.compactToggle) {
      el.compactToggle.setAttribute('aria-pressed', active ? 'true' : 'false');
      el.compactToggle.classList.toggle('is-active', !!active);
      el.compactToggle.textContent = active ? 'Comfortable' : 'Dense';
    }
  }

  function loadCompactPreference() {
    try {
      const stored = window.localStorage?.getItem(COMPACT_TABLE_KEY);
      if (stored === null) return;
      compactPreferenceSet = true;
      setCompactTable(stored === 'true');
    } catch (err) {
      // Ignore storage errors.
    }
  }

  function setDropoffDrawer(open) {
    dropoffDrawerOpen = !!open;
    if (el.dropoffDrawer) {
      el.dropoffDrawer.classList.toggle('is-open', dropoffDrawerOpen);
    }
    if (el.dropoffDrawerOverlay) {
      el.dropoffDrawerOverlay.classList.toggle('is-visible', dropoffDrawerOpen);
    }
    if (el.dropoffDrawerToggle) {
      el.dropoffDrawerToggle.setAttribute('aria-expanded', dropoffDrawerOpen ? 'true' : 'false');
      el.dropoffDrawerToggle.textContent = dropoffDrawerOpen ? 'Close drop-off' : 'Drop-off';
    }
    if (el.vizPanel) {
      el.vizPanel.classList.toggle('is-drawer-open', dropoffDrawerOpen);
    }
  }

  function loadDropoffDrawerPreference() {
    try {
      const stored = window.localStorage?.getItem(DROPOFF_DRAWER_KEY);
      if (stored === null) return;
      setDropoffDrawer(stored === 'true');
    } catch (err) {
      // Ignore storage errors.
    }
  }

  function loadGhostPanelPreference() {
    try {
      const stored = window.localStorage?.getItem('ghostPanelMinimized');
      if (stored === 'true' && el.ghostPanel) {
        el.ghostPanel.classList.add('is-minimized');
        if (el.ghostPanelToggle) {
          el.ghostPanelToggle.textContent = '+';
          el.ghostPanelToggle.setAttribute('aria-label', 'Expand placeholder');
        }
      }
    } catch (err) {
      // Ignore storage errors.
    }
  }

  function updateCuratedCountyOptions(items) {
    if (!el.curatedCounty) return;
    const counties = Array.from(new Set(items.map(item => item.county).filter(Boolean))).sort();
    el.curatedCounty.innerHTML = '';
    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = 'All counties';
    el.curatedCounty.appendChild(defaultOpt);
    counties.forEach(county => {
      const opt = document.createElement('option');
      opt.value = county;
      opt.textContent = county;
      el.curatedCounty.appendChild(opt);
    });
    if (el.curatedCounty instanceof HTMLSelectElement) {
      el.curatedCounty.disabled = counties.length === 0;
    }
  }

  function loadVizTopCountPreference() {
    if (!(el.vizTopRaceCount instanceof HTMLSelectElement)) return;
    try {
      const stored = window.localStorage?.getItem(VIZ_TOP_COUNT_KEY);
      if (!stored) return;
      const count = parseInt(stored, 10);
      if (!Number.isFinite(count)) return;
      const option = Array.from(el.vizTopRaceCount.options).find(opt => Number(opt.value) === count);
      if (option) {
        el.vizTopRaceCount.value = option.value;
        vizTopRaceCount = count;
      }
    } catch (err) {
      // Ignore storage read errors.
    }
  }

  function loadVizOverlayPreference() {
    try {
      const stored = window.localStorage?.getItem(VIZ_OVERLAY_KEY);
      if (stored === null) return;
      vizOverlayEnabled = stored === 'true';
      if (el.vizDropoffOverlayToggle instanceof HTMLInputElement) {
        el.vizDropoffOverlayToggle.checked = vizOverlayEnabled;
      }
    } catch (err) {
      // Ignore storage read errors.
    }
  }

  function loadVizPlaybackPreference() {
    try {
      const stored = window.localStorage?.getItem(VIZ_PLAYBACK_PAUSED_KEY);
      if (stored === null) return;
      const paused = stored === 'true';
      if (paused) {
        vizAutoLocked = true;
        vizAutoPaused = true;
      }
    } catch (err) {
      // Ignore storage read errors.
    }
  }

  function saveVizPlaybackPreference(paused) {
    try {
      window.localStorage?.setItem(VIZ_PLAYBACK_PAUSED_KEY, String(!!paused));
    } catch (err) {
      // Ignore storage write errors.
    }
  }

  function loadDropoffPreferences() {
    try {
      const scaleStored = window.localStorage?.getItem(DROPOFF_SCALE_KEY);
      if (scaleStored === 'absolute' || scaleStored === 'adjusted') {
        dropoffScaleMode = scaleStored;
        if (el.dropoffScale instanceof HTMLSelectElement) {
          el.dropoffScale.value = scaleStored;
        }
      }
      const orderStored = window.localStorage?.getItem(DROPOFF_ORDER_KEY);
      if (orderStored === 'turnout_weighted' || orderStored === 'absolute' || orderStored === 'alphabetical') {
        dropoffOrderStrategy = orderStored;
        if (el.dropoffOrder instanceof HTMLSelectElement) {
          el.dropoffOrder.value = orderStored;
        }
      }
      const limitStored = window.localStorage?.getItem(DROPOFF_COUNTY_LIMIT_KEY);
      if (limitStored) {
        dropoffCountyLimit = limitStored === 'all' ? 0 : Math.max(1, parseInt(limitStored, 10) || 50);
        if (el.dropoffCountyLimit instanceof HTMLSelectElement) {
          el.dropoffCountyLimit.value = limitStored;
        }
      }
    } catch (err) {
      // Ignore storage read errors.
    }
  }

  function updateCuratedStateOptions(items) {
    if (!el.curatedState) return;
    const states = Array.from(new Set(items.map(item => item.state).filter(Boolean))).sort();
    el.curatedState.innerHTML = '';
    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = 'All states';
    el.curatedState.appendChild(defaultOpt);
    states.forEach(state => {
      const opt = document.createElement('option');
      opt.value = state;
      opt.textContent = state;
      el.curatedState.appendChild(opt);
    });
  }

  function renderCuratedDetail(item) {
    stopPreviewCycle();
    curatedSelection = item;
    if (el.curatedRows) el.curatedRows.textContent = item?.row_count != null ? String(item.row_count) : '—';
    if (el.curatedColumns) el.curatedColumns.textContent = item?.column_count != null ? String(item.column_count) : '—';
    if (el.curatedUpdated) el.curatedUpdated.textContent = item?.updated_at || '—';
    if (el.curatedMeta) {
      const parts = [item?.contest, item?.state, item?.county].filter(Boolean).join(' • ');
      el.curatedMeta.textContent = parts || 'No additional metadata available.';
    }
    if (el.curatedLinks) {
      el.curatedLinks.innerHTML = '';
      if (item?.source_url) {
        const link = document.createElement('a');
        link.href = item.source_url;
        link.target = '_blank';
        link.rel = 'noreferrer';
        link.className = 'btn action';
        link.textContent = 'Open source URL';
        el.curatedLinks.appendChild(link);
      }
    }

    updateVisualizationFromCurated(item);
  }

  function clearVisualization() {
    if (el.vizChart) {
      el.vizChart.innerHTML = '<div class="viz-placeholder">Select a dataset to render charts.</div>';
    }
    if (el.vizTable) {
      el.vizTable.innerHTML = '<div class="viz-placeholder">Top rows will appear here for the chosen dataset.</div>';
    }
    if (el.vizYear) el.vizYear.innerHTML = '';
    if (el.vizContest) el.vizContest.innerHTML = '';
    if (el.vizTopRace) el.vizTopRace.innerHTML = '';
    vizRows = [];
    vizYear = '';
    vizContest = '';
    vizTopRaces = [];
    vizAutoLocked = false;
    vizAutoPaused = false;
    vizAutoOrder = [];
    stopVizAutoRotation();
  }

  function renderVizChart(rows) {
    if (!el.vizChart) return;
    el.vizChart.innerHTML = '';
    if (!rows.length) {
      el.vizChart.innerHTML = '<div class="viz-placeholder">No rows available for this dataset.</div>';
      return;
    }
    const buckets = [
      { key: 'dem', label: 'Dem' },
      { key: 'rep', label: 'Rep' },
      { key: 'lib', label: 'Lib' },
      { key: 'grn', label: 'Grn' },
      { key: 'ind', label: 'Ind' },
      { key: 'non', label: 'Nonpart' },
      { key: 'writein', label: 'Write-In' },
      { key: 'other', label: 'Other' }
    ];
    const totals = Object.fromEntries(buckets.map(entry => [entry.key, 0]));
    rows.forEach(row => {
      const bucket = normalizePartyBucket(row.party || row.ballot_party || '');
      totals[bucket] = (totals[bucket] || 0) + (Number(row.votes) || 0);
    });
    const totalVotes = Object.values(totals).reduce((sum, val) => sum + val, 0) || 1;
    const stack = document.createElement('div');
    stack.className = 'viz-stack';
    const segments = buckets
      .map(entry => ({ label: entry.label, value: totals[entry.key], tone: entry.key }))
      .filter(segment => segment.value > 0);
    (segments.length ? segments : [{ label: 'Other', value: totalVotes, tone: 'other' }]).forEach(segment => {
      const seg = document.createElement('div');
      seg.className = `viz-stack-seg viz-stack-${segment.tone}`;
      seg.style.setProperty('--seg-size', `${Math.round((segment.value / totalVotes) * 100)}%`);
      seg.title = `${segment.label}: ${segment.value.toLocaleString()}`;
      stack.appendChild(seg);
    });

    const legend = document.createElement('div');
    legend.className = 'viz-legend';
    segments.forEach(segment => {
      const item = document.createElement('div');
      item.className = `viz-legend-item viz-legend-${segment.tone}`;
      item.textContent = `${segment.label}: ${segment.value.toLocaleString()}`;
      legend.appendChild(item);
    });

    const totalLine = document.createElement('div');
    totalLine.className = 'viz-total';
    totalLine.textContent = `Total votes: ${totalVotes.toLocaleString()}`;

    appendDropoffOverlayToVizStack(stack);

    el.vizChart.appendChild(stack);
    el.vizChart.appendChild(legend);
    el.vizChart.appendChild(totalLine);
  }

  function appendDropoffOverlayToVizStack(stackEl) {
    if (!stackEl || !vizOverlayEnabled) return;
    const scopeRows = getDropoffRowsForYear(vizYear, dropoffData)
      .filter(row => !vizState || normalizeValue(row.state) === normalizeValue(vizState));
    const countySeries = buildCountyDropoffSeries(scopeRows);
    if (!countySeries.length) return;
    const points = countySeries
      .slice()
      .sort((a, b) => Math.abs(b.delta_pct || 0) - Math.abs(a.delta_pct || 0))
      .slice(0, 36);
    const maxAbs = Math.max(1, ...points.map(item => Math.abs(item.delta_pct || 0)));
    const svg = createSvgElement('svg');
    svg.classList.add('viz-dropoff-overlay');
    svg.setAttribute('viewBox', '0 0 1000 120');
    svg.setAttribute('preserveAspectRatio', 'none');

    const mid = createSvgElement('line');
    mid.setAttribute('x1', '0');
    mid.setAttribute('x2', '1000');
    mid.setAttribute('y1', '60');
    mid.setAttribute('y2', '60');
    mid.setAttribute('class', 'viz-dropoff-overlay-mid');
    svg.appendChild(mid);

    const path = createSvgElement('path');
    const coords = points.map((row, idx) => {
      const x = points.length === 1 ? 500 : Math.round((idx / (points.length - 1)) * 1000);
      const y = 60 - Math.round(((Number(row.delta_pct) || 0) / maxAbs) * 55);
      return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
    path.setAttribute('d', coords);
    path.setAttribute('class', 'viz-dropoff-overlay-line');
    svg.appendChild(path);

    stackEl.appendChild(svg);

    const note = document.createElement('div');
    note.className = 'viz-dropoff-overlay-note';
    note.textContent = `Overlay: county drop-off trend (${points.length} counties)`;
    stackEl.insertAdjacentElement('afterend', note);
  }

  function renderVizTable(rows) {
    if (!el.vizTable) return;
    el.vizTable.innerHTML = '';
    if (!rows.length) {
      el.vizTable.innerHTML = '<div class="viz-placeholder">No rows available for this dataset.</div>';
      return;
    }
    const datasetType = rows[0]?.dataset_type || VIZ_DATASET_FINALIZED;
    const sorted = [...rows].sort((a, b) => {
      const aVotes = Number(a.votes ?? 0) || 0;
      const bVotes = Number(b.votes ?? 0) || 0;
      return bVotes - aVotes;
    });
    const table = document.createElement('table');
    table.className = 'viz-table';
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    const headers = datasetType === VIZ_DATASET_DOWN_BALLOT
      ? ['County', 'Party', 'Down-Ballot Votes', 'Presidential Votes', 'Drop-off %']
      : ['County/District', 'Dem Votes', 'Rep Votes', 'Other Votes', 'Write-In Votes', 'Uncategorized Votes', 'Total Votes'];
    headers.forEach(label => {
      const th = document.createElement('th');
      th.textContent = label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    const renderValue = value => {
      if (value == null || value === '') return '—';
      if (typeof value === 'number') return value.toLocaleString();
      return String(value);
    };
    if (datasetType === VIZ_DATASET_DOWN_BALLOT) {
      sorted.slice(0, 8).forEach(row => {
        const tr = document.createElement('tr');
        const cells = [
          row.county,
          row.party,
          row.down_ballot_votes,
          row.presidential_votes,
          row.dropoff_pct
        ];
        cells.forEach(value => {
          const td = document.createElement('td');
          td.textContent = renderValue(value);
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    } else {
      const grouped = new Map();
      sorted.forEach(row => {
        const countyKey = String(row.county || '—').trim() || '—';
        const entry = grouped.get(countyKey) || {
          county: countyKey,
          dem: 0,
          rep: 0,
          other: 0,
          writein: 0,
          uncategorized: 0,
          total: 0,
        };
        const bucket = normalizePartyBucket(row.party || row.ballot_party || '');
        const votes = Number(row.votes || 0) || 0;
        const uncategorized = Number(row.uncategorized_votes || 0) || 0;
        if (bucket === 'dem') entry.dem += votes;
        else if (bucket === 'rep') entry.rep += votes;
        else if (bucket === 'writein') entry.writein += votes;
        else entry.other += votes;
        entry.uncategorized += uncategorized;
        entry.total += votes;
        grouped.set(countyKey, entry);
      });
      const countyRows = Array.from(grouped.values())
        .sort((a, b) => b.total - a.total)
        .slice(0, 12);
      countyRows.forEach(row => {
        const tr = document.createElement('tr');
        const cells = [
          row.county,
          row.dem,
          row.rep,
          row.other,
          row.writein,
          row.uncategorized,
          row.total
        ];
        cells.forEach(value => {
          const td = document.createElement('td');
          td.textContent = renderValue(value);
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    }
    table.appendChild(tbody);
    el.vizTable.appendChild(table);
  }

  function setVizFilters(rows) {
    vizTopRaces = [];
    // Populate Year dropdown
    const years = Array.from(new Set(rows.map(row => getRowYear(row)).filter(Boolean)))
      .filter(Boolean)
      .sort((a, b) => Number(b) - Number(a));
    if (el.vizYear instanceof HTMLSelectElement) {
      el.vizYear.innerHTML = '';
      years.forEach(year => {
        const opt = document.createElement('option');
        opt.value = year;
        opt.textContent = year;
        el.vizYear.appendChild(opt);
      });
      vizYear = years[0] || '';
      if (vizYear) el.vizYear.value = vizYear;
    }

    // Populate State dropdown based on selected year
    updateVizStates();

    const contests = Array.from(new Set(rows.map(row => row.contest).filter(Boolean))).sort();
    if (el.vizContest instanceof HTMLSelectElement) {
      el.vizContest.innerHTML = '';
      contests.forEach(contest => {
        const opt = document.createElement('option');
        opt.value = contest;
        opt.textContent = contest;
        el.vizContest.appendChild(opt);
      });
    }

    updateTopRaces();

    if (!contests.includes(vizContest)) {
      vizContest = vizTopRaces[0] || contests[0] || '';
    }
    if (el.vizContest instanceof HTMLSelectElement && vizContest) {
      el.vizContest.value = vizContest;
    }
    if (el.vizTopRace instanceof HTMLSelectElement && vizContest) {
      el.vizTopRace.value = vizContest;
    }

    if (el.vizParty instanceof HTMLSelectElement) {
      vizParty = el.vizParty.value || '';
    }

    if (el.vizTopRaceCount instanceof HTMLSelectElement) {
      vizTopRaceCount = Math.max(1, parseInt(el.vizTopRaceCount.value, 10) || 5);
    }

    if (!vizAutoLocked) {
      startVizAutoRotation();
    }
  }

  function updateVizStates() {
    // Filter rows by selected year, then extract unique states from warehouse data
    const scopeRows = vizYear
      ? vizRows.filter(row => getRowYear(row) === vizYear)
      : vizRows;
    // Extract states from PostgreSQL warehouse data (ensure sync with database)
    const states = Array.from(new Set(scopeRows.map(row => row.state).filter(Boolean))).sort();
    if (el.vizState instanceof HTMLSelectElement) {
      el.vizState.innerHTML = '';
      states.forEach(state => {
        const opt = document.createElement('option');
        opt.value = state;
        opt.textContent = state;
        el.vizState.appendChild(opt);
      });
      if (states.length && !states.includes(vizState)) {
        vizState = states[0];
      }
      if (vizState) el.vizState.value = vizState;
    }
    updateVizCounties();
  }

  function updateVizCounties() {
    // Filter rows by year + state, then extract unique counties from warehouse data
    const scopeRows = vizYear && vizState
      ? vizRows.filter(row => 
          getRowYear(row) === vizYear &&
          row.state === vizState
        )
      : vizYear
        ? vizRows.filter(row => getRowYear(row) === vizYear)
        : vizRows;
    // Extract counties from PostgreSQL warehouse data (cascading filter from state)
    const counties = Array.from(new Set(scopeRows.map(row => row.county).filter(Boolean))).sort();
    if (el.vizCounty instanceof HTMLSelectElement) {
      el.vizCounty.innerHTML = '';
      counties.forEach(county => {
        const opt = document.createElement('option');
        opt.value = county;
        opt.textContent = county;
        el.vizCounty.appendChild(opt);
      });
      if (counties.length && !counties.includes(vizCounty)) {
        vizCounty = counties[0];
      }
      if (vizCounty) el.vizCounty.value = vizCounty;
    }
  }

  function updateTopRaces() {
    const contestTotals = {};
    const scopeRows = vizYear
      ? vizRows.filter(row => getRowYear(row) === vizYear)
      : vizRows;
    scopeRows.forEach(row => {
      if (!row.contest) return;
      contestTotals[row.contest] = (contestTotals[row.contest] || 0) + (Number(row.votes) || 0);
    });
    const topRaces = Object.entries(contestTotals)
      .sort((a, b) => b[1] - a[1])
      .slice(0, vizTopRaceCount);
    vizTopRaces = topRaces.map(entry => entry[0]);
    if (el.vizTopRace instanceof HTMLSelectElement) {
      el.vizTopRace.innerHTML = '';
      topRaces.forEach(([contest]) => {
        const opt = document.createElement('option');
        opt.value = contest;
        opt.textContent = contest;
        el.vizTopRace.appendChild(opt);
      });
    }
    if (vizTopRaces.length && !vizTopRaces.includes(vizContest)) {
      setVizContest(vizTopRaces[0]);
    }
  }

  function applyVizFilters(rows) {
    let filtered = rows;
    if (vizYear) {
      filtered = filtered.filter(row => getRowYear(row) === vizYear);
    }
    if (vizState) {
      filtered = filtered.filter(row => row.state === vizState);
    }
    if (vizCounty) {
      filtered = filtered.filter(row => row.county === vizCounty);
    }
    if (vizContest) {
      filtered = filtered.filter(row => row.contest === vizContest);
    }
    if (vizParty) {
      filtered = filtered.filter(row => normalizePartyBucket(row.party) === normalizePartyBucket(vizParty));
    }
    return filtered;
  }

  function shuffleArray(items) {
    for (let i = items.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
    return items;
  }

  function refreshViz() {
    const filtered = applyVizFilters(vizRows);
    renderVizChart(filtered);
    renderVizTable(filtered);
  }

  function setVizContest(value, syncTopRace = true) {
    if (!value) return;
    vizContest = value;
    if (el.vizContest instanceof HTMLSelectElement) {
      el.vizContest.value = value;
    }
    if (syncTopRace && el.vizTopRace instanceof HTMLSelectElement) {
      el.vizTopRace.value = value;
    }
    refreshViz();
  }

  function setVizYear(value) {
    if (!value) return;
    vizYear = value;
    if (el.vizYear instanceof HTMLSelectElement) {
      el.vizYear.value = value;
    }
    updateVizStates();
    updateTopRaces();
    refreshViz();
  }

  function setVizState(value) {
    if (!value) return;
    vizState = value;
    if (el.vizState instanceof HTMLSelectElement) {
      el.vizState.value = value;
    }
    updateVizCounties();
    refreshViz();
  }

  function setVizCounty(value) {
    if (!value) return;
    vizCounty = value;
    if (el.vizCounty instanceof HTMLSelectElement) {
      el.vizCounty.value = value;
    }
    refreshViz();
  }

  function setVizDataset(value) {
    if (!value) return;
    vizDataset = value;
    if (el.vizDataset instanceof HTMLSelectElement) {
      el.vizDataset.value = value;
    }
    const sourceRows = getVizSourceRows();
    applyVizDatasetRows(sourceRows);
  }

  function stopVizAutoRotation() {
    if (vizAutoTimer) {
      window.clearInterval(vizAutoTimer);
      vizAutoTimer = null;
    }
  }

  function startVizAutoRotation(resetOrder = true) {
    if (vizAutoLocked || vizAutoPaused) {
      stopVizAutoRotation();
      updateVizAutoToggleLabel();
      return;
    }
    stopVizAutoRotation();
    const states = getVizStateList();
    if (!states.length) return;
    // Always rebuild order from scratch (never persist position)
    vizAutoOrder = [...states];
    // Shuffle for randomized start position each session
    for (let i = vizAutoOrder.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [vizAutoOrder[i], vizAutoOrder[j]] = [vizAutoOrder[j], vizAutoOrder[i]];
    }
    vizAutoIndex = 0;
    setVizStateContext(vizAutoOrder[vizAutoIndex]);
    vizAutoTimer = window.setInterval(() => {
      if (vizAutoLocked) return;
      vizAutoIndex = (vizAutoIndex + 1) % vizAutoOrder.length;
      setVizStateContext(vizAutoOrder[vizAutoIndex]);
    }, 6000);
    updateVizAutoToggleLabel();
  }

  function pauseVizAutoRotation() {
    if (vizAutoLocked) return;
    vizAutoPaused = true;
    stopVizAutoRotation();
    if (el.vizHint) el.vizHint.classList.add('is-visible');
    updateVizAutoToggleLabel();
  }

  function hideVizHint() {
    if (el.vizHint) el.vizHint.classList.remove('is-visible');
  }

  function resumeVizAutoRotation() {
    if (vizAutoLocked || !vizAutoPaused) return;
    vizAutoPaused = false;
    startVizAutoRotation(false);
    hideVizHint();
    updateVizAutoToggleLabel();
  }

  function getVizStateList() {
    // Return sorted list; startVizAutoRotation will shuffle for random start
    const scopeRows = vizYear
      ? vizRows.filter(row => getRowYear(row) === vizYear)
      : vizRows;
    return Array.from(new Set(scopeRows.map(row => String(row.state || '').trim()).filter(Boolean))).sort();
  }

  function pickTopContestForState(state) {
    if (!state) return '';
    const totals = {};
    vizRows.forEach(row => {
      if ((getRowYear(row) || '') !== (vizYear || '')) return;
      if (String(row.state || '').trim() !== state) return;
      if (!row.contest) return;
      totals[row.contest] = (totals[row.contest] || 0) + (Number(row.votes) || 0);
    });
    const sorted = Object.entries(totals).sort((a, b) => b[1] - a[1]);
    return sorted[0]?.[0] || '';
  }

  function setVizStateContext(state) {
    if (!state) return;
    vizState = state;
    if (el.vizState instanceof HTMLSelectElement) {
      el.vizState.value = state;
    }
    updateVizCounties();
    const topContest = pickTopContestForState(state);
    if (topContest) {
      vizContest = topContest;
      if (el.vizContest instanceof HTMLSelectElement) el.vizContest.value = topContest;
      if (el.vizTopRace instanceof HTMLSelectElement) el.vizTopRace.value = topContest;
    }
    refreshViz();
  }

  function stepVizState(step) {
    const states = getVizStateList();
    if (!states.length) return;
    const currentIndex = Math.max(0, states.indexOf(vizState));
    const nextIndex = (currentIndex + step + states.length) % states.length;
    setVizStateContext(states[nextIndex]);
    vizAutoOrder = [...states];
    vizAutoIndex = nextIndex;
  }

  function updateVizAutoToggleLabel() {
    if (!el.vizAutoToggleBtn) return;
    el.vizAutoToggleBtn.textContent = vizAutoPaused ? 'Start' : 'Pause';
  }

  function setPreviewStatus(text) {
    if (!el.vizPreviewStatus) return;
    el.vizPreviewStatus.textContent = text || '';
  }

  function setPreviewState(active) {
    if (el.vizPanel) {
      el.vizPanel.classList.toggle('is-previewing', !!active);
    }
  }

  function ghostPreviewPanels() {
    const panels = [el.vizChart, el.vizTable].filter(Boolean);
    panels.forEach(panel => panel.classList.add('is-ghosting'));
    window.setTimeout(() => {
      panels.forEach(panel => panel.classList.remove('is-ghosting'));
    }, 700);
  }

  function buildPreviewUrl(mode) {
    if (!previewUrl) return null;
    const url = new URL(previewUrl, window.location.origin);
    url.searchParams.set('mode', mode || 'idle');
    url.searchParams.set('limit', '140');
    if (mode === 'active') {
      const state = getSelectedDropoffState();
      const county = getSelectedDropoffCounty();
      const contest = getSelectedDropoffContest();
      const year = getSelectedDropoffYear();
      if (state) url.searchParams.set('state', state);
      if (county) url.searchParams.set('county', county);
      if (contest) url.searchParams.set('contest', contest);
      if (year) url.searchParams.set('year', year);
    }
    return url.toString();
  }

  async function fetchPreviewPayload(mode) {
    const url = buildPreviewUrl(mode);
    if (!url) return null;
    try {
      const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (!response.ok) return null;
      return await response.json().catch(() => null);
    } catch (err) {
      return null;
    }
  }

  async function refreshPreview() {
    if (!previewActive) return;
    if (curatedSelection) return;
    const sourceRows = getVizSourceRows();
    if (sourceRows.length) {
      applyVizDatasetRows(sourceRows);
      const label = vizDataset === VIZ_DATASET_DOWN_BALLOT ? 'DB-Lite Down-Ballot' : 'DB-Lite Finalized';
      setPreviewStatus(`${label} • ${sourceRows.length} rows`);
      ghostPreviewPanels();
      return;
    }
    const payload = await fetchPreviewPayload(previewMode);
    const rows = Array.isArray(payload?.rows) ? payload.rows : [];
    if (!rows.length) {
      setPreviewStatus('Preview unavailable — waiting for warehouse data.');
      clearVisualization();
      return;
    }
    vizRows = rows;
    setVizFilters(vizRows);
    refreshViz();
    ghostPreviewPanels();
    const meta = payload?.meta || {};
    const label = [meta.contest, meta.county, meta.state, meta.year].filter(Boolean).join(' • ');
    setPreviewStatus(label ? `Previewing ${label}` : 'Previewing warehouse sample');
  }

  function startPreviewCycle(mode = 'idle') {
    if (!previewUrl) return;
    previewActive = true;
    previewMode = mode;
    setPreviewState(true);
    if (previewTimer) {
      window.clearInterval(previewTimer);
      previewTimer = null;
    }
    refreshPreview();
    previewTimer = window.setInterval(refreshPreview, 12000);
  }

  function stopPreviewCycle() {
    previewActive = false;
    if (previewTimer) {
      window.clearInterval(previewTimer);
      previewTimer = null;
    }
    setPreviewState(false);
    setPreviewStatus('');
  }

  function setDetectionMode(active) {
    const target = el.dropoffPanel || el.dropoffControls;
    if (target) {
      target.classList.toggle('is-detecting', !!active);
    }
  }

  function normalizeValue(value) {
    return (value || '').toString().trim().toLowerCase();
  }

  function mapDbLiteFinalizedRecord(record) {
    if (!record || typeof record !== 'object') return null;
    const rawParty = record['Ballot Party'] || record['Party'] || record.party || '';
    const isWriteIn = String(record['Is Write In'] || '').toLowerCase() === 'true'
      || String(record['Is Write In'] || '').toLowerCase() === 'yes'
      || String(record['Is Write In'] || '') === '1';
    const partyLabel = isWriteIn ? 'Write-In' : rawParty;
    return {
      dataset_type: VIZ_DATASET_FINALIZED,
      state: record['State'] || record.state || '',
      county: record['County/District'] || record['County'] || record.county || '',
      contest: record['Office'] || record['Contest'] || record.contest || '',
      candidate: record['Ballot Candidate Name'] || record['Candidate'] || record.candidate || '',
      party: partyLabel || record['Party'] || record.party || '',
      votes: parseNumeric(record['Total Votes'] || record['Uncategorized Votes'] || record.votes || 0),
      uncategorized_votes: parseNumeric(record['Uncategorized Votes'] || 0),
      early_votes: parseNumeric(record['Early Votes'] || 0),
      election_day_votes: parseNumeric(record['Election Day Votes'] || 0),
      mail_in_votes: parseNumeric(record['Mail in Votes'] || 0),
      provisional_votes: parseNumeric(record['Provisional Votes'] || 0),
      write_in_votes: isWriteIn ? 1 : 0,
      year: extractYearFromValue(record['Year'] || record['Election Date'] || record['Contest'] || '')
    };
  }

  function mapDbLiteDownBallotRecord(record) {
    if (!record || typeof record !== 'object') return null;
    const downVotes = parseNumeric(record['Down-Ballot Votes'] || record.down_ballot_votes || 0);
    const presidentialVotes = parseNumeric(record['Presidential Votes'] || record.presidential_votes || 0);
    const deltaVotes = downVotes - presidentialVotes;
    const explicitPct = parsePercent(record['Drop-off %'] || record.dropoff_pct || 0);
    const computedPct = presidentialVotes ? (deltaVotes / presidentialVotes) * 100 : 0;
    const eligibleVoters = parseNumeric(
      record['Eligible Voters']
      || record['Total Eligible Voters']
      || record['Registered Voters']
      || record['Voting Age Population']
      || record.eligible_voters
      || record.registered_voters
      || 0
    );
    return {
      dataset_type: VIZ_DATASET_DOWN_BALLOT,
      year: extractYearFromValue(record['Year'] || record.year || ''),
      state: record['State'] || record.state || '',
      county: record['County'] || record.county || '',
      contest: record['Office'] || record.office || 'Down-Ballot',
      party: record['Party'] || record.party || '',
      votes: downVotes || presidentialVotes || parseNumeric(record.votes || 0),
      down_ballot_votes: downVotes,
      presidential_votes: presidentialVotes,
      delta_votes: deltaVotes,
      dropoff_pct: explicitPct || computedPct,
      eligible_voters: eligibleVoters,
      turnout_pct: eligibleVoters ? (presidentialVotes / eligibleVoters) * 100 : 0
    };
  }

  function getVizSourceRows() {
    return vizDataset === VIZ_DATASET_DOWN_BALLOT ? dbLiteDownBallotRows : dbLiteFinalizedRows;
  }

  function applyVizDatasetRows(rows) {
    vizRows = Array.isArray(rows) ? rows : [];
    if (!vizRows.length) {
      clearVisualization();
      return;
    }
    vizAutoLocked = false;
    setVizFilters(vizRows);
    refreshViz();
  }

  function updateVisualizationFromCurated(item) {
    if (!item) {
      clearVisualization();
      return;
    }
    vizAutoLocked = false;
    const sourceRows = getVizSourceRows();
    if (!sourceRows.length) {
      clearVisualization();
      return;
    }
    const filtered = sourceRows.filter(row => {
      const matchState = item.state ? normalizeValue(row.state) === normalizeValue(item.state) : true;
      const matchCounty = item.county ? normalizeValue(row.county) === normalizeValue(item.county) : true;
      const matchContest = item.contest ? normalizeValue(row.contest) === normalizeValue(item.contest) : true;
      const matchYear = item.year ? getRowYear(row) === String(item.year) : true;
      return matchState && matchCounty && matchContest && matchYear;
    });
    applyVizDatasetRows(filtered.length ? filtered : sourceRows);
  }

  function renderCuratedList(items) {
    if (!el.curatedList) return;
    el.curatedList.innerHTML = '';
    if (!items.length) {
      const empty = document.createElement('div');
      empty.className = 'curated-empty';
      empty.textContent = 'No curated datasets match the current filters.';
      el.curatedList.appendChild(empty);
      return;
    }
    items.forEach(item => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `curated-item${curatedSelection?.id === item.id ? ' is-active' : ''}`;
      btn.setAttribute('role', 'listitem');
      const title = document.createElement('div');
      title.className = 'curated-item-title';
      title.textContent = item.title || item.contest || 'Dataset';
      const meta = document.createElement('div');
      meta.className = 'curated-item-meta';
      meta.textContent = [item.state, item.county, item.year].filter(Boolean).join(' • ');
      btn.appendChild(title);
      btn.appendChild(meta);
      btn.addEventListener('click', () => {
        document.querySelectorAll('.curated-item.is-active').forEach(elm => elm.classList.remove('is-active'));
        btn.classList.add('is-active');
        renderCuratedDetail(item);
      });
      el.curatedList.appendChild(btn);
    });
  }

  function filterCuratedItems() {
    let items = curatedItems;
    if (curatedSearch) {
      const q = curatedSearch.toLowerCase();
      items = items.filter(item =>
        [item.title, item.state, item.county, item.contest].some(val => (val || '').toLowerCase().includes(q))
      );
    }
    if (curatedState) items = items.filter(item => item.state === curatedState);
    if (curatedCounty) items = items.filter(item => item.county === curatedCounty);
    renderCuratedList(items);
    const selectedId = curatedSelection?.id || null;
    if (!selectedId || !items.some(item => item.id === selectedId)) {
      curatedSelection = null;
      if (items[0] && !previewActive) {
        renderCuratedDetail(items[0]);
        const firstButton = el.curatedList?.querySelector('.curated-item');
        if (firstButton) firstButton.classList.add('is-active');
      }
    }
  }

  function fetchCuratedDatasets() {
    if (!curatedUrl) return;
    setStatusText(el.curatedStatus, 'Loading curated datasets...');
    fetch(curatedUrl, { headers: { 'Accept': 'application/json' } })
      .then(r => r.json())
      .then(data => {
        curatedItems = Array.isArray(data?.items) ? data.items : [];
        updateCuratedStateOptions(curatedItems);
        updateCuratedCountyOptions(curatedItems);
        filterCuratedItems();
        if (!curatedItems.length && !getVizSourceRows().length) {
          clearVisualization();
        }
        setStatusText(el.curatedStatus, curatedItems.length ? `Loaded ${curatedItems.length} datasets.` : 'No curated datasets available.');
      })
      .catch(err => {
        curatedItems = [];
        renderCuratedList([]);
        setStatusText(el.curatedStatus, `Failed to load curated datasets: ${err?.message || err}`);
      });
  }

  async function fetchDbLiteDataset(url, mapper) {
    if (!url) return { ok: false, rows: [] };
    try {
      const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (!response.ok) return { ok: false, rows: [] };
      const payload = await response.json().catch(() => null);
      const records = Array.isArray(payload?.records) ? payload.records : [];
      const rows = records.map(mapper).filter(Boolean);
      return {
        ok: true,
        rows,
        rowCount: payload?.row_count || rows.length,
        sheetName: payload?.sheet_name || ''
      };
    } catch (err) {
      return { ok: false, rows: [] };
    }
  }

  async function fetchDbLiteFinalized() {
    const result = await fetchDbLiteDataset(dbLiteFinalizedUrl, mapDbLiteFinalizedRecord);
    dbLiteFinalizedRows = result.ok ? result.rows : [];
    if (result.ok && vizDataset === VIZ_DATASET_FINALIZED && !curatedSelection) {
      applyVizDatasetRows(dbLiteFinalizedRows);
      setPreviewStatus(`DB-Lite Finalized • ${result.rowCount || dbLiteFinalizedRows.length} rows`);
    }
  }

  async function fetchDbLiteDownBallot() {
    const result = await fetchDbLiteDataset(dbLiteDownBallotUrl, mapDbLiteDownBallotRecord);
    dbLiteDownBallotRows = result.ok ? result.rows : [];
    loadDropoffData();
    if (result.ok && vizDataset === VIZ_DATASET_DOWN_BALLOT && !curatedSelection) {
      applyVizDatasetRows(dbLiteDownBallotRows);
      setPreviewStatus(`DB-Lite Down-Ballot • ${result.rowCount || dbLiteDownBallotRows.length} rows`);
    }
  }

  function loadDropoffData() {
    dropoffData = Array.isArray(dbLiteDownBallotRows) ? [...dbLiteDownBallotRows] : [];
    hydrateDropoffSelectors(dropoffData);
  }

  function getDropoffRowsForControls(rows = dropoffData) {
    const state = normalizeValue(getSelectedDropoffState());
    const contest = normalizeValue(getSelectedDropoffContest());
    const party = normalizeValue(getSelectedDropoffParty());
    return rows.filter(item => {
      const stateOk = !state || normalizeValue(item.state).includes(state);
      const contestOk = !contest || normalizeValue(item.contest).includes(contest);
      const partyOk = !party || normalizePartyBucket(item.party) === normalizePartyBucket(party);
      return stateOk && contestOk && partyOk;
    });
  }

  function getDropoffRowsForYear(year, rows = dropoffData) {
    const scoped = getDropoffRowsForControls(rows);
    if (!year) return scoped;
    return scoped.filter(item => String(item.year || '') === String(year));
  }

  function buildCountyDropoffSeries(rows) {
    const grouped = new Map();
    rows.forEach(row => {
      const county = String(row.county || '').trim();
      if (!county) return;
      const entry = grouped.get(county) || {
        county,
        down_ballot_votes: 0,
        presidential_votes: 0,
        eligible_voters: 0,
      };
      entry.down_ballot_votes += Number(row.down_ballot_votes || 0) || 0;
      entry.presidential_votes += Number(row.presidential_votes || 0) || 0;
      entry.eligible_voters += Number(row.eligible_voters || 0) || 0;
      grouped.set(county, entry);
    });
    const values = Array.from(grouped.values()).map(entry => {
      const deltaVotes = (entry.down_ballot_votes || 0) - (entry.presidential_votes || 0);
      const percentDelta = entry.presidential_votes
        ? (deltaVotes / entry.presidential_votes) * 100
        : 0;
      const turnoutPct = entry.eligible_voters
        ? ((entry.presidential_votes || 0) / entry.eligible_voters) * 100
        : 0;
      return {
        ...entry,
        delta_votes: deltaVotes,
        delta_pct: percentDelta,
        turnout_pct: turnoutPct,
        adjusted_votes: entry.presidential_votes ? (deltaVotes / entry.presidential_votes) * 10000 : deltaVotes,
      };
    });
    const maxPresidential = Math.max(1, ...values.map(item => item.presidential_votes || 0));
    values.forEach(item => {
      const weight = Math.sqrt((item.presidential_votes || 0) / maxPresidential);
      item.adjusted_pct = item.delta_pct * (Number.isFinite(weight) ? weight : 1);
    });
    return values;
  }

  function hydrateDropoffSelectors(rows) {
    const scopedRows = getDropoffRowsForControls(rows);
    const years = Array.from(new Set(scopedRows.map(row => String(row.year || '')).filter(Boolean)))
      .sort((a, b) => Number(b) - Number(a));
    if (el.dropoffYear instanceof HTMLSelectElement) {
      el.dropoffYear.innerHTML = '';
      years.forEach(year => {
        const opt = document.createElement('option');
        opt.value = year;
        opt.textContent = year;
        el.dropoffYear.appendChild(opt);
      });
      if (!years.includes(dropoffYear)) {
        dropoffYear = years[0] || '';
      }
      if (dropoffYear) el.dropoffYear.value = dropoffYear;
    }
    const currentRows = getDropoffRowsForYear(dropoffYear, rows);
    const countySeries = buildCountyDropoffSeries(currentRows);
    updateDropoffCountyOptions(countySeries);
    renderDropoffGraphs(countySeries);
    updateDropoffSummary(countySeries);
  }

  function updateDropoffCountyOptions(rows) {
    if (!(el.dropoffCounty instanceof HTMLSelectElement)) return;
    const previousValue = el.dropoffCounty.value || '';
    el.dropoffCounty.innerHTML = '';
    rows
      .slice()
      .sort((a, b) => String(a.county).localeCompare(String(b.county)))
      .forEach(row => {
        const opt = document.createElement('option');
        opt.value = row.county;
        opt.textContent = row.county;
        el.dropoffCounty.appendChild(opt);
      });
    if (previousValue && rows.some(row => row.county === previousValue)) {
      el.dropoffCounty.value = previousValue;
    } else if (rows[0]) {
      el.dropoffCounty.value = rows[0].county;
    }
  }

  function formatDropoffValue(value, decimals = 2) {
    const num = Number(value) || 0;
    return num.toLocaleString(undefined, { maximumFractionDigits: decimals, minimumFractionDigits: decimals });
  }

  function createSvgElement(tagName) {
    return document.createElementNS('http://www.w3.org/2000/svg', tagName);
  }

  function renderDivergingCountyGraph(targetEl, rows, config) {
    if (!targetEl) return;
    targetEl.innerHTML = '';
    const title = document.createElement('div');
    title.className = 'dropoff-graph-title';
    title.textContent = config.title;
    targetEl.appendChild(title);

    if (!rows.length) {
      const empty = document.createElement('div');
      empty.className = 'dropoff-summary-note';
      empty.textContent = 'No county data available for the current state/year/contest filter.';
      targetEl.appendChild(empty);
      return;
    }

    const selectedCounty = getSelectedDropoffCounty();
    const maxAbs = Math.max(1, ...rows.map(item => Math.abs(Number(config.value(item)) || 0)));
    const barWidth = rows.length > 120 ? 6 : rows.length > 80 ? 8 : rows.length > 40 ? 10 : 14;
    const gap = rows.length > 80 ? 2 : 3;
    const margin = { top: 14, right: 14, bottom: 76, left: 56 };
    const plotHeight = 190;
    const width = Math.max(720, margin.left + margin.right + rows.length * (barWidth + gap));
    const height = margin.top + plotHeight + margin.bottom;
    const zeroY = margin.top + Math.round(plotHeight / 2);
    const maxBarHeight = Math.round(plotHeight / 2) - 4;

    const scroll = document.createElement('div');
    scroll.className = 'dropoff-graph-scroll';
    const svg = createSvgElement('svg');
    svg.classList.add('dropoff-graph-svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('width', String(width));
    svg.setAttribute('height', String(height));

    for (let i = 0; i <= 4; i += 1) {
      const ratio = i / 4;
      const value = maxAbs - (ratio * maxAbs * 2);
      const y = margin.top + Math.round(ratio * plotHeight);
      const line = createSvgElement('line');
      line.setAttribute('x1', String(margin.left));
      line.setAttribute('x2', String(width - margin.right));
      line.setAttribute('y1', String(y));
      line.setAttribute('y2', String(y));
      line.setAttribute('class', Math.abs(value) < 1e-9 ? 'dropoff-zero-line' : 'dropoff-grid-line');
      svg.appendChild(line);

      const label = createSvgElement('text');
      label.setAttribute('x', String(margin.left - 8));
      label.setAttribute('y', String(y + 4));
      label.setAttribute('text-anchor', 'end');
      label.setAttribute('class', 'dropoff-axis-label');
      label.textContent = config.axisLabel(value);
      svg.appendChild(label);
    }

    const labelStep = Math.max(1, Math.ceil(rows.length / 16));
    rows.forEach((row, index) => {
      const rawValue = Number(config.value(row)) || 0;
      const x = margin.left + index * (barWidth + gap);
      const barHeight = Math.max(1, Math.round((Math.abs(rawValue) / maxAbs) * maxBarHeight));
      const y = rawValue >= 0 ? zeroY - barHeight : zeroY;

      const rect = createSvgElement('rect');
      rect.setAttribute('x', String(x));
      rect.setAttribute('y', String(y));
      rect.setAttribute('width', String(barWidth));
      rect.setAttribute('height', String(barHeight));
      rect.setAttribute('class', rawValue >= 0 ? 'dropoff-bar-pos' : 'dropoff-bar-neg');
      if (selectedCounty && row.county === selectedCounty) {
        rect.setAttribute('stroke', 'rgba(212 175 55 / 1)');
        rect.setAttribute('stroke-width', '1.2');
      }
      const titleNode = createSvgElement('title');
      titleNode.textContent = `${row.county}: ${config.tooltip(rawValue, row)}`;
      rect.appendChild(titleNode);
      svg.appendChild(rect);

      if (index % labelStep === 0 || row.county === selectedCounty) {
        const countyLabel = createSvgElement('text');
        countyLabel.setAttribute('x', String(x + (barWidth / 2)));
        countyLabel.setAttribute('y', String(height - 8));
        countyLabel.setAttribute('text-anchor', 'end');
        countyLabel.setAttribute('transform', `rotate(-58 ${x + (barWidth / 2)} ${height - 8})`);
        countyLabel.setAttribute('class', 'dropoff-axis-label');
        countyLabel.textContent = row.county;
        svg.appendChild(countyLabel);
      }
    });

    scroll.appendChild(svg);
    targetEl.appendChild(scroll);
  }

  function renderDropoffGraphs(seriesRows) {
    const rows = seriesRows.slice().sort((a, b) => {
      if (dropoffOrderStrategy === 'alphabetical') {
        return String(a.county || '').localeCompare(String(b.county || ''));
      }
      const aBase = dropoffMetric === 'percent' ? Math.abs(a.delta_pct || 0) : Math.abs(a.delta_votes || 0);
      const bBase = dropoffMetric === 'percent' ? Math.abs(b.delta_pct || 0) : Math.abs(b.delta_votes || 0);
      if (dropoffOrderStrategy === 'turnout_weighted') {
        const aWeight = Math.sqrt(Math.max(1, Number(a.presidential_votes || 0)));
        const bWeight = Math.sqrt(Math.max(1, Number(b.presidential_votes || 0)));
        return (bBase * bWeight) - (aBase * aWeight);
      }
      return bBase - aBase;
    });
    const limitedRows = dropoffCountyLimit > 0 ? rows.slice(0, dropoffCountyLimit) : rows;

    renderDivergingCountyGraph(el.dropoffChartVotes, limitedRows, {
      title: dropoffScaleMode === 'adjusted'
        ? 'County drop-off delta (votes), adjusted per 10k presidential votes'
        : 'County drop-off delta (votes): positive = gain, negative = loss',
      value: row => (dropoffScaleMode === 'adjusted' ? row.adjusted_votes : row.delta_votes),
      axisLabel: value => dropoffScaleMode === 'adjusted'
        ? `${formatDropoffValue(value, 1)}`
        : `${Math.round(value)}`,
      tooltip: (value, row) => {
        const raw = formatDropoffValue(row.delta_votes, 0);
        const adjusted = formatDropoffValue(row.adjusted_votes, 2);
        return dropoffScaleMode === 'adjusted'
          ? `${adjusted} (adj), raw ${raw}`
          : `${raw} votes`;
      }
    });

    renderDivergingCountyGraph(el.dropoffChartPercent, limitedRows, {
      title: dropoffScaleMode === 'adjusted'
        ? 'County drop-off percent, weighted by turnout size'
        : 'County drop-off percent (delta / presidential votes)',
      value: row => (dropoffScaleMode === 'adjusted' ? row.adjusted_pct : row.delta_pct),
      axisLabel: value => `${formatDropoffValue(value, 1)}%`,
      tooltip: (value, row) => {
        const raw = formatDropoffValue(row.delta_pct, 2);
        const weighted = formatDropoffValue(row.adjusted_pct, 2);
        return dropoffScaleMode === 'adjusted'
          ? `${weighted}% (weighted), raw ${raw}%`
          : `${raw}%`;
      }
    });
  }

  function updateDropoffSummary(seriesRows) {
    if (!el.dropoffSummary) return;
    if (!Array.isArray(seriesRows) || !seriesRows.length) {
      el.dropoffSummary.textContent = 'No drop-off county rows match the current filters.';
      return;
    }
    const selectedCounty = getSelectedDropoffCounty();
    const row = seriesRows.find(item => item.county === selectedCounty) || seriesRows[0];
    const countyLabel = row.county || 'County';
    const deltaVotes = formatDropoffValue(row.delta_votes, 0);
    const deltaPct = formatDropoffValue(row.delta_pct, 2);
    const turnoutPct = row.turnout_pct ? `${formatDropoffValue(row.turnout_pct, 2)}%` : 'n/a';
    const eligible = row.eligible_voters ? formatDropoffValue(row.eligible_voters, 0) : 'n/a';
    const scaleNote = dropoffScaleMode === 'adjusted'
      ? 'Adjusted scale dampens small-county outliers for side-by-side comparison.'
      : 'Absolute scale shows raw deltas.';
    el.dropoffSummary.innerHTML = `${countyLabel} • Δ votes ${deltaVotes} • Δ pct ${deltaPct}% • Pres turnout ${turnoutPct} • Eligible ${eligible}<div class="dropoff-summary-note">${scaleNote}</div>`;
  }

  function refreshDropoffVisuals() {
    const selectedYear = getSelectedDropoffYear() || dropoffYear;
    if (selectedYear) {
      dropoffYear = selectedYear;
    }
    const rows = getDropoffRowsForYear(dropoffYear, dropoffData);
    const countySeries = buildCountyDropoffSeries(rows);
    updateDropoffCountyOptions(countySeries);
    renderDropoffGraphs(countySeries);
    updateDropoffSummary(countySeries);
  }

  function initPipelineSteps() {
    if (!el.pipelineSteps || !el.pipelineDetail) return;
    const stepCopy = {
      ingest: 'Ingest CSV or export from your curated sources into the staging pipeline.',
      validate: 'Validate schema, required columns, and safe formatting rules before QA.',
      qa: 'Run quality checks, confidence scoring, and anomaly detection gates.',
      publish: 'Publish to PostgreSQL and expose in the warehouse view.'
    };
    el.pipelineSteps.querySelectorAll('.pipeline-step').forEach(step => {
      step.addEventListener('click', () => {
        el.pipelineSteps.querySelectorAll('.pipeline-step').forEach(btn => btn.classList.remove('is-active'));
        step.classList.add('is-active');
        const key = step.getAttribute('data-step');
        el.pipelineDetail.textContent = stepCopy[key] || 'Select a step to see what happens next.';
      });
    });
  }

  function getSelectedDropoffCounty() {
    if (el.dropoffCounty instanceof HTMLSelectElement) return el.dropoffCounty.value;
    return '';
  }

  function getSelectedDropoffYear() {
    if (el.dropoffYear instanceof HTMLSelectElement) return el.dropoffYear.value;
    return '';
  }

  function getSelectedDropoffState() {
    if (el.dropoffState instanceof HTMLInputElement) return el.dropoffState.value.trim().toUpperCase();
    return '';
  }

  function getSelectedDropoffContest() {
    if (el.dropoffContest instanceof HTMLInputElement) return el.dropoffContest.value.trim();
    return '';
  }

  function getSelectedDropoffParty() {
    if (el.dropoffParty instanceof HTMLSelectElement) return el.dropoffParty.value.trim();
    return '';
  }

  function normalizePartyBucket(party) {
    const raw = (party || '').toLowerCase();
    if (!raw) return 'other';
    if (/\(\s*r\s*\)/i.test(raw) || raw === 'r') return 'rep';
    if (/\(\s*d\s*\)/i.test(raw) || raw === 'd') return 'dem';
    if (/\(\s*i\s*\)/i.test(raw) || raw === 'i') return 'ind';
    if (/\(\s*l\s*\)/i.test(raw) || raw === 'l') return 'lib';
    if (/\(\s*g\s*\)/i.test(raw) || raw === 'g') return 'grn';
    if (raw.includes('write') || raw.includes('w/i')) return 'writein';
    if (raw.startsWith('dem') || raw.includes('democrat')) return 'dem';
    if (raw.startsWith('rep') || raw.includes('republic')) return 'rep';
    if (raw.startsWith('lib') || raw.includes('libert')) return 'lib';
    if (raw.startsWith('grn') || raw.includes('green')) return 'grn';
    if (raw.startsWith('ind') || raw.includes('independent')) return 'ind';
    if (raw.includes('nonpart') || raw.startsWith('np') || raw.includes('no party') || raw.includes('unaff')) return 'non';
    return 'other';
  }

  // ---------- Upload handling ----------
  if (el.uploadForm) {
    el.uploadForm.addEventListener('submit', e => {
      e.preventDefault();
      // JSDoc cast to satisfy FormData typing expectations in TS checks
      /** @type {HTMLFormElement|null} */
      const uploadForm = (el.uploadForm && el.uploadForm instanceof HTMLFormElement) ? el.uploadForm : null;
      const fd = uploadForm ? new FormData(uploadForm) : new FormData();

      // Optional: basic filename policy (client hint)
      const fileEntry = fd.get('csv_file');
      const file = (fileEntry instanceof File) ? fileEntry : null;
      if (file && file.name && !/\.(csv|txt|tsv|xlsx|xls)$/i.test(file.name)) {
        setStatus(el.uploadStatus, 'error', 'Only .csv, .txt, .tsv, .xls, or .xlsx files are allowed.');
        return;
      }

      setStatus(el.uploadStatus, 'info', 'Uploading...');
      (async () => {
        try {
          const fetchInit = { method: 'POST', body: fd };
          if (csrfToken) {
            const hdrs = new Headers();
            hdrs.append('X-CSRFToken', csrfToken);
            fetchInit.headers = hdrs;
          }
          
          // Use AuthUtils for certificate-aware fetch (upload requires cert)
          const winAny = (typeof window !== 'undefined') ? /** @type {any} */ (window) : null;
          const authUtils = winAny ? winAny.AuthUtils : null;
          
          let resp;
          if (authUtils && typeof authUtils.fetchWithCertHandling === 'function') {
            // Use cert-aware fetch
            resp = await authUtils.fetchWithCertHandling(
              uploadUrl,
              fetchInit,
              true,  // requiresCert
              (_url) => {
                setStatus(el.uploadStatus, 'warning', 'Client certificate required for upload');
              }
            );
          } else {
            // Fallback if AuthUtils not loaded
            resp = await fetch(uploadUrl, fetchInit);
          }
          
          const json = await resp.json().catch(() => ({ success: false, error: 'Upload endpoint did not return JSON' }));
          if (!resp.ok) {
            const errMsg = json && json.error ? json.error : `Server returned ${resp.status}`;
            setStatus(el.uploadStatus, 'error', `Upload failed: ${errMsg}`);
            showErrorToast('Upload failed.');
            return;
          }
          if (json && json.success) {
            setStatus(el.uploadStatus, 'ok', 'Upload successful!');
            showInfoToast('Upload successful.');
            fetchData(true);
          } else {
            setStatus(el.uploadStatus, 'error', `Upload failed: ${json.error || 'Unknown error'}`);
            showErrorToast('Upload failed.');
          }
        } catch (err) {
          setStatus(el.uploadStatus, 'error', `Upload failed: ${err?.message || err}`);
          showErrorToast('Upload failed.');
        }
      })();
    });
  }

  // ---------- Scaffold actions ----------
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  if (el.scaffoldJson) {
    el.scaffoldJson.addEventListener('click', () => {
      setStatus(el.status, 'info', 'Building scaffold...');
      fetch(scaffoldJsonUrl + '?limit=200', { headers: { 'Accept': 'application/json' } })
        .then(r => r.json())
        .then(data => {
          const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
          downloadBlob(blob, 'data_framework_scaffold.json');
          setStatus(el.status, 'ok', 'Scaffold JSON ready.');
          showInfoToast('Scaffold JSON downloaded.');
        })
        .catch(err => {
          setStatus(el.status, 'error', `Scaffold download failed: ${err}`);
          showErrorToast('Scaffold JSON failed.');
        });
    });
  }

  if (el.scaffoldCsv) {
    el.scaffoldCsv.addEventListener('click', () => {
      setStatus(el.status, 'info', 'Preparing scaffold CSV...');
      // navigate to server-provided scaffold CSV endpoint
      window.location.href = scaffoldCsvUrl + '?limit=200';
    });
  }

  // ---------- Column / header builders ----------
  function buildColumns() {
    const keys = rawData.length ? Object.keys(rawData[0]) : [];
    if (!visibleColumns.length) {
      const preferred = DEFAULT_VISIBLE_COLUMNS.filter(key => keys.includes(key));
      visibleColumns = preferred.length ? normalizeColumns(preferred) : normalizeColumns(keys);
    } else {
      // Re-filter existing visibleColumns against new allowlist
      visibleColumns = visibleColumns.filter(k => keys.includes(k) && isAllowedColumn(k));
    }
    buildHeader();
    buildColumnMenu();
  }

  function buildHeader() {
    // Clear existing header cells safely
    while (el.theadRow.firstChild) el.theadRow.removeChild(el.theadRow.firstChild);
    visibleColumns.forEach(key => {
      if (!allowedColumns.has(key)) return;
      const th = document.createElement('th');
      th.scope = 'col';
      th.classList.add('sortable');
      th.dataset.field = key;
      th.tabIndex = 0;
      th.setAttribute('aria-sort', sortBy === key ? sortDir : 'none');
      th.textContent = key.charAt(0).toUpperCase() + key.slice(1);

      const ind = document.createElement('span');
      ind.className = 'sort-indicator';
      th.appendChild(ind);

      const toggleSort = () => {
        if (sortBy !== key) {
          sortBy = key;
          sortDir = 'ascending';
        } else {
          sortDir = cycleSortDirection(sortDir);
          if (sortDir === 'none') sortBy = null;
        }
        [...el.theadRow.children].forEach(h => {
          const hh = /** @type {HTMLElement} */ (h);
          const fld = (hh.dataset && hh.dataset.field) ? hh.dataset.field : '';
          hh.setAttribute('aria-sort', fld === sortBy ? sortDir : 'none');
        });
        render();
      };
      th.addEventListener('click', toggleSort);
      th.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleSort();
        }
      });
      el.theadRow.appendChild(th);
    });
  }

  function buildColumnMenu() {
    if (!el.colMenu) return;
    while (el.colMenu.firstChild) el.colMenu.removeChild(el.colMenu.firstChild);
    const allKeys = Array.from(allowedColumns);
    allKeys.forEach(key => {
      const label = document.createElement('label');
      label.setAttribute('role', 'menuitemcheckbox');

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = visibleColumns.includes(key);
      cb.addEventListener('change', () => {
        if (cb.checked) {
          if (!visibleColumns.includes(key)) visibleColumns.push(key);
        } else {
          visibleColumns = visibleColumns.filter(k => k !== key);
          if (sortBy === key) { sortBy = null; sortDir = 'none'; }
        }
        buildHeader();
        render();
      });

      const span = document.createElement('span');
      span.textContent = key;
      label.appendChild(cb);
      label.appendChild(span);
      el.colMenu.appendChild(label);
    });
  }

  // ---------- Data filtering / sorting ----------
  function getFilteredSorted() {
    let data = rawData;

    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      data = data.filter(row =>
        visibleColumns.some(col => safeGet(row[col]).toLowerCase().includes(q))
      );
    }

    if (sortBy && sortDir !== 'none' && allowedColumns.has(sortBy)) {
      const dirMul = sortDir === 'descending' ? -1 : 1;
      data = [...data].sort((a, b) => {
        const av = safeGet(a[sortBy]);
        const bv = safeGet(b[sortBy]);
        const avn = parseFloat(av);
        const bvn = parseFloat(bv);
        const bothNum = av !== '' && bv !== '' && !isNaN(avn) && !isNaN(bvn);
        const cmp = bothNum
          ? (avn - bvn)
          : av.localeCompare(bv, undefined, { numeric: true, sensitivity: 'base' });
        return cmp * dirMul;
      });
    }
    return data;
  }

  // ---------- Rendering ----------
  function renderSkeleton(rows = SKELETON_ROWS) {
    while (el.tbody.firstChild) el.tbody.removeChild(el.tbody.firstChild);
    for (let i = 0; i < rows; i++) {
      const tr = document.createElement('tr');
      tr.className = 'skeleton';
      visibleColumns.forEach(() => tr.appendChild(document.createElement('td')));
      el.tbody.appendChild(tr);
    }
  }

  function render() {
    const filtered = getFilteredSorted();
    const total = filtered.length;
    const pages = Math.max(1, Math.ceil(total / pageSize));
    page = Math.min(Math.max(1, page), pages);
    const start = (page - 1) * pageSize;
    const slice = filtered.slice(start, start + pageSize);

    while (el.tbody.firstChild) el.tbody.removeChild(el.tbody.firstChild);
    if (!slice.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = Math.max(visibleColumns.length, 1);
      td.textContent = rawData.length
        ? 'No results match the current filters.'
        : 'No data to display.';
      tr.appendChild(td);
      el.tbody.appendChild(tr);
    } else {
      for (const row of slice) {
        const tr = document.createElement('tr');
        for (const col of visibleColumns) {
          const td = document.createElement('td');
          td.textContent = safeGet(row[col]);
          tr.appendChild(td);
        }
        el.tbody.appendChild(tr);
      }
    }

    el.pageInfo.textContent = `Page ${page} of ${pages} • ${total} rows`;
    if (el.first instanceof HTMLButtonElement || el.first instanceof HTMLInputElement) el.first.disabled = page <= 1;
    if (el.prev instanceof HTMLButtonElement || el.prev instanceof HTMLInputElement) el.prev.disabled = page <= 1;
    if (el.next instanceof HTMLButtonElement || el.next instanceof HTMLInputElement) el.next.disabled = page >= pages;
    if (el.last instanceof HTMLButtonElement || el.last instanceof HTMLInputElement) el.last.disabled = page >= pages;

    setStatus(el.status,
      slice.length ? 'info' : (rawData.length ? 'info' : 'error'),
      slice.length ? '' : (rawData.length ? 'No results match the current filters.' : '')
    );
  }

  // ---------- CSV helpers ----------
  function buildVisibleCsv(data) {
    const cols = visibleColumns.filter(c => allowedColumns.has(c));
    const header = cols.join(',');
    const rows = data.map(r =>
      cols.map(c => {
        let v = safeGet(r[c]).replace(/\r?\n/g, ' ').replace(/\r/g, ' ');
        v = v.replace(/"/g, '""');
        return /[",\n]/.test(v) ? `"${v}"` : v;
      }).join(',')
    );
    return [header, ...rows].join('\n');
  }

  function exportCsv() {
    const data = getFilteredSorted();
    if (data.length > MAX_ROWS_EXPORT) {
      showErrorToast(`Export capped at ${MAX_ROWS_EXPORT} rows.`);
      return;
    }
    const csv = buildVisibleCsv(data);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `export_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showInfoToast('Export started.');
  }

  function copyVisibleCsv(e) {
    e?.preventDefault();
    const data = getFilteredSorted();
    const csv = buildVisibleCsv(data);
    (navigator.clipboard?.writeText
      ? navigator.clipboard.writeText(csv)
      : new Promise((res, rej) => {
            try {
              const ta = document.createElement('textarea');
              ta.value = csv;
              // Use CSS-driven offscreen class instead of inline styles
              ta.classList.add('offscreen-temp');
              document.body.appendChild(ta);
              ta.select();
              document.execCommand('copy');
              ta.remove();
              res();
            } catch (er) { rej(er); }
          })
    ).then(() => showInfoToast('Copied CSV to clipboard.'))
     .catch(() => showErrorToast('Copy failed.'));
  }

  // ---------- Event Wiring ----------
  el.search?.addEventListener('input', debounce(e => {
    const tgt = e.target;
    if (tgt instanceof HTMLInputElement || tgt instanceof HTMLTextAreaElement) {
      searchTerm = sanitizeSearch(tgt.value);
      page = 1;
      render();
    }
  }, 150));

  el.pageSize?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement || tgt instanceof HTMLInputElement) {
      pageSize = Math.max(1, parseInt(tgt.value, 10) || 25);
      page = 1;
      render();
    }
  });

  el.first?.addEventListener('click', () => { page = 1; render(); });
  el.prev?.addEventListener('click', () => { page = Math.max(1, page - 1); render(); });
  el.next?.addEventListener('click', () => { page += 1; render(); });
  el.last?.addEventListener('click', () => {
    const total = getFilteredSorted().length;
    page = Math.max(1, Math.ceil(total / pageSize));
    render();
  });

  el.refresh?.addEventListener('click', () => fetchData(true));

  el.resetFilters?.addEventListener('click', () => {
    searchTerm = '';
    if (el.search instanceof HTMLInputElement) el.search.value = '';
    sortBy = null;
    sortDir = 'none';
    page = 1;
    render();
    showInfoToast('Filters reset.');
  });

  el.compactToggle?.addEventListener('click', () => {
    const isActive = el.compactToggle?.getAttribute('aria-pressed') === 'true';
    const next = !isActive;
    setCompactTable(next);
    compactPreferenceSet = true;
    try {
      window.localStorage?.setItem(COMPACT_TABLE_KEY, String(next));
    } catch (err) {
      // Ignore storage write errors.
    }
  });

  el.dropoffDrawerToggle?.addEventListener('click', () => {
    const next = !dropoffDrawerOpen;
    setDropoffDrawer(next);
    try {
      window.localStorage?.setItem(DROPOFF_DRAWER_KEY, String(next));
    } catch (err) {
      // Ignore storage write errors.
    }
  });

  el.dropoffDrawerOverlay?.addEventListener('click', () => {
    setDropoffDrawer(false);
    try {
      window.localStorage?.setItem(DROPOFF_DRAWER_KEY, 'false');
    } catch (err) {
      // Ignore storage write errors.
    }
  });

  // Ghost panel minimize/expand toggle
  el.ghostPanelToggle?.addEventListener('click', () => {
    const isMinimized = el.ghostPanel?.classList.toggle('is-minimized');
    if (el.ghostPanelToggle) {
      el.ghostPanelToggle.textContent = isMinimized ? '+' : '−';
      el.ghostPanelToggle.setAttribute('aria-label', isMinimized ? 'Expand placeholder' : 'Minimize placeholder');
    }
    try {
      window.localStorage?.setItem('ghostPanelMinimized', String(isMinimized));
    } catch (err) {
      // Ignore storage write errors.
    }
  });

  el.exportCsv?.addEventListener('click', exportCsv);
  el.copyVisibleCsv?.addEventListener('click', copyVisibleCsv);

  el.colBtn?.addEventListener('click', e => {
    e.stopPropagation();
    const expanded = el.colBtn.getAttribute('aria-expanded') === 'true';
    el.colBtn.setAttribute('aria-expanded', String(!expanded));
    colWrap.classList.toggle('open', !expanded);
  });

  el.priorityStateSelect?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      priorityState = tgt.value || '';
      fetchPriorityStatus();
    }
  });

  el.priorityYearSelect?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      priorityYear = tgt.value || '';
      fetchPriorityStatus();
    }
  });

  el.curatedSearch?.addEventListener('input', debounce(e => {
    const tgt = e.target;
    if (tgt instanceof HTMLInputElement || tgt instanceof HTMLTextAreaElement) {
      curatedSearch = sanitizeSearch(tgt.value);
      filterCuratedItems();
    }
  }, 150));

  el.curatedState?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      curatedState = tgt.value;
      const stateItems = curatedState ? curatedItems.filter(item => item.state === curatedState) : curatedItems;
      updateCuratedCountyOptions(stateItems);
      curatedCounty = '';
      if (el.curatedCounty instanceof HTMLSelectElement) el.curatedCounty.value = '';
      filterCuratedItems();
    }
  });

  el.curatedCounty?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      curatedCounty = tgt.value;
      filterCuratedItems();
    }
  });

  el.curatedRefresh?.addEventListener('click', () => fetchCuratedDatasets());

  // Pause auto-rotation when hovering over chart, table, OR filter dropdowns
  el.vizChart?.addEventListener('mouseenter', pauseVizAutoRotation);
  el.vizChart?.addEventListener('mouseleave', resumeVizAutoRotation);
  el.vizTable?.addEventListener('mouseenter', pauseVizAutoRotation);
  el.vizTable?.addEventListener('mouseleave', resumeVizAutoRotation);
  el.vizFilters?.addEventListener('mouseenter', pauseVizAutoRotation);
  el.vizFilters?.addEventListener('mouseleave', resumeVizAutoRotation);

  el.vizDataset?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      setVizDataset(tgt.value || VIZ_DATASET_FINALIZED);
      updateVizAutoToggleLabel();
    }
  });

  el.vizYear?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizYear(tgt.value);
      updateVizAutoToggleLabel();
    }
  });

  el.vizState?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizState(tgt.value);
      updateVizAutoToggleLabel();
    }
  });

  el.vizCounty?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizCounty(tgt.value);
      updateVizAutoToggleLabel();
    }
  });

  el.vizTopRaceCount?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizTopRaceCount = Math.max(1, parseInt(tgt.value, 10) || 5);
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      try {
        window.localStorage?.setItem(VIZ_TOP_COUNT_KEY, String(vizTopRaceCount));
      } catch (err) {
        // Ignore storage write errors.
      }
      updateTopRaces();
      refreshViz();
      updateVizAutoToggleLabel();
    }
  });

  el.vizContest?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizContest(tgt.value);
      updateVizAutoToggleLabel();
    }
  });

  el.vizTopRace?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizContest(tgt.value);
      updateVizAutoToggleLabel();
    }
  });

  el.vizParty?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizParty = tgt.value || '';
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      refreshViz();
      updateVizAutoToggleLabel();
    }
  });

  el.vizPrevStateBtn?.addEventListener('click', () => {
    vizAutoLocked = true;
    vizAutoPaused = true;
    stopVizAutoRotation();
    hideVizHint();
    stepVizState(-1);
    saveVizPlaybackPreference(true);
    updateVizAutoToggleLabel();
  });

  el.vizNextStateBtn?.addEventListener('click', () => {
    vizAutoLocked = true;
    vizAutoPaused = true;
    stopVizAutoRotation();
    hideVizHint();
    stepVizState(1);
    saveVizPlaybackPreference(true);
    updateVizAutoToggleLabel();
  });

  el.vizAutoToggleBtn?.addEventListener('click', () => {
    if (vizAutoPaused || vizAutoLocked) {
      vizAutoLocked = false;
      vizAutoPaused = false;
      hideVizHint();
      startVizAutoRotation(false);
      saveVizPlaybackPreference(false);
    } else {
      pauseVizAutoRotation();
      vizAutoLocked = true;
      saveVizPlaybackPreference(true);
    }
    updateVizAutoToggleLabel();
  });

  el.vizDropoffOverlayToggle?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLInputElement) {
      vizOverlayEnabled = !!tgt.checked;
      try {
        window.localStorage?.setItem(VIZ_OVERLAY_KEY, String(vizOverlayEnabled));
      } catch (err) {
        // Ignore storage write errors.
      }
      refreshViz();
    }
  });

  el.dropoffModeDropoff?.addEventListener('click', () => {
    el.dropoffModeDropoff?.classList.add('is-active');
    el.dropoffModeTotals?.classList.remove('is-active');
    dropoffMetric = 'raw';
    el.dropoffMetricRaw?.classList.add('is-active');
    el.dropoffMetricPercent?.classList.remove('is-active');
    refreshDropoffVisuals();
  });

  el.dropoffModeTotals?.addEventListener('click', () => {
    el.dropoffModeTotals?.classList.add('is-active');
    el.dropoffModeDropoff?.classList.remove('is-active');
    dropoffMetric = 'percent';
    el.dropoffMetricPercent?.classList.add('is-active');
    el.dropoffMetricRaw?.classList.remove('is-active');
    refreshDropoffVisuals();
  });

  el.dropoffState?.addEventListener('change', () => {
    loadDropoffData();
    if (previewActive) startPreviewCycle('active');
  });

  el.dropoffContest?.addEventListener('change', () => {
    loadDropoffData();
    if (previewActive) startPreviewCycle('active');
  });

  el.dropoffParty?.addEventListener('change', () => {
    loadDropoffData();
  });

  el.dropoffCounty?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      refreshDropoffVisuals();
    }
    if (previewActive) refreshPreview();
  });

  el.dropoffYear?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      dropoffYear = tgt.value;
      refreshDropoffVisuals();
    }
    if (previewActive) refreshPreview();
  });

  el.dropoffScale?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      dropoffScaleMode = tgt.value === 'adjusted' ? 'adjusted' : 'absolute';
      try {
        window.localStorage?.setItem(DROPOFF_SCALE_KEY, dropoffScaleMode);
      } catch (err) {
        // Ignore storage write errors.
      }
      refreshDropoffVisuals();
    }
  });

  el.dropoffCountyLimit?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      dropoffCountyLimit = tgt.value === 'all' ? 0 : Math.max(1, parseInt(tgt.value, 10) || 50);
      try {
        window.localStorage?.setItem(DROPOFF_COUNTY_LIMIT_KEY, tgt.value || '50');
      } catch (err) {
        // Ignore storage write errors.
      }
      refreshDropoffVisuals();
    }
  });

  el.dropoffOrder?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      dropoffOrderStrategy = tgt.value || 'turnout_weighted';
      try {
        window.localStorage?.setItem(DROPOFF_ORDER_KEY, dropoffOrderStrategy);
      } catch (err) {
        // Ignore storage write errors.
      }
      refreshDropoffVisuals();
    }
  });

  [el.dropoffState, el.dropoffCounty].forEach(target => {
    target?.addEventListener('focus', () => {
      setDetectionMode(true);
      if (previewActive) startPreviewCycle('active');
    });
    target?.addEventListener('blur', () => {
      setDetectionMode(false);
      if (previewActive) startPreviewCycle('idle');
    });
  });

  el.dropoffMetricPercent?.addEventListener('click', () => {
    dropoffMetric = 'percent';
    el.dropoffMetricPercent?.classList.add('is-active');
    el.dropoffMetricRaw?.classList.remove('is-active');
    el.dropoffModeTotals?.classList.add('is-active');
    el.dropoffModeDropoff?.classList.remove('is-active');
    refreshDropoffVisuals();
  });

  el.dropoffMetricRaw?.addEventListener('click', () => {
    dropoffMetric = 'raw';
    el.dropoffMetricRaw?.classList.add('is-active');
    el.dropoffMetricPercent?.classList.remove('is-active');
    el.dropoffModeDropoff?.classList.add('is-active');
    el.dropoffModeTotals?.classList.remove('is-active');
    refreshDropoffVisuals();
  });

  document.addEventListener('click', ev => {
    if (!colWrap) return;
    const tgt = ev.target;
    if (!(tgt instanceof Node) || !colWrap.contains(tgt)) {
      colWrap.classList.remove('open');
      el.colBtn?.setAttribute('aria-expanded', 'false');
    }
  });

  // ---------- Data Fetch ----------
  function fetchData(showLoading = false) {
    if (showLoading) {
      setStatus(el.status, 'info', 'Loading data...');
      renderSkeleton();
    }
    fetch(apiUrl, { headers: { 'Accept': 'application/json' } })
      .then(async r => {
        const ct = (r.headers.get('content-type') || '').toLowerCase();
        if (!r.ok) {
          const text = await r.text().catch(() => '');
            if (text.trim().startsWith('<'))
              throw new Error(`Server error ${r.status}. Received HTML.`);
          throw new Error(`Server error ${r.status}: ${text.slice(0,200)}`);
        }
        if (!ct.includes('application/json')) {
          const text = await r.text().catch(() => '');
          if (text.trim().startsWith('<')) throw new Error('Unexpected HTML response.');
          throw new Error(`Unexpected content-type: ${ct || 'unknown'}`);
        }
        return r.json().catch(e => { throw new Error(`Invalid JSON: ${e.message}`); });
      })
      .then(data => {
        rawData = Array.isArray(data)
          ? data
          : Array.isArray(data?.rows) ? data.rows
          : Array.isArray(data?.items) ? data.items
          : [];
        if (!Array.isArray(rawData)) rawData = [];

        if (!compactPreferenceSet && rawData.length) {
          setCompactTable(rawData.length >= COMPACT_AUTO_THRESHOLD);
        }

        // Trim objects with non-plain types
        rawData = rawData.map(r => {
          if (r && typeof r === 'object' && !Array.isArray(r)) return r;
          return {};
        });

        // Build allowlist / columns
        allowedColumns.clear();
        buildColumns();

        if (!rawData.length) {
          setStatus(el.status, 'error', 'No data found in the database.');
        } else {
          setStatus(el.status, 'ok', `Loaded ${rawData.length} rows.`);
        }
        fetchPriorityStatus();
        render();
      })
      .catch(err => {
        rawData = [];
        allowedColumns.clear();
        buildColumns();
        render();
        const msg = err?.message || String(err);
        if (/does not exist/i.test(msg)) {
          setStatus(el.status, 'error', 'Backend table missing. Waiting for initialization (reload shortly).');
        } else {
          setStatus(el.status, 'error', msg);
        }
        showErrorToast('Failed to load data.');
      });
  }

  // ---------- Init ----------
  initPipelineSteps();
  loadVizTopCountPreference();
  loadVizOverlayPreference();
  loadVizPlaybackPreference();
  loadCompactPreference();
  loadDropoffDrawerPreference();
  loadDropoffPreferences();
  loadGhostPanelPreference();
  if (el.vizDataset instanceof HTMLSelectElement && el.vizDataset.value) {
    vizDataset = el.vizDataset.value;
  }
  if (el.dropoffScale instanceof HTMLSelectElement && el.dropoffScale.value && !window.localStorage?.getItem(DROPOFF_SCALE_KEY)) {
    dropoffScaleMode = el.dropoffScale.value === 'adjusted' ? 'adjusted' : 'absolute';
  }
  if (el.dropoffCountyLimit instanceof HTMLSelectElement && el.dropoffCountyLimit.value && !window.localStorage?.getItem(DROPOFF_COUNTY_LIMIT_KEY)) {
    dropoffCountyLimit = el.dropoffCountyLimit.value === 'all'
      ? 0
      : Math.max(1, parseInt(el.dropoffCountyLimit.value, 10) || 50);
  }
  if (el.dropoffOrder instanceof HTMLSelectElement && el.dropoffOrder.value && !window.localStorage?.getItem(DROPOFF_ORDER_KEY)) {
    dropoffOrderStrategy = el.dropoffOrder.value || 'turnout_weighted';
  }
  if (el.vizDropoffOverlayToggle instanceof HTMLInputElement && window.localStorage?.getItem(VIZ_OVERLAY_KEY) === null) {
    vizOverlayEnabled = !!el.vizDropoffOverlayToggle.checked;
  }
  updateVizAutoToggleLabel();
  startPreviewCycle('idle');
  fetchWorklistOverview();
  fetchPriorityStatus();
  window.setInterval(fetchPriorityStatus, PRIORITY_REFRESH_MS);
  fetchCuratedDatasets();
  fetchDbLiteFinalized();
  fetchDbLiteDownBallot();
  loadDropoffData();
  fetchData(true);
});  