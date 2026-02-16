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
    vizHint: document.getElementById('vizAutoHint'),
    dropoffModeDropoff: document.getElementById('dropoffModeDropoff'),
    dropoffModeTotals: document.getElementById('dropoffModeTotals'),
    dropoffState: document.getElementById('dropoffStateInput'),
    dropoffContest: document.getElementById('dropoffContestInput'),
    dropoffCounty: document.getElementById('dropoffCountySelect'),
    dropoffYear: document.getElementById('dropoffYearSelect'),
    dropoffParty: document.getElementById('dropoffPartySelect'),
    dropoffMetricPercent: document.getElementById('dropoffMetricPercent'),
    dropoffMetricRaw: document.getElementById('dropoffMetricRaw'),
    dropoffChart: document.getElementById('dropoffChart'),
    dropoffSummary: document.getElementById('dropoffSummary'),
    dropoffControls: document.getElementById('dropoffControls'),
    dropoffPanel: document.getElementById('dropoffPanel'),
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
  let curatedItems = [];
  let curatedSelection = null;
  let curatedSearch = '';
  let curatedState = '';
  let curatedCounty = '';
  let vizRows = [];
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
  let previewActive = false;
  let previewMode = 'idle';
  let previewTimer = null;
  let compactPreferenceSet = false;
  let dropoffDrawerOpen = false;
  let dropoffData = [];
  let dropoffMetric = 'percent';
  let dropoffYear = '';
  let focusMode = 'dropoff';
  let electorTotals = [];
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
      const response = await fetch(priorityUrl, { headers: { 'Accept': 'application/json' } });
      if (!response.ok) {
        setPriorityStatus('Priority tracker unavailable.', 'error');
        return;
      }
      const payload = await response.json().catch(() => null);
      if (!payload) {
        setPriorityStatus('Priority tracker unavailable.', 'error');
        return;
      }
      if (payload.error || payload.available === false) {
        const msg = payload.error || 'Priority tracker unavailable.';
        setPriorityStatus(msg, 'error');
        return;
      }
      const summary = formatPrioritySummary(payload);
      setPriorityStatus(summary || 'Priority tracker ready.', payload.missing_total ? 'info' : 'ok');
    } catch (err) {
      setPriorityStatus('Priority tracker unavailable.', 'error');
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
    const totals = { dem: 0, rep: 0, other: 0 };
    rows.forEach(row => {
      const bucket = normalizePartyBucket(row.party);
      totals[bucket] += Number(row.votes) || 0;
    });
    const totalVotes = totals.dem + totals.rep + totals.other || 1;
    const stack = document.createElement('div');
    stack.className = 'viz-stack';

    const segments = [
      { label: 'Dem', value: totals.dem, tone: 'dem' },
      { label: 'Rep', value: totals.rep, tone: 'rep' },
      { label: 'Other', value: totals.other, tone: 'other' }
    ];
    segments.forEach(segment => {
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

    el.vizChart.appendChild(stack);
    el.vizChart.appendChild(legend);
    el.vizChart.appendChild(totalLine);
  }

  function renderVizTable(rows) {
    if (!el.vizTable) return;
    el.vizTable.innerHTML = '';
    if (!rows.length) {
      el.vizTable.innerHTML = '<div class="viz-placeholder">No rows available for this dataset.</div>';
      return;
    }
    const sorted = [...rows].sort((a, b) => {
      const aVotes = Number(a.uncategorized_votes ?? a.votes ?? 0) || 0;
      const bVotes = Number(b.uncategorized_votes ?? b.votes ?? 0) || 0;
      return bVotes - aVotes;
    });
    const table = document.createElement('table');
    table.className = 'viz-table';
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    ['County', 'Ballot Candidate Name', 'Ballot Party', 'Uncategorized Votes', 'EarlyVotes', 'Election Day Votes', 'Mail in Votes', 'Provisional Votes', 'Write-In'].forEach(label => {
      const th = document.createElement('th');
      th.textContent = label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    sorted.slice(0, 6).forEach(row => {
      const tr = document.createElement('tr');
      const cells = [
        row.county,
        row.candidate,
        row.party,
        row.uncategorized_votes ?? row.votes,
        row.early_votes,
        row.election_day_votes,
        row.mail_in_votes,
        row.provisional_votes,
        row.write_in_votes
      ];
      cells.forEach(value => {
        const td = document.createElement('td');
        if (value == null || value === '') {
          td.textContent = '—';
        } else if (typeof value === 'number') {
          td.textContent = value.toLocaleString();
        } else {
          td.textContent = String(value);
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    el.vizTable.appendChild(table);
  }

  function setVizFilters(rows) {
    vizTopRaces = [];
    // Populate Year dropdown
    const years = Array.from(new Set(rows.map(row => String(row.election_date || row.year || '')).filter(Boolean)))
      .map(value => value.slice(0, 4))
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
    const scopeRows = vizYear
      ? vizRows.filter(row => String(row.election_date || row.year || '').startsWith(vizYear))
      : vizRows;
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
    const scopeRows = vizYear && vizState
      ? vizRows.filter(row => 
          String(row.election_date || row.year || '').startsWith(vizYear) &&
          row.state === vizState
        )
      : vizYear
        ? vizRows.filter(row => String(row.election_date || row.year || '').startsWith(vizYear))
        : vizRows;
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
      ? vizRows.filter(row => String(row.election_date || row.year || '').startsWith(vizYear))
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
      filtered = filtered.filter(row => String(row.election_date || row.year || '').startsWith(vizYear));
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

  function stopVizAutoRotation() {
    if (vizAutoTimer) {
      window.clearInterval(vizAutoTimer);
      vizAutoTimer = null;
    }
  }

  function startVizAutoRotation(resetOrder = true) {
    stopVizAutoRotation();
    if (!vizTopRaces.length) return;
    if (resetOrder || !vizAutoOrder.length) {
      vizAutoOrder = shuffleArray([...vizTopRaces]);
      vizAutoIndex = 0;
    }
    setVizContest(vizAutoOrder[vizAutoIndex]);
    vizAutoTimer = window.setInterval(() => {
      if (vizAutoLocked) return;
      vizAutoIndex = (vizAutoIndex + 1) % vizAutoOrder.length;
      setVizContest(vizAutoOrder[vizAutoIndex]);
    }, 6000);
  }

  function pauseVizAutoRotation() {
    if (vizAutoLocked) return;
    vizAutoPaused = true;
    stopVizAutoRotation();
    if (el.vizHint) el.vizHint.classList.add('is-visible');
  }

  function hideVizHint() {
    if (el.vizHint) el.vizHint.classList.remove('is-visible');
  }

  function resumeVizAutoRotation() {
    if (vizAutoLocked || !vizAutoPaused) return;
    vizAutoPaused = false;
    startVizAutoRotation(false);
    hideVizHint();
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

  async function updateVisualizationFromCurated(item) {
    if (!item || !apiUrl) {
      clearVisualization();
      return;
    }
    vizAutoLocked = false;
    const url = buildWarehouseUrl({
      state: item.state,
      county: item.county,
      contest: item.contest,
      year: item.year,
      data_source: 'live',
      limit: 150
    });
    try {
      const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
      const contentType = (response.headers.get('content-type') || '').toLowerCase();
      if (!response.ok || !contentType.includes('application/json')) {
        clearVisualization();
        return;
      }
      const payload = await response.json().catch(() => null);
      vizRows = Array.isArray(payload?.items)
        ? payload.items
        : Array.isArray(payload?.rows)
          ? payload.rows
          : [];
      setVizFilters(vizRows);
      refreshViz();
    } catch (e) {
      clearVisualization();
    }
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
        if (!curatedItems.length) {
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

  function renderDropoffChart(row) {
    if (!el.dropoffChart) return;
    el.dropoffChart.innerHTML = '';
    if (!row) {
      el.dropoffChart.textContent = 'No drop-off data available.';
      return;
    }

    const values = [
      { label: 'Dem', value: dropoffMetric === 'percent' ? row.dem_pct_dropoff : row.dem_dropoff, tone: 'dem' },
      { label: 'Rep', value: dropoffMetric === 'percent' ? row.rep_pct_dropoff : row.rep_dropoff, tone: 'rep' },
      { label: 'Total', value: dropoffMetric === 'percent' ? row.total_pct_dropoff : row.total_dropoff, tone: 'total' }
    ];
    const maxAbs = Math.max(1, ...values.map(v => Math.abs(Number(v.value) || 0)));
    values.forEach(entry => {
      const bar = document.createElement('div');
      bar.className = `dropoff-bar dropoff-${entry.tone}`;
      const label = document.createElement('div');
      label.className = 'dropoff-bar-label';
      label.textContent = entry.label;
      const value = document.createElement('div');
      value.className = 'dropoff-bar-value';
      const numeric = Number(entry.value) || 0;
      const display = dropoffMetric === 'percent' ? `${numeric.toFixed(2)}%` : `${numeric}`;
      value.textContent = display;
      const fill = document.createElement('div');
      const fillPct = Math.min(100, Math.round((Math.abs(numeric) / maxAbs) * 100));
      fill.className = `dropoff-bar-fill${numeric < 0 ? ' is-negative' : ''}`;
      fill.style.setProperty('--bar-fill', `${fillPct}%`);
      bar.appendChild(label);
      bar.appendChild(fill);
      bar.appendChild(value);
      el.dropoffChart.appendChild(bar);
    });
  }

  function renderElectorChart(totals) {
    if (!el.dropoffChart) return;
    el.dropoffChart.innerHTML = '';
    if (!totals) {
      el.dropoffChart.textContent = 'No elector totals available.';
      return;
    }
    const values = [
      { label: 'Dem', value: totals.dem || 0, tone: 'dem' },
      { label: 'Rep', value: totals.rep || 0, tone: 'rep' },
      { label: 'Other', value: totals.other || 0, tone: 'total' }
    ];
    const maxAbs = dropoffMetric === 'percent' && totals.total
      ? 100
      : Math.max(1, ...values.map(v => Math.abs(Number(v.value) || 0)));
    values.forEach(entry => {
      const bar = document.createElement('div');
      bar.className = `dropoff-bar dropoff-${entry.tone}`;
      const label = document.createElement('div');
      label.className = 'dropoff-bar-label';
      label.textContent = entry.label;
      const value = document.createElement('div');
      value.className = 'dropoff-bar-value';
      const numeric = Number(entry.value) || 0;
      const display = dropoffMetric === 'percent' && totals.total
        ? `${((numeric / totals.total) * 100).toFixed(2)}%`
        : `${numeric}`;
      value.textContent = display;
      const fill = document.createElement('div');
      const baseValue = dropoffMetric === 'percent' && totals.total
        ? Math.abs((numeric / totals.total) * 100)
        : Math.abs(numeric);
      const fillPct = Math.min(100, Math.round((baseValue / maxAbs) * 100));
      fill.className = 'dropoff-bar-fill';
      fill.style.setProperty('--bar-fill', `${fillPct}%`);
      bar.appendChild(label);
      bar.appendChild(fill);
      bar.appendChild(value);
      el.dropoffChart.appendChild(bar);
    });
  }

  function updateDropoffSummary(row) {
    if (!el.dropoffSummary) return;
    if (focusMode === 'elector_totals') {
      if (!row) {
        el.dropoffSummary.textContent = 'Select a county to see elector totals.';
        return;
      }
      const suffix = dropoffMetric === 'percent' ? '%' : '';
      const demVal = dropoffMetric === 'percent' && row.total
        ? ((row.dem / row.total) * 100).toFixed(2)
        : row.dem;
      const repVal = dropoffMetric === 'percent' && row.total
        ? ((row.rep / row.total) * 100).toFixed(2)
        : row.rep;
      const otherVal = dropoffMetric === 'percent' && row.total
        ? ((row.other / row.total) * 100).toFixed(2)
        : row.other;
      el.dropoffSummary.textContent = `Totals • Dem ${demVal}${suffix} • Rep ${repVal}${suffix} • Other ${otherVal}${suffix}`;
      return;
    }
    if (!row) {
      el.dropoffSummary.textContent = 'Select a county to see drop-off totals.';
      return;
    }
    const demVal = dropoffMetric === 'percent' ? row.dem_pct_dropoff : row.dem_dropoff;
    const repVal = dropoffMetric === 'percent' ? row.rep_pct_dropoff : row.rep_dropoff;
    const totalVal = dropoffMetric === 'percent' ? row.total_pct_dropoff : row.total_dropoff;
    const suffix = dropoffMetric === 'percent' ? '%' : '';
    el.dropoffSummary.textContent = `${row.county} County • Dem ${demVal}${suffix} • Rep ${repVal}${suffix} • Total ${totalVal}${suffix}`;
  }

  function loadDropoffData() {
    fetchDropoffFromApi().then(ok => {
      if (ok) {
        hydrateDropoffSelectors(dropoffData);
        return;
      }
      dropoffData = [];
      renderDropoffChart(null);
      updateDropoffSummary(null);
    });
  }

  function hydrateDropoffSelectors(rows) {
    const years = Array.from(new Set(rows.map(row => String(row.year || '')).filter(Boolean)))
      .sort((a, b) => Number(b) - Number(a));
    if (el.dropoffYear instanceof HTMLSelectElement) {
      el.dropoffYear.innerHTML = '';
      years.forEach(year => {
        const opt = document.createElement('option');
        opt.value = year;
        opt.textContent = year;
        el.dropoffYear.appendChild(opt);
      });
      dropoffYear = years[0] || '';
      if (dropoffYear) el.dropoffYear.value = dropoffYear;
    }
    const filteredRows = getDropoffRowsForYear(dropoffYear);
    updateDropoffCountyOptions(filteredRows);
    const first = filteredRows[0];
    if (first) {
      if (el.dropoffCounty instanceof HTMLSelectElement) {
        el.dropoffCounty.value = first.county;
      }
      renderDropoffChart(first);
      updateDropoffSummary(first);
    } else {
      renderDropoffChart(null);
      updateDropoffSummary(null);
    }
  }

  function loadElectorTotalsData() {
    const contest = getSelectedDropoffContest();
    if (!contest) {
      renderElectorChart(null);
      updateDropoffSummary(null);
      return;
    }
    fetchElectorTotalsFromApi().then(ok => {
      if (!ok) {
        electorTotals = [];
        renderElectorChart(null);
        updateDropoffSummary(null);
        return;
      }
      const totalsMap = buildElectorTotalsMap(electorTotals);
      const years = Object.keys(totalsMap).sort((a, b) => Number(b) - Number(a));
      if (el.dropoffYear instanceof HTMLSelectElement) {
        el.dropoffYear.innerHTML = '';
        years.forEach(year => {
          const opt = document.createElement('option');
          opt.value = year;
          opt.textContent = year;
          el.dropoffYear.appendChild(opt);
        });
        dropoffYear = years[0] || '';
        if (dropoffYear) el.dropoffYear.value = dropoffYear;
      }
      const counties = totalsMap[dropoffYear] ? Object.keys(totalsMap[dropoffYear]).sort() : [];
      updateDropoffCountyOptions(counties.map(county => ({ county })));
      const firstCounty = counties[0];
      if (firstCounty && el.dropoffCounty instanceof HTMLSelectElement) {
        el.dropoffCounty.value = firstCounty;
      }
      const selection = firstCounty && totalsMap[dropoffYear] ? totalsMap[dropoffYear][firstCounty] : null;
      renderElectorChart(selection || null);
      updateDropoffSummary(selection || null);
    });
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
    if (raw.startsWith('dem')) return 'dem';
    if (raw.startsWith('rep')) return 'rep';
    return 'other';
  }

  function getDropoffRowsForYear(year) {
    if (!year) return dropoffData;
    return dropoffData.filter(item => String(item.year || '') === String(year));
  }

  function updateDropoffCountyOptions(rows) {
    if (!(el.dropoffCounty instanceof HTMLSelectElement)) return;
    el.dropoffCounty.innerHTML = '';
    rows.forEach(row => {
      const opt = document.createElement('option');
      opt.value = row.county;
      opt.textContent = row.county;
      el.dropoffCounty.appendChild(opt);
    });
  }

  function buildWarehouseUrl(params) {
    const url = new URL(apiUrl, window.location.origin);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && String(value).trim() !== '') {
        url.searchParams.set(key, String(value));
      }
    });
    return url.toString();
  }

  async function fetchDropoffFromApi() {
    if (!apiUrl) return false;
    const state = getSelectedDropoffState();
    const contest = getSelectedDropoffContest() || 'Senate';
    if (!state) return false;
    const url = buildWarehouseUrl({
      metric: 'dropoff',
      state,
      contest,
      data_source: 'live'
    });
    try {
      const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (!response.ok) return false;
      const data = await response.json().catch(() => null);
      if (!data || !Array.isArray(data.items)) return false;
      dropoffData = data.items;
      return true;
    } catch (e) {
      return false;
    }
  }

  async function fetchElectorTotalsFromApi() {
    if (!apiUrl) return false;
    const state = getSelectedDropoffState();
    const contest = getSelectedDropoffContest();
    const party = getSelectedDropoffParty();
    if (!state || !contest) return false;
    const url = buildWarehouseUrl({
      metric: 'elector_totals',
      state,
      contest,
      party,
      data_source: 'live'
    });
    try {
      const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (!response.ok) return false;
      const data = await response.json().catch(() => null);
      if (!data || !Array.isArray(data.items)) return false;
      electorTotals = data.items;
      return true;
    } catch (e) {
      return false;
    }
  }

  function buildElectorTotalsMap(items) {
    const map = {};
    items.forEach(item => {
      const year = String(item.year || '');
      const county = item.county || '';
      if (!year || !county) return;
      map[year] = map[year] || {};
      const entry = map[year][county] || { dem: 0, rep: 0, other: 0, total: 0 };
      const bucket = normalizePartyBucket(item.party);
      const votes = Number(item.votes) || 0;
      entry[bucket] += votes;
      entry.total += votes;
      map[year][county] = entry;
    });
    return map;
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
              (url) => {
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
      visibleColumns = normalizeColumns(keys);
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

  el.exportCsv?.addEventListener('click', exportCsv);
  el.copyVisibleCsv?.addEventListener('click', copyVisibleCsv);

  el.colBtn?.addEventListener('click', e => {
    e.stopPropagation();
    const expanded = el.colBtn.getAttribute('aria-expanded') === 'true';
    el.colBtn.setAttribute('aria-expanded', String(!expanded));
    colWrap.classList.toggle('open', !expanded);
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

  el.vizChart?.addEventListener('mouseenter', pauseVizAutoRotation);
  el.vizChart?.addEventListener('mouseleave', resumeVizAutoRotation);
  el.vizTable?.addEventListener('mouseenter', pauseVizAutoRotation);
  el.vizTable?.addEventListener('mouseleave', resumeVizAutoRotation);

  el.vizYear?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizYear(tgt.value);
    }
  });

  el.vizState?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizState(tgt.value);
    }
  });

  el.vizCounty?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizCounty(tgt.value);
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
    }
  });

  el.vizContest?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizContest(tgt.value);
    }
  });

  el.vizTopRace?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      vizAutoLocked = true;
      stopVizAutoRotation();
      hideVizHint();
      setVizContest(tgt.value);
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
    }
  });

  el.dropoffModeDropoff?.addEventListener('click', () => {
    focusMode = 'dropoff';
    el.dropoffModeDropoff?.classList.add('is-active');
    el.dropoffModeTotals?.classList.remove('is-active');
    loadDropoffData();
  });

  el.dropoffModeTotals?.addEventListener('click', () => {
    focusMode = 'elector_totals';
    el.dropoffModeTotals?.classList.add('is-active');
    el.dropoffModeDropoff?.classList.remove('is-active');
    loadElectorTotalsData();
  });

  el.dropoffState?.addEventListener('change', () => {
    if (focusMode === 'elector_totals') {
      loadElectorTotalsData();
    } else {
      loadDropoffData();
    }
    if (previewActive) startPreviewCycle('active');
  });

  el.dropoffContest?.addEventListener('change', () => {
    if (focusMode === 'elector_totals') {
      loadElectorTotalsData();
    } else {
      loadDropoffData();
    }
    if (previewActive) startPreviewCycle('active');
  });

  el.dropoffParty?.addEventListener('change', () => {
    if (focusMode === 'elector_totals') {
      loadElectorTotalsData();
    }
  });

  el.dropoffCounty?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      const year = getSelectedDropoffYear();
      if (focusMode === 'elector_totals') {
        const totalsMap = buildElectorTotalsMap(electorTotals);
        const selection = totalsMap[year]?.[tgt.value] || null;
        renderElectorChart(selection);
        updateDropoffSummary(selection);
      } else {
        const rows = getDropoffRowsForYear(year);
        const row = rows.find(item => item.county === tgt.value);
        renderDropoffChart(row || null);
        updateDropoffSummary(row || null);
      }
    }
    if (previewActive) refreshPreview();
  });

  el.dropoffYear?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      dropoffYear = tgt.value;
      if (focusMode === 'elector_totals') {
        const totalsMap = buildElectorTotalsMap(electorTotals);
        const counties = totalsMap[dropoffYear] ? Object.keys(totalsMap[dropoffYear]).sort() : [];
        updateDropoffCountyOptions(counties.map(county => ({ county })));
        const firstCounty = counties[0] || null;
        if (firstCounty && el.dropoffCounty instanceof HTMLSelectElement) {
          el.dropoffCounty.value = firstCounty;
        }
        const selection = firstCounty ? totalsMap[dropoffYear]?.[firstCounty] : null;
        renderElectorChart(selection);
        updateDropoffSummary(selection);
      } else {
        const rows = getDropoffRowsForYear(dropoffYear);
        updateDropoffCountyOptions(rows);
        const first = rows[0] || null;
        if (first && el.dropoffCounty instanceof HTMLSelectElement) {
          el.dropoffCounty.value = first.county;
        }
        renderDropoffChart(first);
        updateDropoffSummary(first);
      }
    }
    if (previewActive) refreshPreview();
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
    const selectedYear = getSelectedDropoffYear();
    if (focusMode === 'elector_totals') {
      const totalsMap = buildElectorTotalsMap(electorTotals);
      const selected = getSelectedDropoffCounty();
      const selection = selected ? totalsMap[selectedYear]?.[selected] : null;
      renderElectorChart(selection);
      updateDropoffSummary(selection);
    } else {
      const rows = getDropoffRowsForYear(selectedYear);
      const selected = getSelectedDropoffCounty();
      const current = selected ? rows.find(item => item.county === selected) : rows[0];
      renderDropoffChart(current || null);
      updateDropoffSummary(current || null);
    }
  });

  el.dropoffMetricRaw?.addEventListener('click', () => {
    dropoffMetric = 'raw';
    el.dropoffMetricRaw?.classList.add('is-active');
    el.dropoffMetricPercent?.classList.remove('is-active');
    const selectedYear = getSelectedDropoffYear();
    if (focusMode === 'elector_totals') {
      const totalsMap = buildElectorTotalsMap(electorTotals);
      const selected = getSelectedDropoffCounty();
      const selection = selected ? totalsMap[selectedYear]?.[selected] : null;
      renderElectorChart(selection);
      updateDropoffSummary(selection);
    } else {
      const rows = getDropoffRowsForYear(selectedYear);
      const selected = getSelectedDropoffCounty();
      const current = selected ? rows.find(item => item.county === selected) : rows[0];
      renderDropoffChart(current || null);
      updateDropoffSummary(current || null);
    }
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
  loadCompactPreference();
  loadDropoffDrawerPreference();
  startPreviewCycle('idle');
  fetchPriorityStatus();
  window.setInterval(fetchPriorityStatus, PRIORITY_REFRESH_MS);
  fetchCuratedDatasets();
  loadDropoffData();
  fetchData(true);
});  