/**
 * Data Framework UI (refactored with defensive guards & client-side hardening)
 * NOTE: Real SQL injection mitigation must occur server-side via parameterized queries.
 * This client enforces:
 *  - Column name allowlisting (alphanumeric + underscore)
 *  - Sanitized search term (length + control char stripping)
 *  - Strict sort direction cycling / no arbitrary query fragments
 *  - Safe CSV generation (quotes + CR/LF normalized)
 *
 * G2.3.1: preserve jurisdiction name/type through canonical visualization,
 *         keep county-specific tooling explicit, and clarify cached priority metadata.
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
    hydratedUrl ||  // server injects the configured publication endpoint
    ((/** @type {any} */ (window)).__DATA_FRAMEWORK__ && (/** @type {any} */ (window)).__DATA_FRAMEWORK__.apiUrl) ||
    '/api/ballotlens-database';
  // Additional injectable endpoints
  const uploadUrl = cfgEl?.dataset?.uploadUrl || '/upload/input';
  const scaffoldJsonUrl = cfgEl?.dataset?.scaffoldJsonUrl || '/api/data_framework/scaffold';
  const scaffoldCsvUrl = cfgEl?.dataset?.scaffoldCsvUrl || '/api/data_framework/scaffold.csv';
  const curatedUrl = cfgEl?.dataset?.curatedUrl || '/api/data_framework/curated';
  const priorityUrl = cfgEl?.dataset?.priorityUrl || '/api/data_framework/warehouse_status';
  const canonicalFacetsUrl =
    cfgEl?.dataset?.canonicalFacetsUrl || '/api/data_framework/canonical_facets';

  // G3.1C2: Data Framework election-result consumers are canonical-only.
  // Transitional DB-Lite / worklist / legacy preview endpoint identifiers are retired.
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
    vizPlaybackBadge: document.getElementById('vizPlaybackBadge'),
    vizScopeBadge: document.getElementById('vizScopeBadge'),
    vizPreviewStatus: document.getElementById('vizPreviewStatus'),
    vizCountyScopeBadge: document.getElementById('vizCountyScopeBadge'),
    dropoffDrawer: document.getElementById('dropoffDrawer'),
    dropoffDrawerToggle: document.getElementById('dropoffDrawerToggle'),
    dropoffDrawerOverlay: document.getElementById('dropoffDrawerOverlay'),
    vizYear: document.getElementById('vizYearSelect'),
    vizState: document.getElementById('vizStateSelect'),
    vizCounty: document.getElementById('vizCountySelect'),
    vizContest: document.getElementById('vizContestSelect'),
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
    pipelineDetail: document.getElementById('uploadPipelineDetail'),
    readOnlyBanner: document.getElementById('dataFrameworkReadOnlyBanner'),
    readOnlyMessage: document.getElementById('dataFrameworkReadOnlyMessage'),
    evidenceContextBar: document.getElementById('evidenceContextBar'),
    evidenceContextTitle: document.getElementById('evidenceContextTitle'),
    evidenceContextScope: document.getElementById('evidenceContextScope'),
    evidenceContextAnalysis: document.getElementById('evidenceContextAnalysis'),
    evidenceContextReview: document.getElementById('evidenceContextReview'),
    evidenceContextCanonical: document.getElementById('evidenceContextCanonical'),
    analysisContextStatus: document.getElementById('analysisContextStatus'),
    reviewContextStatus: document.getElementById('reviewContextStatus'),
    canonicalContextStatus: document.getElementById('canonicalContextStatus')
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

  let curatedItems = [];
  let curatedSelection = null;
  let curatedSearch = '';
  let curatedState = '';
  let curatedCounty = '';
  let vizRows = [];
  let warehouseVizRows = [];
  // Stable unfiltered canonical pool survives scoped Explore requests.
  let canonicalPreviewRows = [];
  let canonicalFacetPayload = null;
  let canonicalFacetUniversePayload = null;
  let canonicalFacetRequestSeq = 0;
  let canonicalFacetAbortController = null;
  let canonicalDataRequestSeq = 0;
  let canonicalDataAbortController = null;
  // Canonical Record is an independent consumer surface. It must never share
  // request cancellation or scope ownership with Analysis.
  let canonicalRecordRequestSeq = 0;
  let canonicalRecordAbortController = null;
  // Canonical Record selector availability is independent from Analysis scope.
  let canonicalRecordFacetPayload = null;
  let canonicalRecordFacetRequestSeq = 0;
  let canonicalRecordFacetAbortController = null;
  let analysisRowsPossiblyTruncated = false;
  let initialCanonicalQueryScope = null;
  let canonicalQueryScopeHydrated = false;

  let vizDataset = 'warehouse_core';
  let vizYear = '';
  let vizState = '';
  let vizCounty = '';
  let vizContest = '';
  let vizAutoTimer = null;
  let vizTopRaces = [];
  let vizAutoIndex = 0;
  let vizAutoLocked = false;
  let vizTopRaceCount = 5;
  let vizAutoOrder = [];
  let vizAutoPaused = false;
  let vizHoverPaused = false;
  // G2.4.3B1: Preview playback and operator Explore scope are distinct states.
  let vizInteractionMode = 'preview';
  let vizOverlayEnabled = false;
  let vizStatusBase = '';
  let vizUiState = 'idle';
  let previewActive = false;
  let previewMode = 'idle';
  let previewTimer = null;
  let priorityTimer = null;
  let authRestrictedMode = false;
  let _authRestrictionReason = '';
  let authRestrictionNotified = false;
  let canonicalRecordBaseStatus = {
    tone: 'info',
    state: 'idle',
    text: ''
  };
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
  const COMPACT_TABLE_KEY = 'df_table_compact';
  const COMPACT_AUTO_THRESHOLD = 300;
  // G3.1C1.6: canonical publication currently clamps result reads to 1,000 rows.
  // Hitting this limit must be presented as a bounded subset, never as completeness.
  const CANONICAL_CLIENT_ROW_LIMIT = 1000;
  const PRIORITY_REFRESH_MS = 60000;
  const DROPOFF_DRAWER_KEY = 'df_dropoff_drawer';
  const VIZ_OVERLAY_KEY = 'df_viz_overlay_enabled';
  const VIZ_PLAYBACK_PAUSED_KEY = 'df_viz_playback_paused';
  const DROPOFF_ORDER_KEY = 'df_dropoff_order_strategy';
  const DROPOFF_SCALE_KEY = 'df_dropoff_scale_mode';
  const DROPOFF_COUNTY_LIMIT_KEY = 'df_dropoff_county_limit';
  const VIZ_FILTER_SNAPSHOT_KEY = 'df_viz_filter_snapshot_v1';
  const WAREHOUSE_STATUS_SNAPSHOT_KEY = 'df_warehouse_status_snapshot_v1';
  const VIZ_MODE_HELP_DISMISSED_KEY = 'df_viz_mode_help_dismissed_v1';
  const VIZ_DATASET_WAREHOUSE = 'warehouse_core';
  const VIZ_INTERACTION_PREVIEW = 'preview';
  const VIZ_INTERACTION_EXPLORE = 'explore';
  const SHAREABLE_CANONICAL_QUERY_MAX_LEN = 240;
  const SHAREABLE_CANONICAL_QUERY_KEYS = Object.freeze({
    year: 'year',
    state: 'state',
    jurisdiction: 'jurisdiction',
    contest: 'contest',
  });
  const DEFAULT_VISIBLE_COLUMNS = ['state', 'jurisdiction_name', 'jurisdiction_type', 'contest', 'candidate', 'party', 'votes'];
  const COLUMN_LABELS = {
    jurisdiction_name: 'Jurisdiction',
    jurisdiction_type: 'Jurisdiction Type',
    total_votes: 'Total Votes',
    election_year: 'Election Year',
    date_precision: 'Date Precision',
    aggregation_scope: 'Aggregation Scope',
  };

  function isAuthForbiddenStatus(status) {
    return status === 401 || status === 403;
  }

  function shouldRetryStatus(status) {
    return [408, 425, 429, 500, 502, 503, 504].includes(Number(status || 0));
  }

  function sleep(ms) {
    return new Promise(resolve => window.setTimeout(resolve, ms));
  }

  /**
   * @typedef {Object} FetchRetryOptions
   * @property {string=} authReason
   * @property {number=} retries
   * @property {number=} baseDelayMs
   * @property {Record<string, string>=} headers
   */

  /**
   * @param {string} url
   * @param {FetchRetryOptions=} options
   */
  async function fetchJsonWithRetry(url, {
    authReason,
    retries = 2,
    baseDelayMs = 450,
    headers = { 'Accept': 'application/json' },
    signal = undefined
  } = {}) {
    let lastError = null;
    let lastStatus = null;

    for (let attempt = 0; attempt <= retries; attempt += 1) {
      try {
        const response = await fetch(url, { headers, signal });
        lastStatus = response.status;
        if (isAuthForbiddenStatus(response.status)) {
          enterAuthRestrictedMode(authReason || 'Authentication required for protected Data Framework endpoints.');
          return { ok: false, authBlocked: true, status: response.status, data: null, error: 'auth_forbidden' };
        }

        if (response.ok) {
          const data = await response.json().catch(() => null);
          return { ok: true, authBlocked: false, status: response.status, data, error: null };
        }

        if (!(attempt < retries && shouldRetryStatus(response.status))) {
          break;
        }
      } catch (err) {
        if (err && err.name === 'AbortError') {
          return {
            ok: false,
            authBlocked: false,
            aborted: true,
            status: lastStatus,
            data: null,
            error: 'aborted'
          };
        }
        lastError = err;
        if (!(attempt < retries)) {
          break;
        }
      }

      const backoff = Math.min(baseDelayMs * (2 ** attempt) + Math.floor(Math.random() * 180), 2600);
      await sleep(backoff);
    }

    return {
      ok: false,
      authBlocked: false,
      status: lastStatus,
      data: null,
      error: lastError ? String(lastError?.message || lastError) : 'request_failed'
    };
  }

  function enterAuthRestrictedMode(reason = 'Authentication required for protected Data Framework endpoints.') {
    authRestrictedMode = true;
    _authRestrictionReason = reason;
    if (el.readOnlyBanner) {
      if (el.readOnlyMessage) {
        el.readOnlyMessage.textContent = `Read-only mode: ${reason}`;
      }
      el.readOnlyBanner.classList.remove('d-none');
      setUiState(el.readOnlyBanner, 'restricted');
    }
    if (previewTimer) {
      window.clearInterval(previewTimer);
      previewTimer = null;
    }
    if (priorityTimer) {
      window.clearInterval(priorityTimer);
      priorityTimer = null;
    }
    previewActive = false;
    previewMode = 'idle';
    setPreviewState(false);
    setPreviewStatus(
      'Read-only mode: authenticate to enable curated and canonical analysis feeds.',
      'restricted'
    );
    setPriorityStatus(
      'Read-only mode: priority tracker requires authentication.',
      'info',
      'restricted'
    );
    if (el.curatedStatus) {
      setStatusText(
        el.curatedStatus,
        'Read-only mode: authenticate to load curated datasets.',
        'restricted'
      );
    }
    if (!authRestrictionNotified) {
      showInfoToast(reason);
      authRestrictionNotified = true;
    }
  }

  // ---------- Utilities ----------
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
  const safeGet = v => (v == null ? '' : String(v));
  const displayValue = v => (v == null ? '—' : String(v));
  const exportValue = v => (v == null ? 'NULL' : String(v));

  // G2.4.1: propagate selected evidence context without claiming unsupported lineage.
  function setContextText(target, text) {
    if (!target) return;
    target.textContent = text || '';
  }

  function setContextTone(target, tone) {
    if (!target) return;
    target.setAttribute('data-context-tone', tone || 'idle');
  }

  function setRelationState(target, state, text) {
    if (!target) return;
    const relation = target.closest('.evidence-relation');
    if (relation) relation.setAttribute('data-state', state || 'idle');
    target.textContent = text || '';
  }

  function setFlowStepState(stepName, state) {
    const step = document.querySelector(`[data-flow-step="${stepName}"]`);
    if (step) step.setAttribute('data-flow-state', state || 'idle');
  }

  function getEvidenceDatasetLabel() {
    return 'Canonical Production';
  }

  function getEvidenceScopeParts(item) {
    if (!item) return [];
    const parts = [];
    if (item.state) parts.push(`State ${item.state}`);
    if (item.county) parts.push(`Jurisdiction ${item.county}`);
    if (item.contest) parts.push(`Contest ${item.contest}`);
    if (item.year) parts.push(`Metadata year ${item.year}`);
    return parts;
  }

  function getEvidenceMatchAxes(item) {
    if (!item) return [];
    const axes = [];
    if (item.state) axes.push('state');
    if (item.county) axes.push('jurisdiction');
    if (item.contest) axes.push('contest');
    return axes;
  }

  function resetEvidenceRelationshipContext() {
    if (el.evidenceContextBar) el.evidenceContextBar.setAttribute('data-context-state', 'idle');
    setContextText(el.evidenceContextTitle, 'No source evidence selected');
    setContextText(
      el.evidenceContextScope,
      'Select a Source Evidence item to align supported workbench context. No lineage is inferred.'
    );
    setRelationState(el.evidenceContextAnalysis, 'idle', 'Waiting for selection');
    setRelationState(el.evidenceContextReview, 'unlinked', 'Not linked');
    setRelationState(el.evidenceContextCanonical, 'unlinked', 'Lineage not established');

    setContextTone(el.analysisContextStatus, 'idle');
    setContextText(
      el.analysisContextStatus?.querySelector('span:last-child'),
      'No source selected; the current analysis view is independent.'
    );
    setContextTone(el.reviewContextStatus, 'unlinked');
    setContextText(
      el.reviewContextStatus?.querySelector('span:last-child'),
      'No selected source is linked to this review pipeline. Upload and publication actions remain separate governed operations.'
    );
    setContextTone(el.canonicalContextStatus, 'unlinked');
    setContextText(
      el.canonicalContextStatus?.querySelector('span:last-child'),
      'No source-to-canonical lineage is established. Canonical rows remain independently governed.'
    );

    setFlowStepState('source', 'idle');
    setFlowStepState('analysis', 'idle');
    setFlowStepState('review', 'unlinked');
    setFlowStepState('authority', 'unlinked');
    setFlowStepState('canonical', 'unlinked');
  }

  function updateEvidenceRelationshipContext(item, analysisResult = {}) {
    if (!item) {
      resetEvidenceRelationshipContext();
      return;
    }

    const title = item.title || item.contest || 'Selected source evidence';
    const scopeParts = getEvidenceScopeParts(item);
    const analysisStatus = analysisResult.status || 'no-match';
    const analysisCount = Number(analysisResult.count || 0);
    const axes = Array.isArray(analysisResult.axes) ? analysisResult.axes : getEvidenceMatchAxes(item);
    const axesText = axes.length ? axes.join(', ') : 'available scope fields';

    if (el.evidenceContextBar) el.evidenceContextBar.setAttribute('data-context-state', 'selected');
    setContextText(el.evidenceContextTitle, title);
    setContextText(
      el.evidenceContextScope,
      scopeParts.length
        ? `${scopeParts.join(' • ')} • Context only; source-to-canonical lineage is not established.`
        : 'Selected evidence has no usable scope fields. Context only; lineage is not established.'
    );

    setFlowStepState('source', 'selected');
    setFlowStepState('review', 'unlinked');
    setFlowStepState('authority', 'unlinked');
    setFlowStepState('canonical', 'unlinked');

    if (analysisStatus === 'context-match') {
      setRelationState(
        el.evidenceContextAnalysis,
        'context-match',
        `Context match only (${analysisCount.toLocaleString()} rows)`
      );
      setFlowStepState('analysis', 'context-match');
      setContextTone(el.analysisContextStatus, 'context-match');
      setContextText(
        el.analysisContextStatus?.querySelector('span:last-child'),
        `Selected evidence aligns with ${analysisCount.toLocaleString()} ${getEvidenceDatasetLabel()} rows by ${axesText}. This is contextual alignment, not provenance lineage.`
      );
    } else if (analysisStatus === 'feed-unavailable') {
      setRelationState(el.evidenceContextAnalysis, 'no-match', 'Analysis feed unavailable');
      setFlowStepState('analysis', 'no-match');
      setContextTone(el.analysisContextStatus, 'no-match');
      setContextText(
        el.analysisContextStatus?.querySelector('span:last-child'),
        `Selected evidence is retained as context, but ${getEvidenceDatasetLabel()} is not currently available.`
      );
    } else {
      setRelationState(el.evidenceContextAnalysis, 'no-match', 'No scope match');
      setFlowStepState('analysis', 'no-match');
      setContextTone(el.analysisContextStatus, 'no-match');
      setContextText(
        el.analysisContextStatus?.querySelector('span:last-child'),
        `No ${getEvidenceDatasetLabel()} rows match the selected evidence by ${axesText}. The existing analysis view remains independent.`
      );
    }

    setRelationState(el.evidenceContextReview, 'unlinked', 'Not linked');
    setContextTone(el.reviewContextStatus, 'unlinked');
    setContextText(
      el.reviewContextStatus?.querySelector('span:last-child'),
      `Selected evidence: ${title}. No direct source-to-review key is established; upload and review actions do not automatically attach this source.`
    );

    setRelationState(el.evidenceContextCanonical, 'unlinked', 'Lineage not established');
    setContextTone(el.canonicalContextStatus, 'unlinked');
    setContextText(
      el.canonicalContextStatus?.querySelector('span:last-child'),
      `Selected evidence: ${title}. Canonical lineage is not established; the PostgreSQL rows below remain independently governed.`
    );
  }

  function firstPresent(...values) {
    for (const value of values) {
      if (value !== null && value !== undefined && value !== '') return value;
    }
    return null;
  }

  function firstNumeric(...values) {
    for (const value of values) {
      const parsed = parseNumeric(value);
      if (parsed !== null) return parsed;
    }
    return null;
  }
  const UI_STATES = new Set([
    'idle',
    'loading',
    'ready',
    'empty',
    'restricted',
    'error'
  ]);

  function normalizeUiState(state) {
    return UI_STATES.has(state) ? state : 'idle';
  }

  function stateForTone(type) {
    if (type === 'ok') return 'ready';
    if (type === 'error') return 'error';
    return 'idle';
  }

  function setUiState(target, state = 'idle', text = undefined) {
    if (!target) return;
    const normalized = normalizeUiState(state);
    target.dataset.uiState = normalized;
    target.setAttribute('aria-busy', normalized === 'loading' ? 'true' : 'false');
    if (text !== undefined) {
      target.textContent = text || '';
    }
  }

  function setStatus(target, type, text, state = null) {
    if (!target) return;
    target.className = 'status ' + (type === 'ok' ? 'status-ok' : type === 'error' ? 'status-error' : 'status-info');
    setUiState(target, state || stateForTone(type), text);
  }

  function setCanonicalRecordBaseStatus(type, text, state = null) {
    canonicalRecordBaseStatus = {
      tone: type,
      state: normalizeUiState(state || stateForTone(type)),
      text: text || ''
    };
    setStatus(
      el.status,
      canonicalRecordBaseStatus.tone,
      canonicalRecordBaseStatus.text,
      canonicalRecordBaseStatus.state
    );
  }

  function restoreCanonicalRecordBaseStatus() {
    setStatus(
      el.status,
      canonicalRecordBaseStatus.tone,
      canonicalRecordBaseStatus.text,
      canonicalRecordBaseStatus.state
    );
  }
  function sanitizeSearch(raw) {
    if (!raw) return '';
    let s = raw.slice(0, MAX_SEARCH_LEN);
    // Strip control chars except basic whitespace
    s = s.replace(/[\x00-\x08\x0B-\x1F\x7F]/g, '');
    return s.trim();
  }
  function parseNumeric(value) {
    if (value == null || value === '') return null;
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    const cleaned = String(value).replace(/,/g, '').replace(/[^\d.-]/g, '').trim();
    if (!cleaned || cleaned === '-' || cleaned === '.' || cleaned === '-.') return null;
    const num = Number(cleaned);
    return Number.isFinite(num) ? num : null;
  }
  function parsePercent(value) {
    if (value == null || value === '') return null;
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    const cleaned = String(value).replace('%', '').trim();
    if (!cleaned) return null;
    const num = Number(cleaned);
    return Number.isFinite(num) ? num : null;
  }

  function compareNullableNumbersDesc(left, right) {
    const a = parseNumeric(left);
    const b = parseNumeric(right);
    if (a === null && b === null) return 0;
    if (a === null) return 1;
    if (b === null) return -1;
    return b - a;
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

  function setStatusText(target, text, state = 'idle') {
    setUiState(target, state, text);
  }

  function setPriorityStatus(text, tone = 'info', state = null) {
    if (!el.warehousePriorityStatus) return;
    el.warehousePriorityStatus.className = `warehouse-status-strip status status-${tone}`;
    setUiState(
      el.warehousePriorityStatus,
      state || stateForTone(tone === 'ok' ? 'ok' : tone),
      text
    );
  }

  function setPriorityMeta(text) {
    if (!el.warehousePriorityMeta) return;
    el.warehousePriorityMeta.textContent = text || '';
  }

  // Warehouse status is priority metadata only. Canonical selectors are
  // populated from canonical facet authority, never from warehouse coverage.

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
    return [yearText, divText].filter(Boolean).join(' | ');
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

  function readWarehouseStatusSnapshot() {
    try {
      const raw = window.localStorage?.getItem(WAREHOUSE_STATUS_SNAPSHOT_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object') return null;
      return parsed;
    } catch (err) {
      return null;
    }
  }

  function writeWarehouseStatusSnapshot(payload) {
    if (!payload || typeof payload !== 'object') return;
    try {
      const snapshot = {
        payload,
        captured_at: new Date().toISOString(),
      };
      window.localStorage?.setItem(WAREHOUSE_STATUS_SNAPSHOT_KEY, JSON.stringify(snapshot));
    } catch (err) {
      // Ignore storage write errors.
    }
  }

  function applyPriorityPayload(payload, fromCache = false) {
    if (!payload) return false;
    if (payload.error || payload.available === false) {
      const msg = payload.error || 'Priority tracker unavailable.';
      setPriorityStatus(msg, 'error');
      setPriorityMeta('');
      return false;
    }
    // Do not let warehouse-status availability mutate Canonical Record scope.
    lastPriorityPayload = payload;
    const summary = formatPrioritySummary(payload);
    const statusText = fromCache
      ? `${summary || 'Priority snapshot loaded.'} (cached)`
      : (summary || 'Priority tracker ready.');
    setPriorityStatus(
      statusText,
      payload.missing_total ? 'info' : 'ok',
      'ready'
    );
    const baseMeta = formatPriorityMeta(payload);
    const suffix = fromCache ? ' | Priority metadata: cached snapshot' : '';
    setPriorityMeta(`${baseMeta}${suffix}`);
    return true;
  }

  async function fetchPriorityStatus() {
    if (!priorityUrl) return;
    if (authRestrictedMode) return;
    try {
      const url = new URL(priorityUrl, window.location.origin);
      if (priorityState) url.searchParams.set('state', priorityState);
      if (priorityYear) url.searchParams.set('year', priorityYear);
      const result = await fetchJsonWithRetry(url.toString(), {
        authReason: 'Authentication required for Data Framework priority and preview APIs.',
        retries: 2
      });
      if (result.authBlocked) return;
      if (!result.ok) {
        const cached = readWarehouseStatusSnapshot();
        if (cached?.payload && applyPriorityPayload(cached.payload, true)) return;
        setPriorityStatus('Priority tracker unavailable.', 'error');
        setPriorityMeta('');
        return;
      }
      const payload = result.data;
      if (!payload) {
        const cached = readWarehouseStatusSnapshot();
        if (cached?.payload && applyPriorityPayload(cached.payload, true)) return;
        setPriorityStatus('Priority tracker unavailable.', 'error');
        setPriorityMeta('');
        return;
      }
      writeWarehouseStatusSnapshot(payload);
      applyPriorityPayload(payload, false);
    } catch (err) {
      const cached = readWarehouseStatusSnapshot();
      if (cached?.payload && applyPriorityPayload(cached.payload, true)) return;
      setPriorityStatus('Priority tracker unavailable.', 'error');
      setPriorityMeta('');
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
    // G2.4.3B1.2: a fresh page load always begins in Preview auto.
    // Pause/Explore state is intentionally local to this page instance.
    vizAutoPaused = false;
    try {
      window.localStorage?.removeItem(VIZ_PLAYBACK_PAUSED_KEY);
    } catch (err) {
      // Ignore storage cleanup errors.
    }
  }

  function setVizModeHelpCollapsed(collapsed, persist = true) {
    const helpEl = document.getElementById('vizModeHelp');
    if (!helpEl) return;
    helpEl.classList.toggle('is-collapsed', !!collapsed);
    if (!persist) return;
    try {
      window.localStorage?.setItem(VIZ_MODE_HELP_DISMISSED_KEY, collapsed ? 'true' : 'false');
    } catch (err) {
      // Ignore storage write errors.
    }
  }

  function loadVizModeHelpPreference() {
    try {
      const dismissed = window.localStorage?.getItem(VIZ_MODE_HELP_DISMISSED_KEY) === 'true';
      setVizModeHelpCollapsed(dismissed, false);
    } catch (err) {
      setVizModeHelpCollapsed(false, false);
    }
  }

  function saveVizPlaybackPreference(_paused) {
    // Playback state is intentionally not persisted across reloads.
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
    curatedSelection = item || null;
    if (el.curatedRows) el.curatedRows.textContent = item?.row_count != null ? String(item.row_count) : '—';
    if (el.curatedColumns) el.curatedColumns.textContent = item?.column_count != null ? String(item.column_count) : '—';
    if (el.curatedUpdated) el.curatedUpdated.textContent = item?.updated_at || '—';
    if (el.curatedMeta) {
      const parts = [item?.contest, item?.state, item?.county].filter(Boolean).join(' • ');
      el.curatedMeta.textContent = item
        ? (parts || 'No additional metadata available.')
        : 'Pick a dataset to see its summary and source links.';
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

    if (!item) {
      resetEvidenceRelationshipContext();
      return;
    }

    const analysisResult = updateVisualizationFromCurated(item);
    updateEvidenceRelationshipContext(item, analysisResult);
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
    let knownVoteValues = 0;
    rows.forEach(row => {
      const bucket = normalizePartyBucket(row.party || row.ballot_party || '');
      const votes = parseNumeric(row.votes);
      if (votes === null) return;
      knownVoteValues += 1;
      totals[bucket] += votes;
    });

    if (!knownVoteValues) {
      el.vizChart.innerHTML = '<div class="viz-placeholder">Vote totals are not reported for the selected rows.</div>';
      return;
    }

    const totalVotes = Object.values(totals).reduce((sum, val) => sum + val, 0);
    const stack = document.createElement('div');
    stack.className = 'viz-stack';
    const segments = buckets
      .map(entry => ({ label: entry.label, value: totals[entry.key], tone: entry.key }))
      .filter(segment => segment.value > 0);

    if (totalVotes > 0) {
      segments.forEach(segment => {
        const seg = document.createElement('div');
        seg.className = `viz-stack-seg viz-stack-${segment.tone}`;
        seg.style.setProperty('--seg-size', `${Math.round((segment.value / totalVotes) * 100)}%`);
        seg.title = `${segment.label}: ${segment.value.toLocaleString()}`;
        stack.appendChild(seg);
      });
    }

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
      .filter(item => parseNumeric(item.delta_pct) !== null)
      .slice()
      .sort((a, b) => Math.abs(b.delta_pct) - Math.abs(a.delta_pct))
      .slice(0, 36);
    if (!points.length) return;
    const maxAbs = Math.max(1, ...points.map(item => Math.abs(item.delta_pct)));
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
      const y = 60 - Math.round((row.delta_pct / maxAbs) * 55);
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

    const sorted = [...rows].sort((a, b) => compareNullableNumbersDesc(a.votes, b.votes));
    const table = document.createElement('table');
    table.className = 'viz-table';
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    const headers = ['Jurisdiction', 'Type', 'Dem Votes', 'Rep Votes', 'Other Votes', 'Write-In Votes', 'Uncategorized Votes', 'Total Votes'];
    headers.forEach(label => {
      const th = document.createElement('th');
      th.textContent = label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    const renderValue = value => {
      if (value == null || value === '') return '\u2014';
      if (typeof value === 'number') return value.toLocaleString();
      return String(value);
    };

    const hasJurisdictionFocus = !!vizCounty || new Set(
      sorted.map(row => normalizeCountyKey(getVizJurisdictionName(row)))
    ).size <= 1;

    if (hasJurisdictionFocus) {
      headRow.innerHTML = '';
      ['Candidate', 'Party', 'Votes', 'Jurisdiction', 'Type'].forEach(label => {
        const th = document.createElement('th');
        th.textContent = label;
        headRow.appendChild(th);
      });

      sorted
        .slice()
        .sort((a, b) => compareNullableNumbersDesc(a.votes, b.votes))
        .slice(0, 18)
        .forEach(row => {
          const tr = document.createElement('tr');
          const partyBucket = normalizePartyBucket(row.party || row.ballot_party || '');
          const partyLabel = partyBucket === 'writein'
            ? 'Write-In'
            : (row.party || row.ballot_party || 'Other');
          const cells = [
            row.candidate || 'Unspecified Candidate',
            partyLabel,
            parseNumeric(row.votes),
            getVizJurisdictionName(row) || '\u2014',
            getVizJurisdictionType(row) || null,
          ];
          cells.forEach(value => {
            const td = document.createElement('td');
            td.textContent = renderValue(value);
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });

      table.appendChild(tbody);
      el.vizTable.appendChild(table);
      return;
    }

    const grouped = new Map();
    sorted.forEach(row => {
      const jurisdictionLabel = String(getVizJurisdictionName(row) || '\u2014').trim() || '\u2014';
      const jurisdictionType = getVizJurisdictionType(row);
      const jurisdictionKey = `${normalizeCountyKey(jurisdictionLabel) || '\u2014'}::${normalizeCountyKey(jurisdictionType)}`;
      const entry = grouped.get(jurisdictionKey) || {
        jurisdiction: jurisdictionLabel,
        jurisdiction_type: jurisdictionType,
        dem: 0,
        rep: 0,
        other: 0,
        writein: 0,
        uncategorized: 0,
        total: 0,
        dem_missing: false,
        rep_missing: false,
        other_missing: false,
        writein_missing: false,
        uncategorized_missing: false,
        total_missing: false,
      };

      const bucket = normalizePartyBucket(row.party || row.ballot_party || '');
      const bucketKey = ['dem', 'rep', 'writein'].includes(bucket) ? bucket : 'other';
      const votes = parseNumeric(row.votes);
      const uncategorized = parseNumeric(row.uncategorized_votes);

      if (votes === null) {
        entry[`${bucketKey}_missing`] = true;
        entry.total_missing = true;
      } else {
        entry[bucketKey] += votes;
        entry.total += votes;
      }

      if (uncategorized === null) {
        entry.uncategorized_missing = true;
      } else {
        entry.uncategorized += uncategorized;
      }

      grouped.set(jurisdictionKey, entry);
    });

    Array.from(grouped.values())
      .sort((a, b) => b.total - a.total)
      .slice(0, 12)
      .forEach(row => {
        const tr = document.createElement('tr');
        const cells = [
          row.jurisdiction,
          row.jurisdiction_type || null,
          row.dem_missing ? null : row.dem,
          row.rep_missing ? null : row.rep,
          row.other_missing ? null : row.other,
          row.writein_missing ? null : row.writein,
          row.uncategorized_missing ? null : row.uncategorized,
          row.total_missing ? null : row.total
        ];
        cells.forEach(value => {
          const td = document.createElement('td');
          td.textContent = renderValue(value);
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });

    table.appendChild(tbody);
    el.vizTable.appendChild(table);
  }
  function setVizOverlayEnabled(enabled, persist = true) {
    vizOverlayEnabled = !!enabled;
    if (el.vizDropoffOverlayToggle instanceof HTMLInputElement) {
      el.vizDropoffOverlayToggle.checked = vizOverlayEnabled;
    }
    if (!persist) return;
    try {
      window.localStorage?.setItem(VIZ_OVERLAY_KEY, String(vizOverlayEnabled));
    } catch (err) {
      // Ignore storage write errors.
    }
  }

  function syncVizOverlayAvailability() {
    // Governed canonical drop-off derivation is not published yet.
    // Keep the dormant overlay unavailable without retaining DB-Lite authority.
    if (el.vizDropoffOverlayToggle instanceof HTMLInputElement) {
      el.vizDropoffOverlayToggle.disabled = true;
    }
    if (vizOverlayEnabled) {
      setVizOverlayEnabled(false);
    }
  }

  function getUniqueValues(rows, getter) {
    return Array.from(new Set((rows || []).map(getter).filter(Boolean))).sort((a, b) => String(a).localeCompare(String(b)));
  }

  function normalizeCountyKey(value) {
    return String(value || '').trim().toLowerCase().replace(/\s+/g, ' ');
  }

  function getVizJurisdictionName(row) {
    if (!row || typeof row !== 'object') return '';
    return String(firstPresent(row.jurisdiction_name, '') || '').trim();
  }

  function getVizJurisdictionType(row) {
    if (!row || typeof row !== 'object') return '';
    return String(firstPresent(row.jurisdiction_type, '') || '').trim();
  }

  function getUniqueCountyValues(rows, getter) {
    const byKey = new Map();
    (rows || []).forEach(row => {
      const rawValue = String(getter(row) || '').trim();
      if (!rawValue) return;
      const normalizedKey = normalizeCountyKey(rawValue);
      if (!normalizedKey) return;
      if (!byKey.has(normalizedKey)) {
        byKey.set(normalizedKey, rawValue);
      }
    });
    return Array.from(byKey.values()).sort((a, b) => String(a).localeCompare(String(b)));
  }

  function getRowsForYear(rows, year) {
    if (!year) return rows;
    return rows.filter(row => getRowYear(row) === year);
  }

  function getRowsForYearState(rows, year, state) {
    const byYear = getRowsForYear(rows, year);
    if (!state) return byYear;
    return byYear.filter(row => String(row.state || '') === String(state));
  }

  function getRowsForYearStateCounty(rows, year, state, county) {
    const byYearState = getRowsForYearState(rows, year, state);
    if (!county) return byYearState;
    return byYearState.filter(row => getVizJurisdictionName(row) === String(county));
  }

  function setSelectOptions(selectEl, values, preferredValue = '', allowEmpty = false, emptyLabel = 'All') {
    if (!(selectEl instanceof HTMLSelectElement)) return '';
    const normalizedValues = Array.from(new Set((values || []).filter(Boolean).map(v => String(v))));
    const previous = preferredValue || selectEl.value || '';
    selectEl.innerHTML = '';

    if (allowEmpty) {
      const emptyOpt = document.createElement('option');
      emptyOpt.value = '';
      emptyOpt.textContent = emptyLabel;
      selectEl.appendChild(emptyOpt);
    }

    normalizedValues.forEach(value => {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = value;
      selectEl.appendChild(opt);
    });

    const fallback = allowEmpty ? '' : (normalizedValues[0] || '');
    const nextValue = normalizedValues.includes(previous) ? previous : fallback;
    if (nextValue) {
      selectEl.value = nextValue;
    } else if (allowEmpty) {
      selectEl.value = '';
    }
    selectEl.disabled = !normalizedValues.length && !allowEmpty;
    return selectEl.value || '';
  }

  function readVizSnapshots() {
    try {
      const raw = window.localStorage?.getItem(VIZ_FILTER_SNAPSHOT_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (err) {
      return {};
    }
  }

  function writeVizSnapshot(dataset, payload) {
    if (!dataset || !payload) return;
    try {
      const snapshots = readVizSnapshots();
      snapshots[dataset] = {
        ...payload,
        updatedAt: new Date().toISOString(),
      };
      window.localStorage?.setItem(VIZ_FILTER_SNAPSHOT_KEY, JSON.stringify(snapshots));
    } catch (err) {
      // Ignore storage write errors.
    }
  }

  function getVizSnapshot(dataset) {
    const snapshots = readVizSnapshots();
    return snapshots && typeof snapshots === 'object' ? snapshots[dataset] : null;
  }

  function hydrateVizFiltersFromSnapshot(dataset) {
    const snapshot = getVizSnapshot(dataset);
    if (!snapshot) return false;
    const options = snapshot.options || {};
    const selection = snapshot.selection || {};

    if (el.vizYear instanceof HTMLSelectElement) {
      vizYear = setSelectOptions(el.vizYear, options.years || [], selection.vizYear);
    }
    if (el.vizState instanceof HTMLSelectElement) {
      vizState = setSelectOptions(el.vizState, options.states || [], selection.vizState);
    }
    if (el.vizCounty instanceof HTMLSelectElement) {
      vizCounty = setSelectOptions(el.vizCounty, options.counties || [], selection.vizCounty, true, 'All jurisdictions');
    }
    if (el.vizContest instanceof HTMLSelectElement) {
      vizContest = setSelectOptions(el.vizContest, options.contests || [], selection.vizContest);
    }
    return true;
  }

  function saveCurrentVizSnapshot(rows) {
    if (!Array.isArray(rows) || !rows.length) return;
    const years = Array.from(new Set(rows.map(row => getRowYear(row)).filter(Boolean))).sort((a, b) => Number(b) - Number(a));
    const states = getUniqueValues(getRowsForYear(rows, vizYear), row => row.state);
    const counties = getUniqueValues(getRowsForYearState(rows, vizYear, vizState), row => getVizJurisdictionName(row));
    const contests = getUniqueValues(getRowsForYearStateCounty(rows, vizYear, vizState, vizCounty), row => row.contest);

    writeVizSnapshot(vizDataset, {
      rowCount: rows.length,
      options: {
        years,
        states,
        counties,
        contests,
        topRaces: Array.isArray(vizTopRaces) ? [...vizTopRaces] : [],
      },
      selection: {
        vizYear,
        vizState,
        vizCounty,
        vizContest,
      },
    });
  }

  

  function sanitizeCanonicalQueryValue(value) {
    if (value === null || value === undefined) return '';
    const normalized = String(value).trim();
    if (
      !normalized
      || normalized.length > SHAREABLE_CANONICAL_QUERY_MAX_LEN
      || /[\x00-\x1F\x7F]/.test(normalized)
    ) {
      return '';
    }
    return normalized;
  }

  function readCanonicalQueryScopeFromLocation() {
    const params = new URLSearchParams(window.location.search || '');
    return {
      year: sanitizeCanonicalQueryValue(
        params.get(SHAREABLE_CANONICAL_QUERY_KEYS.year)
      ),
      state: sanitizeCanonicalQueryValue(
        params.get(SHAREABLE_CANONICAL_QUERY_KEYS.state)
      ),
      jurisdiction: sanitizeCanonicalQueryValue(
        params.get(SHAREABLE_CANONICAL_QUERY_KEYS.jurisdiction)
      ),
      contest: sanitizeCanonicalQueryValue(
        params.get(SHAREABLE_CANONICAL_QUERY_KEYS.contest)
      ),
    };
  }

  function hasCanonicalQueryScope(filters) {
    return !!(
      filters?.year
      || filters?.state
      || filters?.jurisdiction
      || filters?.contest
    );
  }

  function applyInitialCanonicalQueryScope() {
    initialCanonicalQueryScope = readCanonicalQueryScopeFromLocation();
    if (!hasCanonicalQueryScope(initialCanonicalQueryScope)) return false;

    // Query values are only intent at this point. They are validated against
    // canonicalFacetUniversePayload before the scoped result GET is allowed.
    vizYear = initialCanonicalQueryScope.year;
    vizState = initialCanonicalQueryScope.state;
    vizCounty = initialCanonicalQueryScope.jurisdiction;
    vizContest = initialCanonicalQueryScope.contest;

    // A shareable scope is operator Explore state, never Preview playback.
    // Do not persist this as a playback preference merely because a URL was opened.
    vizInteractionMode = VIZ_INTERACTION_EXPLORE;
    vizAutoLocked = true;
    vizAutoPaused = true;
    vizHoverPaused = false;
    previewActive = false;
    previewMode = 'idle';
    return true;
  }

  function replaceCanonicalQueryScopeInLocation(
    filters = getCanonicalScopeFilters()
  ) {
    if (!canonicalQueryScopeHydrated || !window.history?.replaceState) return;

    const url = new URL(window.location.href);
    const values = {
      [SHAREABLE_CANONICAL_QUERY_KEYS.year]: filters.year || '',
      [SHAREABLE_CANONICAL_QUERY_KEYS.state]: filters.state || '',
      [SHAREABLE_CANONICAL_QUERY_KEYS.jurisdiction]:
        filters.jurisdiction || '',
      [SHAREABLE_CANONICAL_QUERY_KEYS.contest]: filters.contest || '',
    };

    Object.entries(values).forEach(([key, value]) => {
      if (value) url.searchParams.set(key, value);
      else url.searchParams.delete(key);
    });

    const nextLocation = `${url.pathname}${url.search}${url.hash}`;
    const currentLocation =
      `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextLocation !== currentLocation) {
      window.history.replaceState(window.history.state, '', nextLocation);
    }
  }

  function clearCanonicalQueryScopeFromLocation() {
    if (!window.history?.replaceState) return;
    const url = new URL(window.location.href);
    Object.values(SHAREABLE_CANONICAL_QUERY_KEYS)
      .forEach(key => url.searchParams.delete(key));

    const nextLocation = `${url.pathname}${url.search}${url.hash}`;
    const currentLocation =
      `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextLocation !== currentLocation) {
      window.history.replaceState(window.history.state, '', nextLocation);
    }
  }

  // Curated Source Evidence is intentionally not serialized until its API
  // publishes a stable dataset identifier suitable for cross-surface handoff.

  function getCanonicalScopeFilters() {
    return {
      year: vizYear || '',
      state: vizState || '',
      jurisdiction: vizCounty || '',
      contest: vizContest || '',
    };
  }

  function hasCanonicalScope(filters = getCanonicalScopeFilters()) {
    return !!(
      filters.year
      || filters.state
      || filters.jurisdiction
      || filters.contest
    );
  }

  function buildCanonicalFacetUrl(filters = getCanonicalScopeFilters()) {
    const url = new URL(canonicalFacetsUrl, window.location.origin);
    if (filters.year) url.searchParams.set('year', filters.year);
    if (filters.state) url.searchParams.set('state', filters.state);
    if (filters.jurisdiction) url.searchParams.set('jurisdiction', filters.jurisdiction);
    if (filters.contest) url.searchParams.set('contest', filters.contest);
    return url.toString();
  }

  function buildCanonicalDataUrl(filters = getCanonicalScopeFilters()) {
    const url = new URL(apiUrl, window.location.origin);
    url.searchParams.set('limit', '1000');
    if (filters.year) url.searchParams.set('year', filters.year);
    if (filters.state) url.searchParams.set('state', filters.state);
    if (filters.jurisdiction) url.searchParams.set('jurisdiction', filters.jurisdiction);
    if (filters.contest) url.searchParams.set('contest', filters.contest);
    return url.toString();
  }

  function buildCanonicalRecordDataUrl() {
    const url = new URL(apiUrl, window.location.origin);
    url.searchParams.set('limit', String(CANONICAL_CLIENT_ROW_LIMIT));
    // Canonical Record owns its visible State / Year selectors independently
    // from Analysis' vizYear / vizState / vizJurisdiction / vizContest scope.
    if (priorityYear) url.searchParams.set('year', priorityYear);
    if (priorityState) url.searchParams.set('state', priorityState);
    return url.toString();
  }

  function isCanonicalFacetPayload(payload) {
    return !!(
      payload
      && payload.contract === 'canonical_facets_v1'
      && payload.data_source === 'canonical'
      && payload.authority === 'canonical_production'
      && payload.filter_model === 'bidirectional_faceted'
      && payload.semantic_contract?.facet_mode === 'self_excluding'
      && payload.semantic_contract?.lineage === 'not_inferred'
      && payload.semantic_contract?.null === 'preserved_null'
      && payload.semantic_contract?.no_warehouse_fallback === true
      && Array.isArray(payload.years)
      && Array.isArray(payload.states)
      && Array.isArray(payload.jurisdictions)
      && Array.isArray(payload.contests)
    );
  }

  function replaceCanonicalOptions(
    select,
    universeEntries,
    availableEntries,
    selected,
    allLabel,
    valueOf,
    labelOf,
    decorate
  ) {
    if (!(select instanceof HTMLSelectElement)) return selected || '';

    const desired = String(selected || '');
    const universe = Array.isArray(universeEntries) ? universeEntries : [];
    const available = new Set(
      (Array.isArray(availableEntries) ? availableEntries : [])
        .map(entry => String(valueOf(entry) || ''))
        .filter(Boolean)
    );
    const valid = new Set();

    while (select.firstChild) select.removeChild(select.firstChild);

    const all = document.createElement('option');
    all.value = '';
    all.textContent = allLabel;
    all.dataset.availability = 'available';
    select.appendChild(all);

    universe.forEach(entry => {
      const value = String(valueOf(entry) || '');
      if (!value || valid.has(value)) return;
      valid.add(value);

      const option = document.createElement('option');
      const isAvailable = available.has(value);
      const baseLabel = labelOf(entry) || value;

      option.value = value;
      option.dataset.availability = isAvailable ? 'available' : 'unavailable';
      option.textContent = isAvailable
        ? baseLabel
        : `${baseLabel} — no current match`;

      // Preserve a currently-selected valid no-result scope so the user can
      // see and clear it. Other valid-but-unavailable options remain visible
      // but use the native disabled presentation.
      option.disabled = !isAvailable && value !== desired;

      if (!isAvailable) {
        option.title = 'Valid canonical option; no rows match the other active filters.';
      }
      if (decorate) decorate(option, entry);
      select.appendChild(option);
    });

    // Only values outside the canonical universe are invalid and cleared.
    const resolved = desired && valid.has(desired) ? desired : '';
    select.value = resolved;
    return resolved;
  }

  function canonicalJurisdictionOptions(entries) {
    const grouped = new Map();
    (Array.isArray(entries) ? entries : []).forEach(entry => {
      if (!entry || typeof entry !== 'object') return;
      const name = String(entry.name || '').trim();
      const type = String(entry.type || '').trim();
      if (!name) return;
      if (!grouped.has(name)) grouped.set(name, new Set());
      if (type) grouped.get(name).add(type);
    });
    return Array.from(grouped.entries())
      .map(([name, types]) => ({ name, types: Array.from(types).sort() }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }

  function applyCanonicalFacetPayload(payload) {
    if (!isCanonicalFacetPayload(payload)) return false;
    canonicalFacetPayload = payload;

    const universePayload = isCanonicalFacetPayload(canonicalFacetUniversePayload)
      ? canonicalFacetUniversePayload
      : payload;

    vizYear = replaceCanonicalOptions(
      el.vizYear,
      universePayload.years,
      payload.years,
      vizYear,
      'All years',
      value => String(value),
      value => String(value)
    );
    vizState = replaceCanonicalOptions(
      el.vizState,
      universePayload.states,
      payload.states,
      vizState,
      'All states',
      value => String(value),
      value => String(value)
    );

    const universeJurisdictions = canonicalJurisdictionOptions(
      universePayload.jurisdictions
    );
    const availableJurisdictions = canonicalJurisdictionOptions(
      payload.jurisdictions
    );
    vizCounty = replaceCanonicalOptions(
      el.vizCounty,
      universeJurisdictions,
      availableJurisdictions,
      vizCounty,
      'All jurisdictions',
      entry => entry.name,
      entry => entry.types.length
        ? `${entry.name} — ${entry.types.join(' / ')}`
        : entry.name,
      (option, entry) => {
        option.dataset.jurisdictionTypes = entry.types.join('|');
      }
    );

    vizContest = replaceCanonicalOptions(
      el.vizContest,
      universePayload.contests,
      payload.contests,
      vizContest,
      'All contests',
      value => String(value),
      value => String(value)
    );

    updateVizModeBadges();
    renderVizStatus();
    return true;
  }

  function getCanonicalRecordFacetFilters() {
    return {
      year: priorityYear || '',
      state: priorityState || '',
      jurisdiction: '',
      contest: '',
    };
  }

  function applyCanonicalRecordFacetPayload(payload) {
    if (!isCanonicalFacetPayload(payload)) return false;
    canonicalRecordFacetPayload = payload;

    const universePayload = isCanonicalFacetPayload(canonicalFacetUniversePayload)
      ? canonicalFacetUniversePayload
      : payload;

    priorityYear = replaceCanonicalOptions(
      el.priorityYearSelect,
      universePayload.years,
      payload.years,
      priorityYear,
      'All years',
      value => String(value),
      value => String(value)
    );
    priorityState = replaceCanonicalOptions(
      el.priorityStateSelect,
      universePayload.states,
      payload.states,
      priorityState,
      'All states',
      value => String(value),
      value => String(value)
    );

    return true;
  }

  async function fetchCanonicalRecordFacets({ useUniverse = false } = {}) {
    if (
      useUniverse
      && isCanonicalFacetPayload(canonicalFacetUniversePayload)
    ) {
      return applyCanonicalRecordFacetPayload(
        canonicalFacetUniversePayload
      );
    }

    const requestSeq = ++canonicalRecordFacetRequestSeq;

    if (canonicalRecordFacetAbortController) {
      canonicalRecordFacetAbortController.abort();
    }
    canonicalRecordFacetAbortController = new AbortController();

    const result = await fetchJsonWithRetry(
      buildCanonicalFacetUrl(getCanonicalRecordFacetFilters()),
      {
        authReason: 'Authentication required for Canonical Record facets.',
        retries: 2,
        signal: canonicalRecordFacetAbortController.signal,
      }
    );

    if (
      result?.aborted
      || requestSeq !== canonicalRecordFacetRequestSeq
    ) {
      return false;
    }
    if (result?.authBlocked || !result?.ok) return false;
    if (!isCanonicalFacetPayload(result.data)) return false;

    return applyCanonicalRecordFacetPayload(result.data);
  }

  async function fetchCanonicalFacets({ universe = false } = {}) {
    const requestSeq = ++canonicalFacetRequestSeq;
    if (canonicalFacetAbortController) canonicalFacetAbortController.abort();
    canonicalFacetAbortController = new AbortController();

    const filters = universe
      ? { year: '', state: '', jurisdiction: '', contest: '' }
      : getCanonicalScopeFilters();

    const result = await fetchJsonWithRetry(buildCanonicalFacetUrl(filters), {
      authReason: 'Authentication required for canonical Data Framework facets.',
      retries: 2,
      signal: canonicalFacetAbortController.signal,
    });

    if (result?.aborted || requestSeq !== canonicalFacetRequestSeq) return false;
    if (result?.authBlocked || !result?.ok) return false;

    if (!isCanonicalFacetPayload(result.data)) {
      setPreviewStatus('Canonical facet response rejected — authority contract mismatch.');
      return false;
    }

    if (universe || !hasCanonicalScope(filters)) {
      canonicalFacetUniversePayload = result.data;
    }
    return applyCanonicalFacetPayload(result.data);
  }

  function refreshCanonicalExploreScope() {
    if (vizInteractionMode !== VIZ_INTERACTION_EXPLORE) return;
    fetchCanonicalFacets();
    fetchData(false);
  }

  function setVizFilters(rows) {
    if (canonicalFacetPayload && applyCanonicalFacetPayload(canonicalFacetPayload)) {
      return;
    }
    vizTopRaces = [];
    const computedYears = Array.from(new Set(rows.map(row => getRowYear(row)).filter(Boolean))).sort((a, b) => Number(b) - Number(a));
    const years = computedYears;
    if (el.vizYear instanceof HTMLSelectElement) {
      vizYear = setSelectOptions(el.vizYear, years, vizYear);
    } else if (!years.includes(vizYear)) {
      vizYear = years[0] || '';
    }

    updateVizStates();

    if (vizInteractionMode === VIZ_INTERACTION_PREVIEW && !vizAutoLocked && !vizAutoPaused && !vizHoverPaused) {
      startVizAutoRotation();
    }

    syncVizOverlayAvailability();
    saveCurrentVizSnapshot(rows);
  }

  function updateVizStates() {
    // Canonical facets own selector validity. The bounded result page is only
    // rendering data and must never collapse the State universe.
    const facetPayload = isCanonicalFacetPayload(canonicalFacetPayload)
      ? canonicalFacetPayload
      : canonicalFacetUniversePayload;
    const universePayload = isCanonicalFacetPayload(canonicalFacetUniversePayload)
      ? canonicalFacetUniversePayload
      : facetPayload;

    if (
      el.vizState instanceof HTMLSelectElement
      && isCanonicalFacetPayload(facetPayload)
      && isCanonicalFacetPayload(universePayload)
    ) {
      vizState = replaceCanonicalOptions(
        el.vizState,
        universePayload.states,
        facetPayload.states,
        vizState,
        'All states',
        value => String(value),
        value => String(value)
      );
    } else if (!(el.vizState instanceof HTMLSelectElement)) {
      // Non-DOM fallback only. This does not mutate the browser selector.
      const scopeRows = getRowsForYear(vizRows, vizYear);
      const states = getUniqueValues(scopeRows, row => row.state);
      if (!states.includes(vizState)) {
        vizState = states[0] || '';
      }
    }

    updateVizCounties();
  }

  function updateVizCounties() {
    const scopeRows = getRowsForYearState(vizRows, vizYear, vizState);
    const computedCounties = getUniqueCountyValues(scopeRows, row => getVizJurisdictionName(row));
    const counties = computedCounties;
    if (el.vizCounty instanceof HTMLSelectElement) {
      vizCounty = setSelectOptions(el.vizCounty, counties, vizCounty, true, 'All jurisdictions');
    } else if (!counties.includes(vizCounty)) {
      vizCounty = '';
    }

    updateTopRaces();
  }

  function updateTopRaces() {
    const contestScopeRows = getRowsForYearStateCounty(vizRows, vizYear, vizState, vizCounty);
    const fallbackScopeRows = contestScopeRows.length
      ? contestScopeRows
      : getRowsForYearState(vizRows, vizYear, vizState);
    const contestTotals = {};
    fallbackScopeRows.forEach(row => {
      if (!row.contest) return;
      const votes = parseNumeric(row.votes);
      if (votes === null) return;
      contestTotals[row.contest] = (contestTotals[row.contest] || 0) + votes;
    });

    const contestsByVotes = Object.entries(contestTotals)
      .sort((a, b) => b[1] - a[1])
      .map(entry => entry[0]);
    const contestOptions = contestsByVotes;

    if (el.vizContest instanceof HTMLSelectElement) {
      vizContest = setSelectOptions(el.vizContest, contestOptions, vizContest);
    } else if (!contestOptions.includes(vizContest)) {
      vizContest = contestOptions[0] || '';
    }

    const stateScopeRows = getRowsForYearState(vizRows, vizYear, vizState);
    const topTotals = {};
    stateScopeRows.forEach(row => {
      if (!row.contest) return;
      const votes = parseNumeric(row.votes);
      if (votes === null) return;
      topTotals[row.contest] = (topTotals[row.contest] || 0) + votes;
    });

    vizTopRaces = Object.entries(topTotals)
      .sort((a, b) => b[1] - a[1])
      .slice(0, vizTopRaceCount)
      .map(entry => entry[0]);
    if (!vizTopRaces.length) {
      vizTopRaces = contestsByVotes;
    }

    if (
      vizInteractionMode === VIZ_INTERACTION_PREVIEW
      && vizTopRaces.length
      && !vizTopRaces.includes(vizContest)
    ) {
      vizContest = vizTopRaces[0];
      if (el.vizContest instanceof HTMLSelectElement) {
        el.vizContest.value = vizContest;
      }
    }
  }

  function ensureVizSelectionHasData() {
    if (!vizRows.length) return;
    let filtered = applyVizFilters(vizRows);
    if (filtered.length) return;

    updateVizStates();
    filtered = applyVizFilters(vizRows);
    if (filtered.length) return;

    const fallbackRows = getRowsForYearState(vizRows, vizYear, vizState);
    const fallbackContest = fallbackRows[0]?.contest || '';
    if (fallbackContest) {
      vizContest = fallbackContest;
      if (el.vizContest instanceof HTMLSelectElement) {
        el.vizContest.value = fallbackContest;
      }
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
      filtered = filtered.filter(row => normalizeCountyKey(getVizJurisdictionName(row)) === normalizeCountyKey(vizCounty));
    }
    if (vizContest) {
      filtered = filtered.filter(row => row.contest === vizContest);
    }
    return filtered;
  }

  function _shuffleArray(items) {
    for (let i = items.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [items[i], items[j]] = [items[j], items[i]];
    }
    return items;
  }

  function refreshViz() {
    if (vizInteractionMode === VIZ_INTERACTION_PREVIEW) {
      ensureVizSelectionHasData();
    }
    const filtered = applyVizFilters(vizRows);
    renderVizChart(filtered);
    renderVizTable(filtered);
    saveCurrentVizSnapshot(vizRows);
    updateVizModeBadges();
    renderVizStatus();
  }

  function setVizContest(value) {
    if (!value) return;
    vizContest = value;
    if (el.vizContest instanceof HTMLSelectElement) {
      el.vizContest.value = value;
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
    vizCounty = value || '';
    if (el.vizCounty instanceof HTMLSelectElement) {
      el.vizCounty.value = vizCounty;
    }
    refreshViz();
  }

  function setVizDataset(_value) {
    vizDataset = VIZ_DATASET_WAREHOUSE;
    if (el.vizDataset instanceof HTMLSelectElement) {
      el.vizDataset.value = VIZ_DATASET_WAREHOUSE;
    }
    setVizOverlayEnabled(false);
    applyVizDatasetRows(getVizSourceRows());
    syncVizOverlayAvailability();
    updateVizModeBadges();
    renderVizStatus();
  }

  function stopVizAutoRotation() {
    if (vizAutoTimer) {
      window.clearInterval(vizAutoTimer);
      vizAutoTimer = null;
    }
  }

  function stopPreviewFeedTimer() {
    if (previewTimer) {
      window.clearInterval(previewTimer);
      previewTimer = null;
    }
  }

  function getVizPreviewPoolRows() {
    const sourceRows = getVizSourceRows();
    return sourceRows.length ? sourceRows : vizRows;
  }

  function buildVizPreviewFrames(rows = getVizPreviewPoolRows()) {
    const frameMap = new Map();
    (Array.isArray(rows) ? rows : []).forEach(row => {
      const year = getRowYear(row);
      const state = String(row.state || '').trim();
      const contest = String(row.contest || '').trim();
      if (!year || !state || !contest) return;
      const key = `${year}\u0000${state}\u0000${contest}`;
      const votes = parseNumeric(row.votes);
      const current = frameMap.get(key) || {
        year,
        state,
        contest,
        jurisdiction: '',
        totalVotes: 0,
      };
      if (votes !== null) {
        current.totalVotes += votes;
      }
      frameMap.set(key, current);
    });
    return Array.from(frameMap.values())
      .filter(frame => frame.totalVotes > 0)
      .sort((a, b) => {
        if (b.totalVotes !== a.totalVotes) return b.totalVotes - a.totalVotes;
        return `${b.year}|${b.state}|${b.contest}`.localeCompare(`${a.year}|${a.state}|${a.contest}`);
      });
  }

  function setVizPreviewFrame(frame) {
    if (!frame || vizInteractionMode !== VIZ_INTERACTION_PREVIEW) return;
    vizYear = String(frame.year || '');
    vizState = String(frame.state || '');
    vizCounty = String(frame.jurisdiction || '');
    vizContest = String(frame.contest || '');

    if (el.vizYear instanceof HTMLSelectElement) el.vizYear.value = vizYear;
    updateVizStates();

    if (el.vizState instanceof HTMLSelectElement) el.vizState.value = vizState;
    updateVizCounties();

    if (el.vizCounty instanceof HTMLSelectElement) el.vizCounty.value = vizCounty;
    if (el.vizContest instanceof HTMLSelectElement) el.vizContest.value = vizContest;

    refreshViz();
  }

  function startVizAutoRotation(_resetOrder = true) {
    if (
      vizInteractionMode !== VIZ_INTERACTION_PREVIEW
      || vizAutoLocked
      || vizAutoPaused
      || vizHoverPaused
    ) {
      stopVizAutoRotation();
      updateVizAutoToggleLabel();
      return;
    }

    stopVizAutoRotation();
    const frames = buildVizPreviewFrames(vizRows);
    if (!frames.length) return;

    if (_resetOrder || !vizAutoOrder.length) {
      vizAutoOrder = _shuffleArray([...frames]);
      vizAutoIndex = 0;
    } else {
      const availableKeys = new Set(frames.map(frame => `${frame.year}\u0000${frame.state}\u0000${frame.contest}`));
      vizAutoOrder = vizAutoOrder.filter(frame => availableKeys.has(`${frame.year}\u0000${frame.state}\u0000${frame.contest}`));
      if (!vizAutoOrder.length) {
        vizAutoOrder = _shuffleArray([...frames]);
        vizAutoIndex = 0;
      } else {
        vizAutoIndex = Math.min(vizAutoIndex, vizAutoOrder.length - 1);
      }
    }

    setVizPreviewFrame(vizAutoOrder[vizAutoIndex]);
    vizAutoTimer = window.setInterval(() => {
      if (
        vizInteractionMode !== VIZ_INTERACTION_PREVIEW
        || vizAutoLocked
        || vizAutoPaused
        || vizHoverPaused
      ) return;
      vizAutoIndex = (vizAutoIndex + 1) % vizAutoOrder.length;
      setVizPreviewFrame(vizAutoOrder[vizAutoIndex]);
    }, 6000);
    updateVizAutoToggleLabel();
  }

  function pauseVizAutoRotation() {
    if (
      vizInteractionMode !== VIZ_INTERACTION_PREVIEW
      || vizAutoLocked
      || vizAutoPaused
      || vizHoverPaused
    ) return;
    vizHoverPaused = true;
    stopVizAutoRotation();
    if (el.vizHint) {
      el.vizHint.textContent = 'Preview paused while focused - move pointer away to resume.';
      el.vizHint.classList.add('is-visible');
    }
    el.vizPanel?.classList.add('is-focus-paused');
    updateVizAutoToggleLabel();
  }

  function hideVizHint() {
    if (el.vizHint) el.vizHint.classList.remove('is-visible');
  }

  function resumeVizAutoRotation() {
    if (
      vizInteractionMode !== VIZ_INTERACTION_PREVIEW
      || vizAutoLocked
      || vizAutoPaused
      || !vizHoverPaused
    ) return;
    vizHoverPaused = false;
    el.vizPanel?.classList.remove('is-focus-paused');
    startVizAutoRotation(false);
    hideVizHint();
    updateVizAutoToggleLabel();
  }

  function enterVizExploreMode(reason = 'manual') {
    vizInteractionMode = VIZ_INTERACTION_EXPLORE;
    vizAutoLocked = true;
    vizAutoPaused = true;
    vizHoverPaused = false;
    previewActive = false;
    previewMode = 'idle';
    stopPreviewFeedTimer();
    stopVizAutoRotation();
    setPreviewState(false);
    el.vizPanel?.classList.remove('is-focus-paused');
    hideVizHint();
    saveVizPlaybackPreference(true);
    updateVizAutoToggleLabel();
    if (reason && reason !== 'manual') {
      setPreviewStatus(`Explore mode • ${reason}`);
    }
  }

  function enterVizPreviewMode({ paused = false, preserveRows = true } = {}) {
    vizInteractionMode = VIZ_INTERACTION_PREVIEW;
    vizAutoLocked = false;
    vizAutoPaused = !!paused;
    vizHoverPaused = false;
    previewActive = true;
    previewMode = 'idle';
    canonicalQueryScopeHydrated = true;
    clearCanonicalQueryScopeFromLocation();
    setPreviewState(true);
    el.vizPanel?.classList.remove('is-focus-paused');
    hideVizHint();
    saveVizPlaybackPreference(vizAutoPaused);

    const sourceRows = getVizSourceRows();
    if (!preserveRows && sourceRows.length) {
      applyVizDatasetRows(sourceRows);
    }
    if (sourceRows.length) {
      stopPreviewFeedTimer();
      if (!vizAutoPaused) {
        startVizAutoRotation(false);
      }
    } else if (!vizAutoPaused) {
      startPreviewCycle('idle');
    }
    updateVizAutoToggleLabel();
  }

  function stepVizPreviewFrame(step) {
    if (vizInteractionMode !== VIZ_INTERACTION_PREVIEW) {
      enterVizPreviewMode({ paused: true, preserveRows: false });
    } else {
      vizAutoPaused = true;
      vizAutoLocked = false;
      vizHoverPaused = false;
      stopVizAutoRotation();
      saveVizPlaybackPreference(true);
    }

    const frames = buildVizPreviewFrames(vizRows);
    if (!frames.length) {
      updateVizAutoToggleLabel();
      return;
    }

    const keys = new Map(frames.map(frame => [`${frame.year}\u0000${frame.state}\u0000${frame.contest}`, frame]));
    if (!vizAutoOrder.length) {
      vizAutoOrder = _shuffleArray([...frames]);
      vizAutoIndex = 0;
    } else {
      vizAutoOrder = vizAutoOrder
        .map(frame => keys.get(`${frame.year}\u0000${frame.state}\u0000${frame.contest}`))
        .filter(Boolean);
      if (!vizAutoOrder.length) {
        vizAutoOrder = _shuffleArray([...frames]);
        vizAutoIndex = 0;
      }
    }

    vizAutoIndex = (vizAutoIndex + step + vizAutoOrder.length) % vizAutoOrder.length;
    setVizPreviewFrame(vizAutoOrder[vizAutoIndex]);
    updateVizAutoToggleLabel();
  }

  function updateVizAutoToggleLabel() {
    if (!el.vizAutoToggleBtn) return;
    const shouldStart = (
      vizInteractionMode === VIZ_INTERACTION_EXPLORE
      || vizAutoPaused
    );
    el.vizAutoToggleBtn.textContent = shouldStart ? 'Start' : 'Pause';
    updateVizModeBadges();
    renderVizStatus();
  }

  function getVizPlaybackLabel() {
    if (vizInteractionMode === VIZ_INTERACTION_EXPLORE) return 'Explore';
    if (vizAutoPaused) return 'Paused';
    if (vizHoverPaused) return 'Focused';
    return 'Auto';
  }

  function getVizScopeLabel() {
    return vizCounty ? 'Jurisdiction focus' : 'All jurisdictions';
  }

  function updateVizModeBadges() {
    if (el.vizPlaybackBadge) {
      el.vizPlaybackBadge.textContent = `Playback: ${getVizPlaybackLabel()}`;
    }
    if (el.vizScopeBadge) {
      el.vizScopeBadge.textContent = `Scope: ${getVizScopeLabel()}`;
    }
    if (el.vizCountyScopeBadge) {
      el.vizCountyScopeBadge.textContent = vizCounty ? `Jurisdiction: ${vizCounty}` : 'All jurisdictions';
      el.vizCountyScopeBadge.classList.toggle('is-focused', !!vizCounty);
    }
  }

  function renderVizStatus() {
    if (!el.vizPreviewStatus) return;
    const baseText = vizStatusBase || 'Visualization ready';
    const modeLabel = vizInteractionMode === VIZ_INTERACTION_EXPLORE ? 'Explore' : 'Preview';
    setUiState(
      el.vizPreviewStatus,
      vizUiState,
      `${baseText} • Mode: ${modeLabel} • Scope: ${getVizScopeLabel()} • Playback: ${getVizPlaybackLabel()}`
    );
  }

  function setPreviewStatus(text, state = 'idle') {
    vizStatusBase = text || '';
    vizUiState = normalizeUiState(state);
    renderVizStatus();
  }

  function setPreviewState(active) {
    if (el.vizPanel) {
      el.vizPanel.classList.toggle('is-previewing', !!active);
    }
    updateVizModeBadges();
    renderVizStatus();
  }

  function ghostPreviewPanels() {
    const panels = [el.vizChart, el.vizTable].filter(Boolean);
    panels.forEach(panel => panel.classList.add('is-ghosting'));
    window.setTimeout(() => {
      panels.forEach(panel => panel.classList.remove('is-ghosting'));
    }, 700);
  }

  async function refreshPreview() {
    if (!previewActive) return;
    const sourceRows = getVizSourceRows();

    if (!sourceRows.length) {
      setPreviewStatus(
        'Preview waiting for canonical publication rows.',
        'empty'
      );
      return;
    }

    stopPreviewFeedTimer();
    if (!vizRows.length || vizRows !== sourceRows) {
      applyVizDatasetRows(sourceRows);
    }
    setPreviewStatus(`Canonical Production â€¢ ${sourceRows.length} rows`);
    if (!vizAutoPaused && !vizHoverPaused) {
      startVizAutoRotation(false);
    }
  }
  function startPreviewCycle(mode = 'idle') {
    if (authRestrictedMode) return;
    if (vizInteractionMode !== VIZ_INTERACTION_PREVIEW) return;
    previewActive = true;
    previewMode = mode;
    setPreviewState(true);
    stopPreviewFeedTimer();

    const sourceRows = getVizSourceRows();
    if (sourceRows.length) {
      if (!vizRows.length || vizRows !== sourceRows) {
        applyVizDatasetRows(sourceRows);
      }
      if (!vizAutoPaused && !vizHoverPaused) {
        startVizAutoRotation(false);
      }
      return;
    }

    refreshPreview();
    previewTimer = window.setInterval(refreshPreview, 12000);
  }

  function stopPreviewCycle() {
    previewActive = false;
    stopPreviewFeedTimer();
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

  function mapWarehouseVizRecord(record) {
    if (!record || typeof record !== 'object') return null;
    return {
      dataset_type: VIZ_DATASET_WAREHOUSE,
      state: firstPresent(record.state, record.State, '') || '',
      jurisdiction_name: firstPresent(record.jurisdiction_name, '') || '',
      jurisdiction_type: firstPresent(record.jurisdiction_type, '') || '',
      county: firstPresent(record.county, record.County, '') || '',
      contest: firstPresent(record.contest, record.office, record.Office, record.race, '') || '',
      candidate: firstPresent(record.candidate, record.Candidate, record['Ballot Candidate Name'], '') || '',
      party: firstPresent(record.party, record.Party, record['Ballot Party'], '') || '',
      votes: firstNumeric(record.votes, record['Total Votes'], record.total_votes),
      uncategorized_votes: firstNumeric(record.uncategorized_votes, record['Uncategorized Votes']),
      year: extractYearFromValue(firstPresent(record.year, record.election_year, record.election_date, record.timestamp, record.date, '') || '')
    };
  }

  function getVizSourceRows() {
    if (vizInteractionMode === VIZ_INTERACTION_PREVIEW && canonicalPreviewRows.length) {
      return canonicalPreviewRows;
    }
    return warehouseVizRows;
  }

  function applyVizDatasetRows(rows) {
    vizRows = Array.isArray(rows) ? rows : [];
    if (!vizRows.length) {
      // G2.4.3B1.2: a no-result Explore response is a valid result, not a request
      // to reset the operator scope or erase available facet options.
      if (vizInteractionMode === VIZ_INTERACTION_PREVIEW) {
        hydrateVizFiltersFromSnapshot(vizDataset);
      }
      renderVizChart([]);
      renderVizTable([]);
      syncVizOverlayAvailability();
      updateVizModeBadges();
      renderVizStatus();
      return;
    }
    // G2.4.3B1: applying asynchronous rows must never clear an operator Explore lock.
    setVizFilters(vizRows);
    refreshViz();
    syncVizOverlayAvailability();
  }

  function updateVisualizationFromCurated(item) {
    if (!item) {
      return { status: 'idle', count: 0, axes: [] };
    }

    // G3.1C1.6: Source Evidence is relationship context only. Selecting evidence
    // must never change Preview/Explore mode and must never replace Analysis rows.
    const axes = getEvidenceMatchAxes(item);
    const sourceRows = getVizSourceRows();

    if (!sourceRows.length) {
      return { status: 'feed-unavailable', count: 0, axes };
    }

    const filtered = sourceRows.filter(row => {
      const matchState = item.state ? normalizeValue(row.state) === normalizeValue(item.state) : true;
      const matchCounty = item.county ? normalizeValue(getVizJurisdictionName(row)) === normalizeValue(item.county) : true;
      const matchContest = item.contest ? normalizeValue(row.contest) === normalizeValue(item.contest) : true;
      return matchState && matchCounty && matchContest;
    });

    if (!filtered.length) {
      return { status: 'no-match', count: 0, axes };
    }

    return { status: 'context-match', count: filtered.length, axes };
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
      resetEvidenceRelationshipContext();
      if (items[0] && !previewActive) {
        renderCuratedDetail(items[0]);
        const firstButton = el.curatedList?.querySelector('.curated-item');
        if (firstButton) firstButton.classList.add('is-active');
      }
    }
  }

  async function fetchCuratedDatasets() {
    if (!curatedUrl) return;
    if (authRestrictedMode) return;
    setStatusText(
      el.curatedStatus,
      'Loading curated datasets...',
      'loading'
    );
    const result = await fetchJsonWithRetry(curatedUrl, {
      authReason: 'Authentication required for curated datasets and preview feeds.',
      retries: 2
    });
    if (result.authBlocked) return;
    if (!result.ok || !result.data) {
      curatedItems = [];
      curatedSelection = null;
      renderCuratedList([]);
      resetEvidenceRelationshipContext();
      setStatusText(
        el.curatedStatus,
        'Failed to load curated datasets.',
        'error'
      );
      return;
    }
    const data = result.data;
    curatedItems = Array.isArray(data?.items) ? data.items : [];
    updateCuratedStateOptions(curatedItems);
    updateCuratedCountyOptions(curatedItems);
    filterCuratedItems();
    if (!curatedItems.length && !getVizSourceRows().length) {
      clearVisualization();
    }
    setStatusText(
      el.curatedStatus,
      curatedItems.length ? `Loaded ${curatedItems.length} datasets.` : 'No curated datasets available.',
      curatedItems.length ? 'ready' : 'empty'
    );
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
        down_ballot_missing: false,
        presidential_missing: false,
        eligible_voters_missing: false,
      };

      const downVotes = parseNumeric(row.down_ballot_votes);
      const presidentialVotes = parseNumeric(row.presidential_votes);
      const eligibleVoters = parseNumeric(row.eligible_voters);

      if (downVotes === null) entry.down_ballot_missing = true;
      else entry.down_ballot_votes += downVotes;

      if (presidentialVotes === null) entry.presidential_missing = true;
      else entry.presidential_votes += presidentialVotes;

      if (eligibleVoters === null) entry.eligible_voters_missing = true;
      else entry.eligible_voters += eligibleVoters;

      grouped.set(county, entry);
    });

    const values = Array.from(grouped.values()).map(entry => {
      const downVotes = entry.down_ballot_missing ? null : entry.down_ballot_votes;
      const presidentialVotes = entry.presidential_missing ? null : entry.presidential_votes;
      const eligibleVoters = entry.eligible_voters_missing ? null : entry.eligible_voters;
      const deltaVotes = (downVotes !== null && presidentialVotes !== null)
        ? downVotes - presidentialVotes
        : null;
      const percentDelta = (
        deltaVotes !== null
        && presidentialVotes !== null
        && presidentialVotes !== 0
      ) ? (deltaVotes / presidentialVotes) * 100 : null;
      const turnoutPct = (
        presidentialVotes !== null
        && eligibleVoters !== null
        && eligibleVoters !== 0
      ) ? (presidentialVotes / eligibleVoters) * 100 : null;
      const adjustedVotes = (
        deltaVotes !== null
        && presidentialVotes !== null
        && presidentialVotes !== 0
      ) ? (deltaVotes / presidentialVotes) * 10000 : null;

      return {
        ...entry,
        down_ballot_votes: downVotes,
        presidential_votes: presidentialVotes,
        eligible_voters: eligibleVoters,
        delta_votes: deltaVotes,
        delta_pct: percentDelta,
        turnout_pct: turnoutPct,
        adjusted_votes: adjustedVotes,
      };
    });

    const knownPresidential = values
      .map(item => item.presidential_votes)
      .filter(value => value !== null && value > 0);
    const maxPresidential = knownPresidential.length ? Math.max(...knownPresidential) : null;

    values.forEach(item => {
      if (
        item.delta_pct === null
        || item.presidential_votes === null
        || !maxPresidential
      ) {
        item.adjusted_pct = null;
        return;
      }
      const weight = Math.sqrt(item.presidential_votes / maxPresidential);
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
    const num = parseNumeric(value);
    if (num === null) return '—';
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

    const plottableRows = rows
      .map(row => ({ row, value: parseNumeric(config.value(row)) }))
      .filter(item => item.value !== null);

    if (!plottableRows.length) {
      const empty = document.createElement('div');
      empty.className = 'dropoff-summary-note';
      empty.textContent = rows.length
        ? 'Drop-off values are not reported for the current selection.'
        : 'No county data available for the current state/year/contest filter.';
      targetEl.appendChild(empty);
      return;
    }

    const selectedCounty = getSelectedDropoffCounty();
    const maxAbs = Math.max(1, ...plottableRows.map(item => Math.abs(item.value)));
    const barWidth = plottableRows.length > 120 ? 6 : plottableRows.length > 80 ? 8 : plottableRows.length > 40 ? 10 : 14;
    const gap = plottableRows.length > 80 ? 2 : 3;
    const margin = { top: 14, right: 14, bottom: 76, left: 56 };
    const plotHeight = 190;
    const width = Math.max(720, margin.left + margin.right + plottableRows.length * (barWidth + gap));
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

    const labelStep = Math.max(1, Math.ceil(plottableRows.length / 16));
    plottableRows.forEach(({ row, value: rawValue }, index) => {
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
      const aMetric = parseNumeric(dropoffMetric === 'percent' ? a.delta_pct : a.delta_votes);
      const bMetric = parseNumeric(dropoffMetric === 'percent' ? b.delta_pct : b.delta_votes);
      if (aMetric === null && bMetric === null) return 0;
      if (aMetric === null) return 1;
      if (bMetric === null) return -1;
      const aBase = Math.abs(aMetric);
      const bBase = Math.abs(bMetric);
      if (dropoffOrderStrategy === 'turnout_weighted') {
        const aPres = parseNumeric(a.presidential_votes);
        const bPres = parseNumeric(b.presidential_votes);
        const aWeight = Math.sqrt(Math.max(1, aPres === null ? 1 : aPres));
        const bWeight = Math.sqrt(Math.max(1, bPres === null ? 1 : bPres));
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
    const turnoutPct = row.turnout_pct !== null && row.turnout_pct !== undefined
      ? `${formatDropoffValue(row.turnout_pct, 2)}%`
      : 'n/a';
    const eligible = row.eligible_voters !== null && row.eligible_voters !== undefined
      ? formatDropoffValue(row.eligible_voters, 0)
      : 'n/a';
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
          const hdrs = new Headers();
          hdrs.append('Accept', 'application/json');
          hdrs.append('X-Requested-With', 'XMLHttpRequest');
          if (csrfToken) {
            hdrs.append('X-CSRFToken', csrfToken);
          }
          fetchInit.headers = hdrs;
          
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
                if (typeof authUtils.defaultCertRequiredHandler === 'function') {
                  authUtils.defaultCertRequiredHandler(_url || uploadUrl);
                }
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
            fetchCanonicalRecordData(true);
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
    el.scaffoldJson.addEventListener('click', async () => {
      setStatus(el.status, 'info', 'Building scaffold...');
      try {
        const result = await fetchJsonWithRetry(scaffoldJsonUrl + '?limit=200', {
          authReason: 'Authentication required for scaffold JSON endpoint.',
          retries: 1,
        });
        if (result.authBlocked) return;
        if (!result.ok || !result.data) {
          setStatus(el.status, 'error', 'Scaffold download failed.');
          showErrorToast('Scaffold JSON failed.');
          return;
        }
        const blob = new Blob([JSON.stringify(result.data, null, 2)], { type: 'application/json' });
        downloadBlob(blob, 'data_framework_scaffold.json');
        setStatus(el.status, 'ok', 'Scaffold JSON ready.');
        showInfoToast('Scaffold JSON downloaded.');
      } catch (err) {
        setStatus(el.status, 'error', `Scaffold download failed: ${err}`);
        showErrorToast('Scaffold JSON failed.');
      }
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
      th.textContent = COLUMN_LABELS[key] || key
        .split('_')
        .map(part => part ? part.charAt(0).toUpperCase() + part.slice(1) : '')
        .join(' ');

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
          td.textContent = displayValue(row[col]);
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

    if (!slice.length && rawData.length) {
      setStatus(
        el.status,
        'info',
        'No results match the current filters.',
        'empty'
      );
    } else {
      restoreCanonicalRecordBaseStatus();
    }
  }

  // ---------- CSV helpers ----------
  function buildVisibleCsv(data) {
    const cols = visibleColumns.filter(c => allowedColumns.has(c));
    const header = cols.join(',');
    const rows = data.map(r =>
      cols.map(c => {
        let v = exportValue(r[c]).replace(/\r?\n/g, ' ').replace(/\r/g, ' ');
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

  el.refresh?.addEventListener('click', () => fetchCanonicalRecordData(true));

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
      fetchCanonicalRecordFacets();
      fetchPriorityStatus();
      fetchCanonicalRecordData(true);
    }
  });

  el.priorityYearSelect?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      priorityYear = tgt.value || '';
      fetchCanonicalRecordFacets();
      fetchPriorityStatus();
      fetchCanonicalRecordData(true);
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

  // Preview hover/focus freezes transient playback only; it never changes Explore scope.
  el.vizChart?.addEventListener('mouseenter', pauseVizAutoRotation);
  el.vizChart?.addEventListener('mouseleave', resumeVizAutoRotation);
  el.vizTable?.addEventListener('mouseenter', pauseVizAutoRotation);
  el.vizTable?.addEventListener('mouseleave', resumeVizAutoRotation);
  el.vizFilters?.addEventListener('mouseenter', pauseVizAutoRotation);
  el.vizFilters?.addEventListener('mouseleave', resumeVizAutoRotation);

  el.vizDataset?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      enterVizExploreMode('analysis view selected');
      setVizDataset(VIZ_DATASET_WAREHOUSE);
      refreshCanonicalExploreScope();
      updateVizAutoToggleLabel();
    }
  });

  el.vizYear?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      enterVizExploreMode('year selected');
      hideVizHint();
      vizYear = tgt.value || '';
      canonicalQueryScopeHydrated = true;
      replaceCanonicalQueryScopeInLocation();
      refreshCanonicalExploreScope();
      updateVizAutoToggleLabel();
    }
  });

  el.vizState?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      enterVizExploreMode('state selected');
      hideVizHint();
      vizState = tgt.value || '';
      canonicalQueryScopeHydrated = true;
      replaceCanonicalQueryScopeInLocation();
      refreshCanonicalExploreScope();
      updateVizAutoToggleLabel();
    }
  });

  el.vizCounty?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      enterVizExploreMode('jurisdiction selected');
      hideVizHint();
      vizCounty = tgt.value || '';
      canonicalQueryScopeHydrated = true;
      replaceCanonicalQueryScopeInLocation();
      refreshCanonicalExploreScope();
      updateVizAutoToggleLabel();
    }
  });

  el.vizContest?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLSelectElement) {
      enterVizExploreMode('contest selected');
      hideVizHint();
      vizContest = tgt.value || '';
      canonicalQueryScopeHydrated = true;
      replaceCanonicalQueryScopeInLocation();
      refreshCanonicalExploreScope();
      updateVizAutoToggleLabel();
    }
  });

  el.vizPrevStateBtn?.addEventListener('click', () => {
    hideVizHint();
    stepVizPreviewFrame(-1);
  });

  el.vizNextStateBtn?.addEventListener('click', () => {
    hideVizHint();
    stepVizPreviewFrame(1);
  });

  el.vizAutoToggleBtn?.addEventListener('click', () => {
    if (vizInteractionMode === VIZ_INTERACTION_EXPLORE || vizAutoPaused) {
      enterVizPreviewMode({ paused: false, preserveRows: false });
    } else {
      vizAutoPaused = true;
      vizHoverPaused = false;
      stopVizAutoRotation();
      saveVizPlaybackPreference(true);
      hideVizHint();
      updateVizAutoToggleLabel();
    }
  });

  el.vizDropoffOverlayToggle?.addEventListener('change', e => {
    const tgt = e.target;
    if (tgt instanceof HTMLInputElement) {
      setVizOverlayEnabled(!!tgt.checked);
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
    const requestSeq = ++canonicalDataRequestSeq;
    if (canonicalDataAbortController) canonicalDataAbortController.abort();
    canonicalDataAbortController = new AbortController();

    (async () => {
      if (showLoading) {
        setPreviewStatus(
          'Loading Canonical Production Analysis...',
          'loading'
        );
      }

      const result = await fetchJsonWithRetry(buildCanonicalDataUrl(), {
        authReason: 'Authentication required for canonical Analysis feed.',
        retries: 2,
        signal: canonicalDataAbortController.signal,
      });
      if (result?.aborted || requestSeq !== canonicalDataRequestSeq) return;
      if (result.authBlocked) return;
      if (!result.ok || !result.data) {
        throw new Error(`Server error ${result.status || 'unknown'}.`);
      }

      const data = result.data;
      const canonicalRows = Array.isArray(data)
        ? data
        : Array.isArray(data?.rows) ? data.rows
        : Array.isArray(data?.items) ? data.items
        : [];

      warehouseVizRows = canonicalRows
        .map(mapWarehouseVizRecord)
        .filter(Boolean);

      analysisRowsPossiblyTruncated =
        canonicalRows.length >= CANONICAL_CLIENT_ROW_LIMIT;

      if (!hasCanonicalScope() && warehouseVizRows.length) {
        canonicalPreviewRows = [...warehouseVizRows];
      }

      if (warehouseVizRows.length) {
        writeVizSnapshot(VIZ_DATASET_WAREHOUSE, {
          rowCount: warehouseVizRows.length,
          options: {
            years: Array.from(
              new Set(warehouseVizRows.map(row => getRowYear(row)).filter(Boolean))
            ).sort((a, b) => Number(b) - Number(a))
          },
          selection: {
            vizYear,
            vizState,
            vizCounty,
            vizContest,
          },
        });
      }

      if (vizDataset === VIZ_DATASET_WAREHOUSE) {
        // Fresh canonical Analysis rows always win. Evidence is evaluated only
        // after the canonical scope has been rendered, so a no-match cannot
        // leave stale rows from the previous Analysis request.
        applyVizDatasetRows(warehouseVizRows);

        if (curatedSelection) {
          const analysisResult = updateVisualizationFromCurated(curatedSelection);
          updateEvidenceRelationshipContext(curatedSelection, analysisResult);
        }

        const analysisState = warehouseVizRows.length ? 'ready' : 'empty';
        if (analysisRowsPossiblyTruncated) {
          setPreviewStatus(
            `Canonical Production - ${warehouseVizRows.length} rows - API cap reached; totals may be partial`,
            analysisState
          );
        } else {
          setPreviewStatus(
            `Canonical Production - ${warehouseVizRows.length} rows`,
            analysisState
          );
        }
      }
    })().catch(err => {
      if (requestSeq !== canonicalDataRequestSeq || (err && err.name === 'AbortError')) {
        return;
      }
      const msg = err?.message || String(err);
      setPreviewStatus(
        `Canonical Analysis load failed - ${msg}`,
        'error'
      );
      showErrorToast('Failed to load Analysis data.');
    });
  }

  function fetchCanonicalRecordData(showLoading = false) {
    const requestSeq = ++canonicalRecordRequestSeq;
    if (canonicalRecordAbortController) canonicalRecordAbortController.abort();
    canonicalRecordAbortController = new AbortController();

    (async () => {
      if (showLoading) {
        setCanonicalRecordBaseStatus(
          'info',
          'Loading Canonical Record data...',
          'loading'
        );
        renderSkeleton();
      }

      const result = await fetchJsonWithRetry(buildCanonicalRecordDataUrl(), {
        authReason: 'Authentication required for Canonical Record feed.',
        retries: 2,
        signal: canonicalRecordAbortController.signal,
      });
      if (result?.aborted || requestSeq !== canonicalRecordRequestSeq) return;
      if (result.authBlocked) return;
      if (!result.ok || !result.data) {
        throw new Error(`Server error ${result.status || 'unknown'}.`);
      }

      const data = result.data;
      rawData = Array.isArray(data)
        ? data
        : Array.isArray(data?.rows) ? data.rows
        : Array.isArray(data?.items) ? data.items
        : [];
      if (!Array.isArray(rawData)) rawData = [];

      rawData = rawData.map(r => {
        if (r && typeof r === 'object' && !Array.isArray(r)) return r;
        return {};
      });

      if (!compactPreferenceSet && rawData.length) {
        setCompactTable(rawData.length >= COMPACT_AUTO_THRESHOLD);
      }

      allowedColumns.clear();
      buildColumns();

      const scopeParts = [];
      if (priorityYear) scopeParts.push(`Year ${priorityYear}`);
      if (priorityState) scopeParts.push(`State ${priorityState}`);
      const scopeText = scopeParts.length ? ` - ${scopeParts.join(' / ')}` : '';

      if (!rawData.length) {
        setCanonicalRecordBaseStatus(
          'info',
          `No Canonical Record rows found${scopeText}.`,
          'empty'
        );
      } else if (rawData.length >= CANONICAL_CLIENT_ROW_LIMIT) {
        setCanonicalRecordBaseStatus(
          'ok',
          `Loaded first ${rawData.length} Canonical Record rows${scopeText}; API cap reached, result may be partial.`,
          'ready'
        );
      } else {
        setCanonicalRecordBaseStatus(
          'ok',
          `Loaded ${rawData.length} Canonical Record rows${scopeText}.`,
          'ready'
        );
      }

      page = 1;
      render();
    })().catch(err => {
      if (requestSeq !== canonicalRecordRequestSeq || (err && err.name === 'AbortError')) {
        return;
      }

      rawData = [];
      allowedColumns.clear();
      buildColumns();
      render();

      const msg = err?.message || String(err);
      if (/does not exist/i.test(msg)) {
        setCanonicalRecordBaseStatus(
          'error',
          'Canonical Record backend unavailable. Waiting for initialization.',
          'error'
        );
      } else {
        setCanonicalRecordBaseStatus('error', msg, 'error');
      }
      showErrorToast('Failed to load Canonical Record data.');
    });
  }

  // ---------- Init ----------
  resetEvidenceRelationshipContext();
  initPipelineSteps();
  loadVizOverlayPreference();
  loadVizPlaybackPreference();
  loadVizModeHelpPreference();
  loadCompactPreference();
  loadDropoffDrawerPreference();
  loadDropoffPreferences();
  loadGhostPanelPreference();
  const hasInitialCanonicalQueryScope = applyInitialCanonicalQueryScope();
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
  const cachedWarehouseStatus = readWarehouseStatusSnapshot();
  if (cachedWarehouseStatus?.payload) {
    applyPriorityPayload(cachedWarehouseStatus.payload, true);
  }
  updateVizModeBadges();
  renderVizStatus();
  updateVizAutoToggleLabel();

  if (el.vizPanel) {
    el.vizPanel.addEventListener('pointerdown', () => {
      setVizModeHelpCollapsed(true, true);
    }, { once: true });
  }

  async function bootstrapProtectedFeeds() {
    await fetchPriorityStatus();
    if (authRestrictedMode) return;

    // Canonical facet authority defines valid State / Year options. Warehouse
    // status remains contextual priority metadata and never defines record scope.
    const canonicalUniverseReady = await fetchCanonicalFacets({ universe: true });
    if (canonicalUniverseReady) {
      await fetchCanonicalRecordFacets({ useUniverse: true });
    }

    fetchCanonicalRecordData(true);

    priorityTimer = window.setInterval(fetchPriorityStatus, PRIORITY_REFRESH_MS);
    fetchCuratedDatasets();

    if (
      vizInteractionMode === VIZ_INTERACTION_PREVIEW
      && !vizAutoPaused
      && !vizAutoLocked
    ) {
      startPreviewCycle('idle');
    }
  }

  async function bootstrapInitialAnalysisRead() {
    if (!hasInitialCanonicalQueryScope) {
      fetchData(true);
      bootstrapProtectedFeeds();
      return;
    }

    // A URL-provided scope is never sent to the canonical result endpoint
    // until the canonical facet universe has accepted/cleared its values.
    const canonicalUniverseReady = await fetchCanonicalFacets({
      universe: true,
    });
    if (!canonicalUniverseReady) {
      vizYear = '';
      vizState = '';
      vizCounty = '';
      vizContest = '';
      canonicalQueryScopeHydrated = true;
      replaceCanonicalQueryScopeInLocation();
      fetchData(true);
      bootstrapProtectedFeeds();
      return;
    }

    canonicalQueryScopeHydrated = true;
    await fetchCanonicalFacets();
    replaceCanonicalQueryScopeInLocation();
    fetchData(true);

    // Preserve the long-standing protected-bootstrap contract exactly.
    // A deep-link startup may therefore perform a second canonical-universe GET.
    // That duplicate read is bounded and preferable to coupling public Analysis
    // scope to protected-feed bootstrap state.
    bootstrapProtectedFeeds();
  }

  // Canonical publication rows are the sole election-result Analysis feed.
  bootstrapInitialAnalysisRead();
});  