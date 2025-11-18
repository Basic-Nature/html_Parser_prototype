/* run_parser.js
   - Early session room join + queuing (joinedSessions/earlyQueue)
   - Auto history fetch & race protection
   - Prompt auto-response (optional) + numeric index expansion
   - Contest selection menu detection + modal
   - Folder browser: uploads, input, output (search + nested navigation)
   - Retains prior normalization & UI logic
*/
(function () {
  'use strict';
  if (window.__RUN_PARSER_JS_LOADED__) return;
  window.__RUN_PARSER_JS_LOADED__ = true;

  // -------- Config / Feature Flags --------
  const ENABLE_META_STRIP = true;
  const DISPLAY_TRUNCATE = 160;
  const DUP_WINDOW_MS = 600;
  const SHOW_BROWSE_TOOLBAR = false;
  // Respect OS "reduced motion" setting (used by particle effects)
  const PREFERS_REDUCED_MOTION = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  // -------- Utilities --------
  const $  = sel => document.querySelector(sel);
  const $$ = sel => Array.from(document.querySelectorAll(sel));
  const on = (el, ev, fn, opts) => el && el.addEventListener(ev, fn, opts);
  const lsGetJSON = (k, fb) => { try { return JSON.parse(localStorage.getItem(k) || 'null') ?? fb; } catch { return fb; } };
  const lsSetJSON = (k, v) => localStorage.setItem(k, JSON.stringify(v));
  const uniq = arr => Array.from(new Set(arr));
  const nowPerf = () => performance.now();
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  const debounce = (fn, wait = 200) => { let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), wait); }; };
  function extractFirstJSONObject(str) {
    if (typeof str !== 'string') return null;
    let depth = 0, start = -1, inStr = false, escNext = false;
    for (let i=0;i<str.length;i++) {
      const c = str[i];
      if (escNext) { escNext = false; continue; }
      if (c === '\\') { escNext = true; continue; }
      if (c === '"') inStr = !inStr;
      if (inStr) continue;
      if (c === '{') {
        if (depth === 0) start = i;
        depth++;
      } else if (c === '}') {
        depth--;
        if (depth === 0 && start !== -1) return str.slice(start, i+1);
      }
    }
    return null;
  }
  function tryParseJSON(str, allowExtract = true) {
    if (typeof str !== 'string') return null;
    let s = str.trim();
    if (!s.startsWith('{') && allowExtract) {
      const frag = extractFirstJSONObject(s);
      if (frag) s = frag;
    }
    if (!s.startsWith('{')) return null;
    try { return JSON.parse(s); } catch { return null; }
  }
  function stripMetaLines(rawStr) {
    if (!ENABLE_META_STRIP || typeof rawStr !== 'string' || !/(^|\n)\s*(Level:|Type:|Session:)/i.test(rawStr)) return rawStr;
    return rawStr
      .split(/\r?\n/)
      .map(l => l.trim())
      .filter(l => l && !/^Level:\s/i.test(l) && !/^Type:\s/i.test(l) && !/^Session:\s/i.test(l))
      .join('\n');
  }

  // -------- State --------
  let socket = null;
  let cancelRequested = false;
  let activeSessionId = '';
  let batching = false;
  let batchFrag = document.createDocumentFragment();
  let lastFlush = 0;
  const BATCH_INTERVAL = 40;
  const dupMap = new Map();
  const typeCounts = { all: 0 };
  const urlIndexMap = Object.create(null);
  let contestIndexMap = {};
  const MANUAL_UPLOAD_PROMPT_PATTERN = /select a file to parse from uploads/i;
  const manualOverrideState = Object.create(null);
  const manualUploadSelectionBySession = new Map();
  const directUrlDraftBySession = new Map();
  let manualUploadSelection = null; // { relPath, name }
  let manualUploadsInventory = [];
  let pendingManualUploadSelection = null;
  let pendingDirectUrlDraft = '';
  const MAX_DIRECT_URLS = 20;
  const SessionEnums = {
    state: {
      IDLE: 'idle',
      RUNNING: 'running',
      WAITING_PROMPT: 'waiting_prompt',
      CANCELLING: 'cancelling',
      CANCELLED: 'cancelled',
      COMPLETED: 'completed',
      ERROR: 'error',
    },
    phase: {
      PREPARE: 'prepare',
      SOURCE: 'source',
      RUN: 'run',
      RESOLVE: 'resolve',
      REVIEW: 'review',
    },
    stateToPhase: {
      idle: 'prepare',
      running: 'run',
      waiting_prompt: 'resolve',
      cancelling: 'run',
      cancelled: 'review',
      completed: 'review',
      error: 'review',
    },
  };

  let PIPELINE_PHASES = [
    SessionEnums.phase.PREPARE,
    SessionEnums.phase.SOURCE,
    SessionEnums.phase.RUN,
    SessionEnums.phase.RESOLVE,
    SessionEnums.phase.REVIEW,
  ];
  let pipelinePhase = PIPELINE_PHASES[0];
  let lastCelebrationTs = 0;

  fetch('/api/session/enums', { cache: 'no-store' })
    .then(res => (res.ok ? res.json() : null))
    .then(data => {
      if (!data || typeof data !== 'object') return;
      if (data.states && typeof data.states === 'object') {
        SessionEnums.state = { ...SessionEnums.state, ...data.states };
      }
      if (data.phases && typeof data.phases === 'object') {
        SessionEnums.phase = { ...SessionEnums.phase, ...data.phases };
      }
      if (Array.isArray(data.phase_order) && data.phase_order.length) {
        PIPELINE_PHASES = data.phase_order.slice();
        if (!PIPELINE_PHASES.includes(pipelinePhase)) {
          pipelinePhase = PIPELINE_PHASES[0];
        }
      }
      if (data.state_phase_map && typeof data.state_phase_map === 'object') {
        SessionEnums.stateToPhase = { ...SessionEnums.stateToPhase, ...data.state_phase_map };
      }
  })
  .catch(() => {});

  function createSessionMirror() {
    const store = new Map();
    const subscribers = new Set();

    function notify(sessionId) {
      const meta = store.get(sessionId) || null;
      subscribers.forEach(fn => {
        try { fn(sessionId, meta); } catch (err) { void err; }
      });
    }

    const mirror = {
      upsert(meta) {
        if (!meta || typeof meta !== 'object') return;
        const sid = meta.session_id;
        if (!sid) return;
        const existing = store.get(sid) || {};
        const merged = { ...existing, ...meta };
        store.set(sid, merged);
        notify(sid);
      },
      remove(sessionId) {
        if (!sessionId) return;
        store.delete(sessionId);
        notify(sessionId);
      },
      replace(list) {
        store.clear();
        if (Array.isArray(list)) {
          list.forEach(item => mirror.upsert(item));
        }
      },
      get(sessionId) {
        return store.get(sessionId) || null;
      },
      list() {
        return Array.from(store.values());
      },
      subscribe(fn) {
        if (typeof fn !== 'function') return () => {};
        subscribers.add(fn);
        return () => subscribers.delete(fn);
      }
    };

    return mirror;
  }

  const sessionMirror = createSessionMirror();
  sessionMirror.subscribe((sid, meta) => {
    ingestSessionSourceMeta(sid, meta, { fromServer: true });
  });
  let uploadsHasFiles = null;

  function pipelinePhaseIndex(phase) {
    return PIPELINE_PHASES.indexOf(phase);
  }

  function setPipelineHint(text, level = 'info') {
    if (!el || !el.pipelineHint) return;
    if (!text) {
      el.pipelineHint.textContent = '';
      el.pipelineHint.dataset.level = 'info';
      el.pipelineHint.classList.add('hidden');
      return;
    }
    el.pipelineHint.textContent = text;
    el.pipelineHint.dataset.level = level;
    el.pipelineHint.classList.remove('hidden');
  }

  function updatePipelineHintForPhase(extraMessage) {
    if (!el || !el.pipelineHint) return;
    const source = currentFileSource();
    const origin = activeManualSourceOrigin || 'default';
    let message = '';
    let level = 'info';
    if (extraMessage) {
      if (typeof extraMessage === 'object') {
        message = extraMessage.text || '';
        level = extraMessage.level || 'info';
      } else {
        message = String(extraMessage);
      }
    } else {
      switch (pipelinePhase) {
        case 'prepare': {
          if (source === 'uploads') {
            if (uploadsHasFiles === false) {
              message = 'Uploads folder is empty. Use the Uploads section or tap "Show Instructions" for guidance.';
              level = 'warn';
            } else if (origin !== 'default') {
              message = 'Manual uploads already selected. Step "Choose Source" is checked - press Run when ready.';
            } else {
              message = 'Verify your uploads or add new files, then choose the source and press Run when ready.';
            }
          } else {
            message = 'Review URLs and inputs. Switch to “Manual Uploads” if you need to run a local file.';
          }
          break;
        }
        case 'source': {
          if (source === 'uploads') {
            if (uploadsHasFiles === false) {
              message = 'No files detected in Uploads. Add a file first or switch back to Input Folder.';
              level = 'warn';
            } else if (origin !== 'default') {
              message = origin === 'server'
                ? 'Manual uploads pre-selected. Review the file, then continue to Run.'
                : 'Manual uploads selected. Review the file list or press Run when ready.';
              level = 'info';
            } else {
              message = 'Select the file you want to parse from the Uploads panel before running.';
              level = 'action';
            }
          } else {
            message = 'Parser will pull from the Input folder. Press Run when you are ready.';
          }
          break;
        }
        case 'run': {
          message = 'Parser is running. Monitor the activity log for progress and warnings.';
          break;
        }
        case 'resolve': {
          message = 'Respond to the prompt to continue. Use the modal or command box to submit your choice.';
          level = 'action';
          break;
        }
        case 'review': {
          message = 'Download outputs from the Output section or rerun with a different selection.';
          break;
        }
        default:
          message = '';
      }
    }
    setPipelineHint(message, level);
  }

  function setPipelinePhase(phase, { focus = false, force = false } = {}) {
    if (!phase) return;
    const targetIdx = pipelinePhaseIndex(phase);
    if (targetIdx === -1) return;
    const currentIdx = pipelinePhaseIndex(pipelinePhase);
    if (!force && currentIdx > targetIdx) {
      updatePipelineHintForPhase();
      return;
    }
    pipelinePhase = phase;
    if (pipelineControl) {
      pipelineControl.setPhase(phase, { source: 'manual' });
      if (focus) pipelineControl.focusStep(phase, { scroll: false, highlight: true });
    }
    updatePipelineHintForPhase();
  }

  function ensureResolveAttention(flag) {
    if (pipelineControl && typeof pipelineControl.attentionOnly === 'function') {
      pipelineControl.attentionOnly('resolve', !!flag);
    }
  }

  function noteUploadsPresence(hasFiles) {
    if (uploadsHasFiles === hasFiles) return;
    uploadsHasFiles = hasFiles;
    updatePipelineHintForPhase();
  }

  function updatePipelineMetadataForActive() {
    if (!pipelineControl || typeof pipelineControl.setStepState !== 'function') return;
    if (activeManualSource === 'uploads' && activeManualSourceOrigin !== 'default') {
      pipelineControl.setStepState('source', 'done');
    } else {
      pipelineControl.setStepState('source', null);
    }
  }
  // Contest options store (per session) + helpers
  const CONTEST_STORE_KEY = 'contest_opts_by_session_v1';
  let contestOptionsBySession = lsGetJSON(CONTEST_STORE_KEY, {});
  function cloneContestOption(opt) {
    if (!opt || typeof opt !== 'object') return opt;
    const cloned = { ...opt };
    if (opt.metadata && typeof opt.metadata === 'object') {
      try {
        cloned.metadata = JSON.parse(JSON.stringify(opt.metadata));
      } catch {
        cloned.metadata = { ...opt.metadata };
      }
    }
    return cloned;
  }
  function cloneContestOptions(list) {
    return Array.isArray(list) ? list.map(cloneContestOption) : [];
  }
  function getContestOptions(sessionId = activeSessionId) {
    return cloneContestOptions(contestOptionsBySession[sessionId] || []);
  }
  function setContestOptions(sessionId, opts) {
    contestOptionsBySession[sessionId] = cloneContestOptions(opts || []);
    lsSetJSON(CONTEST_STORE_KEY, contestOptionsBySession);
  }
  const tablePreviewBySession = new Map();

  function cloneTablePreviewEntry(entry) {
    if (!entry || typeof entry !== 'object') return null;
    const headers = Array.isArray(entry.headers) ? entry.headers.map(h => String(h)) : [];
    const rows = Array.isArray(entry.rows)
      ? entry.rows.map(row => {
          if (!row || typeof row !== 'object') return {};
          const copy = {};
          Object.keys(row).forEach(key => { copy[key] = row[key]; });
          return copy;
        })
      : [];
    return {
      index: Number(entry.index) || 0,
      total: Number(entry.total) || 0,
      confidence: typeof entry.confidence === 'number' ? entry.confidence : null,
      headers,
      rows,
      contest: entry.contest || '',
      receivedAt: Number(entry.receivedAt) || Date.now(),
    };
  }

  function cloneTablePreviewState(state) {
    if (!state || typeof state !== 'object') return { contest: '', entries: [] };
    return {
      contest: state.contest || '',
      entries: Array.isArray(state.entries)
        ? state.entries.map(entry => cloneTablePreviewEntry(entry)).filter(Boolean)
        : [],
    };
  }

  function getTablePreviewState(sessionId = activeSessionId) {
    return cloneTablePreviewState(tablePreviewBySession.get(sessionId));
  }

  function recordTablePreview(sessionId, raw) {
    if (!sessionId || !raw || typeof raw !== 'object') return;
    const preview = raw.preview;
    if (!preview || typeof preview !== 'object') return;
    const headers = Array.isArray(preview.headers) ? preview.headers.map(h => String(h)) : [];
    const rows = Array.isArray(preview.rows_preview)
      ? preview.rows_preview.map(row => {
          if (!row || typeof row !== 'object') return {};
          const copy = {};
          Object.keys(row).forEach(key => { copy[key] = row[key]; });
          return copy;
        })
      : [];
    const entry = {
      index: Number(raw.candidate_index || raw.preview_index || rows.index) || 1,
      total: Number(raw.candidates_total || raw.total_candidates || preview.candidates_total) || Math.max(rows.length, 0),
      confidence: typeof raw.ml_avg_confidence === 'number' ? raw.ml_avg_confidence : null,
      headers,
      rows,
      contest: raw.contest || preview.contest || '',
      receivedAt: raw.timestamp || Date.now(),
    };
    const state = tablePreviewBySession.get(sessionId) || { contest: entry.contest || '', entries: [] };
    if (entry.contest) state.contest = entry.contest;
    const existingIdx = state.entries.findIndex(e => Number(e.index) === Number(entry.index));
    if (existingIdx >= 0) {
      state.entries[existingIdx] = entry;
    } else {
      state.entries.push(entry);
    }
    state.entries.sort((a, b) => Number(a.index) - Number(b.index));
    // Limit to most recent 12 entries to avoid unbounded growth
    if (state.entries.length > 12) {
      state.entries = state.entries.slice(-12);
    }
    tablePreviewBySession.set(sessionId, state);
    if (typeof document !== 'undefined' && document) {
      try {
        document.dispatchEvent(new CustomEvent('table-preview:updated', {
          detail: { sessionId }
        }));
      } catch (err) {
        void err;
      }
    }
  }

  window.getTablePreviews = (sessionId = activeSessionId) => getTablePreviewState(sessionId);
  // Handy globals for console/UI hooks
  window.getLastContestOptions = () => getContestOptions();
  window.showContestPicker = function(sessionId = activeSessionId) {
    const opts = getContestOptions(sessionId);
    if (!opts.length) return false;
    const ctxSummary = `<div class="small text-muted">${opts.length} option(s)</div>`;
    if (sessionId === activeSessionId) {
      pipelineControl?.markAttention('resolve');
      pipelineControl?.focusStep('resolve', { scroll: false, highlight: true });
      logPanelControl?.expand();
    }

    const restoreKey = sessionId ? `contest:manual:${sessionId}` : 'contest:manual';
    const restoreTitle = 'Contest Selection';
    const restoreDetail = 'Finish choosing the contest for this parser session.';
    const restoreMessage = `${restoreTitle}. ${restoreDetail}`;
    modalRestore.register({
      key: restoreKey,
      sessionId,
      message: restoreMessage,
      title: restoreTitle,
      detail: restoreDetail,
      icon: '🎯',
      buttonLabel: 'Resume Contest',
      buttonTitle: 'Reopen the contest selection dialog',
      reopen: () => window.showContestPicker(sessionId)
    });
    modalRestore.markActive(restoreKey);

    showContestSelectionModal(
      'Select Contest',
      opts,
      ctxSummary,
      (selection) => {
        if (!selection || !selection.length) return;
        respondToPrompt(sessionId, selection.join(','));
        if (sessionId === activeSessionId) {
          pipelineControl?.clearAttention('resolve');
          pipelineControl?.setPhase('resolve');
        }
        modalRestore.clear(restoreKey);
      },
      {
        sessionId,
        onCancel: () => {
          modalRestore.markDismissed(restoreKey);
        },
        onSubmit: () => {
          modalRestore.clear(restoreKey);
        }
      }
    );
    return true;
  };
  let lastPromptContext = null;
  const urlPromptContextBySession = new Map();
  let lastSentSourceBySession = {};
  const sessionSourceMeta = new Map();
  let activeManualSource = 'input';
  let activeManualSourceOrigin = 'default';
  let pendingManualSource = null;
  let pipelineControl = null;
  let logPanelControl = null;

  // Folder browser state
  const ROOT_LABELS = { uploads: 'Uploads', input: 'Input', output: 'Output' };

  function normalizeManualSource(source) {
    return source === 'uploads' ? 'uploads' : 'input';
  }

  function normalizeManualOrigin(origin, source) {
    const normalizedSource = normalizeManualSource(source);
    const lowered = typeof origin === 'string' ? origin.toLowerCase() : '';
    if (lowered === 'user') {
      return normalizedSource === 'input' ? 'default' : 'user';
    }
    if (lowered === 'server') {
      return normalizedSource === 'input' ? 'default' : 'server';
    }
    if (lowered === 'default') {
      return 'default';
    }
    return normalizedSource === 'uploads' ? 'user' : 'default';
  }

  function clearSessionSourceMeta(sessionId, { fromServer = false } = {}) {
    if (!sessionId) return;
    sessionSourceMeta.delete(sessionId);
    if (fromServer) delete lastSentSourceBySession[sessionId];
    if (sessionId === activeSessionId) {
      activeManualSource = 'input';
      activeManualSourceOrigin = 'default';
      syncManualSourceUI();
      updatePipelineMetadataForActive();
      updatePipelineHintForPhase();
    }
  }

  function updateSessionSourceMeta(sessionId, source, origin, { fromServer = false } = {}) {
    const normalizedSource = normalizeManualSource(source);
    const normalizedOrigin = normalizeManualOrigin(origin, normalizedSource);

    if (!sessionId) {
      pendingManualSource = { source: normalizedSource, origin: normalizedOrigin };
      activeManualSource = normalizedSource;
      activeManualSourceOrigin = normalizedOrigin;
      syncManualSourceUI();
      updatePipelineMetadataForActive();
      updatePipelineHintForPhase();
      return;
    }

    sessionSourceMeta.set(sessionId, { source: normalizedSource, origin: normalizedOrigin });
    if (fromServer) {
      lastSentSourceBySession[sessionId] = normalizedSource;
    }
    if (sessionId === activeSessionId) {
      activeManualSource = normalizedSource;
      activeManualSourceOrigin = normalizedOrigin;
      pendingManualSource = null;
      syncManualSourceUI();
      updatePipelineMetadataForActive();
      updatePipelineHintForPhase();
    }
  }

  function ingestSessionSourceMeta(sessionId, meta, { fromServer = true } = {}) {
    if (!sessionId) return;
    if (!meta) {
      clearSessionSourceMeta(sessionId, { fromServer });
      return;
    }
    const source = meta.manual_source ?? meta.file_source ?? meta.source;
    const origin = meta.manual_source_origin ?? meta.source_origin ?? meta.origin;
    updateSessionSourceMeta(sessionId, source, origin, { fromServer });
  }

  function syncManualSourceUI() {
    const desired = activeManualSource === 'uploads' ? 'uploads' : 'input';
    if (el.fileSourceSelect && el.fileSourceSelect.value !== desired) {
      el.fileSourceSelect.value = desired;
    }
    syncSourceClass();
  }

  function ensureManualUploadOption(relPath, label) {
    if (!el.manualUploadSelect) return;
    const value = relPath || '';
    if (!value) return;
    const exists = Array.from(el.manualUploadSelect.options || []).some(opt => opt.value === value);
    if (exists) return;
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label || value;
    opt.dataset.manual = '1';
    el.manualUploadSelect.appendChild(opt);
  }

  function syncManualUploadControls() {
    if (!el.manualUploadSelect) return;
    const rel = manualUploadSelection ? manualUploadSelection.relPath : '';
    ensureManualUploadOption(rel, manualUploadSelection?.name);
    if (el.manualUploadSelect.value !== rel) {
      el.manualUploadSelect.value = rel || '';
    }
    if (el.manualUploadSummary) {
      const summary = manualUploadSelection
        ? `Selected: ${manualUploadSelection.relPath}`
        : 'No file selected.';
      el.manualUploadSummary.textContent = summary;
    }
    if (el.manualUploadClearBtn) {
      el.manualUploadClearBtn.disabled = !manualUploadSelection;
    }
  }

  async function refreshManualUploads({ preserveSelection = true, silent = false } = {}) {
    if (!el.manualUploadSelect) return;
    try {
      const res = await fetch('/api/fs/list?root=uploads', { headers: { 'Cache-Control': 'no-store' } });
      const data = await res.json().catch(() => ({}));
      const basePath = (data.path || '').replace(/\\/g, '/').replace(/^\//, '').trim();
      manualUploadsInventory = Array.isArray(data.entries)
        ? data.entries
            .filter(item => item && item.type === 'file')
            .map(item => {
              const rel = basePath ? `${basePath}/${item.name}` : item.name;
              return {
                name: item.name,
                relPath: rel.replace(/\\/g, '/'),
              };
            })
        : [];
      noteUploadsPresence(manualUploadsInventory.length > 0);
      const select = el.manualUploadSelect;
      const currentRel = (preserveSelection && manualUploadSelection) ? manualUploadSelection.relPath : '';
      const optionsHtml = manualUploadsInventory
        .map(entry => `<option value="${esc(entry.relPath)}">${esc(entry.name)}</option>`)
        .join('');
      select.innerHTML = `<option value="">— No file selected —</option>${optionsHtml}`;
      if (currentRel && manualUploadsInventory.every(entry => entry.relPath !== currentRel)) {
        ensureManualUploadOption(currentRel, manualUploadSelection?.name);
      }
      syncManualUploadControls();
      if (!silent && manualUploadSelection && manualUploadsInventory.every(entry => entry.relPath !== manualUploadSelection.relPath)) {
        // Notification when previously selected file is missing
        el.manualUploadSummary?.classList.add('text-warning');
      } else {
        el.manualUploadSummary?.classList.remove('text-warning');
      }
    } catch (err) {
      manualUploadsInventory = [];
      noteUploadsPresence(false);
      syncManualUploadControls();
      if (!silent) {
        el.manualUploadSummary && (el.manualUploadSummary.textContent = 'Unable to load uploads list.');
      }
    }
  }

  function applyManualUploadSelection(selection, { updateSource = true } = {}) {
    if (!selection || !selection.relPath) {
      manualUploadSelection = null;
      if (activeSessionId) manualUploadSelectionBySession.delete(activeSessionId);
      else pendingManualUploadSelection = null;
      syncManualUploadControls();
      el.manualUploadSummary?.classList.remove('text-warning');
      noteUploadsPresence(manualUploadsInventory.length > 0);
      return;
    }
    const cleanRel = selection.relPath.replace(/\\/g, '/').replace(/^\//, '');
    const name = selection.name || cleanRel.split('/').pop() || cleanRel;
    manualUploadSelection = { relPath: cleanRel, name };
  if (activeSessionId) manualUploadSelectionBySession.set(activeSessionId, manualUploadSelection);
  else pendingManualUploadSelection = manualUploadSelection;
    ensureManualUploadOption(cleanRel, name);
    syncManualUploadControls();
    el.manualUploadSummary?.classList.remove('text-warning');
  noteUploadsPresence(true);
    if (updateSource !== false) {
      updateSessionSourceMeta(activeSessionId, 'uploads', 'user');
      emitManualFileSource();
      updatePipelineMetadataForActive();
      updatePipelineHintForPhase();
    }
  }

  function parseDirectUrlField() {
    const raw = (el.directUrlTextarea?.value || '')
      .split(/\r?\n/)
      .map(line => line.split(','))
      .flat()
      .map(part => part.trim())
      .filter(Boolean);
    const valid = [];
    const invalid = [];
    raw.forEach(url => {
      if (isUserSuppliedSafeHttpUrl(url)) valid.push(url);
      else invalid.push(url);
    });
    return { valid, invalid };
  }

  function updateDirectUrlFeedback() {
    if (!el.directUrlFeedback) return;
    const { valid, invalid } = parseDirectUrlField();
    let message = 'Enter one URL per line.';
    const classList = el.directUrlFeedback.classList;
    classList.remove('text-danger');
    classList.add('text-muted');
    if (valid.length) {
      message = `Will run ${valid.length} URL${valid.length === 1 ? '' : 's'}.`;
    }
    if (valid.length > MAX_DIRECT_URLS) {
      message = `Limit ${MAX_DIRECT_URLS} URLs (currently ${valid.length}).`;
      classList.add('text-danger');
      classList.remove('text-muted');
    } else if (invalid.length) {
      const sample = invalid.slice(0, 2).join(', ');
      const more = invalid.length > 2 ? ` (+${invalid.length - 2} more)` : '';
      message = `Invalid URL${invalid.length === 1 ? '' : 's'}: ${sample}${more}`;
      classList.add('text-danger');
      classList.remove('text-muted');
    }
    el.directUrlFeedback.textContent = message;
  }

  function initManualUploadControl() {
    if (!el.manualUploadSelect) return;
    refreshManualUploads({ preserveSelection: true, silent: true }).catch(() => {});
    on(el.manualUploadRefreshBtn, 'click', () => {
      refreshManualUploads({ preserveSelection: true }).catch(() => {});
    });
    on(el.manualUploadBrowseBtn, 'click', () => {
      showFolderBrowser('uploads', '', (sel) => {
        if (!sel) return;
        const rel = sel.path ? `${sel.path}/${sel.name}` : sel.name;
        applyManualUploadSelection({ relPath: rel, name: sel.name });
      });
    });
    on(el.manualUploadClearBtn, 'click', () => {
      applyManualUploadSelection(null, { updateSource: false });
    });
    on(el.manualUploadSelect, 'change', () => {
      const rel = el.manualUploadSelect.value || '';
      if (!rel) {
        applyManualUploadSelection(null, { updateSource: false });
        return;
      }
      const entry = manualUploadsInventory.find(item => item.relPath === rel);
      applyManualUploadSelection({ relPath: rel, name: entry?.name || rel.split('/').pop() || rel });
    });
  }

  function initDirectUrlControl() {
    if (!el.directUrlTextarea) return;
    updateDirectUrlFeedback();
    if (!activeSessionId && el.directUrlTextarea.value) {
      pendingDirectUrlDraft = el.directUrlTextarea.value;
    }
    on(el.directUrlTextarea, 'input', () => {
      if (activeSessionId) directUrlDraftBySession.set(activeSessionId, el.directUrlTextarea.value);
      else pendingDirectUrlDraft = el.directUrlTextarea.value;
      updateDirectUrlFeedback();
    });
    on(el.directUrlClearBtn, 'click', () => {
      el.directUrlTextarea.value = '';
      if (activeSessionId) directUrlDraftBySession.set(activeSessionId, '');
      else pendingDirectUrlDraft = '';
      updateDirectUrlFeedback();
    });
  }

  // Keep Run button in sync with backend session lock state
  function updateRunButtonLock() {
    if (!el.runBtn) return;
  const meta = sessionMirror.get(activeSessionId) || {};
  const state = String(meta.state || '').toLowerCase();
    const locked = !!meta.locked;
    const busyStates = new Set([
      SessionEnums.state.RUNNING,
      SessionEnums.state.WAITING_PROMPT,
      SessionEnums.state.CANCELLING,
    ]);
    const shouldDisable = locked || busyStates.has(state);
    if (shouldDisable) {
      el.runBtn.disabled = true;
      el.runBtn.classList.add('btn-locked');
      el.runBtn.setAttribute('data-running','true');
    } else {
      el.runBtn.disabled = false;
      el.runBtn.classList.remove('btn-locked');
      el.runBtn.removeAttribute('data-running');
    }
  }

  function applySessionState(sessionId, metaOverride) {
    if (!sessionId) return;
    const meta = metaOverride || sessionMirror.get(sessionId);
    if (!meta) return;
  const state = String(meta.state || '').toLowerCase();
    const mappedPhase = meta.phase || SessionEnums.stateToPhase[state];
    if (sessionId === activeSessionId && mappedPhase) {
      setPipelinePhase(mappedPhase, { force: true });
      ensureResolveAttention(state === SessionEnums.state.WAITING_PROMPT);
      updateRunButtonLock();
    }
  }
  // Track joined rooms & early log queue
  let joinedSessions = new Set();
  let earlyQueue = [];

  window.getActiveSessionId = () => activeSessionId;

  // -------- DOM refs --------
  const el = {
    fileSourceSelect: $('#fileSourceSelect'),
    bypassBtn: $('#toggleOutputBypassBtn'),
    runBtn: $('#runParserBtn'),
    cancelBtn: $('#cancelParserBtn'),
    outputDiv: $('#terminal'),
    promptInput: $('#promptInput'),
    promptForm: $('#promptForm'),
    logFilterSelect: $('#logFilterSelect'),
    outputModeSelect: $('#outputModeSelect'),
    pipelineStepper: $('#pipelineStepper'),
    logPanel: document.querySelector('.log-panel'),
    logPanelBody: $('#logPanelBody'),
    logToggleBtn: $('#toggleLogPanelBtn'),
  pipelineHint: $('#pipelineHint'),
    sessionFooter: $('#sessionFooter'),
    footerPreview: $('#footerPreview'),
    footerFull: $('#footerFull'),
    sessionList: $('#sessionList'),
    addSessionBtn: $('#addSessionBtn'),
    activeSessionSpan: $('#activeSessionId'),
    sessionCount: $('#sessionCount'),
    addUrlForm: $('#addUrlForm'),
    urlList: $('#urlList'),
    saveFiltersBtn: $('#saveFiltersBtn'),
    exportFiltersBtn: $('#exportFiltersBtn'),
    filterPresetSelect: $('#filterPresetSelect'),
    deletePresetBtn: $('#deletePresetBtn'),
    manualUploadSelect: $('#manualUploadSelect'),
    manualUploadSummary: $('#manualUploadSummary'),
    manualUploadRefreshBtn: $('#manualUploadRefreshBtn'),
    manualUploadBrowseBtn: $('#manualUploadBrowseBtn'),
    manualUploadClearBtn: $('#manualUploadClearBtn'),
    directUrlTextarea: $('#directUrlTextarea'),
    directUrlFeedback: $('#directUrlFeedback'),
    directUrlClearBtn: $('#directUrlClearBtn'),
    modalAddDirectUrl: $('#modalAddDirectUrl'),
    modalAddManualUpload: $('#modalAddManualUpload'),
    modalQuickAddContainer: $('#modalQuickAddContainer'),
    modalQuickAddDirectPanel: $('#modalQuickAddDirectPanel'),
    modalQuickAddManualPanel: $('#modalQuickAddManualPanel'),
    modalQuickAddMessage: $('#modalQuickAddMessage'),
    modalQuickAddDirectAdd: $('#modalQuickAddDirectAdd'),
    modalQuickAddDirectCancel: $('#modalQuickAddDirectCancel'),
    modalQuickAddManualDone: $('#modalQuickAddManualDone'),
    modalQuickAddUploadInput: $('#modalQuickAddUploadInput')
  };

  let modalQuickAddContext = null;
  let modalQuickAddHideTimer = null;

  function setModalQuickAddContext(ctx) {
    modalQuickAddContext = ctx || null;
    const mode = modalQuickAddContext?.mode || null;
    if (el.modalAddDirectUrl) {
      el.modalAddDirectUrl.classList.toggle('hidden', mode !== 'url');
    }
    if (el.modalAddManualUpload) {
      el.modalAddManualUpload.classList.toggle('hidden', mode !== 'manual');
    }
    if (!ctx) hideModalQuickAdd();
  }

  function showModalQuickAdd(mode) {
    if (!el.modalQuickAddContainer) return;
    clearTimeout(modalQuickAddHideTimer);
    el.modalQuickAddContainer.classList.remove('hidden');
    if (mode === 'url') {
      el.modalQuickAddDirectPanel?.classList.remove('hidden');
      el.modalQuickAddDirectPanel?.setAttribute('aria-hidden', 'false');
      el.modalQuickAddManualPanel?.classList.add('hidden');
      el.modalQuickAddManualPanel?.setAttribute('aria-hidden', 'true');
      setTimeout(() => el.directUrlTextarea?.focus(), 10);
    } else if (mode === 'manual') {
      el.modalQuickAddManualPanel?.classList.remove('hidden');
      el.modalQuickAddManualPanel?.setAttribute('aria-hidden', 'false');
      el.modalQuickAddDirectPanel?.classList.add('hidden');
      el.modalQuickAddDirectPanel?.setAttribute('aria-hidden', 'true');
      setTimeout(() => el.manualUploadSelect?.focus(), 10);
    }
  }

  function hideModalQuickAdd({ resetMessage = true } = {}) {
    if (!el.modalQuickAddContainer) return;
    el.modalQuickAddContainer.classList.add('hidden');
    el.modalQuickAddDirectPanel?.classList.add('hidden');
    el.modalQuickAddDirectPanel?.setAttribute('aria-hidden', 'true');
    el.modalQuickAddManualPanel?.classList.add('hidden');
    el.modalQuickAddManualPanel?.setAttribute('aria-hidden', 'true');
    if (resetMessage) setModalQuickAddMessage();
    clearTimeout(modalQuickAddHideTimer);
  }

  function setModalQuickAddMessage(message = '', tone = 'info') {
    if (!el.modalQuickAddMessage) return;
    const node = el.modalQuickAddMessage;
    node.classList.add('hidden');
    node.classList.remove('alert-info', 'alert-success', 'alert-danger', 'alert-warning');
    if (!message) {
      node.textContent = '';
      return;
    }
    const toneClass = tone === 'success'
      ? 'alert-success'
      : tone === 'danger'
        ? 'alert-danger'
        : tone === 'warning'
          ? 'alert-warning'
          : 'alert-info';
    node.classList.add('alert', toneClass);
    node.textContent = message;
    node.classList.remove('hidden');
    clearTimeout(modalQuickAddHideTimer);
    modalQuickAddHideTimer = setTimeout(() => {
      if (!node.classList.contains('hidden')) node.classList.add('hidden');
    }, 4000);
  }

  function addUrlOptionsToContext(urls) {
    const ctx = modalQuickAddContext;
    if (!ctx || ctx.mode !== 'url' || !Array.isArray(urls) || !urls.length) return 0;
    const processed = ctx.processedMap || {};
    const sanitized = ctx.sanitized;
    let added = 0;
    urls.forEach(url => {
      if (!sanitized.includes(url)) {
        sanitized.push(url);
        added += 1;
      }
    });
    if (!added) return 0;
    cacheUrlPromptContext(ctx.sessionId, sanitized, processed, ctx.meta);
    const options = buildUrlOptions(sanitized, processed);
    ctx.controller?.updateOptions(options);
    ctx.options = options;
    return added;
  }

  function handleModalQuickAddDirectAdd() {
    const ctx = modalQuickAddContext;
    if (!ctx || ctx.mode !== 'url') return;
    const { valid, invalid } = parseDirectUrlField();
    if (!valid.length) {
      setModalQuickAddMessage(
        invalid.length ? 'Fix invalid URLs before adding them.' : 'Enter at least one URL to add.',
        invalid.length ? 'warning' : 'info'
      );
      return;
    }
    const added = addUrlOptionsToContext(valid);
    if (!added) {
      setModalQuickAddMessage('All URLs already exist in this list.', 'info');
      return;
    }
    setModalQuickAddMessage(`Added ${added} URL${added === 1 ? '' : 's'} to selection list.`, 'success');
  }

  async function handleModalQuickAddUploadSelect(event) {
    const ctx = modalQuickAddContext;
    if (!ctx || ctx.mode !== 'manual') return;
    const input = event?.target;
    const file = input?.files?.[0];
    if (!file) return;
    try {
      setModalQuickAddMessage(`Uploading ${file.name}...`, 'info');
      const formData = new FormData();
      formData.append('data_file', file);
      const resp = await fetch('/upload/uploads', { method: 'POST', body: formData });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const relPath = file.name.replace(/\\/g, '/');
      applyManualUploadSelection({ relPath, name: file.name });
      await refreshManualUploads({ preserveSelection: true });
      setModalQuickAddMessage(`Uploaded '${file.name}' to manual uploads.`, 'success');
      if (!manualUploadSelection || manualUploadSelection.relPath !== relPath) {
        applyManualUploadSelection({ relPath, name: file.name });
      }
    } catch (err) {
      setModalQuickAddMessage(`Upload failed: ${err.message || err}`, 'danger');
    } finally {
      if (input) input.value = '';
    }
  }

  function initModalQuickAdd() {
    if (el.modalAddDirectUrl && !el.modalAddDirectUrl.dataset.quickAddBound) {
      el.modalAddDirectUrl.dataset.quickAddBound = '1';
      on(el.modalAddDirectUrl, 'click', () => {
        if (modalQuickAddContext?.mode === 'url') {
          showModalQuickAdd('url');
          setModalQuickAddMessage();
        }
      });
    }
    if (el.modalAddManualUpload && !el.modalAddManualUpload.dataset.quickAddBound) {
      el.modalAddManualUpload.dataset.quickAddBound = '1';
      on(el.modalAddManualUpload, 'click', () => {
        if (modalQuickAddContext?.mode === 'manual') {
          showModalQuickAdd('manual');
          setModalQuickAddMessage();
        }
      });
    }
    if (el.modalQuickAddDirectAdd && !el.modalQuickAddDirectAdd.dataset.quickAddBound) {
      el.modalQuickAddDirectAdd.dataset.quickAddBound = '1';
      on(el.modalQuickAddDirectAdd, 'click', handleModalQuickAddDirectAdd);
    }
    if (el.modalQuickAddDirectCancel && !el.modalQuickAddDirectCancel.dataset.quickAddBound) {
      el.modalQuickAddDirectCancel.dataset.quickAddBound = '1';
      on(el.modalQuickAddDirectCancel, 'click', () => hideModalQuickAdd());
    }
    if (el.modalQuickAddManualDone && !el.modalQuickAddManualDone.dataset.quickAddBound) {
      el.modalQuickAddManualDone.dataset.quickAddBound = '1';
      on(el.modalQuickAddManualDone, 'click', () => hideModalQuickAdd());
    }
    if (el.modalQuickAddUploadInput && !el.modalQuickAddUploadInput.dataset.quickAddBound) {
      el.modalQuickAddUploadInput.dataset.quickAddBound = '1';
      on(el.modalQuickAddUploadInput, 'change', handleModalQuickAddUploadSelect);
    }
  }

  // Modal helper (single reusable modal shell)
  const Modal = {
    FOCUSABLE_SELECTOR: 'a[href], area[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    RESTORE_TABINDEX_ATTR: 'data-modal-restore-tabindex',
    get() {
      let modal = document.getElementById('downloadModal');
      if (!modal) {
        modal = document.createElement('div');
        modal.id = 'downloadModal';
        modal.className = 'modal fade';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.setAttribute('aria-labelledby', 'downloadModalTitle');
        modal.innerHTML = `
          <div class="modal-dialog modal-lg">
            <div class="modal-content">
              <div class="modal-header">
                <h5 class="modal-title" id="downloadModalTitle">Select URL</h5>
                <div class="modal-header-actions ms-auto d-flex gap-2">
                  <button type="button" class="btn btn-sm btn-outline-secondary hidden" id="modalAddDirectUrl">Add Direct URL</button>
                  <button type="button" class="btn btn-sm btn-outline-secondary hidden" id="modalAddManualUpload">Upload File</button>
                  <button type="button" class="btn-close" id="closeDownloadModal" aria-label="Close"></button>
                </div>
              </div>
              <div class="modal-body">
                <div id="modalQuickAddContainer" class="modal-quick-add hidden" aria-live="polite">
                  <div id="modalQuickAddDirectPanel" class="quick-add-section hidden" aria-hidden="true">
                    <label for="directUrlTextarea" class="dropdown-label">
                      Direct URLs:
                      <span class="help-icon" tabindex="0" aria-label="Direct URLs">
                        ?
                        <span class="custom-tooltip">
                          Paste one or more election result URLs (HTTP or HTTPS). Each valid URL runs immediately and skips the urls.txt prompt.
                        </span>
                      </span>
                    </label>
                    <textarea id="directUrlTextarea" class="form-control" rows="3" placeholder="https://example.gov/results/latest"></textarea>
                    <div class="d-flex justify-content-between align-items-center mt-2 flex-wrap gap-2">
                      <span id="directUrlFeedback" class="text-muted small">Enter one URL per line.</span>
                      <div class="btn-group btn-group-sm" role="group" aria-label="Direct URL actions">
                        <button type="button" class="btn btn-outline-secondary" id="modalQuickAddDirectAdd">Add to List</button>
                        <button type="button" class="btn btn-outline-secondary" id="directUrlClearBtn">Clear</button>
                        <button type="button" class="btn btn-outline-secondary" id="modalQuickAddDirectCancel">Done</button>
                      </div>
                    </div>
                    <hr class="my-3">
                  </div>
                  <div id="modalQuickAddManualPanel" class="quick-add-section hidden" aria-hidden="true">
                    <div class="modal-inline-heading d-flex justify-content-between align-items-start mb-2 flex-wrap gap-2">
                      <label for="manualUploadSelect" class="dropdown-label mb-0">
                        Manual Upload File:
                        <span class="help-icon" tabindex="0" aria-label="Manual Upload File">
                          ?
                          <span class="custom-tooltip">
                            Pick a file from the uploads folder to parse immediately. This sets the source to Manual Uploads when you run the parser.
                          </span>
                        </span>
                      </label>
                      <div class="btn-group btn-group-sm" role="group" aria-label="Manual upload actions">
                        <button type="button" class="btn btn-outline-secondary" id="manualUploadRefreshBtn" title="Refresh uploads list">⟳</button>
                        <button type="button" class="btn btn-outline-secondary" id="manualUploadBrowseBtn" title="Browse uploads">Browse</button>
                        <label for="modalQuickAddUploadInput" class="btn btn-outline-secondary mb-0" title="Upload new file">Upload</label>
                        <input type="file" id="modalQuickAddUploadInput" class="visually-hidden" aria-label="Upload new file to manual uploads">
                      </div>
                    </div>
                    <select id="manualUploadSelect" class="form-select" aria-label="Select manual upload file">
                      <option value="">— No file selected —</option>
                    </select>
                    <div class="d-flex justify-content-between align-items-center mt-2 flex-wrap gap-2">
                      <span id="manualUploadSummary" class="text-muted small">No file selected.</span>
                      <div class="btn-group btn-group-sm" role="group" aria-label="Manual upload selection actions">
                        <button type="button" class="btn btn-outline-secondary" id="manualUploadClearBtn">Clear</button>
                        <button type="button" class="btn btn-outline-secondary" id="modalQuickAddManualDone">Done</button>
                      </div>
                    </div>
                    <hr class="my-3">
                  </div>
                  <div id="modalQuickAddMessage" class="alert hidden mt-2" role="status"></div>
                </div>
                <input type="search" id="downloadSearch" class="form-control mb-2" placeholder="Filter by keyword, type, or format...">
                <div id="downloadSummary" class="mb-2"></div>
                <div id="downloadOptions"></div>
              </div>
              <div class="modal-footer">
                <button type="button" class="btn btn-secondary" id="cancelDownloadModal">Cancel</button>
              </div>
            </div>
          </div>`;
        document.body.appendChild(modal);
        // Refresh handles to newly created elements and bind listeners
        el.modalAddDirectUrl = $('#modalAddDirectUrl');
        el.modalAddManualUpload = $('#modalAddManualUpload');
        el.modalQuickAddContainer = $('#modalQuickAddContainer');
        el.modalQuickAddDirectPanel = $('#modalQuickAddDirectPanel');
        el.modalQuickAddManualPanel = $('#modalQuickAddManualPanel');
        el.modalQuickAddMessage = $('#modalQuickAddMessage');
        el.modalQuickAddDirectAdd = $('#modalQuickAddDirectAdd');
        el.modalQuickAddDirectCancel = $('#modalQuickAddDirectCancel');
        el.modalQuickAddManualDone = $('#modalQuickAddManualDone');
        el.modalQuickAddUploadInput = $('#modalQuickAddUploadInput');
        el.manualUploadSelect = $('#manualUploadSelect');
        el.manualUploadSummary = $('#manualUploadSummary');
        el.manualUploadRefreshBtn = $('#manualUploadRefreshBtn');
        el.manualUploadBrowseBtn = $('#manualUploadBrowseBtn');
        el.manualUploadClearBtn = $('#manualUploadClearBtn');
        el.directUrlTextarea = $('#directUrlTextarea');
        el.directUrlFeedback = $('#directUrlFeedback');
        el.directUrlClearBtn = $('#directUrlClearBtn');
        initManualUploadControl();
        initDirectUrlControl();
        initModalQuickAdd();
      }
      modal.removeAttribute('aria-hidden');
      if (!modal.dataset.modalA11yBound) {
        modal.dataset.modalA11yBound = '1';
        const clearAriaHidden = () => {
          try { modal.removeAttribute('aria-hidden'); } catch (err) { void err; }
        };
        const setInertState = (value) => {
          try {
            if ('inert' in modal) {
              modal.inert = !!value;
            }
            if (value) {
              modal.setAttribute('inert', '');
              modal.setAttribute('data-modal-inert', '1');
            } else {
              modal.removeAttribute('inert');
              modal.removeAttribute('data-modal-inert');
            }
          } catch (err) {
            void err;
          }
          if (value) {
            modal.querySelectorAll(Modal.FOCUSABLE_SELECTOR).forEach(node => {
              if (node.hasAttribute('tabindex')) return;
              node.setAttribute(Modal.RESTORE_TABINDEX_ATTR, '1');
              node.setAttribute('tabindex', '-1');
            });
          } else {
            modal.querySelectorAll(`[${Modal.RESTORE_TABINDEX_ATTR}]`).forEach(node => {
              node.removeAttribute('tabindex');
              node.removeAttribute(Modal.RESTORE_TABINDEX_ATTR);
            });
          }
        };
        const prepareHideState = () => {
          const active = document.activeElement;
          if (active && modal.contains(active) && typeof active.blur === 'function') {
            try { active.blur(); } catch (err) { void err; }
          }
          setInertState(true);
        };
        modal.addEventListener('show.bs.modal', () => {
          setInertState(false);
          clearAriaHidden();
        });
        modal.addEventListener('shown.bs.modal', () => {
          setInertState(false);
          clearAriaHidden();
        });
        modal.addEventListener('hide.bs.modal', () => {
          clearAriaHidden();
          prepareHideState();
        });
        modal.addEventListener('hidden.bs.modal', () => {
          clearAriaHidden();
          setInertState(true);
        });
        clearAriaHidden();
        setInertState(true);
      }
      return {
        modal,
        titleEl: modal.querySelector('.modal-title'),
        searchEl: document.getElementById('downloadSearch'),
        optionsDiv: document.getElementById('downloadOptions'),
        summaryDiv: document.getElementById('downloadSummary'),
        closeBtn: document.getElementById('closeDownloadModal'),
        cancelBtn: document.getElementById('cancelDownloadModal')
      };
    },
    open() {
      const { modal } = this.get();
      hideModalQuickAdd();
      setModalQuickAddMessage();
      const inst = window.bootstrap?.Modal.getOrCreateInstance(modal, { keyboard: true, backdrop: true });
      inst.show();
    },
    close() {
      const { modal } = this.get();
      const inst = window.bootstrap?.Modal.getOrCreateInstance(modal);
      if (modal && modal.contains(document.activeElement)) {
        try { document.activeElement.blur(); } catch (err) { void err; }
      }
      inst?.hide();
      setModalQuickAddContext(null);
      hideModalQuickAdd();
    }
  };

  const PendingOverlay = (() => {
    let bannerEl = null;
    let textEl = null;
    let hideTimer = null;
    let minVisibleUntil = 0;

    function resolveHost() {
      const modalBanner = document.getElementById('modalRestoreBanner');
      if (modalBanner && modalBanner.parentElement) return modalBanner.parentElement;
      const hint = document.getElementById('pipelineHint');
      if (hint && hint.parentElement) return hint.parentElement;
      const main = document.querySelector('.container-main');
      if (main) return main;
      return document.body || document.documentElement;
    }

    function placeBanner(host) {
      if (!host || !bannerEl) return;
      const modalBanner = document.getElementById('modalRestoreBanner');
      const hint = document.getElementById('pipelineHint');
      if (modalBanner && modalBanner.parentElement === host) {
        host.insertBefore(bannerEl, modalBanner);
        return;
      }
      if (hint && hint.parentElement === host) {
        host.insertBefore(bannerEl, hint.nextSibling);
        return;
      }
      if (host.firstChild) host.insertBefore(bannerEl, host.firstChild);
      else host.appendChild(bannerEl);
    }

    function ensureBanner() {
      const host = resolveHost();
      if (!bannerEl) {
        bannerEl = document.createElement('div');
        bannerEl.id = 'appPendingBanner';
        bannerEl.className = 'app-pending-banner hidden';
        bannerEl.setAttribute('role', 'status');
        bannerEl.setAttribute('aria-live', 'polite');
        bannerEl.setAttribute('aria-hidden', 'true');
        bannerEl.innerHTML = `
          <div class="app-pending-banner-shell">
            <div class="app-pending-spinner" aria-hidden="true"></div>
            <div class="app-pending-text">Please wait…</div>
          </div>`;
        textEl = bannerEl.querySelector('.app-pending-text');
        placeBanner(host);
      } else if (!bannerEl.isConnected || (host && bannerEl.parentElement !== host)) {
        placeBanner(host);
      }
      return bannerEl;
    }

    function scheduleHide(delay = 0) {
      if (hideTimer) clearTimeout(hideTimer);
      if (delay <= 0) {
        hideTimer = null;
        performHide(true);
      } else {
        hideTimer = setTimeout(() => performHide(true), delay);
      }
    }

    function performHide(force = false) {
      const el = ensureBanner();
      const now = Date.now();
      if (!force && now < minVisibleUntil) {
        scheduleHide(minVisibleUntil - now);
        return;
      }
      el.classList.add('hidden');
      el.setAttribute('aria-hidden', 'true');
      if (hideTimer) {
        clearTimeout(hideTimer);
        hideTimer = null;
      }
      minVisibleUntil = 0;
    }

    function show(message = 'Please wait…', options = {}) {
      const { minimumMs = 400, autoHideMs = 0 } = options || {};
      const el = ensureBanner();
      if (textEl) textEl.textContent = message;
      el.classList.remove('hidden');
      el.removeAttribute('aria-hidden');
      const now = Date.now();
      minVisibleUntil = Math.max(minVisibleUntil, now + Math.max(0, minimumMs));
      if (hideTimer) {
        clearTimeout(hideTimer);
        hideTimer = null;
      }
      if (autoHideMs && autoHideMs > 0) {
        scheduleHide(autoHideMs);
      }
    }

    function hide() {
      performHide(false);
    }

    return { show, hide };
  })();
  window.PendingOverlay = PendingOverlay;

  function ensureModalRestoreStyles() {
    // Styles now live in webapp/static/css/run_parser.css to satisfy strict CSS loading policies.
  }

  const modalRestore = (() => {
    const contexts = new Map();
    const pendingTimers = new Map();
    let bannerCtx = null;
    let resizeHandler = null;
    let scrollHandler = null;
    let viewportHandler = null;
    let footerObserver = null;
    let anchorObserver = null;
    let observedAnchor = null;
    let lastContainer = null;
    let anchorElement = null;

    const debug = (...args) => {
      if (typeof window !== 'undefined' && window.DEBUG_MODAL_RESTORE) {
        try { console.debug('[modalRestore]', ...args); } catch (err) { void err; }
      }
    };

    const coerceText = (val, fallback = '') => {
      if (typeof val === 'string') {
        const trimmed = val.trim();
        if (trimmed) return trimmed;
      }
      return fallback;
    };

    function deriveTitleFromMessage(message) {
      const raw = coerceText(message);
      if (!raw) return 'Resume Work';
      const firstSentence = raw.split(/[.!?]/)[0] || raw;
      const cleaned = firstSentence
        .replace(/reopen it here/gi, '')
        .replace(/window closed/gi, '')
        .replace(/dialog closed/gi, '')
        .replace(/closed$/i, '')
        .trim();
      if (!cleaned) return 'Resume Work';
      return cleaned.replace(/\s+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    }

    function clearPending(key) {
      if (!key) return;
      const pending = pendingTimers.get(key);
      if (!pending) return;
      if (pending.timeout) {
        clearTimeout(pending.timeout);
      }
      if (pending.modal && pending.listener) {
        try { pending.modal.removeEventListener('hidden.bs.modal', pending.listener); } catch (err) { void err; }
      }
      pendingTimers.delete(key);
    }

    function measureFooterHeight() {
      const footer = document.getElementById('sessionFooter');
      if (!footer) return 0;
      const rect = footer.getBoundingClientRect();
      return Math.max(0, rect.height || (window.innerHeight - rect.top));
    }

    function getSafeInsets() {
      const vv = window.visualViewport;
      if (!vv) {
        return { top: 0, right: 0, bottom: 0, left: 0 };
      }
      const top = Math.max(0, vv.offsetTop || 0);
      const left = Math.max(0, vv.offsetLeft || 0);
      const right = Math.max(0, window.innerWidth - (vv.offsetLeft + vv.width));
      const bottom = Math.max(0, window.innerHeight - (vv.offsetTop + vv.height));
      return { top, right, bottom, left };
    }

    function resolveAnchor() {
      if (anchorElement && document.contains(anchorElement)) return anchorElement;
      const selectors = ['[data-modal-anchor="true"]', '.container-main', '.layout-flex', '#pipelineStepper'];
      for (const sel of selectors) {
        const candidate = document.querySelector(sel);
        if (candidate) {
          anchorElement = candidate;
          if (candidate instanceof HTMLElement && !candidate.dataset.modalAnchor) {
            candidate.dataset.modalAnchor = 'true';
          }
          return anchorElement;
        }
      }
      anchorElement = document.body;
      return anchorElement;
    }

    function relocateBanner(container) {
      if (!container) return;
      const hint = document.getElementById('pipelineHint');
      const main = document.querySelector('.container-main');
      const host = hint?.parentElement || main || document.body || document.documentElement;
      if (!host) return;
      const reference = hint && hint.parentElement === host ? hint.nextSibling : null;
      if (reference) {
        if (reference !== container) host.insertBefore(container, reference);
      } else if (container.parentElement !== host) {
        host.appendChild(container);
      }
    }

    function applyBannerPlacement(container) {
      if (!container) return;
      relocateBanner(container);
      container.classList.add('modal-restore-banner-hosted');
    }

    function ensurePlacementHooks(container) {
      if (!container) return;
      lastContainer = container;
      applyBannerPlacement(container);
    }

    function ensureUI() {
      ensureModalRestoreStyles();
      let container = document.getElementById('modalRestoreBanner');
      if (!container) {
        const host = document.body || document.documentElement;
        if (!host) return null;
        container = document.createElement('div');
        container.id = 'modalRestoreBanner';
        container.className = 'modal-restore-banner hidden';
        container.tabIndex = -1;
        container.setAttribute('role', 'status');
        container.setAttribute('aria-live', 'polite');
        const messageId = 'modalRestoreMessage';
        const titleId = 'modalRestoreTitle';
        const detailId = 'modalRestoreDetail';
        container.setAttribute('aria-describedby', messageId);
        container.setAttribute('aria-labelledby', titleId);
        container.innerHTML = `
          <span class="modal-restore-message" id="${messageId}">Dialog closed. Reopen it here.</span>
          <div class="modal-restore-shell" role="group" aria-labelledby="${titleId}" aria-describedby="${messageId} ${detailId}">
            <div class="modal-restore-badge" aria-hidden="true">↺</div>
            <div class="modal-restore-copy">
              <div class="modal-restore-title" id="${titleId}">Dialog paused</div>
              <div class="modal-restore-detail" id="${detailId}">Reopen to continue where you left off.</div>
            </div>
            <div class="modal-restore-actions">
              <button type="button" class="modal-restore-reopen" aria-describedby="${messageId} ${detailId}">
                <span class="modal-restore-text text-shimmer">Reopen</span>
              </button>
              <button type="button" class="modal-restore-dismiss" aria-label="Dismiss restore banner"></button>
            </div>
          </div>`;
        host.appendChild(container);
      }
      relocateBanner(container);
      const messageEl = container.querySelector('.modal-restore-message');
      const reopenBtn = container.querySelector('.modal-restore-reopen');
      const dismissBtn = container.querySelector('.modal-restore-dismiss');
      const titleEl = container.querySelector('.modal-restore-title');
      const detailEl = container.querySelector('.modal-restore-detail');
      const badgeEl = container.querySelector('.modal-restore-badge');
      const reopenTextEl = reopenBtn ? reopenBtn.querySelector('.modal-restore-text') : null;
      if (titleEl && !titleEl.classList.contains('text-shimmer-soft')) {
        titleEl.classList.add('text-shimmer-soft');
      }
      if (detailEl && !detailEl.classList.contains('text-shimmer-soft')) {
        detailEl.classList.add('text-shimmer-soft');
      }
      if (messageEl && !messageEl.classList.contains('text-shimmer-soft')) {
        messageEl.classList.add('text-shimmer-soft');
      }
      if (reopenTextEl && !reopenTextEl.classList.contains('text-shimmer')) {
        reopenTextEl.classList.add('text-shimmer');
      }
      if (titleEl && !titleEl.id) {
        titleEl.id = 'modalRestoreTitle';
      }
      if (detailEl && !detailEl.id) {
        detailEl.id = 'modalRestoreDetail';
      }
      if (messageEl && reopenBtn) {
        if (!messageEl.id) {
          messageEl.id = 'modalRestoreMessage';
        }
        if (!reopenBtn.getAttribute('aria-describedby')) {
          reopenBtn.setAttribute('aria-describedby', messageEl.id);
        }
        if (!container.getAttribute('aria-describedby')) {
          container.setAttribute('aria-describedby', messageEl.id);
        }
        if (!container.getAttribute('aria-labelledby') && titleEl && titleEl.id) {
          container.setAttribute('aria-labelledby', titleEl.id);
        }
      }
      ensurePlacementHooks(container);
      return { container, messageEl, reopenBtn, dismissBtn, titleEl, detailEl, badgeEl, reopenTextEl };
    }

    function hideBanner() {
      const ui = ensureUI();
      if (!ui) return;
      ui.container.classList.add('hidden');
      debug('banner:hidden');
    }

    function showBanner(ctx) {
      const ui = ensureUI();
      if (!ui) return;
      bannerCtx = ctx;
      ensurePlacementHooks(ui.container);
      const visibleMessage = coerceText(ctx.message, 'Dialog closed. Reopen it here.');
      const heading = coerceText(ctx.title, deriveTitleFromMessage(visibleMessage));
      const baseDetail = coerceText(ctx.detail, visibleMessage);
      const busyMessage = coerceText(ctx.busyMessage, baseDetail);
      const isBusy = ctx.busy === true;
      const baseButtonLabel = coerceText(ctx.buttonLabel, 'Reopen');
      const busyButtonLabel = coerceText(ctx.busyButtonLabel, 'Please wait…');
      const buttonLabel = isBusy ? busyButtonLabel : baseButtonLabel;
      const displayDetail = isBusy ? busyMessage : baseDetail;
      const announcement = coerceText(ctx.announcement, `${heading}. ${displayDetail}`);
      const icon = coerceText(ctx.icon, '↺');
      if (ui.messageEl) ui.messageEl.textContent = announcement;
      if (ui.titleEl) ui.titleEl.textContent = heading;
      if (ui.detailEl) ui.detailEl.textContent = displayDetail;
      if (ui.badgeEl) ui.badgeEl.textContent = icon;
      if (ui.reopenTextEl) ui.reopenTextEl.textContent = buttonLabel;
      if (ui.reopenBtn) {
        ui.reopenBtn.setAttribute('title', coerceText(ctx.buttonTitle, displayDetail || heading));
        ui.reopenBtn.setAttribute('aria-label', `${buttonLabel}: ${heading}`);
        ui.reopenBtn.disabled = isBusy;
        ui.reopenBtn.classList.toggle('busy', isBusy);
        if (isBusy) {
          ui.reopenBtn.setAttribute('aria-disabled', 'true');
          ui.reopenBtn.setAttribute('aria-busy', 'true');
        } else {
          ui.reopenBtn.removeAttribute('aria-disabled');
          ui.reopenBtn.removeAttribute('aria-busy');
        }
      }
      if (ui.dismissBtn) {
        ui.dismissBtn.setAttribute('aria-label', `Dismiss ${heading} restore banner`);
      }
      ui.container.classList.remove('hidden');
      requestAnimationFrame(() => {
        applyBannerPlacement(ui.container);
        if (ctx.scrollIntoView !== false) {
          try { ui.container.scrollIntoView({ behavior: 'smooth', block: 'center' }); } catch (err) { void err; }
        }
        const focusTarget = (!isBusy && ui.reopenBtn) ? ui.reopenBtn : ui.container;
        try { focusTarget.focus({ preventScroll: true }); } catch (err) { void err; }
      });
      if (ui.reopenBtn) {
        ui.reopenBtn.onclick = () => {
          const latest = contexts.get(ctx.key);
          if (latest && latest.busy) return;
          bannerCtx = null;
          hideBanner();
          contexts.set(ctx.key, ctx);
          try { ctx.reopen && ctx.reopen(); } catch (err) { void err; }
        };
      }
      if (ui.dismissBtn) {
        ui.dismissBtn.onclick = () => {
          hideBanner();
          if (bannerCtx && bannerCtx.key === ctx.key) {
            bannerCtx = null;
          }
          contexts.delete(ctx.key);
        };
      }
      debug('banner:shown', ctx.key);
    }

    return {
      register(input) {
        if (!input || !input.key) return null;
        const message = coerceText(input.message, 'Dialog closed. Reopen it here.');
        const ctx = {
          key: input.key,
          message,
          detail: coerceText(input.detail),
          title: coerceText(input.title),
          icon: coerceText(input.icon),
          buttonLabel: coerceText(input.buttonLabel),
          buttonTitle: coerceText(input.buttonTitle),
          announcement: coerceText(input.announcement),
          scrollIntoView: input.scrollIntoView !== false,
          sessionId: input.sessionId || null,
          busy: input.busy === true,
          busyMessage: coerceText(input.busyMessage),
          busyButtonLabel: coerceText(input.busyButtonLabel),
          reopen: () => {
            hideBanner();
            bannerCtx = null;
            try { input.reopen && input.reopen(); } catch (err) { void err; }
          }
        };
        contexts.set(ctx.key, ctx);
        debug('register', ctx.key);
        return ctx;
      },
      markActive(key) {
        if (!key) return;
        clearPending(key);
        if (bannerCtx && bannerCtx.key === key) {
          debug('markActive:hide', key);
          bannerCtx = null;
          hideBanner();
        } else {
          debug('markActive:noop', key);
        }
      },
      markDismissed(key) {
        if (!key) return;
        const ctx = contexts.get(key);
        if (!ctx) return;
        bannerCtx = ctx;
        clearPending(key);

        const finish = () => {
          const activeKey = bannerCtx && bannerCtx.key;
          clearPending(key);
          if (!activeKey || activeKey !== key) {
            debug('markDismissed:skip-show', key);
            return;
          }
          const emit = () => showBanner(ctx);
          if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(emit);
          } else {
            emit();
          }
        };

        let delayMs = 360;
        let modalEl = null;
        if (typeof document !== 'undefined') {
          modalEl = document.getElementById('downloadModal');
        }
        if (modalEl && typeof window !== 'undefined') {
          try {
            const parsed = (value) => {
              if (!value) return 0;
              const segment = String(value).split(',')[0].trim();
              if (segment.endsWith('ms')) return parseFloat(segment);
              if (segment.endsWith('s')) return parseFloat(segment) * 1000;
              return parseFloat(segment) || 0;
            };
            const styles = window.getComputedStyle(modalEl);
            const transition = parsed(styles.transitionDuration);
            const delay = parsed(styles.transitionDelay);
            delayMs = Math.max(280, Math.round(transition + delay + 200));
          } catch (err) {
            void err;
          }
        }

        if (modalEl) {
          const listener = () => finish();
          modalEl.addEventListener('hidden.bs.modal', listener, { once: true });
          const timeout = setTimeout(listener, delayMs);
          pendingTimers.set(key, { timeout, modal: modalEl, listener });
          debug('markDismissed:await-hidden', key, delayMs);
        } else {
          const timeout = setTimeout(() => finish(), delayMs);
          pendingTimers.set(key, { timeout });
          debug('markDismissed:await-timeout', key, delayMs);
        }
      },
      clear(key) {
        if (!key) return;
        clearPending(key);
        contexts.delete(key);
        if (bannerCtx && bannerCtx.key === key) {
          debug('clear:hide', key);
          bannerCtx = null;
          hideBanner();
        }
      },
      setBusy(key, busy, options = {}) {
        if (!key) return;
        const ctx = contexts.get(key);
        if (!ctx) return;
        const wasBusy = ctx.busy === true;
        ctx.busy = !!busy;
        if (options.message !== undefined) {
          ctx.busyMessage = coerceText(options.message, ctx.busyMessage || ctx.detail || ctx.message);
        }
        if (options.buttonLabel !== undefined) {
          ctx.busyButtonLabel = coerceText(options.buttonLabel, ctx.busyButtonLabel || ctx.buttonLabel || 'Please wait…');
        }
        contexts.set(key, ctx);
        if ((ctx.busy || wasBusy) && bannerCtx && bannerCtx.key === key) {
          showBanner(ctx);
        }
      },
      setBusyForSession(sessionId, busy, options = {}) {
        if (!sessionId) return;
        contexts.forEach((ctx) => {
          if (ctx.sessionId && ctx.sessionId === sessionId) {
            this.setBusy(ctx.key, busy, options);
          }
        });
      }
    };
  })();

  // Compact text-input modal for naming folders
  const NameModal = {
    get() {
      let modal = document.getElementById('nameModal');
      if (!modal) {
        modal = document.createElement('div');
        modal.id = 'nameModal';
        modal.className = 'modal fade';
        modal.innerHTML = `
          <div class="modal-dialog modal-sm" role="dialog" aria-modal="true" aria-labelledby="nameModalTitle">
            <div class="modal-content">
              <div class="modal-header">
                <h5 class="modal-title" id="nameModalTitle">New Folder</h5>
                <button type="button" class="btn-close" id="nameModalClose" aria-label="Close"></button>
              </div>
              <div class="modal-body">
                <label for="nameModalInput" class="form-label">Folder name</label>
                <input type="text" id="nameModalInput" class="neon-input" placeholder="e.g. 2024_results">
                <div class="form-hint">Avoid / \\ : * ? " < > | and trailing dots.</div>
                <div id="nameModalError" class="form-error hidden"></div>
              </div>
              <div class="modal-footer">
                <button type="button" class="btn btn-secondary" id="nameModalCancel">Cancel</button>
                <button type="button" class="btn btn-primary btn-gradient-neon" id="nameModalOk">Create</button>
              </div>
            </div>
          </div>`;
        document.body.appendChild(modal);
      }
      return {
        modal,
        titleEl: modal.querySelector('.modal-title'),
        inputEl: document.getElementById('nameModalInput'),
        errEl: document.getElementById('nameModalError'),
        closeBtn: document.getElementById('nameModalClose'),
        cancelBtn: document.getElementById('nameModalCancel'),
        okBtn: document.getElementById('nameModalOk')
      };
    },
    open() {
      const { modal } = this.get();
      const inst = window.bootstrap?.Modal.getOrCreateInstance(modal, { keyboard: true, backdrop: true });
      inst.show();
    },
    close() {
      const { modal } = this.get();
      const inst = window.bootstrap?.Modal.getOrCreateInstance(modal);
      inst?.hide();
    }
  };

  function sanitizeFolderName(s) {
    if (typeof s !== 'string') return '';
    // Strip invalid filesystem chars, trim spaces, disallow trailing dots
    let v = s.replace(/[\\/:*?"<>|]/g, '').trim();
    while (v.endsWith('.')) v = v.slice(0, -1);
    return v.slice(0, 80);
  }

  function askFolderName({ title = 'New Folder', placeholder = 'e.g. reports_2024', defaultValue = '' } = {}, cb) {
    const refs = NameModal.get();
    refs.titleEl.textContent = title;
    refs.inputEl.value = defaultValue || '';
    refs.inputEl.placeholder = placeholder || '';
    refs.errEl.classList.add('hidden');
    refs.errEl.textContent = '';

    const done = (val) => { NameModal.close(); cb && cb(val); cleanup(); };
    const fail = (msg) => { refs.errEl.textContent = msg; refs.errEl.classList.remove('hidden'); };

    function onConfirm() {
      const raw = refs.inputEl.value;
      const name = sanitizeFolderName(raw);
      if (!name) return fail('Please enter a valid folder name.');
      done(name);
    }
    function onKey(e) {
      if (e.key === 'Enter') { e.preventDefault(); onConfirm(); }
      if (e.key === 'Escape') { e.preventDefault(); done(null); }
    }
    function cleanup() {
      refs.okBtn.removeEventListener('click', onConfirm);
      refs.cancelBtn.removeEventListener('click', () => done(null));
      refs.closeBtn.removeEventListener('click', () => done(null));
      refs.inputEl.removeEventListener('keydown', onKey);
      refs.modal.removeEventListener('hidden.bs.modal', onHidden);
    }
    function onCancel() { done(null); }
    function onHidden() { cleanup(); }
    refs.okBtn.addEventListener('click', onConfirm);
    refs.cancelBtn.addEventListener('click', onCancel);
    refs.closeBtn.addEventListener('click', onCancel);
    refs.inputEl.addEventListener('keydown', onKey);
    refs.modal.addEventListener('hidden.bs.modal', onHidden);

    NameModal.open();
    setTimeout(() => refs.inputEl.focus(), 0);
  }
  // --- Download Selection Modal Logic ---
  function showDownloadModal(options, summary, callback) {
    const refs = Modal.get();
    if (!refs) return callback?.(null);
    const { titleEl, searchEl, optionsDiv, summaryDiv, closeBtn, cancelBtn } = refs;
    if (titleEl) titleEl.textContent = 'Select Download';
    let filtered = options.slice();
    let submitted = false;
    function submitOnce(val) {
      if (submitted) return;
      submitted = true;
      // Accessibility: move focus out of modal before hiding to avoid aria-hidden warnings
      try { document.activeElement && document.activeElement.blur && document.activeElement.blur(); } catch {}
      Modal.close();
      callback(val);
    }
    function renderList(filter = '') {
      const q = filter.trim().toLowerCase();
      filtered = options.filter(opt =>
        opt.format.toLowerCase().includes(q) ||
        opt.filename.toLowerCase().includes(q) ||
        opt.contest.toLowerCase().includes(q)
      );
      const groups = {};
      filtered.forEach(opt => {
        const key = opt.contest || 'Other';
        if (!groups[key]) groups[key] = [];
        groups[key].push(opt);
      });
      optionsDiv.innerHTML = '';
      optionsDiv.classList.remove('table-preview-container');
      optionsDiv.classList.add('table-preview-container');
      Object.keys(groups).sort().forEach(group => {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'download-group';
        groupDiv.innerHTML = `<div class="download-group-header"><b>${group}</b> (${groups[group].length})</div>`;
        groups[group].forEach((opt) => {
          const item = document.createElement('div');
          item.className = 'download-option';
          item.tabIndex = 0;
          item.innerHTML = `<span class="badge bg-primary me-2">${opt.format.toUpperCase()}</span>
            <span class="download-filename">${highlight(opt.filename, q)}</span>
            <span class="download-type ms-2">${highlight(opt.contest, q)}</span>`;
          item.onclick = () => { submitOnce(opt.index); };
          item.onkeydown = e => { if (e.key === 'Enter') { submitOnce(opt.index); } };
          groupDiv.appendChild(item);
        });
        optionsDiv.appendChild(groupDiv);
      });
    }
    function highlight(text, q) {
      if (!q) return esc(text);
      return esc(text).replace(new RegExp(q, 'gi'), m => `<mark>${m}</mark>`);
    }
    function hide() { Modal.close(); }
    searchEl.value = '';
    summaryDiv.textContent = summary || '';
    renderList();
    searchEl.oninput = e => renderList(e.target.value);
    closeBtn.onclick = cancelBtn.onclick = hide;
    Modal.open();
    searchEl.focus();
  }

  function openTableStructurePreview(sessionId, state) {
    const data = cloneTablePreviewState(state || getTablePreviewState(sessionId));
    if (!data.entries.length) {
      alert('No table previews captured yet for this session.');
      return;
    }
    const refs = Modal.get();
    if (!refs) return;
    const { modal, titleEl, searchEl, optionsDiv, summaryDiv, closeBtn, cancelBtn } = refs;
    titleEl.textContent = 'Table Structure Preview';
    searchEl.value = '';
    searchEl.placeholder = 'Search columns or values…';
    summaryDiv.innerHTML = `<div class="small text-muted">${esc(data.contest || 'Contest Pending')} • ${data.entries.length} candidate${data.entries.length === 1 ? '' : 's'}</div>`;

    function renderTable(filter = '') {
      const q = filter.trim().toLowerCase();
      optionsDiv.innerHTML = '';
      data.entries.forEach(entry => {
        const headers = entry.headers || [];
        const rows = entry.rows || [];
        if (q) {
          const headerMatch = headers.some(h => h.toLowerCase().includes(q));
          const rowMatch = rows.some(row => Object.values(row || {}).some(v => String(v).toLowerCase().includes(q)));
          if (!headerMatch && !rowMatch) return;
        }
        const card = document.createElement('div');
        card.className = 'table-preview-card';
        const confidenceText = typeof entry.confidence === 'number' ? `conf ${entry.confidence.toFixed(2)}` : 'confidence n/a';
        card.innerHTML = `
          <header class="table-preview-header">
            <div class="table-preview-title">Candidate ${entry.index}/${entry.total || data.entries.length}</div>
            <div class="table-preview-meta">${esc(confidenceText)} • ${headers.length} columns</div>
          </header>
        `;
        const table = document.createElement('table');
        table.className = 'table-preview-grid';
        const thead = document.createElement('thead');
        const headRow = document.createElement('tr');
        headers.forEach(h => {
          const th = document.createElement('th');
          th.textContent = h;
          headRow.appendChild(th);
        });
        thead.appendChild(headRow);
        table.appendChild(thead);
        const tbody = document.createElement('tbody');
        rows.forEach(row => {
          const tr = document.createElement('tr');
          headers.forEach(h => {
            const td = document.createElement('td');
            td.textContent = row && h in row ? String(row[h]) : '';
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
        if (!rows.length) {
          const tr = document.createElement('tr');
          const td = document.createElement('td');
          td.colSpan = Math.max(headers.length, 1);
          td.className = 'table-preview-empty';
          td.textContent = 'No sample rows available.';
          tr.appendChild(td);
          tbody.appendChild(tr);
        }
        table.appendChild(tbody);
        card.appendChild(table);
        optionsDiv.appendChild(card);
      });
      if (!optionsDiv.children.length) {
        const empty = document.createElement('div');
        empty.className = 'contest-empty';
        empty.textContent = 'No table previews match your search.';
        optionsDiv.appendChild(empty);
      }
    }

    const close = () => Modal.close();
    closeBtn.onclick = cancelBtn.onclick = close;
    if (modal) {
      modal.addEventListener('hide.bs.modal', close, { once: true });
    }
    searchEl.oninput = e => renderTable(e.target.value || '');
    renderTable();
    Modal.open();
    setTimeout(() => searchEl.focus(), 0);
  }

  function showContestSelectionModal(title, options, ctxSummaryHtml, onSelect, extras = {}) {
    const refs = Modal.get();
    if (!refs) {
      if (typeof onSelect === 'function') onSelect(null);
      return;
    }
    const { modal, titleEl, searchEl, optionsDiv, summaryDiv, closeBtn, cancelBtn } = refs;
    const placeholder = extras.placeholder || 'Search contests…';
    const onCancel = typeof extras.onCancel === 'function' ? extras.onCancel : null;
    const onSubmit = typeof extras.onSubmit === 'function' ? extras.onSubmit : null;
    const sessionForModal = extras.sessionId || activeSessionId;

    const baseOptions = cloneContestOptions(options || []);
    const summaryIsDefault = !ctxSummaryHtml;
    const baseSummaryHtml = ctxSummaryHtml || `<div class="small text-muted">${baseOptions.length} option(s)</div>`;
    let tableState = getTablePreviewState(sessionForModal);
    let closed = false;
    let lastContestFilter = '';
    let lastTableFilter = '';
    const bundleChildren = new Map();
    const bundleFamilies = new Map();
    const bundleAssignments = new Map();
    const optionOrder = new Map();
    const optionByIndex = new Map();
    const expandedOffices = new Set();
    let viewMode = 'contest';

    const handlePreviewUpdated = (event) => {
      if (!event || !event.detail) return;
      const { sessionId } = event.detail;
      if (sessionId !== sessionForModal) return;
      refreshTableState();
      if (viewMode === 'table') {
        renderTableCards(lastTableFilter);
      } else {
        renderContestList(currentContestList);
      }
    };

    document.addEventListener('table-preview:updated', handlePreviewUpdated);
    const cleanup = () => {
      document.removeEventListener('table-preview:updated', handlePreviewUpdated);
    };

    baseOptions.forEach((opt, idx) => {
      if (!opt || opt.index == null) return;
      if (!optionOrder.has(opt.index)) optionOrder.set(opt.index, idx);
      if (!optionByIndex.has(opt.index)) optionByIndex.set(opt.index, opt);
    });

    const preferredBundleKeys = [
      'bundle_key', 'bundleKey', 'bundle_id', 'bundleId', 'bundle_slug', 'bundleSlug',
      'bundle_hash', 'bundleHash', 'bundle_group', 'bundleGroup', 'bundle_parent_key', 'bundleParentKey',
      'bundle', 'bundle_uid', 'bundleUid', 'bundle_guid', 'bundleGuid',
      'group_key', 'groupKey', 'group_id', 'groupId', 'cluster_key', 'clusterKey',
      'aggregate_key', 'aggregateKey', 'family_key', 'familyKey', 'contest_group_id', 'contestGroupId',
      'contest_bundle', 'contestBundle'
    ];

    const truthyKeys = ['bundle_anchor', 'bundle_primary', 'is_primary', 'primary', 'preferred'];

    function canonicalize(text) {
      return String(text || '').toLowerCase().replace(/\s+/g, ' ').trim();
    }

    function rawOfficeLabel(opt) {
      if (!opt || typeof opt !== 'object') return '';
      const meta = opt.metadata || {};
      const prefer = meta.display_header || meta.bundle_office_label;
      if (prefer) return String(prefer);
      const office = meta.office_title || meta.office || meta.officeName || meta.primary_title;
      if (office) return String(office);
      const label = String(opt.label || '');
      const head = label.includes(' – ') ? label.split(' – ')[0] : label;
      return head || label;
    }

    function normalizeOfficeTitle(text) {
      if (!text) return '';
      let value = stripContestSuffix(String(text));
      value = value.replace(/\s*\(\s*\d+\s+(?:contest|race|entry|variation)s?\s*\)\s*$/i, '');
      value = value.replace(/\s*(?:,|-)?\s*(?:district|dist\.|place|seat|position|ward|post)\s*(?:no\.?\s*)?[#\-]?\s*\d+[a-z]?$/gi, '');
      value = value.replace(/\s*(?:,|-)?\s*(?:subdistrict|sub-?division|area|division)\s*(?:no\.?\s*)?[#\-]?\s*\d+[a-z]?$/gi, '');
      value = value.replace(/\s*(?:,|-)?\s*(?:group|zone|precinct)\s*(?:no\.?\s*)?[#\-]?\s*\d+[a-z]?$/gi, '');
      value = value.replace(/\s+/g, ' ').trim();
      if (!value) value = String(text).trim();
      return value;
    }

    function deriveOfficeTitle(opt) {
      const raw = rawOfficeLabel(opt) || 'Other';
      const normalized = normalizeOfficeTitle(raw);
      return normalized || raw || 'Other';
    }

    function deriveOfficeKey(opt) {
      if (!opt || typeof opt !== 'object') return 'other';
      const meta = opt.metadata || {};
      if (meta.bundle_office_key) return canonicalize(meta.bundle_office_key);
      return canonicalize(deriveOfficeTitle(opt)) || 'other';
    }

    function resolveExplicitBundleKey(meta) {
      if (!meta || typeof meta !== 'object') return '';
      for (const key of preferredBundleKeys) {
        if (!(key in meta)) continue;
        const value = meta[key];
        if (value == null || value === '') continue;
        const str = String(value).trim();
        if (str) return str;
      }
      for (const key of Object.keys(meta)) {
        const value = meta[key];
        if (value == null || value === '') continue;
        if (typeof value !== 'string' && typeof value !== 'number') continue;
        if (!/bundle|cluster|family|group|aggregate|package|set|pack/i.test(key)) continue;
        const str = String(value).trim();
        if (str) return str;
      }
      return '';
    }

    function resolveFallbackBundleKey(opt) {
      if (!opt) return '';
      const meta = opt.metadata || {};
      const label = canonicalize(opt.label);
      if (!label) return '';
      const office = canonicalize(meta.office_title || meta.office || meta.officeName);
      const scope = canonicalize(meta.scope_label || meta.scope || meta.division_scope);
      const variant = canonicalize(meta.variant_label || meta.variant);
      const year = canonicalize(meta.year || meta.election_year);
      const contest = canonicalize(meta.contest_title || meta.contest_name || meta.primary_title);
      const countyCount = Array.isArray(meta.counties) ? meta.counties.filter(Boolean).length : 0;
      const parts = [label, office, scope, variant, year, contest];
      if (countyCount > 1) parts.push(`counties:${countyCount}`);
      const key = parts.filter(Boolean).join('|');
      return key ? `label:${key}` : '';
    }

    function ensureFamily(key, priority = 1) {
      if (!bundleFamilies.has(key)) {
        bundleFamilies.set(key, { key, priority, members: [] });
      }
      const family = bundleFamilies.get(key);
      if (priority === 0 && family.priority !== 0) family.priority = 0;
      return family;
    }

    function addToFamily(key, opt, priority = 1) {
      if (!key || !opt) return;
      const family = ensureFamily(key, priority);
      if (!family.members.includes(opt)) {
        family.members.push(opt);
      }
      if (priority === 0) {
        bundleAssignments.set(opt.index, key);
      } else if (priority === 1 && !bundleAssignments.has(opt.index)) {
        bundleAssignments.set(opt.index, key);
      }
    }

    function hasTruthy(meta, keys) {
      if (!meta || typeof meta !== 'object') return false;
      return keys.some(key => {
        if (!(key in meta)) return false;
        const value = meta[key];
        if (typeof value === 'boolean') return value;
        if (typeof value === 'number') return value !== 0;
        if (typeof value === 'string') {
          const trimmed = value.trim().toLowerCase();
          if (!trimmed) return false;
          return trimmed !== 'false' && trimmed !== '0' && trimmed !== 'no';
        }
        return !!value;
      });
    }

    function pickAggregateOption(options) {
      if (!options.length) return null;
      const ranked = options.map(opt => {
        const meta = opt.metadata || {};
        const order = optionOrder.has(opt.index) ? optionOrder.get(opt.index) : Number.MAX_SAFE_INTEGER;
        const mode = String(meta.bundle_mode || '').toLowerCase();
        let score = 3;
        if (mode === 'aggregate') score = 0;
        else if (hasTruthy(meta, truthyKeys)) score = 1;
        else if (!meta.bundle_member && !hasTruthy(meta, ['bundle_member'])) score = 2;
        return { opt, score, order };
      });
      ranked.sort((a, b) => {
        if (a.score !== b.score) return a.score - b.score;
        return a.order - b.order;
      });
      return ranked[0]?.opt || options[0];
    }

    function ensureArrayField(meta, key) {
      if (!meta || typeof meta !== 'object') return [];
      const current = meta[key];
      if (Array.isArray(current)) return current;
      if (current == null || current === '') {
        meta[key] = [];
        return meta[key];
      }
      meta[key] = [current];
      return meta[key];
    }

    function stripContestSuffix(label) {
      if (!label) return '';
      return String(label).replace(/\s*\(\s*\d+\s+contest[s]?\s*\)\s*$/i, '').trim();
    }

    function shouldAutoBundleFallback(opt) {
      if (!opt || typeof opt !== 'object') return false;
      const meta = opt.metadata || {};
      const mode = String(meta.bundle_mode || '').toLowerCase();
      if (mode === 'aggregate' || meta.bundle_member || hasTruthy(meta, truthyKeys)) return true;
      const contestTypes = [];
      if (Array.isArray(meta.contest_types)) contestTypes.push(...meta.contest_types);
      if (meta.contest_type) contestTypes.push(meta.contest_type);
      const normalizedTypes = contestTypes.map(val => String(val).toLowerCase());
      if (normalizedTypes.some(type => type.includes('candidate') || type.includes('office'))) return true;
      const office = String(meta.office_title || meta.office || '').toLowerCase();
      if (office) {
        if (/commissioner|representative|senator|council|board|judge|attorney|sheriff|mayor|superintendent/.test(office)) return true;
      }
      const scope = String(meta.scope_label || '').toLowerCase();
      if (scope && scope !== 'statewide') return true;
      const counties = Array.isArray(meta.counties) ? meta.counties.filter(Boolean) : [];
      if (counties.length > 1) return true;
      const divisions = Array.isArray(meta.division_identifiers) ? meta.division_identifiers.filter(Boolean) : [];
      if (divisions.length > 1) return true;
      return false;
    }

    function attachOfficeAggregateChildren() {
      const officeAggregates = new Map();
      const officeMembers = new Map();

      baseOptions.forEach(opt => {
        if (!opt || opt.index == null) return;
        const officeKey = deriveOfficeKey(opt);
        if (!officeKey) return;
        if (!officeMembers.has(officeKey)) officeMembers.set(officeKey, []);
        officeMembers.get(officeKey).push(opt);
        if (isAggregateOption(opt)) {
          if (!officeAggregates.has(officeKey)) officeAggregates.set(officeKey, []);
          officeAggregates.get(officeKey).push(opt);
        }
      });

      officeAggregates.forEach((aggregateList, officeKey) => {
        const roster = officeMembers.get(officeKey) || [];
        const nonAggregates = roster.filter(entry => !isAggregateOption(entry));
        if (!nonAggregates.length) return;

        const sortedAggregates = aggregateList.slice().sort((a, b) => {
          const sizeA = Number((a.metadata || {}).bundle_size) || 0;
          const sizeB = Number((b.metadata || {}).bundle_size) || 0;
          if (sizeA !== sizeB) return sizeB - sizeA;
          return (optionOrder.get(a.index) || 0) - (optionOrder.get(b.index) || 0);
        });

        const remaining = new Set(nonAggregates);
        sortedAggregates.forEach(aggregate => {
          const meta = aggregate.metadata || (aggregate.metadata = {});
          const existingChildren = (bundleChildren.get(aggregate.index) || []).filter(Boolean);
          existingChildren.forEach(child => remaining.delete(child));

          const newlyAssigned = [];
          remaining.forEach(child => {
            if (child === aggregate) return;
            const childMeta = child.metadata || (child.metadata = {});
            if (childMeta.bundle_member && childMeta.bundle_parent_index != null && childMeta.bundle_parent_index !== aggregate.index) {
              return;
            }
            newlyAssigned.push(child);
          });

          if (!existingChildren.length && !newlyAssigned.length) return;

          const combined = existingChildren.concat(newlyAssigned);
          if (!combined.length) return;

          const bundleKey = meta.bundle_key || meta.aggregate_key || meta.family_key || `office:${officeKey}`;
          meta.bundle_key = bundleKey;
          bundleChildren.set(aggregate.index, combined);
          meta.bundle_child_count = combined.length;

          const inferredTotal = Number(meta.bundle_size);
          const totalForSummary = Number.isFinite(inferredTotal) && inferredTotal >= combined.length + 1
            ? inferredTotal
            : combined.length + 1;
          meta.bundle_size = totalForSummary;

          combined.forEach(child => {
            const childMeta = child.metadata || (child.metadata = {});
            childMeta.bundle_member = true;
            childMeta.bundle_parent_index = aggregate.index;
            if (!childMeta.bundle_key) childMeta.bundle_key = bundleKey;
            if (!childMeta.bundle_mode) childMeta.bundle_mode = 'member';
            remaining.delete(child);
          });

          applyAggregatePresentation(aggregate, combined, totalForSummary);
        });
      });
    }

    function insertUniqueLine(list, line, front = false) {
      if (!Array.isArray(list)) return;
      const str = String(line || '').trim();
      if (!str) return;
      const lower = str.toLowerCase();
      const existingIndex = list.findIndex(entry => String(entry).toLowerCase() === lower);
      if (existingIndex !== -1) {
        if (front && existingIndex !== 0) {
          const [existing] = list.splice(existingIndex, 1);
          list.unshift(existing);
        }
        return;
      }
      if (front) list.unshift(str);
      else list.push(str);
    }

    function applyAggregatePresentation(aggregate, childList, expectedTotal) {
      if (!aggregate) return;
      const children = Array.isArray(childList) ? childList.filter(Boolean) : [];
      if (!children.length) return;
      const meta = aggregate.metadata || (aggregate.metadata = {});
      const inferredTotal = Number.isFinite(expectedTotal) && expectedTotal > 0 ? Number(expectedTotal) : children.length + 1;
      const totalContests = inferredTotal;

      const unionCounties = new Set(Array.isArray(meta.counties) ? meta.counties.filter(Boolean).map(String) : []);
      const scopeSet = new Set();
      const divisionSet = new Set();
      const variantSet = new Set();

      const collect = (entryMeta) => {
        if (!entryMeta || typeof entryMeta !== 'object') return;
        if (Array.isArray(entryMeta.counties)) {
          entryMeta.counties.filter(Boolean).forEach(val => unionCounties.add(String(val)));
        }
        const scopePrimary = entryMeta.scope_label || entryMeta.scope;
        if (scopePrimary) scopeSet.add(String(scopePrimary));
        if (Array.isArray(entryMeta.scope_labels)) {
          entryMeta.scope_labels.filter(Boolean).forEach(val => scopeSet.add(String(val)));
        }
        if (Array.isArray(entryMeta.division_identifiers)) {
          entryMeta.division_identifiers.filter(Boolean).forEach(val => divisionSet.add(String(val)));
        }
        if (entryMeta.variant_label) variantSet.add(String(entryMeta.variant_label));
        if (Array.isArray(entryMeta.variant_labels)) {
          entryMeta.variant_labels.filter(Boolean).forEach(val => variantSet.add(String(val)));
        }
      };

      collect(meta);
      children.forEach(child => collect(child.metadata || (child.metadata = {})));

      meta.counties = Array.from(unionCounties);
      if (meta.counties.length) {
        meta.county_label = `${meta.counties.length} county${meta.counties.length === 1 ? '' : 'ies'}`;
      }

      const totalCounties = unionCounties.size;
      const summaryParts = [`${totalContests} contest${totalContests === 1 ? '' : 's'} grouped`];
      if (totalCounties > 1) summaryParts.push(`${totalCounties} counties`);
      else if (totalCounties === 1) summaryParts.push('1 county');
      if (divisionSet.size > 1) summaryParts.push(`${divisionSet.size} districts`);
      if (scopeSet.size === 1) summaryParts.push(`${Array.from(scopeSet)[0]} scope`);
      else if (scopeSet.size > 1) summaryParts.push(`${scopeSet.size} scope types`);
      if (variantSet.size > 1) summaryParts.push(`${variantSet.size} variants`);

      const summaryLine = summaryParts.join(' • ');
      meta.aggregate_summary_line = summaryLine;
      aggregate.meta = summaryLine;

      const displayDetails = ensureArrayField(meta, 'display_details');
      insertUniqueLine(displayDetails, summaryLine, true);
      const summaryList = ensureArrayField(meta, 'summary');
      insertUniqueLine(summaryList, summaryLine, true);
      insertUniqueLine(summaryList, `${totalContests} contest${totalContests === 1 ? '' : 's'}`);

      const childCount = children.length;
      meta.bundle_child_count = Math.max(childCount, totalContests - 1);

      const originalLabel = String(aggregate.label || '');
      const initialOffice = meta.office_title || meta.office || stripContestSuffix(originalLabel.split(' – ')[0]);
      const labelBase = initialOffice || stripContestSuffix(originalLabel) || 'Contest';
      const normalizedLabelBase = stripContestSuffix(labelBase);
      aggregate.label = `${normalizedLabelBase} (${totalContests} contest${totalContests === 1 ? '' : 's'})`;
      const finalOffice = initialOffice || normalizedLabelBase;
      if (finalOffice) {
        meta.office_title = finalOffice;
        meta.office = finalOffice;
        if (!meta.display_header) meta.display_header = finalOffice;
      }
    }

    function hydrateProvidedBundles() {
      baseOptions.forEach(opt => {
        if (!opt || opt.index == null) return;
        const meta = opt.metadata || {};
        const mode = String(meta.bundle_mode || '').toLowerCase();
        if (mode !== 'aggregate') return;
        const indices = Array.isArray(meta.bundle_member_indices) ? meta.bundle_member_indices : [];
        if (!indices.length) return;
        const existing = bundleChildren.get(opt.index) || [];
        const seen = new Set(existing);
        indices.forEach(idx => {
          const child = optionByIndex.get(idx);
          if (!child || child === opt) return;
          if (!seen.has(child)) {
            seen.add(child);
            existing.push(child);
          }
          const childMeta = child.metadata || (child.metadata = {});
          childMeta.bundle_member = true;
          childMeta.bundle_parent_index = opt.index;
          if (!childMeta.bundle_key && meta.bundle_key) childMeta.bundle_key = meta.bundle_key;
          if (!childMeta.bundle_mode) childMeta.bundle_mode = 'member';
        });
        if (existing.length) {
          bundleChildren.set(opt.index, existing);
          if (typeof meta.bundle_child_count !== 'number' || meta.bundle_child_count < existing.length) {
            meta.bundle_child_count = existing.length;
          }
          const expectedTotal = Number(meta.bundle_size);
          const totalForSummary = Number.isFinite(expectedTotal) && expectedTotal > 0 ? expectedTotal : existing.length + 1;
          meta.bundle_size = totalForSummary;
          applyAggregatePresentation(opt, existing, totalForSummary);
        }
      });

      baseOptions.forEach(opt => {
        if (!opt || opt.index == null) return;
        const meta = opt.metadata || {};
        if (!meta.bundle_member || meta.bundle_parent_index == null) return;
        if (!bundleChildren.has(meta.bundle_parent_index)) return;
        const siblings = bundleChildren.get(meta.bundle_parent_index);
        if (!siblings.includes(opt)) siblings.push(opt);
      });
    }

    function isAggregateOption(opt) {
      if (!opt || typeof opt !== 'object') return false;
      const meta = opt.metadata || {};
      const mode = String(meta.bundle_mode || '').toLowerCase();
      if (mode === 'aggregate') return true;
      if (hasTruthy(meta, ['bundle_summary', 'is_aggregate', 'aggregate', 'aggregate_entry'])) return true;
      const label = String(opt.label || '');
      if (/\(\s*\d+\s+(contest|race|entry)/i.test(label)) return true;
      if (/all\s+(counties|contests|districts|races)/i.test(label)) return true;
      return false;
    }

    baseOptions.forEach(opt => {
      if (!opt || opt.index == null) return;
      const key = resolveExplicitBundleKey(opt.metadata || {});
      if (!key) return;
      addToFamily(`explicit:${key}`, opt, 0);
    });

    for (const [key, family] of Array.from(bundleFamilies.entries())) {
      if (family.priority === 0 && (!family.members || family.members.length <= 1)) {
        family.members.forEach(opt => {
          if (bundleAssignments.get(opt.index) === key) {
            bundleAssignments.delete(opt.index);
          }
        });
        bundleFamilies.delete(key);
      }
    }

    baseOptions.forEach(opt => {
      if (!opt || opt.index == null) return;
      if (bundleAssignments.has(opt.index)) return;
      if (!shouldAutoBundleFallback(opt)) return;
      const key = resolveFallbackBundleKey(opt);
      if (!key) return;
      addToFamily(`fallback:${key}`, opt, 1);
    });

    for (const [key, family] of Array.from(bundleFamilies.entries())) {
      if (!family.members || family.members.length <= 1) {
        family.members.forEach(opt => {
          if (bundleAssignments.get(opt.index) === key) {
            bundleAssignments.delete(opt.index);
          }
        });
        bundleFamilies.delete(key);
      }
    }

    bundleFamilies.forEach(family => {
      if (!family.members || family.members.length <= 1) return;
      const aggregate = pickAggregateOption(family.members);
      if (!aggregate) return;
      const children = family.members.filter(opt => opt !== aggregate);
      if (!children.length) return;

      const aggregateMeta = aggregate.metadata || (aggregate.metadata = {});
      let bundleKey = resolveExplicitBundleKey(aggregateMeta);
      if (!bundleKey) bundleKey = family.key.replace(/^(explicit|fallback):/, '');
      aggregateMeta.bundle_mode = 'aggregate';
      aggregateMeta.bundle_key = bundleKey;
      aggregateMeta.bundle_child_count = children.length;
      delete aggregateMeta.bundle_member;
      delete aggregateMeta.bundle_parent_index;

      const unionCounties = new Set(Array.isArray(aggregateMeta.counties) ? aggregateMeta.counties.map(c => String(c)) : []);
      family.members.forEach(opt => {
        const meta = opt.metadata || {};
        if (Array.isArray(meta.counties)) {
          meta.counties.filter(Boolean).forEach(c => unionCounties.add(String(c)));
        }
      });
      aggregateMeta.counties = Array.from(unionCounties);

      const existingSize = Number(aggregateMeta.bundle_size);
      const totalForSummary = Number.isFinite(existingSize) && existingSize > 0 ? existingSize : family.members.length;
      aggregateMeta.bundle_size = totalForSummary;

      const details = ensureArrayField(aggregateMeta, 'display_details');
      const variationsLine = `${totalForSummary} contest variation${totalForSummary === 1 ? '' : 's'}`;
      insertUniqueLine(details, variationsLine, true);

      children.forEach(child => {
        const meta = child.metadata || (child.metadata = {});
        meta.bundle_member = true;
        meta.bundle_parent_index = aggregate.index;
        meta.bundle_key = bundleKey;
        if (!meta.bundle_mode) meta.bundle_mode = 'member';
      });

      applyAggregatePresentation(aggregate, children, totalForSummary);
      bundleChildren.set(aggregate.index, children.slice());
    });

    hydrateProvidedBundles();
    attachOfficeAggregateChildren();
    bundleChildren.forEach(list => list.sort((a, b) => String(a.label || '').localeCompare(String(b.label || ''))));
    const expandedBundles = new Set();
    const OFFICE_COLLAPSE_THRESHOLD = typeof extras.collapseThreshold === 'number'
      ? Math.max(3, Number(extras.collapseThreshold))
      : 18;
    let currentContestList = baseOptions.slice();

    function refreshTableState() {
      tableState = getTablePreviewState(sessionForModal);
      return tableState;
    }

    titleEl.textContent = title || 'Select Contest';
    searchEl.value = '';
    searchEl.placeholder = placeholder;

    function requiresGrouping(opt) {
      const meta = opt.metadata || {};
      if (meta.bundle_mode === 'aggregate') {
        return true;
      }
      const variants = Number(meta.variants || (Array.isArray(meta.contest_ids) ? meta.contest_ids.length : 0));
      if (variants > 1) return true;
      const counties = Array.isArray(meta.counties) ? meta.counties.filter(Boolean) : [];
      if (counties.length > 1) {
        const scope = (meta.scope_label || '').toLowerCase();
        if (scope !== 'single county') return true;
      }
      const scopes = Array.isArray(meta.division_scopes) ? meta.division_scopes : [];
      if (scopes.some(s => s && s.toLowerCase() !== 'single-county')) {
        return (Array.isArray(meta.counties) ? meta.counties.length : 0) > 1;
      }
      return false;
    }

    function createBadge(text, extraClass = '') {
      if (!text) return '';
      const cls = extraClass ? ` ${extraClass}` : '';
      return `<span class="contest-badge${cls}">${esc(String(text))}</span>`;
    }

    function formatCounties(meta) {
      const counties = Array.isArray(meta.counties) ? meta.counties.filter(Boolean) : [];
      if (!counties.length) return '';
      const preview = counties.slice(0, 4).map(c => esc(String(c)));
      if (counties.length > 4) preview.push(esc(`+${CountiesRemaining(counties.length - 4)}`));
      const label = meta.county_label ? esc(String(meta.county_label)) : 'Counties';
      return `${label}: ${preview.join(', ')}`;
    }

    function CountiesRemaining(n) {
      return `${n} more`;
    }

    function normalizeContestQuestion(meta) {
      if (!meta || typeof meta !== 'object') return '';
      const direct = typeof meta.question === 'string' ? meta.question.trim() : '';
      if (direct) return direct;
      if (Array.isArray(meta.questions)) {
        for (const entry of meta.questions) {
          if (typeof entry === 'string') {
            const trimmed = entry.trim();
            if (trimmed) return trimmed;
          }
        }
      }
      return '';
    }

    function buildDetails(meta, fallback, exclude) {
      meta = meta || {};
      const lines = [];
      const variant = meta.variant_label;
      const excludeKey = typeof exclude === 'string' && exclude.trim()
        ? exclude.trim().toLowerCase()
        : '';
      const shouldSkip = (val) => {
        if (!excludeKey) return false;
        const candidate = String(val || '').trim().toLowerCase();
        return candidate === excludeKey;
      };
      if (variant && !shouldSkip(variant)) lines.push(String(variant));
      const details = Array.isArray(meta.display_details)
        ? meta.display_details
        : meta.display_details ? [meta.display_details] : [];
      details.forEach(val => {
        if (!val) return;
        const lower = String(val).toLowerCase();
        if (!shouldSkip(val) && !lines.some(existing => existing.toLowerCase() === lower)) {
          lines.push(String(val));
        }
      });
      const summary = Array.isArray(meta.summary)
        ? meta.summary
        : meta.summary ? [meta.summary] : [];
      summary.forEach(val => {
        if (!val) return;
        const lower = String(val).toLowerCase();
        if (!shouldSkip(val) && !lines.some(existing => existing.toLowerCase() === lower)) {
          lines.push(String(val));
        }
      });
      if (!lines.length && fallback && !shouldSkip(fallback)) lines.push(String(fallback));
      return lines.map(val => esc(val)).join(' • ');
    }

    function createOption(opt, arg) {
      const options = typeof arg === 'boolean' ? { isChild: arg } : (arg || {});
      const {
        isChild = false,
        extraClass = '',
        disableCountyPreview = false,
      } = options;
      const meta = opt.metadata || {};
      const questionText = normalizeContestQuestion(meta);
      const scopeBadge = createBadge(meta.scope_label, 'badge-scope');
      const groupedBadge = requiresGrouping(opt) ? createBadge('Grouped', 'badge-group') : '';
      const bundleBadge = !isChild && Number.isFinite(Number(meta.bundle_child_count)) && Number(meta.bundle_child_count) >= 1
        ? createBadge(`${Number(meta.bundle_child_count) + 1} variations`, 'badge-bundle-size')
        : '';
      const variants = Number(meta.variants || (Array.isArray(meta.contest_ids) ? meta.contest_ids.length : 0));
      const variantBadge = variants > 1 ? createBadge(`${variants} IDs`, 'badge-variant') : '';
      const counties = Array.isArray(meta.counties) ? meta.counties.filter(Boolean) : [];
      const countyBadge = !disableCountyPreview && counties.length > 1 ? createBadge(`${counties.length} counties`, 'badge-count') : '';
      const yearBadge = meta.year ? createBadge(meta.year, 'badge-year') : '';
      const confidence = typeof meta.confidence === 'number' ? createBadge(`conf ${meta.confidence.toFixed(2)}`, 'badge-confidence') : '';
      const badgeLine = [groupedBadge, bundleBadge, scopeBadge, variantBadge, countyBadge, yearBadge, confidence].filter(Boolean).join('');
      const countiesText = disableCountyPreview ? '' : formatCounties(meta);
      const detailText = buildDetails(meta, opt.meta, questionText);
      const questionHtml = questionText ? `<div class="contest-question">${esc(questionText)}</div>` : '';
      const item = document.createElement('button');
      item.type = 'button';
      const classNames = ['contest-option'];
      if (isChild) classNames.push('contest-option-child');
      if (extraClass) classNames.push(extraClass);
      item.className = classNames.join(' ');
      item.dataset.index = String(opt.index);
      item.innerHTML = `
        <div class="contest-line">
          <span class="contest-index">[${esc(String(opt.index))}]</span>
          <span class="contest-title">${esc(opt.label || '')}</span>
        </div>
        ${badgeLine ? `<div class="contest-meta-line">${badgeLine}</div>` : ''}
        ${countiesText ? `<div class="contest-counties">${countiesText}</div>` : ''}
        ${questionHtml}
        ${detailText ? `<div class="contest-details">${detailText}</div>` : ''}
      `;
      if (questionText) item.title = questionText;
      else if (opt.label) item.title = opt.label;
      item.onclick = () => closeWith('submit', [opt.index]);
      return item;
    }

    function matches(opt, query) {
      const q = query.toLowerCase();
      if (String(opt.index).toLowerCase().includes(q)) return true;
      if ((opt.label || '').toLowerCase().includes(q)) return true;
      if ((opt.meta || '').toLowerCase().includes(q)) return true;
      const meta = opt.metadata || {};
      const fields = [meta.scope_label, meta.variant_label, meta.office_title, meta.primary_title];
      if (fields.some(val => val && String(val).toLowerCase().includes(q))) return true;
      if (Array.isArray(meta.counties) && meta.counties.some(c => String(c).toLowerCase().includes(q))) return true;
      const flatDetails = [];
      if (Array.isArray(meta.display_details)) flatDetails.push(...meta.display_details);
      else if (meta.display_details) flatDetails.push(meta.display_details);
      if (Array.isArray(meta.summary)) flatDetails.push(...meta.summary);
      else if (meta.summary) flatDetails.push(meta.summary);
      return flatDetails.some(val => val && String(val).toLowerCase().includes(q));
    }

    function groupOptions(list) {
      const officeMap = new Map();
      list.forEach(opt => {
        const key = deriveOfficeKey(opt) || 'other';
        const meta = opt.metadata || {};
        const candidateLabel = meta.display_header || meta.bundle_office_label || deriveOfficeTitle(opt);
        if (!officeMap.has(key)) {
          officeMap.set(key, {
            key,
            office: normalizeOfficeTitle(candidateLabel) || 'Other',
            options: [],
            hasExplicit: Boolean(meta.display_header || meta.bundle_office_label)
          });
        }
        const entry = officeMap.get(key);
        entry.options.push(opt);
        if (!entry.hasExplicit) {
          const normalized = normalizeOfficeTitle(candidateLabel);
          if (normalized && (!entry.office || normalized.length > entry.office.length)) {
            entry.office = normalized;
          }
          if (meta.display_header || meta.bundle_office_label) {
            entry.hasExplicit = true;
          }
        }
      });
      return Array.from(officeMap.values()).sort((a, b) => a.office.localeCompare(b.office));
    }

    function renderContestList(list) {
      const latest = refreshTableState();
      optionsDiv.innerHTML = '';
      currentContestList = list.slice();
      let renderedCount = 0;
      let visibleGrouped = 0;
      if (!list.length) {
        const empty = document.createElement('div');
        empty.className = 'contest-empty';
        empty.textContent = 'No contests match your search.';
        optionsDiv.appendChild(empty);
      } else {
        const renderAggregateOption = (aggregateOpt, officeKey) => {
          if (!aggregateOpt) return null;
          const wrapper = document.createElement('div');
          wrapper.className = 'contest-bundle';
          const header = document.createElement('div');
          header.className = 'contest-bundle-header';
          const meta = aggregateOpt.metadata || {};
          const children = (bundleChildren.get(aggregateOpt.index) || []).slice();
          const availableChildCount = children.length;
          const storedChildCount = Number(meta.bundle_child_count);
          const bundleSize = Number(meta.bundle_size);
          const totalContests = Number.isFinite(bundleSize) && bundleSize > 0
            ? bundleSize
            : Number.isFinite(storedChildCount) && storedChildCount >= 0
              ? storedChildCount + 1
              : availableChildCount + 1;
          const toggleContestCount = Math.max(totalContests, availableChildCount || 0);
          const isExpanded = expandedBundles.has(aggregateOpt.index);
          const primaryBtn = createOption(aggregateOpt, {
            disableCountyPreview: true,
            extraClass: 'contest-option-bundle-primary'
          });

          if (availableChildCount > 0) {
            const toggle = document.createElement('button');
            toggle.type = 'button';
            toggle.className = 'contest-bundle-toggle';
            toggle.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
            const actionText = isExpanded ? 'Hide' : 'Show';
            const quantityText = `${toggleContestCount} contest${toggleContestCount === 1 ? '' : 's'}`;
            const toggleLabel = `${actionText} ${quantityText} for ${aggregateOpt.label || 'this bundle'}`;
            toggle.setAttribute('aria-label', toggleLabel);
            toggle.innerHTML = `<span class="bundle-caret" aria-hidden="true"></span><span class="visually-hidden">${esc(toggleLabel)}</span>`;
            toggle.onclick = () => {
              const willExpand = !expandedBundles.has(aggregateOpt.index);
              if (willExpand) {
                expandedBundles.add(aggregateOpt.index);
                if (officeKey) expandedOffices.add(officeKey);
              } else {
                expandedBundles.delete(aggregateOpt.index);
                if (officeKey) expandedOffices.delete(officeKey);
              }
              renderContestList(currentContestList);
            };
            header.appendChild(toggle);
          } else {
            wrapper.classList.add('contest-bundle-static');
          }

          header.appendChild(primaryBtn);
          wrapper.appendChild(header);

          renderedCount += 1;
          if (requiresGrouping(aggregateOpt)) visibleGrouped += 1;

          if (availableChildCount && isExpanded) {
            const childContainer = document.createElement('div');
            childContainer.className = 'contest-bundle-children';
            children.sort((aChild, bChild) => String(aChild.label || '').localeCompare(String(bChild.label || '')));
            children.forEach(childOpt => {
              childContainer.appendChild(createOption(childOpt, { isChild: true }));
              renderedCount += 1;
              if (requiresGrouping(childOpt)) visibleGrouped += 1;
            });
            wrapper.appendChild(childContainer);
          }

          return wrapper;
        };

        const offices = groupOptions(list);
        offices.forEach(group => {
          const officeKey = group.key || canonicalize(group.office || '');
          const section = document.createElement('section');
          section.className = 'contest-group';

          const aggregateOptions = [];
          const regularOptions = [];
          group.options.forEach(opt => {
            if (isAggregateOption(opt)) aggregateOptions.push(opt);
            else regularOptions.push(opt);
          });
          const aggregateSet = new Set(aggregateOptions);

          const totalRegular = regularOptions.length;
          const searchActive = !!lastContestFilter.trim();
          const hasAggregate = aggregateOptions.length > 0;
          const aggregatedTotal = hasAggregate
            ? aggregateOptions.reduce((max, opt) => {
                const meta = opt.metadata || {};
                const size = Number(meta.bundle_size);
                const childCount = Number(meta.bundle_child_count);
                const inferred = Number.isFinite(size) && size > 0
                  ? size
                  : Number.isFinite(childCount) && childCount >= 0
                    ? childCount + 1
                    : totalRegular;
                return Math.max(max, inferred || 0);
              }, 0)
            : 0;
          const displayCount = hasAggregate
            ? (aggregatedTotal || totalRegular || aggregateOptions.length)
            : totalRegular;

          const header = document.createElement('div');
          header.className = 'contest-group-title';
          header.innerHTML = `${esc(group.office)} <span class="contest-count">(${displayCount || group.options.length})</span>`;

          const shouldCollapse = !searchActive && ((hasAggregate && totalRegular > 0) || totalRegular > OFFICE_COLLAPSE_THRESHOLD);
          if (!shouldCollapse && !searchActive) {
            expandedOffices.add(officeKey);
          }
          const officeExpanded = searchActive || !shouldCollapse || expandedOffices.has(officeKey);

          if (shouldCollapse) {
            const toggle = document.createElement('button');
            toggle.type = 'button';
            toggle.className = 'contest-group-toggle';
            toggle.setAttribute('aria-expanded', officeExpanded ? 'true' : 'false');
            const expandTotal = hasAggregate
              ? (aggregatedTotal || totalRegular || aggregateOptions.length)
              : totalRegular;
            toggle.textContent = officeExpanded ? 'Hide contests' : `Show ${expandTotal} contests`;
            const bundleIndexes = aggregateOptions
              .filter(opt => (bundleChildren.get(opt.index) || []).length > 0)
              .map(opt => opt.index);
            toggle.onclick = () => {
              if (officeExpanded) {
                expandedOffices.delete(officeKey);
                bundleIndexes.forEach(idx => expandedBundles.delete(idx));
              } else {
                expandedOffices.add(officeKey);
                bundleIndexes.forEach(idx => expandedBundles.add(idx));
              }
              renderContestList(currentContestList);
            };
            header.appendChild(toggle);
          }

          section.appendChild(header);

          if (aggregateOptions.length) {
            const aggregateContainer = document.createElement('div');
            aggregateContainer.className = 'contest-aggregate-container';
            aggregateOptions.forEach(opt => {
              const node = renderAggregateOption(opt, officeKey);
              if (node) aggregateContainer.appendChild(node);
            });
            if (aggregateContainer.children.length) {
              section.appendChild(aggregateContainer);
            }
          }

          if (!officeExpanded && shouldCollapse) {
            const collapsedNote = document.createElement('div');
            collapsedNote.className = 'contest-collapsed-note';
            const collapseTotal = hasAggregate
              ? (aggregatedTotal || totalRegular)
              : totalRegular;
            const collapseLabel = collapseTotal === 1 ? 'contest' : 'contests';
            collapsedNote.textContent = hasAggregate
              ? `Expand to view all ${collapseTotal} grouped ${collapseLabel}.`
              : `Expand to view ${collapseTotal} individual ${collapseLabel}.`;
            section.appendChild(collapsedNote);
            optionsDiv.appendChild(section);
            return;
          }

          const scopeMap = new Map();
          regularOptions.forEach(opt => {
            const meta = opt.metadata || {};
            const scopeRaw = meta.scope_label || 'General';
            const scope = String(scopeRaw || 'General').trim() || 'General';
            const key = scope.toLowerCase();
            if (!scopeMap.has(key)) scopeMap.set(key, { scope, options: [] });
            scopeMap.get(key).options.push(opt);
          });

          const scopes = Array.from(scopeMap.values()).sort((a, b) => a.scope.localeCompare(b.scope));
          scopes.forEach(bucket => {
            const block = document.createElement('div');
            block.className = 'contest-subgroup';
            block.innerHTML = `<div class="contest-subgroup-title">${esc(bucket.scope)}</div>`;
            const sortedOptions = bucket.options.slice().sort((a, b) => {
              const rankFor = (entry) => {
                const meta = entry.metadata || {};
                if (meta.bundle_mode === 'aggregate') return 0;
                if (meta.bundle_member) return 2;
                return 1;
              };
              const rankDiff = rankFor(a) - rankFor(b);
              if (rankDiff !== 0) return rankDiff;
              return String(a.label || '').localeCompare(String(b.label || ''));
            });
            sortedOptions.forEach(opt => {
              if (aggregateSet.has(opt)) return;
              const meta = opt.metadata || {};
              if (meta.bundle_member && !lastContestFilter) {
                return;
              }
              block.appendChild(createOption(opt, { isChild: !!meta.bundle_member }));
              renderedCount += 1;
              if (requiresGrouping(opt)) visibleGrouped += 1;
            });
            if (block.children.length > 1) {
              section.appendChild(block);
            }
          });

          optionsDiv.appendChild(section);
        });
      }
      if (summaryDiv) {
        if (summaryIsDefault) {
          summaryDiv.innerHTML = `<div class="small text-muted">${renderedCount} option${renderedCount === 1 ? '' : 's'}</div>`;
        } else {
          summaryDiv.innerHTML = baseSummaryHtml;
        }
        const filteredCount = renderedCount;
        const groupedCount = visibleGrouped;
        const stats = document.createElement('div');
        stats.className = 'small contest-summary-hint text-shimmer';
        const summaryParts = [`${filteredCount} shown`, `${groupedCount} grouped contest${groupedCount === 1 ? '' : 's'}`];
        if (latest.entries.length) {
          const last = latest.entries[latest.entries.length - 1];
          if (last && Array.isArray(last.headers) && last.headers.length) {
            summaryParts.push(`${last.headers.length} column preview`);
          }
        } else {
          summaryParts.push('preview pending');
        }
        stats.textContent = summaryParts.join(' • ');
        summaryDiv.appendChild(stats);
        const previewBtn = document.createElement('button');
        previewBtn.type = 'button';
        previewBtn.className = 'btn btn-outline-info btn-sm contest-preview-btn';
        previewBtn.textContent = 'Show Table Preview';
        if (!latest.entries.length) {
          previewBtn.title = 'No table previews captured yet; opens placeholder view.';
        }
        previewBtn.onclick = () => {
          switchToTableView({ focusSearch: true });
        };
        summaryDiv.appendChild(previewBtn);
      }
    }

    function applyContestFilter(query) {
      const current = query || '';
      lastContestFilter = current;
      const trimmed = current.trim();
      if (!trimmed) {
        renderContestList(baseOptions);
        return;
      }
      const lowered = trimmed.toLowerCase();
      const filtered = baseOptions.filter(opt => matches(opt, lowered));
      renderContestList(filtered);
    }

    function renderTableCards(filter = '') {
      const latest = refreshTableState();
      const q = filter.trim().toLowerCase();
      const entries = latest.entries.slice();
      const filteredEntries = q
        ? entries.filter(entry => {
            const headers = entry.headers || [];
            const rows = entry.rows || [];
            const headerMatch = headers.some(h => h.toLowerCase().includes(q));
            const rowMatch = rows.some(row => Object.values(row || {}).some(v => String(v).toLowerCase().includes(q)));
            return headerMatch || rowMatch;
          })
        : entries;

      optionsDiv.innerHTML = '';
      if (!entries.length) {
        const empty = document.createElement('div');
        empty.className = 'contest-empty';
        empty.textContent = 'Table preview not available yet. Run the parser to capture structure samples.';
        optionsDiv.appendChild(empty);
      } else if (!filteredEntries.length) {
        const empty = document.createElement('div');
        empty.className = 'contest-empty';
        empty.textContent = 'No table previews match your search.';
        optionsDiv.appendChild(empty);
      } else {
        filteredEntries.forEach(entry => {
          const headers = entry.headers || [];
          const rows = entry.rows || [];
          const card = document.createElement('div');
          card.className = 'table-preview-card';
          const confidenceText = typeof entry.confidence === 'number' ? `conf ${entry.confidence.toFixed(2)}` : 'confidence n/a';
          card.innerHTML = `
            <header class="table-preview-header">
              <div class="table-preview-title">Candidate ${entry.index}/${entry.total || entries.length}</div>
              <div class="table-preview-meta">${esc(confidenceText)} • ${headers.length} columns</div>
            </header>
          `;
          const table = document.createElement('table');
          table.className = 'table-preview-grid';
          const thead = document.createElement('thead');
          const headRow = document.createElement('tr');
          headers.forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            headRow.appendChild(th);
          });
          thead.appendChild(headRow);
          table.appendChild(thead);
          const tbody = document.createElement('tbody');
          rows.forEach(row => {
            const tr = document.createElement('tr');
            headers.forEach(h => {
              const td = document.createElement('td');
              td.textContent = row && h in row ? String(row[h]) : '';
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
          });
          if (!rows.length) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = Math.max(headers.length, 1);
            td.className = 'table-preview-empty';
            td.textContent = 'No sample rows available.';
            tr.appendChild(td);
            tbody.appendChild(tr);
          }
          table.appendChild(tbody);
          card.appendChild(table);
          optionsDiv.appendChild(card);
        });
      }

      if (summaryDiv) {
        summaryDiv.innerHTML = `<div class="small text-muted">${esc(latest.contest || 'Contest Pending')} • ${latest.entries.length} candidate${latest.entries.length === 1 ? '' : 's'}</div>`;
        const backBtn = document.createElement('button');
        backBtn.type = 'button';
        backBtn.className = 'btn btn-outline-secondary btn-sm contest-preview-btn';
        backBtn.textContent = 'Back to Contest List';
        backBtn.onclick = () => {
          switchToContestView({ focusSearch: true });
        };
        summaryDiv.appendChild(backBtn);
      }
    }

    function switchToContestView({ focusSearch = false } = {}) {
      searchEl.placeholder = placeholder;
      searchEl.value = lastContestFilter;
      searchEl.oninput = event => {
        applyContestFilter(event.target.value || '');
      };
      applyContestFilter(lastContestFilter);
      if (focusSearch) setTimeout(() => searchEl.focus(), 0);
    }

    function switchToTableView({ focusSearch = false } = {}) {
      searchEl.placeholder = 'Search columns or values…';
      searchEl.value = lastTableFilter;
      searchEl.oninput = event => {
        lastTableFilter = event.target.value || '';
        renderTableCards(lastTableFilter);
      };
      renderTableCards(lastTableFilter);
      if (focusSearch) setTimeout(() => searchEl.focus(), 0);
    }

    function closeWith(mode, payload) {
      if (closed) return;
      closed = true;
      cleanup();
      if (mode === 'submit' && typeof onSubmit === 'function') {
        try { onSubmit(payload); } catch (err) { void err; }
      }
      Modal.close();
      if (mode === 'submit') {
        if (typeof onSelect === 'function') onSelect(payload);
      } else {
        if (typeof onCancel === 'function') onCancel();
        if (typeof onSelect === 'function') onSelect(null);
      }
    }

    closeBtn.onclick = cancelBtn.onclick = () => closeWith('cancel');
    if (modal) {
      modal.addEventListener('hide.bs.modal', () => {
        if (closed) return;
        closeWith('cancel');
      }, { once: true });
    }

    switchToContestView();
    Modal.open();
    setTimeout(() => searchEl.focus(), 0);
  }

  // Parse “[i] Label (meta)” lines from a backend menu string
  function parseIndexedMenu(message) {
    const opts = [];
    const lines = String(message || '').split(/\r?\n/);
    for (const line of lines) {
      const m = line.match(/^\s*\[(\d+)\]\s+(.+?)\s*$/);
      if (!m) continue;
      const idx = Number(m[1]);
      let label = m[2];
      let meta = '';
      const mm = label.match(/^(.+?)\s+\((.+)\)\s*$/);
      if (mm) { label = mm[1]; meta = mm[2]; }
      opts.push({ index: idx, label, meta });
    }
    return opts;
  }

  // Generic selection modal reusing download modal shell
  function showIndexedSelectionModalWithContext(title, options, ctxSummaryHtml, onSelect, extras = {}) {
    const refs = Modal.get();
    if (!refs) {
      if (typeof onSelect === 'function') onSelect(null);
      return;
    }
    const { modal, titleEl, searchEl, optionsDiv, summaryDiv, closeBtn, cancelBtn } = refs;
    const { onCancel, onSubmit } = extras || {};

    let closed = false;

    const finalize = (mode, payload) => {
      if (closed) return;
      closed = true;
      if (mode === 'cancel' && typeof onCancel === 'function') {
        try { onCancel(); } catch (err) { void err; }
      }
      if (mode === 'submit' && typeof onSubmit === 'function') {
        try { onSubmit(payload); } catch (err) { void err; }
      }
    };

    const closeModal = (mode, payload) => {
      finalize(mode, payload);
      Modal.close();
      if (typeof onSelect === 'function') {
        if (mode === 'submit') onSelect(payload);
        else onSelect(null);
      }
    };

    const cancelSelection = () => closeModal('cancel', null);
    const emitSelection = (value) => closeModal('submit', value);

    titleEl.textContent = title || 'Select';
    summaryDiv.innerHTML = ctxSummaryHtml || `${options.length} option(s)`;

    const PAGE_SIZE = 200;
    let filtered = options.slice();
    let rendered = 0;

    const createOptionElement = (option) => {
      const item = document.createElement('div');
      item.className = 'download-option';
      item.tabIndex = 0;
      item.innerHTML = `<b>[${option.index}]</b> ${esc(option.label || '')}${option.meta ? ` <small>(${esc(option.meta)})</small>` : ''}`;
      const select = () => emitSelection([option.index]);
      item.onclick = select;
      item.onkeydown = (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          select();
        }
      };
      optionsDiv.appendChild(item);
    };

    const appendLoadMoreButton = () => {
      const remaining = filtered.length - rendered;
      if (remaining <= 0) return;
      const moreBtn = document.createElement('button');
      moreBtn.type = 'button';
      moreBtn.className = 'btn btn-primary btn-sm mt-1em';
      moreBtn.textContent = `Show more (${remaining} remaining)`;
      moreBtn.onclick = () => {
        const start = rendered;
        const newEnd = Math.min(filtered.length, rendered + PAGE_SIZE);
        for (let i = start; i < newEnd; i += 1) {
          createOptionElement(filtered[i]);
        }
        rendered = newEnd;
        const left = filtered.length - rendered;
        if (left > 0) {
          moreBtn.textContent = `Show more (${left} remaining)`;
        } else {
          moreBtn.remove();
        }
      };
      optionsDiv.appendChild(moreBtn);
    };

    function renderList(query = '') {
      const normalized = query.toLowerCase().trim();
      filtered = normalized
        ? options.filter(o =>
            String(o.index).includes(normalized) ||
            (o.label || '').toLowerCase().includes(normalized) ||
            (o.meta || '').toLowerCase().includes(normalized)
          )
        : options.slice();

      rendered = 0;
      optionsDiv.innerHTML = '';

      const initialEnd = Math.min(filtered.length, PAGE_SIZE);
      for (let i = 0; i < initialEnd; i += 1) {
        createOptionElement(filtered[i]);
      }
      rendered = initialEnd;

      appendLoadMoreButton();
    }

    searchEl.value = '';
    renderList();
    searchEl.oninput = (event) => renderList(event.target.value);
    closeBtn.onclick = cancelBtn.onclick = cancelSelection;

    if (modal) {
      modal.addEventListener('hide.bs.modal', () => {
        if (closed) return;
        finalize('cancel', null);
        if (typeof onSelect === 'function') onSelect(null);
      }, { once: true });
    }

    Modal.open();
    searchEl.focus();
  }
  function getManualState(sessionId) {
    if (!sessionId) return null;
    const key = String(sessionId);
    if (!manualOverrideState[key]) {
      manualOverrideState[key] = {
        files: [],
        folder: 'uploads',
        baseDir: '',
        lastPromptHash: null,
        lastReset: 0,
        lastCount: null,
        lastSelection: null
      };
    }
    return manualOverrideState[key];
  }

  function resetManualState(sessionId, meta = {}) {
    const state = getManualState(sessionId);
    if (!state) return null;
    state.files = [];
    state.lastReset = Date.now();
    state.lastPromptHash = null;
    state.lastSelection = null;
    state.lastCount = typeof meta.count === 'number' ? meta.count : null;
    if (meta.folder) state.folder = meta.folder;
    if (meta.baseDir) state.baseDir = meta.baseDir;
    return state;
  }

  function addManualFileOption(sessionId, index, label) {
    const state = getManualState(sessionId);
    if (!state) return null;
    const idx = Number(index);
    const trimmed = (label || '').trim();
    if (!Number.isFinite(idx) || !trimmed) return state;
    if (state.files.some(f => f.index === idx || f.label === trimmed)) return state;
    state.files.push({ index: idx, label: trimmed, value: String(idx), meta: '' });
    state.files.sort((a, b) => a.index - b.index);
    return state;
  }

  function getManualFiles(sessionId) {
    const state = getManualState(sessionId);
    return state ? state.files.slice() : [];
  }

  function ensureSectionExpanded(sectionId, options = {}) {
    if (!sectionId) return;
    const panel = document.getElementById(sectionId);
    if (!panel) return;
    const opts = typeof options === 'object' && options !== null ? options : {};
    const btn = document.querySelector(`.collapsible-btn[data-target="${sectionId}"]`);
    const wasHidden = panel.classList.contains('hidden');
    if (wasHidden) {
      panel.classList.remove('hidden');
      if (btn) btn.setAttribute('aria-expanded', 'true');
    }
    panel.querySelectorAll('.folder-panel').forEach(fp => fp._refresh && fp._refresh());
    const container = panel.closest('.section');
    if (!container) return;

    const highlight = opts.highlight !== false;
    const scrollIntoView = !!opts.scrollIntoView;
    const focusButton = !!opts.focusButton;
    const flashClass = typeof opts.flashClass === 'string' && opts.flashClass.trim() ? opts.flashClass.trim() : '';
    const flashDuration = Number.isFinite(opts.flashDuration) ? Math.max(0, opts.flashDuration) : 2400;

    if (highlight) {
      container.classList.add('manual-highlight');
      setTimeout(() => container.classList.remove('manual-highlight'), 1600);
    }

    if (flashClass) {
      container.classList.add(flashClass);
      setTimeout(() => container.classList.remove(flashClass), flashDuration);
    }

    if (scrollIntoView) {
      try {
        container.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } catch (err) {
        void err;
      }
    }

    if (focusButton && btn) {
      try {
        btn.focus({ preventScroll: true });
      } catch (err) {
        void err;
      }
    }
  }

  function clearParserCompletionVisuals() {
    document.querySelectorAll('.container-main.parser-finished').forEach(node => {
      node.classList.remove('parser-finished');
      node.removeAttribute('data-celebrate');
    });
    document.querySelectorAll('.container-main .celebrate-sparkles').forEach(node => node.remove());
    const outputSection = document.getElementById('outputSection');
    if (outputSection) {
      const container = outputSection.closest('.section');
      if (container) {
        container.classList.remove('celebrate-pulse');
      }
    }
  }

  function spawnCelebrateSparkles(hostElement) {
    if (PREFERS_REDUCED_MOTION) return;
    const host = hostElement || document.querySelector('.container-main') || document.body;
    if (!host) return;
    const sparkles = document.createElement('div');
    sparkles.className = 'celebrate-sparkles';
    const total = 14;
    for (let i = 0; i < total; i += 1) {
      const dot = document.createElement('span');
      dot.className = 'sparkle';
      dot.style.setProperty('--x', `${Math.random() * 100}%`);
      dot.style.setProperty('--y', `${Math.random() * 100}%`);
      dot.style.setProperty('--delay', `${(Math.random() * 0.4).toFixed(2)}s`);
      dot.style.setProperty('--drift', `${(Math.random() * 28 + 18).toFixed(1)}px`);
      sparkles.appendChild(dot);
    }
    host.appendChild(sparkles);
    setTimeout(() => sparkles.remove(), 2600);
  }

  function celebrateParserCompletion() {
    const now = Date.now();
    if (now - lastCelebrationTs < 1000) return;
    lastCelebrationTs = now;
    ensureSectionExpanded('outputSection', {
      scrollIntoView: true,
      focusButton: true,
      flashClass: 'celebrate-pulse',
      flashDuration: 2600,
    });
    const container = document.querySelector('.container-main');
    if (container) {
      container.classList.add('parser-finished');
      container.setAttribute('data-celebrate', 'output-ready');
      spawnCelebrateSparkles(container);
      setTimeout(() => {
        container.classList.remove('parser-finished');
        container.removeAttribute('data-celebrate');
      }, 5200);
    } else {
      spawnCelebrateSparkles(document.body);
    }
  }

  function respondToPrompt(sessionId, value, options = {}) {
    if (!socket || !sessionId || value == null) return;
    const opts = typeof options === 'object' && options !== null ? options : {};
    const message = typeof opts.message === 'string' && opts.message.trim()
      ? opts.message.trim()
      : 'Processing selection…';
    const minimumMs = Number.isFinite(opts.minimumMs) ? Math.max(0, opts.minimumMs) : 600;
    const autoHideMs = Number.isFinite(opts.autoHideMs) ? Math.max(0, opts.autoHideMs) : 12000;
    const buttonLabel = typeof opts.buttonLabel === 'string' && opts.buttonLabel.trim()
      ? opts.buttonLabel.trim()
      : 'Please wait…';
    const overlayEnabled = opts.showOverlay !== false;

    try {
      if (overlayEnabled) {
        PendingOverlay.show(message, { minimumMs, autoHideMs });
      }
      if (modalRestore && typeof modalRestore.setBusyForSession === 'function') {
        modalRestore.setBusyForSession(sessionId, true, {
          message,
          buttonLabel
        });
      }
    } catch (err) {
      void err;
    }
    socket.emit('parser_prompt', { session_id: sessionId, value: String(value) });
  }

  function cacheUrlPromptContext(sessionId, urls, processed = {}, meta = {}) {
    if (!sessionId || !Array.isArray(urls)) return [];
    const sanitized = urls
      .map(u => (typeof u === 'string' ? u.trim() : ''))
      .filter(u => u && !/^(?:\.{3}|…)/.test(u));
    const processedMap = (processed && typeof processed === 'object') ? processed : {};
    const metaCopy = (meta && typeof meta === 'object') ? { ...meta } : {};
    urlPromptContextBySession.set(sessionId, {
      urls: sanitized.slice(),
      processed: processedMap,
      meta: metaCopy,
    });
    urlIndexMap[sessionId] = Object.fromEntries(sanitized.map((url, idx) => [String(idx + 1), url]));
    return sanitized;
  }

  function getUrlPromptContext(sessionId = activeSessionId) {
    return urlPromptContextBySession.get(sessionId) || null;
  }

  function hidePromptInput() {
    if (!el.promptInput) return;
    el.promptInput.disabled = true;
    el.promptInput.value = '';
    el.promptInput.parentElement?.classList.add('hidden');
  }

  function showPromptInput(placeholder = 'Type a command...') {
    if (!el.promptInput) return;
    el.promptInput.placeholder = placeholder;
    el.promptInput.disabled = false;
    el.promptInput.parentElement?.classList.remove('hidden');
  }

  function openManualSelectionModal(sessionId, promptMeta = {}) {
    const state = getManualState(sessionId);
    if (!state) return;
    const files = getManualFiles(sessionId);
    const placeholder = promptMeta.placeholder || 'Enter index or filename…';
    const summaryParts = [];
    const count = Number.isFinite(state.lastCount) ? state.lastCount : files.length;
    if (count) summaryParts.push(`${count} file(s) detected`);
    if (state.folder) summaryParts.push(`Source: ${state.folder}`);
    if (state.baseDir) {
      const normalized = state.baseDir.replace(/\\/g, '/');
      summaryParts.push(`Path: ${normalized}`);
    }
    const summaryHtml = summaryParts.length
      ? `<div class="small text-muted">${summaryParts.map(part => esc(part)).join(' • ')}</div>`
      : '';

    pipelineControl?.setPhase('source');
    pipelineControl?.markAttention('source');
    pipelineControl?.focusStep('source', { scroll: false, highlight: true });
    logPanelControl?.expand();

    hidePromptInput();

    if (!files.length) {
      showFolderBrowser('uploads', '', (sel) => {
        if (!sel) {
          showPromptInput(placeholder);
          return;
        }
        const rel = sel.path ? `${sel.path}/${sel.name}` : sel.name;
        respondToPrompt(sessionId, rel);
        pipelineControl?.clearAttention('source');
        state.lastSelection = rel;
        setTimeout(() => showPromptInput('Type a command...'), 600);
      });
      return;
    }

    const restoreKey = `manual:${sessionId}`;
    const restoreTitle = 'Manual Uploads';
    const restoreDetail = 'Resume picking the upload you want to parse.';
    const restoreMessage = promptMeta.restoreMessage || `${restoreTitle}. ${restoreDetail}`;
    modalRestore.register({
      key: restoreKey,
      sessionId,
      message: restoreMessage,
      title: restoreTitle,
      detail: restoreDetail,
      icon: '📦',
      buttonLabel: 'Resume Uploads',
      buttonTitle: 'Reopen the manual uploads picker',
      reopen: () => openManualSelectionModal(sessionId, promptMeta)
    });
    modalRestore.markActive(restoreKey);

    showIndexedSelectionModalWithContext(
      'Select Upload File',
      files.map(file => ({ index: file.index, label: file.label, meta: file.meta || '' })),
      summaryHtml || `${files.length} option(s)`,
      (selection) => {
        if (!selection || !selection.length) {
          showPromptInput(placeholder);
          return;
        }
        const idx = selection[0];
        const match = files.find(f => String(f.index) === String(idx));
        const value = match ? match.value : String(idx);
        respondToPrompt(sessionId, value);
        pipelineControl?.clearAttention('source');
        state.lastSelection = value;
        modalRestore.clear(restoreKey);
        setTimeout(() => showPromptInput('Type a command...'), 600);
      },
      {
        onCancel: () => {
          showPromptInput(placeholder);
          modalRestore.markDismissed(restoreKey);
        },
        onSubmit: () => modalRestore.clear(restoreKey)
      }
    );
  }

  function openContestSelectionModal(sessionId, options, ctxSummaryHtml = '', meta = {}) {
    if (!sessionId || !Array.isArray(options) || !options.length) {
      showPromptInput(meta.placeholder || 'Enter contest index…');
      return;
    }

    pipelineControl?.setPhase('resolve');
    pipelineControl?.markAttention('resolve');
    pipelineControl?.focusStep('resolve', { scroll: false, highlight: true });
    logPanelControl?.expand();

    hidePromptInput();

    const restoreKey = `contest:${sessionId}`;
    const restoreTitle = meta.restoreTitle || 'Contest Selection';
    const restoreDetail = meta.restoreDetail || 'Contest choice still needs your input.';
    const restoreMessage = meta.restoreMessage || `${restoreTitle}. ${restoreDetail}`;
    const optionSnapshot = cloneContestOptions(options);
    modalRestore.register({
      key: restoreKey,
      sessionId,
      message: restoreMessage,
      title: restoreTitle,
      detail: restoreDetail,
      icon: meta.restoreIcon || '🎯',
      buttonLabel: meta.restoreButtonLabel || 'Resume Contest',
      buttonTitle: meta.restoreButtonTitle || 'Reopen the contest selection dialog',
      reopen: () => openContestSelectionModal(sessionId, cloneContestOptions(optionSnapshot), ctxSummaryHtml, meta)
    });
    modalRestore.markActive(restoreKey);

    showContestSelectionModal(
      meta.title || 'Select Contest',
      options,
      ctxSummaryHtml || `${options.length} option(s)`,
      (selection) => {
        if (!selection || !selection.length) {
          showPromptInput(meta.placeholder || 'Enter contest index…');
          return;
        }
        const payload = selection.join(',');
        respondToPrompt(sessionId, payload);
        pipelineControl?.clearAttention('resolve');
        modalRestore.clear(restoreKey);
        setTimeout(() => showPromptInput('Type a command...'), 600);
      },
      {
        placeholder: meta.placeholder || 'Enter contest index…',
        onCancel: () => {
          showPromptInput(meta.placeholder || 'Enter contest index…');
          modalRestore.markDismissed(restoreKey);
        },
        onSubmit: () => modalRestore.clear(restoreKey),
        sessionId
      }
    );
  }

  function toTitleCase(val) {
    const s = String(val || '').replace(/_/g, ' ').trim();
    return s ? s.replace(/\b\w/g, c => c.toUpperCase()) : '';
  }

  function buildUrlOptions(urls, processedMap = {}) {
    return urls.map((url, idx) => {
      const entry = processedMap[url];
      const rawStatus = entry && typeof entry.status === 'string' ? entry.status.trim().toLowerCase() : '';
      const statusKey = rawStatus || 'unprocessed';
      const friendlyStatus = toTitleCase(statusKey);
      const flagged = !!(entry && (entry.flagged_for_review || entry.flagged));
      const metaParts = [];
      if (friendlyStatus) metaParts.push(friendlyStatus);
      if (flagged) metaParts.push('Flagged');
      return {
        index: idx + 1,
        label: url,
        meta: metaParts.join(' • '),
        statusKey,
      };
    });
  }

  function openUrlSelectionModal(sessionId, urls, processed = {}, meta = {}) {
    const placeholder = meta.placeholder || 'Enter URL index or filter…';
    const hintText = meta.hint || 'Search or filter (state:/county:/text)';
    const sanitized = cacheUrlPromptContext(sessionId, urls, processed, meta);
    if (!sessionId || !sanitized.length) {
      showPromptInput(placeholder);
      return;
    }
    if (sessionId !== activeSessionId) return;

    const context = getUrlPromptContext(sessionId) || { processed: {} };
    const processedMap = (context.processed && typeof context.processed === 'object') ? context.processed : {};

    const options = buildUrlOptions(sanitized, processedMap);

    if (sessionId === activeSessionId) {
      lastPromptContext = { kind: 'url', session_id: sessionId };
    }

    const statusCounts = options.reduce((acc, opt) => {
      const key = opt.statusKey || 'unprocessed';
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
    const statusSummary = Object.entries(statusCounts)
      .filter(([, count]) => count > 0)
      .map(([status, count]) => `${count} ${toTitleCase(status)}`)
      .join(' • ');
    const summaryLines = [
      `${options.length} URL(s)`
    ];
    if (statusSummary) summaryLines.push(statusSummary);
    if (hintText) summaryLines.push(hintText);
    const summaryHtml = `<div class="small text-muted">${summaryLines.map(line => esc(line)).join('<br>')}</div>`;

    pipelineControl?.setPhase('source');
    pipelineControl?.markAttention('source');
    pipelineControl?.focusStep('source', { scroll: false, highlight: true });
    logPanelControl?.expand();

    hidePromptInput();

    const restoreKey = `url:${sessionId}`;
    const restoreTitle = meta.restoreTitle || 'URL Selection';
    const restoreDetail = meta.restoreDetail || 'Pick which URL to process next.';
    const restoreMessage = meta.restoreMessage || `${restoreTitle}. ${restoreDetail}`;
    const reopenUrls = sanitized.slice();
    modalRestore.register({
      key: restoreKey,
      sessionId,
      message: restoreMessage,
      title: restoreTitle,
      detail: restoreDetail,
      icon: meta.restoreIcon || '🔗',
      buttonLabel: meta.restoreButtonLabel || 'Resume URLs',
      buttonTitle: meta.restoreButtonTitle || 'Reopen the URL selection dialog',
      reopen: () => openUrlSelectionModal(sessionId, reopenUrls.slice(), processed, meta)
    });
    modalRestore.markActive(restoreKey);

    showIndexedSelectionModalWithContext(
      meta.title || 'Select URL',
      options,
      summaryHtml,
      (selection) => {
        if (!selection || !selection.length) {
          showPromptInput(placeholder);
          return;
        }
        const payload = selection.join(',');
        respondToPrompt(sessionId, payload);
        pipelineControl?.clearAttention('source');
        modalRestore.clear(restoreKey);
        setTimeout(() => showPromptInput('Type a command...'), 600);
      },
      {
        onCancel: () => {
          showPromptInput(placeholder);
          modalRestore.markDismissed(restoreKey);
        },
        onSubmit: () => modalRestore.clear(restoreKey)
      }
    );
  }

  window.showUrlPicker = function(sessionId = activeSessionId) {
    if (sessionId !== activeSessionId) return false;
    const ctx = getUrlPromptContext(sessionId);
    if (!ctx || !Array.isArray(ctx.urls) || !ctx.urls.length) return false;
    openUrlSelectionModal(sessionId, ctx.urls, ctx.processed || {}, ctx.meta || {});
    return true;
  };

  function updatePipelineForStatusLog(message) {
    if (!pipelineControl || !message) return;
    const lower = String(message).toLowerCase();
    if (/parser connected|session started|launching parser/.test(lower)) {
      pipelineControl.setPhase('run');
    }
    if (/prompt response received/.test(lower)) {
      pipelineControl.clearAttention('source');
      pipelineControl.clearAttention('resolve');
      pipelineControl.setPhase('run');
    }
    if (/run cancelled|cancellation requested|run canceled/.test(lower)) {
      pipelineControl.clearAttention('resolve');
      pipelineControl.setPhase('prepare');
      pipelineControl.focusStep('prepare', { scroll: false, highlight: false });
    }
    if (/parser run completed|completed! output csv|manual upload parse succeeded/i.test(lower)) {
      pipelineControl.setPhase('review');
      pipelineControl.focusStep('review', { scroll: false, highlight: true });
      celebrateParserCompletion();
    }
  }

  // Folder browser (uploads, input, output) with search + breadcrumbs
  async function apiListDir(root, path = '') {
    const qs = new URLSearchParams({ root, path });
    const urls = [
      `/api/fs/list?${qs.toString()}`,   // preferred
      `/api/list_dir?${qs.toString()}`   // fallback
    ];
    for (const u of urls) {
      try {
        const r = await fetch(u, { method: 'GET' });
        if (!r.ok) continue;
        const d = await r.json();
        if (d && Array.isArray(d.entries)) return d;
      } catch { /* ignore */ }
    }
    throw new Error('Directory listing API not available');
  }
  // Top-level styled folder panel (used in sidebar sections)
  function mountFolderPanel(root, panelEl) {
    if (!panelEl) return;
    panelEl.innerHTML = '';
    const toolbar = document.createElement('div');
    toolbar.className = 'folder-toolbar';

    const search = document.createElement('input');
    search.type = 'search';
    search.className = 'form-control folder-search';
    search.placeholder = `Search ${ROOT_LABELS[root] || root}…`;
    search.setAttribute('aria-label', `Search ${root} files`);

    const refreshBtn = document.createElement('button');
    refreshBtn.type = 'button';
    refreshBtn.className = 'btn btn-primary btn-ghost';
    refreshBtn.title = 'Refresh';
    refreshBtn.innerHTML = '⟳';

    // Minimal controls
    const upBtn = document.createElement('button');
    upBtn.type = 'button';
    upBtn.className = 'btn btn-secondary btn-ghost hidden';
    upBtn.title = 'Up one folder';
    upBtn.innerHTML = '⬆️';

    const newBtn = document.createElement('button');
    newBtn.type = 'button';
    newBtn.className = 'btn btn-new-folder icon-only btn-gradient-neon';
    newBtn.title = 'New Folder';
    newBtn.setAttribute('aria-label', 'New Folder');
    newBtn.textContent = '+';

    const pathSpan = document.createElement('span');
    pathSpan.className = 'folder-count folder-path';
    pathSpan.textContent = '/';
    pathSpan.title = `/${ROOT_LABELS[root] || root}`;
    pathSpan.addEventListener('click', () => { cwd = ''; refresh(); });

    const results = document.createElement('div');
    results.className = 'folder-results';
    results.innerHTML = `<div class="download-option">Loading…</div>`;

    toolbar.appendChild(search);
    toolbar.appendChild(refreshBtn);
    toolbar.appendChild(upBtn);
    toolbar.appendChild(newBtn);
    toolbar.appendChild(pathSpan);
    panelEl.appendChild(toolbar);
    panelEl.appendChild(results);

    let cwd = ''; // track current path for inline navigation
    let allEntries = [];
    let viewEntries = [];
    let page = 0;
    const PAGE_SIZE = 50;

    function joinPath(base, name) {
      return base ? `${base}/${name}` : name;
    }

    function render() {
      const end = Math.min(viewEntries.length, (page + 1) * PAGE_SIZE);
      const slice = viewEntries.slice(0, end);

      results.innerHTML = '';
      if (!slice.length) {
        results.innerHTML = `<div class="download-option">No files found.</div>`;
      } else {
        slice.forEach(ent => {
          const row = document.createElement('div');
          row.className = 'download-option folder-row';

          const icon = ent.type === 'dir' ? '📁' : '📄';
          const description = ent.type === 'dir' ? `Open folder ${ent.name}` : `Use ${ent.name}`;
          const nameBtn = document.createElement('button');
          nameBtn.type = 'button';
          nameBtn.className = 'item-name';
          nameBtn.title = ent.name;
          nameBtn.setAttribute('aria-label', description);
          nameBtn.textContent = `${icon} ${ent.name}`;

          const activateFile = () => {
            const rel = joinPath(cwd, ent.name);
            if (activeSessionId) {
              respondToPrompt(activeSessionId, rel);
            }
          };

          const openFolder = async () => {
            cwd = joinPath(cwd, ent.name);
            await refresh();
          };

          const actions = document.createElement('div');
          actions.className = 'file-actions';

          if (ent.type === 'file') {
            const useBtn = document.createElement('button');
            useBtn.type = 'button';
            useBtn.className = 'btn btn-primary btn-sm btn-ghost';
            useBtn.textContent = 'Use';
            useBtn.title = `Use ${ent.name}`;
            useBtn.addEventListener('click', (e) => {
              e.stopPropagation();
              const rel = joinPath(cwd, ent.name);
              if (activeSessionId) {
                respondToPrompt(activeSessionId, rel);
              }
            });

            const dl = document.createElement('a');
            dl.href = `/download_fs?root=${encodeURIComponent(root)}&path=${encodeURIComponent(cwd)}&name=${encodeURIComponent(ent.name)}`;
            dl.className = 'btn btn-success btn-sm btn-ghost';
            dl.innerText = 'Download';
            dl.title = `Download ${ent.name}`;
            dl.addEventListener('click', e => e.stopPropagation());

            const del = document.createElement('button');
            del.type = 'button';
            del.className = 'btn btn-danger btn-sm btn-ghost';
            del.innerText = 'Delete';
            del.title = `Delete ${ent.name}`;
            del.addEventListener('click', async (e) => {
              e.stopPropagation();
              if (!confirm(`Delete ${ent.name}?`)) return;
              try {
                const r = await fetch('/api/fs/delete', {
                  method: 'POST',
                  headers: { 'Content-Type':'application/json; charset=utf-8' },
                  body: JSON.stringify({ root, path: cwd, name: ent.name })
                });
                const d = await r.json().catch(()=>({}));
                if (!r.ok || !d.success) alert(d.error || 'Failed to delete file.');
                await refresh();
              } catch { alert('Network error.'); }
            });

            actions.appendChild(useBtn);
            actions.appendChild(dl);
            actions.appendChild(del);
            row.addEventListener('click', () => {
              activateFile();
            });
            nameBtn.addEventListener('click', (e) => {
              e.stopPropagation();
              activateFile();
            });
          } else {
            // Directory: click to open
            row.addEventListener('click', () => {
              openFolder();
            });
            nameBtn.addEventListener('click', async (e) => {
              e.stopPropagation();
              await openFolder();
            });

            // Folder delete (Shift = recursive)
            const delDir = document.createElement('button');
            delDir.type = 'button';
            delDir.className = 'btn btn-danger btn-sm btn-ghost';
            delDir.innerText = 'Delete';
            delDir.title = 'Delete folder (Shift for recursive)';
            delDir.addEventListener('click', async (e) => {
              e.stopPropagation();
              const recursive = e.shiftKey;
              const ok = confirm(`Delete folder "${ent.name}"${recursive ? ' and all its contents' : ''}?`);
              if (!ok) return;
              try {
                const r = await fetch('/api/fs/delete', {
                  method: 'POST',
                  headers: { 'Content-Type':'application/json; charset=utf-8' },
                  body: JSON.stringify({ root, path: cwd, name: ent.name, recursive })
                });
                const d = await r.json().catch(()=>({}));
                if (!r.ok || !d.success) alert(d.error || 'Failed to delete folder.');
                await refresh();
              } catch { alert('Network error.'); }
            });
            actions.appendChild(delDir);
          }

          row.appendChild(nameBtn);
          row.appendChild(actions);
          results.appendChild(row);
        });
      }

      if (end < viewEntries.length) {
        const more = document.createElement('button');
        more.type = 'button';
        more.className = 'btn btn-primary btn-sm mt-1em';
        more.innerText = `Show more (${viewEntries.length - end} remaining)`;
        more.onclick = () => { page++; render(); };
        results.appendChild(more);
      }
    }

    function applyFilter() {
      const q = (search.value || '').toLowerCase().trim();
      page = 0;
      viewEntries = !q ? allEntries.slice()
                       : allEntries.filter(e => e.name.toLowerCase().includes(q) || (e.type || '').toLowerCase().includes(q));
      // Sort: dirs first, then by name
      viewEntries.sort((a,b) => (a.type !== b.type) ? (a.type === 'dir' ? -1 : 1) : a.name.localeCompare(b.name));
      render();
    }

    async function refresh() {
      results.innerHTML = `<div class="download-option">Loading…</div>`;
      try {
        const listing = await apiListDir(root, cwd);
        allEntries = Array.isArray(listing.entries) ? listing.entries : [];
      } catch {
        allEntries = [];
      }
      pathSpan.textContent = `/${cwd || ''}`;
      upBtn.classList.toggle('hidden', !cwd);
      applyFilter();
    }

    // Persist search per-root
    const LS_KEY = `folder_search_${root}`;
    search.value = localStorage.getItem(LS_KEY) || '';
    search.addEventListener('input', debounce(() => {
      localStorage.setItem(LS_KEY, search.value);
      applyFilter();
    }, 200));

    refreshBtn.addEventListener('click', refresh);
    upBtn.addEventListener('click', async () => {
      if (!cwd) return;
      const parts = cwd.split('/').filter(Boolean);
      parts.pop();
      cwd = parts.join('/');
      await refresh();
    });
    newBtn.addEventListener('click', async () => {
      askFolderName({ title: `New ${ROOT_LABELS[root] || root} Folder`, placeholder: 'e.g. results_2024' }, async (name) => {
        if (!name) return;
        try {
          const r = await fetch('/api/fs/mkdir', {
            method: 'POST',
            headers: { 'Content-Type':'application/json; charset=utf-8' },
            body: JSON.stringify({ root, path: cwd, name })
          });
          const d = await r.json().catch(()=>({}));
          if (!r.ok || !d.success) alert(d.error || 'Failed to create folder.');
          await refresh();
        } catch { alert('Network error.'); }
      });
    });

    // Expose refresh so collapsible open can refresh list
    panelEl._refresh = refresh;
    refresh();
  }

  function initFolderPanels() {
    const inputPanel   = document.querySelector('#inputSection  #inputFolderPanel');
    const outputPanel  = document.querySelector('#outputSection #outputFolderPanel');
    const uploadsPanel = document.querySelector('#uploadsSection #uploadsFolderPanel');
    if (inputPanel)  mountFolderPanel('input', inputPanel);
    if (outputPanel) mountFolderPanel('output', outputPanel);
    if (uploadsPanel) mountFolderPanel('uploads', uploadsPanel);
  }
  // Folder browser (uploads, input, output) with search + breadcrumbs
  function showFolderBrowser(root, initialPath = '', onSelect, options = {}) {
    if (typeof onSelect === 'object' && options === undefined) {
      options = onSelect || {};
      onSelect = undefined;
    }
    options = options || {};
    const selectCallback = typeof onSelect === 'function' ? onSelect : () => {};
    const label = ROOT_LABELS[root] || root;
    const restoreKey = options.restoreKey ?? `folder:${root}`;
    const restoreTitle = options.restoreTitle ?? `${label} Browser`;
    const restoreDetail = options.restoreDetail ?? `Reopen to keep browsing ${label.toLowerCase()} items.`;
    const restoreMessage = options.restoreMessage ?? `${restoreTitle}. ${restoreDetail}`;
    const restoreIcon = options.restoreIcon ?? (root === 'output' ? '📤' : root === 'uploads' ? '📁' : '🗂️');
    const restoreButtonLabel = options.restoreButtonLabel ?? 'Resume Browsing';
    const restoreButtonTitle = options.restoreButtonTitle ?? `Reopen the ${label} browser`;
    const skipRegister = options.skipRegister === true;

    if (restoreKey && !skipRegister) {
      modalRestore.register({
        key: restoreKey,
        message: restoreMessage,
        title: restoreTitle,
        detail: restoreDetail,
        icon: restoreIcon,
        buttonLabel: restoreButtonLabel,
        buttonTitle: restoreButtonTitle,
        scrollIntoView: options.scrollIntoView !== false,
        reopen: () => showFolderBrowser(root, initialPath, onSelect, { ...options, skipRegister: true })
      });
    }

    const refs = Modal.get();
    if (!refs) {
      selectCallback(null);
      return;
    }
    const { modal, titleEl, searchEl, optionsDiv, summaryDiv, closeBtn, cancelBtn } = refs;

    if (restoreKey) modalRestore.markActive(restoreKey);

    let closed = false;
    const finish = (reason, payload = null) => {
      if (closed) return;
      closed = true;
      if (reason === 'submit') {
        if (restoreKey) modalRestore.clear(restoreKey);
        selectCallback(payload);
      } else {
        if (restoreKey) modalRestore.markDismissed(restoreKey);
        selectCallback(null);
      }
    };

    if (modal) {
      modal.addEventListener('hide.bs.modal', () => finish('cancel'), { once: true });
    }

    const shutdown = (reason, payload = null) => {
      if (closed) return;
      Modal.close();
      finish(reason, payload);
    };

    let cwd = initialPath || '';
    let allEntries = []; // {name,type:'dir'|'file', size, modified}

    titleEl.textContent = `Browse ${ROOT_LABELS[root] || root}`;
    summaryDiv.textContent = '';

    function makeCrumb(path) {
      const parts = path.split('/').filter(Boolean);
      const spans = [];
      let acc = '';
      const rootSpan = document.createElement('span');
      rootSpan.className = 'crumb';
      rootSpan.textContent = ROOT_LABELS[root] || root;
      rootSpan.onclick = () => { cwd = ''; refresh(); };
      spans.push(rootSpan);
      for (let i=0;i<parts.length;i++) {
        const sep = document.createElement('span');
        sep.textContent = ' / ';
        spans.push(sep);
        acc += (acc ? '/' : '') + parts[i];
        const s = document.createElement('span');
        s.className = 'crumb';
        s.textContent = parts[i];
        s.onclick = () => { cwd = acc; refresh(); };
        spans.push(s);
      }
      const wrap = document.createElement('div');
      wrap.className = 'folder-breadcrumb';
      spans.forEach(n => wrap.appendChild(n));
      return wrap;
    }
    function renderList(filter = '') {
      const q = (filter||'').toLowerCase().trim();
      let entries = allEntries.slice();
      if (q) {
        entries = entries.filter(e =>
          e.name.toLowerCase().includes(q) || (e.type||'').toLowerCase().includes(q)
        );
      }
      entries.sort((a,b) => {
        if (a.type !== b.type) return a.type === 'dir' ? -1 : 1;
        return a.name.localeCompare(b.name);
      });

      optionsDiv.innerHTML = '';
      optionsDiv.appendChild(makeCrumb(cwd));

      const tools = document.createElement('div');
      tools.className = 'folder-actions-bar';

      const newBtn = document.createElement('button');
      newBtn.type = 'button';
      newBtn.className = 'btn btn-sm btn-new-folder icon-only btn-gradient-neon ms-2';
      newBtn.title = 'New Folder';
      newBtn.setAttribute('aria-label', 'New Folder');
      newBtn.textContent = '+';
      newBtn.onclick = async () => {
        askFolderName({ title: `New ${ROOT_LABELS[root] || root} Folder`, placeholder: 'e.g. reports' }, async (name) => {
          if (!name) return;
          try {
            const r = await fetch('/api/fs/mkdir', {
              method: 'POST',
              headers: { 'Content-Type':'application/json; charset=utf-8' },
              body: JSON.stringify({ root, path: cwd, name })
            });
            const d = await r.json().catch(()=>({}));
            if (!r.ok || !d.success) alert(d.error || 'Failed to create folder.');
            await refresh();
          } catch { alert('Network error.'); }
        });
      };
      tools.appendChild(newBtn);
      optionsDiv.appendChild(tools);

      if (cwd) {
        const up = document.createElement('div');
        up.className = 'download-option';
        up.innerHTML = `⬆️ <b>[..]</b> <small>Up one level</small>`;
        up.onclick = () => {
          const parts = cwd.split('/').filter(Boolean);
          parts.pop();
          cwd = parts.join('/');
          refresh();
        };
        optionsDiv.appendChild(up);
      }

      entries.forEach(ent => {
        const item = document.createElement('div');
        item.className = 'download-option';
        item.tabIndex = 0;
        if (ent.type === 'dir') {
          // Folder row
          const label = document.createElement('span');
          label.innerHTML = `📁 <b>${esc(ent.name)}</b>`;
          item.appendChild(label);

          const actions = document.createElement('span');
          actions.className = 'ms-2';
          const openBtn = document.createElement('button');
          openBtn.type = 'button';
          openBtn.className = 'btn btn-sm btn-secondary me-1';
          openBtn.textContent = 'Open';
          openBtn.onclick = (e) => { e.stopPropagation(); cwd = (cwd ? cwd + '/' : '') + ent.name; refresh(); };

          const delBtn = document.createElement('button');
          delBtn.type = 'button';
          delBtn.className = 'btn btn-sm btn-danger';
          delBtn.textContent = 'Delete';
          delBtn.title = 'Delete folder (empty folders only; hold Shift for recursive)';
          delBtn.onclick = async (e) => {
            e.stopPropagation();
            const recursive = e.shiftKey; // Shift+Click -> recursive delete
            const ok = confirm(`Delete folder "${ent.name}"${recursive ? ' and all its contents' : ''}?`);
            if (!ok) return;
            try {
              const r = await fetch('/api/fs/delete', {
                method: 'POST',
                headers: { 'Content-Type':'application/json; charset=utf-8' },
                body: JSON.stringify({ root, path: cwd, name: ent.name, recursive })
              });
              const d = await r.json().catch(()=>({}));
              if (!r.ok || !d.success) alert(d.error || 'Failed to delete folder.');
              await refresh();
            } catch { alert('Network error.'); }
          };

          actions.appendChild(openBtn);
          actions.appendChild(delBtn);
          item.appendChild(actions);

          // Also allow clicking whole row to open
          item.onclick = () => { cwd = (cwd ? cwd + '/' : '') + ent.name; refresh(); };
          item.onkeydown = (e) => { if (e.key === 'Enter') { cwd = (cwd ? cwd + '/' : '') + ent.name; refresh(); } };
        } else {
          // File row
          const meta = [];
          if (ent.size != null) meta.push(`${ent.size} bytes`);
          if (ent.modified) meta.push(new Date(ent.modified).toLocaleString());
          const metaStr = meta.length ? ` <small>(${esc(meta.join(' • '))})</small>` : '';

          const label = document.createElement('span');
          label.innerHTML = `📄 ${esc(ent.name)}${metaStr}`;
          item.appendChild(label);

          const actions = document.createElement('span');
          actions.className = 'ms-2';

          const useBtn = document.createElement('button');
          useBtn.type = 'button';
          useBtn.className = 'btn btn-sm btn-primary me-1';
          useBtn.textContent = 'Use';
          useBtn.title = `Use ${ent.name}`;
          useBtn.onclick = (e) => {
            e.stopPropagation();
            const rel = cwd ? `${cwd}/${ent.name}` : ent.name;
            shutdown('submit', { root, path: cwd, name: ent.name });
            if (activeSessionId) {
              respondToPrompt(activeSessionId, rel);
            }
          };

          const dl = document.createElement('a');
          dl.href = `/download_fs?root=${encodeURIComponent(root)}&path=${encodeURIComponent(cwd)}&name=${encodeURIComponent(ent.name)}`;
          dl.className = 'btn btn-sm btn-success me-1';
          dl.textContent = 'Download';
          dl.onclick = (e) => e.stopPropagation();

          const delBtn = document.createElement('button');
          delBtn.type = 'button';
          delBtn.className = 'btn btn-sm btn-danger';
          delBtn.textContent = 'Delete';
          delBtn.onclick = async (e) => {
            e.stopPropagation();
            if (!confirm(`Delete file "${ent.name}"?`)) return;
            try {
              const r = await fetch('/api/fs/delete', {
                method: 'POST',
                headers: { 'Content-Type':'application/json; charset=utf-8' },
                body: JSON.stringify({ root, path: cwd, name: ent.name })
              });
              const d = await r.json().catch(()=>({}));
              if (!r.ok || !d.success) alert(d.error || 'Failed to delete file.');
              await refresh();
            } catch { alert('Network error.'); }
          };

          actions.appendChild(useBtn);
          actions.appendChild(dl);
          actions.appendChild(delBtn);
          item.appendChild(actions);

          // Keep row click as "Use"
          item.onclick = () => {
            const rel = cwd ? `${cwd}/${ent.name}` : ent.name;
            shutdown('submit', { root, path: cwd, name: ent.name });
            if (activeSessionId) {
              respondToPrompt(activeSessionId, rel);
            }
          };
          item.onkeydown = (e) => { if (e.key === 'Enter') { item.onclick(); } };
        }

        optionsDiv.appendChild(item);
      });

      summaryDiv.textContent = `${entries.length} item(s)`;
    }

    async function refresh() {
      optionsDiv.innerHTML = `<div class="download-option">Loading…</div>`;
      try {
        const listing = await apiListDir(root, cwd);
        allEntries = Array.isArray(listing.entries) ? listing.entries : [];
      } catch (e) {
        allEntries = [];
        optionsDiv.innerHTML = `<div class="download-option">Folder API not available.</div>`;
      }
      renderList(searchEl.value);
    }

    searchEl.value = '';
    searchEl.oninput = e => renderList(e.target.value);
    closeBtn.onclick = cancelBtn.onclick = () => shutdown('cancel');

    Modal.open();
    searchEl.focus();
    refresh();
  }

  // Dynamic type selector
  let logTypeSelect = $('#logTypeFilterSelect');
  if (!logTypeSelect) {
    logTypeSelect = document.createElement('select');
    logTypeSelect.id = 'logTypeFilterSelect';
    logTypeSelect.className = 'log-select ms-1em';
    logTypeSelect.setAttribute('aria-label','Filter logs by type');
    logTypeSelect.innerHTML = `<option value="all">All Types</option>`;
    const label = document.createElement('label');
    label.htmlFor = 'logTypeFilterSelect';
    label.className = 'log-label';
    label.textContent = 'Type:';
    label.id = 'logTypeFilterLabel';
    logTypeSelect.setAttribute('aria-labelledby','logTypeFilterLabel');
    const container = document.querySelector('.logger-mode-section');
    if (container) {
      container.appendChild(label);
      container.appendChild(logTypeSelect);

      if (SHOW_BROWSE_TOOLBAR) {
        // Folder browser buttons (optional)
        const browseWrap = document.createElement('div');
        browseWrap.className = 'browse-toolbar';

        const mkBtn = (txt, root) => {
          const b = document.createElement('button');
          b.type = 'button';
          b.className = 'btn btn-sm btn-secondary';
          b.textContent = txt;
          b.onclick = () => {
            showFolderBrowser(root, '', (sel) => {
              if (!sel) return;
              const rel = sel.path ? `${sel.path}/${sel.name}` : sel.name;
              if (activeSessionId) {
                respondToPrompt(activeSessionId, rel);
              }
            });
          };
          return b;
        };
        browseWrap.appendChild(mkBtn('Browse Uploads', 'uploads'));
        browseWrap.appendChild(mkBtn('Browse Input', 'input'));
        browseWrap.appendChild(mkBtn('Browse Output', 'output'));
        container.appendChild(browseWrap);
      }
    }
  }
  el.outputModeSelect?.setAttribute('aria-label','Select output delivery mode');
  el.logFilterSelect?.setAttribute('aria-label','Filter logs by level');

  // Canonical levels and types
  const CANONICAL_LEVELS = [
    'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL', 'TRACE'
  ];
  const CANONICAL_TYPES = [
    'status', 'input', 'output', 'manual_override', 'ai_analysis', 'stream', 'router',
    'handler', 'batch', 'download', 'browser', 'validation', 'exception', 'cancel',
    'summary', 'cache', 'prompt', 'heartbeat', 'database', 'delete', 'selector', 'other'
  ];

  // Pre-populate log level filter
  if (el.logFilterSelect) {
    el.logFilterSelect.innerHTML = '<option value="all">All</option>' +
      CANONICAL_LEVELS.map(lvl => `<option value="${lvl.toLowerCase()}">${lvl}</option>`).join('');
  }

  // Pre-populate log type filter
  if (logTypeSelect) {
    logTypeSelect.innerHTML = '<option value="all">All Types</option>' +
      CANONICAL_TYPES.map(type => `<option value="${type}">${type.replace(/_/g, ' ')}</option>`).join('');
  }

  // Use these sets for filter option tracking
  const seenLogTypes = new Set(['all', ...CANONICAL_TYPES]);
  const dynamicLevels = new Set(['all', ...CANONICAL_LEVELS.map(l => l.toLowerCase())]);

  // -------- Sessions --------
  function syncSessionCache(ids) {
    let source = Array.isArray(ids) ? ids.slice() : [];
    if (!source.length) {
      source = sessionMirror.list().map(meta => meta.session_id).filter(Boolean);
    }
    const normalized = uniq(source.filter(Boolean));
    lsSetJSON('active_sessions', normalized);
    return normalized;
  }

  function getSessions() {
    const cached = lsGetJSON('active_sessions', []);
    const merged = [...cached, ...sessionMirror.list().map(meta => meta.session_id || '')];
    return syncSessionCache(merged);
  }

  function highlightActiveSessionBtn() {
    if (!el.sessionList) return;
    const sid = activeSessionId;
    $$('.session-btn').forEach(b => b.classList.toggle('active', b.dataset.sid === sid));
  }

  function setActiveSession(id) {
    activeSessionId = id || '';
    localStorage.setItem('session_id', activeSessionId);
    if (el.activeSessionSpan) el.activeSessionSpan.textContent = activeSessionId;
    highlightActiveSessionBtn();
    pipelineControl?.reset();
    if (activeSessionId) {
      const meta = sessionMirror.get(activeSessionId);
      if (meta) {
        const source = meta.manual_source ?? meta.file_source ?? meta.source;
        const origin = meta.manual_source_origin ?? meta.source_origin ?? meta.origin;
        updateSessionSourceMeta(activeSessionId, source, origin, { fromServer: true });
      } else if (pendingManualSource) {
        updateSessionSourceMeta(
          activeSessionId,
          pendingManualSource.source,
          pendingManualSource.origin,
          { fromServer: false }
        );
      } else {
        clearSessionSourceMeta(activeSessionId, { fromServer: true });
      }
    } else {
      activeManualSource = 'input';
      activeManualSourceOrigin = 'default';
      syncManualSourceUI();
      updatePipelineMetadataForActive();
      updatePipelineHintForPhase();
    }
    logPanelControl?.collapse();
    if (activeSessionId) {
      if (pendingManualUploadSelection && !manualUploadSelectionBySession.has(activeSessionId)) {
        manualUploadSelectionBySession.set(activeSessionId, pendingManualUploadSelection);
      }
      manualUploadSelection = manualUploadSelectionBySession.get(activeSessionId) || null;
      if (pendingDirectUrlDraft && !directUrlDraftBySession.has(activeSessionId)) {
        directUrlDraftBySession.set(activeSessionId, pendingDirectUrlDraft);
      }
    } else {
      manualUploadSelection = manualUploadSelection || pendingManualUploadSelection || null;
    }
    if (activeSessionId) {
      pendingManualUploadSelection = null;
      pendingDirectUrlDraft = '';
    }
    syncManualUploadControls();
    const directDraft = activeSessionId
      ? (directUrlDraftBySession.get(activeSessionId) || '')
      : (pendingDirectUrlDraft || el.directUrlTextarea?.value || '');
    if (el.directUrlTextarea) {
      el.directUrlTextarea.value = directDraft;
    }
    updateDirectUrlFeedback();
    if (earlyQueue.length && el.outputDiv) {
      earlyQueue.forEach(d => renderParserOutput(d));
      flushBatch();
      earlyQueue = [];
    }
    applySessionState(activeSessionId);
  }

  function updateSessionCount() {
    if (el.sessionCount) el.sessionCount.textContent = `(${getSessions().length})`;
  }

  function joinSession(id) {
    if (!socket || !id || joinedSessions.has(id)) return;
    socket.emit('join', { session_id: id });
    joinedSessions.add(id);
  }

  function renderSessionList() {
    if (!el.sessionList) return;
    el.sessionList.innerHTML = '';
    getSessions().forEach(sid => {
      const btn = document.createElement('button');
      btn.textContent = sid;
      btn.className = 'session-btn w-100';
      btn.dataset.sid = sid;
      btn.addEventListener('click', () => {
        setActiveSession(sid);
        clearOutput();
        joinSession(sid);
        loadSessionLogs(sid);
        restoreCachedLogs(sid);
      });
      const remove = document.createElement('span');
      remove.className = 'session-remove';
      remove.textContent = '✖';
      remove.addEventListener('click', e => {
        e.stopPropagation();
        socket?.emit('delete_session', { session_id: sid });
        const filtered = getSessions().filter(s => s !== sid);
        lsSetJSON('active_sessions', filtered);
        if (activeSessionId === sid) {
          setActiveSession(filtered[0] || '');
          if (filtered[0]) joinSession(filtered[0]);
        }
        renderSessionList();
      });
      btn.appendChild(remove);
      el.sessionList.appendChild(btn);
    });
    highlightActiveSessionBtn();
    updateSessionCount();
  }

  // --- Simple per-session log cache in localStorage (bounded) ---
  const LOG_CACHE_KEY = 'session_log_cache_v1';
  function loadCache() { return lsGetJSON(LOG_CACHE_KEY, {}); }
  function saveCache(cache) { lsSetJSON(LOG_CACHE_KEY, cache); }
  function appendCacheLog(sid, obj) {
    if (!sid || !obj) return;
    const cache = loadCache();
    const arr = cache[sid] = cache[sid] || [];
    arr.push(obj);
    if (arr.length > 400) arr.splice(0, arr.length - 300);
    saveCache(cache);
  }
  function restoreCachedLogs(sid) {
    const cache = loadCache();
    const arr = cache[sid];
    if (!arr || !arr.length || !el.outputDiv) return;
    clearOutput();
    arr.forEach(l => renderParserOutput(l));
    flushBatch();
  }

  function addNewSession() {
    const sid = 'sess_' + Math.random().toString(36).slice(2, 11);
    const sessions = getSessions();
    sessions.push(sid);
    lsSetJSON('active_sessions', sessions);
    setActiveSession(sid);
    renderSessionList();
    joinSession(sid);
    clearOutput();
  }

  // -------- Output batching --------
  function clearOutput() { if (el.outputDiv) el.outputDiv.innerHTML = ''; }
  function scrollOutput() { if (el.outputDiv) el.outputDiv.scrollTop = el.outputDiv.scrollHeight; }
  function flushBatch() {
    if (!batching) return;
    if (el.outputDiv && batchFrag.childNodes.length) {
      el.outputDiv.appendChild(batchFrag);
      batchFrag = document.createDocumentFragment();
      scrollOutput();
      filterLogs();
    }
    batching = false;
  }
  function scheduleFlush() {
    const t = nowPerf();
    if (!batching) {
      batching = true;
      lastFlush = t;
      setTimeout(flushBatch, BATCH_INTERVAL);
    } else if (t - lastFlush > BATCH_INTERVAL) {
      flushBatch();
    }
  }

  // -------- Normalization --------
  function inferType(msgLower) {
    if (msgLower.includes('heartbeat')) return 'heartbeat';
    if (msgLower.includes('cancellation')) return 'cancel';
    if (msgLower.includes('launching parser') || msgLower.includes('session started') ||
        msgLower.includes('manual file source set') || msgLower.includes('socket connected') ||
        msgLower.includes('client disconnected') || msgLower.includes('re-associated socket')) return 'status';
    if (msgLower.includes('no usable urls')) return 'input';
    if (msgLower.includes('processed successfully') || msgLower.includes('urls failed')) return 'summary';
    if (msgLower.includes('exception') || msgLower.includes('traceback')) return 'exception';
    if (msgLower.includes('validation')) return 'validation';
    if (msgLower.includes('output') && msgLower.includes('saved')) return 'output';
    return 'other';
  }

  function normalizeLog(raw) {
    const original_raw = typeof raw === 'string'
      ? raw
      : (() => { try { return JSON.stringify(raw); } catch { return String(raw); } })();

    if (typeof raw === 'string') {
      raw = stripMetaLines(raw);
      const cand = extractFirstJSONObject(raw);
      if (cand) {
        try { raw = JSON.parse(cand); } catch {}
      }
    }

    let obj = raw;

    if (typeof obj === 'string') {
      const parsed = tryParseJSON(obj, true);
      obj = parsed || { level: 'INFO', message: obj, type: 'raw' };
    }
    if (!obj || typeof obj !== 'object')
      obj = { level: 'INFO', message: String(raw), type: 'raw' };

    const visited = new Set();
    for (let i = 0; i < 5; i++) {
      if (!obj || typeof obj !== 'object' || visited.has(obj)) break;
      visited.add(obj);
      const m = obj.message;
      let promoted = null;
      if (typeof m === 'string') {
        const p = tryParseJSON(m, true);
        if (p && typeof p === 'object') promoted = p;
      } else if (m && typeof m === 'object' && (m.level || m.type || m.message || m.status)) {
        promoted = m;
      }
      if (promoted) {
        if (!promoted.session_id && obj.session_id) promoted.session_id = obj.session_id;
        if (!promoted.level && obj.level) promoted.level = obj.level;
        if (!promoted.type && obj.type) promoted.type = obj.type;
        obj = promoted;
      } else break;
    }

    if (typeof obj.message === 'string' && obj.message.trim().startsWith('{')) {
      const inner = tryParseJSON(obj.message.trim(), false);
      if (inner && typeof inner === 'object') {
        if (!inner.session_id && obj.session_id) inner.session_id = obj.session_id;
        obj = { ...inner, _from_embedded: true };
      }
    }

    obj.level = (obj.level || 'INFO').toString().toUpperCase();
    let providedType = (obj.type || '').toString().trim();
    if (!providedType && obj.message && typeof obj.message === 'object' && obj.message.type)
      providedType = String(obj.message.type);

    let full_text;
    if (obj.message == null) full_text = '';
    else if (typeof obj.message === 'object') {
      try { full_text = JSON.stringify(obj.message); } catch { full_text = String(obj.message); }
    } else full_text = String(obj.message);

    const lowerMsg = full_text.toLowerCase();
    const TYPE_CANON = {
      cancellation: 'cancel',
      canc: 'cancel',
      manual: 'manual_override',
      manualoverride: 'manual_override',
      ai: 'ai_analysis',
      analysis: 'ai_analysis',
      anomalies: 'ai_analysis',
      streamresults: 'stream',
      dl: 'download',
      download: 'download',
      handler: 'handler',
      batch: 'batch',
      browser: 'browser',
      out: 'output',
      outputfile: 'output',
      fatal: 'fatal'
    };

    function deriveType() {
      if (providedType) {
        const key = providedType.toLowerCase().replace(/[\s-]/g, '_');
        return TYPE_CANON[key] || key;
      }
      if ('anomalies' in obj || 'integrity_issues' in obj) return 'ai_analysis';
      if ('output_file' in obj || 'output_dir' in obj) return 'output';
      if ('manual_file' in obj) return 'manual_override';
      if ('contests' in obj) return 'stream';
      if ('flagged' in obj && Array.isArray(obj.flagged)) return 'ai_analysis';
      if ('handler' in obj && !('router' in obj)) return 'handler';
      if (/router/i.test(full_text)) return 'router';
      if (/batch mode/i.test(full_text)) return 'batch';
      if (/download/i.test(lowerMsg)) return 'download';
      if (/browser/i.test(lowerMsg)) return 'browser';
      if (/heartbeat/.test(lowerMsg)) return 'heartbeat';
      if (/validation/.test(lowerMsg)) return 'validation';
      if (/summary/.test(lowerMsg)) return 'summary';
      if (/cache/.test(lowerMsg)) return 'cache';
      if (/prompt/.test(lowerMsg)) return 'prompt';
      if (/manual override/.test(lowerMsg)) return 'manual_override';
      if (/streaming results/.test(lowerMsg)) return 'stream';
      if (/navigating to/.test(lowerMsg)) return 'status';
      if (/csv written/.test(lowerMsg) || /output bypass/.test(lowerMsg)) return 'output';
      if (/no urls/.test(lowerMsg) || /loaded \d+ raw urls/.test(lowerMsg)) return 'input';
      if (/exception/.test(lowerMsg) || /traceback/.test(lowerMsg)) return 'exception';
      return 'other';
    }

    let t = deriveType();
    if (t === 'other') {
      const alt = inferType(lowerMsg);
      if (alt && alt !== 'other') t = alt;
    }
    t = TYPE_CANON[t] || t;
    if (t === 'cancellation') t = 'cancel';
    obj.type = t;

    if (obj.status === 'alive' && !full_text)
      full_text = `[heartbeat] ${obj.session_id || ''}`.trim();

    let ts = obj.timestamp;
    if (ts == null) ts = Date.now();
    ts = Number(ts);
    obj.timestamp = ts < 10_000_000_000 ? ts * 1000 : ts;

    if (!obj.session_id)
      obj.session_id = obj.sessionId || obj.sid || activeSessionId || '';

    obj.full_text = full_text;
    obj.display_text = full_text.length > DISPLAY_TRUNCATE
      ? full_text.slice(0, DISPLAY_TRUNCATE) + '…'
      : full_text;
    obj.original_raw = original_raw;

    return obj;
  }

  const SHOW_HEARTBEAT_LINES = false;

  const levelIconMap = {
    INFO:'🛈', DEBUG:'⚙️', WARNING:'⚠️', ERROR:'⛔', CRITICAL:'🚨',
    TRACE:'🔍'
  };
  const levelColorMap = {
    INFO:'#00ffe7', DEBUG:'#8ecae6', WARNING:'#ffd166', ERROR:'#eb4f43', CRITICAL:'#ff006e',
    TRACE:'#ff8c00'
  };
  const typeColorMap = {
    status:           '#264653',
    input:            '#38bdf8',
    output:           '#c084fc',
    manual_override:  '#fb923c',
    ai_analysis:      '#f472b6',
    stream:           '#34d399',
    router:           '#fde047',
    handler:          '#f9a8d4',
    batch:            '#67e8f9',
    download:         '#93c5fd',
    browser:          '#d8b4fe',
    validation:       '#fcd34d',
    exception:        '#ff7b00',
    cancel:           '#f87171',
    summary:          '#a78bfa',
    cache:            '#a3a3a3',
    prompt:           '#d4af37',
    heartbeat:        '#60a5fa',
    database:         '#86efac',
    delete:           '#f87171',
    other:            '#555'
  };
  function hashColor(seed) {
    let h = 0;
    for (let i=0;i<seed.length;i++) h = (h * 131 + seed.charCodeAt(i)) >>> 0;
    const hue = h % 360;
    const sat = 55 + (h >> 3) % 25;
    const light = 45 + (h >> 6) % 20;
    return `hsl(${hue} ${sat}% ${light}%)`;
  }

  function ensureFilterOptions(selectEl, valueSet, incoming) {
    const v = (incoming||'').toLowerCase();
    if (!selectEl || !v || valueSet.has(v)) return;
    valueSet.add(v);
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = formatTypeLabel(v);
    selectEl.appendChild(opt);
  }

  function formatTypeLabel(v) {
    const base = v.replace(/_/g,' ').replace(/^\w/,c=>c.toUpperCase());
    const ct = typeCounts[v] ?? 0;
    return ct ? `${base} (${ct})` : base;
  }

  function bumpTypeCount(v) {
    if (!v) return;
    typeCounts[v] = (typeCounts[v] || 0) + 1;
    const opt = logTypeSelect?.querySelector(`option[value="${v}"]`);
    if (opt) opt.textContent = formatTypeLabel(v);
  }
  function shouldSuppressDuplicate(sig) {
    const now = Date.now();
    if (dupMap.size > 350) {
      for (const [k,v] of dupMap) if (now - v > DUP_WINDOW_MS) dupMap.delete(k);
    }
    const prev = dupMap.get(sig);
    if (prev && now - prev < DUP_WINDOW_MS) return true;
    dupMap.set(sig, now);
    return false;
  }

  function closeAllDetails(except) {
    $$('.log-line.expanded').forEach(line => {
      if (line === except) return;
      line.classList.remove('expanded');
      const main = line.querySelector('.log-main');
      const details = line.querySelector('.log-details');
      if (main && details) {
        main.setAttribute('aria-expanded','false');
        details.classList.add('hidden');
      }
    });
  }

  function renderParserOutput(raw) {
    if (!el.outputDiv) return;
    if (typeof raw === 'object' && raw && /no usable urls/i.test(String(raw.message||'')) && !document.getElementById('empty-url-hint')) {
      const hint = document.createElement('div');
      hint.id = 'empty-url-hint';
      hint.className = 'empty-url-hint';
      hint.innerHTML = `
        <div class="hint-box">
          <strong>No usable URLs found.</strong><br>
          Edit <code>webapp/parser/urls.txt</code> (remove comments / ensure validity) then press Run again.
        </div>`;
      el.outputDiv.appendChild(hint);
    }
    if (typeof raw === 'string') {
      const trimmed = raw.trim();
      if (/^(Level:|Type:|Session:)\s/i.test(trimmed)) return;
    }

    const obj = normalizeLog(raw);
    if (!obj || typeof obj !== 'object' || !obj.type || !obj.level) return;

    if (obj.type === 'heartbeat' && !SHOW_HEARTBEAT_LINES) return;

    if (!levelIconMap[obj.level]) {
      levelIconMap[obj.level] = '🧩';
      levelColorMap[obj.level] = hashColor(obj.level);
      ensureFilterOptions(el.logFilterSelect, dynamicLevels, obj.level.toLowerCase());
    }
    if (!typeColorMap[obj.type]) typeColorMap[obj.type] = hashColor(obj.type);

    ensureFilterOptions(el.logFilterSelect, dynamicLevels, obj.level.toLowerCase());
    const normType = obj.type.toLowerCase();
    if (obj.session_id === activeSessionId && pipelineControl) {
      if (normType === 'output' || normType === 'summary') {
        pipelineControl.setPhase('review');
      }
    }
    if (logPanelControl && typeof logPanelControl.isCollapsed === 'function') {
      const severe = obj.level === 'ERROR' || obj.level === 'CRITICAL';
      const warn = obj.level === 'WARNING';
      if (logPanelControl.isCollapsed() && (severe || warn)) logPanelControl.expand();
    }
    ensureFilterOptions(logTypeSelect, seenLogTypes, normType);
    bumpTypeCount(normType);

    const sig = ['v6', obj.level, obj.type, obj.session_id || '', obj.full_text].join('|');
    if (shouldSuppressDuplicate(sig)) return;

    let timeStr = '';
    if (obj.timestamp && !isNaN(obj.timestamp)) {
      const d = new Date(obj.timestamp);
      if (!isNaN(d.getTime())) timeStr = d.toLocaleTimeString();
    }

    const levelIcon = levelIconMap[obj.level] || '🛈';

    const wrapper = document.createElement('div');
    wrapper.className = 'log-line';
    wrapper.dataset.level = obj.level;
    wrapper.dataset.type  = obj.type;
    wrapper.classList.add('level-' + obj.level, 'type-' + obj.type.replace(/[^a-z0-9_-]/gi,'_'));

    const detailsId = 'logd_' + Math.random().toString(36).slice(2,10);

    wrapper.innerHTML = `
      <div class="log-main" tabindex="0" aria-controls="${detailsId}" aria-expanded="false" role="button">
        <span class="log-time">${timeStr ? '['+esc(timeStr)+']' : ''}</span>
        <span class="log-level" title="${esc(obj.level)}">${levelIcon}</span>
        <span class="log-type-pill">${esc(obj.type)}</span>
        <span class="log-msg">${esc(obj.display_text)||'&nbsp;'}</span>
        ${obj.display_text !== obj.full_text ? '<span class="log-more-hint" aria-hidden="true"> +</span>' : ''}
      </div>
      <div class="log-details hidden" id="${detailsId}" role="region" aria-label="Log details"></div>
    `;

    const mainRow  = wrapper.querySelector('.log-main');
    const detailEl = wrapper.querySelector('.log-details');

    function buildDetails() {
      if (detailEl._built) return;
      const exclude = new Set([
        'level','type','session_id','sessionId','sid','timestamp','message','msg',
        'status','_suppressRender','full_text','display_text','original_raw','_from_embedded'
      ]);
      const extra = Object.entries(obj).filter(([k]) => !exclude.has(k));
      let parsedObj = null, pretty = '';
      if (obj.full_text && obj.full_text.trim().startsWith('{')) {
        try { parsedObj = JSON.parse(obj.full_text); pretty = JSON.stringify(parsedObj, null, 2); } catch {}
      }
      detailEl.innerHTML = `
        <div class="log-detail-grid">
          <div><span class="ld-k">Session:</span><span class="ld-v">${esc(obj.session_id||'—')}</span></div>
          <div><span class="ld-k">Timestamp:</span><span class="ld-v">${esc(String(obj.timestamp))}</span></div>
          <div><span class="ld-k">Level:</span><span class="ld-v">${esc(obj.level)}</span></div>
          <div><span class="ld-k">Type:</span><span class="ld-v">${esc(obj.type)}</span></div>
          <div class="ld-span"><span class="ld-k">Message:</span><span class="ld-v">${esc(obj.full_text)}</span></div>
          <div class="ld-span"><span class="ld-k">Original Raw:</span><span class="ld-v">${esc(obj.original_raw)}</span></div>
          ${extra.map(([k,v]) => `
            <div class="ld-span">
              <span class="ld-k">${esc(k)}:</span>
              <span class="ld-v">${esc(typeof v === 'object' ? JSON.stringify(v) : String(v))}</span>
            </div>`).join('')}
          ${ pretty ? `
            <div class="ld-span json-prewrap">
              <div class="json-head">
                <span class="ld-k">Parsed JSON:</span>
                <button type="button" class="btn btn-sm btn-copy-json">Copy</button>
              </div>
              <pre>${esc(pretty)}</pre>
            </div>` : '' }
        </div>
      `;
      const copyBtn = detailEl.querySelector('.btn-copy-json');
      if (copyBtn && pretty) {
        copyBtn.addEventListener('click', () => {
          navigator.clipboard?.writeText(pretty).catch(()=>{});
          copyBtn.textContent = 'Copied';
          setTimeout(()=> copyBtn.textContent='Copy', 1400);
        });
      }
      detailEl._built = true;
    }

    function toggleDetails(explicit) {
      const show = explicit !== undefined ? explicit : detailEl.classList.contains('hidden');
      if (show) {
        closeAllDetails(wrapper);
        buildDetails();
      }
      detailEl.classList.toggle('hidden', !show);
      mainRow.setAttribute('aria-expanded', String(show));
      wrapper.classList.toggle('expanded', show);
    }

    mainRow.addEventListener('click', e => {
      if (e.metaKey || e.ctrlKey || e.shiftKey) return;
      toggleDetails();
    });
    mainRow.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleDetails(); }
      else if (e.key === 'Escape') toggleDetails(false);
      else if (e.key === 'ArrowRight') toggleDetails(true);
      else if (e.key === 'ArrowLeft') toggleDetails(false);
    });

    batchFrag.appendChild(wrapper);
    scheduleFlush();
  }

  // -------- Filters --------
  function filterLogs() {
    const levelFilter = el.logFilterSelect ? el.logFilterSelect.value : 'all';
    const typeFilter  = logTypeSelect ? logTypeSelect.value : 'all';
    $$('.log-line').forEach(line => {
      const lvl = (line.dataset.level||'').toLowerCase();
      const typ = (line.dataset.type||'').toLowerCase();
      const show =
        (levelFilter === 'all' || lvl === levelFilter) &&
        (typeFilter === 'all'  || typ === typeFilter);
      line.classList.toggle('filtered-out', !show);
    });
  }

  // --- Filter Presets ---
  const PRESET_KEY = 'log_filter_presets_v1';
  function getFilterPresets() { return lsGetJSON(PRESET_KEY, {}); }
  function setFilterPresets(obj) { lsSetJSON(PRESET_KEY, obj); }
  function populatePresetSelect() {
    if (!el.filterPresetSelect) return;
    const presets = getFilterPresets();
    const names = Object.keys(presets).sort();
    el.filterPresetSelect.innerHTML = '<option value="">Presets...</option>' +
      names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join('');
  }
  function saveCurrentPreset(name) {
    if (!name) return;
    const presets = getFilterPresets();
    presets[name] = {
      level: el.logFilterSelect?.value || 'all',
      type:  logTypeSelect?.value || 'all'
    };
    setFilterPresets(presets);
    populatePresetSelect();
  }
  function applyPreset(name) {
    const presets = getFilterPresets();
    const p = presets[name];
    if (!p) return;
    if (el.logFilterSelect && p.level) el.logFilterSelect.value = p.level;
    if (logTypeSelect && p.type) logTypeSelect.value = p.type;
    filterLogs();
  }
  function deletePreset(name) {
    const presets = getFilterPresets();
    if (!presets[name]) return;
    delete presets[name];
    setFilterPresets(presets);
    populatePresetSelect();
  }
  function exportPreset(name) {
    const presets = getFilterPresets();
    let data;
    if (name && presets[name]) data = presets[name];
    else data = {
      current: {
        level: el.logFilterSelect?.value || 'all',
        type:  logTypeSelect?.value || 'all'
      }
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type:'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `parser_filters_${(name||'current').replace(/\W+/g,'_')}.json`;
    document.body.appendChild(a);
    a.click();
    setTimeout(()=>{ URL.revokeObjectURL(a.href); a.remove(); }, 400);
  }
  function initFilterPresets() {
    populatePresetSelect();
    if (el.saveFiltersBtn) on(el.saveFiltersBtn,'click', () => {
      const defName = 'preset_' + new Date().toISOString().slice(11,19).replace(/:/g,'');
      const name = prompt('Enter preset name', defName);
      if (!name) return;
      saveCurrentPreset(name.trim());
    });
    if (el.filterPresetSelect) on(el.filterPresetSelect,'change', e => {
      const v = e.target.value;
      if (v) applyPreset(v);
    });
    if (el.deletePresetBtn) on(el.deletePresetBtn,'click', () => {
      const cur = el.filterPresetSelect?.value;
      if (cur && confirm(`Delete preset "${cur}"?`)) deletePreset(cur);
    });
    if (el.exportFiltersBtn) on(el.exportFiltersBtn,'click', () => {
      const cur = el.filterPresetSelect?.value;
      exportPreset(cur);
    });
  }

  // -------- Source / Bypass --------
  function currentFileSource() {
    return activeManualSource === 'uploads' ? 'uploads' : 'input';
  }
  function syncSourceClass() {
    const inputSection = document.getElementById('inputSection');
    const uploadsSection = document.getElementById('uploadsSection');
    const fileSource = currentFileSource();

    document.body.classList.toggle('source-uploads', fileSource === 'uploads');
    document.body.classList.toggle('source-input', fileSource === 'input');

    if (inputSection) {
      inputSection.classList.toggle('hidden', fileSource !== 'input');
      inputSection.parentElement?.classList.toggle('hidden', fileSource !== 'input');
    }
    if (uploadsSection) {
      uploadsSection.classList.toggle('hidden', fileSource !== 'uploads');
      uploadsSection.parentElement?.classList.toggle('hidden', fileSource !== 'uploads');
    }
  }
  function emitManualFileSource() {
    if (!socket || !activeSessionId) return;
    const src = currentFileSource();
    if (lastSentSourceBySession[activeSessionId] === src) return;
    lastSentSourceBySession[activeSessionId] = src;
    socket.emit('set_manual_source', {
      session_id: activeSessionId,
      file_source: src,
      origin: activeManualSourceOrigin,
    });
  }
  function applyBypassState(onState) {
    document.body.classList.toggle('output-bypass', !!onState);
    if (el.bypassBtn) {
      el.bypassBtn.setAttribute('aria-pressed', onState ? 'true':'false');
      el.bypassBtn.textContent = onState ? 'Use Output Folder' : 'Bypass Output';
    }
  }

  // -------- Run / Cancel --------
  function animateButton(button) {
    if (!button) return;
    const r = button.getBoundingClientRect();
    const x = r.left + r.width / 2;
    const y = r.top + r.height / 2 + window.scrollY;
    spawnParticleBurst(x, y, 34, levelColorMap.INFO);
  }

  function runParser() {
    if (!socket || !el.runBtn) return;
    clearParserCompletionVisuals();
    const stored = localStorage.getItem('session_id');
    if (!activeSessionId && stored) {
      setActiveSession(stored);
      joinSession(stored);
      if (earlyQueue.length && el.outputDiv) {
        earlyQueue.forEach(d => renderParserOutput(d));
        flushBatch();
        earlyQueue = [];
      }
    }
    if (!activeSessionId) {
      addNewSession();
      if (earlyQueue.length && el.outputDiv) {
        earlyQueue.forEach(d => renderParserOutput(d));
        flushBatch();
        earlyQueue = [];
      }
    } else {
      joinSession(activeSessionId);
      if (earlyQueue.length && el.outputDiv) {
        earlyQueue.forEach(d => renderParserOutput(d));
        flushBatch();
        earlyQueue = [];
      }
    }
    const { valid: directUrls, invalid: invalidUrls } = parseDirectUrlField();
    if (invalidUrls.length) {
      alert(`Fix ${invalidUrls.length} invalid URL${invalidUrls.length === 1 ? '' : 's'} before running.`);
      el.directUrlTextarea?.focus();
      return;
    }
    if (directUrls.length > MAX_DIRECT_URLS) {
      alert(`Please limit direct URLs to ${MAX_DIRECT_URLS}.`);
      el.directUrlTextarea?.focus();
      return;
    }

    let desiredSource = (el.fileSourceSelect?.value === 'uploads') ? 'uploads' : 'input';
    if (manualUploadSelection && desiredSource !== 'uploads') {
      desiredSource = 'uploads';
      if (el.fileSourceSelect) el.fileSourceSelect.value = 'uploads';
    }
    if (directUrls.length && desiredSource === 'uploads') {
      alert('Direct URLs are ignored when Manual Uploads is selected. Clear the upload selection or switch the File Source.');
      return;
    }

    const desiredOrigin = desiredSource === 'uploads' ? 'user' : 'default';
    updateSessionSourceMeta(activeSessionId, desiredSource, desiredOrigin);
    emitManualFileSource();
  pipelineControl?.clearAttention('resolve');
  pipelineControl?.setPhase('run');
  pipelineControl?.focusStep('run', { scroll: false, highlight: true });
  logPanelControl?.expand();
    animateButton(el.runBtn);
    el.runBtn.disabled = true;
    el.runBtn.setAttribute('data-running','true');
    el.runBtn.textContent = 'Running...';
    const dispatchRun = () => {
      const payload = {
        session_id: activeSessionId,
        file_source: desiredSource,
        manual_source_origin: desiredOrigin,
      };
      if (manualUploadSelection && desiredSource === 'uploads') {
        payload.manual_upload_path = manualUploadSelection.relPath;
        payload.manual_upload_name = manualUploadSelection.name;
      }
      if (directUrls.length) {
        payload.direct_urls = directUrls;
      }
      socket.emit('run_parser', payload);
      setTimeout(() => socket && socket.emit('get_session_history', { session_id: activeSessionId }), 600);
    };

    if (joinedSessions.has(activeSessionId)) {
      dispatchRun();
    } else {
      socket.once('joined', function(data) {
        if (data.session_id === activeSessionId) {
          dispatchRun();
        }
      });
      joinSession(activeSessionId);
    }
    setTimeout(() => { if (!el.runBtn.getAttribute('data-running')) el.runBtn.disabled = false; }, 4000);
  }

  // -------- Prompt --------
  function handlePromptSubmit(e) {
    e.preventDefault();
    if (!socket || !el.promptInput || !activeSessionId) return;
    let raw = el.promptInput.value.trim();
    const urlMap = urlIndexMap[activeSessionId] || {};

    // Quick command: reopen last contest picker
    if (/^\/?contests$/i.test(raw)) {
      if (!window.showContestPicker()) alert('No cached contest options for this session yet.');
      el.promptInput.value = '';
      return;
    }
    // Expand comma-separated numbers for contests (or URLs if present)
    if (/^\d+(?:\s*,\s*\d+)*$/.test(raw)) {
      const nums = raw.split(/\s*,\s*/).filter(Boolean);
      if (lastPromptContext && lastPromptContext.kind === 'contest' && Object.keys(contestIndexMap).length) {
        raw = nums.filter(n => contestIndexMap[n] != null).join(',');
      } else if (lastPromptContext && lastPromptContext.kind === 'url') {
        raw = nums.join(',');
      } else if (Object.keys(urlMap).length) {
        raw = nums.map(n => urlMap[n] || n).join(',');
      }
    } else {
      const m = raw.match(/^\[?(\d+)\]?$/);
      if (m) {
        const n = m[1];
        if (lastPromptContext && lastPromptContext.kind === 'contest' && contestIndexMap[n] != null) {
          raw = n;
        } else if (lastPromptContext && lastPromptContext.kind === 'url') {
          raw = n;
        } else if (urlMap[n]) {
          raw = urlMap[n];
        }
      }
    }

    respondToPrompt(activeSessionId, raw, { message: 'Submitting response…' });
    el.promptInput.value = '';
    pipelineControl?.clearAttention('resolve');
    pipelineControl?.setPhase('resolve');
  }

  // -------- URLs --------
  function renderUrlSidebar(urls) {
    const sidebar = document.getElementById('urlSidebarBlock');
    if (!sidebar) return;
    const searchBox = sidebar.querySelector('.url-search-box');
    const listBox = sidebar.querySelector('#urlLinesBox');
    if (!searchBox || !listBox) return;

    function updateList(filter = '') {
      let filtered = urls;
      const q = filter.trim().toLowerCase();
      if (q) {
        if (q.startsWith('state:')) {
          filtered = urls.filter(u => u.toLowerCase().includes(q.slice(6).trim()));
        } else if (q.startsWith('county:')) {
          filtered = urls.filter(u => u.toLowerCase().includes(q.slice(7).trim()));
        } else {
          filtered = urls.filter(u => u.toLowerCase().includes(q));
        }
      }
      listBox.innerHTML = filtered.slice(0, 40).map((u, i) => {
        const short = u.length > 60 ? u.slice(0, 57) + '…' : u;
        return `<div class="url-sidebar-item" title="${u}" data-url="${encodeURIComponent(u)}">[${i+1}] ${short}</div>`;
      }).join('') +
        (filtered.length > 40 ? `<div class="url-sidebar-more">...and ${filtered.length-40} more</div>` : '');
      listBox.querySelectorAll('.url-sidebar-item').forEach(el => {
        el.onclick = () => {
          const url = decodeURIComponent(el.getAttribute('data-url'));
          if (typeof window.getActiveSessionId === 'function') {
            const sid = getActiveSessionId();
            if (sid) respondToPrompt(sid, url, { message: 'Processing URL…' });
          }
        };
      });
    }
    searchBox.removeEventListener('input', searchBox._urlSearchHandler || (() => {}));
    searchBox._urlSearchHandler = e => updateList(e.target.value);
    searchBox.addEventListener('input', searchBox._urlSearchHandler);
    updateList();
  }

  function fetchUrls() {
    fetch('/api/urls')
      .then(r => r.json())
      .then(d => {
        const list = d.urls || [];
        renderUrlSidebar(list);
      })
      .catch(() => {
        renderUrlSidebar([]);
      });
  }

  // -------- Output Mode --------
  function setOutputMode() {
    if (!socket) return;
    socket.emit('set_output_mode', { session_id: activeSessionId, mode: el.outputModeSelect.value });
  }

  // -------- Particles --------
  let particleCanvas, pctx, particleEffects = [], particleAnimating = false;
  function ensureParticleCanvas() {
    if (particleCanvas) return;
    particleCanvas = document.createElement('canvas');
    particleCanvas.className = 'particle-canvas';
    document.body.appendChild(particleCanvas);
    pctx = particleCanvas.getContext('2d');
    resizeParticleCanvas();
    window.addEventListener('resize', resizeParticleCanvas);
  }
  function resizeParticleCanvas() {
    if (!particleCanvas) return;
    particleCanvas.width = window.innerWidth;
    particleCanvas.height = window.innerHeight;
  }
  function spawnParticleBurst(x, y, count = 28, color = '#00ffe7') {
    if (PREFERS_REDUCED_MOTION) return;
    ensureParticleCanvas();
    const rectTop = window.scrollY;
    const originY = y - rectTop;
    const particles = [];
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i / count) + (Math.random()*0.6);
      const speed = 40 + Math.random()*140;
      particles.push({
        x,
        y: originY,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        life: 0,
        maxLife: 450 + Math.random()*350,
        size: 2 + Math.random()*3,
        hueShift: Math.random()*40 - 20,
        baseColor: color
      });
    }
    particleEffects.push({ particles, start: performance.now() });
    if (!particleAnimating) {
      particleAnimating = true;
      requestAnimationFrame(particleFrame);
    }
  }
  function particleFrame() {
    if (!pctx || !particleCanvas) { particleAnimating = false; return; }
    pctx.clearRect(0,0,particleCanvas.width,particleCanvas.height);
    const still = [];
    for (const effect of particleEffects) {
      let alive = false;
      for (const p of effect.particles) {
        p.life += 16;
        if (p.life < p.maxLife) {
          alive = true;
          const t = p.life / p.maxLife;
          p.x += p.vx * 0.016;
          p.y += p.vy * 0.016 + 40 * 0.016 * t;
          const alpha = 1 - (t*t);
          const grad = pctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size*3);
          grad.addColorStop(0, hexToRgba(p.baseColor, alpha));
          grad.addColorStop(1, hexToRgba(p.baseColor, 0));
          pctx.fillStyle = grad;
          pctx.beginPath();
          pctx.arc(p.x, p.y, p.size*2, 0, Math.PI*2);
          pctx.fill();
        }
      }
      if (alive) still.push(effect);
    }
    particleEffects = still;
    if (particleEffects.length) requestAnimationFrame(particleFrame);
    else {
      particleAnimating = false;
      pctx.clearRect(0,0,particleCanvas.width,particleCanvas.height);
    }
  }
  function hexToRgba(hex, a) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!m) return `rgba(0,255,231,${a})`;
    const r = parseInt(m[1],16), g = parseInt(m[2],16), b = parseInt(m[3],16);
    return `rgba(${r},${g},${b},${a})`;
  }

  // -------- Footer UI --------
  function initFooter() {
    if (!el.sessionFooter) return;
    on(el.footerPreview,'click',()=>{
      el.sessionFooter.classList.replace('minimized','expanded');
      const r = el.footerPreview.getBoundingClientRect();
      spawnParticleBurst(r.left + r.width/2, r.top + r.height/2 + window.scrollY, 26, '#00ffe7');
    });
    on(el.footerFull,'click',e=>{
      if (e.target.closest('.session-btn')) return;
      el.sessionFooter.classList.replace('expanded','minimized');
      const r = el.footerFull.getBoundingClientRect();
      spawnParticleBurst(r.left + r.width/2, r.top + 12 + window.scrollY, 22, '#eb4f43');
    });
  }

  // -------- Connection banners --------
  function showDisconnectedMessage() {
    if (!el.outputDiv || $('#socket-disconnect-msg')) return;
    const div = document.createElement('div');
    div.id = 'socket-disconnect-msg';
    div.className = 'socket-disconnect-banner';
    div.textContent = 'Connection lost… reconnecting (auto)';
    el.outputDiv.appendChild(div);
  }
  function hideDisconnectedMessage() { $('#socket-disconnect-msg')?.remove(); }

  // -------- Socket --------
  function loadSessionLogs(sid) { socket && socket.emit('get_session_history', { session_id: sid }); }

  function connectSocket() {
    if (socket && typeof socket.disconnect === 'function') socket.disconnect();

    let prevSessionId = localStorage.getItem('session_id') || '';
    try {
      if (prevSessionId.startsWith('{')) {
        const parsed = JSON.parse(prevSessionId);
        if (parsed.session_id) prevSessionId = parsed.session_id;
      }
    } catch {}

    socket = window.socket = io({
      query: { prev_session_id: prevSessionId },
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 2000,
      reconnectionDelayMax: 10000,
      transports: ['websocket'],
      pingInterval: 10000,
      pingTimeout: 60000
    });

    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('connect_error', handleConnectError);
    socket.on('session_id', handleSessionId);
    socket.on('session_history', handleSessionHistory);
    socket.on('parser_output', handleParserOutput);
    socket.on('output_bypass_state', ({ output_bypass }) => applyBypassState(!!output_bypass));
    socket.on('manual_source_state', handleManualSourceState);
    socket.on('session_list', handleSessionList);
  socket.on('session_state', handleSessionState);
    socket.on('session_deleted', handleSessionDeleted);
    socket.on('session_heartbeat', handleSessionHeartbeat);
    // Full contest list (paged modal) from backend, with context summary
    socket.on('contest_options', (payload) => {
      try {
        const { session_id, context, total_count, options } = payload || {};
        if (!Array.isArray(options) || !session_id) return;

        const normalized = options.map(opt => ({
          index: Number(opt.index ?? opt[0] ?? 0),
          label: opt.label ?? opt.name ?? String(opt.title ?? opt[1] ?? opt),
          meta: opt.meta ?? opt.summary ?? ''
        }));

        setContestOptions(session_id, normalized);
        contestIndexMap = Object.fromEntries(normalized.map(o => [String(o.index), o.label]));
        lastPromptContext = { kind: 'contest', options: normalized.map(o => `[${o.index}] ${o.label}`), session_id };

        if (session_id !== activeSessionId || !normalized.length) return;

        const ctxSummary = `
          <div class="small text-muted">
            ${total_count ?? normalized.length} option(s)
            ${context?.state && context.state.toLowerCase() !== 'unknown' ? ` • State: ${esc(context.state)}` : ''}
            ${context?.county && context.county.toLowerCase() !== 'unknown' ? ` • County: ${esc(context.county)}` : ''}
            ${context?.handler ? ` • Handler: ${esc(context.handler)}` : ''}
            ${context?.input_file ? ` • File: ${esc(context.input_file)}` : ''}
          </div>`.trim();

        openContestSelectionModal(session_id, normalized, ctxSummary, { placeholder: 'Enter contest index…', title: 'Select Contest' });
      } catch (e) {
        console.error('contest_options handler error:', e);
      }
    });
    socket.emit('get_sessions');

    function handleConnect() {
      hideDisconnectedMessage();
      joinedSessions.clear();
      renderSessionList();
      const sessions = getSessions();
      if (activeSessionId && sessions.includes(activeSessionId)) {
        joinSession(activeSessionId);
        socket.emit('get_session_history', { session_id: activeSessionId });
      } else if (sessions.length) {
        setActiveSession(sessions[0]);
        joinSession(sessions[0]);
        socket.emit('get_session_history', { session_id: sessions[0] });
      }
    }

    function handleDisconnect(reason) {
      showDisconnectedMessage();
      console.warn('Socket disconnected:', reason);
    }

    function handleConnectError(err) {
      showDisconnectedMessage();
      console.error('Socket connect error:', err);
    }

    function handleSessionId(data) {
      const sid = typeof data === 'string' ? data : (data && data.session_id) || '';
      const sessions = getSessions();
      if (sid && !sessions.includes(sid)) {
        sessions.push(sid);
        lsSetJSON('active_sessions', sessions);
      }
      setActiveSession(sid);
      joinSession(sid);
      renderSessionList();
      socket.emit('get_session_history', { session_id: sid });
    }

    function handleSessionHistory(data) {
      if (!data || !Array.isArray(data.logs)) return;
      data.logs.forEach(l => renderParserOutput(l));
      flushBatch();
      const lastPrompt = data.logs.slice().reverse().find(l => l.type === 'prompt');
      if (lastPrompt && el.promptInput) {
        el.promptInput.placeholder = lastPrompt.full_text || 'Type a command...';
        el.promptInput.parentElement?.classList.remove('hidden');
        el.promptInput.disabled = false;
        el.promptInput.focus();
        pipelineControl?.markAttention('resolve');
        pipelineControl?.focusStep('resolve', { scroll: false, highlight: false });
        logPanelControl?.expand();
      }
    }

    function handleParserOutput(d) {
      let overlayHidden = false;
      const hideOverlay = (options = {}) => {
        if (overlayHidden) return;
        overlayHidden = true;
        const opts = typeof options === 'object' && options !== null ? options : {};
        const delayMs = Number.isFinite(opts.delayMs) && opts.delayMs > 0 ? opts.delayMs : 0;
        const afterFrame = opts.afterFrame === true;
        const runner = () => { try { PendingOverlay.hide(); } catch (err) { void err; } };
        if (afterFrame && typeof requestAnimationFrame === 'function') {
          requestAnimationFrame(() => requestAnimationFrame(runner));
        } else if (delayMs > 0) {
          setTimeout(runner, delayMs);
        } else {
          runner();
        }
      };

      if (!activeSessionId) {
        hideOverlay();
        earlyQueue.push(d);
        return;
      }
      let handledCustomPrompt = false;
      if (d && typeof d.message === 'string') {
        updatePipelineForStatusLog(d.message);
      }
      if (d && d.session_id) {
        if (modalRestore && typeof modalRestore.setBusyForSession === 'function') {
          modalRestore.setBusyForSession(d.session_id, false);
        }
        if (d.type === 'manual_override') {
          const sessionId = d.session_id;
          const msg = String(d.message || d.full_text || '');
          const state = getManualState(sessionId);
          const foundMatch = msg.match(/Found\s+(\d+)\s+file/i);
          if (foundMatch) {
            const count = Number(foundMatch[1]);
            const folderMatch = msg.match(/in\s+'([^']+)'/i);
            const folder = folderMatch ? folderMatch[1] : state?.folder;
            resetManualState(sessionId, { count, folder });
            if (sessionId === activeSessionId) {
              pipelineControl?.setPhase('source');
              pipelineControl?.markAttention('source');
              pipelineControl?.focusStep('source', { scroll: false, highlight: true });
              ensureSectionExpanded('uploadsSection');
              if (window.matchMedia && window.matchMedia('(max-width: 900px)').matches && !document.body.classList.contains('sidebar-open')) {
                document.getElementById('sidebarToggleBtn')?.click();
              }
            }
          }
          const itemMatch = msg.match(/\[ManualOverride\]\s*\[(\d+)\]\s*(.+)$/i);
          if (itemMatch) addManualFileOption(sessionId, Number(itemMatch[1]), itemMatch[2]);
          const dirMatch = msg.match(/uploads_dir\s*=\s*(.+)$/i);
          if (dirMatch && state) state.baseDir = dirMatch[1].trim();
          if (/manual upload mode/i.test(msg) && sessionId === activeSessionId) {
            ensureSectionExpanded('uploadsSection');
            if (window.matchMedia && window.matchMedia('(max-width: 900px)').matches && !document.body.classList.contains('sidebar-open')) {
              document.getElementById('sidebarToggleBtn')?.click();
            }
          }
          if (/using manual upload/i.test(msg) && sessionId === activeSessionId) {
            pipelineControl?.clearAttention('source');
            pipelineControl?.setPhase('run');
          }
        }

        if (d.type === 'prompt' && MANUAL_UPLOAD_PROMPT_PATTERN.test(String(d.message || ''))) {
          const sessionId = d.session_id;
          const state = getManualState(sessionId);
          if (state) state.lastPromptHash = `${sessionId || ''}|${d.timestamp || ''}`;
          if (sessionId === activeSessionId) {
            handledCustomPrompt = true;
            ensureSectionExpanded('uploadsSection');
            if (window.matchMedia && window.matchMedia('(max-width: 900px)').matches && !document.body.classList.contains('sidebar-open')) {
              document.getElementById('sidebarToggleBtn')?.click();
            }
            openManualSelectionModal(sessionId, { placeholder: "Enter index or filename…" });
          }
        }

        const urlPromptList = d.context && Array.isArray(d.context.urls) ? d.context.urls : null;
        if (urlPromptList && urlPromptList.length) {
          const processed = d.context && typeof d.context.processed === 'object' ? d.context.processed : {};
          const meta = {
            placeholder: 'Enter URL index or filter…',
            title: 'Select URL',
            hint: 'Search or filter (state:/county:/text)'
          };
          if (d.session_id === activeSessionId) {
            handledCustomPrompt = true;
            openUrlSelectionModal(d.session_id, urlPromptList, processed, meta);
          } else {
            cacheUrlPromptContext(d.session_id, urlPromptList, processed, meta);
          }
        }
      }
      if (d && d.type === 'contest_options' && Array.isArray(d.options)) {
        const normalized = d.options.map(o => cloneContestOption({
          index: Number(o.index),
          label: o.label,
          meta: o.meta || '',
          metadata: o.metadata || {}
        }));
        setContestOptions(d.session_id, normalized);

        if (d.session_id === activeSessionId && normalized.length) {
          handledCustomPrompt = true;
          contestIndexMap = Object.fromEntries(normalized.map(o => [String(o.index), o.label]));
          lastPromptContext = { kind: 'contest', options: normalized.map(o => `[${o.index}] ${o.label}`), session_id: d.session_id };
          const ctx = d.context || {};
          const ctxSummary = `
            <div class="small text-muted">
              ${normalized.length} option(s)
              ${ctx.state && ctx.state.toLowerCase() !== 'unknown' ? ' • State: ' + esc(ctx.state) : ''}
              ${ctx.county && ctx.county.toLowerCase() !== 'unknown' ? ' • County: ' + esc(ctx.county) : ''}
              ${ctx.year ? ' • Year: ' + esc(String(ctx.year)) : ''}
              ${ctx.input_file ? ' • File: ' + esc(ctx.input_file) : ''}
            </div>`.trim();

          openContestSelectionModal(d.session_id, normalized, ctxSummary, { placeholder: 'Enter contest index…' });
          hideOverlay({ afterFrame: true });
        }
      }

      // If a generic prompt carries contest context (context.kind === 'contest'), normalize it
      if (d && d.type === 'prompt' && d.context && d.context.kind === 'contest' && Array.isArray(d.context.options)) {
        const optStrings = d.context.options;
        const parsed = optStrings.map((s, i) => {
          // Try to parse "[idx] Title (meta)" if present; fallback to raw
            const m = s.match(/^\s*\[(\d+)\]\s+(.+?)(?:\s+\(([^)]+)\))?\s*$/);
            if (m) {
              return {
                index: Number(m[1]),
                label: m[2],
                meta: m[3] || '',
                metadata: {}
              };
            }
            return { index: i, label: s, meta: '', metadata: {} };
        });
        if (parsed.length) {
          setContestOptions(d.session_id, parsed);
          if (d.session_id === activeSessionId) {
            handledCustomPrompt = true;
            const ctx = d.context;
            const ctxSummary = `
              <div class="small text-muted">
                ${parsed.length} option(s)
                ${ctx.state && ctx.state.toLowerCase() !== 'unknown' ? ' • State: ' + esc(ctx.state) : ''}
                ${ctx.county && ctx.county.toLowerCase() !== 'unknown' ? ' • County: ' + esc(ctx.county) : ''}
                ${ctx.year ? ' • Year: ' + esc(String(ctx.year)) : ''}
              </div>`.trim();
            contestIndexMap = Object.fromEntries(parsed.map(o => [String(o.index), o.label]));
            lastPromptContext = { kind: 'contest', options: parsed.map(o => `[${o.index}] ${o.label}`), session_id: d.session_id };
            openContestSelectionModal(d.session_id, parsed, ctxSummary, { placeholder: 'Enter contest index…' });
            hideOverlay({ afterFrame: true });
          }
        }
      }
      // Normalize message for display (fallback to context/description if message is blank)
      const msg = (d && typeof d.message === 'string' && d.message.trim())
        ? d.message
        : (d && typeof d.context === 'string' && d.context.trim())
          ? d.context
          : (d && typeof d.description === 'string' && d.description.trim())
            ? d.description
            : '';
      if (msg && d) d.message = msg;
      
      // Detect and render backend contest menus (“Available contests:”)
      if (d && typeof d.message === 'string' && /available contests:/i.test(d.message)) {
        const options = parseIndexedMenu(d.message).map(opt => ({ ...opt, metadata: {} }));
        if (options.length) {
          setContestOptions(d.session_id, options);
          contestIndexMap = Object.fromEntries(options.map(o => [String(o.index), o.label]));
          lastPromptContext = { kind: 'contest', options: options.map(o => `[${o.index}] ${o.label}`), session_id: d.session_id };
          const ctxSummary = `<div class="small text-muted">${options.length} option(s)</div>`;
          if (d.session_id === activeSessionId) {
            handledCustomPrompt = true;
            openContestSelectionModal(d.session_id, options, ctxSummary, { placeholder: 'Enter contest index…' });
            hideOverlay({ afterFrame: true });
          }
        }
      }

      if (d && typeof d.message === 'string' && /no match; try again/i.test(d.message)) {
        const sid = d.session_id || activeSessionId;
        if (sid === activeSessionId) {
          const options = getContestOptions(sid);
          if (options.length) {
            handledCustomPrompt = true;
            const ctxSummary = `<div class="small text-muted">${options.length} option(s)</div>`;
            openContestSelectionModal(sid, options, ctxSummary, { placeholder: 'Enter contest index…' });
            hideOverlay({ afterFrame: true });
          }
        }
      }

        // Prompt handling
      if (d && d.type === 'prompt' && d.session_id === activeSessionId && !handledCustomPrompt) {
        const ctx = d.context || {};
        pipelineControl?.markAttention('resolve');
        pipelineControl?.focusStep('resolve', { scroll: false, highlight: true });
        logPanelControl?.expand();
        // Prefer confirmed (shape: [[fmt, href, group, filename], ...])
        if (Array.isArray(ctx.confirmed) && ctx.confirmed.length) {
          const opts = ctx.confirmed.map((arr, i) => ({
            index: i,
            format: String(arr?.[0] || ''),
            filename: String(arr?.[3] || ''),
            contest: String(arr?.[2] || 'Other'),
            href: String(arr?.[1] || '')
          }));
          // Hide inline prompt while modal is used
          if (el.promptInput && el.promptInput.parentElement) el.promptInput.parentElement.classList.add('hidden');
          showDownloadModal(opts, ctx.summary || '', function(selectedIdx) {
            const value = selectedIdx == null ? 'n' : String(selectedIdx);
            respondToPrompt(activeSessionId, value, { message: 'Processing download choice…' });
            pipelineControl?.clearAttention('resolve');
            pipelineControl?.setPhase('resolve');
          });
        } else if (Array.isArray(ctx.options) && ctx.options.length) {
          // Fallback: ctx.options as display strings (parse best-effort)
          const opts = ctx.options.map((opt, i) => {
            const s = String(opt || '');
            const m = /^(\w+)\s+\(([^)]+)\)\s+\[([^\]]*)\]/.exec(s) || [];
            return {
              index: i,
              format: m[1] || s.split(' ')[0] || '',
              filename: m[2] || s,
              contest: m[3] || ''
            };
          });
          if (el.promptInput && el.promptInput.parentElement) el.promptInput.parentElement.classList.add('hidden');
          showDownloadModal(opts, ctx.summary || '', function(selectedIdx) {
            const value = selectedIdx == null ? 'n' : String(selectedIdx);
            respondToPrompt(activeSessionId, value, { message: 'Processing download choice…' });
            pipelineControl?.clearAttention('resolve');
            pipelineControl?.setPhase('resolve');
          });
        } else {
          showPromptModal(d.message, function(userInput) {
            respondToPrompt(activeSessionId, userInput, { message: 'Submitting response…' });
          });
        }
      }

      // Normal log handling
      renderParserOutput(d);
      if (d && d.session_id) appendCacheLog(d.session_id, d);

      // Status: completed logic
      if (d && d.session_id === activeSessionId && (d.type === 'status' || d.type === 'cancel' || d.type === 'error')) {
        const msg = String(d.message || '');
        if (/completed/i.test(msg)) {
          pipelineControl?.clearAttention('resolve');
          pipelineControl?.focusStep('review', { scroll: false, highlight: true });
          logPanelControl?.expand();
          if (el.runBtn) {
            el.runBtn.disabled = false;
            el.runBtn.removeAttribute('data-running');
            el.runBtn.textContent = 'Run Parser';
            updateRunButtonLock();
          }
        } else if (/cancel/i.test(msg) || /run cancelled/i.test(msg)) {
          pipelineControl?.clearAttention('resolve');
          pipelineControl?.setPhase('prepare');
          if (el.runBtn) {
            el.runBtn.disabled = false;
            el.runBtn.removeAttribute('data-running');
            el.runBtn.textContent = 'Run Parser';
            updateRunButtonLock();
          }
        } else if (/error/i.test(msg) || /failed/i.test(msg)) {
          if (el.runBtn) {
            el.runBtn.disabled = false;
            el.runBtn.removeAttribute('data-running');
            el.runBtn.textContent = 'Run Parser';
            updateRunButtonLock();
          }
        }
      }

      hideOverlay();
    }

    function handleManualSourceState({ session_id, file_source, manual_source_origin }) {
      updateSessionSourceMeta(session_id, file_source, manual_source_origin, { fromServer: true });
    }

    function handleSessionList(data) {
      if (!Array.isArray(data.sessions)) return;
      sessionMirror.replace(data.sessions);
      const ids = syncSessionCache(data.sessions.map(s => (s && typeof s === 'object' && s.session_id) ? s.session_id : s));
      renderSessionList();
      if (!ids.includes(activeSessionId)) setActiveSession(ids[0] || '');
      applySessionState(activeSessionId);
    }

    function handleSessionDeleted({ session_id }) {
      sessionMirror.remove(session_id);
      manualUploadSelectionBySession.delete(session_id);
      directUrlDraftBySession.delete(session_id);
      const filtered = getSessions().filter(s => s !== session_id);
      syncSessionCache(filtered);
      if (activeSessionId === session_id) setActiveSession(filtered[0] || '');
      renderSessionList();
    }

    function handleSessionState(payload) {
      if (!payload || typeof payload !== 'object') return;
      const sid = payload.session_id || (payload.metadata && payload.metadata.session_id);
      if (!sid) return;
      try { PendingOverlay.hide(); } catch (err) { void err; }
      if (modalRestore && typeof modalRestore.setBusyForSession === 'function') {
        modalRestore.setBusyForSession(sid, false);
      }
      const meta = (payload.metadata && typeof payload.metadata === 'object')
        ? payload.metadata
        : { session_id: sid, state: payload.state, phase: payload.phase };
      sessionMirror.upsert(meta);
      syncSessionCache();
      if (sid === activeSessionId) {
        applySessionState(sid, meta);
      }
    }

    function handleSessionHeartbeat({ session_id }) {
      const btn = document.querySelector(`.session-btn[data-sid="${session_id}"]`);
      if (!btn) return;
      let hb = btn.querySelector('.heartbeat-indicator');
      if (!hb) {
        hb = document.createElement('span');
        hb.className = 'heartbeat-indicator';
        hb.innerHTML = `
          <svg width="36" height="18" viewBox="0 0 36 18" class="hb-svg">
            <polyline class="ekg-wave" points="0,9 6,9 9,3 12,15 15,9 18,9 21,6 24,12 27,9 36,9"
              fill="none" stroke="#00ffe7" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
          </svg>`;
        btn.appendChild(hb);
      }
      const wave = hb.querySelector('.ekg-wave');
      wave.classList.add('pulse');
      clearTimeout(hb._flatlineTimeout);
      hb._flatlineTimeout = setTimeout(() => {
        wave.setAttribute('points','0,9 36,9');
        wave.classList.remove('pulse');
        wave.classList.add('flatline');
      }, 3000);
    }

    // Prompt Modal (basic)
    function showPromptModal(message, callback) {
      let context = null;
      try {
        context = typeof message === 'object' ? message : null;
      } catch {}
      if (!context && window.lastPromptContext) context = window.lastPromptContext;

      if (context && Array.isArray(context.options) && context.confirmed) {
        const opts = context.options.map((opt, i) => {
          const m = /^(\w+)\s+\(([^)]+)\)\s+\[([^\]]*)\]/.exec(opt);
          return {
            index: i,
            format: m ? m[1] : '',
            filename: m ? m[2] : opt,
            contest: m ? m[3] : '',
            raw: opt
          };
        });
        const summary = context.summary || '';
        showDownloadModal(opts, summary, function(selectedIdx) {
          if (selectedIdx == null) {
            callback('n');
          } else {
            callback(String(selectedIdx));
          }
        });
        window.lastPromptContext = context;
        return;
      }

      if (!el.promptInput) return;
      el.promptInput.placeholder = typeof message === 'string' ? message : "Enter value:";
      el.promptInput.disabled = false;
      el.promptInput.value = '';
      el.promptInput.parentElement?.classList.remove('hidden');
      el.promptInput.focus();
      pipelineControl?.markAttention('resolve');
      pipelineControl?.focusStep('resolve', { scroll: false, highlight: true });
      logPanelControl?.expand();
      el.promptInput.onkeydown = function(e) {
        if (e.key === 'Enter') {
          e.preventDefault();
          const val = el.promptInput.value.trim();
          el.promptInput.value = '';
          el.promptInput.disabled = true;
          el.promptInput.parentElement?.classList.add('hidden');
          callback(val);
          pipelineControl?.clearAttention('resolve');
          pipelineControl?.setPhase('resolve');
        }
        if (e.key === 'Escape') {
          el.promptInput.value = '';
          el.promptInput.disabled = true;
          el.promptInput.parentElement?.classList.add('hidden');
          callback('');
          pipelineControl?.clearAttention('resolve');
          pipelineControl?.setPhase('resolve');
        }
      };
    }
  }
  // Add a single cleanup for navigation/close instead of 'unload'
  function cleanupOnPageHide() {
    try { if (window.socket && window.socket.connected) window.socket.disconnect(); } catch {}
  }
  // Prefer pagehide (fires on bfcache) instead of unload/beforeunload
  window.addEventListener('pagehide', cleanupOnPageHide, { once: true });
  // -------- Init sub-blocks --------
  function initFileSource() {
    if (!el.fileSourceSelect) return;
    const qpSource = new URLSearchParams(location.search).get('source');
    if (qpSource && ['uploads','input'].includes(qpSource.toLowerCase()))
      el.fileSourceSelect.value = qpSource.toLowerCase();
    syncSourceClass();
    on(el.fileSourceSelect,'change', () => {
      const src = el.fileSourceSelect.value === 'uploads' ? 'uploads' : 'input';
      updateSessionSourceMeta(activeSessionId, src, 'user');
      emitManualFileSource();
      pipelineControl?.setPhase('source');
    });
  }
  function initOutputBypass() {
    on(el.bypassBtn,'click', () => {
      if (!socket) return;
      el.bypassBtn.disabled = true;
      socket.emit('toggle_output_bypass', { session_id: activeSessionId });
      setTimeout(()=> el.bypassBtn && (el.bypassBtn.disabled = false), 400);
    });
  }
  function initCollapsibles() {
    $$('.collapsible-btn[data-target]').forEach(btn => {
      on(btn,'click', () => {
        const id = btn.getAttribute('data-target');
        const panel = id && document.getElementById(id);
        if (!panel) return;
        const wasHidden = panel.classList.contains('hidden');
        panel.classList.toggle('hidden');
        // If becoming visible, refresh any mounted folder panels inside
        if (wasHidden) {
          panel.querySelectorAll('.folder-panel').forEach(fp => fp._refresh && fp._refresh());
        }
      });
    });
  }
  function initQuickActions() {
    on($('#actionClearOutput'),'click', clearOutput);
    on($('#actionCopyOutput'),'click', () => {
      if (!el.outputDiv) return;
      const text = el.outputDiv.innerText || '';
      if (!text) return;
      if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(text).catch(()=>{});
      } else {
        const ta = document.createElement('textarea');
        ta.className = 'clipboard-temp';
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch {}
        ta.remove();
      }
    });
  }
  function initSessionActions() { on(el.addSessionBtn,'click', addNewSession); }
  function initRunCancel() {
    on(el.runBtn,'click', runParser);
    on(el.cancelBtn,'click', e => {
      e.preventDefault();
      if (cancelRequested || !socket) return;
      cancelRequested = true;
      socket.emit('cancel_parser', { session_id: activeSessionId });
      renderParserOutput({ level:'CANCELLED', type:'cancel', message:'Cancellation requested', session_id: activeSessionId });
      if (el.cancelBtn) {
        el.cancelBtn.disabled = true;
        el.cancelBtn.textContent = 'Canceling...';
      }
      setTimeout(() => {
        cancelRequested = false;
        if (el.cancelBtn) {
          el.cancelBtn.disabled = false;
          el.cancelBtn.textContent = 'Cancel';
        }
      }, 3500);
    });
  }
  function initPrompt() {
    on(el.promptForm,'submit', handlePromptSubmit);
    on(el.promptInput,'keydown', e => { if (e.key === 'Escape') e.target.value=''; });
  }
  function initLogFilters() {
    on(el.logFilterSelect,'change', filterLogs);
    on(logTypeSelect,'change', filterLogs);
  }
  function initUrlList() {
    if (el.addUrlForm) {
      on(el.addUrlForm,'submit', e => {
        e.preventDefault();
        const urlField = $('#newUrl');
        const urlVal = urlField?.value.trim();
        if (!urlVal) return;
        if (!isUserSuppliedSafeHttpUrl(urlVal)) {
          alert('Invalid URL (require http/https, no credentials).');
          return;
        }
        fetch('/api/urls', {
          method:'POST',
          headers:{'Content-Type':'application/json; charset=utf-8'},
          body: JSON.stringify({ url: urlVal })
        })
        .then(r=>r.json())
        .then(res => {
          if (res.success) {
            fetchUrls();
            el.addUrlForm.reset();
          } else {
            alert(res.error || 'Failed to add URL.');
          }
        })
        .catch(()=> alert('Network error adding URL.'));
      });
    }
    const refreshBtn = document.getElementById('refreshUrlListBtn');
    if (refreshBtn) {
      on(refreshBtn, 'click', () => {
        fetchUrls();
      });
    }
  }
  function isUserSuppliedSafeHttpUrl(u) {
    try {
      const x = new URL(u);
      if (!/^https?:$/.test(x.protocol)) return false;
      if (x.username || x.password) return false;
      return true;
    } catch { return false; }
  }
  function initOutputMode() { on(el.outputModeSelect,'change', setOutputMode); }
  function initFolderBrowseButtons() {
    // Remove any header-level “Browse” buttons; inline panel replaces the modal.
    $$('.collapsible-btn[data-target]').forEach(btn => {
      let sib = btn.nextElementSibling;
      while (sib) {
        const isBtn = sib.tagName === 'BUTTON';
        const looksBrowse = /browse/i.test((sib.textContent || '') + ' ' + (sib.title || ''));
        if (isBtn && looksBrowse) {
          const next = sib.nextElementSibling;
          sib.remove();
          sib = next;
          continue;
        }
        break;
      }
    });
    // Also remove any leftover preview elements we might have injected previously
    document.querySelectorAll('.folder-preview')?.forEach(n => n.remove());
  }
  function initSidebarToggle() {
    const aside = document.getElementById('sidebar') || document.querySelector('aside.sidebar');
    const section = document.getElementById('urlSidebarBlock');           // inner section (desktop collapse)
    const drawerBtn = document.getElementById('sidebarToggleBtn');        // mobile drawer toggle
    const backdrop = document.getElementById('sidebarBackdrop');          // mobile drawer backdrop
    const collapseBtn = document.getElementById('toggleUrlSidebarBtn');   // optional desktop collapse button
    const main = document.querySelector('.container-main');
    const nav = document.querySelector('.navbar');
    const footer = document.getElementById('sessionFooter');
    const mql = window.matchMedia('(max-width: 900px)');
    const mobileFloatMql = window.matchMedia('(max-width: 700px)');
    const bodyEl = document.body;
    let lastFocus = null;
    let untrap = null;
    let resetSwipe = () => {};

    const getFocusable = (root) =>
      Array.from(root.querySelectorAll('a[href], button, textarea, input, select, [tabindex]:not([tabindex="-1"])'))
        .filter(el => !el.hasAttribute('disabled') && !el.getAttribute('aria-hidden'));

    function trapFocus(container) {
      const nodes = getFocusable(container);
      if (!nodes.length) return () => {};
      const first = nodes[0], last = nodes[nodes.length - 1];
      function onKey(e) {
        if (e.key !== 'Tab') return;
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
      document.addEventListener('keydown', onKey);
      return () => document.removeEventListener('keydown', onKey);
    }

    function setInert(on) {
      // Prefer inert when available; fall back to aria-hidden
      [nav, main, footer].forEach(el => {
        if (!el) return;
        if (on) {
          if ('inert' in el) el.inert = true;
          el.setAttribute('aria-hidden', 'true');
        } else {
          if ('inert' in el) el.inert = false;
          el.removeAttribute('aria-hidden');
        }
      });
    }

    const FLOAT_CLASS = 'sidebar-float-enabled';
    const applyFloatState = () => {
      const enabled = mobileFloatMql.matches && !!(aside && drawerBtn);
      bodyEl.classList.toggle(FLOAT_CLASS, enabled);
      if (!enabled) resetSwipe?.();
    };
    applyFloatState();
    mobileFloatMql.addEventListener?.('change', applyFloatState);

    // Mobile drawer behavior (off-canvas)
    if (aside && drawerBtn && backdrop) {
      let swipeZone = aside.querySelector('.sidebar-swipe-zone');
      if (!swipeZone) {
        swipeZone = document.createElement('div');
        swipeZone.className = 'sidebar-swipe-zone';
        swipeZone.setAttribute('aria-hidden', 'true');
        aside.appendChild(swipeZone);
      }

      const swipeState = {
        active: false,
        pointerId: null,
        startX: 0,
        startY: 0,
        deltaX: 0,
        locked: false,
      };
      const SWIPE_ACTIVATION = 26;
      const SWIPE_THRESHOLD = 120;
      const SWIPE_SLOPE_RATIO = 1.35;

      function resetSwipeState(forceTransform = true) {
        swipeState.active = false;
        swipeState.pointerId = null;
        swipeState.deltaX = 0;
        swipeState.locked = false;
        aside.style.transition = '';
        if (forceTransform) {
          aside.style.transform = '';
          aside.classList.remove('sidebar-dragging');
        }
      }

      resetSwipe = () => resetSwipeState(true);

      const open = () => {
        lastFocus = document.activeElement;
        resetSwipeState();
        document.body.classList.add('sidebar-open');
        drawerBtn.setAttribute('aria-expanded', 'true');
        setInert(true);
        untrap = trapFocus(aside);
        setTimeout(() => aside.querySelector('.url-search-box')?.focus(), 0);
      };
      const close = () => {
        resetSwipeState();
        document.body.classList.remove('sidebar-open');
        drawerBtn.setAttribute('aria-expanded', 'false');
        setInert(false);
        if (untrap) { untrap(); untrap = null; }
        (lastFocus && typeof lastFocus.focus === 'function') ? lastFocus.focus() : drawerBtn.focus();
      };
      const toggle = () => (document.body.classList.contains('sidebar-open') ? close() : open());

      drawerBtn.addEventListener('click', toggle);
      backdrop.addEventListener('click', close);
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.body.classList.contains('sidebar-open')) close();
      });
      mql.addEventListener?.('change', (e) => { if (!e.matches) close(); });

      function onSwipePointerDown(event) {
        if (!mobileFloatMql.matches) return;
        if (!document.body.classList.contains('sidebar-open')) return;
        if (event.pointerType && event.pointerType !== 'touch' && event.pointerType !== 'pen') return;
        swipeState.active = true;
        swipeState.pointerId = event.pointerId;
        swipeState.startX = event.clientX;
        swipeState.startY = event.clientY;
        swipeState.deltaX = 0;
        swipeState.locked = false;
        aside.classList.add('sidebar-dragging');
        aside.style.transition = 'none';
        try { swipeZone.setPointerCapture(event.pointerId); } catch { /* no-op */ }
      }

      function onSwipePointerMove(event) {
        if (!swipeState.active || event.pointerId !== swipeState.pointerId) return;
        const deltaX = event.clientX - swipeState.startX;
        const deltaY = event.clientY - swipeState.startY;
        if (!swipeState.locked) {
          if (Math.abs(deltaX) < SWIPE_ACTIVATION) return;
          if (Math.abs(deltaX) < Math.abs(deltaY) * SWIPE_SLOPE_RATIO) {
            cancelSwipe(event);
            return;
          }
          swipeState.locked = true;
        }
        event.preventDefault();
        const translate = Math.min(0, deltaX);
        swipeState.deltaX = translate;
        aside.style.transform = `translateX(${translate}px)`;
      }

      function finalizeSwipe(event) {
        if (!swipeState.active || event.pointerId !== swipeState.pointerId) return;
        try { swipeZone.releasePointerCapture(event.pointerId); } catch { /* no-op */ }
        aside.style.transition = '';
        aside.classList.remove('sidebar-dragging');
        const distance = Math.abs(swipeState.deltaX);
        const shouldClose = swipeState.locked && distance >= SWIPE_THRESHOLD;
        resetSwipeState(!shouldClose);
        if (shouldClose) {
          close();
        } else {
          aside.style.transform = '';
        }
      }

      function cancelSwipe(event) {
        if (!swipeState.active || (event && event.pointerId !== swipeState.pointerId)) return;
        if (event?.pointerId != null) {
          try { swipeZone.releasePointerCapture(event.pointerId); } catch { /* ignore */ }
        }
        aside.style.transition = '';
        resetSwipeState();
      }

      swipeZone.addEventListener('pointerdown', onSwipePointerDown);
      swipeZone.addEventListener('pointermove', onSwipePointerMove);
      swipeZone.addEventListener('pointerup', finalizeSwipe);
      swipeZone.addEventListener('pointercancel', cancelSwipe);
    }

    // Desktop: collapse/expand the URL section
    if (collapseBtn && section) {
      collapseBtn.addEventListener('click', () => {
        if (mql.matches) { drawerBtn?.click(); return; }
        const hidden = section.classList.toggle('hidden');
        collapseBtn.setAttribute('aria-expanded', String(!hidden));
        collapseBtn.textContent = hidden ? '➡️' : '⬅️';
        if (!hidden) section.querySelector('.url-search-box')?.focus();
      });
    }
  }
  function initPipelineStepper() {
    const container = el.pipelineStepper;
    if (!container) return null;
    const steps = Array.from(container.querySelectorAll('.pipeline-step[data-step-id]'));
    if (!steps.length) return null;

    const order = steps.map(step => step.dataset.stepId).filter(Boolean);
    const indexLookup = new Map(order.map((id, idx) => [id, idx]));
    let activeId = order[0] || null;
    let attentionId = null;
    let manualLockIdx = activeId && indexLookup.has(activeId) ? indexLookup.get(activeId) : -1;
    const highlightTimers = new WeakMap();
    const visibleRatios = new Map();
    const forcedStates = new Map();

    const targets = new Map();
    document.querySelectorAll('[data-step-target]').forEach(node => {
      const id = node.getAttribute('data-step-target');
      if (!id || !indexLookup.has(id)) return;
      if (node.closest('.navbar')) return;
      const existing = targets.get(id);
      if (!existing || node.offsetTop < existing.offsetTop) targets.set(id, node);
    });

    function highlightTarget(node) {
      if (!node || !node.classList) return;
      const prev = highlightTimers.get(node);
      if (prev) clearTimeout(prev);
      node.classList.add('step-target-highlight');
      highlightTimers.set(node, setTimeout(() => {
        node.classList.remove('step-target-highlight');
        highlightTimers.delete(node);
      }, 1400));
    }

    function applyForcedStates() {
      forcedStates.forEach((state, stepId) => {
        if (!state || !indexLookup.has(stepId)) return;
        const step = steps[indexLookup.get(stepId)];
        if (!step) return;
        if (step.dataset.state === 'active' && state === 'done') return;
        step.dataset.state = state;
        if (state !== 'attention') step.classList.remove('attention-only');
      });
    }

    function reapplyStates() {
      if (activeId && indexLookup.has(activeId)) {
        setPhaseInternal(activeId, { source: 'forced' });
      } else if (order[0]) {
        setPhaseInternal(order[0], { source: 'forced' });
      } else {
        applyForcedStates();
      }
    }

    function setPhaseInternal(id, { attention = false, source = 'manual' } = {}) {
      if (!indexLookup.has(id)) return;
      const idx = indexLookup.get(id);

      if (source === 'manual') {
        manualLockIdx = idx;
      } else if (source === 'observer' && manualLockIdx >= 0 && idx < manualLockIdx) {
        return;
      }

      if (!attention && attentionId && indexLookup.get(attentionId) <= idx) attentionId = null;
      if (attention) attentionId = id;

      activeId = id;
      steps.forEach((step, i) => {
        step.removeAttribute('data-state');
        if (i < idx) step.dataset.state = 'done';
        else if (i === idx) {
          step.dataset.state = attentionId === id ? 'attention' : 'active';
          step.classList.remove('attention-only');
        }
      });
      applyForcedStates();
    }

    if (activeId) setPhaseInternal(activeId, { source: 'manual' });

    function setAttentionOnly(id, flag) {
      if (!indexLookup.has(id)) return;
      const step = steps[indexLookup.get(id)];
      if (!step) return;
      step.classList.toggle('attention-only', !!flag);
    }

    function focusTarget(id, options = {}) {
      const node = targets.get(id);
      if (!node) return;
      if (options.scroll !== false) {
        try { node.scrollIntoView({ behavior: 'smooth', block: 'start', inline: 'nearest' }); } catch {}
      }
      if (options.highlight !== false) highlightTarget(node);
    }

    steps.forEach(step => {
      step.addEventListener('click', () => {
        const id = step.dataset.stepId;
        if (!id) return;
        setPhaseInternal(id, { source: 'manual' });
        focusTarget(id, { highlight: true });
      });
      step.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); step.click(); }
      });
    });

    const focusHandler = (event) => {
      const node = event.target?.closest?.('[data-step-target]');
      if (!node) return;
      const id = node.getAttribute('data-step-target');
      if (!indexLookup.has(id)) return;
      setPhaseInternal(id, { source: 'manual' });
    };
    document.addEventListener('focusin', focusHandler);

    if (window.IntersectionObserver) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          const id = entry.target.getAttribute('data-step-target');
          if (!id || !indexLookup.has(id)) return;
          if (entry.isIntersecting) visibleRatios.set(id, entry.intersectionRatio);
          else visibleRatios.delete(id);
        });
        if (!visibleRatios.size) return;
        let chosen = null;
        let bestRatio = 0.25;
        visibleRatios.forEach((ratio, id) => {
          if (ratio > bestRatio) {
            bestRatio = ratio;
            chosen = id;
          } else if (ratio === bestRatio && chosen) {
            if (indexLookup.get(id) > indexLookup.get(chosen)) chosen = id;
          }
        });
        if (chosen && chosen !== activeId) setPhaseInternal(chosen, { source: 'observer' });
      }, { threshold: [0.2, 0.35, 0.5] });
      targets.forEach(node => observer.observe(node));
    }

    return {
      setPhase(id, opts = {}) { setPhaseInternal(id, { ...opts, source: opts.source || 'manual' }); },
      markAttention(id) { setPhaseInternal(id, { attention: true, source: 'manual' }); },
      clearAttention(id) {
        if (!id || !indexLookup.has(id)) return;
        if (attentionId === id) attentionId = null;
        const targetId = (activeId && indexLookup.has(activeId)) ? activeId : (order[0] || id);
        setPhaseInternal(targetId, { source: 'manual' });
      },
      focusStep(id, opts = {}) {
        if (!indexLookup.has(id)) return;
        setPhaseInternal(id, { attention: opts.attention, source: opts.source || 'manual' });
        focusTarget(id, opts);
      },
      reset() {
        attentionId = null;
        manualLockIdx = order.length ? 0 : -1;
        const first = order[0];
        steps.forEach(step => step.removeAttribute('data-state'));
        steps.forEach(step => step.classList.remove('attention-only'));
        if (first) setPhaseInternal(first, { source: 'manual' });
      },
      releaseLock() { manualLockIdx = -1; },
      getActive() { return activeId; },
      attentionOnly(id, flag) { setAttentionOnly(id, flag); },
      setStepState(id, state) {
        if (!indexLookup.has(id)) return;
        if (!state) forcedStates.delete(id);
        else forcedStates.set(id, state);
        reapplyStates();
      },
    };
  }
  function initLogPanelToggle() {
    const body = el.logPanelBody;
    const btn = el.logToggleBtn;
    const panel = el.logPanel;
    if (!body || !btn || !panel) return null;

    function sync() {
      const open = !body.classList.contains('collapsed');
      btn.setAttribute('aria-expanded', String(open));
      btn.textContent = open ? 'Hide Log' : 'Show Log';
      panel.classList.toggle('is-open', open);
    }

    function setOpen(open) {
      body.classList.toggle('collapsed', !open);
      sync();
    }

    on(btn, 'click', () => setOpen(body.classList.contains('collapsed')));
    sync();

    return {
      setOpen,
      expand: () => setOpen(true),
      collapse: () => setOpen(false),
      toggle: () => setOpen(body.classList.contains('collapsed')),
      isCollapsed: () => body.classList.contains('collapsed')
    };
  }
  function initBootstrapUI() {
    if (!window.bootstrap) return;
    if (document.body.dataset.allowStyleAttr !== "1") return;
    $$('[data-bs-toggle="tooltip"]').forEach(n => window.bootstrap.Tooltip.getOrCreateInstance(n));
    $$('[data-bs-toggle="popover"]').forEach(n => window.bootstrap.Popover.getOrCreateInstance(n));
  }
  function initConfirmDelegation() {
    document.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-confirm]');
      if (!btn) return;
      const msg = btn.getAttribute('data-confirm') || 'Are you sure?';
      if (!window.confirm(msg)) e.preventDefault();
    });
  }
  function portalTooltip(icon) {
    function open() {
      const rect = icon.getBoundingClientRect();
      const isMobile = window.innerWidth < 700;
      const spaceRight = window.innerWidth - rect.right;
      const tooltipWidth = Math.min(340, window.innerWidth - 32);

      icon.classList.add('is-open');
      const useBelow = isMobile || spaceRight < tooltipWidth + 16;
      icon.classList.toggle('tooltip-below', useBelow);
      icon.classList.toggle('tooltip-right', !useBelow);
    }
    function close() {
      icon.classList.remove('is-open', 'tooltip-below', 'tooltip-right');
    }

    icon.addEventListener('mouseenter', open);
    icon.addEventListener('focus', open);
    icon.addEventListener('mouseleave', close);
    icon.addEventListener('blur', close);
    icon.addEventListener('touchstart', (e) => {
      e.preventDefault();
      icon.classList.contains('is-open') ? close() : open();
    }, { passive: false });
    document.addEventListener('touchstart', (e) => { if (!icon.contains(e.target)) close(); });
    document.addEventListener('click', (e) => { if (!icon.contains(e.target)) close(); });
    icon.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
  }

  // -------- Init master --------
  function init() {
    initBootstrapUI();
    initConfirmDelegation();
    pipelineControl = initPipelineStepper();
    logPanelControl = initLogPanelToggle();
    pipelineControl?.reset();
  updatePipelineMetadataForActive();
    initFileSource();
  initManualUploadControl();
  initDirectUrlControl();
  initModalQuickAdd();
    initOutputBypass();
    initCollapsibles();
    initQuickActions();
    initFooter();
    initSessionActions();
    initRunCancel();
    initPrompt();
    initLogFilters();
    initUrlList();
    initOutputMode();
    initSidebarToggle();
    initFolderPanels();
    initFolderBrowseButtons();
    initFilterPresets();
    fetchUrls();
    connectSocket();

    const btn = document.getElementById('toggleInstructionsBtn');
    const panel = document.getElementById('instructionsPanel');
    if (btn && panel) {
      btn.addEventListener('click', function () {
        const expanded = btn.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-expanded', String(!expanded));
        panel.classList.toggle('hidden');
        btn.textContent = expanded ? '📖 Show Instructions' : '📖 Hide Instructions';
        if (!expanded) panel.focus();
      });
    }

    $$('.help-icon').forEach(portalTooltip);

    if (!activeSessionId && el.outputDiv) {
      el.outputDiv.innerHTML = `<div class="no-session-hint">
        No active session. Click “+ New Session” or press “Run Parser”.
      </div>`;
    }

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') flushBatch();
    });
  }

  document.addEventListener('DOMContentLoaded', init);
})();