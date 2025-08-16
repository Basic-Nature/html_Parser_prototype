/* run_parser.js
   - Early session room join + queuing (joinedSessions/earlyQueue)
   - Auto history fetch & race protection
   - Prompt auto-response (optional) + numeric index expansion
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

  // Auto prompt (respond "all" to selection prompt if user does nothing)
  const AUTO_PROMPT_ENABLED = true;
  const AUTO_PROMPT_PATTERN = /enter indices.*(all)/i;
  const AUTO_PROMPT_RESPONSE = 'all';

  // -------- Utilities --------
  const $  = sel => document.querySelector(sel);
  const $$ = sel => Array.from(document.querySelectorAll(sel));
  const on = (el, ev, fn, opts) => el && el.addEventListener(ev, fn, opts);
  const lsGetJSON = (k, fb) => { try { return JSON.parse(localStorage.getItem(k) || 'null') ?? fb; } catch { return fb; } };
  const lsSetJSON = (k, v) => localStorage.setItem(k, JSON.stringify(v));
  const uniq = arr => Array.from(new Set(arr));
  const nowPerf = () => performance.now();
  const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

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
    if (!ENABLE_META_STRIP || typeof rawStr !== 'string') return rawStr;
    if (!/(^|\n)\s*(Level:|Type:|Session:)/i.test(rawStr)) return rawStr;
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
  const seenLogTypes = new Set(['all']);
  const typeCounts = { all: 0 }; 
  let urlIndexMap = {};
  let lastSentSourceBySession = {};
  let sessionMetaIndex = {};
  // Keep Run button in sync with backend session lock state
  function updateRunButtonLock() {
    if (!el.runBtn) return;
    const meta = sessionMetaIndex[activeSessionId];
    const locked = meta && meta.locked;
    if (locked) {
      el.runBtn.disabled = true;
      el.runBtn.classList.add('btn-locked');
      el.runBtn.setAttribute('data-running','true');
    } else {
      // Only re-enable if not currently marked running by a fresh click
      if (!el.runBtn.getAttribute('data-running')) {
        el.runBtn.disabled = false;
      }
      el.runBtn.classList.remove('btn-locked');
      if (!locked) el.runBtn.removeAttribute('data-running');
    }
  }
  // New: track joined rooms & early log queue
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
    deletePresetBtn: $('#deletePresetBtn')
  };

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
    }
  }
  el.outputModeSelect?.setAttribute('aria-label','Select output delivery mode');
  el.logFilterSelect?.setAttribute('aria-label','Filter logs by level');

  // -------- Sessions --------
  function getSessions() {
    const list = lsGetJSON('active_sessions', []);
    const normalized = uniq(list.map(s => (s && typeof s === 'object' && s.session_id) ? s.session_id : s).filter(Boolean));
    lsSetJSON('active_sessions', normalized);
    return normalized;
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
    // Flush any early queued logs
    if (earlyQueue.length && el.outputDiv) {
      earlyQueue.forEach(d => renderParserOutput(d));
      flushBatch();
      earlyQueue = [];
    }
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
        restoreCachedLogs(sid); // show cached immediately
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
        const key = providedType.toLowerCase().replace(/[\s\-]/g,'_');
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
    // Fallback to legacy heuristic if still other
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
    SUCCESS:'✅', CANCELLED:'🛑', CANCEL:'🛑', EXCEPTION:'⛔',
    PROGRESS:'⏳', PROMPT:'💬', SUMMARY:'🧾', FATAL:'💥'
  };
  const levelColorMap = {
    INFO:'#00ffe7', DEBUG:'#8ecae6', WARNING:'#ffd166', ERROR:'#eb4f43', CRITICAL:'#ff006e',
    SUCCESS:'#74c69d', CANCELLED:'#eb4f43', CANCEL:'#eb4f43', EXCEPTION:'#ff7b00',
    PROGRESS:'#5fa8d3', PROMPT:'#d4af37', SUMMARY:'#9d4edd', FATAL:'#ff2e2e', OTHER:'#ffffff'
  };
  const typeColorMap = {
    status:'#86efac',
    input:'#38bdf8',
    output:'#c084fc',
    manual_override:'#fb923c',
    ai_analysis:'#f472b6',
    stream:'#34d399',
    router:'#fde047',
    validation:'#fcd34d',
    cancel:'#f87171',
    heartbeat:'#60a5fa',
    summary:'#a78bfa',
    cache:'#a3a3a3',
    handler:'#f9a8d4',
    batch:'#67e8f9',
    download:'#93c5fd',
    browser:'#d8b4fe',
    exception:'#ff7b00',
    error:'#eb4f43',
    fatal:'#ff006e',
    other:'#ffffff',
    prompt:'#d4af37'
  };
  function hashColor(seed) {
    let h = 0;
    for (let i=0;i<seed.length;i++) h = (h * 131 + seed.charCodeAt(i)) >>> 0;
    const hue = h % 360;
    const sat = 55 + (h >> 3) % 25;
    const light = 45 + (h >> 6) % 20;
    return `hsl(${hue} ${sat}% ${light}%)`;
  }

  const dynamicLevels = new Set(['all']);

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
    // update option label if it exists
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

  function maybeAutoRespondPrompt(obj) {
    if (!AUTO_PROMPT_ENABLED) return;
    if (obj.type === 'prompt' && AUTO_PROMPT_PATTERN.test(obj.full_text || '')) {
      setTimeout(() => {
        if (!el.promptInput || el.promptInput.value.trim()) return;
        socket.emit('parser_prompt', { session_id: activeSessionId, value: AUTO_PROMPT_RESPONSE });
      }, 500);
    }
  }

  function renderParserOutput(raw) {
    if (!el.outputDiv) return;
    // If empty-URL condition appears, inject hint once
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
    maybeAutoRespondPrompt(obj);

    if (obj.type === 'heartbeat' && !SHOW_HEARTBEAT_LINES) return;

    if (obj.type === 'cancel' && (obj.level === 'DEBUG' || obj.level === 'INFO')) {
      if (!levelIconMap[obj.level]) levelIconMap[obj.level] = '🛑';
    }

    if (!levelIconMap[obj.level]) {
      levelIconMap[obj.level] = '🧩';
      levelColorMap[obj.level] = hashColor(obj.level);
      ensureFilterOptions(el.logFilterSelect, dynamicLevels, obj.level.toLowerCase());
    }
    if (!typeColorMap[obj.type]) typeColorMap[obj.type] = hashColor(obj.type);

    ensureFilterOptions(el.logFilterSelect, dynamicLevels, obj.level.toLowerCase());
    const normType = obj.type.toLowerCase();
    ensureFilterOptions(logTypeSelect, seenLogTypes, normType);
    bumpTypeCount(normType);

    const sig = ['v5', obj.level, obj.type, obj.session_id || '', obj.full_text].join('|');
    if (shouldSuppressDuplicate(sig)) return;

    let timeStr = '';
    if (obj.timestamp && !isNaN(obj.timestamp)) {
      const d = new Date(obj.timestamp);
      if (!isNaN(d.getTime())) timeStr = d.toLocaleTimeString();
    }

    if (obj.type === 'prompt' && el.promptInput)
      el.promptInput.placeholder = obj.full_text || 'Type a command...';

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
  // Wire preset functions to UI
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
    return (el.fileSourceSelect && el.fileSourceSelect.value === 'uploads') ? 'uploads' : 'input';
  }
  function syncSourceClass() {
    document.body.classList.toggle('source-uploads', currentFileSource() === 'uploads');
  }
  function emitManualFileSource() {
    if (!socket || !activeSessionId) return;
    const src = currentFileSource();
    if (lastSentSourceBySession[activeSessionId] === src) return;
    lastSentSourceBySession[activeSessionId] = src;
    socket.emit('set_manual_source', { session_id: activeSessionId, file_source: src });
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
    // Prefer existing last stored session before auto-creating
    const stored = localStorage.getItem('session_id');
    if (!activeSessionId && stored) {
      setActiveSession(stored);
      joinSession(stored);
    }
    if (!activeSessionId) {
      addNewSession();
    } else {
      joinSession(activeSessionId);
    }
    emitManualFileSource();
    clearOutput();
    animateButton(el.runBtn);
    el.runBtn.disabled = true;
    el.runBtn.setAttribute('data-running','true');
    el.runBtn.textContent = 'Running...';
    socket.emit('run_parser', { session_id: activeSessionId, file_source: currentFileSource() });
    setTimeout(() => socket && socket.emit('get_session_history', { session_id: activeSessionId }), 600);
    // Re-enable only after a session_list shows unlocked
    setTimeout(() => { if (!el.runBtn.getAttribute('data-running')) el.runBtn.disabled = false; }, 4000);
  }

  // -------- Prompt --------
  function handlePromptSubmit(e) {
    e.preventDefault();
    if (!socket || !el.promptInput || !activeSessionId) return;
    let raw = el.promptInput.value.trim(); // allow empty (backend may treat as cancel)
    if (/^\d+(,\s*\d+)*$/.test(raw)) {
      raw = raw.split(/,\s*/).map(n => urlIndexMap[n] || n).join(',');
    } else {
      const mSingle = raw.match(/^\[?(\d+)\]?$/);
      if (mSingle && urlIndexMap[mSingle[1]]) raw = urlIndexMap[mSingle[1]];
    }
    socket.emit('parser_prompt', { session_id: activeSessionId, value: raw });
    el.promptInput.value = '';
  }

  // -------- URLs --------

  function buildUrlSidebarBlock(urls) {
    let host = document.getElementById('urlSidebarBlock');
    if (!host) {
      host = document.createElement('div');
      host.id = 'urlSidebarBlock';
      host.className = 'section';
      host.innerHTML = '<h2>urls.txt</h2><div class="url-lines" id="urlLinesBox"></div>';
      const sb = document.querySelector('.sidebar');
      if (sb) sb.prepend(host);
    }
    const box = host.querySelector('#urlLinesBox');
    if (!box) return;
    if (!urls.length) {
      box.innerHTML = '<span>(empty)</span>';
      return;
    }
    box.innerHTML = urls.map((u,i)=>`<span>[${i+1}] ${esc(u)}</span>`).join('');
  }

  function renderUrlList(urls) {
    if (!el.urlList) return;
    urlIndexMap = {};
    if (!urls.length) {
      el.urlList.innerHTML = `<ul class="url-list empty"><li>No URLs in urls.txt.</li></ul>`;
      return;
    }
    el.urlList.innerHTML = `<ul class="url-list">${
      urls.map((u,i)=> {
        const idx = i+1; urlIndexMap[idx] = u;
        return `<li data-url-idx="${idx}">[${idx}] ${esc(u)}</li>`;
      }).join('')
    }</ul>`;
  }

  function fetchUrls() {
    fetch('/api/urls')
      .then(r=>r.json())
      .then(d=>{
        const list = d.urls||[];
        renderUrlList(list);      // existing (center) list
        buildUrlSidebarBlock(list); // new sidebar block
      })
      .catch(()=>{
        renderUrlList([]);
        buildUrlSidebarBlock([]);
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
      reconnectionDelay: 800,
      transports: ['websocket'],          // force WS (avoid polling duplicates)
      pingInterval: 20000,                // MUST match server ping_interval (ms)
      pingTimeout: 60000                  // MUST match server ping_timeout (ms)
    });

    socket.on('connect', () => {
      hideDisconnectedMessage();
      joinedSessions.clear();
      renderSessionList();
      // Auto-join and fetch history for every remembered session
      const sessions = getSessions();
      sessions.forEach(s => {
        joinSession(s);
        socket.emit('get_session_history', { session_id: s });
      });
      // Prefer previously active
      const last = localStorage.getItem('session_id');
      if (last && sessions.includes(last)) {
        setActiveSession(last);
        restoreCachedLogs(last);
      } else if (sessions.length) {
        setActiveSession(sessions[0]);
        restoreCachedLogs(sessions[0]);
      }
    });
    socket.on('disconnect', showDisconnectedMessage);
    socket.on('connect_error', showDisconnectedMessage);

    socket.on('session_id', data => {
      const sid = typeof data === 'string' ? data : (data && data.session_id) || '';
      const sessions = getSessions();
      if (sid && !sessions.includes(sid)) { sessions.push(sid); lsSetJSON('active_sessions', sessions); }
      setActiveSession(sid);
      joinSession(sid);
      renderSessionList();
      socket.emit('get_session_history', { session_id: sid });
    });

    socket.on('session_history', data => {
      if (!data || !Array.isArray(data.logs)) return;
      clearOutput();
      data.logs.forEach(l => renderParserOutput(l));
      flushBatch();
    });

  socket.on('parser_output', d => {
      if (!activeSessionId) {
        earlyQueue.push(d);
        return;
      }
      renderParserOutput(d);
      if (d && d.session_id) appendCacheLog(d.session_id, d);
      if (d && d.session_id === activeSessionId && d.type === 'status') {
        if (/completed/i.test(d.message||'')) {
          if (el.runBtn) {
            el.runBtn.disabled = false;
            el.runBtn.removeAttribute('data-running');
            el.runBtn.textContent = 'Run Parser';
            updateRunButtonLock();
          }
        }
      }
  });

    socket.on('output_bypass_state', ({ output_bypass }) => applyBypassState(!!output_bypass));
    socket.on('manual_source_state', ({ session_id, file_source }) => {
      if (session_id === activeSessionId && el.fileSourceSelect) {
        el.fileSourceSelect.value = file_source;
        syncSourceClass();
      }
    });
    socket.on('session_list', data => {
      if (Array.isArray(data.sessions)) {
        const ids = data.sessions.map(s => (s && typeof s === 'object' && s.session_id) ? s.session_id : s);
        sessionMetaIndex = {};
        data.sessions.forEach(s => {
          if (s && s.session_id) sessionMetaIndex[s.session_id] = s;
        });
        lsSetJSON('active_sessions', ids);
        renderSessionList();
        if (!ids.includes(activeSessionId)) setActiveSession(ids[0] || '');
        updateRunButtonLock();
      }
    });
    socket.on('session_deleted', ({ session_id }) => {
      const filtered = getSessions().filter(s => s !== session_id);
      lsSetJSON('active_sessions', filtered);
      if (activeSessionId === session_id) setActiveSession(filtered[0] || '');
      renderSessionList();
    });

    socket.on('session_heartbeat', ({ session_id }) => {
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
    });

    socket.emit('get_sessions');
  }

  // -------- Init sub-blocks --------
  function initFileSource() {
    if (!el.fileSourceSelect) return;
    const qpSource = new URLSearchParams(location.search).get('source');
    if (qpSource && ['uploads','input'].includes(qpSource.toLowerCase()))
      el.fileSourceSelect.value = qpSource.toLowerCase();
    syncSourceClass();
    on(el.fileSourceSelect,'change', () => { syncSourceClass(); emitManualFileSource(); });
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
        panel && panel.classList.toggle('hidden');
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
  function initBootstrapUI() {
    if (!window.bootstrap) return;
    if (document.body.dataset.allowStyleAttr !== "1") return;
    $$('[data-bs-toggle="tooltip"]').forEach(n => window.bootstrap.Tooltip.getOrCreateInstance(n));
    $$('[data-bs-toggle="popover"]').forEach(n => window.bootstrap.Popover.getOrCreateInstance(n));
  }

  // -------- Init master --------
  function init() {
    initBootstrapUI();
    initFileSource();
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
    initFilterPresets();
    fetchUrls();
    connectSocket();

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