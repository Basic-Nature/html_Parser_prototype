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
  const typeCounts = { all: 0 };
  let urlIndexMap = {};
  let contestIndexMap = {};
  // Contest options store (per session) + helpers
  const CONTEST_STORE_KEY = 'contest_opts_by_session_v1';
  let contestOptionsBySession = lsGetJSON(CONTEST_STORE_KEY, {});
  let lastContestOptions = contestOptionsBySession[activeSessionId] || []; // [{index,label,meta}]
  function getContestOptions(sessionId = activeSessionId) {
    return (contestOptionsBySession[sessionId] || []).slice();
  }
  function setContestOptions(sessionId, opts) {
    contestOptionsBySession[sessionId] = (opts || []).slice();
    lsSetJSON(CONTEST_STORE_KEY, contestOptionsBySession);
    if (sessionId === activeSessionId) lastContestOptions = contestOptionsBySession[sessionId];
  }
  // Handy globals for console/UI hooks
  window.getLastContestOptions = () => getContestOptions();
  window.showContestPicker = function(sessionId = activeSessionId) {
    const opts = getContestOptions(sessionId);
    if (!opts.length) return false;
    showIndexedSelectionModal('Select Contest', opts, (selection) => {
      if (!selection) return;
      if (socket) socket.emit('parser_prompt', { session_id: sessionId, value: selection.join(',') });
    });
    return true;
  };
  let lastPromptContext = null;
  let lastSentSourceBySession = {};
  let sessionMetaIndex = {};

  // Folder browser state
  const ROOT_LABELS = { uploads: 'Uploads', input: 'Input', output: 'Output' };

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
      if (!el.runBtn.getAttribute('data-running')) {
        el.runBtn.disabled = false;
      }
      el.runBtn.classList.remove('btn-locked');
      if (!locked) el.runBtn.removeAttribute('data-running');
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

  // Modal helper (single reusable modal shell)
  const Modal = {
    get() {
      let modal = document.getElementById('downloadModal');
      // Auto-create a minimal modal if missing (fallback)
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
                <h5 class="modal-title" id="downloadModalTitle">Select</h5>
                <button type="button" class="btn-close" id="closeDownloadModal" aria-label="Close"></button>
              </div>
              <div class="modal-body">
                <input type="search" id="downloadSearch" class="form-control mb-2" placeholder="Filter...">
                <div id="downloadSummary" class="mb-2"></div>
                <div id="downloadOptions"></div>
              </div>
              <div class="modal-footer">
                <button type="button" class="btn btn-secondary" id="cancelDownloadModal">Cancel</button>
              </div>
            </div>
          </div>`;
        document.body.appendChild(modal);
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
      const inst = window.bootstrap?.Modal.getOrCreateInstance(modal, { keyboard: true, backdrop: true });
      inst.show();
    },
    close() {
      const { modal } = this.get();
      const inst = window.bootstrap?.Modal.getOrCreateInstance(modal);
      inst?.hide();
    }
  };

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
  function showIndexedSelectionModal(title, options, onSelect) {
    const refs = Modal.get();
    if (!refs) return onSelect(null);
    const { titleEl, searchEl, optionsDiv, summaryDiv, closeBtn, cancelBtn } = refs;

    titleEl.textContent = title || 'Select';
    summaryDiv.textContent = `${options.length} option(s)`;

    function renderList(q = '') {
      const query = (q || '').toLowerCase().trim();
      const filtered = !query
        ? options
        : options.filter(o =>
            String(o.index).includes(query) ||
            o.label.toLowerCase().includes(query) ||
            (o.meta || '').toLowerCase().includes(query)
          );

      optionsDiv.innerHTML = '';
      filtered.forEach(o => {
        const item = document.createElement('div');
        item.className = 'download-option';
        item.tabIndex = 0;
        item.innerHTML = `<b>[${o.index}]</b> ${esc(o.label)}${o.meta ? ` <small>(${esc(o.meta)})</small>` : ''}`;
        item.onclick = () => { hide(); onSelect([o.index]); };
        item.onkeydown = (e) => { if (e.key === 'Enter') { hide(); onSelect([o.index]); } };
        optionsDiv.appendChild(item);
      });
    }

    function hide() { Modal.close(); }
    searchEl.value = '';
    renderList('');
    searchEl.oninput = e => renderList(e.target.value);
    closeBtn.onclick = cancelBtn.onclick = () => { hide(); onSelect(null); };
    Modal.open();
    searchEl.focus();
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
          row.tabIndex = 0;
          row.setAttribute('role','button');

          const name = document.createElement('div');
          name.className = 'item-name';
          const icon = ent.type === 'dir' ? '📁' : '📄';
          name.title = ent.name;
          name.innerText = `${icon} ${ent.name}`;

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
              if (socket && activeSessionId) {
                socket.emit('parser_prompt', { session_id: activeSessionId, value: rel });
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
            // Clicking the row uses the file (quick action)
            row.addEventListener('click', () => {
              const rel = joinPath(cwd, ent.name);
              if (socket && activeSessionId) {
                socket.emit('parser_prompt', { session_id: activeSessionId, value: rel });
              }
            });
          } else {
            // Directory: click to open
            row.addEventListener('click', async () => {
              cwd = joinPath(cwd, ent.name);
              await refresh();
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

          row.appendChild(name);
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
  function showFolderBrowser(root, initialPath = '', onSelect) {
    const refs = Modal.get();
    if (!refs) return onSelect(null);
    const { titleEl, searchEl, optionsDiv, summaryDiv, closeBtn, cancelBtn } = refs;

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
            hide();
            onSelect({ root, path: cwd, name: ent.name });
            if (socket && activeSessionId) {
              socket.emit('parser_prompt', { session_id: activeSessionId, value: rel });
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
            hide();
            onSelect({ root, path: cwd, name: ent.name });
            if (socket && activeSessionId) {
              socket.emit('parser_prompt', { session_id: activeSessionId, value: rel });
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

    function hide() { Modal.close(); }

    searchEl.value = '';
    searchEl.oninput = e => renderList(e.target.value);
    closeBtn.onclick = cancelBtn.onclick = () => { hide(); onSelect(null); };

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
              if (socket && activeSessionId) {
                socket.emit('parser_prompt', { session_id: activeSessionId, value: rel });
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
    // sync lastContestOptions for this session
    lastContestOptions = getContestOptions(activeSessionId);
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
    return (el.fileSourceSelect && el.fileSourceSelect.value === 'uploads') ? 'uploads' : 'input';
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
    emitManualFileSource();
    animateButton(el.runBtn);
    el.runBtn.disabled = true;
    el.runBtn.setAttribute('data-running','true');
    el.runBtn.textContent = 'Running...';
    socket.once('joined', function(data) {
      if (data.session_id === activeSessionId) {
        socket.emit('run_parser', { session_id: activeSessionId, file_source: currentFileSource() });
        setTimeout(() => socket && socket.emit('get_session_history', { session_id: activeSessionId }), 600);
      }
    });
    joinSession(activeSessionId);
    setTimeout(() => { if (!el.runBtn.getAttribute('data-running')) el.runBtn.disabled = false; }, 4000);
  }

  // -------- Prompt --------
  function handlePromptSubmit(e) {
    e.preventDefault();
    if (!socket || !el.promptInput || !activeSessionId) return;
    let raw = el.promptInput.value.trim();

    // Quick command: reopen last contest picker
    if (/^\/?contests$/i.test(raw)) {
      if (!window.showContestPicker()) alert('No cached contest options for this session yet.');
      el.promptInput.value = '';
      return;
    }
    // Expand comma-separated numbers for contests (or URLs if present)
    if (/^\d+(?:\s*,\s*\d+)*$/.test(raw)) {
      const nums = raw.split(/\s*,\s*/);
      if (lastPromptContext && lastPromptContext.kind === 'contest' && Object.keys(contestIndexMap).length) {
        raw = nums.filter(n => contestIndexMap[n] != null).join(',');
      } else {
        raw = nums.map(n => urlIndexMap[n] || n).join(',');
      }
    } else {
      const m = raw.match(/^\[?(\d+)\]?$/);
      if (m) {
        const n = m[1];
        if (lastPromptContext && lastPromptContext.kind === 'contest' && contestIndexMap[n] != null) {
          raw = n;
        } else if (urlIndexMap[n]) {
          raw = urlIndexMap[n];
        }
      }
    }

    socket.emit('parser_prompt', { session_id: activeSessionId, value: raw });
    el.promptInput.value = '';
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
          if (window.socket && window.getActiveSessionId) {
            socket.emit('parser_prompt', { session_id: getActiveSessionId(), value: url });
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
    socket.on('session_deleted', handleSessionDeleted);
    socket.on('session_heartbeat', handleSessionHeartbeat);

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
      }
    }

    function handleParserOutput(d) {
      if (!activeSessionId) {
        earlyQueue.push(d);
        return;
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
        const options = parseIndexedMenu(d.message);
        if (options.length) {
          contestIndexMap = Object.fromEntries(options.map(o => [String(o.index), o.label]));
          setContestOptions(d.session_id, options);
          lastPromptContext = { kind: 'contest', options: options.map(o => `[${o.index}] ${o.label}`), session_id: d.session_id };
          showIndexedSelectionModal('Select Contest', options, (selection) => {
            if (!selection) return;
            const val = selection.join(',');
            socket && socket.emit('parser_prompt', { session_id: d.session_id, value: val });
          });
        }
      }

      // Prompt handling
      if (d && d.type === 'prompt' && d.session_id === activeSessionId) {
        const ctx = d.context || {};
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
            socket.emit('parser_prompt', {
              session_id: activeSessionId,
              value: selectedIdx == null ? 'n' : String(selectedIdx)
            });
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
            socket.emit('parser_prompt', {
              session_id: activeSessionId,
              value: selectedIdx == null ? 'n' : String(selectedIdx)
            });
          });
        } else {
          showPromptModal(d.message, function(userInput) {
            socket.emit('parser_prompt', { session_id: activeSessionId, value: userInput });
          });
        }
      }

      // Normal log handling
      renderParserOutput(d);
      if (d && d.session_id) appendCacheLog(d.session_id, d);

      // Status: completed logic
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
    }

    function handleManualSourceState({ session_id, file_source }) {
      if (session_id === activeSessionId && el.fileSourceSelect) {
        el.fileSourceSelect.value = file_source;
        syncSourceClass();
      }
    }

    function handleSessionList(data) {
      if (!Array.isArray(data.sessions)) return;
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

    function handleSessionDeleted({ session_id }) {
      const filtered = getSessions().filter(s => s !== session_id);
      lsSetJSON('active_sessions', filtered);
      if (activeSessionId === session_id) setActiveSession(filtered[0] || '');
      renderSessionList();
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
      el.promptInput.onkeydown = function(e) {
        if (e.key === 'Enter') {
          e.preventDefault();
          const val = el.promptInput.value.trim();
          el.promptInput.value = '';
          el.promptInput.disabled = true;
          el.promptInput.parentElement?.classList.add('hidden');
          callback(val);
        }
        if (e.key === 'Escape') {
          el.promptInput.value = '';
          el.promptInput.disabled = true;
          el.promptInput.parentElement?.classList.add('hidden');
          callback('');
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
    let lastFocus = null;
    let untrap = null;

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

    // Mobile drawer behavior (off-canvas)
    if (aside && drawerBtn && backdrop) {
      const open = () => {
        lastFocus = document.activeElement;
        document.body.classList.add('sidebar-open');
        drawerBtn.setAttribute('aria-expanded', 'true');
        setInert(true);
        untrap = trapFocus(aside);
        setTimeout(() => aside.querySelector('.url-search-box')?.focus(), 0);
      };
      const close = () => {
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