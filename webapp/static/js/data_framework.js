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
  if (window.bootstrap) {
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
      .forEach(el => bootstrap.Tooltip.getOrCreateInstance(el));
    document.querySelectorAll('[data-bs-toggle="popover"]')
      .forEach(el => bootstrap.Popover.getOrCreateInstance(el));
  }

  // ---------- Config hydration ----------
  const cfgEl = document.getElementById('dataFrameworkConfig');
  const hydratedUrl = cfgEl?.dataset.apiUrl;
  const apiUrl =
    hydratedUrl ||  // server now injects absolute path via url_for
    (window.__DATA_FRAMEWORK__ && window.__DATA_FRAMEWORK__.apiUrl) ||
    '/api/warehouse_election_results';

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
    resetFilters: document.getElementById('resetFiltersBtn'),
    refresh: document.getElementById('refreshBtn'),
    colBtn: document.getElementById('columnChooserBtn'),
    colMenu: document.getElementById('columnChooserMenu'),
    copyVisibleCsv: document.getElementById('copyVisibleCsv'),
    uploadForm: document.getElementById('uploadForm'),
    uploadStatus: document.getElementById('uploadStatus')
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
  let pageSize = parseInt(el.pageSize?.value || '25', 10);

  // ---------- Constants / Policies ----------
  const COL_NAME_RX = /^[A-Za-z0-9_]{1,64}$/;
  const MAX_SEARCH_LEN = 1200;
  const MAX_VISIBLE_COLS = 200;
  const MAX_ROWS_EXPORT = 200000; // safeguard client memory for CSV
  const SKELETON_ROWS = 6;

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

  // ---------- Upload handling ----------
  if (el.uploadForm) {
    el.uploadForm.addEventListener('submit', e => {
      e.preventDefault();
      const fd = new FormData(el.uploadForm);

      // Optional: basic filename policy (client hint)
      const file = fd.get('csv_file');
      if (file && file.name && !/\.csv$/i.test(file.name)) {
        setStatus(el.uploadStatus, 'error', 'Only .csv files are allowed.');
        return;
      }

      setStatus(el.uploadStatus, 'info', 'Uploading...');
      fetch('/api/upload_csv', { method: 'POST', body: fd })
        .then(r => r.json().catch(() => ({ success: false, error: 'Upload endpoint did not return JSON' })))
        .then(data => {
          if (data.success) {
            setStatus(el.uploadStatus, 'ok', 'Upload successful!');
            showInfoToast('Upload successful.');
            fetchData(true);
          } else {
            setStatus(el.uploadStatus, 'error', `Upload failed: ${data.error || 'Unknown error'}`);
            showErrorToast('Upload failed.');
          }
        })
        .catch(err => {
          setStatus(el.uploadStatus, 'error', `Upload failed: ${err}`);
          showErrorToast('Upload failed.');
        });
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
    el.theadRow.innerHTML = '';
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
        [...el.theadRow.children].forEach(h =>
          h.setAttribute('aria-sort', h.dataset.field === sortBy ? sortDir : 'none'));
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
    el.colMenu.innerHTML = '';
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
        const bothNum = av !== '' && bv !== '' && !isNaN(av) && !isNaN(bv);
        const cmp = bothNum
          ? (Number(av) - Number(bv))
          : av.localeCompare(bv, undefined, { numeric: true, sensitivity: 'base' });
        return cmp * dirMul;
      });
    }
    return data;
  }

  // ---------- Rendering ----------
  function renderSkeleton(rows = SKELETON_ROWS) {
    el.tbody.innerHTML = '';
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

    el.tbody.innerHTML = '';
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
    el.first.disabled = page <= 1;
    el.prev.disabled = page <= 1;
    el.next.disabled = page >= pages;
    el.last.disabled = page >= pages;

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
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
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
    searchTerm = sanitizeSearch(e.target.value);
    page = 1;
    render();
  }, 150));

  el.pageSize?.addEventListener('change', e => {
    pageSize = Math.max(1, parseInt(e.target.value, 10) || 25);
    page = 1;
    render();
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
    if (el.search) el.search.value = '';
    sortBy = null;
    sortDir = 'none';
    page = 1;
    render();
    showInfoToast('Filters reset.');
  });

  el.exportCsv?.addEventListener('click', exportCsv);
  el.copyVisibleCsv?.addEventListener('click', copyVisibleCsv);

  el.colBtn?.addEventListener('click', e => {
    e.stopPropagation();
    const expanded = el.colBtn.getAttribute('aria-expanded') === 'true';
    el.colBtn.setAttribute('aria-expanded', String(!expanded));
    colWrap.classList.toggle('open', !expanded);
  });

  document.addEventListener('click', ev => {
    if (!colWrap) return;
    if (!colWrap.contains(ev.target)) {
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
  fetchData(true);
});  