document.addEventListener('DOMContentLoaded', () => {
  // Prefer server-provided URL, fallback to default
  const apiUrl = (window.__DATA_FRAMEWORK__ && window.__DATA_FRAMEWORK__.apiUrl) || '/api/warehouse_election_results';

  // Elements
  const theadRow = document.getElementById('table-header');
  const tbody = document.getElementById('table-body');
  const statusEl = document.getElementById('tableStatus');

  const searchInput = document.getElementById('globalSearch');
  const pageSizeSelect = document.getElementById('pageSizeSelect');
  const firstBtn = document.getElementById('firstPageBtn');
  const prevBtn = document.getElementById('prevPageBtn');
  const nextBtn = document.getElementById('nextPageBtn');
  const lastBtn = document.getElementById('lastPageBtn');
  const pageInfo = document.getElementById('pageInfo');

  const exportBtn = document.getElementById('exportCsvBtn');
  const resetBtn = document.getElementById('resetFiltersBtn');
  const refreshBtn = document.getElementById('refreshBtn');
  const colBtn = document.getElementById('columnChooserBtn');
  const colMenu = document.getElementById('columnChooserMenu');
  const colWrap = colBtn?.parentElement;

  // Upload form (moved from inline script)
  const uploadStatus = document.getElementById('uploadStatus');
  const uploadForm = document.getElementById('uploadForm');

  // State
  let rawData = [];
  let visibleColumns = [];
  let sortBy = null; // column key
  let sortDir = 'none'; // 'ascending' | 'descending' | 'none'
  let searchTerm = '';
  let page = 1;
  let pageSize = parseInt(pageSizeSelect?.value || '25', 10);

  // Helpers
  function setStatus(el, type, text) {
    if (!el) return;
    el.classList.remove('status-info', 'status-ok', 'status-error');
    el.classList.add('status', type === 'ok' ? 'status-ok' : type === 'error' ? 'status-error' : 'status-info');
    el.textContent = text || '';
  }
  function safeGet(v) { return v == null ? '' : String(v); }
  function debounce(fn, ms) {
    let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  // Upload
  if (uploadForm) {
    uploadForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const formData = new FormData(uploadForm);
      setStatus(uploadStatus, 'info', 'Uploading...');
      fetch('/api/upload_csv', { method: 'POST', body: formData })
        .then(r => r.json().catch(() => ({ success:false, error:'Upload endpoint did not return JSON' })))
        .then(data => {
          if (data.success) {
            setStatus(uploadStatus, 'ok', 'Upload successful!');
            fetchData();
          } else {
            setStatus(uploadStatus, 'error', `Upload failed: ${data.error || 'Unknown error'}`);
          }
        })
        .catch(err => setStatus(uploadStatus, 'error', `Upload failed: ${err}`));
    });
  }

  // Build columns and header
  function buildColumns() {
    const keys = rawData.length ? Object.keys(rawData[0]) : [];
    if (!visibleColumns.length) visibleColumns = [...keys];
    buildHeader(keys);
    buildColumnMenu(keys);
  }

  function buildHeader(keys) {
    theadRow.innerHTML = '';
    keys.forEach(key => {
      if (!visibleColumns.includes(key)) return;
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = key.charAt(0).toUpperCase() + key.slice(1);
      th.classList.add('sortable');
      th.setAttribute('role', 'columnheader');
      th.setAttribute('tabindex', '0');
      th.dataset.field = key;
      th.setAttribute('aria-sort', sortBy === key ? sortDir : 'none');

      const ind = document.createElement('span');
      ind.className = 'sort-indicator';
      th.appendChild(ind);

      const toggleSort = () => {
        if (sortBy !== key) {
          sortBy = key; sortDir = 'ascending';
        } else {
          sortDir = sortDir === 'ascending' ? 'descending' : sortDir === 'descending' ? 'none' : 'ascending';
          if (sortDir === 'none') sortBy = null;
        }
        [...theadRow.children].forEach(th2 =>
          th2.setAttribute('aria-sort', th2.dataset.field === sortBy ? sortDir : 'none')
        );
        render();
      };
      th.addEventListener('click', toggleSort);
      th.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleSort(); } });

      theadRow.appendChild(th);
    });
  }

  function buildColumnMenu(keys) {
    if (!colMenu) return;
    colMenu.innerHTML = '';
    keys.forEach(key => {
      const label = document.createElement('label');
      label.setAttribute('role', 'menuitemcheckbox');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = visibleColumns.includes(key);
      cb.addEventListener('change', () => {
        if (cb.checked) { if (!visibleColumns.includes(key)) visibleColumns.push(key); }
        else { visibleColumns = visibleColumns.filter(k => k !== key); }
        buildHeader(keys);
        render();
      });
      const text = document.createElement('span');
      text.textContent = key;
      label.appendChild(cb); label.appendChild(text);
      colMenu.appendChild(label);
    });
  }

  function getFilteredSorted() {
    let data = [...rawData];

    // Search
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      data = data.filter(row =>
        visibleColumns.some(col => safeGet(row[col]).toLowerCase().includes(q))
      );
    }

    // Sort
    if (sortBy) {
      data.sort((a, b) => {
        const av = safeGet(a[sortBy]);
        const bv = safeGet(b[sortBy]);
        if (!isNaN(av) && !isNaN(bv)) {
          return (Number(av) - Number(bv)) * (sortDir === 'descending' ? -1 : 1);
        }
        return av.localeCompare(bv, undefined, { numeric: true, sensitivity: 'base' }) *
               (sortDir === 'descending' ? -1 : 1);
      });
    }
    return data;
  }

  function renderSkeleton(rows = 6) {
    tbody.innerHTML = '';
    for (let i = 0; i < rows; i++) {
      const tr = document.createElement('tr');
      tr.className = 'skeleton';
      visibleColumns.forEach(() => tr.appendChild(document.createElement('td')));
      tbody.appendChild(tr);
    }
  }

  function render() {
    const filtered = getFilteredSorted();

    // Pagination
    const total = filtered.length;
    const pages = Math.max(1, Math.ceil(total / pageSize));
    page = Math.min(Math.max(1, page), pages);
    const start = (page - 1) * pageSize;
    const slice = filtered.slice(start, start + pageSize);

    // Body
    tbody.innerHTML = '';
    if (!slice.length) {
      const tr = document.createElement('tr');
      const td = document.createElement('td');
      td.colSpan = Math.max(visibleColumns.length, 1);
      td.textContent = rawData.length ? 'No results match the current filters.' : 'No data to display.';
      tr.appendChild(td);
      tbody.appendChild(tr);
    } else {
      slice.forEach(row => {
        const tr = document.createElement('tr');
        visibleColumns.forEach(col => {
          const td = document.createElement('td');
          td.textContent = safeGet(row[col]);
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
    }

    // Pagination UI
    pageInfo.textContent = `Page ${page} of ${pages} • ${total} rows`;
    firstBtn.disabled = page <= 1;
    prevBtn.disabled = page <= 1;
    nextBtn.disabled = page >= pages;
    lastBtn.disabled = page >= pages;

    setStatus(statusEl, slice.length ? 'info' : (rawData.length ? 'info' : 'error'), slice.length ? '' : (rawData.length ? 'No results match the current filters.' : ''));
  }

  // Events
  searchInput?.addEventListener('input', debounce((e) => { searchTerm = e.target.value.trim(); page = 1; render(); }, 150));
  pageSizeSelect?.addEventListener('change', (e) => { pageSize = parseInt(e.target.value, 10) || 25; page = 1; render(); });
  firstBtn?.addEventListener('click', () => { page = 1; render(); });
  prevBtn?.addEventListener('click', () => { page = Math.max(1, page - 1); render(); });
  nextBtn?.addEventListener('click', () => { page = page + 1; render(); });
  lastBtn?.addEventListener('click', () => {
    const total = getFilteredSorted().length;
    page = Math.max(1, Math.ceil(total / pageSize));
    render();
  });
  refreshBtn?.addEventListener('click', () => fetchData());

  exportBtn?.addEventListener('click', () => {
    const data = getFilteredSorted();
    const cols = visibleColumns;
    const csv = [
      cols.join(','),
      ...data.map(r => cols.map(c => {
        const v = safeGet(r[c]).replace(/"/g, '""');
        return /[",\n]/.test(v) ? `"${v}"` : v;
      }).join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `export_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  });

  resetBtn?.addEventListener('click', () => {
    if (searchInput) searchInput.value = '';
    searchTerm = ''; sortBy = null; sortDir = 'none'; page = 1;
    visibleColumns = []; // reset to all on rebuild
    buildColumns(); render();
  });

  // Column chooser
  colBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    const expanded = colBtn.getAttribute('aria-expanded') === 'true';
    colBtn.setAttribute('aria-expanded', String(!expanded));
    colWrap.classList.toggle('open', !expanded);
  });
  document.addEventListener('click', (e) => {
    if (!colWrap) return;
    if (!colWrap.contains(e.target)) {
      colWrap.classList.remove('open');
      colBtn?.setAttribute('aria-expanded', 'false');
    }
  });

  // Robust fetch with HTML fallback detection
  function fetchData() {
    setStatus(statusEl, 'info', 'Loading data...');
    renderSkeleton();
    fetch(apiUrl, { headers: { 'Accept': 'application/json' } })
      .then(async (r) => {
        const ct = (r.headers.get('content-type') || '').toLowerCase();
        if (!r.ok) {
          const text = await r.text().catch(() => '');
          // If HTML, show concise hint
          if (text.trim().startsWith('<')) {
            throw new Error(`Server error ${r.status}. Received HTML (likely an error page).`);
          }
          throw new Error(`Server error ${r.status}: ${text.slice(0,200)}`);
        }
        // Prefer JSON, but detect HTML
        if (ct.includes('application/json')) {
          try { return await r.json(); }
          catch (e) {
            const text = await r.text().catch(() => '');
            if (text.trim().startsWith('<')) {
              throw new Error(`Invalid JSON: received HTML content.`);
            }
            throw new Error(`Invalid JSON from API: ${e.message}`);
          }
        } else {
          const text = await r.text().catch(() => '');
          if (text.trim().startsWith('<')) {
            throw new Error(`Unexpected HTML response from API.`);
          }
          throw new Error(`Unexpected content-type: ${ct || 'unknown'}`);
        }
      })
      .then(data => {
        // Accept array, {rows: [...]}, or {items: [...]}
        rawData = Array.isArray(data)
          ? data
          : (Array.isArray(data?.rows) ? data.rows
          : (Array.isArray(data?.items) ? data.items : []));
        if (!rawData.length) {
          setStatus(statusEl, 'error', 'No data found in the database.');
        } else {
          setStatus(statusEl, 'ok', `Loaded ${rawData.length} rows.`);
        }
        buildColumns(); render();
      })
      .catch(err => {
        rawData = [];
        buildColumns(); render();
        // Normalize common JSON parse error shown earlier
        const msg = (err && err.message) ? err.message : String(err);
        if (/Unexpected token .*<!doctype/i.test(msg) || /received html/i.test(msg)) {
          setStatus(statusEl, 'error', 'Error loading data: API returned HTML instead of JSON. Check server logs or endpoint URL.');
        } else {
          setStatus(statusEl, 'error', `Error loading data: ${msg}`);
        }
      });
  }

  // Initial load
  fetchData();
});