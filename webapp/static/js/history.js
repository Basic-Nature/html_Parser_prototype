/**
 * history.js
 * Client-side hardening against unvalidated URL redirection.
 * Defense-in-depth:
 *  - Strict allow‑list of internal route prefixes.
 *  - Canonicalization & repeated validation (pre & post decode).
 *  - Blocks protocol / scheme smuggling (javascript:, data:, etc.).
 *  - Neutralizes protocol-relative (//host) and mixed control chars.
 *  - Optional integrity check on click (re-validates live target).
 *  - Guards ?next= and any data-* navigation attributes.
 */
document.addEventListener('DOMContentLoaded', () => {
  const qs  = sel => document.querySelector(sel);
  const qsa = sel => Array.from(document.querySelectorAll(sel));

  // ---- Configurable allow-list (prefix-based) ----
  const INTERNAL_ROUTE_PREFIXES = Object.freeze([
    '/',               // root
    '/history',
    '/ballot_lens',
    '/data_framework',
    '/api/',
  ]);

  // Max length for any user-supplied redirect param
  const MAX_URL_LENGTH = 512;

  // ---- Canonicalize & validate helpers ----
  function stripControls(str) {
    return str.replace(/[\u0000-\u001F\u007F\s]+/g,'');
  }

  function safeDecode(raw) {
    try { return decodeURIComponent(raw); } catch { return raw; }
  }

  function canonicalize(raw) {
    if (!raw) return '';
    raw = raw.trim();
    if (raw.length > MAX_URL_LENGTH) raw = raw.slice(0, MAX_URL_LENGTH);
    raw = stripControls(raw);
    // Remove surrounding quotes if any
    if ((raw.startsWith('"') && raw.endsWith('"')) || (raw.startsWith("'") && raw.endsWith("'"))) {
      raw = raw.slice(1, -1);
    }
    // First decode pass
    raw = safeDecode(raw);
    // Collapse multiple slashes except protocol part
    if (!/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(raw)) {
      raw = raw.replace(/\/{2,}/g,'/'); // avoid path confusion (not touching leading // yet)
    }
    return raw;
  }

  function isDangerousScheme(str) {
    const lowered = str.toLowerCase();
    return /^(javascript:|data:|vbscript:|file:|mailto:|ws:|wss:)/.test(lowered);
  }

  function isSafeInternalUrl(raw) {
    if (!raw) return false;
    raw = canonicalize(raw);

    // Block protocol-relative //host
    if (raw.startsWith('//')) return false;

    // Absolute with scheme -> must be same-origin & safe scheme
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(raw)) {
      if (isDangerousScheme(raw)) return false;
      let u;
      try { u = new URL(raw); } catch { return false; }
      if (u.origin !== window.location.origin) return false;
      raw = u.pathname + u.search + u.hash;
    }

    // In-page anchor only
    if (raw.startsWith('#')) return true;

    // Must start with /
    if (!raw.startsWith('/')) return false;

    // Second decode pass (defend double-encoding attempts producing scheme)
    const twice = safeDecode(raw);
    if (isDangerousScheme(twice)) return false;
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(twice)) return false; // smuggled scheme after second decode
    // Enforce prefix allow-list
    return INTERNAL_ROUTE_PREFIXES.some(p => {
      if (p === '/') return true; // root covers all internal routes (keep if desired)
      const withSlash = p.endsWith('/') ? p : p + '/';
      return twice === p || twice.startsWith(withSlash);
    });
  }

  // Re-validate right before navigation (defensive)
  function guardedNavigate(targetRaw) {
    if (isSafeInternalUrl(targetRaw)) {
      const url = canonicalize(targetRaw);
      // Use assign to preserve history (change to replace if desired)
      window.location.assign(url);
    } else {
      showToast('toastError','Blocked unsafe navigation.');
    }
  }

  // ---- Query param handling (e.g., ?next=) ----
  const urlParams = new URLSearchParams(window.location.search);
  const nextParamRaw = urlParams.get('next');
  let SAFE_NEXT = null;
  if (nextParamRaw && isSafeInternalUrl(nextParamRaw)) {
    SAFE_NEXT = canonicalize(nextParamRaw);
  }
  // (No auto-redirect; developers can expose an explicit button referencing SAFE_NEXT.)

  // ---- Sanitize existing data-safe-nav / data-redirect anchors ----
  qsa('a[data-safe-nav], a[data-redirect]').forEach(a => {
    const raw = a.getAttribute('data-safe-nav') || a.getAttribute('data-redirect');
    if (!isSafeInternalUrl(raw)) {
      a.removeAttribute('data-safe-nav');
      a.removeAttribute('data-redirect');
      a.addEventListener('click', ev => {
        ev.preventDefault();
        showToast('toastError','Blocked unsafe navigation.');
      });
    } else {
      // Normalize to canonical path
      const canon = canonicalize(raw);
      if (a.hasAttribute('data-safe-nav')) a.setAttribute('data-safe-nav', canon);
      if (a.hasAttribute('data-redirect')) a.setAttribute('data-redirect', canon);
    }
  });

  // ---- External links: enforce rel noopener/noreferrer ----
  qsa('a[href]').forEach(a => {
    const href = a.getAttribute('href');
    if (!href) return;
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(href)) { // absolute
      try {
        const u = new URL(href, window.location.origin);
        if (u.origin !== window.location.origin) {
          // External
          const rel = (a.getAttribute('rel') || '').toLowerCase();
          const needed = ['noopener','noreferrer'];
          needed.forEach(flag => {
            if (!rel.includes(flag)) {
              a.setAttribute('rel', (rel ? rel + ' ' : '') + flag);
            }
          });
        }
      } catch { /* ignore */ }
    }
  });

  // ---- Bootstrap UI init ----
  qsa('[data-bs-toggle="tooltip"]').forEach(el => bootstrap.Tooltip.getOrCreateInstance(el));
  qsa('[data-bs-toggle="popover"]').forEach(el => bootstrap.Popover.getOrCreateInstance(el));

  // ---- Toast helper ----
  const showToast = (id, msg) => {
    const el = qs('#'+id);
    if (!el) return;
    const body = el.querySelector('.toast-body');
    if (body && typeof msg === 'string') body.textContent = msg;
    bootstrap.Toast.getOrCreateInstance(el).show();
  };

  // ---- Expand / Collapse all ----
  document.addEventListener('click', e => {
    const tgt = e.target;
    const btn = (tgt instanceof Element) ? tgt.closest('[data-action]') : null;
    if (!btn) return;
    const action = btn.getAttribute('data-action');
    if (action === 'expand-all' || action === 'collapse-all') {
      qsa('#runHistoryAccordion .accordion-collapse').forEach(coll => {
        const inst = bootstrap.Collapse.getOrCreateInstance(coll, { toggle:false });
        action === 'expand-all' ? inst.show() : inst.hide();
      });
    }
  }, { passive:true });

  // ---- Copy / Download JSON & guarded navigation ----
  function getRunJson(runId) {
    const pre = qs(`.run-json[data-run-id="${runId}"]`);
    return pre ? pre.textContent : '';
  }

  document.addEventListener('click', e => {
    const tgt = e.target;
    const btn = (tgt instanceof Element) ? tgt.closest('[data-action="copy-json"],[data-action="download-json"],[data-safe-nav],[data-redirect]') : null;
    if (!btn) return;

    // Guarded nav (re-validate on click)
    if (btn.hasAttribute('data-safe-nav') || btn.hasAttribute('data-redirect')) {
      e.preventDefault();
      const targetRaw = btn.getAttribute('data-safe-nav') || btn.getAttribute('data-redirect');
      return guardedNavigate(targetRaw);
    }

    const runId  = btn.getAttribute('data-run-id');
    const action = btn.getAttribute('data-action');
    if (!runId || !action) return;
    const jsonTxt = getRunJson(runId);
    if (!jsonTxt) return showToast('toastError','No JSON found.');

    if (action === 'copy-json') {
      if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(jsonTxt)
          .then(()=> showToast('toastInfo','Copied JSON to clipboard.'))
          .catch(()=> showToast('toastError','Copy failed.'));
      } else {
        try {
          const ta = document.createElement('textarea');
          ta.value = jsonTxt;
          // Use CSS-driven offscreen class instead of inline styles
          ta.classList.add('offscreen-temp');
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
          showToast('toastInfo','Copied JSON to clipboard.');
        } catch {
          showToast('toastError','Copy failed.');
        }
      }
    } else if (action === 'download-json') {
      const blob = new Blob([jsonTxt], { type:'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `run-${runId}.json`;
      document.body.appendChild(a);
      a.click();
      setTimeout(()=> {
        URL.revokeObjectURL(a.href);
        a.remove();
      }, 150);
      showToast('toastInfo','Download started.');
    }
  });

  // ---- Filtering ----
  const filterInput  = qs('#runFilterInput');
  const statusFilter = qs('#statusFilter');
  const sourceFilter = qs('#sourceFilter');
  const clearBtn     = qs('#clearFilters');

  function applyFilters() {
    const text   = (filterInput?.value || '').toLowerCase();
    const status = statusFilter?.value;
    const source = sourceFilter?.value;

    const match = (rid, sid, st, src) => {
      if (text && !(rid.includes(text) || sid.includes(text) || st.includes(text) || src.includes(text))) return false;
      if (status && st !== status) return false;
      if (source && src !== source) return false;
      return true;
    };

    qsa('.run-row').forEach(row => {
      const rid = (row.dataset.runId || '').toLowerCase();
      const sid = (row.dataset.sessionId || '').toLowerCase();
      const st  = (row.dataset.status || '').toLowerCase();
      const src = (row.dataset.source || '').toLowerCase();
      // Toggle presentation via CSS class instead of inline style
      row.classList.toggle('hidden', !match(rid,sid,st,src));
    });

    qsa('.run-accordion-item').forEach(item => {
      const rid = (item.dataset.runId || '').toLowerCase();
      const sid = (item.dataset.sessionId || '').toLowerCase();
      const st  = (item.dataset.status || '').toLowerCase();
      const src = (item.dataset.source || '').toLowerCase();
      // Toggle presentation via CSS class instead of inline style
      item.classList.toggle('hidden', !match(rid,sid,st,src));
    });
  }

  [filterInput, statusFilter, sourceFilter].forEach(el => {
    if (!el) return;
    el.addEventListener('input', applyFilters);
    el.addEventListener('change', applyFilters);
  });

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (filterInput)  filterInput.value  = '';
      if (statusFilter) statusFilter.value = '';
      if (sourceFilter) sourceFilter.value = '';
      applyFilters();
    });
  }

  // ---- Smooth scroll (anchors only) ----
  qsa('[data-scroll-to]').forEach(a => {
    a.addEventListener('click', ev => {
      const href = a.getAttribute('href');
      if (!href || !href.startsWith('#run-card-')) return;
      ev.preventDefault();
      const target = qs(href);
      if (target) {
        target.scrollIntoView({ behavior:'smooth', block:'start' });
        const collapseEl = target.querySelector('.accordion-collapse');
        if (collapseEl && !collapseEl.classList.contains('show')) {
          bootstrap.Collapse.getOrCreateInstance(collapseEl).show();
        }
      }
    });
  });

  // Initial filter + optional exposure of safe next (if needed)
  applyFilters();
  if (SAFE_NEXT) {
    // If you want to surface a safe navigation button:
    // const safeBtn = qs('#safeNextBtn'); if (safeBtn) safeBtn.dataset.safeNav = SAFE_NEXT;
  }
});