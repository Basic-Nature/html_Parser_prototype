"use strict";
/* nav_guard.js
   Shared client-side defense vs unvalidated URL redirection.
   - Validates any navigation originating from data-safe-nav.
   - Blocks protocol-relative, foreign origins, dangerous schemes.
   - Adds rel noopener/noreferrer to target=_blank links.
*/
document.addEventListener('DOMContentLoaded', () => {
  const DANGER = /^(javascript|data|vbscript|file|ws|wss|mailto):/i;
  const ALLOW_PREFIXES = ['/', '/history', '/ballot-lens', '/data_framework', '/api/'];
  const MAX_LEN = 512;

  const strip = s => s.replace(/[\u0000-\u001F\u007F]+/g,'').trim();
  const decodeMulti = (v, n=3) => {
    let cur = v;
    for (let i=0;i<n;i++){
      try {
        const d = decodeURIComponent(cur);
        if (d === cur) break;
        cur = d;
      } catch { break; }
    }
    return cur;
  };

  function canonical(raw) {
    if (!raw) return '';
    raw = strip(raw).slice(0, MAX_LEN);
    if ((raw.startsWith('"') && raw.endsWith('"'))||(raw.startsWith("'")&&raw.endsWith("'")))
      raw = raw.slice(1,-1);
    raw = decodeMulti(raw);
    if (raw.startsWith('//')) return ''; // protocol-relative block
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(raw)) {
      if (DANGER.test(raw)) return '';
      try {
        const u = new URL(raw);
        if (u.origin !== window.location.origin) return '';
        raw = u.pathname + u.search + u.hash;
      } catch { return ''; }
    }
    if (raw.startsWith('#')) return raw;
    raw = raw.replace(/\/{2,}/g,'/');
    return raw;
  }

  function isAllowed(path) {
    if (!path) return false;
    if (path.startsWith('#')) return true;
    if (!path.startsWith('/')) return false;
    const twice = decodeMulti(path);
    if (DANGER.test(twice)) return false;
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(twice)) return false;
    return ALLOW_PREFIXES.some(p => {
      if (p === '/') return true;
      const withSlash = p.endsWith('/') ? p : p + '/';
      return twice === p || twice.startsWith(withSlash);
    });
  }

  function guardedNavigate(raw) {
    const c = canonical(raw);
    if (isAllowed(c)) window.location.assign(c);
    else console.warn('Blocked unsafe navigation', raw);
  }

  document.addEventListener('click', e => {
    const tgt = e.target;
    const a = (tgt instanceof Element) ? tgt.closest('a[data-safe-nav]') : null;
    if (!a) return;
    const target = a.getAttribute('data-safe-nav') || a.getAttribute('href');
    if (!target) return;
    const c = canonical(target);
    if (!isAllowed(c)) {
      e.preventDefault();
      console.warn('Blocked unsafe navigation', target);
    } else if (c !== target) {
      e.preventDefault();
      guardedNavigate(c);
    }
  }, { capture:true });

  // Harden target=_blank links
  document.querySelectorAll('a[target="_blank"]').forEach(a => {
    const rel = (a.getAttribute('rel')||'').toLowerCase();
    const needed = ['noopener','noreferrer'];
    needed.forEach(flag=>{
      if (!rel.includes(flag))
        a.setAttribute('rel', (rel ? rel + ' ' : '') + flag);
    });
  });
});