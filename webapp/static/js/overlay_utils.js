// Adaptive overlay utilities: inject CSS and position overlays to remain visible
// Lightweight, CSP-friendly helper (no external deps)
(function () {
  'use strict';

  function debounce(fn, wait) {
    let t = null;
    return function () {
      const args = arguments;
      clearTimeout(t);
      t = setTimeout(function () { fn.apply(null, args); }, wait);
    };
  }

  function injectStyles() {
    try {
      // Use external stylesheet to satisfy strict CSP (avoid inline <style>)
      var existing = document.querySelector('link[data-generated-by="overlay_utils"]');
      if (existing) return;
      var l = document.createElement('link');
      l.rel = 'stylesheet';
      l.href = '/static/css/overlay_utils.css';
      l.setAttribute('data-generated-by', 'overlay_utils');
      document.head && document.head.appendChild(l);
      // also ensure dynamic overlay stylesheet is present (for CSSOM rules)
      var dyn = document.querySelector('link[data-generated-by="overlay_dynamic"]');
      if (!dyn) {
        var d = document.createElement('link');
        d.rel = 'stylesheet';
        d.href = '/static/css/overlay_dynamic.css';
        d.setAttribute('data-generated-by', 'overlay_dynamic');
        document.head && document.head.appendChild(d);
      }
    } catch (e) {
      console && console.debug && console.debug('overlay_utils: injectStyles failed to add link', e);
    }
  }

  // CSSOM dynamic stylesheet manager for per-overlay position rules
  let _dynamicSheet = null;
  let _overlayIdCounter = 1;

  function _ensureDynamicSheet(cb) {
    if (_dynamicSheet) { cb(_dynamicSheet); return; }
    try {
      // find the link we just added
      const links = Array.from(document.styleSheets || []).filter(function (s) {
        try { return s.href && s.href.indexOf('/static/css/overlay_dynamic.css') !== -1; } catch (e) { return false; }
      });
      if (links.length > 0) {
        _dynamicSheet = links[0];
        cb(_dynamicSheet);
        return;
      }
      // if not available yet, wait for load on the link element
      const ln = document.querySelector('link[data-generated-by="overlay_dynamic"]');
      if (ln) {
        ln.addEventListener('load', function () {
          const sheets = Array.from(document.styleSheets).filter(function (s) { try { return s.href && s.href.indexOf('/static/css/overlay_dynamic.css') !== -1; } catch (e) { return false; } });
          _dynamicSheet = sheets[0] || null;
          cb(_dynamicSheet);
        }, { once: true });
        // if already loaded, try again synchronously
        setTimeout(function () { _ensureDynamicSheet(cb); }, 50);
        return;
      }
      // fallback: create a link synchronously and wait
      var d2 = document.createElement('link');
      d2.rel = 'stylesheet';
      d2.href = '/static/css/overlay_dynamic.css';
      d2.setAttribute('data-generated-by', 'overlay_dynamic');
      d2.addEventListener('load', function () { _ensureDynamicSheet(cb); }, { once: true });
      document.head && document.head.appendChild(d2);
    } catch (e) {
      cb(null);
    }
  }

  function _setOverlayPositionRule(overlayId, cssText) {
    _ensureDynamicSheet(function (sheet) {
      try {
        if (!sheet) return;
        // remove previous rule for this id if exists
        const selector = '.overlay-pos-' + overlayId;
        // iterate rules and remove matching selectorText
        const rules = sheet.cssRules || sheet.rules || [];
        for (let i = rules.length - 1; i >= 0; i--) {
          try {
            if (rules[i] && rules[i].selectorText === selector) sheet.deleteRule(i);
          } catch (e) {}
        }
        // insert new rule
        const rule = selector + ' { position: fixed; ' + cssText + ' }';
        try { sheet.insertRule(rule, sheet.cssRules.length); } catch (e) { /* ignore */ }
      } catch (e) {}
    });
  }

  function _removeOverlayPositionRule(overlayId) {
    _ensureDynamicSheet(function (sheet) {
      try {
        if (!sheet) return;
        const rules = sheet.cssRules || sheet.rules || [];
        for (let i = rules.length - 1; i >= 0; i--) {
          try { if (rules[i] && rules[i].selectorText === ('.overlay-pos-' + overlayId)) sheet.deleteRule(i); } catch (e) {}
        }
      } catch (e) {}
    });
  }

  function findAnchorForOverlay(overlay) {
    // preferred: data-anchor-id, then aria-labelledby / aria-describedby, then aria-controls reverse
    try {
      /**
       * @param {Element|null|undefined} el
       * @returns {el is HTMLElement}
       */
      function isHTMLElement(el) {
        return el instanceof HTMLElement;
      }
      const aid = overlay.getAttribute && overlay.getAttribute('data-anchor-id');
      if (aid) return document.getElementById(aid) || document.querySelector('[data-overlay-anchor="' + aid + '"]');
      // try aria-controls reverse lookup
      const id = overlay.id;
      if (id) {
        const btn = document.querySelector('[aria-controls="' + id + '"]') || document.querySelector('[data-target="' + id + '"]');
        if (btn) return btn;
      }
      // fallback: previousElementSibling or button with data-anchor
      const prev = overlay.previousElementSibling;
      if (isHTMLElement(prev)) return prev;
    } catch (e) {}
    return null;
  }

  function positionOverlay(overlay) {
    try {
      if (!overlay || !(overlay instanceof HTMLElement)) return;
      // max-height controlled by CSS (avoid inline style to respect CSP)

      // remove alignment classes first
      overlay.classList.remove('overlay--align-right');
      overlay.classList.remove('overlay--align-top');
      overlay.classList.remove('overlay--center');

      // Prefer CSSOM positioning: keep overlays outside anchors and set fixed positions via dynamic stylesheet
      var useCssom = true;
      if (useCssom) {
        try {
          // ensure overlay has an id for rule scoping
          var oid = overlay.getAttribute('data-overlay-id');
          if (!oid) {
            oid = String(_overlayIdCounter++);
            overlay.setAttribute('data-overlay-id', oid);
            overlay.classList.add('overlay-pos-' + oid);
          }
          // make sure overlay is direct child of body
          if (overlay.parentElement !== document.body) document.body.appendChild(overlay);

          // compute desired fixed left/top to keep overlay visible
          const anchor = findAnchorForOverlay(overlay);
          const oRect = overlay.getBoundingClientRect();
          let left = Math.max(8, (anchor ? anchor.getBoundingClientRect().left : (window.innerWidth - oRect.width) / 2));
          let top = Math.max(8, (anchor ? (anchor.getBoundingClientRect().bottom + 6) : (window.innerHeight - oRect.height) / 2));
          // prefer flipping to top if not enough space below
          if (anchor) {
            const aRect = anchor.getBoundingClientRect();
            const spaceBelow = window.innerHeight - aRect.bottom;
            const spaceAbove = aRect.top;
            if (spaceBelow < oRect.height && spaceAbove > spaceBelow) {
              top = Math.max(8, aRect.top - oRect.height - 6);
            }
            // clamp to viewport width
            if (left + oRect.width > window.innerWidth - 8) left = Math.max(8, window.innerWidth - oRect.width - 8);
          } else {
            // center horizontally
            left = Math.max(8, Math.round((window.innerWidth - oRect.width) / 2));
            top = Math.max(8, Math.round((window.innerHeight - oRect.height) / 2));
          }

          // set max-height via CSS rule as well (so overlay scrolls internally)
          const maxh = Math.max(60, Math.floor(window.innerHeight - top - 12));
          const cssText = 'left: ' + left + 'px; top: ' + top + 'px; max-height: ' + maxh + 'px; overflow: auto; z-index: 9999;';
          _setOverlayPositionRule(oid, cssText);
          // ensure aria / open classes
          overlay.classList.remove('overlay-attached', 'overlay-top', 'align-right');
          overlay.classList.remove('overlay--center');
          return;
        } catch (e) {
          // if CSSOM approach fails, fall back to class-driven attached behavior below
        }
      }

      const anchor = findAnchorForOverlay(overlay);
      if (!anchor || !(anchor instanceof HTMLElement)) {
        // center if no anchor: ensure overlay is a child of body and use CSS centered class
        if (overlay.parentElement !== document.body) {
          document.body.appendChild(overlay);
        }
        overlay.classList.remove('overlay-attached', 'overlay-top', 'align-right');
        overlay.classList.add('overlay--center');
        return;
      }

      const aRect = anchor.getBoundingClientRect();
      // compute where overlay would land if placed below
      const preferBelow = true;
      const oRect = overlay.getBoundingClientRect();
      const spaceBelow = window.innerHeight - aRect.bottom;
      const spaceAbove = aRect.top;

      // Attach overlay into the anchor to use CSS for positioning (avoid inline styles)
      try {
        // mark anchor as positioning context
        if (!anchor.classList.contains('overlay-anchor')) anchor.classList.add('overlay-anchor');
        // move overlay into anchor if not already
        if (overlay.parentElement !== anchor) anchor.appendChild(overlay);
        // clear center class and ensure attached class
        overlay.classList.remove('overlay--center');
        overlay.classList.add('overlay-attached');

        // decide align-right vs default left
        const wouldOverflowRight = (aRect.left + oRect.width) > window.innerWidth - 12;
        if (wouldOverflowRight) overlay.classList.add('align-right'); else overlay.classList.remove('align-right');

        // decide top flip
        if (preferBelow && spaceBelow < Math.min(200, oRect.height) && spaceAbove > spaceBelow) {
          overlay.classList.add('overlay-top');
        } else {
          overlay.classList.remove('overlay-top');
        }
      } catch (e) {
        // fallback: ensure centered
        if (overlay.parentElement !== document.body) document.body.appendChild(overlay);
        overlay.classList.remove('overlay-attached', 'overlay-top', 'align-right');
        overlay.classList.add('overlay--center');
      }
    } catch (e) {
      console && console.debug && console.debug('overlay_utils.positionOverlay err', e);
    }
  }

  function positionAllOverlays() {
    try {
      const overlays = Array.from(document.querySelectorAll('.overlay, .nav-dropdown, #navMoreDropdown'));
      overlays.forEach(function (ov) {
        // only position visible overlays
        if (!ov || !(ov instanceof HTMLElement)) return;
        const visible = ov.offsetParent !== null && ov.getAttribute('aria-hidden') !== 'true';
        if (visible) positionOverlay(ov);
      });
    } catch (e) {}
  }

  const debouncedPositionAll = debounce(positionAllOverlays, 120);

  function install() {
    injectStyles();
    // reposition on resize/orientation
    window.addEventListener('resize', debouncedPositionAll);
    window.addEventListener('orientationchange', debouncedPositionAll);

    // reposition on global clicks (useful after menus open)
    document.addEventListener('click', debounce(function () {
      debouncedPositionAll();
    }, 50), true);

    // close overlays on Escape and reposition on open
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape' || ev.key === 'Esc') {
        try {
          const open = Array.from(document.querySelectorAll('.overlay, .nav-dropdown, #navMoreDropdown'))
            .filter(function (el) { return (el instanceof HTMLElement) && el.offsetParent !== null && el.getAttribute('aria-hidden') !== 'true'; });
          open.forEach(function (o) {
            if (o.setAttribute) o.setAttribute('aria-hidden', 'true');
            o.classList.remove('open');
            try {
              var oid = o.getAttribute && o.getAttribute('data-overlay-id');
              if (oid) _removeOverlayPositionRule(oid);
            } catch (e) {}
          });
        } catch (e) {}
      }
    });

    // MutationObserver: watch for added overlays or attribute changes
    try {
      const mo = new MutationObserver(debounce(function () { debouncedPositionAll(); }, 80));
      mo.observe(document.documentElement || document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class', 'aria-hidden'] });
    } catch (e) {}

    // initial pass after DOM ready
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
      setTimeout(positionAllOverlays, 80);
    } else {
      document.addEventListener('DOMContentLoaded', function () { setTimeout(positionAllOverlays, 80); });
    }
  }

  // Focus-trap and body scroll-lock
  const trapMap = new WeakMap();
  let overlayOpenCount = 0;
  let previousBodyOverflow = null;

  function bodyLock() {
    try {
      if (overlayOpenCount === 0) {
        document.documentElement.classList.add('overlay-open');
      }
      overlayOpenCount++;
    } catch (e) {}
  }

  function bodyUnlock() {
    try {
      overlayOpenCount = Math.max(0, overlayOpenCount - 1);
      if (overlayOpenCount === 0) {
        document.documentElement.classList.remove('overlay-open');
      }
    } catch (e) {}
  }

  function focusableElementsWithin(el) {
    if (!el || !el.querySelectorAll) return [];
    return Array.from(el.querySelectorAll('a[href], button:not([disabled]), textarea, input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])')).filter(function (i) { return (i instanceof HTMLElement) && i.offsetParent !== null; });
  }

  function enableFocusTrap(overlay) {
    try {
      if (!(overlay instanceof HTMLElement)) return;
      if (trapMap.has(overlay)) return;
      const prev = document.activeElement;
      const saved = { prev: prev };
      trapMap.set(overlay, saved);

      // ensure overlay is focusable
      if (!overlay.hasAttribute('tabindex')) overlay.setAttribute('tabindex', '-1');

      const handler = function (ev) {
        if (ev.key === 'Tab') {
          const focusables = focusableElementsWithin(overlay);
          if (focusables.length === 0) {
            ev.preventDefault();
            overlay.focus();
            return;
          }
          const first = focusables[0];
          const last = focusables[focusables.length - 1];
          if (!ev.shiftKey && document.activeElement === last) {
            ev.preventDefault();
            first.focus();
          } else if (ev.shiftKey && document.activeElement === first) {
            ev.preventDefault();
            last.focus();
          }
        }
      };
      saved.keyHandler = handler;
      overlay.addEventListener('keydown', handler);

      // focus first focusable or the overlay itself
      const focusables = focusableElementsWithin(overlay);
      setTimeout(function () {
        if (focusables.length) focusables[0].focus(); else overlay.focus();
      }, 10);

      bodyLock();
      overlay.setAttribute('aria-hidden', 'false');
      overlay.classList.add('open');
    } catch (e) { console && console.debug && console.debug('enableFocusTrap err', e); }
  }

  function disableFocusTrap(overlay) {
    try {
      if (!(overlay instanceof HTMLElement)) return;
      const saved = trapMap.get(overlay);
      if (!saved) return;
      if (saved.keyHandler) overlay.removeEventListener('keydown', saved.keyHandler);
      trapMap.delete(overlay);
      overlay.setAttribute('aria-hidden', 'true');
      overlay.classList.remove('open');
      try { var oid2 = overlay.getAttribute && overlay.getAttribute('data-overlay-id'); if (oid2) _removeOverlayPositionRule(oid2); } catch (e) {}
      bodyUnlock();
      // restore focus
      try { if (saved.prev && saved.prev.focus) saved.prev.focus(); } catch (e) {}
    } catch (e) { console && console.debug && console.debug('disableFocusTrap err', e); }
  }

  // Wire into mutation observer: when overlay becomes visible -> enable trap; when hidden -> disable
  try {
    const mo2 = new MutationObserver(function (records) {
      records.forEach(function (rec) {
        try {
          const target = rec.target;
          if (!(target instanceof HTMLElement)) return;
          if (target.matches && (target.matches('.overlay') || target.matches('.nav-dropdown') || target.id === 'navMoreDropdown')) {
            const visible = target.offsetParent !== null && target.getAttribute('aria-hidden') !== 'true';
            const wantsTrap = target.getAttribute('data-focus-trap') === 'true' || target.hasAttribute('data-trap');
            if (visible && wantsTrap) {
              enableFocusTrap(target);
            } else if (!visible) {
              disableFocusTrap(target);
            }
          }
        } catch (e) {}
      });
    });
    mo2.observe(document.documentElement || document.body, { subtree: true, attributes: true, attributeFilter: ['class', 'style', 'aria-hidden'], childList: true });
  } catch (e) {}

  // Expose for debugging & manual control
  window['__overlay_utils__'] = {
    positionAllOverlays: positionAllOverlays,
    positionOverlay: positionOverlay,
    enableFocusTrap: enableFocusTrap,
    disableFocusTrap: disableFocusTrap,
    bodyLock: bodyLock,
    bodyUnlock: bodyUnlock,
  };

  // Install on load
  try { install(); } catch (e) { console && console.debug && console.debug('overlay_utils install failed', e); }

})();
