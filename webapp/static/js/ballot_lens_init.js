// @ts-nocheck
// CSP-friendly initialization script for ballot_lens
// - Polyfills/addEventListener defensive shim (moved from inline template)
// - Socket.IO config extraction (moved from inline template)
// This file is intentionally minimal and avoids touching the DOM aggressively.
(function () {
  'use strict';
  try {
    if (typeof document !== 'undefined' && typeof document.addEventListener !== 'function') {
      if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        document.addEventListener = function (evt, cb, opts) {
          window.addEventListener(evt, cb, opts);
          return function () { try { window.removeEventListener(evt, cb, opts); } catch (e) {} };
        };
      } else {
        document.addEventListener = function () { return function () {}; };
      }
    }
  } catch (e) {
    // Best-effort only; don't break the page
  }

  // Socket.IO config extraction: read data attribute from vendor script tag
  try {
    var configEl = document.querySelector('script[data-socketio-config]');
    var defaultConfig = { transports: ['websocket', 'polling'], upgrade: true, pingInterval: 10000, pingTimeout: 60000 };
    if (configEl) {
      try {
        var configData = configEl.getAttribute('data-socketio-config');
        if (configData && typeof configData === 'string' && configData.trim().length > 2) {
          var trimmed = configData.trim();
          if (trimmed.charAt(0) === '{' && trimmed.charAt(trimmed.length - 1) === '}') {
            var parsed = JSON.parse(trimmed);
            window.__SOCKETIO_CONFIG__ = Object.keys(parsed).length > 0 ? parsed : defaultConfig;
          } else {
            window.__SOCKETIO_CONFIG__ = defaultConfig;
          }
        } else {
          window.__SOCKETIO_CONFIG__ = defaultConfig;
        }
      } catch (e) {
        console && console.warn && console.warn('Failed to parse Socket.IO config, using defaults:', e);
        window.__SOCKETIO_CONFIG__ = defaultConfig;
      }
    } else {
      window.__SOCKETIO_CONFIG__ = defaultConfig;
    }
  } catch (e) {
    window.__SOCKETIO_CONFIG__ = { transports: ['websocket', 'polling'], upgrade: true, pingInterval: 10000, pingTimeout: 60000 };
  }

  // Attach delegated handlers for elements with `data-confirm` or `data-action` attributes
  try {
    document.addEventListener('click', function (ev) {
      try {
        var tgt = ev.target;
        while (tgt && tgt !== document) {
          // Only operate on element-like targets
          if (tgt && typeof tgt.getAttribute === 'function') {
            // Handle data-confirm
            var confirmMsg = tgt.getAttribute('data-confirm');
            if (confirmMsg) {
              if (!window.confirm(confirmMsg)) {
                ev.preventDefault();
                ev.stopPropagation();
                return false;
              }
              return true;
            }

            // Handle data-action (simple built-in actions)
            var action = tgt.getAttribute('data-action');
            if (action) {
              try {
                if (action === 'modal-remove') {
                  var m = tgt.closest && tgt.closest('.modal');
                  if (m && m.parentNode) m.parentNode.removeChild(m);
                  ev.preventDefault();
                  return true;
                }
                if (action === 'modal-hide' || action === 'modal-close') {
                  var m2 = tgt.closest && tgt.closest('.modal');
                  if (m2 && m2.classList) m2.classList.add('hidden');
                  ev.preventDefault();
                  return true;
                }
              } catch (e) {
                // ignore handler errors
              }
            }
          }
          tgt = tgt && (tgt.parentNode || tgt.parentElement) ? (tgt.parentNode || tgt.parentElement) : null;
        }
      } catch (e) {
        // ignore
      }
    }, true);
  } catch (e) {
    // noop
  }

  // Prevent default submit for forms that should be handled client-side (class: run-filters)
  try {
    document.addEventListener('submit', function (ev) {
      try {
        var form = ev.target;
        if (form && form.classList && form.classList.contains && form.classList.contains('run-filters')) {
          ev.preventDefault();
        }
      } catch (e) {}
    }, true);
  } catch (e) {}
})();

// Defer-load overlay utils (CSP-friendly dynamic loader)
(function () {
  try {
    var s = document.createElement('script');
    s.src = '/static/js/overlay_utils.js';
    s.defer = true;
    s.async = false;
    s.setAttribute('data-generated-by', 'ballot_lens_init');
    document.head && document.head.appendChild(s);
  } catch (e) {
    console && console.debug && console.debug('Failed to load overlay_utils', e);
  }
})();
