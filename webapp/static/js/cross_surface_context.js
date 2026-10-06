(function (root, factory) {
  'use strict';

  const api = factory();

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }

  if (!root) {
    return;
  }

  root.ElectionPulseCrossSurfaceContext = Object.freeze(api);

  if (!root.document || !root.location) {
    return;
  }

  const apply = () => {
    api.applyContextNavigation(
      root.document,
      root.location.href,
      root.location.origin
    );
  };

  if (root.document.readyState === 'loading') {
    root.document.addEventListener('DOMContentLoaded', apply, { once: true });
  } else {
    apply();
  }
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  const CONTEXT_KEYS = Object.freeze(['state', 'year']);
  const TARGET_PATHS = Object.freeze(['/worklist', '/data_framework']);

  function buildContextHref(sourceHref, targetHref, markerPath, origin) {
    if (
      typeof sourceHref !== 'string'
      || typeof targetHref !== 'string'
      || typeof markerPath !== 'string'
      || typeof origin !== 'string'
    ) {
      return null;
    }

    try {
      const source = new URL(sourceHref, origin);
      const target = new URL(targetHref, origin);

      if (source.origin !== origin || target.origin !== origin) {
        return null;
      }
      if (!TARGET_PATHS.includes(markerPath) || target.pathname !== markerPath) {
        return null;
      }

      CONTEXT_KEYS.forEach((key) => {
        const value = source.searchParams.get(key);
        if (value === null || value.trim() === '') {
          target.searchParams.delete(key);
        } else {
          target.searchParams.set(key, value);
        }
      });

      return `${target.pathname}${target.search}${target.hash}`;
    } catch (_error) {
      return null;
    }
  }

  function decorateAnchor(anchor, sourceHref, origin) {
    if (!anchor || typeof anchor.getAttribute !== 'function') {
      return false;
    }

    const markerPath = anchor.getAttribute('data-o4e-context-nav');
    const targetHref = anchor.getAttribute('href');
    const nextHref = buildContextHref(
      sourceHref,
      targetHref,
      markerPath,
      origin
    );

    if (nextHref === null) {
      return false;
    }

    anchor.setAttribute('href', nextHref);
    return true;
  }

  function applyContextNavigation(documentRef, sourceHref, origin) {
    if (!documentRef || typeof documentRef.querySelectorAll !== 'function') {
      return 0;
    }

    let updated = 0;
    documentRef.querySelectorAll('a[data-o4e-context-nav]').forEach((anchor) => {
      if (decorateAnchor(anchor, sourceHref, origin)) {
        updated += 1;
      }
    });
    return updated;
  }

  return Object.freeze({
    CONTEXT_KEYS,
    TARGET_PATHS,
    buildContextHref,
    decorateAnchor,
    applyContextNavigation,
  });
});
