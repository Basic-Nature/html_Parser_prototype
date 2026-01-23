(function(){
  // Lightweight helpers to safely access EventTarget values and node containment
  // Exposed on `window.__tl_helpers` for reuse across this large file.
  try {
          const W = /** @type {any} */ (window);
          W.__tl_helpers = W.__tl_helpers || {};
          /**
           * @typedef {Event & { target?: EventTarget | null }} DomEvent
           * @typedef {(ev: DomEvent) => string} TargetValueFn
           * @typedef {(ev: DomEvent) => boolean} TargetCheckedFn
           * @typedef {(ev: DomEvent, sel: string) => Element | null} TargetClosestFn
  /* LogRecord typedef consolidated at top of file. */

    /** @type {TlHelpers} */
    W.__tl_helpers = /** @type {TlHelpers} */ (W.__tl_helpers || {});

    /**
     * Safely extract value from common form event targets.
     * @type {TargetValueFn}
     */
    /** @param {Event & { target?: EventTarget | null }} ev */
    W.__tl_helpers.targetValue = function(ev) {
      try {
      const t = ev && ev.target;
      return (t instanceof HTMLInputElement || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) ? t.value : '';
      } catch (/** @type {any} */ e) {
      return '';
      }
    };
    /**
     * @interface TlHelpersInterface
     * @property {TargetValueFn} targetValue
     * @property {TargetCheckedFn} targetChecked
     * @property {TargetClosestFn} targetClosest
     * @property {NodeContainsFn} nodeContains
     */

    /**
     * Safely extract checked state from common form event targets.
     * @type {TargetCheckedFn}
     */
    /** @param {Event & { target?: EventTarget | null }} ev */
    W.__tl_helpers.targetChecked = function(ev) {
      try { const t = ev && ev.target; return (t instanceof HTMLInputElement) ? t.checked : false; } catch (/** @type {any} */ e) { return false; }
    };
    /**
     * Safely find the closest ancestor matching selector from an event target.
     * Uses the shared TargetClosestFn typedef: (ev: DomEvent, sel: string) => Element | null
     * @type {TargetClosestFn}
     */
    /** @param {Event & { target?: EventTarget | null }} ev */
    /** @param {string} sel */
    W.__tl_helpers.targetClosest = /** @type {TargetClosestFn} */ (function(ev, sel) {
      try {
      const t = ev && ev.target;
      return (t instanceof Element) ? t.closest(sel) : null;
      } catch (/** @type {any} */ e) {
      return null;
      }
    });
    /**
     * Safely test node containment from a container node to a target node.
     * @type {NodeContainsFn}
     */
    /** @param {Node} container
     *  @param {Node} t
     */
    W.__tl_helpers.nodeContains = /** @type {NodeContainsFn} */ (function(container, t) {
      try { return (t instanceof Node) && container && container.contains && container.contains(t); } catch (/** @type {any} */ e) { return false; }
    });
    /**
     * Safely get element value (for inputs/selects/textareas) or empty string.
     * @param {Element|null|undefined} el
     * @returns {string}
     */
    W.__tl_helpers.elValue = function(el) {
      try {
        return (el instanceof HTMLInputElement || el instanceof HTMLSelectElement || el instanceof HTMLTextAreaElement) ? el.value : '';
      } catch (/** @type {any} */ e) {
        return '';
      }
    };
    /**
     * Safely set element value where supported.
     * @param {Element|null|undefined} el
     * @param {string} v
     */
    W.__tl_helpers.setElValue = function(el, v) {
      try {
        if (el instanceof HTMLInputElement || el instanceof HTMLSelectElement || el instanceof HTMLTextAreaElement) el.value = String(v);
      } catch (/** @type {any} */ e) { /* noop */ }
    };
    /**
     * Safely get checkbox/radio checked state.
     * @param {Element|null|undefined} el
     * @returns {boolean}
     */
    W.__tl_helpers.elChecked = function(el) {
      try { return (el instanceof HTMLInputElement) ? !!el.checked : false; } catch (/** @type {any} */ e) { return false; }
    };
    /**
     * Safely set checked where supported.
     * @param {Element|null|undefined} el
     * @param {boolean} v
     */
    W.__tl_helpers.setElChecked = function(el, v) {
      try { if (el instanceof HTMLInputElement) el.checked = !!v; } catch (/** @type {any} */ e) { /* noop */ }
    };
    /**
     * Safe click helper: calls .click() if available
     * @param {Element|null|undefined} el
     */
    W.__tl_helpers.safeClick = function(el) {
      try { if (el && typeof /** @type {any} */ (el).click === 'function') /** @type {any} */ (el).click(); } catch (/** @type {any} */ e) { /* noop */ }
    };
    /**
     * Safely set disabled on buttons/inputs
     * @param {Element|null|undefined} el
     * @param {boolean} v
     */
    W.__tl_helpers.setDisabled = function(el, v) {
      try { if (el instanceof HTMLButtonElement || el instanceof HTMLInputElement) el.disabled = !!v; } catch (/** @type {any} */ e) { /* noop */ }
    };
    /**
     * Bound-property helpers for storing listener refs on elements.
     * @param {Element|null|undefined} el
     * @param {string} prop
     * @param {any} val
     */
    W.__tl_helpers.setBound = function(el, prop, val) { try { if (el && typeof el === 'object') el[prop] = val; } catch (/** @type {any} */ e) {} };
    W.__tl_helpers.getBound = function(el, prop) { try { return el && typeof el === 'object' ? el[prop] : undefined; } catch (/** @type {any} */ e) { return undefined; } };
  } catch (/** @type {any} */ e) { /* ignore helper install errors */ }
})();

/* --------------------------------------------------------------------------
 * Canonical typedefs -- single source of truth to avoid duplicate JSDoc defs
 * Consolidate commonly-reused typedef names here. If you add new typedefs,
 * append them to this block rather than repeating the same @typedef later.
 * -------------------------------------------------------------------------- */
/**
 * @template T
 * @typedef {T} VirtualItem
 */

/** @typedef {Event & { target?: EventTarget | null }} DomEvent */
/** @typedef {(ev: DomEvent) => string} TargetValueFn */
/** @typedef {(ev: DomEvent) => boolean} TargetCheckedFn */
/** @typedef {(ev: DomEvent, sel: string) => Element | null} TargetClosestFn */
/** @typedef {(container: Node, t: Node) => boolean} NodeContainsFn */
/** @typedef {{ targetValue: TargetValueFn, targetChecked: TargetCheckedFn, targetClosest: TargetClosestFn, nodeContains: NodeContainsFn }} TlHelpers */

/** @typedef {Object.<string, any>} PreviewRow */
/** @typedef {PreviewRow[]} PreviewData */

/** @typedef {Object} LogRecord */
/** @typedef {Object.<string, any>} ParserOutputEvent */
/** @typedef {Object} ContestOptionsPayload */

/** @typedef {{ index?: number, value?: string|number, label?: string, meta?: string, metadata?: Object.<string, any> }} PromptOption */
/** @typedef {{ bundled?: boolean, isChild?: boolean }} CreateBtnOptions */
/** @typedef {Object} Result */
/** @typedef {(e: Event) => void} ClickHandler */
/** @typedef {Object} ResultCardButton */
/** @typedef {string|number} PromptValue */
/** @typedef {Object} ParserPromptPayload */
/** @callback EmitPromptFn */
/** @callback SubmitPromptFn */

const __tl_window = /** @type {any} */ (typeof window !== 'undefined' ? window : {});

// Use the unified sidebar controller; legacy sidebar handlers should no-op.
try {
  __tl_window.__tl_sidebarUnified = true;
} catch (e) {
  /* noop */
}


// Fallback handlers for drawer and sidebar toggles (ensure mobile buttons work)
(function(){
  document.addEventListener('DOMContentLoaded', function(){
    function initSourceCarousel(){
      var carousel = document.getElementById('sourceCarousel');
      if (!carousel) return;
      var track = document.getElementById('sourceCarouselTrack');
      if (!track) return;
      var slides = Array.prototype.slice.call(carousel.querySelectorAll('.source-carousel-slide'));
      var dots = Array.prototype.slice.call(carousel.querySelectorAll('.source-dot'));
      if (!slides.length || !dots.length) return;
      var index = 0;

      function setActive(nextIndex){
        index = Math.max(0, Math.min(nextIndex, slides.length - 1));
        track.style.transform = 'translateX(' + (-index * 100) + '%)';
        slides.forEach(function(slide, idx){
          var isActive = idx === index;
          slide.classList.toggle('is-active', isActive);
          slide.setAttribute('aria-hidden', String(!isActive));
        });
        dots.forEach(function(dot, idx){
          var isActive = idx === index;
          dot.classList.toggle('is-active', isActive);
          dot.setAttribute('aria-selected', String(isActive));
        });
      }

      dots.forEach(function(dot){
        dot.addEventListener('click', function(){
          var idx = parseInt(dot.getAttribute('data-index') || '0', 10);
          if (!Number.isNaN(idx)) setActive(idx);
        });
      });

      var startX = 0;
      var startY = 0;
      var dragging = false;

      function onTouchStart(evt){
        var target = evt.target;
        if (target && (target.closest('input, textarea, select, button, a') || target.isContentEditable)) return;
        var touch = evt.touches && evt.touches[0];
        if (!touch) return;
        dragging = true;
        startX = touch.clientX;
        startY = touch.clientY;
      }

      function onTouchEnd(evt){
        if (!dragging) return;
        dragging = false;
        var touch = (evt.changedTouches && evt.changedTouches[0]) || (evt.touches && evt.touches[0]);
        if (!touch) return;
        var dx = touch.clientX - startX;
        var dy = touch.clientY - startY;
        if (Math.abs(dx) < 40 || Math.abs(dx) < Math.abs(dy)) return;
        if (dx < 0) {
          setActive(index + 1);
        } else {
          setActive(index - 1);
        }
      }

      carousel.addEventListener('touchstart', onTouchStart, {passive: true});
      carousel.addEventListener('touchend', onTouchEnd, {passive: true});
      setActive(0);
    }

    function attachDrawerHandle(){
      if (__tl_window.__tl_sidebarUnified) return;
      var dh = document.getElementById('drawerHandle') || document.querySelector('.drawer-handle');
      if(!dh) return;
      dh.addEventListener('click', function(){
        var targetId = dh.getAttribute('aria-controls') || 'logDrawer';
        var drawer = document.getElementById(targetId) || document.querySelector('#logDrawer');
        if(drawer){
          var isExpanded = dh.getAttribute('aria-expanded') === 'true';
          dh.setAttribute('aria-expanded', String(!isExpanded));
          drawer.classList.toggle('minimized', isExpanded);
          drawer.classList.toggle('expanded', !isExpanded);
        } else if (typeof window.openRight === 'function'){
          try{ window.openRight(); }catch(e){ console.debug(e); }
        }
      }, {passive:true});
    }

    function attachSidebarToggle(){
      if (__tl_window.__tl_sidebarUnified) return;
      var st = document.getElementById('sidebarToggleBtn');
      var stAny = /** @type {any} */ (st);
      if(!st || stAny.__tlLeftBound) return;
      stAny.__tlLeftBound = true;
      st.addEventListener('click', function(){
        var sidebar = document.getElementById('sidebar') || document.querySelector('.sidebar-left, .sidebar');
        if(sidebar){
          var isOpen = sidebar.classList.toggle('sidebar-open');
          document.body.classList.toggle('no-scroll', isOpen);
          st.setAttribute('aria-expanded', String(isOpen));
        } else if (typeof window.openLeft === 'function'){
          try{ window.openLeft(); }catch(e){ console.debug(e); }
        }
      }, {passive:true});
    }

    function compactDrawerCheck(){
      var dh = document.querySelector('.drawer-handle');
      if(!dh) return;
      try{
        var rect = dh.getBoundingClientRect();
        if(rect.width < 140 || window.innerWidth <= 640) dh.classList.add('compact'); else dh.classList.remove('compact');
      }catch(e){ /* ignore */ }
    }

    attachDrawerHandle();
    attachSidebarToggle();
    initSourceCarousel();
    compactDrawerCheck();
    window.addEventListener('resize', compactDrawerCheck);
  });
})();

// Parser tools dropdown: manage aria-expanded, focus trap, and outside clicks
(function(){
  /**
   * @typedef {HTMLAnchorElement|HTMLButtonElement|HTMLInputElement|HTMLSelectElement|HTMLTextAreaElement|HTMLElement} FocusableElement
   *
   * @typedef {Object} FocusableDescendantsInterface
   * @property {(root: Element) => FocusableElement[]} focusableDescendants
   */

  /**
   * Return focusable, visible descendants of a root element.
   * @param {Element} root
   * @returns {FocusableElement[]}
   */
  /** @type {FocusableDescendantsInterface['focusableDescendants']} */
  const focusableDescendants = function(root){
    const sel = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    return /** @type {FocusableElement[]} */ (Array.from(root.querySelectorAll(sel)).filter(el => (el instanceof HTMLElement) && el.offsetParent !== null));
  };

  /**
   * @typedef {HTMLElement & { _onKey?: (e: KeyboardEvent) => void }} ParserToolsDropdown
   * @typedef {HTMLElement} ParserToolsToggle
   */

  /**
   * Close the parser tools dropdown and clean up event handlers.
   * @param {ParserToolsDropdown | null | undefined} dropdown
   * @param {ParserToolsToggle | null | undefined} toggle
   * @returns {void}
   */
  function closeParserTools(
    /** @type {ParserToolsDropdown | null | undefined} */ dropdown,
    /** @type {ParserToolsToggle | null | undefined} */ toggle
  ){
    if(!dropdown || !toggle) return;
    dropdown.setAttribute('aria-hidden', 'true');
    toggle.setAttribute('aria-expanded','false');
    dropdown.classList.remove('open');
    try{ toggle.focus(); }catch(e){}
    // remove stored key handler (use any-cast for custom property)
    const ddAny = /** @type {any} */ (dropdown);
    if(ddAny && ddAny._onKey) { document.removeEventListener('keydown', ddAny._onKey); try { delete ddAny._onKey; } catch (/** @type {any} */ e) {} }
  }

  function openParserTools(
    /** @type {ParserToolsDropdown | null | undefined} */ dropdown,
    /** @type {ParserToolsToggle | null | undefined} */ toggle
  ){
    if(!dropdown || !toggle) return;
    dropdown.setAttribute('aria-hidden', 'false');
    toggle.setAttribute('aria-expanded','true');
    dropdown.classList.add('open');
    const items = focusableDescendants(dropdown);
    if(items.length) items[0].focus();

    // trap focus inside
    /** @param {KeyboardEvent} e */
    function onKey(e){
      if(e.key === 'Escape'){
        closeParserTools(dropdown,toggle);
      }
      if(e.key === 'Tab'){
        const focusables = focusableDescendants(dropdown);
        if(focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length -1];
        if(e.shiftKey && document.activeElement === first){
          e.preventDefault(); last.focus();
        } else if(!e.shiftKey && document.activeElement === last){
          e.preventDefault(); first.focus();
        }
      }
    }
    document.addEventListener('keydown', onKey);
    // store cleanup on element for removal later (use any-cast)
    const ddAny = /** @type {any} */ (dropdown);
    ddAny._onKey = onKey;
  }

  document.addEventListener('DOMContentLoaded', function(){
    const toggle = document.getElementById('btnToggleRightSidebar');
    const rightSidebar = document.querySelector('.sidebar-right');
    if(!toggle || !rightSidebar) return;

    // initialize attributes
    try {
      const isCollapsed = document.body.classList.contains('right-sidebar-collapsed');
      const isOpen = rightSidebar.classList.contains('open') || rightSidebar.classList.contains('sidebar-open');
      toggle.setAttribute('aria-expanded', (isOpen || !isCollapsed) ? 'true' : 'false');
    } catch (e) {}
  });
})();

/**
 * Smart Elections Parser - Modern UI JavaScript
 * Phase 1: Core Layout, SheetJS Integration, Component Interactions
 */

// ============================================
// Configuration & Constants
// ============================================

const CONFIG = {
  toastDuration: 4000,
  logBufferSize: 500,
  searchDebounceMs: 300,
  sessionRefreshMs: 1000,
  maxPreviewRows: 500,
  virtualScrollThreshold: 100,
  virtualScrollItemHeight: 48,
  virtualScrollBuffer: 10,
  maxDirectUrls: 20, // Maximum URLs for batch processing
};

// Accessor to get left toggle element when needed (avoid early DOM query)
function getToggleLeftBtn() { return document.getElementById('sidebarToggleBtn'); }

// Defensive guard: ensure `document.addEventListener` exists and is callable.
// Some injected or third-party code can accidentally overwrite it; avoid a hard crash
// by falling back to `window.addEventListener` or a no-op.
        try {
          // Always wrap document.addEventListener with a safe shim that:
          // - delegates to the original `document.addEventListener` when present
          // - falls back to `window.addEventListener` when needed
          // - ALWAYS returns a cleanup/unsubscribe function so callers can safely call it
          if (typeof document !== 'undefined') {
            const _origAdd = document.addEventListener && typeof document.addEventListener === 'function' ? document.addEventListener.bind(document) : null;
            const _origRemove = document.removeEventListener && typeof document.removeEventListener === 'function' ? document.removeEventListener.bind(document) : null;
            /**
             * @interface NativeEventListener
             * @param {Event|string} event
             * @param {EventListenerOrEventListenerObject} listener
             * @param {boolean|AddEventListenerOptions} [options]
             * @returns {void}
             */

            /**
             * @typedef {() => void} RemoveListenerFn
             */

            /**
             * @typedef {(evt: string, cb: EventListenerOrEventListenerObject, opts?: boolean|AddEventListenerOptions) => RemoveListenerFn} AddEventListenerShim
             */

            Object.defineProperty(document, 'addEventListener', {
              configurable: true,
              enumerable: false,
              writable: false,
              /**
               * Shimmed addEventListener that delegates to original handlers and
               * returns a cleanup function for safe removal.
               * @type {AddEventListenerShim}
               * @param {string} evt
               * @param {EventListenerOrEventListenerObject} cb
               * @param {boolean|AddEventListenerOptions=} opts
               */
              value: function(evt, cb, opts) {
              try {
                if (_origAdd) {
                _origAdd(evt, cb, opts);
                } else if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
                window.addEventListener(evt, cb, opts);
                }
              } catch (/** @type {any} */ err) {
                // swallow delegate errors but ensure cleanup is returned
              }
              return /** @type {RemoveListenerFn} */ (function() {
                try {
                if (_origRemove) {
                  _origRemove(evt, cb, opts);
                } else if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
                  window.removeEventListener(evt, cb, opts);
                }
                } catch (/** @type {any} */ e) {
                /* ignore */
                }
              });
              }
            });
          }
        } catch (/** @type {any} */ e) {
          // ignore
        }

// ============================================
// Advanced Features State Management
// ============================================

const AdvancedFeatures = (() => {
  // Filter presets storage
  const filterPresets = new Map();
  const PRESETS_KEY = 'parser_filter_presets';
  
  // Session state tracking
  let currentSessionId = null;
  let directUrlDraftBySession = new Map();
  
  // Load presets from localStorage
  function loadPresets() {
    try {
      const stored = localStorage.getItem(PRESETS_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        Object.entries(parsed).forEach(([name, filters]) => {
          filterPresets.set(name, filters);
        });
      }
    } catch (/** @type {any} */ err) {
      console.warn('[Presets] Failed to load:', err);
    }
  }
  
  // Save presets to localStorage
  function savePresets() {
    try {
      const obj = {};
      filterPresets.forEach((filters, name) => {
        obj[name] = filters;
      });
      localStorage.setItem(PRESETS_KEY, JSON.stringify(obj));
    } catch (err) {
      console.warn('[Presets] Failed to save:', err);
    }
  }
  
  // Get current filter state
  function getCurrentFilters() {
    const confEl = document.getElementById('filterConfidence');
    const stateEl = document.getElementById('filterState');
    const levelEl = document.getElementById('filterLevel');
    const confidence = (confEl instanceof HTMLInputElement || confEl instanceof HTMLSelectElement || confEl instanceof HTMLTextAreaElement) ? confEl.value : '0';
    const state = (stateEl instanceof HTMLInputElement || stateEl instanceof HTMLSelectElement || stateEl instanceof HTMLTextAreaElement) ? stateEl.value : '';
    const level = (levelEl instanceof HTMLInputElement || levelEl instanceof HTMLSelectElement || levelEl instanceof HTMLTextAreaElement) ? levelEl.value : '';
    return { confidence, state, level };
  }
  
  // Apply filters
  /**
   * @typedef {Object} FilterValues
   * @property {string|number} [confidence]
   * @property {string} [state]
   * @property {string} [level]
   *
   * @typedef {HTMLInputElement|HTMLSelectElement|HTMLTextAreaElement|null} MaybeFormElement
   *
   * @callback ApplyFiltersFn
   * @param {FilterValues} filters
   * @returns {void}
   */

  /** @type {ApplyFiltersFn} */
  function applyFilters(filters) {
    const { confidence, state, level } = filters;
    /** @type {MaybeFormElement} */
    const confEl = /** @type {MaybeFormElement} */ (document.getElementById('filterConfidence'));
    /** @type {MaybeFormElement} */
    const stateEl = /** @type {MaybeFormElement} */ (document.getElementById('filterState'));
    /** @type {MaybeFormElement} */
    const levelEl = /** @type {MaybeFormElement} */ (document.getElementById('filterLevel'));
    
    if (confEl instanceof HTMLInputElement || confEl instanceof HTMLSelectElement || confEl instanceof HTMLTextAreaElement) confEl.value = String(typeof confidence === 'number' ? confidence : (confidence ?? '0'));
    if (stateEl instanceof HTMLInputElement || stateEl instanceof HTMLSelectElement || stateEl instanceof HTMLTextAreaElement) stateEl.value = String(state ?? '');
    if (levelEl instanceof HTMLInputElement || levelEl instanceof HTMLSelectElement || levelEl instanceof HTMLTextAreaElement) levelEl.value = String(level ?? '');
    
    // Update confidence label
    /** @type {HTMLElement | null} */
    const labelEl = document.getElementById('filterConfidenceValue');
    if (labelEl && (confEl instanceof HTMLInputElement || confEl instanceof HTMLSelectElement || confEl instanceof HTMLTextAreaElement)) labelEl.textContent = confEl.value + '%+';
    
    // Trigger filter update (call via any-cast to avoid TS window property errors)
    try { (/** @type {any} */ (window)).applyLogFilters && (/** @type {any} */ (window)).applyLogFilters(); } catch (/** @type {any} */ err) { /* ignore */ }
  }
  
  return {
    filterPresets,
    loadPresets,
    savePresets,
    getCurrentFilters,
    applyFilters,
    directUrlDraftBySession,
    get currentSessionId() { return currentSessionId; },
    set currentSessionId(id) { currentSessionId = id; }
  };
})();

// ============================================
// PHASE 2: Error Handling & Recovery
// ============================================

const ErrorBoundary = (() => {
  const errorLog = [];
  const maxErrors = 50;
  
  /** @param {any} error */
  function logError(error, context = '') {
    const timestamp = new Date().toISOString();
    const errorInfo = {
      timestamp,
      message: error?.message || String(error),
      context,
      stack: error?.stack || '',
      recovered: false
    };
    
    errorLog.push(errorInfo);
    if (errorLog.length > maxErrors) {
      errorLog.shift();
    }
    
    console.error(`[ErrorBoundary] ${context}:`, error);
    return errorInfo;
  }
  
  /**
   * @template T
   * @callback SyncFunction
   * @returns {T}
   */

  /**
   * @typedef {Object} LoggedErrorInfo
   * @property {string} timestamp
   * @property {string} message
   * @property {string} context
   * @property {string} stack
   * @property {boolean} recovered
   */

  /**
   * Execute a synchronous function with error capture, logging and a fallback.
   * @template T
   * @param {SyncFunction<T>} fn - Synchronous function to execute.
   * @param {string} [context='anonymous'] - Short label for logging context.
   * @param {T|null} [fallback=null] - Value to return when execution fails.
   * @returns {T|null}
   */
  function safeExecute(fn, context = 'anonymous', fallback = null) {
    try {
      return fn();
    } catch (/** @type {any} */ error) {
      /** @type {LoggedErrorInfo} */
      const logged = logError(error, context);
      logged.recovered = true;
      showErrorNotification(error, context);
      return fallback;
    }
  }
  
  /**
   * @callback AsyncOperation
   * @returns {Promise<any>}
   */

  /**
   * Execute an async function with error capture and notification.
   * @param {AsyncOperation} asyncFn - Asynchronous function to execute.
   * @param {string} [context='async_operation'] - Short label for logging context.
   * @returns {Promise<any|void>} Resolves with the asyncFn result or void on error.
   */
  async function safeAsync(asyncFn, context = 'async_operation') {
    try {
      return await asyncFn();
    } catch (/** @type {any} */ error) {
      logError(error, context);
      showErrorNotification(error, context);
      // swallow and return void on error (preserve previous behavior)
      return;
    }
  }
  
  /**
   * @typedef {Object} ErrorLike
   * @property {string} [message]
   * @property {string} [stack]
   */

  /**
   * @typedef {HTMLDivElement} ToastElement
   */

  /**
   * Display a brief error notification toast.
   * @param {Error|ErrorLike|null|undefined} error
   * @param {string} context
   * @returns {void}
   */
  function showErrorNotification(error, context) {
    /** @type {string} */
    const message = `Error in ${context}: ${error?.message || 'Unknown error'}`;
    /** @type {ToastElement} */
    const toast = /** @type {ToastElement} */ (document.createElement('div'));
    toast.className = 'error-toast notification-toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    // setTimeout returns number in browsers
    setTimeout(() => toast.remove(), 5000);
  }
  
  function getErrorLog() {
    return [...errorLog];
  }
  
  function clearErrorLog() {
    errorLog.length = 0;
  }
  
  return {
    safeExecute,
    safeAsync,
    logError,
    getErrorLog,
    clearErrorLog
  };
})();

// ============================================
// PHASE 2: Performance Utilities
// ============================================

// Debouncing utility for search/filter inputs
/**
 * @callback AnyFunction
 * @param {...any} args
 * @returns any
 */

/**
 * @typedef {number|undefined|null} TimeoutId
 */

/**
 * @callback DebouncedFunction
 * @param {...any} args
 * @returns void
 */

/**
 * Create a debounced wrapper for a function.
 * @param {AnyFunction} fn
 * @param {number} delay
 * @returns {DebouncedFunction}
 */
function debounce(fn, delay) {
  /** @type {TimeoutId} */
  let timeoutId = undefined;
  return function(...args) {
    try { if (typeof timeoutId !== 'undefined' && timeoutId !== null) clearTimeout(timeoutId); } catch (/** @type {any} */ e) {}
    timeoutId = /** @type {TimeoutId} */ (/** @type {any} */ (setTimeout(() => fn.apply(this, args), delay)));
  };
}

// Virtual scrolling manager for large option lists
const VirtualScroll = (() => {
  let isEnabled = false;
  let allItems = [];
  let visibleRange = { start: 0, end: 0 };
  let scrollTop = 0;
  let containerHeight = 0;
  
  /**
   * @typedef {Object} VirtualScrollContainer
   * @property {number} clientHeight
   */

  /**
   * Enable virtual scrolling when items exceed the threshold.
   * @param {Array<any>} items
   * @param {VirtualScrollContainer} container
   * @returns {boolean}
   */
  function enable(items, container) {
    if (items.length < CONFIG.virtualScrollThreshold) {
      isEnabled = false;
      return false;
    }
    
    isEnabled = true;
    allItems = items;
    containerHeight = container.clientHeight || 400;
    
    const itemsPerPage = Math.ceil(containerHeight / CONFIG.virtualScrollItemHeight);
    visibleRange.start = 0;
    visibleRange.end = itemsPerPage + CONFIG.virtualScrollBuffer;
    
    return true;
  }
  
  function getVisibleItems() {
    if (!isEnabled) return allItems;
    return allItems.slice(visibleRange.start, visibleRange.end);
  }
  
  /**
   * @typedef {Object} VirtualScrollRange
   * @property {number} start
   * @property {number} end
   */

  /**
   * Update visible window based on new scroll top position.
   * @param {number} newScrollTop
   * @returns {void}
   */
  function updateScroll(newScrollTop) {
    if (!isEnabled) return;

    /** @type {number} */
    scrollTop = newScrollTop;

    /** @type {number} */
    const itemsPerPage = Math.ceil(containerHeight / CONFIG.virtualScrollItemHeight);

    /** @type {number} */
    const startIdx = Math.floor(scrollTop / CONFIG.virtualScrollItemHeight);

    /** @type {VirtualScrollRange} */
    visibleRange.start = Math.max(0, startIdx - CONFIG.virtualScrollBuffer);
    visibleRange.end = Math.min(allItems.length, startIdx + itemsPerPage + CONFIG.virtualScrollBuffer);
  }
  
  function getTotalHeight() {
    return isEnabled ? allItems.length * CONFIG.virtualScrollItemHeight : 0;
  }
  
  function getOffsetY() {
    return isEnabled ? visibleRange.start * CONFIG.virtualScrollItemHeight : 0;
  }
  
  function reset() {
    isEnabled = false;
    allItems = [];
    visibleRange = { start: 0, end: 0 };
    scrollTop = 0;
  }
  
  return { 
    enable, 
    getVisibleItems, 
    updateScroll, 
    getTotalHeight, 
    getOffsetY, 
    reset, 
    get isEnabled() { return isEnabled; } 
  };
})();

// ============================================
// PHASE 2: Table Preview (P2.3)
// ============================================

const TablePreview = (() => {
  /* PreviewRow/PreviewData typedefs consolidated at top of file. */
  /**
   * @interface RenderPreviewOptions
   * @property {number} [maxRows]
   */

  /**
   * Render a simple HTML table preview for an array of row objects.
   * @param {PreviewData} data
   * @param {number} [maxRows=5]
   * @returns {string}
   */
  function renderPreview(data, maxRows = 5) {
    if (!Array.isArray(data) || !data.length) return '<p class="text-muted">No data to preview</p>';

    /** @type {PreviewRow[]} */
    const rows = data.slice(0, maxRows);
    /** @type {string[]} */
    const keys = Object.keys(rows[0] || {});

    let html = '<table class="preview-table"><thead><tr>';
    keys.forEach(k => html += `<th>${escapeHtml(k)}</th>`);
    html += '</tr></thead><tbody>';

    rows.forEach(/** @param {PreviewRow} row */ (row) => {
      html += '<tr>';
      keys.forEach(k => html += `<td>${escapeHtml(String(row[k] || ''))}</td>`);
      html += '</tr>';
    });

    html += '</tbody></table>';
    if (data.length > maxRows) html += `<p class="text-muted small">${data.length - maxRows} more rows...</p>`;

    return html;
  }
  
  /* PreviewRow/PreviewData typedefs consolidated at top of file. */
  /**
   * @typedef {HTMLDivElement & { _onKey?: (e: KeyboardEvent) => void }} PreviewModalElement
   */

  /**
   * Show a modal with a small table preview.
   * @param {string} title
   * @param {PreviewData} data
   * @returns {void}
   */
  function showPreviewModal(title, data) {
    const modal = /** @type {PreviewModalElement} */ (document.createElement('div'));
    modal.className = 'modal preview-modal';
    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <h3>${escapeHtml(title)}</h3>
          <button class="modal-close" aria-label="Close preview">×</button>
        </div>
        <div class="modal-body">
          ${renderPreview(data)}
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary preview-continue">Continue</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    /** @type {Element | null} */
    const closeBtn = modal.querySelector('.modal-close');
    if (closeBtn instanceof Element) closeBtn.addEventListener('click', () => modal.remove());
    /** @type {Element | null} */
    const cont = modal.querySelector('.preview-continue');
    if (cont instanceof Element) cont.addEventListener('click', () => modal.remove());
  }
  
  return { renderPreview, showPreviewModal };
})();

// ============================================
// PHASE 2: Session Restore (P2.4)
// ============================================

const SessionRestore = (() => {
  const RESTORE_KEY = 'smartElectionsRestore';
  
  /**
   * @typedef {Object} SessionRestoreState
   * @property {number} timestamp
   * @property {string|null} sessionId
   * @property {Array<string|null>} urls
   * @property {Array<string>} searches
   */

  /**
   * Save a lightweight restore snapshot to sessionStorage.
   * @param {any} data
   * @returns {void}
   */
  function saveState(data) {
    try {
      /** @type {SessionRestoreState} */
      const state = {
          timestamp: Date.now(),
          sessionId: /** @type {string|null} */ (currentSessionId || null),
          urls: /** @type {Array<string|null>} */ (Array.from(document.querySelectorAll('[data-url]')).map(el => (el instanceof Element) ? el.getAttribute('data-url') : null)),
          searches: /** @type {string[]} */ (Array.from(document.querySelectorAll('input[type="search"]')).map(el => (el instanceof HTMLInputElement) ? el.value : '')),
        };
      sessionStorage.setItem(RESTORE_KEY, JSON.stringify(state));
    } catch (/** @type {any} */ e) {
      ErrorBoundary.logError(e, 'SessionRestore.saveState');
    }
  }
  
  function hasRestoreData() {
    const data = sessionStorage.getItem(RESTORE_KEY);
    return data && JSON.parse(data).timestamp > (Date.now() - 3600000); // 1 hour
  }
  
  function showRestoreBanner() {
    const data = sessionStorage.getItem(RESTORE_KEY);
    if (!data) return;
    
    const state = JSON.parse(data);
    const banner = document.createElement('div');
    banner.className = 'restore-banner';
    banner.innerHTML = `
      <div class="restore-content">
        <span>📋 Restore session from ${new Date(state.timestamp).toLocaleTimeString()}?</span>
        <button class="btn btn-sm btn-primary" id="btnRestoreYes">Restore</button>
        <button class="btn btn-sm btn-secondary" id="btnRestoreNo">Dismiss</button>
      </div>
    `;
    document.body.prepend(banner);
    
    const btnYes = document.getElementById('btnRestoreYes');
    const btnNo = document.getElementById('btnRestoreNo');
    if (btnYes instanceof Element) btnYes.addEventListener('click', () => {
      // Restore state
      banner.remove();
      showToast('Session restored', 'success');
    });
    if (btnNo instanceof Element) btnNo.addEventListener('click', () => {
      sessionStorage.removeItem(RESTORE_KEY);
      banner.remove();
    });
  }
  
  return { saveState, hasRestoreData, showRestoreBanner };
})();

// ============================================
// PHASE 2: Accessibility (P2.5)
// ============================================

function enhanceAccessibility() {
  // Add keyboard navigation (Enter, Escape)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      hidePrompt();
    }
    if (e.key === 'Enter' && e.ctrlKey && activePromptOptions.length === 1) {
      submitPrompt(activePromptOptions[0].index);
    }
  });
  
  // Add ARIA labels to dynamic content
  /**
   * @callback AddAriaLabelFn
   * @param {string} selector
   * @param {string} label
   * @returns {void}
   */

  /**
   * Lightweight interface describing elements which may accept aria attributes.
   * @typedef {HTMLElement} AriaElement
   */

  /**
   * Set an accessible aria-label on matching elements if one is not already present.
   * @type {AddAriaLabelFn}
   */
  const addAriaLabel = /** @type {AddAriaLabelFn} */ (function(selector, label) {
    /** @type {NodeListOf<Element>} */
    const nodes = document.querySelectorAll(selector);
    nodes.forEach(/** @param {Element} el */ (el) => {
      try {
        const maybeEl = /** @type {AriaElement} */ (el);
        if (!maybeEl.getAttribute('aria-label')) maybeEl.setAttribute('aria-label', label);
      } catch (/** @type {any} */ e) {
        // ignore elements that cannot accept attributes
      }
    });
  });
  
  addAriaLabel('.prompt-option', 'Contest option');
  addAriaLabel('.prompt-bundle-toggle', 'Expand/collapse bundle');
  addAriaLabel('.badge', 'Metadata badge');
  addAriaLabel('[data-tab]', 'Tab button');
  
  // Mark live regions
  const liveRegions = ['#logOutput', '#promptOptions', '#sessionsList'];
  liveRegions.forEach(sel => {
    const el = $(sel);
    if (el && !el.getAttribute('aria-live')) {
      el.setAttribute('aria-live', 'polite');
      el.setAttribute('aria-atomic', 'false');
    }
  });
  
  console.log('[Accessibility] Enhanced with keyboard nav and ARIA labels');
}

// Mobile sidebar toggle and touch-to-close support
function initSidebarMobile() {
  try {
    if (__tl_window.__tl_sidebarUnified) return;
    const leftSidebar = document.querySelector('.sidebar-left, #sidebar');
    const rightSidebar = document.querySelector('.sidebar-right');
    const backdrop = document.querySelector('.sidebar-backdrop') || document.querySelector('.mobile-sidebar-overlay');
    
    if (!leftSidebar && !rightSidebar) return;
    
    // Process both sidebars
    [leftSidebar, rightSidebar].forEach(sidebar => {
      if (!sidebar) return;
      
      const isRightSidebar = sidebar.classList.contains('sidebar-right');
      
      let touchStartX = 0;
      let touchStartY = 0;
      let touchCurrentX = 0;
      let tracking = false;
      let swipeTarget = null;
      
      // INTERACTIVE ELEMENTS that should NOT trigger swipe-to-close
      const interactiveSelectors = [
        'input',
        'textarea',
        'select',
        'button',
        'a',
        '.url-sidebar-item',
        '.source-card',
        '.control-group',
        '.form-control',
        '.btn',
        '[role="button"]',
        '[tabindex="0"]'
      ].join(', ');
      
      const openSidebar = () => {
        sidebar.classList.add('sidebar-open');
        if (backdrop) backdrop.classList.add('visible');
        document.body.style.overflow = 'hidden';
        document.body.style.position = 'fixed';
        document.body.style.width = '100%';
      };
      
      const closeSidebar = () => {
        sidebar.classList.remove('sidebar-open');
        if (backdrop) backdrop.classList.remove('visible');
        document.body.style.overflow = '';
        document.body.style.position = '';
        document.body.style.width = '';
      };
      
      sidebar.addEventListener('touchstart', (ev) => {
        if (!sidebar.classList.contains('sidebar-open')) return;
        
        const t = /** @type {TouchEvent} */(ev).touches && /** @type {TouchEvent} */(ev).touches[0];
        if (!t) return;
        
        // Check if touch started on interactive element
        swipeTarget = /** @type {TouchEvent} */(ev).target;
        const isInteractive = /** @type {HTMLElement} */(swipeTarget).closest(interactiveSelectors);
        
        // Only track swipe if NOT on interactive element
        if (!isInteractive) {
          touchStartX = t.clientX;
          touchStartY = t.clientY;
          tracking = true;
        } else {
          tracking = false;
        }
      }, { passive: true });
      
      sidebar.addEventListener('touchmove', (ev) => {
        if (!tracking) return;
        
        const t = /** @type {TouchEvent} */(ev).touches && /** @type {TouchEvent} */(ev).touches[0];
        if (!t) return;
        
        touchCurrentX = t.clientX;
        const dx = touchCurrentX - touchStartX;
        const dy = t.clientY - touchStartY;
        
        // Only prevent scroll if swipe is clearly horizontal
        const isHorizontalSwipe = Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 15;
        
        if (isHorizontalSwipe) {
          // Check swipe direction matches sidebar position
          const swipingLeft = dx < 0;
          const swipingRight = dx > 0;
          
          const shouldClose = (isRightSidebar && swipingRight) || (!isRightSidebar && swipingLeft);
          
          if (shouldClose) {
            ev.preventDefault(); // Only prevent if we're going to close
          }
        }
      }, { passive: false });
      
      sidebar.addEventListener('touchend', () => {
        if (!tracking) return;
        
        tracking = false;
        const dx = touchCurrentX - touchStartX;
        
        // Threshold: 80px (increased for reliability)
        if (Math.abs(dx) > 80) {
          const swipingLeft = dx < 0;
          const swipingRight = dx > 0;
          
          const shouldClose = (isRightSidebar && swipingRight) || (!isRightSidebar && swipingLeft);
          
          if (shouldClose) {
            closeSidebar();
          }
        }
        
        touchStartX = touchCurrentX = touchStartY = 0;
        swipeTarget = null;
      }, { passive: true });
      
      // Toggle button handling for this sidebar
      const toggle = isRightSidebar 
        ? document.getElementById('btnToggleRightSidebar')
        : document.querySelector('.sidebar-toggle');
      
      if (toggle) {
        toggle.addEventListener('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (sidebar.classList.contains('sidebar-open')) {
            closeSidebar();
          } else {
            openSidebar();
          }
        });
      }
      
      // Initialize state
      if (window.innerWidth <= 768) {
        closeSidebar();
      }
    });
    
    // Backdrop click to close
    if (backdrop) {
      backdrop.addEventListener('click', () => {
        const openSidebar = document.querySelector('.sidebar-left.sidebar-open, .sidebar-right.sidebar-open');
        if (openSidebar) {
          openSidebar.classList.remove('sidebar-open');
          backdrop.classList.remove('visible');
          document.body.style.overflow = '';
          document.body.style.position = '';
          document.body.style.width = '';
        }
      });
    }
    
  } catch (e) {
    ErrorBoundary.logError(e, 'initSidebarMobile');
  }
}

// Initialize scroll indicators for sidebars
function initScrollIndicators() {
  const sidebars = document.querySelectorAll('.sidebar-left, .sidebar-right');
  
  sidebars.forEach(sidebar => {
    function updateScrollIndicator() {
      const hasScroll = sidebar.scrollHeight > sidebar.clientHeight;
      sidebar.classList.toggle('has-scroll', hasScroll);
    }
    
    sidebar.addEventListener('scroll', updateScrollIndicator);
    window.addEventListener('resize', updateScrollIndicator);
    updateScrollIndicator();
  });
}

// Initialize results preview bar functionality
function initResultsPreviewBar() {
  const previewBar = document.getElementById('resultsPreviewBar');
  const resultCountBadge = document.getElementById('resultCountBadge');
  const lastUpdatedPreview = document.getElementById('lastUpdatedPreview');
  const resultsGrid = document.getElementById('resultsGrid');
  
  if (!previewBar) return;
  
  // Sync result count and last updated time
  function updateResultsPreview() {
    if (resultsGrid && resultCountBadge) {
      const resultCards = resultsGrid.querySelectorAll('.result-card, .card');
      const count = resultCards.length;
      resultCountBadge.textContent = `${count} result${count !== 1 ? 's' : ''}`;
    }
    
    if (lastUpdatedPreview) {
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      lastUpdatedPreview.textContent = timeStr;
    }
  }
  
  // Auto-expand when parser runs (when pulse loader shows)
  const pulseLoader = document.getElementById('pulseLoader');
  if (pulseLoader) {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
          const isLoading = !pulseLoader.classList.contains('hidden');
          if (isLoading && !previewBar.open) {
            previewBar.open = true;
          }
        }
      });
    });
    observer.observe(pulseLoader, { attributes: true });
  }
  
  // Watch for results grid changes
  if (resultsGrid) {
    const gridObserver = new MutationObserver(() => {
      updateResultsPreview();
    });
    gridObserver.observe(resultsGrid, { childList: true, subtree: true });
  }
  
  // Initial update
  updateResultsPreview();
  
  // Update time every minute
  setInterval(updateResultsPreview, 60000);
}

// Initialize mobile sidebar handlers on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  initSidebarMobile();
  initScrollIndicators();
  initResultsPreviewBar();
});

// Calculate actual viewport height (accounts for mobile browser chrome)
function setViewportHeight() {
  const vh = window.innerHeight * 0.01;
  document.documentElement.style.setProperty('--vh', `${vh}px`);
}

// Debounced resize handler for viewport height and sidebar state
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    setViewportHeight();
    // Re-check sidebar state
    const sidebar = document.getElementById('sidebar');
    if (sidebar && window.innerWidth > 768) {
      sidebar.classList.remove('sidebar-open');
      const backdrop = document.querySelector('.sidebar-backdrop') || document.querySelector('.mobile-sidebar-overlay');
      if (backdrop) backdrop.classList.remove('visible');
      document.body.style.overflow = '';
      document.body.style.position = '';
      document.body.style.width = '';
    }
  }, 250);
});

// Update on orientation change
window.addEventListener('orientationchange', setViewportHeight);

// Initialize viewport height
setViewportHeight();

// Show flagged details in a dedicated modal with simple filters
/**
 * @typedef {Object} FlaggedItem
 * @property {string} [url]
 * @property {string} [status]
 * @property {string|string[]} [reasons]
 * @property {Object<string, any>} [metadata_excerpt]
 * @property {number|string} [timestamp]
 */

/**
 * @typedef {Object} SortState
 * @property {string} key
 * @property {number} dir
 */

/**
 * @param {FlaggedItem[]} flagged
 * @param {string} report_path
 */
function showFlaggedModal(flagged, report_path) {
  try {
    const existing = document.getElementById('flaggedModal');
    if (existing) existing.remove();
    const reportName = report_path ? report_path.replace(/\\/g, '/').split('/').pop() : '';
    // Persistent sort state per report
    const persistedKey = reportName ? `flagged_sort_${reportName}` : 'flagged_sort_global';
    /** @type {SortState} */
    let currentSort = { key: '', dir: 1 };
    try {
      const p = localStorage.getItem(persistedKey);
      if (p) {
        const parsed = JSON.parse(p);
        if (parsed && parsed.key) currentSort = parsed;
      }
    } catch (e) {
      /* ignore */
    }

    const modal = document.createElement('div');
    modal.id = 'flaggedModal';
    modal.className = 'flagged-modal';
    modal.innerHTML = `
      <div class="flagged-modal-content">
        <div class="flagged-modal-header">
          <h3>Flagged Details (${flagged.length})</h3>
          <div class="flagged-controls">
            <input id="flaggedFilter" placeholder="Filter by URL or reason" class="input-sm" />
            <input id="flaggedMinConf" type="number" min="0" max="1" step="0.01" placeholder="Min confidence" class="input-sm w-110" />
            <button id="flaggedExportCSV" class="btn btn-sm">Export CSV</button>
            <button id="flaggedExportJSON" class="btn btn-sm">Export JSON</button>
            ${reportName ? `<a class="btn btn-sm btn-outline" id="flaggedDownload" href="/download_fs?root=output&path=reports&name=${encodeURIComponent(reportName)}" target="_blank" rel="noopener">Download report</a>` : ''}
            <button id="flaggedClose" class="btn btn-sm">Close</button>
          </div>
        </div>
        <div class="flagged-modal-body">
          <table class="flagged-table"><thead><tr><th data-key="url">URL</th><th data-key="status">Status</th><th data-key="reasons">Reasons</th><th data-key="confidence">Confidence</th><th data-key="metadata">Metadata</th></tr></thead><tbody id="flaggedTableBody"></tbody></table>
        </div>
      </div>
    `;
    document.body.appendChild(modal);

    // close on ESC
    /** @param {KeyboardEvent} ev */
    const escHandler = (ev) => { if (ev.key === 'Escape') { modal.remove(); document.removeEventListener('keydown', escHandler); } };
    document.addEventListener('keydown', escHandler);

    /** @type {HTMLElement | null} */
    const tbody = modal.querySelector('#flaggedTableBody');

    /**
     * @param {FlaggedItem} item
     * @param {string} key
     * @returns {string|number}
     */
    function getSortValue(item, key) {
      if (key === 'url') return (item.url || '').toLowerCase();
      if (key === 'status') return (item.status || '').toLowerCase();
      if (key === 'reasons') return (Array.isArray(item.reasons) ? item.reasons.join(' ') : (item.reasons || '')).toLowerCase();
      if (key === 'confidence') {
        const m = item.metadata_excerpt || {};
        return Number(m.extraction_confidence ?? m.quality_metrics?.extraction_confidence ?? NaN) || 0;
      }
      if (key === 'metadata') return JSON.stringify(item.metadata_excerpt || {}).toLowerCase();
      return '';
    }

    /**
     * Render provided rows into table body.
     * @param {FlaggedItem[]} list
     */
    function renderRows(list) {
      if (!tbody) return;
      tbody.innerHTML = '';
      // apply sort
      const rowsList = Array.isArray(list) ? [...list] : [];
      if (currentSort.key) {
        rowsList.sort((a,b) => {
          const va = getSortValue(a, currentSort.key);
          const vb = getSortValue(b, currentSort.key);
          if (va < vb) return -1 * currentSort.dir;
          if (va > vb) return 1 * currentSort.dir;
          return 0;
        });
      }

      for (const f of rowsList) {
        const reasons = Array.isArray(f.reasons) ? f.reasons.join(', ') : (f.reasons || '');
        const metaObj = f.metadata_excerpt || {};
        const confVal = (metaObj && (metaObj.extraction_confidence || metaObj.quality_metrics?.extraction_confidence || metaObj.extraction_confidence === 0)) ? (Number(metaObj.extraction_confidence ?? metaObj.quality_metrics?.extraction_confidence) ) : '';
        const metaStrJson = JSON.stringify(metaObj || {});
        const metaEsc = escapeHtml(metaStrJson);
        const metaDataAttr = encodeURIComponent(metaStrJson);
        const urlText = escapeHtml(f.url || f['url'] || '');
        const status = escapeHtml(f.status || '');
        const tr = document.createElement('tr');
        // build per-row extra actions (Open output / Jump to CSV row) when metadata provides paths/indexes
        const rowMeta = f.metadata_excerpt || {};
        let openLinkHtml = '';
        const possibleFile = rowMeta.output_file || rowMeta.output_file_path || rowMeta.output_path || rowMeta.output_filename || '';
        if (possibleFile) {
          const base = String(possibleFile).split(/[\\\\\/]/).pop();
          openLinkHtml = `<a class="btn btn-xs" href="/download_fs?root=output&path=&name=${encodeURIComponent(base)}" target="_blank" rel="noopener">Open output</a>`;
        }
        let jumpBtnHtml = '';
        const rowIndex = rowMeta.output_row || rowMeta.output_row_index || rowMeta.row_index || '';
        if (rowIndex !== '' && rowIndex !== undefined && rowIndex !== null) {
          jumpBtnHtml = `<button class="btn btn-xs jump-row" data-row="${escapeHtml(String(rowIndex))}">Jump to CSV row ${escapeHtml(String(rowIndex))}</button>`;
        }

        tr.innerHTML = `
          <td><a href="${escapeHtml(f.url || '')}" target="_blank" rel="noopener">${urlText}</a></td>
          <td>${status}</td>
          <td>${escapeHtml(reasons)}</td>
          <td>${confVal !== '' ? Number(confVal).toFixed(2) : ''}</td>
          <td><div class="flex-col-gap-start">
            <div class="flex-row-gap-center">${openLinkHtml}${jumpBtnHtml}<button class="copy-meta btn btn-xs" data-meta="${metaDataAttr}">Copy</button></div>
            <pre class="small muted pre-wrap no-margin">${metaEsc}</pre>
          </div></td>
        `;
        tbody.appendChild(tr);
      }

      // attach copy handlers
      tbody.querySelectorAll('.copy-meta').forEach(btn => {
        btn.addEventListener('click', () => {
          const meta = (btn instanceof HTMLElement) ? decodeURIComponent(btn.getAttribute('data-meta') || '') : '';
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(meta).then(() => showToast('Metadata copied', 'success'), () => showToast('Copy failed', 'error'));
          } else {
            const ta = document.createElement('textarea');
            ta.value = meta;
            document.body.appendChild(ta);
            ta.select();
            try { document.execCommand('copy'); showToast('Metadata copied', 'success'); } catch (e) { showToast('Copy failed', 'error'); }
            ta.remove();
          }
        });
      });
    }

    // delegated handlers for copy and jump actions
    /** @param {Event} ev */
    tbody.addEventListener('click', function(ev) {
      const target = ev.target;
      if (!target || !(target instanceof Element)) return;
      if (target.classList.contains('copy-meta')) {
        const metaRaw = target.getAttribute('data-meta');
        try {
          const obj = JSON.parse(metaRaw);
          const text = JSON.stringify(obj, null, 2);
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(()=>{
              target.textContent = 'Copied';
              setTimeout(()=> target.textContent = 'Copy', 1200);
            }).catch(()=>{ promptCopyFallback(text, target); });
          } else {
            promptCopyFallback(text, target);
          }
        } catch (e) {
          // noop
        }
      }
      if (target.classList.contains('jump-row')) {
        const rowIdx = target.getAttribute('data-row');
        // try to find associated metadata for output file
        const tr = target.closest('tr');
        const copyBtn = tr && tr.querySelector('.copy-meta');
        let meta = null;
        if (copyBtn) {
          try { meta = JSON.parse(copyBtn.getAttribute('data-meta')); } catch (e) { meta = null; }
        }
        if (meta && (meta.output_file || meta.output_file_path || meta.output_filename)) {
          const possibleFile = meta.output_file || meta.output_file_path || meta.output_filename || '';
          const base = String(possibleFile).split(/[\\\/]/).pop();
          // call server to locate viewer page (build index if needed)
          fetch(`/csv_locate?root=output&path=&name=${encodeURIComponent(base)}&row=${encodeURIComponent(rowIdx)}`)
            .then(r => r.json())
            .then(j => {
              if (j && j.viewer) {
                window.open(j.viewer, '_blank');
              } else {
                // fallback: open the CSV normally
                const dl = `/download_fs?root=output&path=&name=${encodeURIComponent(base)}`;
                window.open(dl, '_blank');
                setTimeout(()=> alert(`Opened output file. Search for row ${rowIdx} in the downloaded CSV.`), 200);
              }
            }).catch(()=>{
              const dl = `/download_fs?root=output&path=&name=${encodeURIComponent(base)}`;
              window.open(dl, '_blank');
              setTimeout(()=> alert(`Opened output file. Search for row ${rowIdx} in the downloaded CSV.`), 200);
            });
        } else if (reportName) {
          const rpt = `/download_fs?root=output&path=reports&name=${encodeURIComponent(reportName)}`;
          window.open(rpt, '_blank');
          setTimeout(()=> alert(`Report opened. If an output CSV exists, open it and search for row ${rowIdx}.`), 200);
        } else {
          alert(`Row: ${rowIdx}. Open the output CSV and search for this row index.`);
        }
      }
    });

    /**
     * Fallback copy prompt
     * @param {string} text
     * @param {Element} targetBtn
     */
    function promptCopyFallback(text, targetBtn) {
      const ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); targetBtn.textContent = 'Copied'; setTimeout(()=> targetBtn.textContent = 'Copy', 1200); } catch (e) { alert('Copy failed — open the metadata and copy manually.'); }
      ta.remove();
    }

    // initial render
    renderRows(flagged);

    // filtering
    /** @type {HTMLInputElement | null} */
    const filterInput = modal.querySelector('#flaggedFilter');
    /** @type {HTMLInputElement | null} */
    const confInput = modal.querySelector('#flaggedMinConf');
    function applyFilter() {
      const q = (filterInput instanceof HTMLInputElement ? (filterInput.value || '') : '').toLowerCase().trim();
      const minConf = parseFloat(confInput instanceof HTMLInputElement ? confInput.value : 'NaN');
      const filtered = flagged.filter(f => {
        let ok = true;
        if (q) {
          const hay = `${f.url || ''} ${Array.isArray(f.reasons) ? f.reasons.join(' ') : (f.reasons || '')} ${JSON.stringify(f.metadata_excerpt || {})}`.toLowerCase();
          ok = hay.indexOf(q) !== -1;
        }
        if (ok && !isNaN(minConf)) {
          const meta = f.metadata_excerpt || {};
          const confVal = Number(meta.extraction_confidence ?? meta.quality_metrics?.extraction_confidence ?? NaN);
          if (!isNaN(confVal)) ok = confVal >= minConf;
          else ok = false;
        }
        return ok;
      });
      renderRows(filtered);
    }
    filterInput.addEventListener('input', debounce(applyFilter, 150));
    confInput.addEventListener('input', debounce(applyFilter, 150));

    // header sort handlers + UI indicators
    function updateHeaderIndicators() {
      modal.querySelectorAll('.flagged-table thead th').forEach(th => {
        if (!(th instanceof HTMLElement)) return;
        const k = th.getAttribute('data-key') || '';
        if (k && currentSort.key === k) {
          th.classList.add('active');
          th.setAttribute('data-sort-dir', String(currentSort.dir));
        } else {
          th.classList.remove('active');
          th.removeAttribute('data-sort-dir');
        }
        th.style.cursor = k ? 'pointer' : '';
      });
    }

    modal.querySelectorAll('.flagged-table thead th').forEach(th => {
      if (!(th instanceof HTMLElement)) return;
      th.addEventListener('click', () => {
        const key = th.getAttribute('data-key');
        if (!key) return;
        if (currentSort.key === key) currentSort.dir *= -1; else { currentSort.key = key; currentSort.dir = 1; }
        // persist
        try { localStorage.setItem(persistedKey, JSON.stringify(currentSort)); } catch (e) {}
        updateHeaderIndicators();
        applyFilter();
      });
    });
    updateHeaderIndicators();

    // export handlers
    function exportJSON() {
      const payload = JSON.stringify(flagged, null, 2);
      const blob = new Blob([/** @type {any} */ (payload)], { type: 'application/json' });
      const name = reportName ? `${reportName.replace(/\.json$/,'')}_flagged.json` : `flagged_${Date.now()}.json`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }

    function exportCSV() {
      const keys = new Set();
      flagged.forEach(f => { Object.keys(f.metadata_excerpt || {}).forEach(k=>keys.add(k)); });
      const metaKeys = Array.from(keys);
      const header = ['url','status','reasons','timestamp',...metaKeys];
      const rows = flagged.map(f => {
        const meta = f.metadata_excerpt || {};
        const row = [f.url || '', f.status || '', Array.isArray(f.reasons)?f.reasons.join('; '):(f.reasons||''), f.timestamp || ''];
        metaKeys.forEach(k => row.push(meta[k] ?? ''));
        return row.map(v => `"${String(v).replace(/"/g,'""')}"`).join(',');
      });
      const csv = [header.join(','), ...rows].join('\n');
      const blob = new Blob([/** @type {any} */ (csv)], { type: 'text/csv' });
      const name = reportName ? `${reportName.replace(/\.json$/,'')}_flagged.csv` : `flagged_${Date.now()}.csv`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    }

    const exCSV = modal.querySelector('#flaggedExportCSV');
    if (exCSV) exCSV.addEventListener('click', exportCSV);
    const exJSON = modal.querySelector('#flaggedExportJSON');
    if (exJSON) exJSON.addEventListener('click', exportJSON);

    // close handler
    const closeBtn = modal.querySelector('#flaggedClose');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => { document.removeEventListener('keydown', escHandler); modal.remove(); });
    }
  } catch (e) {
    ErrorBoundary.logError(e, 'showFlaggedModal');
  }
}

// ============================================
// PHASE 2: Integration Tests (P2.6)
// ============================================

async function runIntegrationTests() {
  const tests = {
    largeDataset: () => {
      const largeArray = Array.from({length: 1000}, (_, i) => ({
        index: i + 1,
        label: `Option ${i + 1}`,
        metadata: { confidence: Math.random() }
      }));
      
      VirtualScroll.enable(largeArray, {clientHeight: 400});
      return VirtualScroll.isEnabled && VirtualScroll.getVisibleItems().length < largeArray.length;
    },
    
    errorBoundary: () => {
      let caught = false;
      ErrorBoundary.safeExecute(() => {
        throw new Error('Test error');
      }, 'test');
      
      const log = ErrorBoundary.getErrorLog();
      caught = log.some(e => e.context === 'test' && e.recovered);
      ErrorBoundary.clearErrorLog();
      return caught;
    },
    
    debounce: async () => {
      let callCount = 0;
      const fn = debounce(() => callCount++, 100);
      fn(); fn(); fn();
      await new Promise(r => setTimeout(r, 150));
      return callCount === 1;
    }
  };
  
  const results = {};
  for (const [name, test] of Object.entries(tests)) {
    try {
      results[name] = await Promise.resolve(test());
    } catch (e) {
      results[name] = false;
      ErrorBoundary.logError(e, `Test: ${name}`);
    }
  }
  
  const passed = Object.values(results).filter(Boolean).length;
  console.log(`[Integration Tests] ${passed}/${Object.keys(tests).length} passed`, results);
  return results;
}

// ============================================
// PHASE 3: Visual Enhancements & UX Polish
// ============================================

/**
 * P3.1: Color-Coded Logs
 * Apply color coding to log entries based on level
 */
const LogColorCoding = (() => {
  const levelColors = {
    'ERROR': { bg: '#3d1a1a', border: '#dc2626', text: '#fca5a5' },
    'CRITICAL': { bg: '#3d1a1a', border: '#991b1b', text: '#fca5a5' },
    'WARNING': { bg: '#3d2a1a', border: '#ea580c', text: '#fdba74' },
    'INFO': { bg: '#1a2a3d', border: '#3b82f6', text: '#93c5fd' },
    'DEBUG': { bg: '#1a2a2a', border: '#10b981', text: '#6ee7b7' },
    'TRACE': { bg: '#2a1a3d', border: '#8b5cf6', text: '#c4b5fd' }
  };
  
  /**
   * @typedef {Object} LevelColor
   * @property {string} bg
   * @property {string} border
   * @property {string} text
   */

  /* LevelColorMap typedef consolidated inline; see `levelColors` object above. */

  /**
   * Apply a level-specific CSS class to an element.
   * Removes any existing level classes derived from levelColors and adds the class for the provided level.
   * @param {HTMLElement} element
   * @param {string} level - Level key (e.g., "ERROR", "INFO")
   * @returns {void}
   */
  function applyColorToElement(element, level) {
    // Remove all level classes first
    Object.keys(levelColors).forEach(lvl => {
      element.classList.remove(`log-level-${lvl.toLowerCase()}`);
    });
    // Add the appropriate level class
    const levelClass = `log-level-${level.toLowerCase()}`;
    element.classList.add(levelClass);
  }
  
  /**
   * @typedef {Object.<string, LevelColor>} LevelColorMap
   */

  /**
   * Return the color mapping for a given log level.
   * @param {string} level
  * @returns {LevelColor}
   */
  function getLevelColor(level) {
    return levelColors[level] || levelColors['INFO'];
  }
  
  return { applyColorToElement, getLevelColor, levelColors };
})();

/**
 * P3.2: Type Badges
 * Visual identification of log message sources
 */
const LogTypeBadges = (() => {
  const typeConfig = {
    'status': { icon: '📊', color: '#3b82f6', label: 'Status' },
    'input': { icon: '📥', color: '#10b981', label: 'Input' },
    'output': { icon: '📤', color: '#8b5cf6', label: 'Output' },
    'error': { icon: '❌', color: '#dc2626', label: 'Error' },
    'exception': { icon: '⚠️', color: '#ea580c', label: 'Exception' },
    'prompt': { icon: '💬', color: '#06b6d4', label: 'Prompt' },
    'router': { icon: '🔀', color: '#6366f1', label: 'Router' },
    'handler': { icon: '⚙️', color: '#0ea5e9', label: 'Handler' },
    'download': { icon: '⬇️', color: '#14b8a6', label: 'Download' },
    'browser': { icon: '🌐', color: '#3b82f6', label: 'Browser' },
    'batch': { icon: '📦', color: '#8b5cf6', label: 'Batch' },
    'cancel': { icon: '🛑', color: '#dc2626', label: 'Cancel' },
    'summary': { icon: '📋', color: '#10b981', label: 'Summary' },
    'heartbeat': { icon: '💓', color: '#6b7280', label: 'Heartbeat' }
  };
  
  /**
   * @typedef {Object} TypeConfigEntry
   * @property {string} icon
   * @property {string} color
   * @property {string} label
   */

  /**
   * @typedef {Object.<string, TypeConfigEntry>} TypeConfigMap
   */

  /**
   * Create an HTML badge for a log type.
   * @param {string} type
   * @returns {string}
   */
  function createBadge(type) {
    /** @type {TypeConfigEntry} */
    const config = /** @type {TypeConfigEntry} */ (typeConfig[type] || { icon: '📌', color: '#6b7280', label: type });
    // Use CSS classes instead of inline styles for CSP compliance
    const safeType = (type || 'info').toLowerCase().replace(/[^a-z0-9]/g, '-');
    const typeClass = `log-type-${safeType}`;
    return `<span class="log-type-badge ${typeClass}">${config.icon} ${config.label}</span>`;
  }
  
  /**
   * Get configuration entry for a given log type.
   * @param {string} type
   * @returns {TypeConfigEntry}
   */
  function getTypeConfig(type) {
    return typeConfig[type] || { icon: '📌', color: '#6b7280', label: type };
  }
  
  return { createBadge, getTypeConfig, typeConfig };
})();

/**
 * P3.3: Search Highlighting
 * Highlight matching text in log messages
 */
const SearchHighlighter = (() => {
  /**
   * @typedef {Object} HighlightMatch
   * @property {string} original - original matched text
   * @property {string} highlighted - HTML highlighted fragment
   *
   * @callback HighlightTextFn
   * @param {string} text - full text to search within
   * @param {string} searchTerm - term to highlight
   * @returns {string} HTML string with <mark class="search-highlight"> wrappers
   */

  /** @type {HighlightTextFn} */
  function highlightText(text, searchTerm) {
    if (!searchTerm || !text) return escapeHtml(text);

    const escaped = escapeHtml(text);
    const regex = new RegExp(`(${escapeRegex(searchTerm)})`, 'gi');
    return escaped.replace(regex, '<mark class="search-highlight">$1</mark>');
  }
  
  function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  
  function clearHighlights() {
    $$('.search-highlight').forEach(el => {
      const parent = el.parentNode;
      parent.replaceChild(document.createTextNode(el.textContent), el);
      parent.normalize();
    });
  }
  
  return { highlightText, clearHighlights };
})();

/**
 * P3.4: Advanced Export
 * Export logs in multiple formats (JSON, CSV, Markdown)
 */
const AdvancedExport = (() => {
  /* LogRecord typedef consolidated at top of file. */
  /**
   * @typedef {Object} AdvancedExportInterface
   * @property {(logs: LogRecord[], filename?: string) => void} exportAsJSON
   * @property {(logs: LogRecord[], filename?: string) => void} exportAsCSV
   * @property {(logs: LogRecord[], filename?: string) => void} exportAsMarkdown
   */

  /**
   * Export logs as a JSON file.
   * @param {LogRecord[]|any[]} logs
   * @param {string} [filename='parser_logs.json']
   * @returns {void}
   */
  function exportAsJSON(logs, filename = 'parser_logs.json') {
    const data = JSON.stringify(logs, null, 2);
    downloadBlob(data, filename, 'application/json');
  }
  
  /* LogRecord typedef consolidated at top of file. */

  /**
   * Export logs as CSV.
   * @param {LogRecord[]} logs
   * @param {string} [filename='parser_logs.csv']
   * @returns {void}
   */
  function exportAsCSV(logs, filename = 'parser_logs.csv') {
    /** @type {string[]} */
    const headers = ['Timestamp', 'Level', 'Type', 'Message', 'Session ID'];
    /** @type {Array<Array<string>>} */
    const rows = logs.map(log => [
      new Date(log.timestamp).toISOString(),
      log.level || '',
      log.type || '',
      (log.message || '').replace(/"/g, '""'),
      log.sessionId || ''
    ]);
    
    const csv = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');
    
    downloadBlob(csv, filename, 'text/csv');
  }
  
  /**
   * @typedef {Object} MarkdownExportEntry
   * @property {number|string} timestamp
   * @property {string} [level]
   * @property {string} [type]
   * @property {string} [message]
   * @property {string} [sessionId]
   */

  /**
   * Export logs as a Markdown file.
   * @param {MarkdownExportEntry[]|any[]} logs
   * @param {string} [filename='parser_logs.md']
   * @returns {void}
   */
  function exportAsMarkdown(logs, filename = 'parser_logs.md') {
    const header = '# Parser Logs\n\n';
    const timestamp = `**Exported:** ${new Date().toISOString()}\n`;
    const count = `**Total Logs:** ${logs.length}\n\n`;
    const divider = '---\n\n';
    
    /**
     * @typedef {Object} MarkdownExportLog
     * @property {number} timestamp
     * @property {string} level
     * @property {string} type
     * @property {string} message
     * @property {string|null} sessionId
     */

    /** @type {string} */
    const entries = logs.map(/** @param {MarkdownExportLog} log */ (log) => {
      /** @type {string} */
      const time = new Date(log.timestamp).toLocaleString();
      /** @type {string} */
      const level = log.level || 'INFO';
      /** @type {string} */
      const type = log.type || 'info';
      /** @type {string} */
      const msg = log.message || '';

      return `### ${level} - ${type}\n**Time:** ${time}\n**Session:** ${log.sessionId || 'N/A'}\n\n${msg}\n\n`;
    }).join(divider);
    
    const markdown = header + timestamp + count + divider + entries;
    downloadBlob(markdown, filename, 'text/markdown');
  }
  
  /**
   * @typedef {string|Blob|ArrayBuffer|ArrayBufferView|Uint8Array|Int8Array|Uint8ClampedArray|Int16Array|Uint16Array|Int32Array|Uint32Array|Float32Array|Float64Array} BlobContent
   *
   * @callback DownloadBlobFn
   * @param {BlobContent} content - The content to write into the blob.
   * @param {string} filename - Suggested filename for download.
   * @param {string} mimeType - MIME type for the blob (e.g., "application/json").
   * @returns {void}
   */

  /** @type {DownloadBlobFn} */
  function downloadBlob(content, filename, mimeType) {
    const blob = new Blob([/** @type {any} */ (content)], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    showToast(`Exported ${filename}`, 'success');
  }
  
  return { exportAsJSON, exportAsCSV, exportAsMarkdown };
})();

/**
 * P4.1: Keyboard Reference Guide
 * Visual guide for keyboard shortcuts
 */
const KeyboardGuide = (() => {
  const shortcuts = [
    { key: 'Escape', description: 'Close modal/prompt' },
    { key: 'Ctrl+Enter', description: 'Submit single option' },
    { key: 'Ctrl+S', description: 'Save current filter preset' },
    { key: 'Ctrl+E', description: 'Export logs as JSON' },
    { key: 'Ctrl+Shift+E', description: 'Export logs as CSV' },
    { key: 'Ctrl+/', description: 'Show keyboard shortcuts' },
    { key: 'Ctrl+L', description: 'Clear log output' },
    { key: 'Ctrl+F', description: 'Focus search input' }
  ];
  
  function show() {
    const modal = document.createElement('div');
    modal.className = 'modal keyboard-guide-modal';
    modal.innerHTML = `
      <div class="modal-content modal-large-max">
        <div class="modal-header">
          <h3>⌨️ Keyboard Shortcuts</h3>
          <button class="modal-close" aria-label="Close shortcuts guide">×</button>
        </div>
        <div class="modal-body">
          <div class="shortcuts-list">
            ${shortcuts.map(s => `
              <div class="shortcut-row">
                <kbd class="shortcut-key">${escapeHtml(s.key)}</kbd>
                <span class="shortcut-desc">${escapeHtml(s.description)}</span>
              </div>
            `).join('')}
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn btn-primary" data-action="modal-remove">Got it</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
    const closeBtn = modal.querySelector('.modal-close');
    if (closeBtn instanceof HTMLElement) closeBtn.addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => {
      const tgt = (e && e.target && (e.target instanceof Node)) ? e.target : null;
      if (tgt === modal) modal.remove();
    });
  }
  
  return { show, shortcuts };
})();

const STATES = [
  'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
  'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
  'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
  'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
  'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
];

// ============================================
// Socket.IO Setup (existing integration)
// ============================================

const socket = io({
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: 5,
});

let currentSessionId = null;
let activePromptMessage = null;
let activePromptOptions = [];
let bundleExpandedState = new Map(); // Track which bundles are expanded
let selectedPromptOptions = new Set(); // Multi-select tracking

// ===== DIAGNOSTIC: Log all Socket.IO events =====
const allEventsReceivedBySocket = [];
const oneventOrig = socket.onevent;
/** @param {any} packet */
socket.onevent = function(packet) {
  const eventName = packet.data[0];
  const eventData = packet.data[1];
  
  // Log to console for debugging
  console.debug(`[Socket.IO:${eventName}]`, eventData);
  
  // Store in array for inspection
  allEventsReceivedBySocket.push({
    timestamp: new Date().toISOString(),
    event: eventName,
    data: JSON.parse(JSON.stringify(eventData)) // deep clone for safety
  });
  
  // Keep only last 100 events to avoid memory leak
  if (allEventsReceivedBySocket.length > 100) {
    allEventsReceivedBySocket.shift();
  }
  
  // Call original handler
  return oneventOrig.call(this, packet);
};

// Export for debugging in browser console
window.debugSocketIO = {
  getAllEvents: () => allEventsReceivedBySocket,
  getLastEvent: () => allEventsReceivedBySocket[allEventsReceivedBySocket.length - 1],
  // getEventsByName consolidated at top-level typed helper
  getEventsByName: /** @type {(name: string) => Array<{ timestamp: string; event: string; data: any }>} */ ((name) => allEventsReceivedBySocket.filter(e => e.event === name)),
  getCurrentSession: () => currentSessionId,
  getModalState: () => ({
    promptTitle: document.getElementById('promptTitle')?.textContent,
    promptMessage: document.getElementById('promptMessage')?.textContent,
    optionsCount: document.getElementById('promptOptions')?.querySelectorAll('.prompt-option').length,
    modalHidden: document.getElementById('promptModal')?.classList.contains('hidden')
  })
};

socket.on('connect', () => {
  console.log('[Socket.IO] Connected:', socket.id);
});

/**
 * @typedef {{ session_id: string }} SessionIdPayload
 */

socket.on('session_id', /** @param {SessionIdPayload} data */ (data) => {
  /** @type {string} */
  currentSessionId = data.session_id;
  console.log('[Session] ID:', currentSessionId);
  updateSessionsList();
  
  // Restore Direct URL draft for this session if exists
  AdvancedFeatures.currentSessionId = currentSessionId;
  /** @type {string|undefined} */
  const savedDraft = AdvancedFeatures.directUrlDraftBySession.get(currentSessionId);
  /** @type {HTMLElement | null} */
  const textarea = document.getElementById('directUrlTextarea');
  if (textarea instanceof HTMLTextAreaElement && savedDraft) {
    textarea.value = savedDraft;
    parseDirectUrlField();
  }
});

/* ParserOutputContext/ParserOutputEvent typedefs consolidated at top of file. */

socket.on('parser_output', /** @param {ParserOutputEvent} data */ (data) => {
  ErrorBoundary.safeAsync(async () => {
    // DEBUG: Log all incoming parser_output events
    console.debug('[Socket.IO parser_output]', {
      type: data?.type,
      messagePreview: typeof data?.message === 'string' ? data.message.substring(0, 80) : data?.message,
      hasContext: !!data?.context,
      contextKeys: data?.context ? Object.keys(data.context) : [],
      fullData: data
    });

    ingestQualityMetrics(data);

    addLog(data);
    handlePromptLog(data);
    SessionRestore.saveState(data); // P2.4: Save state for recovery
    // Show pending overlay for processing messages (P1.4)
    if (data.type === 'status' && data.message?.includes('Processing')) {
      PendingOverlay.show(data.message, 300);
    }
  }, 'socket:parser_output');
});

/* ContestOption/ContestOptionsPayload typedefs consolidated at top of file. */

socket.on('contest_options', /** @param {ContestOptionsPayload} data */ (data) => {
  ErrorBoundary.safeExecute(
    /** @type {() => void} */ (() => {
      console.debug('[Socket.IO contest_options]', {
        optionsCount: data?.options?.length,
        optionsSample: data?.options?.slice(0, 3),
        context: data?.context,
        fullData: data
      });
      handleContestOptions(data);
    }),
    'socket:contest_options'
  );
});

/**
 * @typedef {Object<string, any>} SessionMetadata
 *
 * @typedef {Object} SessionStatePayload
 * @property {string} [session_id]
 * @property {string} [state]
 * @property {string} [phase]
 * @property {SessionMetadata} [metadata]
 * @property {number|string} [timestamp]
 */
socket.on('session_state', /** @param {SessionStatePayload} data */ (data) => {
  ErrorBoundary.safeExecute(
    /** @type {() => void} */ (() => {
      console.log('[Session State]', data);
      updateProgressCard(data);
      updateOverviewStrip(data);
      updateSessionsList();
    }),
    'socket:session_state'
  );
});

/**
 * @typedef {Object} SessionSummary
 * @property {string} session_id
 * @property {string} [state]
 * @property {string} [phase]
 * @property {Object.<string, any>} [metadata]
 */

/**
 * @typedef {Object} SessionListPayload
 * @property {SessionSummary[]} [sessions]
 */

socket.on('session_list', /** @param {SessionListPayload} data */ (data) => {
  ErrorBoundary.safeExecute(() => {
    updateSessionsList(data.sessions);
  }, 'socket:session_list');
});

// Run lifecycle events: started / progress / summary
/**
 * @typedef {Object} RunStartedPayload
 * @property {string} session_id
 * @property {number} [timestamp]
 * @property {number} [total_entries]
 * @property {Object.<string, any>} [metadata]
 */

/**
 * @typedef {HTMLDivElement & { id: string, className: string }} RunSummaryPanelElement
 */

socket.on('run_started', /** @param {RunStartedPayload} data */ (data) => {
  ErrorBoundary.safeExecute(() => {
    console.info('[Run] started', data);
    showToast(`Run started (${data.session_id})`, 'info');
    /** @type {RunSummaryPanelElement | null} */
    let panel = /** @type {RunSummaryPanelElement | null} */ (document.getElementById('runSummaryPanel'));
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'runSummaryPanel';
      panel.className = 'run-summary-panel';
      const container = document.getElementById('runControls') || document.body;
      container.prepend(panel);
    }
    panel.innerHTML = `<strong>Run:</strong> ${escapeHtml(data.session_id)} — <em>started</em> <span class="small muted">(${new Date(data.timestamp*1000).toLocaleString()})</span>`;
  }, 'socket:run_started');
});

/**
 * @typedef {Object} RunProgressPayload
 * @property {string} session_id
 * @property {number} [total_entries]
 * @property {number} [processed]
 * @property {Object<string, number>} [status_counts]
 * @property {string} [status]
 */

socket.on('run_progress', /** @param {RunProgressPayload} data */ (data) => {
  ErrorBoundary.safeExecute(() => {
    // data: { session_id, total_entries, processed, status_counts }
    /** @type {HTMLElement | null} */
    let panel = /** @type {HTMLElement | null} */ (document.getElementById('runSummaryPanel'));
    if (!panel) return;
    /** @type {number} */
    const total = Number(data.total_entries || 0);
    /** @type {number} */
    const processed = Number(data.processed || 0);
    /** @type {number} */
    const pct = total ? Math.round((processed / total) * 100) : 0;
    panel.innerHTML = `<strong>Run:</strong> ${escapeHtml(data.session_id)} — ${pct}% (${processed}/${total})`;
  }, 'socket:run_progress');
});

/**
 * @typedef {Object} RunErrorEntry
 * @property {string} [url]
 * @property {string|number} [status]
 * @property {string} [error]
 */

/**
 * @typedef {Object} RunConfidenceMetrics
 * @property {number} [avg]
 * @property {number} [min]
 * @property {number} [max]
 * @property {number} [median]
 * @property {number} [count]
 */

/**
 * @typedef {Object} FlaggedDetail
 * @property {string} [url]
 * @property {string} [status]
 * @property {string|string[]} [reasons]
 * @property {Object<string, any>} [metadata_excerpt]
 * @property {number|string} [timestamp]
 */

/**
 * @typedef {Object} RunSummary
 * @property {Object.<string, number>} [status_counts]
 * @property {number} [total_entries]
 * @property {number} [flagged_count]
 * @property {RunConfidenceMetrics} [confidence_metrics]
 * @property {RunErrorEntry[]} [errors]
 * @property {FlaggedDetail[]} [flagged_details]
 */

/**
 * @typedef {Object} RunSummaryPayload
 * @property {string} session_id
 * @property {number} [timestamp]
 * @property {RunSummary} [summary]
 * @property {string} [report_path]
 */

socket.on('run_summary', /**
 * @param {RunSummaryPayload} data
 */ (data) => {
  ErrorBoundary.safeExecute(() => {
    console.info('[Run] summary', data);
    showToast('Run completed', 'success');
    /** @type {HTMLElement | null} */
    let panel = document.getElementById('runSummaryPanel');
    if (!panel) {
      panel = document.createElement('div');
      panel.id = 'runSummaryPanel';
      panel.className = 'run-summary-panel';
      const container = document.getElementById('runControls') || document.body;
      container.prepend(panel);
    }
    /** @type {RunSummary} */
    const summary = data.summary || {};
    /** @type {Object.<string, number>} */
    const counts = summary.status_counts || {};
    /** @type {number} */
    const total = summary.total_entries || 0;
    let html = `<strong>Run:</strong> ${escapeHtml(data.session_id)} — <em>completed</em> <span class="small muted">(${new Date((data.timestamp || Date.now())*1000).toLocaleString()})</span>`;
    html += `<div class="mt-2">Total: ${total} — `;
    html += Object.entries(counts).map(([k,v])=>`${escapeHtml(k)}: ${v}`).join(' · ');
    html += `</div>`;
    // flagged count
    if (typeof summary.flagged_count !== 'undefined') {
      html += `<div class="mt-1">Flagged for review: ${Number(summary.flagged_count)}</div>`;
    }
    // confidence metrics
    /** @type {RunConfidenceMetrics} */
    const conf = summary.confidence_metrics || {};
    if (conf && conf.count) {
      html += `<div class="mt-1">Confidence — avg: ${Number(conf.avg).toFixed(2)} min: ${Number(conf.min).toFixed(2)} max: ${Number(conf.max).toFixed(2)} median: ${Number(conf.median).toFixed(2)} (n=${conf.count})</div>`;
    }
    // errors list (collapsible)
    /** @type {RunErrorEntry[]} */
    const errors = summary.errors || [];
    if (Array.isArray(errors) && errors.length) {
      html += `<div class="mt-2"><details><summary>Errors (${errors.length})</summary><ul class="small">`;
      for (const e of errors.slice(0, 20)) {
        const msg = e.error ? ` — ${escapeHtml(e.error)}` : '';
        html += `<li>${escapeHtml(e.url || e['url'] || String(e))} (${escapeHtml(String(e.status || ''))})${msg}</li>`;
      }
      if (errors.length > 20) html += `<li class="muted small">...and ${errors.length-20} more</li>`;
      html += `</ul></details></div>`;
    }
    // flagged_details (expanded, limited view)
    /** @type {FlaggedDetail[]} */
    const flagged = summary.flagged_details || [];
    if (Array.isArray(flagged) && flagged.length) {
      // store last flagged set for modal access
      window.__lastRunFlagged = flagged;
      window.__lastRunReportPath = data.report_path || '';
      html += `<div class="mt-2"><details><summary>Flagged Details (${flagged.length})</summary><ul class="small flagged-list">`;
      for (const f of flagged.slice(0, 20)) {
        const reasons = Array.isArray(f.reasons) ? f.reasons.join(', ') : (f.reasons || '');
        const meta = f.metadata_excerpt ? escapeHtml(JSON.stringify(f.metadata_excerpt)) : '';
        const when = f.timestamp ? ` (${escapeHtml(String(f.timestamp))})` : '';
        const urlText = escapeHtml(f.url || f["url"] || '');
        // link to report if available
        let reportLink = '';
        if (data.report_path) {
          const parts = data.report_path.replace(/\\/g, '/').split('/');
          const name = parts[parts.length-1] || data.report_path;
          const href = `/download_fs?root=output&path=reports&name=${encodeURIComponent(name)}`;
          reportLink = ` <a href="${href}" target="_blank" rel="noopener">View report</a>`;
        }
        html += `<li><strong>${urlText}</strong>${when} — ${escapeHtml(f.status || '')} — ${escapeHtml(reasons)}${reportLink}<pre class="muted small pre-wrap mt-6px">${meta}</pre></li>`;
      }
      if (flagged.length > 20) html += `<li class="muted small">...and ${flagged.length-20} more</li>`;
      html += `</ul></details></div>`;
      html += `<div class="mt-1"><button id="btnViewFlagged" class="btn btn-sm btn-primary">View flagged details</button></div>`;
    }
    if (data.report_path) {
      const parts = data.report_path.replace(/\\/g, '/').split('/');
      const name = parts[parts.length-1] || data.report_path;
      const href = `/download_fs?root=output&path=reports&name=${encodeURIComponent(name)}`;
      html += `<div class="mt-2"><a href="${href}" target="_blank" rel="noopener">Download report</a></div>`;
    }
    panel.innerHTML = html;
    // attach listener for modal open if present
    try {
      const btn = document.getElementById('btnViewFlagged');
      if (btn) {
        btn.addEventListener('click', () => {
          const flagged = window.__lastRunFlagged || [];
          const rp = window.__lastRunReportPath || '';
          showFlaggedModal(flagged, rp);
        });
      }
    } catch (e) {
      console.warn('Failed to attach flagged modal handler', e);
    }
  }, 'socket:run_summary');
});

/**
 * @typedef {Object} SessionClonedPayload
 * @property {string} [old_session]  // original session id (optional)
 * @property {string} new_session    // newly created session id
 */

socket.on('session_cloned', /** @param {SessionClonedPayload} data */ (data) => {
  ErrorBoundary.safeExecute(() => {
    console.log('[Session Cloned]', data);
    showToast(`Session cloned: ${data.new_session}`, 'success');
    updateSessionsList();
  }, 'socket:session_cloned');
});

/**
 * @typedef {Object} SessionDeletedPayload
 * @property {string} [session_id]    // Logical session id that was deleted
 * @property {string} [reason]        // Optional human-readable reason
 * @property {Object.<string, any>} [metadata] // Additional payload metadata
 */

socket.on('session_deleted', /** @param {SessionDeletedPayload} data */ (data) => {
  ErrorBoundary.safeExecute(() => {
    console.log('[Session Deleted]', data);
    showToast('Session deleted', 'info');
    updateSessionsList();
  }, 'socket:session_deleted');
});

// ============================================
// State Management
// ============================================

const state = {
  results: [],
  sessions: [],
  logs: [],
  filters: {
    search: '',
    confidence: 0,
    state: '',
    level: '',
  },
  selectedResults: new Set(),
  currentFile: null,
  autoScroll: true,
};

// ============================================
// Pending Overlay for Long Operations
// ============================================

const PendingOverlay = (() => {
  let element = null;
  let hideTimer = null;
  
  function create() {
    if (element) return element;
    element = document.createElement('div');
    element.id = 'pendingOverlay';
    element.className = 'pending-overlay hidden';
    element.innerHTML = `
      <div class="pending-overlay-content">
        <div class="spinner"></div>
        <div class="pending-text">Processing...</div>
      </div>
    `;
    document.body.appendChild(element);
    return element;
  }
  
  function show(message = 'Processing...', minDuration = 500) {
    if (!element) create();
    element.querySelector('.pending-text').textContent = message;
    element.classList.remove('hidden');
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = setTimeout(() => hide(), minDuration);
  }
  
  function hide() {
    if (!element) return;
    element.classList.add('hidden');
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
  }
  
  return { show, hide };
})();

// Expose overlay toggle for headless checks / tests
try {
  window.setOverlayVisible = function(visible){
    if (visible) PendingOverlay.show(''); else PendingOverlay.hide();
    return true;
  };
} catch (/** @type {any} */ e) { /* ignore */ }

// Diagnostic: expose layout metrics for headless runs
try {
  (/** @type {any} */ (window)).dumpLayoutMetrics = function(){
    /** @param {string} sel */
    const snap = (sel) => {
      const el = document.querySelector(sel);
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      const styles = getComputedStyle(el);
      return {
        width: rect.width,
        height: rect.height,
        top: rect.top,
        left: rect.left,
        paddingInline: [styles.paddingLeft, styles.paddingRight],
        marginInline: [styles.marginLeft, styles.marginRight],
        gap: styles.gap || styles.columnGap || '',
        display: styles.display,
        maxWidth: styles.maxWidth
      };
    };
    return {
      ts: Date.now(),
      viewport: { w: window.innerWidth, h: window.innerHeight },
      resultsHeader: snap('.results-header'),
      resultsGrid: snap('.results-grid'),
      drawer: snap('#logDrawer'),
      drawerHandle: snap('.drawer-handle'),
      footer: snap('#sessionFooter')
    };
  };
} catch (/** @type {any} */ e) { /* ignore */ }

// ============================================
// Filter Presets for Log Console
// ============================================

const filterPresets = (() => {
  const STORAGE_KEY = 'logFilterPresets';
  
  /**
   * @typedef {Object} FilterState
   * @property {string} [search]
   * @property {string} [level]
   * @property {string} [type]
   *
   * @typedef {Object} PresetEntry
   * @property {string} search
   * @property {string} level
   * @property {string} type
   * @property {number} timestamp
   *
   * @typedef {Object.<string, PresetEntry>} PresetMap
   */

  /**
   * Save a named filter preset to localStorage.
   * @param {string} name
   * @param {FilterState} filters
   * @returns {void}
   */
  function save(name, filters) {
    if (!name || !filters) return;
    /** @type {PresetMap} */
    const presets = /** @type {PresetMap} */ (JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'));
    presets[name] = {
      search: filters.search || '',
      level: filters.level || '',
      type: filters.type || '',
      timestamp: Date.now()
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
    updatePresetDropdown();
  }
  
  /**
   * Load a named preset from localStorage.
   * @param {string} name
   * @returns {PresetEntry|null}
   */
  function load(name) {
    /** @type {PresetMap} */
    const presets = /** @type {PresetMap} */ (JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'));
    return presets[name] || null;
  }
  
  /**
   * Delete a named preset.
   * @param {string} name
   * @returns {void}
   */
  function deletePreset(name) {
    /** @type {PresetMap} */
    const presets = /** @type {PresetMap} */ (JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'));
    delete presets[name];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
    updatePresetDropdown();
  }
  
  function list() {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
  }
  
  function updatePresetDropdown() {
    const select = $('#logFilterPresetSelect');
    if (!select) return;
    const presets = list();
    const options = [
      '<option value="">— Save new preset —</option>',
      '<option value="__separator__" disabled>─────────</option>'
    ];
    Object.keys(presets).sort().forEach(name => {
      options.push(`<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`);
    });
    select.innerHTML = options.join('');
  }
  
  /**
   * @typedef {Object} FilterPresetEntry
   * @property {string} [search]
   * @property {string} [level]
   * @property {string} [type]
   * @property {number} [timestamp]
   */

  /**
   * Apply a named preset from storage to the UI filters.
   * @param {string} name
   * @returns {void}
   */
  function applyPreset(name) {
    /** @type {FilterPresetEntry|null} */
    const preset = /** @type {FilterPresetEntry|null} */ (load(name));
    if (!preset) return;
    state.filters = { ...state.filters, ...preset };
    renderLogs();
    /** @type {HTMLElement | null} */
    const logSearchInputEl = /** @type {HTMLElement | null} */ (document.getElementById('logSearchInput'));
    if (logSearchInputEl instanceof HTMLInputElement || logSearchInputEl instanceof HTMLTextAreaElement) logSearchInputEl.value = preset.search || '';
    /** @type {HTMLElement | null} */
    const logLevelFilterEl = /** @type {HTMLElement | null} */ (document.getElementById('logLevelFilter'));
    if (logLevelFilterEl instanceof HTMLInputElement || logLevelFilterEl instanceof HTMLSelectElement || logLevelFilterEl instanceof HTMLTextAreaElement) logLevelFilterEl.value = preset.level || '';
  }
  
  return { save, load, deletePreset, list, updatePresetDropdown, applyPreset };
})();

// ============================================
// Utility Functions
// ============================================

/**
 * @typedef {HTMLElement | SVGElement | null} DomElement
 *
 * Query helper interfaces
 * @typedef {(selector: string) => DomElement} QuerySelectorFn
 */

/**
 * Lightweight single-element selector helper.
 * Returns the first Element matching the selector or null when not found.
 * Kept minimal to avoid changing runtime behavior.
 * @type {QuerySelectorFn}
 * @param {string} selector
 * @returns {DomElement}
 */
function $(selector) {
  return document.querySelector(selector);
}

/**
 * @template {Element} T
 * @typedef {(selector: string) => NodeListOf<T>} QuerySelectorAllFn
 *
 * Lightweight multi-element selector helper.
 * Returns a live NodeList of matching Elements or an empty NodeList when none found.
 * Keeping a generic T allows better editor intellisense when used with specific element casts.
 *
 * @type {QuerySelectorAllFn<Element>}
 * @param {string} selector
 * @returns {NodeListOf<Element>}
 */
function $$(selector) {
  return document.querySelectorAll(selector);
}

/**
 * @typedef {'info'|'success'|'warning'|'error'} ToastType
 * @typedef {HTMLElement & { _timeoutId?: number | null }} ToastElement
 */

/**
 * Show a transient toast notification.
 * @param {string} message
 * @param {ToastType} [type='info']
 * @param {number} [duration=CONFIG.toastDuration]
 * @returns {void}
 */
function showToast(message, type = 'info', duration = CONFIG.toastDuration) {
  /** @type {ToastElement} */
  const toast = /** @type {ToastElement} */ (document.createElement('div'));
  toast.className = `toast ${type} fade-in`;
  
  /** @type {Record<ToastType, string>} */
  const icons = {
    info: 'ℹ️',
    success: '✓',
    warning: '⚠️',
    error: '✗',
  };
  
  toast.innerHTML = `
    <div class="toast-icon">${icons[type] || type}</div>
    <div class="toast-message">${escapeHtml(message)}</div>
    <button class="toast-close">×</button>
  `;
  
  const container = /** @type {HTMLElement | null} */ ($('#toastContainer'));
  if (container) container.appendChild(toast);
  else document.body.appendChild(toast);

  const closeBtn = toast.querySelector('.toast-close');
  if (closeBtn instanceof HTMLElement) {
    closeBtn.addEventListener('click', () => {
      toast.remove();
    });
  }
  
  // Ensure removal after duration (store timeout id in case future code wants to clear)
  const toId = /** @type {TimeoutId} */ (/** @type {any} */ (setTimeout(() => {
    try {
      toast.style.animation = 'slideOutRight 300ms ease';
      setTimeout(() => toast.remove(), 300);
    } catch (e) {
      try { toast.remove(); } catch (err) {}
    }
  }, duration)));
  toast._timeoutId = toId;
}

function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

/**
 * @typedef {'B'|'KB'|'MB'|'GB'} ByteUnit
 *
 * @typedef {Object} ByteFormatter
 * @property {(bytes: number) => string} formatBytes
 */

/**
 * Format a byte count into a human-readable string.
 * @type {(bytes: number) => string}
 * @param {number} bytes
 * @returns {string}
 */
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  /** @type {number} */
  const k = 1024;
  /** @type {ByteUnit[]} */
  const sizes = /** @type {ByteUnit[]} */ (['B', 'KB', 'MB', 'GB']);
  /** @type {number} */
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * @typedef {string|number|Date} DateInput
 */

/**
 * Format a date-like value into a locale string.
 * @param {DateInput} date
 * @returns {string}
 */
function formatDate(date) {
  const d = new Date(date);
  return d.toLocaleString();
}

/**
 * @typedef {'high-confidence'|'medium-confidence'|'low-confidence'} ConfidenceClass
 */

/**
 * Classifier interface for confidence-based labels.
 * @typedef {Object} ConfidenceClassifier
 * @property {(confidence: number) => ConfidenceClass} classify
 */

/**
 * Map a numeric confidence into a CSS classification string.
 * @param {number} confidence
 * @returns {ConfidenceClass}
 */
function parseConfidenceClass(confidence) {
  if (confidence >= 90) return 'high-confidence';
  if (confidence >= 70) return 'medium-confidence';
  return 'low-confidence';
}

// ============================================
// Log Management
// ============================================

/* ParserOutputEvent/LogRecord typedefs consolidated at top of file. */

/**
 * Add a normalized log entry to the UI buffer.
 * @param {ParserOutputEvent} logObj
 * @returns {void}
 */
function addLog(logObj) {
  ErrorBoundary.safeExecute(() => {
    /** @type {LogRecord} */
    const normalized = {
      timestamp: Number(logObj.timestamp || Date.now()),
      level: String(logObj.level || 'INFO'),
      type: String(logObj.type || 'info'),
      message: String(logObj.message || ''),
      sessionId: logObj.session_id || currentSessionId || null,
    };
    
    state.logs.push(normalized);
    
    // Keep buffer size manageable
    if (state.logs.length > CONFIG.logBufferSize) {
      state.logs.shift();
    }
    
    // Update counts
    updateLogCounts();
    
    // Apply filter and render
    renderLogs();
    
    // Auto-scroll if enabled
    if (state.autoScroll) {
      const logOutput = $('#logOutput');
      try {
        if (logOutput) logOutput.scrollTop = logOutput.scrollHeight;
      } catch (e) { /* ignore scroll errors */ }
    }
  }, 'addLog');
}

function updateLogCounts() {
  const errors = state.logs.filter(l => l.level === 'ERROR').length;
  const warnings = state.logs.filter(l => l.level === 'WARNING').length;
  const infos = state.logs.filter(l => l.level === 'INFO').length;
  
  $('#errorCount').textContent = String(errors);
  $('#warningCount').textContent = String(warnings);
  $('#infoCount').textContent = String(infos);
}

function renderLogs() {
  const filtered = state.logs.filter(log => {
    if (state.filters.level && log.level !== state.filters.level) return false;
    if (state.filters.search) {
      const searchLower = state.filters.search.toLowerCase();
      const msgLower = (log.message || '').toLowerCase();
      if (!msgLower.includes(searchLower)) return false;
    }
    return true;
  });
  
  const logOutput = $('#logOutput');
  logOutput.innerHTML = filtered.map(log => {
    const typeBadge = LogTypeBadges.createBadge(log.type);
    const highlightedMsg = state.filters.search 
      ? SearchHighlighter.highlightText(log.message, state.filters.search)
      : escapeHtml(log.message);
    
    // Use CSS classes instead of inline styles for CSP compliance
    const levelClass = `log-level-${(log.level || 'INFO').toLowerCase()}`;
    
    return `
      <div class="log-line ${levelClass}">
        <span class="log-timestamp">${new Date(log.timestamp).toLocaleTimeString()}</span>
        <span class="log-level">${log.level}</span>
        ${typeBadge}
        <div class="log-message">${highlightedMsg}</div>
      </div>
    `;
  }).join('');
}

// ============================================
// Results Management (SheetJS Integration)
// ============================================

/* Result typedef consolidated at top of file. */

/**
 * Create a result card HTML fragment for the given result.
 * @param {Result} result
 * @returns {string}
 */
function createResultCard(result) {
  const confClass = parseConfidenceClass(result.confidence || 0);
  
  return `
    <div class="result-card" data-result-id="${result.id}">
      <div class="card-header">
        <div class="card-icon">📊</div>
        <div class="card-title">
          <div class="card-name">${escapeHtml(result.name)}</div>
          <span class="card-type-badge">${result.type.toUpperCase()}</span>
        </div>
      </div>
      
      <div class="card-stats">
        <div class="card-stat">
          <span class="stat-label">Rows</span>
          <span class="stat-value">${(result.rows || 0).toLocaleString()}</span>
        </div>
        <div class="card-stat">
          <span class="stat-label">Confidence</span>
          <span class="stat-value ${confClass}">${(result.confidence || 0).toFixed(1)}%</span>
        </div>
      </div>
      
      <div class="card-preview">${result.preview || 'No preview available'}</div>
      
      <div class="card-actions">
        <button class="btn-sm btn-preview" data-result-id="${result.id}">👁 Preview</button>
        <button class="btn-sm btn-download" data-result-id="${result.id}">⬇ Download</button>
        <input type="checkbox" class="card-checkbox" data-result-id="${result.id}" id="select-result-${result.id}" name="select-result-${result.id}">
      </div>
    </div>
  `;
}

function renderResults() {
  const filtered = state.results.filter(r => {
    if (state.filters.search && !r.name.toLowerCase().includes(state.filters.search.toLowerCase())) {
      return false;
    }
    if (state.filters.confidence && (r.confidence || 0) < state.filters.confidence) {
      return false;
    }
    if (state.filters.state && !r.state?.includes(state.filters.state)) {
      return false;
    }
    return true;
  });
  // Attach non-inline handlers to comply with CSP (avoid inline attributes)
  attachResultHandlers();
  

function attachResultHandlers() {
  // Preview buttons
  $$('.btn-preview').forEach(btn => {
    const prev = /** @type {any} */ ((/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.getBound ? (/** @type {any} */ (window)).__tl_helpers.getBound(btn, '_boundPreview') : null);
    if (prev && btn.removeEventListener) btn.removeEventListener('click', prev);
    const id = btn.getAttribute('data-result-id');
    /** @type {ClickHandler} */
    const handler = /** @type {ClickHandler} */ (function(e) {
      /** @type {Event} */ (e);
      try { e.preventDefault(); } catch (/** @type {any} */ _){ }
      previewFile(id);
    });
    btn.addEventListener('click', handler);
    if ((/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.setBound) {
      (/** @type {any} */ (window)).__tl_helpers.setBound(btn, '_boundPreview', handler);
    } else {
      /** @type {any} */ (btn)._boundPreview = handler;
    }
  });
  // Download buttons
  $$('.btn-download').forEach(btn => {
    const prev = /** @type {any} */ ((/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.getBound ? (/** @type {any} */ (window)).__tl_helpers.getBound(btn, '_boundDownload') : null);
    if (prev && btn.removeEventListener) btn.removeEventListener('click', prev);
    const id = btn.getAttribute('data-result-id');
    /** @type {ClickHandler} */
    const downloadHandler = /** @type {ClickHandler} */ (function(e) {
      /** @type {Event} */ (e);
      try { e.preventDefault(); } catch (/** @type {any} */ _) { /* ignore */ }
      downloadFile(/** @type {string} */ (id));
    });
    btn.addEventListener('click', downloadHandler);
    if ((/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.setBound) {
      (/** @type {any} */ (window)).__tl_helpers.setBound(btn, '_boundDownload', downloadHandler);
    } else {
      /** @type {any} */ (btn)._boundDownload = downloadHandler;
    }
  });
  // Checkboxes
  $$('.card-checkbox').forEach(cb => {
    const id = cb.getAttribute('data-result-id');
    /**
     * @typedef {HTMLElement & { _boundChange?: EventListenerOrEventListenerObject }} ResultCardCheckbox
     * Represents a checkbox element within a result card with an optional stored listener reference.
     */

    /** @type {ResultCardCheckbox} */
    const cbEl = /** @type {ResultCardCheckbox} */ (cb);

    /** @type {string} */
    const checkboxResultId = String(id);
    /**
     * @typedef {(e: Event) => void} ChangeHandler
     */

    /** @type {ChangeHandler} */
    const handler = /** @type {ChangeHandler} */ (function(e) {
      // Normalize event and prevent default when available
      try { if (e && typeof e.preventDefault === 'function') e.preventDefault(); } catch (err) {}
      try { toggleSelectResult(id); } catch (err) { /* swallow */ }
    });
    const prevCb = /** @type {any} */ ((/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.getBound ? (/** @type {any} */ (window)).__tl_helpers.getBound(cb, '_boundChange') : null);
    if (prevCb && cb.removeEventListener) cb.removeEventListener('change', prevCb);
    cb.addEventListener('change', handler);
    if ((/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.setBound) {
      (/** @type {any} */ (window)).__tl_helpers.setBound(cb, '_boundChange', handler);
    } else {
      /** @type {any} */ (cb)._boundChange = handler;
    }
    // Initialize checked state from `state.selectedResults` (guard element type)
    try { if (cb instanceof HTMLInputElement) cb.checked = state.selectedResults.has(id); } catch (e) {}
  });
}
  const grid = $('#resultsGrid');
  const emptyState = $('#emptyState');
  if (filtered.length === 0) {
    grid.classList.add('hidden');
    emptyState.classList.remove('hidden');
    emptyState.classList.add('flex');
  } else {
    grid.classList.remove('hidden');
    grid.innerHTML = filtered.map(r => createResultCard(r)).join('');
    emptyState.classList.add('hidden');
    emptyState.classList.remove('flex');
  }
}

// ============================================
// File Preview Modal with SheetJS
// ============================================

/* Result typedef consolidated at top of file. */

/**
 * Preview a file by result id.
 * @param {string} resultId
 * @returns {void}
 */
function previewFile(resultId) {
  /** @type {Result|undefined} */
  const result = /** @type {Result|undefined} */ (state.results.find(r => r.id === resultId));
  if (!result) return;
  
  state.currentFile = result;
  
  // Update modal header
  const previewTitle = /** @type {HTMLElement|null} */ ($('#previewTitle'));
  if (previewTitle) previewTitle.textContent = `Preview: ${result.name}`;
  
  // Load and parse file based on type
  loadFilePreview(result);
  
  // Show modal
  const previewModal = /** @type {HTMLElement|null} */ ($('#previewModal'));
  if (previewModal) previewModal.classList.remove('hidden');
}

/**
 * @typedef {Object} FilePreviewResult
 * @property {string} id
 * @property {string} name
 * @property {string} type
 * @property {number} [rows]
 * @property {number} [columns]
 * @property {number} [confidence]
 * @property {string} [preview]
 * @property {string} [state]
 * @property {string} [county]
 * @property {string} [handler]
 * @property {number} [timestamp]
 */

/**
 * Load and display a preview for a given result file.
 * @param {FilePreviewResult} result
 * @returns {void}
 */
function loadFilePreview(result) {
  // In real implementation, fetch the actual file
  // For now, show sample data

  if (result.type === 'csv' || result.type === 'xlsx') {
    displayTablePreview(result);
  } else if (result.type === 'json') {
    displayJsonPreview(result);
  }

  displayFileInfo(result);
}

/**
 * @typedef {string[]} TableRow
 * @typedef {TableRow[]} TableData
 */

/**
 * Display a simple table preview for a result.
 * @param {FilePreviewResult} result
 * @returns {void}
 */
function displayTablePreview(result) {
  // Simulated data - in production, load actual file
  /** @type {TableData} */
  const sampleData = [
    ['Candidate', 'Votes', 'Percentage', 'Party'],
    ['Alice Johnson', '45234', '52.3%', 'Democratic'],
    ['Bob Smith', '41123', '47.7%', 'Republican'],
  ];
  
  /** @type {HTMLTableElement} */
  const table = /** @type {HTMLTableElement} */ ($('#previewTable'));
  table.innerHTML = '';
  
  // Headers
  /** @type {HTMLTableSectionElement} */
  const thead = document.createElement('thead');
  thead.innerHTML = `
    <tr>
      ${sampleData[0].map(h => `<th>${escapeHtml(h)}</th>`).join('')}
    </tr>
  `;
  table.appendChild(thead);
  
  // Body
  /** @type {HTMLTableSectionElement} */
  const tbody = document.createElement('tbody');
  sampleData.slice(1, CONFIG.maxPreviewRows + 1).forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = row.map(cell => `<td>${escapeHtml(cell)}</td>`).join('');
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

/**
 * @typedef {{ name: string, votes: number }} Candidate
 * @typedef {{ contest: string, candidates: Candidate[] }} JsonPreviewData
 */

/**
 * Display a JSON preview for a result.
 * @param {FilePreviewResult} result
 * @returns {void}
 */
function displayJsonPreview(result) {
  /** @type {HTMLElement | null} */
  const tabContent = /** @type {HTMLElement | null} */ ($('#tabPreview'));
  /** @type {JsonPreviewData} */
  const sampleJson = {
    contest: 'County Attorney',
    candidates: [
      { name: 'Alice Brown', votes: 45234 },
      { name: 'Bob Smith', votes: 41123 },
    ],
  };

  /** @type {HTMLPreElement} */
  const pre = /** @type {HTMLPreElement} */ (document.createElement('pre'));
  pre.textContent = JSON.stringify(sampleJson, null, 2);
  pre.style.background = 'var(--bg-primary)';
  pre.style.padding = 'var(--spacing-lg)';
  pre.style.borderRadius = 'var(--radius-md)';
  pre.style.overflow = 'auto';

  if (tabContent) {
    tabContent.innerHTML = '';
    tabContent.appendChild(pre);
  }
}

/**
 * @typedef {Object} FileInfo
 * @property {string} name
 * @property {number} [rows]
 * @property {number} [columns]
 * @property {number} [confidence]
 * @property {string} [handler]
 * @property {number|string|Date} [timestamp]
 */

/**
 * Update file info panel with metadata from a result.
 * @param {FileInfo} result
 * @returns {void}
 */
function displayFileInfo(result) {
  $('#infoFileName').textContent = result.name;
  $('#infoRows').textContent = (result.rows || 0).toLocaleString();
  $('#infoColumns').textContent = String(result.columns ?? 'N/A');
  $('#infoConfidence').textContent = (result.confidence || 0).toFixed(1) + '%';
  $('#infoHandler').textContent = result.handler || 'unknown';
  $('#infoTimestamp').textContent = formatDate(result.timestamp || Date.now());
}

// ============================================
// Session Management
// ============================================

function updateSessionsList(sessions = state.sessions) {
  state.sessions = sessions || state.sessions;
  
  const list = $('#sessionsList');
  if (!list) return; // Guard against missing element
  
  if (!state.sessions.length) {
    list.innerHTML = '<p class="text-muted small">No sessions</p>';
    return;
  }
  
  list.innerHTML = state.sessions.map(session => `
    <div class="session-card ${session.id === currentSessionId ? 'active' : ''}">
      <div class="session-id">${session.id}</div>
      <div class="session-progress">
        <span class="session-status ${session.status || 'pending'}"></span>
        ${session.progress || 'Initializing...'}
      </div>
    </div>
  `).join('');
  
  $('#sessionCount').textContent = String(state.sessions.length);
}

/**
 * @typedef {Object} ProgressSessionData
 * @property {string} [session_id]
 * @property {string} [state]
 * @property {string} [phase]
 */

/**
 * Update the small progress card UI with session progress.
 * @param {ProgressSessionData | null | undefined} sessionData
 * @returns {void}
 */
function updateProgressCard(sessionData) {
  /** @type {HTMLElement | null} */
  const progressCard = /** @type {HTMLElement | null} */ ($('#progressCard'));
  if (!progressCard) return; // Element doesn't exist in DOM

  if (!sessionData || sessionData.state === 'IDLE') {
    progressCard.style.display = 'none';
    return;
  }

  progressCard.style.display = 'block';
  /** @type {HTMLElement | null} */
  const progressSessionEl = /** @type {HTMLElement | null} */ ($('#progressSessionId'));
  /** @type {HTMLElement | null} */
  const progressStatusEl = /** @type {HTMLElement | null} */ ($('#progressStatus'));
  /** @type {HTMLElement | null} */
  const progressStagesEl = /** @type {HTMLElement | null} */ ($('#progressStages'));

  if (progressSessionEl) progressSessionEl.textContent = /** @type {string} */ (sessionData.session_id);
  if (progressStatusEl) progressStatusEl.textContent = /** @type {string} */ (sessionData.state);

  // Update phases
  if (progressStagesEl) {
    /** @type {string[]} */
    const phases = ['PREPARE', 'SOURCE', 'RUN', 'REVIEW'];
    const stagesHtml = phases.map(phase => {
      /** @type {string} */
      let className = '';
      if (phase === sessionData.phase) className = 'active';
      else if (phases.indexOf(phase) < phases.indexOf(sessionData.phase)) className = 'completed';
      return `<div class="stage ${className}">${phase}</div>`;
    }).join('');

    progressStagesEl.innerHTML = stagesHtml;
  }
}

/**
 * Update overview strip (run summary + expected artifacts badge).
 * @param {SessionStatePayload | null | undefined} sessionData
 */
function updateOverviewStrip(sessionData) {
  const meta = (sessionData && typeof sessionData === 'object') ? (sessionData.metadata || {}) : {};
  /** @type {HTMLElement | null} */
  const sessionEl = /** @type {HTMLElement | null} */ (document.getElementById('summarySessionId'));
  /** @type {HTMLElement | null} */
  const sourceEl = /** @type {HTMLElement | null} */ (document.getElementById('summarySource'));
  /** @type {HTMLElement | null} */
  const bypassEl = /** @type {HTMLElement | null} */ (document.getElementById('summaryBypass'));
  /** @type {HTMLElement | null} */
  const modeEl = /** @type {HTMLElement | null} */ (document.getElementById('summaryMode'));
  /** @type {HTMLElement | null} */
  const batchEl = /** @type {HTMLElement | null} */ (document.getElementById('summaryBatch'));
  /** @type {HTMLElement | null} */
  const runBadge = /** @type {HTMLElement | null} */ (document.getElementById('runModeBadge'));
  /** @type {HTMLElement | null} */
  const artifactBadge = /** @type {HTMLElement | null} */ (document.getElementById('artifactGuideBadge'));

  const sessionId = sessionData?.session_id || meta.session_id || currentSessionId || '—';
  const source = meta.manual_source || meta.manualSource || meta.manual_source_origin || (/** @type {HTMLInputElement | null} */ (document.querySelector('input[name="fileSource"]:checked')))?.value || 'input';
  const bypass = typeof meta.output_bypass === 'boolean' ? meta.output_bypass : !!(/** @type {HTMLInputElement | null} */ (document.getElementById('outputBypass')))?.checked;
  const mode = (meta.output_mode || meta.mode || 'live').toString().toLowerCase();
  const batch = typeof meta.batch_mode === 'boolean' ? meta.batch_mode : !!(/** @type {HTMLInputElement | null} */ (document.getElementById('batchMode')))?.checked;

  if (sessionEl) sessionEl.textContent = sessionId || '—';
  if (sourceEl) sourceEl.textContent = source || 'input';
  if (bypassEl) bypassEl.textContent = bypass ? 'on' : 'off';
  if (modeEl) modeEl.textContent = batch ? 'batch' : (mode || 'live');
  if (batchEl) batchEl.textContent = batch ? 'on' : 'off';

  const resetBadge = (el) => {
    if (!el) return;
    el.classList.remove('badge-live', 'badge-batch', 'badge-source-input', 'badge-source-uploads');
    el.classList.add('badge-soft');
  };

  if (runBadge) {
    resetBadge(runBadge);
    runBadge.textContent = batch ? 'batch' : (mode || 'live');
    runBadge.classList.add(batch ? 'badge-batch' : 'badge-live');
  }

  if (artifactBadge) {
    resetBadge(artifactBadge);
    const sourceVariant = (source || '').toLowerCase() === 'uploads' ? 'badge-source-uploads' : 'badge-source-input';
    artifactBadge.classList.add(sourceVariant);
    artifactBadge.textContent = (source || 'input');

    // Micro metric: highlight when quality metrics are present in metadata
    if (meta && typeof meta === 'object' && meta.quality_metrics && typeof meta.quality_metrics === 'object') {
      const q = meta.quality_metrics;
      const conf = (typeof q.extraction_confidence === 'number') ? `${q.extraction_confidence.toFixed(1)}%` : null;
      if (conf) artifactBadge.textContent = `${artifactBadge.textContent} • ${conf}`;
      if (typeof q.extraction_confidence === 'number') {
        setQualityPill(Number(q.extraction_confidence));
      }
    } else if (typeof meta?.extraction_confidence === 'number') {
      const confNum = Number(meta.extraction_confidence);
      artifactBadge.textContent = `${artifactBadge.textContent} • ${confNum.toFixed(1)}%`;
      setQualityPill(confNum);
    }
  }
}

/**
 * Extract quality metrics (extraction_confidence) from various log shapes.
 * @param {any} source
 * @returns {{quality_metrics?: any, extraction_confidence?: number} | null}
 */
function coerceQualityPayload(source) {
  if (!source || typeof source !== 'object') return null;
  const qm = source.quality_metrics || source.qualityMetrics || null;
  const conf = source.extraction_confidence ?? source.extractionConfidence ?? (qm && (qm.extraction_confidence ?? qm.extractionConfidence));
  const confNum = (typeof conf === 'number' && Number.isFinite(conf)) ? conf : null;
  if (qm || confNum !== null) {
    return {
      quality_metrics: qm || (confNum !== null ? { extraction_confidence: confNum } : undefined),
      extraction_confidence: confNum !== null ? confNum : undefined,
    };
  }
  return null;
}

/**
 * Merge quality metrics into SessionMirror + overview pill from parser_output events.
 * @param {ParserOutputEvent} log
 */
function ingestQualityMetrics(log) {
  const sid = log?.session_id || currentSessionId;
  if (!sid) return;

  /** @type {{quality_metrics?: any, extraction_confidence?: number} | null} */
  let payload = coerceQualityPayload(log);
  if (!payload) payload = coerceQualityPayload(log?.context);
  if (!payload) payload = coerceQualityPayload(log?.metadata);

  if (!payload) {
    const msg = log?.message;
    if (msg && typeof msg === 'object') {
      payload = coerceQualityPayload(msg);
    } else if (typeof msg === 'string' && msg.trim().startsWith('{')) {
      try {
        const parsed = JSON.parse(msg);
        payload = coerceQualityPayload(parsed);
      } catch (e) {
        /* ignore */
      }
    }
  }

  if (!payload) return;

  const merged = {
    session_id: sid,
    quality_metrics: payload.quality_metrics || SessionMirror.get(sid)?.quality_metrics,
    extraction_confidence: payload.extraction_confidence ?? SessionMirror.get(sid)?.extraction_confidence,
  };

  SessionMirror.upsert(merged);
  updateOverviewStrip({ session_id: sid, metadata: merged });
  if (payload.extraction_confidence !== undefined) {
    setQualityPill(payload.extraction_confidence);
  }
}

/**
 * Ensure a confidence pill with mini-meter is rendered in the Expected Artifacts card.
 * @param {number} confidence
 */
function setQualityPill(confidence) {
  const body = document.querySelector('article[aria-label="Expected artifacts"] .artifact-card-body');
  if (!body) return;
  let pill = document.getElementById('qualityPill');
  if (!pill) {
    pill = document.createElement('div');
    pill.id = 'qualityPill';
    pill.className = 'quality-pill';
    pill.innerHTML = '<span class="pill-label">Confidence</span><span class="pill-value">–</span><span class="mini-meter"><span class="mini-meter-fill"></span></span>';
    body.appendChild(pill);
  }
  const val = Math.max(0, Math.min(100, Number(confidence) || 0));
  const valueEl = pill.querySelector('.pill-value');
  const fillEl = pill.querySelector('.mini-meter-fill');
  if (valueEl) valueEl.textContent = `${val.toFixed(1)}%`;
  if (fillEl instanceof HTMLElement) {
    fillEl.style.width = `${val}%`;
    fillEl.style.opacity = '1';
  }
}

// ============================================
// Event Listeners: Sidebar Controls
// ============================================

// File Source Toggle
$$('input[name="fileSource"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    socket.emit('set_manual_source', {
      session_id: currentSessionId,
      file_source: (/** @type {any} */ (window)).__tl_helpers.targetValue(e),
    });
  });
});

// Output Bypass Toggle
$('#outputBypass').addEventListener('change', () => {
  socket.emit('toggle_output_bypass', {
    session_id: currentSessionId,
  });
});

// Run Parser Button
$$('#btnRunParser, #btnRunParser2').forEach(btn => {
  btn.addEventListener('click', () => {
    const fileSourceEl = document.querySelector('input[name="fileSource"]:checked');
    const fileSource = (fileSourceEl instanceof HTMLInputElement) ? fileSourceEl.value : '';
    const payload = {
      session_id: currentSessionId,
      file_source: fileSource,
    };
    
    // Add direct URLs if selected
    if (fileSource === 'direct') {
      const urls = parseDirectUrlField();
      if (urls.length === 0) {
        showToast('Please enter at least one valid URL', 'warning');
        return;
      }
      payload.direct_urls = urls;
    }
    
    // Add batch mode flag
    const batchModeCheckbox = $('#batchMode');
    if (batchModeCheckbox && (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.elChecked(batchModeCheckbox)) {
      payload.batch_mode = true;
    }
    
    socket.emit('ballot_lens', payload);
    (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.setDisabled($('#btnRunParser2'), true);
    (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.setDisabled($('#btnCancel'), false);
    
    // Update current session in advanced features
    AdvancedFeatures.currentSessionId = currentSessionId;
    
    showToast('Parser started...', 'info');
  });
});

// Cancel Button
$('#btnCancel').addEventListener('click', () => {
  socket.emit('cancel_parser', {
    session_id: currentSessionId,
  });
  (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.setDisabled($('#btnRunParser2'), false);
  (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.setDisabled($('#btnCancel'), true);
});

// ============================================
// Event Listeners: Filters
// ============================================

$('#searchResults').addEventListener('input', (e) => {
  state.filters.search = (/** @type {any} */ (window)).__tl_helpers.targetValue(e);
  renderResults();
});

$('#filterConfidence').addEventListener('input', (e) => {
  state.filters.confidence = parseInt((/** @type {any} */ (window)).__tl_helpers.targetValue(e), 10);
  $('#filterConfidenceValue').textContent = (/** @type {any} */ (window)).__tl_helpers.targetValue(e) + '%+';
  renderResults();
});

$('#filterState').addEventListener('change', (e) => {
  state.filters.state = (/** @type {any} */ (window)).__tl_helpers.targetValue(e);
  renderResults();
});

$('#filterLevel').addEventListener('change', (e) => {
  state.filters.level = (/** @type {any} */ (window)).__tl_helpers.targetValue(e);
  renderLogs();
});

// ============================================
// Event Listeners: Log Drawer
// ============================================

const drawerHandle = document.getElementById('drawerHandle');
if (drawerHandle) {
  // initialize aria-expanded from current drawer state
  try {
    const logDrawerInit = document.getElementById('logDrawer');
    if (logDrawerInit) {
      drawerHandle.setAttribute('aria-expanded', logDrawerInit.classList.contains('expanded') ? 'true' : 'false');
    }
  } catch (e) { /* ignore */ }

  drawerHandle.addEventListener('click', () => {
    const logDrawer = document.getElementById('logDrawer');
    if (logDrawer) {
      logDrawer.classList.toggle('minimized');
      logDrawer.classList.toggle('expanded');
      const expanded = logDrawer.classList.contains('expanded');
      try { drawerHandle.setAttribute('aria-expanded', expanded ? 'true' : 'false'); } catch (e) {}
    }
  });
}

// ============================================
// Log Drawer: Auto-sync with Legacy Sidebar Width
// ============================================

(function syncDrawerToLegacySidebar(){
  const legacySidebar = document.getElementById('sidebar');
  const logDrawer = $('#logDrawer');
  const root = document.documentElement;
  
  function updateDrawerOffset() {
    if (!legacySidebar || !logDrawer) return;
    const width = legacySidebar.offsetWidth;
    if (width > 0) {
      root.style.setProperty('--drawer-left-offset', width + 'px');
      // Keep the drawer full-width; do not offset into the main grid.
      logDrawer.style.left = '';
      logDrawer.style.right = '';
    }
  }
  
  // Initial sync
  setTimeout(updateDrawerOffset, 100); // Let DOM settle
  
  // Sync on resize
  window.addEventListener('resize', updateDrawerOffset);
  
  // Observer for sidebar visibility changes
  const observer = new MutationObserver(() => {
    requestAnimationFrame(updateDrawerOffset);
  });
  if (legacySidebar) {
    observer.observe(legacySidebar, {
      attributes: true,
      attributeFilter: ['style', 'class'],
      characterData: false,
      subtree: false
    });
  }
})();

// ============================================
// Event Listeners: Mobile Sidebars (Unified)
// ============================================

// Initialize unified mobile sidebar controls after DOM is ready so elements exist.
document.addEventListener('DOMContentLoaded', function initUnifiedMobileSidebars(){
  const legacySidebar = document.getElementById('sidebar');
  const rightSidebar = document.querySelector('.sidebar-right');
  const sidebarBackdrop = null;
  const toggleLeftBtn = $('#sidebarToggleBtn');
  const toggleRightBtn = $('#btnToggleRightSidebar');
  const overlay = $('#mobileSidebarOverlay');
  const isDesktop = () => window.innerWidth > 768;

  function setRightCollapsed(collapsed){
    if (!rightSidebar) return;
    const rs = /** @type {HTMLElement} */ (rightSidebar);
    if (collapsed) {
      document.body.classList.add('right-sidebar-collapsed');
      rightSidebar.classList.add('is-collapsed');
      rightSidebar.classList.remove('open', 'sidebar-open', 'centered-tool-window');
      document.body.classList.remove('sidebar-right-open', 'no-scroll');
      if (isDesktop()) {
        setOverlayVisible(false);
      }
      try { document.body.style.overflow = ''; } catch (e) {}
      try {
        rs.style.width = '';
        rs.style.maxWidth = '';
        rs.style.padding = '';
        rs.style.opacity = '';
        rs.style.transform = '';
        rs.style.pointerEvents = '';
        rs.style.overflow = '';
      } catch (e) {}
    } else {
      document.body.classList.remove('right-sidebar-collapsed');
      rightSidebar.classList.remove('is-collapsed');
      if (isDesktop()) {
        setOverlayVisible(false);
      }
      try { document.body.style.overflow = ''; } catch (e) {}
      try {
        rs.style.width = '';
        rs.style.maxWidth = '';
        rs.style.padding = '';
        rs.style.opacity = '';
        rs.style.transform = '';
        rs.style.pointerEvents = '';
        rs.style.overflow = '';
      } catch (e) {}
    }
  }

  /**
   * Keep the toggle's aria-expanded in sync with the sidebar state.
   */
  function syncToggleAria() {
    if (!toggleRightBtn || !rightSidebar) return;
    const isOpen = rightSidebar.classList.contains('open') || rightSidebar.classList.contains('sidebar-open');
    toggleRightBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  }

  /**
   * Toggle floating state for the left menu button when the left sidebar is off-canvas.
   */
  function syncLeftToggleFloating() {
    if (!toggleLeftBtn) return;
    const btn = /** @type {HTMLElement} */ (toggleLeftBtn);
    const sidebar = legacySidebar;
    let presented = false;
    if (sidebar && typeof sidebar.getBoundingClientRect === 'function') {
      const rect = sidebar.getBoundingClientRect();
      const styles = window.getComputedStyle(sidebar);
      const hiddenByStyle = styles.display === 'none' || styles.visibility === 'hidden' || styles.opacity === '0' || styles.pointerEvents === 'none';
      const zeroSize = rect.width <= 0 || rect.height <= 0;
      const vw = window.innerWidth || document.documentElement.clientWidth || 0;
      const offCanvas = rect.right <= 0 || rect.left >= vw;
      presented = !hiddenByStyle && !zeroSize && !offCanvas;
    }
    const shouldFloat = !presented;
    btn.classList.toggle('sidebar-toggle-floating', shouldFloat);
  }

  /**
   * @typedef {HTMLElement} OverlayElement
   */

  /**
   * Set visibility for overlay/backdrop elements and body scroll state.
   * @param {boolean} visible
   * @returns {void}
   */
  function setOverlayVisible(visible) {
    if (isDesktop()) {
      /** @type {OverlayElement[]} */
      const targets = [];
      if (overlay) targets.push(/** @type {OverlayElement} */ (overlay));
      targets.forEach((el) => {
        try {
          el.classList.remove('visible');
          el.setAttribute('aria-hidden', 'true');
        } catch (e) {}
      });
      try { document.body.classList.remove('no-scroll'); } catch (e) {}
      try { document.body.style.overflow = ''; } catch (e) {}
      return;
    }
    /** @type {OverlayElement[]} */
    const targets = [];
    if (overlay) targets.push(/** @type {OverlayElement} */ (overlay));
    targets.forEach((el) => {
      try {
        if (visible) el.classList.add('visible'); else el.classList.remove('visible');
        el.setAttribute('aria-hidden', visible ? 'false' : 'true');
      } catch (e) {}
    });
    try {
      if (visible) document.body.classList.add('no-scroll'); else document.body.classList.remove('no-scroll');
    } catch (e) {}
    try {
      document.body.style.overflow = visible ? 'hidden' : '';
    } catch (e) {}
  }

  // Ensure initial state starts closed and aria matches
  if (rightSidebar) {
    rightSidebar.classList.remove('open', 'sidebar-open', 'centered-tool-window');
    document.body.classList.remove('sidebar-right-open', 'no-scroll');
    syncToggleAria();
  }
  if (isDesktop()) {
    setRightCollapsed(false);
    syncToggleAria();
  }
  syncLeftToggleFloating();

  function closeAll() {
    if (legacySidebar) legacySidebar.classList.remove('sidebar-open');
    if (rightSidebar) {
      rightSidebar.classList.remove('open');
      rightSidebar.classList.remove('sidebar-open');
      rightSidebar.classList.remove('centered-tool-window');
    }
    if (isDesktop()) {
      setRightCollapsed(true);
    }
    setOverlayVisible(false);
    document.body.classList.remove('no-scroll');
    document.body.classList.remove('sidebar-right-open');
    syncToggleAria();
    if (toggleLeftBtn) {
      try { toggleLeftBtn.setAttribute('aria-expanded', 'false'); } catch (e) {}
    }
    syncLeftToggleFloating();
  }

  function openLeft() {
    if (!legacySidebar) return;
    legacySidebar.classList.add('sidebar-open');
    setOverlayVisible(true);
    document.body.classList.add('no-scroll');
    if (toggleLeftBtn) {
      try { toggleLeftBtn.setAttribute('aria-expanded', 'true'); } catch (e) {}
    }
    syncLeftToggleFloating();
  }

  function openRight() {
    if (!rightSidebar) return;
    setRightCollapsed(false);
    rightSidebar.classList.add('open');
    rightSidebar.classList.add('sidebar-open');
    // Always dock to the right; keep the centered overlay class off for predictable layout
    rightSidebar.classList.remove('centered-tool-window');
    setOverlayVisible(true);
    document.body.classList.add('no-scroll');
    document.body.classList.add('sidebar-right-open');
    syncToggleAria();
  }

  // Left sidebar toggle is handled by the consolidated controller below; keep right sidebar bindings here.

  // Modern right sidebar toggle
  if (toggleRightBtn) {
    const rightBtnAny = /** @type {any} */ (toggleRightBtn);
    if (!rightBtnAny.__tlRightBound) {
      rightBtnAny.__tlRightBound = true;
      toggleRightBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!rightSidebar) return;
        if (isDesktop()) {
          const isCollapsed = document.body.classList.contains('right-sidebar-collapsed');
          setRightCollapsed(!isCollapsed);
          try { toggleRightBtn.setAttribute('aria-expanded', isCollapsed ? 'true' : 'false'); } catch (e) {}
          syncToggleAria();
          return;
        }
        const isOpen = rightSidebar.classList.contains('open');
        if (isOpen) closeAll(); else openRight();
        syncToggleAria();
      });
    }
  }

  // Left sidebar toggle (ensure it always toggles the unified controller)
  if (toggleLeftBtn) {
    const leftBtnAny = /** @type {any} */ (toggleLeftBtn);
    if (!leftBtnAny.__tlLeftBound) {
      leftBtnAny.__tlLeftBound = true;
      toggleLeftBtn.addEventListener('click', (e) => {
        e.preventDefault();
        try {
          e.stopPropagation();
          /** @type {any} */ (e).__tlSidebarHandled = true;
        } catch (err) {}
        if (!legacySidebar) return;
        const isOpen = legacySidebar.classList.contains('sidebar-open');
        if (isOpen) closeAll(); else openLeft();
      });
    }
  }

  function syncRightSidebarMode() {
    if (!rightSidebar) return;
    if (isDesktop()) {
      return;
    }
    // On mobile, never keep the desktop-collapsed state.
    setRightCollapsed(false);
    rightSidebar.classList.remove('is-collapsed');
    document.body.classList.remove('right-sidebar-collapsed');
  }

  syncRightSidebarMode();
  window.addEventListener('resize', syncRightSidebarMode);
  window.addEventListener('resize', syncLeftToggleFloating);

  if (legacySidebar && typeof MutationObserver !== 'undefined') {
    const leftObserver = new MutationObserver(() => {
      try { syncLeftToggleFloating(); } catch (e) {}
    });
    leftObserver.observe(legacySidebar, { attributes: true, attributeFilter: ['style', 'class'] });
  }

  // Backdrop/overlay clicks close all (ensure both elements are covered)
  if (overlay) {
    overlay.addEventListener('click', closeAll);
  }

  // Escape key closes all sidebars
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeAll();
  });

  // Delegated fallback: ensure toggle buttons work even if elements are replaced dynamically.
  // This listens at document level for clicks on the toggle buttons and invokes the
  // exposed open/close helpers to keep behavior robust against DOM replacements.
  document.addEventListener('click', function delegatedSidebarToggle(e) {
    try {
      if (__tl_window.__tl_sidebarUnified) return;
      const tgt = /** @type {any} */ (e.target);
      const btn = (tgt && tgt.closest) ? tgt.closest('#btnToggleRightSidebar') : null;
      if (!btn) return;
      e.preventDefault();
      // Dev-only health logging: only on localhost to avoid noise in prod
      const isLocal = (typeof window !== 'undefined') && (window.location && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'));
      if (isLocal && console && console.debug) console.debug('[health] delegatedSidebarToggle invoked for', btn.id || btn);
      if (btn.id === 'btnToggleRightSidebar') {
        if (typeof window.openRight === 'function') return window.openRight();
        // fallback: toggle classes directly
        const rs = document.querySelector('.sidebar-right');
        if (!rs) return;
        const isOpen = rs.classList.contains('open');
        if (isOpen) {
          rs.classList.remove('open', 'sidebar-open', 'centered-tool-window');
          document.body.classList.remove('no-scroll', 'sidebar-right-open');
          syncToggleAria();
          if (isLocal && console && console.debug) console.debug('[health] right sidebar closed via delegate');
        } else {
            rs.classList.remove('centered-tool-window');
            rs.classList.add('open', 'sidebar-open');
          document.body.classList.add('no-scroll', 'sidebar-right-open');
          syncToggleAria();
          if (isLocal && console && console.debug) console.debug('[health] right sidebar opened via delegate');
        }
        return;
      }
    } catch (err) {
      /* noop */
    }
  }, true);

// Dev-only: confirm delegated listener is installed when running locally
try {
  if (typeof window !== 'undefined' && window.location && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    if (console && console.debug) console.debug('[health] delegatedSidebarToggle listener installed');
  }
} catch (err) {
  /* noop */
}

  // Auto-close on resize to desktop
  window.addEventListener('resize', () => {
    if (window.innerWidth > 768) closeAll();
  });

  // Observe sidebar class changes (e.g., external scripts) to keep aria-expanded honest
  try {
    if (rightSidebar && typeof MutationObserver !== 'undefined') {
      const mo = new MutationObserver(syncToggleAria);
      mo.observe(rightSidebar, { attributes: true, attributeFilter: ['class'] });
    }
  } catch (e) { /* ignore */ }

  // Final sync on load
  syncToggleAria();

  // Defensive: close on outside click when sidebar is open and user taps/tabs away
  document.addEventListener('click', (ev) => {
    if (!rightSidebar || !toggleRightBtn) return;
    const isOpen = rightSidebar.classList.contains('open');
    if (!isOpen) return;
    const target = ev.target;
    const insideToggle = (target instanceof Node) && toggleRightBtn.contains(target);
    const insideSidebar = (target instanceof Node) && rightSidebar.contains(target);
    if (!insideToggle && !insideSidebar) {
      closeAll();
      syncToggleAria();
    }
  }, true);
  // Expose control hooks for automated tests and debug consoles
  try {
    window.openLeft = openLeft;
    window.openRight = openRight;
    window.closeAll = closeAll;
    window.setOverlayVisible = setOverlayVisible;
  } catch (e) {
    /* ignore */
  }
});

const btnClearLogs = $('#btnClearLogs');
if (btnClearLogs) {
  btnClearLogs.addEventListener('click', () => {
    state.logs = [];
    renderLogs();
    updateLogCounts();
    showToast('Logs cleared', 'info');
  });
}

const btnCopyLogs = $('#btnCopyLogs');
if (btnCopyLogs) {
  btnCopyLogs.addEventListener('click', async () => {
    const text = state.logs.map(l => {
      const ts = new Date(l.timestamp).toLocaleTimeString();
      const typeLabel = l.type ? `[${l.type}]` : '';
      return `[${ts}] ${l.level} ${typeLabel} ${l.message}`;
    }).join('\n');
    try {
      await navigator.clipboard.writeText(text || '');
      showToast('Logs copied to clipboard', 'success');
    } catch (err) {
      showToast('Clipboard not available. Use Export instead.', 'warning');
    }
  });
}

const btnExportLogs = $('#btnExportLogs');
if (btnExportLogs) {
  btnExportLogs.addEventListener('click', () => {
    const csv = state.logs.map(l => 
      `${new Date(l.timestamp).toISOString()},${l.level},${l.type},"${l.message.replace(/"/g, '""')}"`
    ).join('\n');
    
    const blob = new Blob(
      [/** @type {any} */ ('timestamp,level,type,message\n' + csv)],
      { type: 'text/csv' }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `parser_logs_${Date.now()}.csv`;
    a.click();
    showToast('Logs exported', 'success');
  });
}

const btnToggleScroll = $('#btnToggleScroll');
if (btnToggleScroll) {
  btnToggleScroll.textContent = state.autoScroll ? 'Pin' : 'Unpin';
  btnToggleScroll.addEventListener('click', () => {
    state.autoScroll = !state.autoScroll;
    btnToggleScroll.textContent = state.autoScroll ? 'Pin' : 'Unpin';
    showToast(`Auto-scroll ${state.autoScroll ? 'enabled' : 'disabled'}`, 'info');
  });
}

// ============================================
// Event Listeners: Modal
// ============================================

function hidePreviewModal(options) {
  const opts = options || {};
  const showRestore = !!opts.showRestore;
  const previewModal = $('#previewModal');
  if (previewModal) previewModal.classList.add('hidden');
  if (showRestore && state.currentFile) {
    try {
      if (typeof ModalRestoreBanner !== 'undefined') {
        ModalRestoreBanner.show('preview', {
          title: 'Preview hidden',
          detail: state.currentFile.name || 'Reopen last preview',
          buttonLabel: 'Reopen preview',
          onRestore: () => {
            if (state.currentFile) {
              previewFile(state.currentFile.id);
            }
          }
        });
      }
    } catch (e) { /* noop */ }
  } else {
    try {
      if (typeof ModalRestoreBanner !== 'undefined') {
        ModalRestoreBanner.clear('preview');
      }
    } catch (e) { /* noop */ }
  }
}

const btnClosePreview = $('#btnClosePreview');
if (btnClosePreview) {
  btnClosePreview.addEventListener('click', () => {
    hidePreviewModal({ showRestore: true });
  });
}

const btnClosePreviewAlt = $('#btnClosePreviewAlt');
if (btnClosePreviewAlt) {
  btnClosePreviewAlt.addEventListener('click', () => {
    hidePreviewModal({ showRestore: true });
  });
}

// Tab switching in modal
    $$('.tab-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    // Remove active from all
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-content').forEach(c => c.classList.remove('active'));
    
    // Add active to clicked (safe guard)
    const tgt = (e && e.target && (e.target instanceof Element)) ? e.target : null;
    if (tgt) {
      tgt.classList.add('active');
      const tabName = tgt.getAttribute('data-tab') || '';
      const tabEl = tabName ? document.querySelector(`#tab${tabName.charAt(0).toUpperCase() + tabName.slice(1)}`) : null;
      if (tabEl instanceof Element) tabEl.classList.add('active');
    }
  });
});

$('#btnDownloadPreview')?.addEventListener('click', () => {
  if (state.currentFile) {
    downloadFile(state.currentFile.id);
  }
});

// ============================================
// Event Listeners: Results
// ============================================

/**
 * @typedef {Set<string>} SelectedResultsSet
 * @typedef {HTMLButtonElement} BulkExportButton
 */

/**
 * Toggle selection of a result id in the selectedResults set.
 * @param {string} resultId
 * @returns {void}
 */
function toggleSelectResult(resultId) {
  if (state.selectedResults.has(resultId)) {
    state.selectedResults.delete(resultId);
  } else {
    state.selectedResults.add(resultId);
  }
  
  // Update button state (guard + typed cast)
  const btn = /** @type {BulkExportButton | null} */ ($('#btnBulkExport'));
  if (btn) btn.disabled = state.selectedResults.size === 0;
}

$('#btnBulkExport')?.addEventListener('click', () => {
  if (state.selectedResults.size === 0) {
    showToast('No results selected', 'warning');
    return;
  }
  
  const selected = Array.from(state.selectedResults).map(id => 
    state.results.find(r => r.id === id)
  ).filter(Boolean);
  
  showToast(`Exporting ${selected.length} file(s)...`, 'info');
  
  // In production, fetch files and create ZIP
  setTimeout(() => {
    showToast(`Successfully exported ${selected.length} file(s)`, 'success');
  }, 1000);
});

  {
    const btn = $('#btnRefreshResults');
    if (btn) {
      btn.addEventListener('click', (ev) => {
        // Ignore synthetic/programmatic clicks; accept only user-initiated events
        try { if (ev && ev.isTrusted === false) return; } catch (e) { /* ignore */ }
        // Guard against accidental double-fire while an async refresh is in flight.
        if (btn.dataset.busy === '1') return;
        btn.dataset.busy = '1';
        btn.classList.add('is-loading');
        // Slight delay on the toast to avoid blink perception on fast refreshes
        const toastTimer = setTimeout(() => showToast('Refreshing results...', 'info'), 180);
        // In production, fetch updated results from API
        setTimeout(() => {
          btn.dataset.busy = '0';
          btn.classList.remove('is-loading');
          clearTimeout(toastTimer);
          showToast('Results refreshed', 'success');
        }, 420);
      });
    }
  }

// ============================================
// File Operations (Stubs for Production)
// ============================================

/* Result typedef consolidated at top of file. */

/**
 * Trigger a download for a result by id.
 * @param {string} resultId
 * @returns {void}
 */
function downloadFile(resultId) {
  /** @type {Result|undefined} */
  const result = /** @type {Result|undefined} */ (state.results.find(r => r.id === resultId));
  if (!result) return;
  
  showToast(`Downloading ${result.name}...`, 'info');
  
  // In production: fetch actual file
  setTimeout(() => {
    showToast(`${result.name} downloaded`, 'success');
  }, 1000);
}

// ============================================
// Command Palette
// ============================================

const commands = [
  { title: 'Run Parser', description: 'Start parsing', shortcut: 'Ctrl+Enter', action: () => (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.safeClick($('#btnRunParser2')) },
  { title: 'Cancel Parser', description: 'Stop parsing', shortcut: 'Ctrl+Shift+C', action: () => (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.safeClick($('#btnCancel')) },
  { title: 'Clear Logs', description: 'Clear debug console', shortcut: 'Ctrl+K', action: () => (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.safeClick($('#btnClearLogs')) },
  { title: 'Toggle Theme', description: 'Switch dark/light mode', shortcut: 'Ctrl+Shift+T', action: () => toggleTheme() },
  { title: 'Export Logs', description: 'Download debug logs', shortcut: 'Ctrl+Shift+E', action: () => (/** @type {any} */ (window)).__tl_helpers.safeClick($('#btnExportLogs')) },
];

// Safety: ensure overlays start hidden even if cache or styles misbehave
const commandPaletteInit = $('#commandPalette');
if (commandPaletteInit) commandPaletteInit.classList.add('hidden');
const previewModalInit = $('#previewModal');
if (previewModalInit) previewModalInit.classList.add('hidden');
const promptModalInit = $('#promptModal');
if (promptModalInit) promptModalInit.classList.add('hidden');

const btnCommandPalette = $('#btnCommandPalette');
if (btnCommandPalette) {
  btnCommandPalette.addEventListener('click', () => {
    const commandPalette = $('#commandPalette');
    const commandInput = $('#commandInput');
    if (commandPalette) commandPalette.classList.remove('hidden');
    if (commandInput) commandInput.focus();
  });
}

const commandInput = $('#commandInput');
if (commandInput) {
  commandInput.addEventListener('input', (e) => {
    const query = (/** @type {any} */ (window)).__tl_helpers.targetValue(e).toLowerCase();
    const results = commands.filter(c => 
      c.title.toLowerCase().includes(query) || 
      c.description.toLowerCase().includes(query)
    );
    
    const commandResults = $('#commandResults');
    if (commandResults) {
      commandResults.innerHTML = results.map((cmd, idx) => `
        <div class="command-item" data-idx="${idx}">
          <div class="command-text">
            <div class="command-title">${cmd.title}</div>
            <div class="command-description">${cmd.description}</div>
          </div>
          <div class="command-shortcut">${cmd.shortcut}</div>
        </div>
      `).join('');
      // Attach non-inline handlers
      Array.from(commandResults.querySelectorAll('.command-item')).forEach((el) => {
        try {
          const idxAttr = el.getAttribute('data-idx');
          const i = idxAttr ? Number(idxAttr) : NaN;
          if (!isNaN(i)) el.addEventListener('click', () => executeCommand(i));
        } catch (/** @type {any} */ _e) { /* noop */ }
      });
    }
  });
}

/**
 * @typedef {Object} CommandEntry
 * @property {string} title
 * @property {string} description
 * @property {string} shortcut
 * @property {() => void} action
 */

/**
 * Execute a command by its index in the commands array.
 * @param {number} index
 * @returns {void}
 */
function executeCommand(index) {
  commands[index].action();
  const commandPalette = $('#commandPalette');
  if (commandPalette) commandPalette.classList.add('hidden');
}

// Close command palette on ESC
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const commandPalette = $('#commandPalette');
    if (commandPalette) commandPalette.classList.add('hidden');
    const previewModal = $('#previewModal');
    if (previewModal && !previewModal.classList.contains('hidden')) {
      hidePreviewModal({ showRestore: true });
    }
    const promptModal = $('#promptModal');
    if (promptModal && !promptModal.classList.contains('hidden')) {
      hidePrompt({ preserveState: true, showRestore: true, reason: 'close' });
    }
  }
  
  // Open command palette with Ctrl+Shift+P
    if (e.ctrlKey && e.shiftKey && e.key === 'P') {
    e.preventDefault();
    (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.safeClick($('#btnCommandPalette'));
  }
});

// ============================================
// Prompt Handling (interactive server prompts)
// ============================================

const promptTitleEl = $('#promptTitle');
const promptMessageEl = $('#promptMessage');
const promptInputEl = $('#promptInput');
const promptSearchEl = $('#promptSearch');
const promptOptionsEl = $('#promptOptions');

/* PromptOption typedef consolidated at top of file. */

/**
 * Context object that may accompany parser_output prompt logs.
 * @typedef {Object} PromptContext
 * @property {Array<string>} [urls] - Direct URL list for URL-selection prompts.
 * @property {Array<string|Object>} [options] - Contest/options array (strings or objects).
 * @property {Object.<string, any>} [processed] - Processed-info map keyed by URL.
 * @property {string} [title] - Optional prompt title override.
 * @property {string} [placeholder] - Optional placeholder text for prompt input/search.
 * @property {Object.<string, any>} [metadata] - Any additional metadata.
 */

/**
 * Parser output event (lightweight subset used by handlePromptLog).
 * Reuses the broader ParserOutputEvent typedef if present; duplicated fields here
 * for clarity and local tooling.
 * @typedef {Object} LocalParserOutputEvent
 * @property {string} [type]
 * @property {string} [message]
 * @property {string} [session_id]
 * @property {PromptContext} [context]
 */

/**
 * Inspect incoming parser_output log entries and surface interactive prompts.
 * @param {LocalParserOutputEvent} data
 */
function handlePromptLog(data) {
  ErrorBoundary.safeExecute(() => {
    /** @type {string} */
    const message = typeof data?.message === 'string' ? data.message : '';
    /** @type {boolean} */
    const isPrompt = data?.type === 'prompt' || message.toUpperCase().includes('[PROMPT]');
    /** @type {PromptContext} */
    const ctx = /** @type {PromptContext} */ (data?.context || {});
    /** @type {PromptOption[]} */
    let options = [];

    // DEBUG: Log what we're receiving
    console.debug('[handlePromptLog] Received data:', {
      messagePreview: message.substring(0, 100),
      hasContext: !!ctx,
      contextKeys: Object.keys(ctx || {}),
      contextData: ctx
    });

    // URL selection prompt
    if (Array.isArray(ctx.urls) && ctx.urls.length) {
      console.debug('[handlePromptLog] Found URLs in context:', ctx.urls.length);
      options = ctx.urls.map((u, idx) => (/** @type {PromptOption} */ ({
        index: idx + 1,
        label: u,
        meta: ctx.processed && ctx.processed[u] ? ctx.processed[u].status || '' : '',
      })));
    }

    // Contest/options style prompt
    if (!options.length && Array.isArray(ctx.options) && ctx.options.length) {
      console.debug('[handlePromptLog] Found options in context:', ctx.options.length);
      options = ctx.options.map((opt, idx) => {
        if (typeof opt === 'string') {
          const m = opt.match(/^\s*\[(\d+)\]\s+(.+?)(?:\s+\(([^)]+)\))?\s*$/);
          if (m) return /** @type {PromptOption} */ ({ index: Number(m[1]), label: m[2], meta: m[3] || '' });
          return /** @type {PromptOption} */ ({ index: idx + 1, label: opt, meta: '' });
        }
        if (opt && typeof opt === 'object') {
          return /** @type {PromptOption} */ ({
            index: Number(opt.index ?? idx + 1),
            label: opt.label || opt.title || opt.name || `Option ${idx + 1}`,
            meta: opt.meta || opt.summary || '',
            metadata: opt
          });
        }
        return /** @type {PromptOption} */ ({ index: idx + 1, label: String(opt), meta: '' });
      });
    }

    if (isPrompt && message) {
      console.debug('[handlePromptLog] Displaying prompt with', options.length, 'options');
      showPrompt({
        title: ctx.title || 'Action required',
        message,
        options,
        placeholder: ctx.placeholder,
      });
    } else {
      console.warn('[handlePromptLog] Not a prompt or empty message. isPrompt:', isPrompt, 'message:', message.substring(0, 50));
    }
  }, 'handlePromptLog');
}

/**
 * @typedef {Object} IncomingContestOption
 * @property {number|string} [index]
 * @property {string} [label]
 * @property {string} [name]
 * @property {string} [title]
 * @property {string} [meta]
 * @property {Object<string, any>} [metadata]
 */

/* ContestOptionsPayload typedef consolidated at top of file. */

/**
 * Normalized choice used by the UI prompt.
 * @typedef {Object} ContestOptionChoice
 * @property {number} index
 * @property {string} label
 * @property {string} meta
 */

/**
 * Handle incoming contest_options socket payload and surface a selection prompt.
 * @param {ContestOptionsPayload} payload
 */
function handleContestOptions(payload) {
  ErrorBoundary.safeExecute(() => {
    /** @type {ContestOptionChoice[]} */
    /**
     * @typedef {Object} IncomingContestOption
     * @property {number|string} [index]
     * @property {string} [label]
     * @property {string} [name]
     * @property {string} [title]
     * @property {string} [meta]
     * @property {Object.<string, any>} [metadata]
     */

    /**
     * @typedef {Object} ContestOptionChoice
     * @property {number} index
     * @property {string} label
     * @property {string} meta
     */

    /** @type {ContestOptionChoice[]} */
    const options = Array.isArray(payload?.options)
      ? /** @type {ContestOptionChoice[]} */ (payload.options.map(
        /**
         * @param {IncomingContestOption|string} opt
         * @param {number} idx
         * @returns {ContestOptionChoice}
         */
        (opt, idx) => ({
        index: Number((opt && typeof opt === 'object' ? (opt.index ?? idx + 1) : (idx + 1))),
        label: (typeof opt === 'string'
          ? opt
          : (opt && (opt.label || opt.name || opt.title)) || `Option ${idx + 1}`),
        meta: (opt && typeof opt === 'object' ? (opt.meta || (opt.metadata && opt.metadata.summary) || '') : '')
        })
      ))
      : [];

    if (!options.length) {
      console.warn('[handleContestOptions] No options provided');
      return;
    }

    /** @type {Object<string, any>} */
    const ctx = payload?.context || {};
    /** @type {string} */
    const message = ctx.message || 'Select a contest';

    showPrompt({
      title: 'Select Contest',
      message,
      options,
      placeholder: 'Search or click to choose',
    });
  }, 'handleContestOptions');
}

function renderPromptOptions(filterText = '') {
  ErrorBoundary.safeExecute(() => {
    if (!promptOptionsEl) return;
    const needle = filterText.toLowerCase();
    const filtered = activePromptOptions.filter(opt => {
      const label = String(opt.label || '').toLowerCase();
      const meta = String(opt.meta || '').toLowerCase();
      const scopeLabel = opt.metadata?.scope_label ? String(opt.metadata.scope_label).toLowerCase() : '';
      return !needle || label.includes(needle) || meta.includes(needle) || scopeLabel.includes(needle);
    });

    if (!filtered.length) {
      promptOptionsEl.innerHTML = '<div class="text-muted small">No options. Enter a response above.</div>';
      return;
    }

    // Group options by bundle_key if available (P1.1 Bundle Grouping)
    const groups = new Map();
    filtered.forEach(opt => {
      const meta = opt.metadata || {};
      const bundleKey = meta.bundle_key || meta.bundle_parent_index;
      
      if (bundleKey && meta.bundle_mode === 'aggregate') {
        if (!groups.has(bundleKey)) {
          groups.set(bundleKey, {
            parent: opt,
            children: [],
            expanded: bundleExpandedState.get(bundleKey) || false
          });
        }
      } else if (bundleKey && meta.bundle_member) {
        const group = groups.get(bundleKey);
        if (group) group.children.push(opt);
        else if (!groups.has(opt.index)) {
          groups.set(opt.index, { parent: opt, children: [], expanded: false });
        }
      } else {
        if (!groups.has(opt.index)) {
          groups.set(opt.index, { parent: opt, children: [], expanded: false });
        }
      }
    });

    promptOptionsEl.innerHTML = '';

    // Standard rendering (virtual scroll removed to avoid inline style mutations under CSP)
    for (const [key, group] of groups) {
      const elem = renderGroupElement(group, key);
      promptOptionsEl.appendChild(elem);
    }
  
  updateSelectionSummary();
  }, 'renderPromptOptions');
}

// Helper: Render a group element (for both virtual and standard rendering)
/* PromptOption typedef consolidated at top of file. */

/**
 * Group descriptor for bundled prompt options.
 * @typedef {Object} PromptGroup
 * @property {PromptOption} parent
 * @property {PromptOption[]} children
 * @property {boolean} expanded
 */

/**
 * Render a group element (for both virtual and standard rendering)
 * @param {PromptGroup} group
 * @param {string|number} key
 * @returns {HTMLElement}
 */
function renderGroupElement(group, key) {
  /** @type {PromptOption} */
  const parent = group.parent;
  /** @type {PromptOption[]} */
  const children = group.children || [];
  /** @type {boolean} */
  const expanded = Boolean(group.expanded);

  if (!children.length) {
    // Single option (not grouped)
    return /** @type {HTMLElement} */ (createPromptOptionButton(parent));
  }

  // Bundle with children
  /** @type {HTMLDivElement} */
  const wrapper = document.createElement('div');
  wrapper.className = 'prompt-bundle';

  // Bundle header with toggle
  /** @type {HTMLDivElement} */
  const header = document.createElement('div');
  header.className = 'prompt-bundle-header';

  /** @type {HTMLButtonElement} */
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'prompt-bundle-toggle';
  toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
  toggle.textContent = expanded ? '▼' : '▶';
  toggle.addEventListener('click', (e) => {
    e.preventDefault();
    bundleExpandedState.set(key, !bundleExpandedState.get(key));
    renderPromptOptions((/** @type {any} */ (window)).__tl_helpers.elValue(promptSearchEl) || '');
  });

  header.appendChild(toggle);
  header.appendChild(/** @type {HTMLElement} */ (createPromptOptionButton(parent, { bundled: true })));
  wrapper.appendChild(header);

  // Children (show if expanded)
  if (expanded && children.length) {
    /** @type {HTMLDivElement} */
    const childContainer = document.createElement('div');
    childContainer.className = 'prompt-bundle-children';
    children.forEach(/** @param {PromptOption} child */ (child) => {
      const childBtn = /** @type {HTMLElement} */ (createPromptOptionButton(child, { isChild: true }));
      childContainer.appendChild(childBtn);
    });
    wrapper.appendChild(childContainer);
  }

  return wrapper;
}

// Stub for old rendering (replaced above)
function renderPromptOptions_OLD(filterText = '') {
  ErrorBoundary.safeExecute(() => {
    if (!promptOptionsEl) return;
    const needle = filterText.toLowerCase();
    const filtered = activePromptOptions.filter(opt => {
      const label = String(opt.label || '').toLowerCase();
      const meta = String(opt.meta || '').toLowerCase();
      const scopeLabel = opt.metadata?.scope_label ? String(opt.metadata.scope_label).toLowerCase() : '';
      return !needle || label.includes(needle) || meta.includes(needle) || scopeLabel.includes(needle);
    });

    if (!filtered.length) {
      promptOptionsEl.innerHTML = '<div class="text-muted small">No options. Enter a response above.</div>';
      return;
    }

    // Group options by bundle_key if available (P1.1 Bundle Grouping)
    const groups = new Map();
    filtered.forEach(opt => {
      const meta = opt.metadata || {};
      const bundleKey = meta.bundle_key || meta.bundle_parent_index;
      
      if (bundleKey && meta.bundle_mode === 'aggregate') {
        if (!groups.has(bundleKey)) {
          groups.set(bundleKey, {
            parent: opt,
            children: [],
            expanded: bundleExpandedState.get(bundleKey) || false
          });
        }
      } else if (bundleKey && meta.bundle_member) {
        const group = groups.get(bundleKey);
        if (group) group.children.push(opt);
        else if (!groups.has(opt.index)) {
          groups.set(opt.index, { parent: opt, children: [], expanded: false });
        }
      } else {
        if (!groups.has(opt.index)) {
          groups.set(opt.index, { parent: opt, children: [], expanded: false });
        }
      }
    });

    promptOptionsEl.innerHTML = '';

    // Render each group
    for (const [key, group] of groups) {
      const { parent, children, expanded } = group;
      
      if (!children.length) {
        // Single option (not grouped)
        const btn = createPromptOptionButton(parent);
        promptOptionsEl.appendChild(btn);
      } else {
        // Bundle with children
        const wrapper = document.createElement('div');
        wrapper.className = 'prompt-bundle';
        
        // Bundle header with toggle
        const header = document.createElement('div');
        header.className = 'prompt-bundle-header';
        
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'prompt-bundle-toggle';
        toggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
        toggle.textContent = expanded ? '▼' : '▶';
        toggle.addEventListener('click', (e) => {
          e.preventDefault();
          bundleExpandedState.set(key, !bundleExpandedState.get(key));
          renderPromptOptions((/** @type {any} */ (window)).__tl_helpers.elValue(promptSearchEl) || '');
        });
        
        header.appendChild(toggle);
        header.appendChild(createPromptOptionButton(parent, { bundled: true }));
        wrapper.appendChild(header);
        
        // Children (show if expanded)
        if (expanded && children.length) {
          const childContainer = document.createElement('div');
          childContainer.className = 'prompt-bundle-children';
            // Render each child in the bundle with explicit types
            /** @type {PromptOption[]} */
            (children).forEach(/** @param {PromptOption} child */ (child) => {
            /** @type {HTMLElement} */
            const childBtn = /** @type {HTMLElement} */ (createPromptOptionButton(child, { isChild: true }));
            childContainer.appendChild(childBtn);
            });
          wrapper.appendChild(childContainer);
      }
      
      promptOptionsEl.appendChild(wrapper);
    }
  }
  
  updateSelectionSummary();
  }, 'renderPromptOptions');
}

/* PromptOption / CreateBtnOptions typedefs consolidated at top of file. */

/**
 * Create a prompt option button element for the prompt modal.
 * @param {PromptOption} opt
 * @param {CreateBtnOptions} [options]
 * @returns {HTMLElement | null}
 */
function createPromptOptionButton(opt, options = {}) {
  return ErrorBoundary.safeExecute(() => {
    const { bundled = false, isChild = false } = options;
    /** @type {HTMLButtonElement} */
    const btn = /** @type {HTMLButtonElement} */ (document.createElement('button'));
    btn.type = 'button';
    btn.className = 'prompt-option' + (isChild ? ' prompt-option-child' : '') + (bundled ? ' prompt-option-bundled' : '');
    
    /** @type {Object.<string, any>} */
    const meta = opt.metadata || {};
    const bundleSize = meta.bundle_child_count ? meta.bundle_child_count + 1 : 0;
    /** @type {string[]} */
    const badges = [];
    
    // P1.2 Metadata Badges
    if (meta.scope_label) badges.push(`<span class="badge badge-scope">${escapeHtml(meta.scope_label)}</span>`);
    if (bundleSize && bundled) badges.push(`<span class="badge badge-bundle">${bundleSize} variations</span>`);
    if (Array.isArray(meta.counties) && meta.counties.length > 1) badges.push(`<span class="badge badge-counties">${meta.counties.length} counties</span>`);
    if (meta.year) badges.push(`<span class="badge badge-year">${meta.year}</span>`);
    if (typeof meta.confidence === 'number') {
      const confClass = meta.confidence >= 0.85 ? 'high' : meta.confidence >= 0.70 ? 'medium' : 'low';
      badges.push(`<span class="badge badge-confidence badge-conf-${confClass}">conf ${meta.confidence.toFixed(2)}</span>`);
    }
    if (meta.variants || (Array.isArray(meta.contest_ids) && meta.contest_ids.length > 1)) {
      const count = meta.variants || meta.contest_ids.length;
      badges.push(`<span class="badge badge-variants">${count} IDs</span>`);
    }
    
    // P2.1 Multi-Select Checkbox
    const hasCheckbox = !isChild && activePromptOptions.length > 1;
    let checkboxHtml = '';
    if (hasCheckbox) {
      const isChecked = selectedPromptOptions.has(opt.index);
      checkboxHtml = `<input type="checkbox" class="prompt-option-checkbox" value="${escapeHtml(String(opt.index))}" ${isChecked ? 'checked' : ''} />`;
    }
    
    btn.innerHTML = `
      ${checkboxHtml}
      <div>
        <div class="label">[${opt.index ?? opt.value ?? '?'}] ${escapeHtml(opt.label || '')}</div>
        ${badges.length ? `<div class="badges">${badges.join('')}</div>` : ''}
        ${opt.meta ? `<div class="meta">${escapeHtml(opt.meta)}</div>` : ''}
      </div>
    `;
    
    // Checkbox event handler
    /** @type {HTMLInputElement | null} */
    const checkbox = /** @type {HTMLInputElement | null} */ (btn.querySelector('.prompt-option-checkbox'));
    if (checkbox) {
      checkbox.addEventListener('change', (e) => {
        if ((/** @type {any} */ (window)).__tl_helpers.targetChecked(e)) {
          selectedPromptOptions.add(opt.index);
        } else {
          selectedPromptOptions.delete(opt.index);
        }
        updateSelectionSummary();
      });
      checkbox.addEventListener('click', (e) => e.stopPropagation());
    } else {
      // Single-click auto-submit for single options
      btn.addEventListener('click', () => submitPrompt(String(opt.index ?? opt.value ?? opt.label)));
    }
    
    return btn;
  }, 'createPromptOptionButton', null);
}

function updateSelectionSummary() {
  ErrorBoundary.safeExecute(() => {
    const count = selectedPromptOptions.size;
    const summaryEl = document.getElementById('promptSelectionSummary');
    if (summaryEl) {
      if (count > 0) {
        summaryEl.textContent = `${count} contest${count === 1 ? '' : 's'} selected`;
        summaryEl.classList.remove('hidden');
      } else {
        summaryEl.classList.add('hidden');
      }
    }
  }, 'updateSelectionSummary');
}

function showPrompt({ title = 'Action required', message = '', options = [], placeholder = '' }) {
  ErrorBoundary.safeExecute(() => {
    try {
      if (typeof ModalRestoreBanner !== 'undefined') {
        ModalRestoreBanner.clear('prompt');
      }
    } catch (e) { /* noop */ }
    activePromptMessage = message;
    activePromptOptions = Array.isArray(options) ? options : [];

    console.debug('[showPrompt] Displaying:', {
      title,
      messagePreview: message.substring(0, 100),
      optionsCount: activePromptOptions.length,
      optionsSample: activePromptOptions.slice(0, 3),
      placeholder
    });

    if (promptTitleEl) {
      promptTitleEl.textContent = title;
      console.debug('[showPrompt] Set title to:', title);
    }
    if (promptMessageEl) {
      promptMessageEl.textContent = message || 'Please choose an option';
      console.debug('[showPrompt] Set message');
    }
    if (promptInputEl) {
      (/** @type {any} */ (window)).__tl_helpers.setElValue(promptInputEl, '');
      if (placeholder && promptInputEl instanceof HTMLInputElement) promptInputEl.placeholder = placeholder;
    }
    if (promptSearchEl) {
      (/** @type {any} */ (window)).__tl_helpers.setElValue(promptSearchEl, '');
      if (promptSearchEl instanceof HTMLInputElement) promptSearchEl.placeholder = placeholder || 'Filter options...';
    }
    renderPromptOptions('');

    const promptModal = $('#promptModal');
    if (promptModal) {
      promptModal.classList.remove('hidden');
      console.debug('[showPrompt] Modal made visible');
    }
    if (promptSearchEl) {
      promptSearchEl.focus();
    } else if (promptInputEl) {
      promptInputEl.focus();
    }
  }, 'showPrompt');
}

/**
/* Prompt typedefs consolidated at top of file. */

/**
 * Submit a prompt response to the server.
 * @param {string|number|undefined|null} [forcedValue] - Optional forced value (clicked option or explicit index)
 * @returns {void}
 */
function submitPrompt(/** @type {string|number|undefined|null} */ forcedValue) {
  ErrorBoundary.safeExecute(() => {
    /** @type {string} */
    let value;

    // If forced value provided (clicked option), use it
    if (forcedValue) {
      value = String(forcedValue);
    } else if (selectedPromptOptions.size > 0) {
      // Otherwise, use comma-separated selected indices (P2.1 multi-select)
      /** @type {number[]} */
      const selArr = /** @type {number[]} */ (Array.from(selectedPromptOptions));
      selArr.sort((a, b) => Number(a) - Number(b));
      value = selArr.join(',');
    } else {
      // Fall back to text input
      value = (/** @type {any} */ (window)).__tl_helpers.elValue(promptInputEl) || '';
    }

    if (!value) {
      showToast('Please select an option or enter a response', 'warning');
      return;
    }

    socket.emit('parser_prompt', /** @type {ParserPromptPayload} */ ({
      session_id: currentSessionId,
      value,
    }));
    hidePrompt({ preserveState: false, showRestore: false, reason: 'submit' });
  }, 'submitPrompt');
}

function hidePrompt(options) {
  ErrorBoundary.safeExecute(() => {
    const opts = options || {};
    const preserveState = !!opts.preserveState;
    const showRestore = !!opts.showRestore;
    const reason = opts.reason || 'close';
    const snapshot = showRestore ? {
      title: promptTitleEl ? promptTitleEl.textContent : 'Action required',
      message: activePromptMessage || (promptMessageEl ? promptMessageEl.textContent : ''),
      options: Array.isArray(activePromptOptions) ? activePromptOptions.slice() : [],
      placeholder: (promptInputEl instanceof HTMLInputElement) ? promptInputEl.placeholder : ''
    } : null;
    const promptModal = $('#promptModal');
    if (promptModal) promptModal.classList.add('hidden');
    document.body.classList.remove('no-scroll');
    if (showRestore && snapshot && (snapshot.message || (snapshot.options && snapshot.options.length))) {
      try {
        if (typeof ModalRestoreBanner !== 'undefined') {
          ModalRestoreBanner.show('prompt', {
            title: 'Prompt hidden',
            detail: snapshot.message || 'Reopen to continue',
            buttonLabel: 'Reopen prompt',
            onRestore: () => {
              activePromptMessage = snapshot.message || '';
              activePromptOptions = snapshot.options || [];
              showPrompt({
                title: snapshot.title || 'Action required',
                message: snapshot.message || '',
                options: snapshot.options || [],
                placeholder: snapshot.placeholder || ''
              });
            }
          });
        }
      } catch (e) { /* noop */ }
    } else if (reason !== 'close') {
      try {
        if (typeof ModalRestoreBanner !== 'undefined') {
          ModalRestoreBanner.clear('prompt');
        }
      } catch (e) { /* noop */ }
    }
    if (!preserveState) {
      activePromptMessage = null;
      activePromptOptions = [];
      selectedPromptOptions.clear();
      bundleExpandedState.clear();
    }
  }, 'hidePrompt');
}

const btnSubmitPrompt = $('#btnSubmitPrompt');
  if (btnSubmitPrompt) {
    btnSubmitPrompt.addEventListener('click', () => submitPrompt());
  }

const btnCancelPrompt = $('#btnCancelPrompt');
if (btnCancelPrompt) {
  btnCancelPrompt.addEventListener('click', () => {
    submitPrompt('cancel');
  });
}

const btnClosePrompt = $('#btnClosePrompt');
if (btnClosePrompt) {
  btnClosePrompt.addEventListener('click', () => hidePrompt({ preserveState: true, showRestore: true, reason: 'close' }));
}

const promptInputField = $('#promptInput');
if (promptInputField) {
  promptInputField.addEventListener('keydown', (e) => {
    const ke = /** @type {KeyboardEvent} */ (e);
    if (ke.key === 'Enter') {
      ke.preventDefault();
      submitPrompt();
    }
  });
}

// Initialize filter presets (P1.3) with debouncing (P2.2)
if (promptSearchEl) {
  const debouncedRender = debounce((value) => {
    renderPromptOptions(value);
  }, CONFIG.searchDebounceMs);
  
  promptSearchEl.addEventListener('input', (e) => {
    debouncedRender((/** @type {any} */ (window)).__tl_helpers.targetValue(e));
  });
}

// Hook up filter preset UI (P1.3)
document.addEventListener('DOMContentLoaded', () => {
  // Initialize Phase 2 features
  enhanceAccessibility(); // P2.5: Accessibility
  if (SessionRestore.hasRestoreData()) {
    SessionRestore.showRestoreBanner(); // P2.4: Session restore
  }
  
  // Initialize Advanced Features (Phase 3-4)
  initDirectUrlControl();
  initFilterPresets();
  initSessionActions();
  initKeyboardShortcuts(); // Consolidated keyboard shortcuts
  
  // Run integration tests only when explicitly enabled (dev flag or query param)
  const shouldRunIntegrationTests = (
    window.location.hostname === 'localhost'
    && ((/** @type {any} */ (window)).__ENABLE_INTEGRATION_TESTS__ === true || window.location.search.includes('runTests=1'))
  );
  if (shouldRunIntegrationTests) {
    runIntegrationTests().catch(e => ErrorBoundary.logError(e, 'Integration Tests'));
  }
  
  filterPresets.updatePresetDropdown();
  
  const saveBtn = $('#btnSaveFilterPreset');
  if (saveBtn) {
    saveBtn.addEventListener('click', () => {
      const name = prompt('Enter preset name:');
      if (name) {
        filterPresets.save(name, state.filters);
        showToast(`Preset "${name}" saved`, 'success');
      }
    });
  }
  
  const deleteBtn = $('#btnDeleteFilterPreset');
  if (deleteBtn) {
    deleteBtn.addEventListener('click', () => {
      const select = $('#logFilterPresetSelect');
      const name = (/** @type {any} */ (window)).__tl_helpers.elValue(select);
      if (name && confirm(`Delete preset "${name}"?`)) {
        filterPresets.deletePreset(name);
        showToast(`Preset "${name}" deleted`, 'info');
      }
    });
  }
  
  const select = $('#logFilterPresetSelect');
  if (select) {
    select.addEventListener('change', (e) => {
      const val = (/** @type {any} */ (window)).__tl_helpers.targetValue(e);
      if (val && val !== '__separator__') {
          filterPresets.applyPreset(val);
        }
    });
  }
  
  // Phase 3: Export buttons
  const exportJsonBtn = $('#btnExportJSON');
  if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', () => AdvancedExport.exportAsJSON(state.logs));
  }
  
  const exportCsvBtn = $('#btnExportCSV');
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => AdvancedExport.exportAsCSV(state.logs));
  }
  
  const exportMdBtn = $('#btnExportMarkdown');
  if (exportMdBtn) {
    exportMdBtn.addEventListener('click', () => AdvancedExport.exportAsMarkdown(state.logs));
  }
  
  const showShortcutsBtn = $('#btnShowKeyboardShortcuts');
  if (showShortcutsBtn) {
    showShortcutsBtn.addEventListener('click', () => KeyboardGuide.show());
  }
});

if (promptSearchEl) {
  promptSearchEl.addEventListener('input', (e) => {
    renderPromptOptions((/** @type {any} */ (window)).__tl_helpers.targetValue(e) || '');
  });
}

// ============================================
// Theme Management
// ============================================

function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', isDark ? 'light' : 'dark');
  localStorage.setItem('theme', isDark ? 'light' : 'dark');
  showToast(`Theme switched to ${isDark ? 'light' : 'dark'} mode`, 'success');
}

// Load theme from localStorage
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);

// ============================================
// Populate Filter States
// ============================================

STATES.forEach(state => {
  const option = document.createElement('option');
  option.value = state;
  option.textContent = state;
  $('#filterState').appendChild(option);
});

// ============================================
// Advanced Features: Direct URL Input
// ============================================

function parseDirectUrlField() {
  const textarea = document.getElementById('directUrlTextarea');
  const feedback = document.getElementById('directUrlFeedback');
  if (!textarea || !feedback) return [];
  
  const raw = (textarea instanceof HTMLTextAreaElement ? textarea.value : '') || '';
  const lines = raw.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  
  const urls = [];
  const errors = [];
  
  for (const line of lines) {
    if (urls.length >= CONFIG.maxDirectUrls) {
      errors.push(`Exceeded maximum of ${CONFIG.maxDirectUrls} URLs`);
      break;
    }
    
    try {
      const parsed = new URL(line);
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        errors.push(`Invalid protocol: ${line.substring(0, 40)}`);
        continue;
      }
      if (parsed.username || parsed.password) {
        errors.push(`URLs with auth not allowed: ${line.substring(0, 40)}`);
        continue;
      }
      urls.push(line);
    } catch (err) {
      errors.push(`Invalid URL: ${line.substring(0, 40)}`);
    }
  }
  
  // Update feedback
  let msg = '';
  if (errors.length > 0) {
    msg = `⚠️ ${errors[0]}`;
    feedback.className = 'text-danger';
  } else if (urls.length > 0) {
    msg = `✓ ${urls.length} valid URL${urls.length > 1 ? 's' : ''}`;
    feedback.className = 'text-success';
  } else {
    msg = 'Enter one URL per line.';
    feedback.className = 'text-muted';
  }
  feedback.textContent = msg;
  
  return urls;
}

function initDirectUrlControl() {
  const textarea = document.getElementById('directUrlTextarea');
  const clearBtn = document.getElementById('directUrlClearBtn');
  const directRadio = document.querySelector('input[name="fileSource"][value="direct"]');
  const advancedSection = document.querySelector('.advanced-option[data-source="direct"]');
  
  if (!textarea || !directRadio || !advancedSection) return;
  
  const isTextarea = textarea instanceof HTMLTextAreaElement;
  const isDirectRadio = directRadio instanceof HTMLInputElement;

  // Show/hide based on radio selection
  function updateVisibility() {
    if (isDirectRadio && directRadio.checked) {
      if (advancedSection instanceof Element) advancedSection.classList.remove('hidden');
      if (isTextarea) parseDirectUrlField();
    } else {
      if (advancedSection instanceof Element) advancedSection.classList.add('hidden');
    }
  }
  
  document.querySelectorAll('input[name="fileSource"]').forEach(radio => {
    if (radio instanceof HTMLInputElement) radio.addEventListener('change', updateVisibility);
  });
  
  // Live validation
  if (isTextarea) {
    textarea.addEventListener('input', debounce(() => {
      parseDirectUrlField();
      // Save draft per session
      if (AdvancedFeatures.currentSessionId) {
        AdvancedFeatures.directUrlDraftBySession.set(
          AdvancedFeatures.currentSessionId,
          textarea.value
        );
      }
    }, 500));
  }
  
  // Clear button
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (isTextarea) textarea.value = '';
      if (isTextarea) parseDirectUrlField();
      if (AdvancedFeatures.currentSessionId) {
        AdvancedFeatures.directUrlDraftBySession.delete(AdvancedFeatures.currentSessionId);
      }
    });
  }
  
  updateVisibility();
}

// ============================================
// Manual Upload File Selection (from classic)
// ============================================

const ManualUploadManager = (() => {
  let inventory = [];
  let currentSelection = null;
  
  /**
   * @typedef {Object} ManualUploadPath
   * @property {string} relPath - Normalized relative path (forward-slash separated)
   * @property {string} name - File name (last segment)
   * @property {string} dir - Directory portion (may be empty)
   */

  /**
   * Parse a provided path-like string into a normalized upload relative path object.
   * @param {string|undefined|null} pathStr
   * @returns {ManualUploadPath|null}
   */
  function parseManualUploadPath(pathStr) {
    if (!pathStr || typeof pathStr !== 'string') return null;
    const normalized = pathStr.replace(/\\/g, '/').trim().replace(/^\/+|\/+$/g, '');
    if (!normalized) return null;
    const parts = normalized.split('/');
    return {
      relPath: normalized,
      name: parts[parts.length - 1],
      dir: parts.slice(0, -1).join('/') || ''
    };
  }
  
  async function refreshInventory(options = {}) {
    const { preserveSelection = true, silent = false } = options;
    
    if (!silent) {
      showToast('Refreshing uploads...', 'info', 1500);
    }
    
    try {
      const response = await fetch('/api/fs/list?root=uploads&path=');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      if (!Array.isArray(data.entries)) {
        throw new Error('Invalid response format');
      }
      
      /**
       * @typedef {{ name: string, type: string, size?: number|null, modified?: number|null }} UploadEntry
       * @typedef {{ relPath: string, name: string, size: number, modified: number }} NormalizedUpload
       */
      inventory = data.entries
        .filter(
          /**
           * @param {any} e
           * @returns {e is UploadEntry}
           */
          function isUploadFile(e) {
        try {
          return !!e &&
        e.type === 'file' &&
        typeof e.name === 'string' &&
        (typeof e.size === 'number' || e.size === null || typeof e.size === 'undefined') &&
        (typeof e.modified === 'number' || e.modified === null || typeof e.modified === 'undefined');
        } catch (/** @type {any} */ _err) {
          return false;
        }
          }
        )
        .map(
          /** @param {UploadEntry} e @returns {NormalizedUpload} */
          e => ({
        name: e.name,
        relPath: e.name,
        size: e.size || 0,
        modified: e.modified || Date.now()
          })
        )
        .sort(
          /** @param {NormalizedUpload} a
           *  @param {NormalizedUpload} b
           *  @returns {number}
           */
          (a, b) => b.modified - a.modified
        );

      /** @type {NormalizedUpload[]} */
      inventory = /** @type {NormalizedUpload[]} */ (inventory);
      
      updateManualUploadUI();
      
      if (!silent) {
        showToast(`Found ${inventory.length} file(s) in uploads`, 'success', 2000);
      }
      
      // Try to restore selection
      if (preserveSelection && currentSelection) {
        /**
         * @typedef {Object} NormalizedUploadLocal
         * @property {string} relPath
         * @property {string} name
         * @property {number} size
         * @property {number} modified
         */

        /** @type {NormalizedUploadLocal|undefined} */
        /** @type {NormalizedUploadLocal | undefined} */
        const found = /** @type {NormalizedUploadLocal | undefined} */ (inventory.find(
          /** @param {any} f @returns {boolean} */ (f) => f && typeof f.relPath === 'string' && f.relPath === currentSelection.relPath
        ));
        /**
         * @typedef {Object} UploadEntry
         * @property {string} name
         * @property {'file'|'dir'} type
         * @property {number|null|undefined} [size]
         * @property {number|null|undefined} [modified]
         */

        /* NormalizedUpload typedef defined earlier; reuse canonical typedef. */

        /** Ensure inventory is treated as NormalizedUpload[] for tooling/type hints */
        /** @type {NormalizedUpload[]} */
        inventory = /** @type {any} */ (inventory || []);
        if (found) {
          applySelection(found, { updateSource: false });
        } else {
          currentSelection = null;
        }
      }
      
      return inventory;
    } catch (err) {
      console.error('[ManualUpload] Refresh failed:', err);
      if (!silent) {
        showToast(`Failed to refresh uploads: ${err.message}`, 'error');
      }
      return [];
    }
  }
  
  function updateManualUploadUI() {
    const select = document.getElementById('manualUploadSelect');
    const summary = document.getElementById('manualUploadSummary');
    const selectEl = select instanceof HTMLSelectElement ? select : null;
    
    if (!selectEl) return;
    
    // Clear and rebuild options
    selectEl.innerHTML = '<option value="">— Choose a file —</option>';
    
    /* NormalizedUpload typedef defined earlier; reuse canonical typedef. */

    inventory.forEach((/** @type {NormalizedUpload} */ file, /** @type {number} */ idx) => {
      /** @type {HTMLOptionElement} */
      const option = document.createElement('option');
      option.value = file.relPath;
      option.textContent = file.name;
      if (option instanceof HTMLElement) {
      option.setAttribute('data-size', String(file.size || 0));
      option.setAttribute('data-modified', String(file.modified || 0));
      }
      selectEl.appendChild(option);
    });
    
    // Restore selection
    if (currentSelection) {
      selectEl.value = currentSelection.relPath;
    }
    
    // Update summary
    if (summary) {
      if (currentSelection) {
        const sizeKB = ((currentSelection.size || 0) / 1024).toFixed(1);
        summary.textContent = `Selected: ${currentSelection.name} (${sizeKB} KB)`;
        summary.className = 'text-success small';
      } else {
        summary.textContent = inventory.length > 0 ? `${inventory.length} file(s) available` : 'No files uploaded';
        summary.className = 'text-muted small';
      }
    }
  }
  
  /* NormalizedUpload typedef defined earlier; reuse canonical typedef. */
  /**
   * @typedef {Object} ApplySelectionOptions
   * @property {boolean} [updateSource]
   */

  /**
   * Apply a selection from the uploads inventory.
   * @param {NormalizedUpload | null | undefined} file
   * @param {ApplySelectionOptions} [options]
   * @returns {void}
   */
  function applySelection(file, options = {}) {
    const { updateSource = true } = options;
    
    if (!file || !file.relPath) {
      currentSelection = null;
      updateManualUploadUI();
      return;
    }
    
    currentSelection = { ...file };
    updateManualUploadUI();
    
    // Emit to server
    if (updateSource && currentSessionId) {
      socket.emit('set_manual_source', {
        session_id: currentSessionId,
        file_source: 'uploads',
        origin: 'user'
      });
    }
    
    // Update pipeline phase
    if (PipelineManager && PipelineManager.getPhase() === 'prepare') {
      PipelineManager.setPhase('source');
    }
    
    showToast(`Selected: ${file.name}`, 'success', 2000);
  }
  
  function clearSelection() {
    currentSelection = null;
    const select = document.getElementById('manualUploadSelect');
    if (select instanceof HTMLSelectElement) select.value = '';
    updateManualUploadUI();
    showToast('Selection cleared', 'info', 1500);
  }
  
  function init() {
    const select = document.getElementById('manualUploadSelect');
    const selectEl = select instanceof HTMLSelectElement ? select : null;
    const refreshBtn = document.getElementById('manualUploadRefreshBtn');
    const clearBtn = document.getElementById('manualUploadClearBtn');
    const uploadRadio = document.querySelector('input[name="fileSource"][value="uploads"]');

    if (!selectEl) return;
    
    // Selection change handler
    selectEl.addEventListener('change', (e) => {
      const value = (/** @type {any} */ (window)).__tl_helpers.targetValue(e);
      if (!value) {
        clearSelection();
        return;
      }
      
      /**
       * Normalized upload object used in UI lists.
       * @typedef {Object} NormalizedUpload
       * @property {string} relPath - Normalized relative path (forward-slash separated)
       * @property {string} name - File name (last segment)
       * @property {number} [size] - File size in bytes
       * @property {number} [modified] - Unix ms timestamp of last modification
       */

      /** @type {NormalizedUpload|undefined} */
      const file = /** @type {NormalizedUpload|undefined} */ (inventory.find(
        /** @param {any} f @returns {boolean} */ (f) => f && f.relPath === value
      ));
      if (file) {
        applySelection(file);
      }
    });
    
    // Refresh button
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        refreshInventory({ preserveSelection: true, silent: false });
      });
    }
    
    // Clear button
    if (clearBtn) {
      clearBtn.addEventListener('click', clearSelection);
    }
    
    // Auto-refresh when uploads radio is selected
    if (uploadRadio instanceof HTMLInputElement) {
      uploadRadio.addEventListener('change', () => {
        if (uploadRadio.checked) {
          refreshInventory({ preserveSelection: true, silent: true });
        }
      });
    }
    
    // Initial load
    refreshInventory({ preserveSelection: false, silent: true });
  }
  
  return {
    init,
    refreshInventory,
    applySelection,
    clearSelection,
    getInventory: () => [...inventory],
    getCurrentSelection: () => currentSelection ? { ...currentSelection } : null
  };
})();

// ============================================
// Advanced Features: Filter Presets
// ============================================

function initFilterPresets() {
  AdvancedFeatures.loadPresets();
  
  const presetSelect = document.getElementById('filterPresetSelect');
  const saveBtn = document.getElementById('saveFiltersBtn');
  const deleteBtn = document.getElementById('deletePresetBtn');
  
  if (!presetSelect || !saveBtn || !deleteBtn) return;
  const presetSel = presetSelect instanceof HTMLSelectElement ? presetSelect : null;
  
  // Populate preset dropdown
  function refreshPresetList() {
    // Clear existing options except first
    if (presetSel) {
      while (presetSel.options.length > 1) {
        presetSel.remove(1);
      }
    }
    
    AdvancedFeatures.filterPresets.forEach((filters, name) => {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      if (presetSel) presetSel.appendChild(option);
    });

    if (presetSel) presetSel.value = '';
  }

  if (presetSel) {
    presetSel.addEventListener('change', () => {
      const selected = presetSel.value;
      if (!selected) return;

      const filters = AdvancedFeatures.filterPresets.get(selected);
      if (filters) {
        AdvancedFeatures.applyFilters(filters);
        showToast(`Loaded preset: ${selected}`, 'success');
      }
    });
  }
  
  // Save current filters as preset
  saveBtn.addEventListener('click', () => {
    const name = prompt('Enter preset name:');
    if (!name || !name.trim()) return;
    
    const filters = AdvancedFeatures.getCurrentFilters();
    AdvancedFeatures.filterPresets.set(name.trim(), filters);
    AdvancedFeatures.savePresets();
    refreshPresetList();
    showToast(`Saved preset: ${name}`, 'success');
  });
  
  // Delete selected preset
  deleteBtn.addEventListener('click', () => {
    const selected = presetSel ? presetSel.value : '';
    if (!selected) {
      showToast('Select a preset to delete', 'warning');
      return;
    }
    
    if (!confirm(`Delete preset "${selected}"?`)) return;
    
    AdvancedFeatures.filterPresets.delete(selected);
    AdvancedFeatures.savePresets();
    refreshPresetList();
    showToast(`Deleted preset: ${selected}`, 'info');
  });
  
  refreshPresetList();
}

// ============================================
// Advanced Features: Session Actions
// ============================================

function initSessionActions() {
  const cloneBtn = document.getElementById('btnCloneSession');
  const exportBtn = document.getElementById('btnExportSession');
  const clearBtn = document.getElementById('btnClearSession');
  
  if (cloneBtn) {
    cloneBtn.addEventListener('click', () => {
      if (!AdvancedFeatures.currentSessionId) {
        showToast('No active session to clone', 'warning');
        return;
      }
      
      if (window.socket && window.socket.connected) {
        window.socket.emit('clone_session', { 
          session_id: AdvancedFeatures.currentSessionId 
        });
        showToast('Cloning session...', 'info');
      } else {
        showToast('Socket not connected', 'error');
      }
    });
  }
  
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!state.results || state.results.length === 0) {
        showToast('No data to export', 'warning');
        return;
      }
      
      const dataStr = JSON.stringify(state.results, null, 2);
      const blob = new Blob([/** @type {any} */ (dataStr)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `session_${AdvancedFeatures.currentSessionId || 'export'}_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
      showToast('Data exported', 'success');
    });
  }
  
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      if (!confirm('Clear all logs for this session?')) return;
      
      if (window.clearSessionLogs) {
        window.clearSessionLogs();
        showToast('Logs cleared', 'info');
      } else {
        // Fallback
        const logContainer = document.getElementById('logEntries');
        if (logContainer) logContainer.innerHTML = '';
        showToast('Logs cleared (UI only)', 'info');
      }
    });
  }
}

/**
 * Canonical typedef for NormalizedUpload used by the manual uploads UI.
 * @typedef {Object} NormalizedUpload
 * @property {string} relPath - Normalized relative path (forward-slash separated)
 * @property {string} name - File name (last segment)
 * @property {number} [size] - File size in bytes
 * @property {number} [modified] - Unix ms timestamp of last modification
 */

// ============================================
// Advanced Features: Keyboard Shortcuts
// ============================================

function initKeyboardShortcuts() {
  const shortcuts = {
    'Ctrl+E': () => {
      // Export JSON
      (/** @type {any} */ (window)).__tl_helpers.safeClick(document.getElementById('btnExportSession'));
    },
    'Ctrl+Shift+E': () => {
      // Export CSV (if available)
      const exportCsvBtn = document.querySelector('[data-action="export-csv"]');
      if (exportCsvBtn instanceof HTMLElement) (/** @type {any} */ (window)).__tl_helpers.safeClick(exportCsvBtn);
    },
    'Ctrl+L': () => {
      // Clear logs
      (/** @type {any} */ (window)).__tl_helpers.safeClick(document.getElementById('btnClearSession'));
    },
    'Ctrl+Shift+C': () => {
      // Clone session
      (/** @type {any} */ (window)).__tl_helpers.safeClick(document.getElementById('btnCloneSession'));
    },
    'Ctrl+Shift+P': () => {
      // Open command palette
      const palette = document.getElementById('commandPalette');
      if (palette && palette.classList.contains('hidden')) {
        palette.classList.remove('hidden');
        palette.querySelector('input')?.focus();
      }
    },
    'Ctrl+/': () => {
      // Show shortcuts help
      showShortcutsHelp();
    },
    'Escape': () => {
      // Close modals
      document.querySelectorAll('.modal:not(.hidden)').forEach(modal => {
        modal.classList.add('hidden');
      });
    }
  };
  
  document.addEventListener('keydown', (e) => {
    const key = [];
    if (e.ctrlKey || e.metaKey) key.push('Ctrl');
    if (e.shiftKey) key.push('Shift');
    if (e.altKey) key.push('Alt');
    key.push(e.key);
    
    const combo = key.join('+');
    const handler = shortcuts[combo];
    
    if (handler) {
      e.preventDefault();
      ErrorBoundary.safeExecute(handler, `Keyboard shortcut: ${combo}`);
    }
  });
}

function showShortcutsHelp() {
  const helpText = `
    <h3>Keyboard Shortcuts</h3>
    <ul class="unstyled-list">
      <li><kbd>Ctrl+E</kbd> - Export session data (JSON)</li>
      <li><kbd>Ctrl+Shift+E</kbd> - Export as CSV</li>
      <li><kbd>Ctrl+L</kbd> - Clear logs</li>
      <li><kbd>Ctrl+Shift+C</kbd> - Clone current session</li>
      <li><kbd>Ctrl+Shift+P</kbd> - Open command palette</li>
      <li><kbd>Ctrl+/</kbd> - Show this help</li>
      <li><kbd>Escape</kbd> - Close modals</li>
    </ul>
  `;
  
  const existing = document.getElementById('shortcutsModal');
  if (existing) {
    existing.classList.remove('hidden');
    return;
  }
  
  const modal = document.createElement('div');
  modal.id = 'shortcutsModal';
  modal.className = 'modal';
  modal.innerHTML = `
    <div class="modal-backdrop"></div>
    <div class="modal-content">
      ${helpText}
      <button class="btn btn-primary" data-action="modal-hide">Close</button>
    </div>
  `;
  document.body.appendChild(modal);
  modal.classList.remove('hidden');
}

// ============================================
// Data Loading: Real API (no sample fixtures)
// ============================================

const pulseLoaderEl = $('#pulseLoader');
const pulseLoaderSubtitleEl = $('#pulseLoaderSubtitle');

function setLoadingState(isLoading, subtitleText = '') {
  if (!pulseLoaderEl) return;
  const grid = $('#resultsGrid');
  const emptyState = $('#emptyState');

  if (isLoading) {
    if (grid) grid.classList.add('hidden');
    if (emptyState) emptyState.classList.add('hidden');
    pulseLoaderEl.classList.remove('hidden');
    pulseLoaderEl.setAttribute('aria-hidden', 'false');
    pulseLoaderEl.setAttribute('aria-busy', 'true');
    if (subtitleText && pulseLoaderSubtitleEl) {
      pulseLoaderSubtitleEl.textContent = subtitleText;
    }
    document.body.classList.add('pulse-loading');
  } else {
    pulseLoaderEl.classList.add('hidden');
    pulseLoaderEl.setAttribute('aria-hidden', 'true');
    pulseLoaderEl.setAttribute('aria-busy', 'false');
    document.body.classList.remove('pulse-loading');
  }
}

/**
 * Fetch results from warehouse API and transform to UI format.
 * If API returns no rows, show empty state; never inject demo fixtures.
 */
async function loadRealData() {
  setLoadingState(true, 'Calibrating election pulse...');
  try {
    console.log('[API] Fetching results from warehouse...');
    const response = await fetch('/api/warehouse_election_results?limit=50', {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      credentials: 'same-origin',
    });

    if (!response.ok) {
      throw new Error(`API returned ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    const items = Array.isArray(data.items) ? data.items : [];

    if (items.length === 0) {
      console.warn('[API] No results found in warehouse');
      state.results = [];
      renderResults();
      showToast('No results available yet.', 'info');
      return;
    }

    // Transform warehouse schema to UI results format
    /**
     * @typedef {Object} WarehouseItem
     * @property {string|number} [id]
     * @property {string} [contest]
     * @property {string} [county]
     * @property {string} [format]
     * @property {number} [row_count]
     * @property {number} [column_count]
     * @property {number|string} [confidence_score]
     * @property {string} [state]
     * @property {string} [handler_name]
     * @property {string|number|Date} [created_at]
     * @property {string} [source_url]
     * @property {string} [preview_html]
     * @property {string} [preview_text]
     */

    /**
     * @typedef {Object} UIResult
     * @property {string} id
     * @property {string} name
     * @property {string} type
     * @property {number} rows
     * @property {number} columns
     * @property {number} confidence
     * @property {string} state
     * @property {string} county
     * @property {string} handler
     * @property {number} timestamp
     * @property {string} source_url
     * @property {string} preview
     */

    /** @type {WarehouseItem[]} */
    const warehouseItems = /** @type {any} */ (items);

    /** @type {number} */
    const minConfidence = Number((state.filters && state.filters.confidence) || 0);

    const mappedResults = warehouseItems.map((item, idx) => {
      let confidence = null;
      if (item.confidence_score !== undefined && item.confidence_score !== null && !Number.isNaN(Number(item.confidence_score))) {
        confidence = Number(item.confidence_score);
        // If service returns 0-1, scale to percentage; if already 0-100, keep as-is.
        if (confidence <= 1) confidence = confidence * 100;
      }

      return {
        id: String(item.id || idx + 1),
        name: item.contest || item.county || `Result #${idx + 1}`,
        type: (item.format || 'csv').toLowerCase(),
        rows: item.row_count || 0,
        columns: item.column_count || 0,
        confidence: confidence === null ? 0 : confidence,
        state: item.state || 'N/A',
        county: item.county || '',
        handler: item.handler_name || 'unknown',
        timestamp: item.created_at ? new Date(item.created_at).getTime() : Date.now(),
        source_url: item.source_url || '',
        preview: item.preview_html || item.preview_text || '(No preview available)',
      };
    }).filter(r => (r.confidence || 0) >= minConfidence);

    state.results = mappedResults;

    console.log(`[API] Loaded ${state.results.length} results from warehouse`);
    renderResults();
  } catch (/** @type {any} */ error) {
    console.error('[API] Failed to load real data:', error);
    state.results = [];
    renderResults();
    showToast(`Failed to load results: ${error.message}`, 'warning');
  } finally {
    setLoadingState(false);
  }
}

// ============================================
// Theme Management
// ============================================

const ThemeManager = (() => {
  const THEME_KEY = 'parser_theme';
  
  function getCurrentTheme() {
    return localStorage.getItem(THEME_KEY) || 'dark';
  }
  
  /**
   * @typedef {'light'|'dark'} ThemeName
   *
   * @typedef {Object} ThemeManagerInterface
   * @property {(theme: ThemeName | string) => void} setTheme
   * @property {() => void} init
   * @property {() => ThemeName} getCurrentTheme
   * @property {(theme: ThemeName) => void} toggleTheme
   */

  /**
   * Set the current theme.
   * Accepts a ThemeName or arbitrary string (invalid values default to 'dark').
   * @param {ThemeName | string} theme
   * @returns {void}
   */
  function setTheme(theme) {
    const validTheme = theme === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', validTheme);
    localStorage.setItem(THEME_KEY, validTheme);
    updateThemeToggle(validTheme);
  }
  
  /**
   * Update the theme toggle button's visual state and label.
   * @param {ThemeName|string} theme
   * @returns {void}
   */
  function updateThemeToggle(theme) {
    const toggleBtn = document.getElementById('themeToggle');
    const themeLabel = document.getElementById('themeLabel');
    
    if (toggleBtn) {
      toggleBtn.classList.toggle('theme-light', theme === 'light');
      toggleBtn.classList.toggle('theme-dark', theme === 'dark');
    }
    
    if (themeLabel) {
      themeLabel.textContent = theme === 'dark' ? 'Dark' : 'Light';
    }
  }
  
  function toggleTheme() {
    const current = getCurrentTheme();
    const next = current === 'light' ? 'dark' : 'light';
    setTheme(next);
    showToast(`Switched to ${next} theme`, 'info', 2000);
  }
  
  function init() {
    const savedTheme = getCurrentTheme();
    setTheme(savedTheme);
    
    // Wire up the new theme toggle button in the right sidebar
    const toggleBtn = document.getElementById('themeToggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', toggleTheme);
    }
  }
  
  return {
    init,
    toggleTheme,
    getCurrentTheme,
    setTheme
  };
})();

// ============================================
// Swipe Gesture Handler for Sidebars
// ============================================

const SwipeHandler = (() => {
  let touchStartX = 0;
  let touchStartY = 0;
  let touchStartTime = 0;
  let isSwiping = false;
  
  const SWIPE_THRESHOLD = 50; // Minimum distance for a swipe
  const SWIPE_TIME_LIMIT = 300; // Maximum time for a swipe (ms)
  const SWIPE_VELOCITY_THRESHOLD = 0.3; // Minimum velocity (pixels/ms)
  
  /**
   * Handle touch start event
   * @param {TouchEvent} e
   * @param {HTMLElement} sidebar
   */
  function handleTouchStart(e, sidebar) {
    if (!e.touches || e.touches.length === 0) return;
    
    const touch = e.touches[0];
    touchStartX = touch.clientX;
    touchStartY = touch.clientY;
    touchStartTime = Date.now();
    isSwiping = false;
  }
  
  /**
   * Handle touch move event
   * @param {TouchEvent} e
   * @param {HTMLElement} sidebar
   */
  function handleTouchMove(e, sidebar) {
    if (!e.touches || e.touches.length === 0) return;
    
    const touch = e.touches[0];
    const deltaX = touch.clientX - touchStartX;
    const deltaY = touch.clientY - touchStartY;
    
    // Detect if this is a horizontal swipe (not vertical scroll)
    if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 10) {
      isSwiping = true;
    }
  }
  
  /**
   * Handle touch end event
   * @param {TouchEvent} e
   * @param {HTMLElement} sidebar
   * @param {string} sidebarType - 'left' or 'right'
   */
  function handleTouchEnd(e, sidebar, sidebarType) {
    if (!isSwiping) return;
    
    const touch = e.changedTouches && e.changedTouches[0];
    if (!touch) return;
    
    const deltaX = touch.clientX - touchStartX;
    const deltaY = touch.clientY - touchStartY;
    const deltaTime = Date.now() - touchStartTime;
    const velocity = Math.abs(deltaX) / deltaTime;
    
    // Check if it's a valid swipe
    const isHorizontalSwipe = Math.abs(deltaX) > Math.abs(deltaY);
    const meetsThreshold = Math.abs(deltaX) > SWIPE_THRESHOLD;
    const withinTimeLimit = deltaTime < SWIPE_TIME_LIMIT;
    const meetsVelocity = velocity > SWIPE_VELOCITY_THRESHOLD;
    
    if (isHorizontalSwipe && (meetsThreshold || meetsVelocity) && withinTimeLimit) {
      // Right sidebar: swipe right to close
      if (sidebarType === 'right' && deltaX > 0) {
        closeSidebar(sidebar);
      }
      // Left sidebar: swipe left to close
      else if (sidebarType === 'left' && deltaX < 0) {
        closeSidebar(sidebar);
      }
    }
    
    // Reset
    isSwiping = false;
  }
  
  /**
   * Close a sidebar
   * @param {HTMLElement} sidebar
   */
  function closeSidebar(sidebar) {
    if (!sidebar) return;
    
    sidebar.classList.remove('open', 'sidebar-open', 'centered-tool-window');
    document.body.classList.remove('no-scroll', 'sidebar-right-open');
    
    // Update toggle button aria-expanded
    const toggleBtn = document.getElementById('btnToggleRightSidebar');
    if (toggleBtn) {
      toggleBtn.setAttribute('aria-expanded', 'false');
    }
  }
  
  /**
   * Initialize swipe handlers for sidebars
   */
  function init() {
    const rightSidebar = document.querySelector('.sidebar-right');
    const leftSidebar = document.querySelector('.sidebar-left');
    
    if (rightSidebar instanceof HTMLElement) {
      rightSidebar.addEventListener('touchstart', (e) => handleTouchStart(/** @type {TouchEvent} */(e), rightSidebar), { passive: true });
      rightSidebar.addEventListener('touchmove', (e) => handleTouchMove(/** @type {TouchEvent} */(e), rightSidebar), { passive: true });
      rightSidebar.addEventListener('touchend', (e) => handleTouchEnd(/** @type {TouchEvent} */(e), rightSidebar, 'right'), { passive: true });
    }
    
    if (leftSidebar instanceof HTMLElement) {
      leftSidebar.addEventListener('touchstart', (e) => handleTouchStart(/** @type {TouchEvent} */(e), leftSidebar), { passive: true });
      leftSidebar.addEventListener('touchmove', (e) => handleTouchMove(/** @type {TouchEvent} */(e), leftSidebar), { passive: true });
      leftSidebar.addEventListener('touchend', (e) => handleTouchEnd(/** @type {TouchEvent} */(e), leftSidebar, 'left'), { passive: true });
    }
  }
  
  return {
    init
  };
})();

// ============================================
// Pipeline Phase System (from classic)
// ============================================

const PipelineManager = (() => {
  const PHASES = ['prepare', 'source', 'run', 'resolve', 'review'];
  let currentPhase = 'prepare';
  let pipelineHintEl = null;
  
  function getPhaseIndex(phase) {
    return PHASES.indexOf(phase);
  }
  
  function setPhase(phase, options = {}) {
    if (!PHASES.includes(phase)) return;
    
    currentPhase = phase;
    const phaseEvent = new CustomEvent('pipeline:phase-change', {
      detail: { phase, options }
    });
    document.dispatchEvent(phaseEvent);
    
    updatePhaseHint();
    
    if (options.focus) {
      // Focus relevant UI element based on phase
      focusPhaseElement(phase);
    }
  }
  
  function updatePhaseHint(customMessage = null) {
    if (!pipelineHintEl) {
      pipelineHintEl = document.querySelector('.pipeline-hint');
      if (!pipelineHintEl) return;
    }
    
    let message = '';
    let level = 'info';
    
    if (customMessage) {
      if (typeof customMessage === 'object') {
        message = customMessage.text || '';
        level = customMessage.level || 'info';
      } else {
        message = String(customMessage);
      }
    } else {
      const hints = {
        prepare: 'Review inputs and choose your data source. Press Run when ready.',
        source: 'Source selected. Verify your selection and press Run to begin parsing.',
        run: 'Parser is running. Monitor the log for progress and warnings.',
        resolve: 'Action required. Respond to the prompt to continue processing.',
        review: 'Parsing complete. Download outputs or run again with different settings.'
      };
      message = hints[currentPhase] || '';
    }
    
    pipelineHintEl.textContent = message;
    if (pipelineHintEl instanceof HTMLElement) {
      pipelineHintEl.setAttribute('data-level', String(level));
      pipelineHintEl.classList.toggle('hidden', !message);
    }
  }
  
  function focusPhaseElement(phase) {
    const focusMap = {
      prepare: '#urlLinesBox',
      source: '#manualUploadSelect',
      run: '#logContainer',
      resolve: '#promptInput',
      review: '#outputFolderPanel'
    };
    
    const selector = focusMap[phase];
    if (selector) {
      const el = document.querySelector(selector);
      el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
  
  function init() {
    // Create pipeline hint element if it doesn't exist (prefer host container instead of navbar)
    if (!pipelineHintEl) {
      pipelineHintEl = document.querySelector('.pipeline-hint');
    }

    if (!pipelineHintEl) {
      const host = document.getElementById('pipelineHintHost')
        || document.querySelector('.content-shell')
        || document.querySelector('.modern-layout');
      pipelineHintEl = document.createElement('div');
      pipelineHintEl.className = 'pipeline-hint';
      if (pipelineHintEl instanceof HTMLElement) pipelineHintEl.setAttribute('data-level', 'info');
      if (host) {
        host.appendChild(pipelineHintEl);
      } else {
        const navbar = document.querySelector('.navbar-modern');
        navbar?.appendChild(pipelineHintEl);
      }
    }
    
    setPhase('prepare');
  }
  
  return {
    init,
    setPhase,
    updatePhaseHint,
    getPhase: () => currentPhase,
    PHASES
  };
})();

// ============================================
// Modal Utility
// ============================================

const Modal = (() => {
  let modalEl = null;
  let refs = null;
  
  function ensureModal() {
    if (modalEl) return;
    
    modalEl = document.createElement('div');
    modalEl.id = 'genericModal';
    modalEl.className = 'modal fade';
    modalEl.setAttribute('role', 'dialog');
    modalEl.setAttribute('aria-modal', 'true');
    modalEl.setAttribute('aria-labelledby', 'genericModalTitle');
    
    modalEl.innerHTML = `
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="genericModalTitle">Modal</h5>
            <button type="button" class="btn-close" id="closeGenericModal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <input type="search" id="genericModalSearch" class="form-control mb-2" placeholder="Filter...">
            <div id="genericModalSummary" class="mb-2"></div>
            <div id="genericModalOptions"></div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" id="cancelGenericModal">Cancel</button>
          </div>
        </div>
      </div>
    `;
    
    document.body.appendChild(modalEl);
    
    refs = {
      modal: modalEl,
      titleEl: modalEl.querySelector('.modal-title'),
      searchEl: $('#genericModalSearch'),
      optionsDiv: $('#genericModalOptions'),
      summaryDiv: $('#genericModalSummary'),
      closeBtn: $('#closeGenericModal'),
      cancelBtn: $('#cancelGenericModal')
    };
  }
  
  function get() {
    ensureModal();
    return refs;
  }
  
  function open() {
    ensureModal();
    const inst = window.bootstrap?.Modal.getOrCreateInstance(modalEl, { keyboard: true, backdrop: true });
    inst?.show();
  }
  
  function close() {
    if (!modalEl) return;
    const inst = window.bootstrap?.Modal.getOrCreateInstance(modalEl);
    inst?.hide();
  }
  
  return {
    get,
    open,
    close
  };
})();

// ============================================
// Modal Restore Banner (from classic)
// ============================================

const ModalRestoreBanner = (() => {
  let bannerEl = null;
  let contexts = new Map(); // key -> { sessionId, message, title, detail, buttonLabel, onRestore }
  let activeBannerKey = null;
  
  function createBanner() {
    if (bannerEl) return bannerEl;
    
    bannerEl = document.createElement('div');
    bannerEl.id = 'modalRestoreBanner';
    bannerEl.className = 'modal-restore-banner hidden';
    bannerEl.setAttribute('role', 'status');
    bannerEl.setAttribute('aria-live', 'polite');
    
    const container = document.querySelector('.modern-layout') || document.body;
    container.appendChild(bannerEl);
    
    return bannerEl;
  }
  
  function show(key, context) {
    if (!key || !context) return;
    
    contexts.set(key, context);
    activeBannerKey = key;
    
    const banner = createBanner();
    const { message, title, detail, buttonLabel, onRestore } = context;
    
    banner.innerHTML = `
      <div class="restore-content">
        <div class="restore-icon">↺</div>
        <div class="restore-text">
          <div class="restore-title">${escapeHtml(title || 'Dialog paused')}</div>
          <div class="restore-detail">${escapeHtml(detail || message || 'Reopen to continue')}</div>
        </div>
        <button type="button" class="btn-sm btn-primary restore-btn">${escapeHtml(buttonLabel || 'Reopen')}</button>
        <button type="button" class="btn-icon-sm restore-dismiss" aria-label="Dismiss">×</button>
      </div>
    `;
    
    const reopenBtn = banner.querySelector('.restore-btn');
    const dismissBtn = banner.querySelector('.restore-dismiss');
    
    if (reopenBtn && typeof onRestore === 'function') {
      reopenBtn.addEventListener('click', () => {
        hide();
        onRestore();
      });
    }
    
    if (dismissBtn) {
      dismissBtn.addEventListener('click', () => {
        contexts.delete(key);
        hide();
      });
    }
    
    banner.classList.remove('hidden');
    
    // Position at bottom of main content (above drawer)
    setTimeout(() => {
      const mainContent = document.querySelector('.main-content');
      if (mainContent) {
        const rect = mainContent.getBoundingClientRect();
        banner.style.bottom = 'calc(var(--drawer-left-offset, 300px) + 60px)';
        banner.style.left = `${rect.left + 16}px`;
        banner.style.right = `${window.innerWidth - rect.right + 16}px`;
      }
    }, 50);
  }
  
  function hide() {
    if (bannerEl) {
      bannerEl.classList.add('hidden');
    }
    activeBannerKey = null;
  }
  
  function clear(key) {
    if (key) {
      contexts.delete(key);
      if (activeBannerKey === key) {
        hide();
      }
    } else {
      contexts.clear();
      hide();
    }
  }
  
  return {
    show,
    hide,
    clear,
    isActive: () => activeBannerKey !== null
  };
})();

// ============================================
// URL List Manager
// ============================================

const UrlListManager = (() => {
  let cachedUrls = [];
  let cachedMeta = [];
  let selectedState = '';
  let selectedCounty = '';
  let lastSearch = '';

  /**
   * Extract state/county hints from a URL using lightweight heuristics.
   * @param {string} url
   * @returns {{ url: string, state: string|null, county: string|null }}
   */
  function extractMeta(url) {
    const lower = url.toLowerCase();
    let state = null;
    let county = null;

    // State detection: query param ?state=XX or path token XX where XX is a known state code
    const qsState = lower.match(/[?&]state=([a-z]{2})/i);
    if (qsState && qsState[1]) {
      const cand = qsState[1].toUpperCase();
      if (STATES.includes(cand)) state = cand;
    }
    if (!state) {
      const tokens = url.split(/[^A-Za-z0-9]+/).filter(Boolean);
      const found = tokens.find(t => {
        const upper = t.toUpperCase();
        return upper.length === 2 && STATES.includes(upper);
      });
      if (found) state = found.toUpperCase();
    }

    // County detection: query param county=, tokens containing "county", or path segment preceding "county"
    const qsCounty = lower.match(/[?&]county=([^&#\/]*)/i);
    if (qsCounty && qsCounty[1]) {
      county = qsCounty[1];
    } else {
      const countyToken = lower.match(/([a-z0-9]+)[-_]?county/);
      if (countyToken && countyToken[1]) county = countyToken[1];
      if (!county) {
        const parts = lower.split(/\//).filter(Boolean);
        const countyIdx = parts.findIndex(p => p.includes('county'));
        if (countyIdx > 0 && parts[countyIdx - 1]) {
          county = parts[countyIdx - 1];
        }
      }
    }

    const tidyCounty = county
      ? county.split(/[_\-\s]+/).map(p => p ? p[0].toUpperCase() + p.slice(1) : '').join(' ').trim()
      : null;

    return { url, state, county: tidyCounty || null };
  }

  function populateTaxonomy(meta) {
    const stateSelect = $('#urlStateFilter');
    const countySelect = $('#urlCountyFilter');
    if (!stateSelect || !(stateSelect instanceof HTMLSelectElement) || !countySelect || !(countySelect instanceof HTMLSelectElement)) return;

    const statesSeen = Array.from(new Set(meta.map(m => m.state).filter(Boolean))).sort();

    // Reset state options
    stateSelect.innerHTML = '<option value="">— Filter by state —</option>';
    statesSeen.forEach(st => {
      const opt = document.createElement('option');
      opt.value = st;
      opt.textContent = st;
      stateSelect.appendChild(opt);
    });

    // Reset county options
    countySelect.innerHTML = '<option value="">— Select a state first —</option>';
    countySelect.disabled = !selectedState;

    if (selectedState) {
      const counties = Array.from(new Set(meta.filter(m => m.state === selectedState && m.county).map(m => m.county))).sort();
      if (counties.length) {
        countySelect.innerHTML = '<option value="">— Filter by county —</option>';
        counties.forEach(c => {
          const opt = document.createElement('option');
          opt.value = c;
          opt.textContent = c;
          countySelect.appendChild(opt);
        });
        countySelect.disabled = false;
      } else {
        countySelect.innerHTML = `<option value="">No counties detected for ${selectedState}</option>`;
        countySelect.disabled = true;
      }
    }

    // Restore selections if still present
    if (selectedState) {
      stateSelect.value = selectedState;
    }
    if (selectedCounty) {
      const match = Array.from(countySelect.options).some(o => o.value === selectedCounty);
      countySelect.value = match ? selectedCounty : '';
      if (!match) selectedCounty = '';
    }
  }

  function renderUrlList(meta, filter = '', stateFilter = '', countyFilter = '') {
    const listBox = $('#urlLinesBox');
    if (!listBox) return;

    let filtered = meta;

    if (stateFilter) filtered = filtered.filter(m => m.state === stateFilter);
    if (countyFilter) filtered = filtered.filter(m => m.county === countyFilter);

    const q = filter.trim().toLowerCase();
    if (q) {
      if (q.startsWith('state:')) {
        const stateQuery = q.slice(6).trim();
        filtered = filtered.filter(m => (m.state || '').toLowerCase().includes(stateQuery));
      } else if (q.startsWith('county:')) {
        const countyQuery = q.slice(7).trim();
        filtered = filtered.filter(m => (m.county || '').toLowerCase().includes(countyQuery));
      } else {
        filtered = filtered.filter(m => m.url.toLowerCase().includes(q));
      }
    }

    const maxDisplay = 40;
    const items = filtered.slice(0, maxDisplay).map((metaItem, index) => {
      const url = metaItem.url;
      const short = url.length > 60 ? url.slice(0, 57) + '…' : url;
      const metaLabel = [metaItem.state, metaItem.county].filter(Boolean).join(' • ');
      const badge = metaLabel ? `<span class="url-meta">${escapeHtml(metaLabel)}</span>` : '';
      return `<div class="url-sidebar-item" title="${escapeHtml(url)}" data-url="${encodeURIComponent(url)}" role="button" tabindex="0">[${index + 1}] ${escapeHtml(short)} ${badge}</div>`;
    }).join('');

    const more = filtered.length > maxDisplay
      ? `<div class="url-sidebar-more">...and ${filtered.length - maxDisplay} more URLs</div>`
      : '';

    listBox.innerHTML = items + more;

    // Attach click handlers
    listBox.querySelectorAll('.url-sidebar-item').forEach(el => {
      el.addEventListener('click', () => {
        const url = decodeURIComponent(el.getAttribute('data-url'));
        const directUrlTextarea = /** @type {HTMLTextAreaElement|null} */ (document.getElementById('directUrlTextarea'));
        const directRadio = document.querySelector('input[name="fileSource"][value="direct"]');
        if (directUrlTextarea && directRadio instanceof HTMLInputElement && directRadio.checked) {
          const lines = directUrlTextarea.value.split(/\r?\n/).filter(Boolean);
          if (!lines.includes(url)) {
            lines.push(url);
            directUrlTextarea.value = lines.join('\n');
          }
          try { directUrlTextarea.dispatchEvent(new Event('input', { bubbles: true })); } catch (/** @type {any} */ _e) { /* noop */ }
        } else {
          const directUrlField = $('#directUrlField');
          if (directUrlField) {
            (/** @type {any} */ (window)).__tl_helpers.setElValue(directUrlField, url);
            try { directUrlField.dispatchEvent(new Event('input', { bubbles: true })); } catch (/** @type {any} */ _e) { /* noop */ }
          }
        }
      });

      // Keyboard accessibility
      el.addEventListener('keydown', (e) => {
        const ke = /** @type {KeyboardEvent} */ (e);
        if (ke.key === 'Enter' || ke.key === ' ') {
          ke.preventDefault();
          (/** @type {any} */ (window)).__tl_helpers && (/** @type {any} */ (window)).__tl_helpers.safeClick(el);
        }
      });
    });
  }

  async function fetchUrls() {
    try {
      const response = await fetch('/api/urls');
      const data = await response.json();
      cachedUrls = data.urls || [];
      cachedMeta = cachedUrls.map(extractMeta);
      populateTaxonomy(cachedMeta);
      renderUrlList(cachedMeta, lastSearch, selectedState, selectedCounty);
      return cachedUrls;
    } catch (/** @type {any} */ error) {
      console.error('[UrlListManager] Failed to fetch URLs:', error);
      cachedUrls = [];
      cachedMeta = [];
      renderUrlList([], lastSearch, selectedState, selectedCounty);
      return [];
    }
  }

  function init() {
    const searchBox = $('.url-search-box');
    const refreshBtn = $('#refreshUrlListBtn');
    const collapseBtn = $('#btnCollapseUrls');
    const urlsContainer = $('.urls-container');
    const instructionsEl = document.querySelector('.source-instructions');
    let urlsPlaceholder = null;
    let instructionsWrapper = null;
    const stateSelect = $('#urlStateFilter');
    const countySelect = $('#urlCountyFilter');

    if (searchBox) {
      searchBox.addEventListener('input', (e) => {
        lastSearch = (/** @type {any} */ (window)).__tl_helpers.targetValue(e);
        renderUrlList(cachedMeta, lastSearch, selectedState, selectedCounty);
      });
    }

    if (stateSelect && stateSelect instanceof HTMLSelectElement) {
      stateSelect.addEventListener('change', (e) => {
        selectedState = (/** @type {any} */ (window)).__tl_helpers.targetValue(e) || '';
        selectedCounty = '';
        populateTaxonomy(cachedMeta);
        const currentSearch = searchBox instanceof HTMLInputElement ? searchBox.value : '';
        renderUrlList(cachedMeta, currentSearch, selectedState, selectedCounty);
      });
    }

    if (countySelect && countySelect instanceof HTMLSelectElement) {
      countySelect.addEventListener('change', (e) => {
        selectedCounty = (/** @type {any} */ (window)).__tl_helpers.targetValue(e) || '';
        const currentSearch = searchBox instanceof HTMLInputElement ? searchBox.value : '';
        renderUrlList(cachedMeta, currentSearch, selectedState, selectedCounty);
      });
    }

    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        fetchUrls();
      });
    }

    // URL section collapse toggle (default to collapsed)
    if (collapseBtn && urlsContainer) {
      if (instructionsEl && !instructionsWrapper) {
        instructionsWrapper = document.createElement('div');
        instructionsWrapper.className = 'source-instructions-content';
        while (instructionsEl.firstChild) {
          instructionsWrapper.appendChild(instructionsEl.firstChild);
        }
        instructionsEl.appendChild(instructionsWrapper);
      }
      if (!urlsPlaceholder) {
        urlsPlaceholder = document.createElement('div');
        urlsPlaceholder.className = 'urls-container-placeholder';
        urlsContainer.parentNode && urlsContainer.parentNode.insertBefore(urlsPlaceholder, urlsContainer);
      }

      const urlsCollapsed = localStorage.getItem('urlsCollapsed') !== 'false'; // Default to collapsed
      if (urlsCollapsed) {
        urlsContainer.classList.add('collapsed');
        collapseBtn.classList.add('collapsed');
        if (instructionsWrapper) {
          instructionsWrapper.classList.add('hidden');
        }
        if (instructionsEl && urlsContainer.parentNode !== instructionsEl) {
          instructionsEl.appendChild(urlsContainer);
        }
      }
      
      collapseBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const isCollapsed = urlsContainer.classList.toggle('collapsed');
        collapseBtn.classList.toggle('collapsed');
        localStorage.setItem('urlsCollapsed', String(isCollapsed));
        if (isCollapsed) {
          if (instructionsWrapper) {
            instructionsWrapper.classList.add('hidden');
          }
          if (instructionsEl && urlsContainer.parentNode !== instructionsEl) {
            instructionsEl.appendChild(urlsContainer);
          }
        } else {
          if (instructionsWrapper) {
            instructionsWrapper.classList.remove('hidden');
          }
          if (urlsPlaceholder && urlsContainer.parentNode !== urlsPlaceholder.parentNode) {
            urlsPlaceholder.parentNode && urlsPlaceholder.parentNode.insertBefore(urlsContainer, urlsPlaceholder);
          }
        }
      });
    }

    // Initial load
    fetchUrls();
  }
  
  return {
    init,
    fetchUrls,
    refresh: fetchUrls,
    getUrls: () => [...cachedUrls]
  };
})();

// ============================================
// Artifact Panels (Input/Output/Uploads)
// ============================================

const ArtifactPanels = (() => {
  const ROOT_CONFIG = {
    input: { allowDownload: true, label: 'input' },
    output: { allowDownload: true, label: 'output' },
    uploads: { allowDownload: false, label: 'uploads' },
  };
  const DISPLAY_LIMIT = 5;
  const SEARCH_LIMIT = 25;
  const entriesByRoot = new Map();
  const searchTimers = new Map();

  function sortEntries(entries) {
    return [...entries].sort((a, b) => {
      const ma = typeof a.modified === 'number' ? a.modified : 0;
      const mb = typeof b.modified === 'number' ? b.modified : 0;
      if (mb !== ma) return mb - ma;
      return String(a.name || '').localeCompare(String(b.name || ''), undefined, { sensitivity: 'base' });
    });
  }

  function updatePreview(root, entries) {
    const itemsEl = document.getElementById(`${root}PreviewItems`);
    if (!itemsEl) return;
    if (!entries || !entries.length) {
      itemsEl.textContent = 'none yet';
      return;
    }
    const names = entries.slice(0, DISPLAY_LIMIT).map(e => e.name).filter(Boolean);
    itemsEl.textContent = names.length ? names.join(' • ') : 'none yet';
  }

  function formatBytes(bytes) {
    if (typeof bytes !== 'number' || isNaN(bytes) || bytes < 0) return '';
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.min(sizes.length - 1, Math.floor(Math.log(bytes) / Math.log(k)));
    return `${Math.round((bytes / Math.pow(k, i)) * 10) / 10} ${sizes[i]}`;
  }

  function buildDownloadLink(root, path, name) {
    const query = new URLSearchParams({ root, path: path || '', name });
    return `/download_fs?${query.toString()}`;
  }

  function renderPanel(panelEl, entries, config, root, path, options) {
    if (!panelEl) return;
    const limit = (options && typeof options.limit === 'number') ? options.limit : DISPLAY_LIMIT;
    const list = document.createElement('div');
    list.className = 'folder-list';

    if (!entries.length) {
      const empty = document.createElement('div');
      empty.className = 'folder-row muted';
      empty.textContent = 'No files yet';
      list.appendChild(empty);
      panelEl.innerHTML = '';
      panelEl.appendChild(list);
      return;
    }

    entries.slice(0, limit).forEach((entry) => {
      const row = document.createElement('div');
      row.className = 'folder-row';
      row.setAttribute('data-type', entry.type || 'file');
      const icon = entry.type === 'dir' ? '📁' : '📄';
      const sizeText = entry.type === 'file' ? formatBytes(entry.size) : 'Folder';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'folder-name';
      nameSpan.textContent = entry.name;

      const metaSpan = document.createElement('span');
      metaSpan.className = 'folder-meta';
      metaSpan.textContent = sizeText || '';

      const actions = document.createElement('span');
      actions.className = 'folder-actions';
      if (config.allowDownload && entry.type === 'file') {
        const link = document.createElement('a');
        link.href = buildDownloadLink(root, path, entry.name);
        link.textContent = 'Download';
        link.rel = 'noopener noreferrer';
        link.className = 'folder-link';
        actions.appendChild(link);
      } else if (entry.type === 'file') {
        const badge = document.createElement('span');
        badge.className = 'badge badge-alert';
        badge.textContent = 'blocked';
        actions.appendChild(badge);
      }

      row.innerHTML = `${icon}`;
      row.appendChild(nameSpan);
      row.appendChild(metaSpan);
      row.appendChild(actions);
      list.appendChild(row);
    });

    if (entries.length > limit) {
      const more = document.createElement('div');
      more.className = 'folder-row muted';
      more.textContent = `...and ${entries.length - limit} more`;
      list.appendChild(more);
    }

    panelEl.innerHTML = '';
    panelEl.appendChild(list);
  }

  async function loadPanel(panelEl) {
    if (!panelEl) return;
    const root = panelEl.getAttribute('data-root');
    if (!root) return;
    const path = panelEl.getAttribute('data-path') || '';
    const config = ROOT_CONFIG[root] || { allowDownload: false, label: root || '' };

    try {
      const resp = await fetch(`/api/fs/list?root=${encodeURIComponent(root || '')}&path=${encodeURIComponent(path)}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const entries = sortEntries(Array.isArray(data.entries) ? data.entries : []);
      entriesByRoot.set(root, entries);
      updatePreview(root, entries);
      renderPanel(panelEl, entries, config, root || '', path, { limit: DISPLAY_LIMIT });
    } catch (e) {
      console.error('[ArtifactPanels] Failed to load panel', e);
      panelEl.innerHTML = '<div class="folder-row muted">Unable to load</div>';
    }
  }

  function applySearch(root, query) {
    if (!root) return;
    const entries = entriesByRoot.get(root) || [];
    const panel = document.querySelector(`.folder-panel[data-root="${root}"]`);
    if (!panel) return;
    const config = ROOT_CONFIG[root] || { allowDownload: false, label: root || '' };
    const path = panel.getAttribute('data-path') || '';
    const normalized = (query || '').trim().toLowerCase();
    const filtered = normalized
      ? entries.filter((e) => {
          const name = String(e.name || '').toLowerCase();
          const meta = String(e.type || '').toLowerCase();
          return name.includes(normalized) || meta.includes(normalized);
        })
      : entries;
    renderPanel(panel, filtered, config, root, path, { limit: normalized ? SEARCH_LIMIT : DISPLAY_LIMIT });
  }

  function init() {
    const panels = Array.from(document.querySelectorAll('.folder-panel'));
    if (!panels.length) return;
    panels.forEach((panel) => {
      // Preserve locked styling for uploads
      const root = panel.getAttribute('data-root');
      if (root === 'uploads') panel.classList.add('locked');
      loadPanel(panel);
    });

    ['input', 'output'].forEach((root) => {
      const input = document.getElementById(`${root}ArtifactSearch`);
      if (!(input instanceof HTMLInputElement)) return;
      input.addEventListener('input', () => {
        if (searchTimers.has(root)) clearTimeout(searchTimers.get(root));
        const timer = setTimeout(() => applySearch(root, input.value), 250);
        searchTimers.set(root, timer);
      });
    });
  }

  return { init, refreshAll: init, refreshPanel: loadPanel };
})();

// ============================================
// Session Mirror (cross-tab synchronization)
// ============================================

const SessionMirror = (() => {
  const store = new Map();
  const subscribers = new Set();
  
  function notify(sessionId) {
    const meta = store.get(sessionId) || null;
    subscribers.forEach(fn => {
      try {
        fn(sessionId, meta);
      } catch (err) {
        console.error('[SessionMirror] Subscriber error:', err);
      }
    });
  }
  
  function upsert(meta) {
    if (!meta || typeof meta !== 'object') return;
    const sid = meta.session_id;
    if (!sid) return;
    
    const existing = store.get(sid) || {};
    const merged = { ...existing, ...meta };
    store.set(sid, merged);
    notify(sid);
  }
  
  function remove(sessionId) {
    if (!sessionId) return;
    store.delete(sessionId);
    notify(sessionId);
  }
  
  function replace(list) {
    store.clear();
    if (Array.isArray(list)) {
      list.forEach(item => upsert(item));
    }
  }
  
  function get(sessionId) {
    return store.get(sessionId) || null;
  }
  
  function list() {
    return Array.from(store.values());
  }
  
  function subscribe(fn) {
    if (typeof fn !== 'function') return () => {};
    subscribers.add(fn);
    return () => subscribers.delete(fn);
  }
  
  return {
    upsert,
    remove,
    replace,
    get,
    list,
    subscribe
  };
})();

// ============================================
// Table Structure Preview
// ============================================

const TablePreviewManager = (() => {
  const previewsBySession = new Map();
  
  function cloneEntry(entry) {
    if (!entry || typeof entry !== 'object') return null;
    return {
      index: Number(entry.index) || 0,
      total: Number(entry.total) || 0,
      confidence: typeof entry.confidence === 'number' ? entry.confidence : null,
      headers: Array.isArray(entry.headers) ? entry.headers.map(h => String(h)) : [],
      rows: Array.isArray(entry.rows) ? entry.rows.map(row => ({ ...row })) : [],
      contest: entry.contest || '',
      receivedAt: Number(entry.receivedAt) || Date.now(),
    };
  }
  
  function cloneState(state) {
    if (!state || typeof state !== 'object') {
      return { contest: '', entries: [] };
    }
    return {
      contest: state.contest || '',
      entries: Array.isArray(state.entries)
        ? state.entries.map(cloneEntry).filter(Boolean)
        : [],
    };
  }
  
  function getState(sessionId) {
    return cloneState(previewsBySession.get(sessionId));
  }
  
  function record(sessionId, raw) {
    if (!sessionId || !raw || typeof raw !== 'object') return;
    const preview = raw.preview;
    if (!preview || typeof preview !== 'object') return;
    
    const entry = {
      index: Number(raw.candidate_index || raw.preview_index || 1) || 1,
      total: Number(raw.candidates_total || raw.total_candidates || preview.candidates_total) || 0,
      confidence: typeof raw.ml_avg_confidence === 'number' ? raw.ml_avg_confidence : null,
      headers: Array.isArray(preview.headers) ? preview.headers.map(h => String(h)) : [],
      rows: Array.isArray(preview.rows_preview)
        ? preview.rows_preview.map(row => ({ ...row }))
        : [],
      contest: raw.contest || preview.contest || '',
      receivedAt: raw.timestamp || Date.now(),
    };
    
    const state = previewsBySession.get(sessionId) || { contest: entry.contest || '', entries: [] };
    if (entry.contest) state.contest = entry.contest;
    
    const existingIdx = state.entries.findIndex(e => Number(e.index) === Number(entry.index));
    if (existingIdx >= 0) {
      state.entries[existingIdx] = entry;
    } else {
      state.entries.push(entry);
    }
    
    state.entries.sort((a, b) => Number(a.index) - Number(b.index));
    
    // Limit to most recent 12 entries
    if (state.entries.length > 12) {
      state.entries = state.entries.slice(-12);
    }
    
    previewsBySession.set(sessionId, state);
    
    document.dispatchEvent(new CustomEvent('table-preview:updated', {
      detail: { sessionId }
    }));
  }
  
  function showPreview(sessionId) {
    const state = getState(sessionId);
    if (!state.entries || state.entries.length === 0) {
      console.warn('[TablePreview] No preview data available for session:', sessionId);
      return;
    }
    
    const modal = Modal.get();
    if (!modal) return;
    
    const { titleEl, optionsDiv, summaryDiv } = modal;
    titleEl.textContent = `Table Preview: ${state.contest || 'Contest'}`;
    summaryDiv.textContent = `${state.entries.length} preview(s) available`;
    
    optionsDiv.innerHTML = '';
    optionsDiv.classList.add('table-preview-container');
    
    state.entries.forEach(entry => {
      const previewDiv = document.createElement('div');
      previewDiv.className = 'table-preview-entry';
      
      const header = document.createElement('div');
      header.className = 'preview-header';
      header.innerHTML = `
        <strong>Candidate ${entry.index}/${entry.total}</strong>
        ${entry.confidence !== null ? `<span class="badge bg-info ms-2">${(entry.confidence * 100).toFixed(1)}% confidence</span>` : ''}
      `;
      previewDiv.appendChild(header);
      
      if (entry.headers.length > 0) {
        const table = document.createElement('table');
        table.className = 'table table-sm table-preview';
        
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        entry.headers.forEach(h => {
          const th = document.createElement('th');
          th.textContent = h;
          headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        const tbody = document.createElement('tbody');
        entry.rows.slice(0, 5).forEach(row => {
          const tr = document.createElement('tr');
          entry.headers.forEach(h => {
            const td = document.createElement('td');
            td.textContent = row[h] || '';
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        
        previewDiv.appendChild(table);
        
        if (entry.rows.length > 5) {
          const more = document.createElement('div');
          more.className = 'preview-more';
          more.textContent = `...and ${entry.rows.length - 5} more rows`;
          previewDiv.appendChild(more);
        }
      }
      
      optionsDiv.appendChild(previewDiv);
    });
    
    Modal.open();
  }
  
  return {
    record,
    getState,
    showPreview
  };
})();

// ============================================
// Enhanced Folder Browser
// ============================================

const FolderBrowser = (() => {
  const ROOT_LABELS = {
    input: 'Input Files',
    uploads: 'Uploads',
    output: 'Output'
  };
  
  async function fetchDirectory(root, path = '') {
    try {
      const response = await fetch(`/api/fs/list?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`);
      const data = await response.json();
      return data.entries || [];
    } catch (/** @type {any} */ error) {
      console.error('[FolderBrowser] Failed to fetch directory:', error);
      return [];
    }
  }
  
  async function createFolder(root, path, name) {
    try {
      const response = await fetch('/api/fs/mkdir', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json; charset=utf-8' },
        body: JSON.stringify({ root, path, name })
      });
      const data = await response.json();
      return data.success;
    } catch (/** @type {any} */ error) {
      console.error('[FolderBrowser] Failed to create folder:', error);
      return false;
    }
  }
  
  function show(root, initialPath = '', onSelect, options = {}) {
    const modal = Modal.get();
    if (!modal) {
      onSelect?.(null);
      return;
    }
    
    const { titleEl, searchEl, optionsDiv, summaryDiv, closeBtn, cancelBtn } = modal;
    const label = ROOT_LABELS[root] || root;
    const isUploads = root === 'uploads';
    
    let cwd = initialPath || '';
    let allEntries = [];
    let submitted = false;
    
    const finish = (selected) => {
      if (submitted) return;
      submitted = true;
      Modal.close();
      onSelect?.(selected);
    };
    
    function renderBreadcrumb() {
      const parts = cwd.split('/').filter(Boolean);
      const crumbs = [];
      
      const rootCrumb = document.createElement('span');
      rootCrumb.className = 'crumb';
      rootCrumb.textContent = label;
      rootCrumb.addEventListener('click', () => { cwd = ''; refresh(); });
      crumbs.push(rootCrumb);
      
      let acc = '';
      parts.forEach((part, i) => {
        const sep = document.createElement('span');
        sep.textContent = ' / ';
        crumbs.push(sep);
        
        acc += (acc ? '/' : '') + part;
        const partCrumb = document.createElement('span');
        partCrumb.className = 'crumb';
        partCrumb.textContent = part;
        const currentPath = acc;
        partCrumb.addEventListener('click', () => { cwd = currentPath; refresh(); });
        crumbs.push(partCrumb);
      });
      
      const breadcrumb = document.createElement('div');
      breadcrumb.className = 'folder-breadcrumb';
      crumbs.forEach(c => breadcrumb.appendChild(c));
      return breadcrumb;
    }
    
    function renderList(filter = '') {
      const q = filter.trim().toLowerCase();
      let entries = allEntries.slice();
      
      if (q) {
        entries = entries.filter(e =>
          e.name.toLowerCase().includes(q) ||
          (e.type || '').toLowerCase().includes(q)
        );
      }
      
      entries.sort((a, b) => {
        if (a.type !== b.type) return a.type === 'dir' ? -1 : 1;
        return a.name.localeCompare(b.name);
      });
      
      optionsDiv.innerHTML = '';
      optionsDiv.appendChild(renderBreadcrumb());
      
      // Toolbar
      const toolbar = document.createElement('div');
      toolbar.className = 'folder-actions-bar';
      
      const newFolderBtn = document.createElement('button');
      newFolderBtn.type = 'button';
      newFolderBtn.className = 'btn btn-sm';
      newFolderBtn.textContent = '+ New Folder';
      newFolderBtn.addEventListener('click', async () => {
        const name = prompt('Enter folder name:');
        if (name) {
          const success = await createFolder(root, cwd, name);
          if (success) {
            await refresh();
          } else {
            alert('Failed to create folder.');
          }
        }
      });
      toolbar.appendChild(newFolderBtn);
      optionsDiv.appendChild(toolbar);
      
      // Parent directory link
      if (cwd) {
        const upLink = document.createElement('div');
        upLink.className = 'download-option';
        upLink.innerHTML = '⬆️ <b>[..]</b> <small>Up one level</small>';
        upLink.addEventListener('click', () => {
          const parts = cwd.split('/').filter(Boolean);
          parts.pop();
          cwd = parts.join('/');
          refresh();
        });
        optionsDiv.appendChild(upLink);
      }
      
      // Directory entries
      entries.forEach(entry => {
        const item = document.createElement('div');
        item.className = 'download-option';
        item.tabIndex = 0;
        
        const icon = entry.type === 'dir' ? '📁' : '📄';
        const sizeText = entry.size !== null && entry.type === 'file'
          ? `<small class="text-muted ms-2">${formatBytes(entry.size)}</small>`
          : '';
        const blocked = isUploads && entry.type === 'file';
        const blockedBadge = blocked ? ' <span class="badge badge-alert ms-2">blocked</span>' : '';

        item.innerHTML = `${icon} <b>${escapeHtml(entry.name)}</b>${sizeText}${blockedBadge}`;
        
        item.addEventListener('click', () => {
          if (entry.type === 'dir') {
            cwd = cwd ? `${cwd}/${entry.name}` : entry.name;
            refresh();
          } else if (!blocked) {
            finish({ root, path: cwd, name: entry.name, fullPath: cwd ? `${cwd}/${entry.name}` : entry.name });
          }
        });

        item.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && item instanceof HTMLElement && !blocked) {
            item.click();
          }
        });
        
        optionsDiv.appendChild(item);
      });
      
      if (entries.length === 0 && !cwd) {
        const empty = document.createElement('div');
        empty.className = 'text-muted text-center p-3';
        empty.textContent = 'No files or folders found.';
        optionsDiv.appendChild(empty);
      }
    }
    
    async function refresh() {
      allEntries = await fetchDirectory(root, cwd);
      renderList(searchEl.value || '');
    }
    
    titleEl.textContent = `Browse ${label}`;
      summaryDiv.textContent = isUploads ? 'Downloads are disabled in uploads; use uploads only for staging.' : '';
    searchEl.value = '';
    searchEl.addEventListener('input', (e) => renderList((/** @type {any} */ (window)).__tl_helpers.targetValue(e)));
    if (closeBtn instanceof Element) closeBtn.addEventListener('click', () => finish(null));
    if (cancelBtn instanceof Element) cancelBtn.addEventListener('click', () => finish(null));
    
    Modal.open();
    refresh();
  }
  
  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 10) / 10 + ' ' + sizes[i];
  }
  
  return {
    show
  };
})();

// ============================================
// Download Modal
// ============================================

const DownloadModal = (() => {
  function show(options, summary, callback) {
    const modal = Modal.get();
    if (!modal) {
      callback?.(null);
      return;
    }
    
    const { titleEl, searchEl, optionsDiv, summaryDiv, closeBtn, cancelBtn } = modal;
    titleEl.textContent = 'Select Download';
    summaryDiv.textContent = summary || '';
    
    let submitted = false;
    
    const finish = (value) => {
      if (submitted) return;
      submitted = true;
      Modal.close();
      callback?.(value);
    };
    
    function renderList(filter = '') {
      const q = filter.trim().toLowerCase();
      const filtered = options.filter(opt =>
        opt.format.toLowerCase().includes(q) ||
        opt.filename.toLowerCase().includes(q) ||
        opt.contest.toLowerCase().includes(q)
      );
      
      // Group by contest
      const groups = {};
      filtered.forEach(opt => {
        const key = opt.contest || 'Other';
        if (!groups[key]) groups[key] = [];
        groups[key].push(opt);
      });
      
      optionsDiv.innerHTML = '';
      optionsDiv.classList.add('table-preview-container');
      
      Object.keys(groups).sort().forEach(groupName => {
        const groupDiv = document.createElement('div');
        groupDiv.className = 'download-group';
        
        const header = document.createElement('div');
        header.className = 'download-group-header';
        header.innerHTML = `<b>${escapeHtml(groupName)}</b> (${groups[groupName].length})`;
        groupDiv.appendChild(header);
        
        groups[groupName].forEach(opt => {
          const item = document.createElement('div');
          item.className = 'download-option';
          item.tabIndex = 0;
          item.innerHTML = `
            <span class="badge bg-primary me-2">${escapeHtml(opt.format.toUpperCase())}</span>
            <span class="download-filename">${highlight(opt.filename, q)}</span>
            <span class="download-type ms-2">${highlight(opt.contest, q)}</span>
          `;
          
          item.addEventListener('click', () => finish(opt.index));
          item.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') finish(opt.index);
          });
          
          groupDiv.appendChild(item);
        });
        
        optionsDiv.appendChild(groupDiv);
      });
    }
    
    function highlight(text, query) {
      if (!query) return escapeHtml(text);
      const regex = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
      return escapeHtml(text).replace(regex, match => `<mark>${match}</mark>`);
    }
    
    searchEl.value = '';
    searchEl.addEventListener('input', (e) => renderList((/** @type {any} */ (window)).__tl_helpers.targetValue(e)));
    if (closeBtn instanceof Element) closeBtn.addEventListener('click', () => finish(null));
    if (cancelBtn instanceof Element) cancelBtn.addEventListener('click', () => finish(null));
    
    renderList();
    Modal.open();
    searchEl.focus();
  }
  
  return {
    show
  };
})();

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  console.log('[Parser UI] Initializing modern interface...');
  
  // Initialize theme manager
  ThemeManager.init();
  
  // Initialize swipe gesture handler for sidebars
  SwipeHandler.init();
  
  // Initialize pipeline phase system
  PipelineManager.init();
  
  // Initialize manual upload file selection
  ManualUploadManager.init();
  
  // Initialize URL list
  UrlListManager.init();

  // Initialize artifact folder panels (input/output/uploads)
  ArtifactPanels.init();

  // Navigation overflow ("More" menu) for small screens
  const navLinks = Array.from(document.querySelectorAll('.navbar-links .nav-link'));
  const navMoreToggle = document.getElementById('btnNavMore');
  const navMoreDropdown = document.getElementById('navMoreDropdown');
  function setHiddenWithInert(el, hidden) {
    if (!el) return;
    try {
      // Prefer native inert when available
      if ('inert' in el) {
        el.inert = hidden;
      } else if (hidden) {
        el.setAttribute('inert', '');
      } else {
        el.removeAttribute('inert');
      }
    } catch (e) {
      // ignore inert errors
    }
    // Manage focusable descendants to avoid axe complaining about focusable children inside aria-hidden
    try {
      const focusable = el.querySelectorAll('a[href], button, input, select, textarea, [tabindex]');
      focusable.forEach((f) => {
        if (!(f instanceof HTMLElement)) return;
        if (hidden) {
          if (f.hasAttribute('tabindex')) {
            f.setAttribute('data-_savedTabindex', f.getAttribute('tabindex') || '');
          } else {
            f.setAttribute('data-_savedTabindex', 'none');
          }
          f.setAttribute('tabindex', '-1');
        } else {
          const saved = f.getAttribute('data-_savedTabindex');
          if (saved && saved !== 'none') {
            f.setAttribute('tabindex', saved);
          } else {
            f.removeAttribute('tabindex');
          }
          try { f.removeAttribute('data-_savedTabindex'); } catch (e) {}
        }
      });
    } catch (e) {
      /* ignore */
    }
    try { el.setAttribute('aria-hidden', hidden ? 'true' : 'false'); } catch (e) {}
  }

  // Ensure dropdown is initialized as inert/hidden as early as possible to avoid
  // headless snapshots where aria-hidden contains focusable children.
  try {
    // Start hidden by default; apply inert/tabindex suppression now
    if (navMoreDropdown) setHiddenWithInert(navMoreDropdown, true);
  } catch (e) {
    /* ignore */
  }

  function setNavDropdown(open) {
    if (!navMoreDropdown || !navMoreToggle) return;
    navMoreDropdown.classList.toggle('open', open);
    // Ensure dropdown is visible to headless checks by applying inline styles
    try {
      if (open) {
        navMoreDropdown.style.display = 'block';
        navMoreDropdown.style.opacity = '1';
        navMoreDropdown.style.zIndex = '20000';
        // Position dropdown near toggle to ensure it's in-viewport for headless tests
        try {
          const r = navMoreToggle.getBoundingClientRect();
          navMoreDropdown.style.position = 'fixed';
          navMoreDropdown.style.left = `${Math.max(6, Math.round(r.left))}px`;
          navMoreDropdown.style.top = `${Math.round(r.bottom + 6)}px`;
          navMoreDropdown.style.minWidth = '160px';
        } catch (errPos) {
          /* ignore */
        }
      } else {
        navMoreDropdown.style.display = 'none';
        navMoreDropdown.style.opacity = '';
        navMoreDropdown.style.zIndex = '';
        navMoreDropdown.style.position = '';
        navMoreDropdown.style.left = '';
        navMoreDropdown.style.top = '';
        navMoreDropdown.style.minWidth = '';
      }
    } catch (e) {}
    // Use inert/tabindex management to avoid focusable children inside aria-hidden
    try { setHiddenWithInert(navMoreDropdown, !open); } catch (e) {}
    try { navMoreDropdown.toggleAttribute('hidden', !open); } catch (e) {}
    navMoreToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  function toggleNavDropdown() {
    if (!navMoreDropdown) return;
    const isOpen = navMoreDropdown.classList.contains('open');
    setNavDropdown(!isOpen);
  }

  // Ensure a dedicated close helper exists so event handlers can call it safely.
  // Some older builds referenced `closeNavDropdown` directly; provide a stable
  // implementation that delegates to `setNavDropdown(false)`.
  function closeNavDropdown() {
    try {
      if (navMoreDropdown && navMoreDropdown.classList.contains('open')) {
        setNavDropdown(false);
      }
    } catch (e) {
      /* ignore */
    }
  }

  /*
   * Test note: `tools/check_nav_dropdown.js` (Puppeteer) exercises
   * `syncNavOverflow()` and `closeNavDropdown()` to detect runtime
    /** @typedef {HTMLDivElement & { _timeoutId?: any }} ToastElement */


  function syncNavOverflow() {
    if (!navMoreDropdown) return;
    // Clear first so previously cloned links are not re-counted on resize/rerender.
    navMoreDropdown.innerHTML = '';
    // Recompute links dynamically so changes to the DOM are reflected.
    const navbarLinksContainer = document.querySelector('.navbar-links');
    const currentLinks = navbarLinksContainer
      ? Array.from(navbarLinksContainer.querySelectorAll(':scope > .nav-link'))
      : Array.from(document.querySelectorAll('.navbar-links .nav-link')).filter((link) => !(navMoreDropdown && navMoreDropdown.contains(link)));
    if (!currentLinks.length) {
      // ensure dropdown is inert/hidden when nothing to show
      setHiddenWithInert(navMoreDropdown, true);
      return;
    }
    currentLinks.forEach((link) => {
      try {
        const clone = /** @type {HTMLElement} */ (link.cloneNode(true));
        clone.addEventListener('click', closeNavDropdown);
        navMoreDropdown.appendChild(clone);
      } catch (/** @type {any} */ _e) {
        // ignore clone failures
      }
    });
    // After cloning, immediately ensure it's inert when hidden so axe won't flag focusable children.
    setHiddenWithInert(navMoreDropdown, true);
  }

  if (navMoreToggle && navMoreDropdown) {
    syncNavOverflow();
    // Observe changes to the navbar links container so overflow stays in sync
    try {
      /**
       * @typedef {Element & { _navOverflowObserver?: MutationObserver }} NavLinksContainer
       */
      const navbarLinksContainer = document.querySelector('.navbar-links');
      if (navbarLinksContainer && typeof MutationObserver !== 'undefined') {
        const mo = new MutationObserver(debounce(() => {
          try { syncNavOverflow(); } catch (e) { /* ignore */ }
        }, 120));
        mo.observe(navbarLinksContainer, { childList: true, subtree: true, attributes: true });
        // store reference for potential teardown/debugging
        try { /** @type {NavLinksContainer} */ (navbarLinksContainer)._navOverflowObserver = mo; } catch (e) { /* ignore */ }
      }
    } catch (e) {
      /* ignore observer failures */
    }
    navMoreToggle.addEventListener('click', (e) => {
      e.preventDefault();
      toggleNavDropdown();
    });
    document.addEventListener('click', (e) => {
      if (!navMoreDropdown.classList.contains('open')) return;
      const tgt = (e && e.target && (e.target instanceof Node)) ? e.target : null;
      if ((/** @type {any} */ (window)).__tl_helpers.nodeContains(navMoreDropdown, tgt)) return;
      if (navMoreToggle === tgt || (/** @type {any} */ (window)).__tl_helpers.nodeContains(navMoreToggle, tgt)) return;
      closeNavDropdown();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeNavDropdown();
    });
    const debouncedSyncNavOverflow = debounce(syncNavOverflow, 200);
    window.addEventListener('resize', () => {
      try { debouncedSyncNavOverflow(); } catch (e) {}
      if (window.innerWidth > 720) closeNavDropdown();
    });
    // Also handle orientation changes on mobile
    window.addEventListener('orientationchange', () => {
      try { debouncedSyncNavOverflow(); } catch (e) {}
    });
  }

  // Expose helpers for headless checks and legacy scripts
  try {
    if (typeof window !== 'undefined') {
      const w = /** @type {any} */ (window);
      w.syncNavOverflow = syncNavOverflow;
      w.closeNavDropdown = closeNavDropdown;
    }
  } catch (e) { /* ignore */ }

  // -------------------------------
  // Card row height sync (JS resize sync)
  // -------------------------------
  function syncCardHeights() {
    try {
      const grid = document.querySelector('.results-grid');
      if (!grid) return;

      // Unwrap any previously created row wrappers to avoid nesting
      const prevRows = Array.from(grid.querySelectorAll(':scope > .results-row'));
      prevRows.forEach((row) => {
        try {
          while (row.firstChild) grid.insertBefore(row.firstChild, row);
          row.remove();
        } catch (e) { /* ignore */ }
      });

      const cards = Array.from(grid.querySelectorAll(':scope > .result-card'));
      if (!cards.length) return;

      // Group cards by their visual top offset
      const rows = new Map();
      cards.forEach((c) => {
        const top = Math.round((/** @type {HTMLElement} */ (c)).getBoundingClientRect().top);
        const list = rows.get(top) || [];
        list.push(c);
        rows.set(top, list);
      });

      // Sort row keys to preserve visual order
      const sortedTops = Array.from(rows.keys()).sort((a,b) => a - b);
      sortedTops.forEach((top) => {
        const rowCards = rows.get(top) || [];
        let max = 0;
        rowCards.forEach((c) => { max = Math.max(max, /** @type {HTMLElement} */ (c).offsetHeight); });

        // Create a wrapper that will carry the CSS custom property for the row
        const wrapper = document.createElement('div');
        wrapper.className = 'results-row';
        // use display: contents to avoid adding an extra box to layout while still providing inheritance
        wrapper.style.display = 'contents';
        wrapper.style.setProperty('--row-min-height', max + 'px');

        // Append wrapper and move row cards into it in order
        grid.appendChild(wrapper);
        rowCards.forEach((c) => { wrapper.appendChild(c); });
      });
    } catch (e) {
      /* ignore measurement errors */
    }
  }

  const debouncedSyncCardHeights = debounce(syncCardHeights, 120);
  // initial run
  try { syncCardHeights(); } catch (e) {}
  // wire to resize/orientation
  window.addEventListener('resize', () => { try { debouncedSyncCardHeights(); } catch (e) {} });
  window.addEventListener('orientationchange', () => { try { debouncedSyncCardHeights(); } catch (e) {} });

  // Observe DOM changes in the results grid to re-run sync (e.g., cards added/removed)
  try {
    const resultsGrid = document.querySelector('.results-grid');
    if (resultsGrid && typeof MutationObserver !== 'undefined') {
      /** @typedef {Element & { _cardSizeObserver?: MutationObserver }} ResultsGridContainer */
      const ro = new MutationObserver(debounce(() => { try { syncCardHeights(); } catch (e) {} }, 120));
      ro.observe(resultsGrid, { childList: true, subtree: true, attributes: true });
      try { /** @type {ResultsGridContainer} */ (resultsGrid)._cardSizeObserver = ro; } catch (e) {}
    }
  } catch (e) { /* ignore observer failures */ }

  // Ensure navbar action buttons reliably trigger expected behaviors.
  // Uses event delegation on `.navbar-actions` so buttons that are moved in the DOM
  // or re-rendered by templates still respond without needing re-binding.
  (function ensureNavbarBindings() {
    const navbarActions = document.querySelector('.navbar-actions');
    if (!navbarActions) return;

    function handleNavbarActionClick(e) {
      const btn = (/** @type {any} */ (window)).__tl_helpers.targetClosest(e, 'button') || ((e.target instanceof Element && e.target.tagName === 'BUTTON') ? e.target : null);
      if (!btn) return;
      const id = btn.id;
      try {
        switch (id) {
          case 'btnCommandPalette': {
            const cp = $('#commandPalette'); const ci = $('#commandInput');
            if (cp) cp.classList.remove('hidden');
            if (ci) ci.focus();
            break;
          }
          case 'btnNotifications': {
            // Lightweight notifications placeholder
            try { showToast('No notifications', 'info'); } catch (err) { console.debug('Notifications not wired', err); }
            break;
          }
          case 'btnTheme': {
            try { ThemeManager.toggleTheme(); } catch (err) { console.debug('Theme toggle failed', err); }
            break;
          }
          case 'btnToggleRightSidebar': {
            if (typeof (/** @type {any} */ (window)).openRight === 'function') { try { (/** @type {any} */ (window)).openRight(); } catch (e) { /* swallow */ } } else { try { btn.click(); } catch (err) {} }
            break;
          }
          case 'btnNavMore': {
            try { toggleNavDropdown(); } catch (err) { btn.click(); }
            break;
          }
          default: break;
        }
      } catch (err) {
        console.debug('Navbar action handler error', err);
      }
    }

    // Attach single delegated listener (idempotent)
    navbarActions.removeEventListener('click', handleNavbarActionClick);
    navbarActions.addEventListener('click', handleNavbarActionClick);
  })();
  
  // Load real data from warehouse API (with fallback to sample data)
  loadRealData();
  
  // Initialize state filters
  $$('#filterState option').forEach(opt => {
    const optVal = /** @type {HTMLOptionElement} */ (opt).value;
    if (!STATES.includes(optVal)) {
      STATES.forEach(state => {
        if (!Array.from($$('#filterState option')).some(o => /** @type {HTMLOptionElement} */ (o).value === state)) {
          const option = document.createElement('option');
          option.value = state;
          option.textContent = state;
          $('#filterState').appendChild(option);
        }
      });
    }
  });
  
  // Request initial session ID
  socket.emit('join', {
    username: localStorage.getItem('username') || 'anonymous',
  });
  
  // Socket.IO handlers for Session Mirror
  socket.on('session_list', (data) => {
    if (data && Array.isArray(data.sessions)) {
      SessionMirror.replace(data.sessions);
    }
  });
  
  socket.on('session_state', (data) => {
    if (data && data.metadata) {
      SessionMirror.upsert(data.metadata);
    }
    updateOverviewStrip(data);
  });
  
  socket.on('session_deleted', (data) => {
    if (data && data.session_id) {
      SessionMirror.remove(data.session_id);
    }
  });
  
  // Socket.IO handler for Table Preview
  socket.on('parser_output', (log) => {
    if (log && log.type === 'table_preview' && log.session_id) {
      TablePreviewManager.record(log.session_id, log);
    }
  });

  // ============================================
  // Footer Session List Toggle
  // ============================================
  const sessionFooter = $('#sessionFooter');
  const footerPreview = $('#footerPreview');
  
  if (sessionFooter && footerPreview) {
    // Toggle expanded state on click
    footerPreview.addEventListener('click', (e) => {
      e.stopPropagation();
      sessionFooter.classList.toggle('expanded');
      localStorage.setItem('footerExpanded', String(sessionFooter.classList.contains('expanded')));
    });
    
    // Restore previously expanded state
    const wasExpanded = localStorage.getItem('footerExpanded') === 'true';
    if (wasExpanded) {
      sessionFooter.classList.add('expanded');
    }
    
    // Close footer when clicking outside
    document.addEventListener('click', (e) => {
      const tgt = (e && e.target && (e.target instanceof Node)) ? e.target : null;
      if (!((/** @type {any} */ (window)).__tl_helpers.nodeContains(sessionFooter, tgt)) && sessionFooter.classList.contains('expanded')) {
        sessionFooter.classList.remove('expanded');
      }
    });
  }
  
    // Sidebar toggle (off-canvas) behavior for small screens
    if ((/** @type {any} */ (window)).__tl_sidebarUnified) {
      // Unified sidebar controller active; skip legacy toggle wiring to avoid conflicts.
      updateSessionsList();
      console.log('[Parser UI] Initialization complete');
      return;
    }
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    const sidebarRight = document.querySelector('.sidebar-right');
    const sidebarEl = document.getElementById('sidebar') || sidebarRight;
    const sidebarBackdrop = document.querySelector('.sidebar-backdrop') || document.querySelector('.mobile-sidebar-overlay');

    // If the primary .sidebar-toggle is inside an off-canvas #sidebar, force it to be fixed/visible on small screens
    try {
      if (sidebarToggle && window.matchMedia && window.matchMedia('(max-width: 768px)').matches) {
        // Prefer CSS-driven styling; add a helper class that provides fixed positioning
        sidebarToggle.classList.add('sidebar-toggle-floating');
        sidebarToggle.setAttribute('aria-expanded', sidebarEl && sidebarEl.classList.contains('sidebar-open') ? 'true' : 'false');
      }
    } catch (err) {
      // ignore
    }

    // Ensure a floating left-toggle exists on mobile so automation can click it even when #sidebar is off-canvas
    let floatingLeftToggle = document.getElementById('sidebarToggleFloating');
    if (!floatingLeftToggle && window.matchMedia && window.matchMedia('(max-width: 768px)').matches) {
      try {
        floatingLeftToggle = document.createElement('button');
        floatingLeftToggle.id = 'sidebarToggleFloating';
        floatingLeftToggle.className = 'sidebar-toggle sidebar-toggle-floating';
        floatingLeftToggle.setAttribute('aria-controls', 'sidebar');
        floatingLeftToggle.setAttribute('aria-expanded', 'false');
        floatingLeftToggle.setAttribute('title', 'Toggle sidebar');
        floatingLeftToggle.innerText = '\u2630';
        // Rely on CSS for styling; add helper class for floating appearance
        floatingLeftToggle.classList.add('sidebar-toggle-floating');
        document.body.appendChild(floatingLeftToggle);
      } catch (e) {
        // ignore DOM creation errors
      }
    }

    function openSidebar() {
      if (sidebarEl) sidebarEl.classList.add('sidebar-open');
      if (sidebarRight) {
        sidebarRight.classList.add('sidebar-open');
        sidebarRight.classList.add('open');
      }
      if (sidebarBackdrop) sidebarBackdrop.classList.add('visible');
      document.body.classList.add('no-scroll');
      document.body.classList.add('sidebar-right-open');
    }

    function closeSidebar() {
      if (sidebarEl) sidebarEl.classList.remove('sidebar-open');
      if (sidebarRight) {
        sidebarRight.classList.remove('sidebar-open');
        sidebarRight.classList.remove('open');
      }
      if (sidebarBackdrop) sidebarBackdrop.classList.remove('visible');
      document.body.classList.remove('no-scroll');
      document.body.classList.remove('sidebar-right-open');
    }

    function toggleSidebar() {
      const isOpen = (sidebarEl && sidebarEl.classList.contains('sidebar-open')) || (sidebarRight && (sidebarRight.classList.contains('sidebar-open') || sidebarRight.classList.contains('open')));
      if (isOpen) closeSidebar(); else openSidebar();
    }

    if (sidebarToggle) {
      sidebarToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleSidebar();
      });
    }

    // Bind floating left toggle if present
    if (floatingLeftToggle) {
      floatingLeftToggle.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleSidebar();
        try { floatingLeftToggle.setAttribute('aria-expanded', String((sidebarEl && sidebarEl.classList.contains('sidebar-open')) ? 'true' : 'false')); } catch (err) {}
      });
    }

    if (sidebarBackdrop) {
      sidebarBackdrop.addEventListener('click', () => closeSidebar());
    }

    // Close sidebar on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        closeSidebar();
        if (sessionFooter && sessionFooter.classList.contains('expanded')) sessionFooter.classList.remove('expanded');
      }
    });

    // Ensure sidebar closes on navigation actions
    document.addEventListener('click', (e) => {
      const target = (e && e.target && (e.target instanceof Element)) ? e.target : null;
      if (!sidebarRight) return;
      if (sidebarRight.classList.contains('sidebar-open') && target && !sidebarRight.contains(target) && !target.closest('.sidebar-toggle')) {
        closeSidebar();
      }
    });

    updateSessionsList();
  
  console.log('[Parser UI] Initialization complete');
});
