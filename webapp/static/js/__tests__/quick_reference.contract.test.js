/* eslint-env jest */
/**
 * Contract tests for quick_reference.js — QuickReference module.
 * Verifies search/filter behavior, section scroll, and keyboard shortcut handling.
 */

const fs = require('fs');
const path = require('path');

describe('QuickReference module contract', () => {
  function loadScript() {
    const src = fs.readFileSync(
      path.join(__dirname, '..', 'quick_reference.js'),
      'utf8'
    );
    const script = document.createElement('script');
    script.textContent = src;
    document.head.appendChild(script);
  }

  beforeAll(() => {
    document.head.innerHTML = '';
    // Minimal DOM required by QuickReference.init()
    document.body.innerHTML = [
      '<div id="keyboard-shortcuts"><h2>Shortcuts</h2></div>',
      '<div id="feature-finder"><h2>Feature Finder</h2></div>',
      '<div id="help"><h2>Help</h2></div>',
      '<div id="quickRefContent">',
      '  <table>',
      '    <thead><tr><th>Feature</th><th>Location</th></tr></thead>',
      '    <tbody>',
      '      <tr><td>Parser</td><td>webapp/parser</td></tr>',
      '      <tr><td>Database</td><td>webapp/db</td></tr>',
      '      <tr><td>Auth</td><td>webapp/auth</td></tr>',
      '    </tbody>',
      '  </table>',
      '</div>',
      '<div class="nav-card"><h3>Parser Overview</h3><p>Main parser section</p></div>',
      '<button id="homeBtn" type="button">Home</button>',
    ].join('');

    // Stub localStorage (no-op writes, null reads)
    // Stub localStorage via Storage.prototype (jsdom non-configurable property workaround)
    jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {});
    jest.spyOn(Storage.prototype, 'getItem').mockReturnValue(null);

    // Stub clipboard API
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText: jest.fn().mockResolvedValue(undefined) },
    });

    loadScript();
    // In jest/jsdom readyState is 'complete', so the script calls init() immediately.
  });

  afterAll(() => {
    jest.restoreAllMocks();
  });

  // ─── Search box injection ─────────────────────────────────────────────────

  test('injects a search box above the table', () => {
    const searchBox = document.getElementById('feature-search');
    expect(searchBox).not.toBeNull();
    expect(searchBox.tagName).toBe('INPUT');
    expect(searchBox.getAttribute('type')).toBe('text');
  });

  test('search box has accessible aria-label', () => {
    const searchBox = document.getElementById('feature-search');
    expect(searchBox.getAttribute('aria-label')).toBeTruthy();
  });

  // ─── Search / filter behavior ─────────────────────────────────────────────

  test('search input filters table rows — matching rows visible', () => {
    const searchBox = /** @type {HTMLInputElement} */ (document.getElementById('feature-search'));
    searchBox.value = 'parser';
    searchBox.dispatchEvent(new Event('input'));

    const rows = Array.from(document.querySelectorAll('table tbody tr'));
    const visible = rows.filter((r) => !r.classList.contains('qr-row-hidden'));

    expect(visible.length).toBeGreaterThanOrEqual(1);
    expect(visible.some((r) => r.textContent.toLowerCase().includes('parser'))).toBe(true);
  });

  test('search input hides non-matching rows', () => {
    const searchBox = /** @type {HTMLInputElement} */ (document.getElementById('feature-search'));
    searchBox.value = 'database';
    searchBox.dispatchEvent(new Event('input'));

    const rows = Array.from(document.querySelectorAll('table tbody tr'));
    const hidden = rows.filter((r) => r.classList.contains('qr-row-hidden'));
    const visible = rows.filter((r) => !r.classList.contains('qr-row-hidden'));

    // "Database" row visible, "Parser" and "Auth" hidden
    expect(visible.some((r) => r.textContent.toLowerCase().includes('database'))).toBe(true);
    expect(hidden.length).toBeGreaterThanOrEqual(1);
  });

  test('empty search query reveals all table rows', () => {
    const searchBox = /** @type {HTMLInputElement} */ (document.getElementById('feature-search'));
    searchBox.value = '';
    searchBox.dispatchEvent(new Event('input'));

    const hidden = document.querySelectorAll('table tbody tr.qr-row-hidden');
    expect(hidden).toHaveLength(0);
  });

  test('search query is persisted to localStorage', () => {
    const searchBox = /** @type {HTMLInputElement} */ (document.getElementById('feature-search'));
    searchBox.value = 'auth';
    searchBox.dispatchEvent(new Event('input'));

    expect(Storage.prototype.setItem).toHaveBeenCalledWith('quickref_search', 'auth');
  });

  // ─── Keyboard shortcut: Escape blurs focused input ────────────────────────

  test('Escape key blurs a focused input element', () => {
    const searchBox = /** @type {HTMLInputElement} */ (document.getElementById('feature-search'));
    searchBox.focus();
    const blurSpy = jest.spyOn(searchBox, 'blur');

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    expect(blurSpy).toHaveBeenCalled();
    blurSpy.mockRestore();
  });

  // ─── Keyboard shortcut: / focuses search (only when not in an input) ───────

  test('/ shortcut does not scroll feature-finder when user is typing in an input', () => {
    const searchBox = /** @type {HTMLInputElement} */ (document.getElementById('feature-search'));
    searchBox.focus();

    const featureFinder = document.getElementById('feature-finder');
    const scrollMock = jest.fn();
    if (featureFinder) featureFinder.scrollIntoView = scrollMock;

    document.dispatchEvent(
      new KeyboardEvent('keydown', { key: '/', bubbles: true, cancelable: true })
    );

    expect(scrollMock).not.toHaveBeenCalled();
    searchBox.blur();
  });

  // ─── QuickReference is accessible on window ───────────────────────────────

  test('QuickReference module is accessible on window', () => {
    const qr = /** @type {any} */ (window).QuickReference;
    expect(qr).toBeDefined();
    expect(typeof qr.init).toBe('function');
    expect(typeof qr.scrollToSection).toBe('function');
    expect(typeof qr.focusSearchBox).toBe('function');
  });

  test('scrollToSection focuses section element by id', () => {
    // scrollToSection finds the h2 inside the section by text content
    const heading = Array.from(document.querySelectorAll('h2')).find(
      (h) => h.textContent.includes('Feature Finder')
    );
    const scrollMock = jest.fn();
    if (heading) heading.scrollIntoView = scrollMock;

    const qr = /** @type {any} */ (window).QuickReference;
    qr.scrollToSection('feature-finder');

    expect(scrollMock).toHaveBeenCalled();
  });

  test('focusSearchBox focuses the search input element', () => {
    const searchBox = /** @type {HTMLInputElement} */ (document.getElementById('feature-search'));
    const focusSpy = jest.spyOn(searchBox, 'focus');

    const qr = /** @type {any} */ (window).QuickReference;
    qr.focusSearchBox();

    expect(focusSpy).toHaveBeenCalled();
    focusSpy.mockRestore();
  });
});
