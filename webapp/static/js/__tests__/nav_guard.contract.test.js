/* eslint-env jest */
/**
 * Contract tests for nav_guard.js — client-side URL safety guard.
 * Verifies that dangerous schemes are blocked, allowed internal paths pass,
 * and target=_blank links receive rel="noopener noreferrer" hardening.
 */

const fs = require('fs');
const path = require('path');

describe('nav_guard URL safety contract', () => {
  let warnSpy;

  beforeAll(() => {
    // Place target=_blank links in DOM BEFORE loading the script so the
    // DOMContentLoaded handler can process them for rel-attribute hardening.
    document.body.innerHTML = [
      '<a id="blank-no-rel" href="https://external.example.com" target="_blank"></a>',
      '<a id="blank-with-noopener" href="https://external.example.com" target="_blank" rel="noopener"></a>',
      '<a id="internal-link" href="/history" data-safe-nav="/history"></a>',
    ].join('');

    const src = fs.readFileSync(path.join(__dirname, '..', 'nav_guard.js'), 'utf8');
    const script = document.createElement('script');
    script.textContent = src;
    document.head.appendChild(script);
    // Manually fire DOMContentLoaded so the IIFE listener runs in jsdom.
    document.dispatchEvent(new Event('DOMContentLoaded'));
  });

  beforeEach(() => {
    warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  /**
   * Helper: inject a link with data-safe-nav, dispatch click, remove it.
   * Returns the click MouseEvent so callers can inspect defaultPrevented.
   */
  function clickSafeNavLink(href) {
    const a = document.createElement('a');
    a.href = href;
    a.setAttribute('data-safe-nav', href);
    document.body.appendChild(a);
    const evt = new MouseEvent('click', { bubbles: true, cancelable: true });
    a.dispatchEvent(evt);
    a.remove();
    return evt;
  }

  function blockedWarningCount() {
    return warnSpy.mock.calls.filter((args) => String(args[0]).includes('Blocked')).length;
  }

  // ─── Dangerous-scheme blocking ───────────────────────────────────────────

  test('blocks javascript: scheme and emits console.warn', () => {
    clickSafeNavLink('javascript:alert(1)');
    expect(blockedWarningCount()).toBeGreaterThanOrEqual(1);
  });

  test('blocks data: scheme (potential XSS vector)', () => {
    clickSafeNavLink('data:text/html,<h1>xss</h1>');
    expect(blockedWarningCount()).toBeGreaterThanOrEqual(1);
  });

  test('blocks vbscript: scheme', () => {
    clickSafeNavLink('vbscript:MsgBox("xss")');
    expect(blockedWarningCount()).toBeGreaterThanOrEqual(1);
  });

  test('blocks protocol-relative //host URLs', () => {
    clickSafeNavLink('//evil.com/steal');
    expect(blockedWarningCount()).toBeGreaterThanOrEqual(1);
  });

  test('blocks percent-encoded javascript: scheme (double-encoding evasion)', () => {
    // %6A%61%76%61%73%63%72%69%70%74%3A = "javascript:"
    clickSafeNavLink('%6A%61%76%61%73%63%72%69%70%74%3Aalert%281%29');
    expect(blockedWarningCount()).toBeGreaterThanOrEqual(1);
  });

  test('click preventDefault is called for blocked javascript: link', () => {
    const evt = clickSafeNavLink('javascript:alert(1)');
    expect(evt.defaultPrevented).toBe(true);
  });

  // ─── Allowed internal paths ───────────────────────────────────────────────

  test('allows /history without blocking warning', () => {
    clickSafeNavLink('/history');
    expect(blockedWarningCount()).toBe(0);
  });

  test('allows /ballot_lens without blocking warning', () => {
    clickSafeNavLink('/ballot_lens');
    expect(blockedWarningCount()).toBe(0);
  });

  test('allows /data_framework without blocking warning', () => {
    clickSafeNavLink('/data_framework');
    expect(blockedWarningCount()).toBe(0);
  });

  test('allows /api/ prefixed routes without blocking warning', () => {
    clickSafeNavLink('/api/election_data');
    expect(blockedWarningCount()).toBe(0);
  });

  // ─── target=_blank rel hardening ──────────────────────────────────────────

  test('adds noopener and noreferrer to target=_blank link with no rel', () => {
    const a = document.getElementById('blank-no-rel');
    const rel = (a.getAttribute('rel') || '').toLowerCase();
    expect(rel).toContain('noopener');
    expect(rel).toContain('noreferrer');
  });

  test('adds noreferrer to target=_blank link that only had noopener', () => {
    const a = document.getElementById('blank-with-noopener');
    const rel = (a.getAttribute('rel') || '').toLowerCase();
    expect(rel).toContain('noreferrer');
  });
});
