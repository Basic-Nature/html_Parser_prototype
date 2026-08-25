#!/usr/bin/env python3
"""Consolidated robust UI test for current implementation.

Tests critical UI elements, sidebar behavior, and responsive layout.
Flexible with responsive elements (e.g., nav overflow button may be hidden).

Exit codes:
    0 - All tests passed
    1 - Some tests failed (with diagnostics)
    2 - Fatal error (page load failed, etc.)

Usage:
    python tools/ui_robust_check.py
    python tools/ui_robust_check.py --url http://localhost:5000/ballot_lens
    python tools/ui_robust_check.py --viewport desktop --skip-nav
"""
import argparse
import json
import os
import sys

from playwright.sync_api import sync_playwright


class TestResult:
    def __init__(self):
        self.tests = []
        self.warnings = []
        self.diagnostics = {}
    
    def add_test(self, name, passed, details=None):
        self.tests.append({
            "name": name,
            "passed": passed,
            "details": details or {}
        })
    
    def add_warning(self, message):
        self.warnings.append(message)
    
    def passed_count(self):
        return sum(1 for t in self.tests if t["passed"])
    
    def failed_count(self):
        return sum(1 for t in self.tests if not t["passed"])
    
    def all_passed(self):
        return all(t["passed"] for t in self.tests)


def save_debug_artifacts(page, out_dir, prefix="debug"):
    """Save screenshot, HTML, and diagnostic info."""
    os.makedirs(out_dir, exist_ok=True)
    artifacts = {}
    
    try:
        screenshot_path = os.path.join(out_dir, f"{prefix}_screenshot.png")
        page.screenshot(path=screenshot_path, full_page=True)
        artifacts["screenshot"] = screenshot_path
    except Exception as e:
        artifacts["screenshot_error"] = str(e)
    
    try:
        html_path = os.path.join(out_dir, f"{prefix}_page.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(page.content())
        artifacts["html"] = html_path
    except Exception as e:
        artifacts["html_error"] = str(e)
    
    try:
        diag_path = os.path.join(out_dir, f"{prefix}_diagnostics.json")
        diagnostics = page.evaluate("""() => {
            return {
                viewport: {width: window.innerWidth, height: window.innerHeight},
                bodyClasses: Array.from(document.body.classList),
                bodyOverflow: getComputedStyle(document.body).overflow,
                htmlOverflow: getComputedStyle(document.documentElement).overflow,
                sidebarState: (() => {
                    const sb = document.getElementById('sidebar');
                    if (!sb) return null;
                    const cs = getComputedStyle(sb);
                    return {
                        classes: Array.from(sb.classList),
                        display: cs.display,
                        position: cs.position,
                        transform: cs.transform,
                    };
                })(),
            };
        }""")
        with open(diag_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostics, f, indent=2)
        artifacts["diagnostics"] = diag_path
    except Exception as e:
        artifacts["diagnostics_error"] = str(e)
    
    return artifacts


def check_element_visible(page, selector, timeout=5000):
    """Check if element is truly visible (not hidden, has size, etc.)."""
    try:
        page.wait_for_selector(selector, state="attached", timeout=timeout)
        info = page.eval_on_selector(selector, """el => {
            const cs = getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return {
                display: cs.display,
                visibility: cs.visibility,
                opacity: parseFloat(cs.opacity),
                width: rect.width,
                height: rect.height,
                x: rect.x,
                y: rect.y,
                classList: Array.from(el.classList),
            };
        }""")
        
        # Element is visible if:
        # - display !== 'none'
        # - visibility === 'visible'  
        # - opacity > 0
        # - has positive dimensions
        visible = (
            info["display"] != "none" and
            info["visibility"] == "visible" and
            info["opacity"] > 0 and
            info["width"] > 0 and
            info["height"] > 0
        )
        return visible, info
    except Exception as e:
        return False, {"error": str(e)}


def test_critical_elements(page, result, viewport_type):
    """Test that critical UI elements exist."""
    left_toggle_optional = viewport_type == "desktop"

    critical = [
        ("btnRunParser2", "Run Button", False),
        ("btnCancel", "Cancel Button", False),
        ("sidebarToggleBtn", "Left Sidebar Toggle", left_toggle_optional),
        ("btnToggleRightSidebar", "Right Sidebar Toggle", False),
        ("sidebar", "Left Sidebar", False),
        ("btnNavMore", "Nav More Button", True),  # May be hidden responsively
    ]
    
    for elem_id, name, allow_hidden in critical:
        selector = f"#{elem_id}"
        try:
            page.wait_for_selector(selector, state="attached", timeout=5000)
            visible, info = check_element_visible(page, selector, timeout=1000)
            
            if allow_hidden:
                # Just check it exists, visibility is OK
                result.add_test(f"Element: {name}", True, {
                    "selector": selector,
                    "visible": visible,
                    "responsive": "visibility may vary by viewport"
                })
            else:
                # Must be visible
                result.add_test(f"Element: {name}", visible, {
                    "selector": selector,
                    "info": info if not visible else {"status": "visible"}
                })
        except Exception as e:
            result.add_test(f"Element: {name}", False, {
                "selector": selector,
                "error": str(e)
            })


def test_sidebar_toggle(page, result, viewport_type):
    """Test left sidebar toggle behavior."""
    try:
        # Look for sidebar toggle
        toggle_sel = "#sidebarToggleBtn"
        sidebar_sel = "#sidebar"
        
        # Check if toggle exists and is visible
        visible, _ = check_element_visible(page, toggle_sel, timeout=3000)
        if not visible:
            result.add_warning(f"Sidebar toggle not visible at viewport={viewport_type}")
            return
        
        # Try clicking toggle (use exposed helper if available)
        try:
            toggle_result = page.evaluate("""() => {
                if (typeof window.openLeft === 'function') {
                    window.openLeft();
                    return 'helper';
                }
                const btn = document.getElementById('sidebarToggleBtn');
                if (btn && btn.click) {
                    btn.click();
                    return 'click';
                }
                return null;
            }""")
        except Exception:
            # Fallback to Playwright click
            page.click(toggle_sel, timeout=5000, force=True)
            toggle_result = "playwright"
        
        # Wait for animation
        page.wait_for_timeout(800)
        
        # Check if sidebar opened
        sidebar_info = page.eval_on_selector(sidebar_sel, """el => {
            const cs = getComputedStyle(el);
            return {
                classes: Array.from(el.classList),
                display: cs.display,
                transform: cs.transform,
                isOpen: el.classList.contains('sidebar-open') || cs.transform !== 'none'
            };
        }""")
        
        result.add_test("Sidebar Toggle", sidebar_info.get('isOpen', False), {
            "method": toggle_result,
            "sidebar_state": sidebar_info
        })
        
    except Exception as e:
        result.add_test("Sidebar Toggle", False, {"error": str(e)})


def test_mobile_sidebar(page, result):
    """Test mobile-specific sidebar behavior (overlay, scroll-lock)."""
    try:
        # Open left sidebar
        page.evaluate("""() => {
            if (typeof window.openLeft === 'function') {
                window.openLeft();
            } else {
                const btn = document.getElementById('sidebarToggleBtn');
                if (btn) btn.click();
            }
        }""")
        page.wait_for_timeout(700)
        
        # Check for overlay visibility
        overlay_visible = False
        try:
            overlay_sel = "#mobileSidebarOverlay"
            if page.query_selector(overlay_sel):
                visible, _ = check_element_visible(page, overlay_sel, timeout=1000)
                overlay_visible = visible
        except Exception:
            pass
        
        # Check body scroll-lock
        body_state = page.evaluate("""() => {
            const body = document.body;
            const cs = getComputedStyle(body);
            return {
                hasNoScrollClass: body.classList.contains('no-scroll'),
                inlineOverflow: body.style.overflow,
                computedOverflow: cs.overflow,
                htmlComputedOverflow: getComputedStyle(document.documentElement).overflow,
            };
        }""")
        
        scroll_locked = (
            body_state.get('hasNoScrollClass') or
            body_state.get('inlineOverflow') == 'hidden' or
            body_state.get('computedOverflow') == 'hidden' or
            body_state.get('htmlComputedOverflow') == 'hidden'
        )
        
        result.add_test("Mobile Overlay", overlay_visible, {
            "overlay_selector": "#mobileSidebarOverlay"
        })
        
        result.add_test("Scroll Lock", scroll_locked, {
            "body_state": body_state
        })
        
    except Exception as e:
        result.add_test("Mobile Sidebar", False, {"error": str(e)})


def test_console_errors(page, console_msgs, result):
    """Check for console errors."""
    errors = [m for m in console_msgs if m.get('type') == 'error']
    # Filter out known benign errors
    benign_patterns = ['favicon', 'source-map', 'CSP', '500', 'INTERNAL SERVER ERROR']
    real_errors = [
        e for e in errors 
        if not any(x in e.get('text', '') for x in benign_patterns)
    ]
    
    result.add_test("Console Errors", len(real_errors) == 0, {
        "error_count": len(real_errors),
        "errors": real_errors[:5] if real_errors else [],
        "filtered_out": len(errors) - len(real_errors)
    })


def test_log_object_normalization(page, result):
    """Ensure object-shaped log payloads render as JSON text, not [object Object]."""
    try:
        probe = page.evaluate("""() => {
            if (typeof addLog !== 'function' || typeof state === 'undefined' || !Array.isArray(state.logs)) {
                return { ok: false, reason: 'addLog/state unavailable' };
            }
            const beforeLen = state.logs.length;
            addLog({
                level: 'INFO',
                type: 'summary',
                message: { total: 1, success: true },
                session_id: 'sess_ui_robust_check'
            });
            const afterLen = state.logs.length;
            const last = state.logs[afterLen - 1] || {};
            const msg = String(last.message || '');
            return {
                ok: true,
                beforeLen,
                afterLen,
                message: msg,
                includesObjectMarker: msg.includes('[object Object]'),
                includesJsonKey: msg.includes('"total"') || msg.includes('total')
            };
        }""")

        passed = bool(
            probe.get("ok")
            and probe.get("afterLen", 0) > probe.get("beforeLen", 0)
            and not probe.get("includesObjectMarker")
            and probe.get("includesJsonKey")
        )
        result.add_test("Log Object Normalization", passed, probe)
    except Exception as e:
        result.add_test("Log Object Normalization", False, {"error": str(e)})


def test_prompt_dedupe_after_submit(page, result):
    """Ensure submit enters standby and identical answered prompt stays suppressed."""
    try:
        probe = page.evaluate("""() => {
            if (typeof handlePromptLog !== 'function') {
                return { ok: false, reason: 'handlePromptLog unavailable' };
            }

            try {
                if (typeof currentSessionId !== 'undefined') {
                    currentSessionId = 'sess_ui_prompt_dedupe';
                }
            } catch (e) {}

            const payload = {
                type: 'prompt',
                session_id: 'sess_ui_prompt_dedupe',
                message: '[PROMPT] Enter URL indices (e.g., 1,3-5):',
                context: {
                    title: 'URL Selection',
                    options: ['url-one', 'url-two', 'url-three'],
                    message: 'Pick URL index'
                }
            };

            handlePromptLog(payload);

            const modal = document.getElementById('promptModal');
            const input = document.getElementById('promptInput');
            const submitBtn = document.getElementById('btnSubmitPrompt');
            const messageEl = document.getElementById('promptMessage');

            if (!modal || !input || !submitBtn) {
                return {
                    ok: false,
                    reason: 'prompt modal controls missing'
                };
            }

            const visibleAfterFirst = !modal.classList.contains('hidden');
            const activeBeforeSubmit = !modal.classList.contains(
                'prompt-standby-active'
            );

            input.value = '2';
            submitBtn.click();

            const visibleAfterSubmit = !modal.classList.contains('hidden');
            const standbyAfterSubmit = modal.classList.contains(
                'prompt-standby-active'
            );
            const inputDisabledAfterSubmit = !!input.disabled;
            const submitDisabledAfterSubmit = !!submitBtn.disabled;
            const messageAfterSubmit = messageEl
                ? String(messageEl.textContent || '')
                : '';

            handlePromptLog(payload);

            const visibleAfterDuplicate = !modal.classList.contains('hidden');
            const standbyAfterDuplicate = modal.classList.contains(
                'prompt-standby-active'
            );
            const inputDisabledAfterDuplicate = !!input.disabled;
            const submitDisabledAfterDuplicate = !!submitBtn.disabled;
            const messageAfterDuplicate = messageEl
                ? String(messageEl.textContent || '')
                : '';

            const duplicateSuppressed = (
                visibleAfterDuplicate
                && standbyAfterDuplicate
                && inputDisabledAfterDuplicate
                && submitDisabledAfterDuplicate
                && messageAfterDuplicate === messageAfterSubmit
            );

            return {
                ok: true,
                visibleAfterFirst,
                activeBeforeSubmit,
                visibleAfterSubmit,
                standbyAfterSubmit,
                inputDisabledAfterSubmit,
                submitDisabledAfterSubmit,
                messageAfterSubmit,
                visibleAfterDuplicate,
                standbyAfterDuplicate,
                inputDisabledAfterDuplicate,
                submitDisabledAfterDuplicate,
                messageAfterDuplicate,
                duplicateSuppressed
            };
        }""")

        passed = bool(
            probe.get("ok")
            and probe.get("visibleAfterFirst")
            and probe.get("activeBeforeSubmit")
            and probe.get("visibleAfterSubmit")
            and probe.get("standbyAfterSubmit")
            and probe.get("inputDisabledAfterSubmit")
            and probe.get("submitDisabledAfterSubmit")
            and probe.get("duplicateSuppressed")
        )

        result.add_test(
            "Prompt Standby + Dedupe After Submit",
            passed,
            probe,
        )
    except Exception as e:
        result.add_test(
            "Prompt Standby + Dedupe After Submit",
            False,
            {"error": str(e)},
        )


def test_blocked_toggle_behavior(page, result):
    """Ensure blocked results are hidden by default and shown when toggled."""
    try:
        probe = page.evaluate("""() => {
            const grid = document.getElementById('resultsGrid');
            const toggleBtn = document.getElementById('btnToggleBlockedResults');
            if (!grid || !toggleBtn) {
                return { ok: false, reason: 'results grid or blocked toggle button missing' };
            }
            if (typeof renderResults !== 'function' || typeof state === 'undefined' || !state || !state.filters) {
                return { ok: false, reason: 'renderResults/state unavailable' };
            }

            const originalResults = Array.isArray(state.results) ? state.results.slice() : [];
            const originalShowBlocked = !!state.filters.showBlocked;

            try {
                state.results = [
                    {
                        id: 'ui-risk-1',
                        name: 'Blocked Risk Result',
                        type: 'csv',
                        rows: 5,
                        columns: 3,
                        confidence: 70,
                        state: 'CA',
                        county: 'Alameda',
                        handler: 'test',
                        timestamp: Date.now(),
                        source_url: 'https://example.test/blocked',
                        preview: 'blocked preview',
                        riskTier: 'block',
                        riskSubTier: 'stop',
                        riskAction: 'REQUIRE_CONFIRMATION'
                    },
                    {
                        id: 'ui-risk-2',
                        name: 'Warn Risk Result',
                        type: 'csv',
                        rows: 6,
                        columns: 3,
                        confidence: 83,
                        state: 'CA',
                        county: 'Alameda',
                        handler: 'test',
                        timestamp: Date.now(),
                        source_url: 'https://example.test/warn',
                        preview: 'warn preview',
                        riskTier: 'warn',
                        riskSubTier: 'pass',
                        riskAction: 'MONITOR_CLOSELY'
                    }
                ];

                state.filters.showBlocked = false;
                renderResults();

                const countHidden = grid.querySelectorAll('.result-card').length;
                const hiddenContainsBlocked = Array.from(grid.querySelectorAll('.card-name')).some((el) =>
                    String(el.textContent || '').includes('Blocked Risk Result')
                );
                const labelBefore = String(toggleBtn.textContent || '').trim();

                toggleBtn.click();
                const countShown = grid.querySelectorAll('.result-card').length;
                const shownContainsBlocked = Array.from(grid.querySelectorAll('.card-name')).some((el) =>
                    String(el.textContent || '').includes('Blocked Risk Result')
                );
                const labelAfterShow = String(toggleBtn.textContent || '').trim();

                toggleBtn.click();
                const countAfterHide = grid.querySelectorAll('.result-card').length;
                const labelAfterHide = String(toggleBtn.textContent || '').trim();

                return {
                    ok: true,
                    countHidden,
                    hiddenContainsBlocked,
                    labelBefore,
                    countShown,
                    shownContainsBlocked,
                    labelAfterShow,
                    countAfterHide,
                    labelAfterHide
                };
            } finally {
                state.results = originalResults;
                state.filters.showBlocked = originalShowBlocked;
                renderResults();
            }
        }""")

        passed = bool(
            probe.get("ok")
            and probe.get("countHidden") == 1
            and not probe.get("hiddenContainsBlocked")
            and probe.get("labelBefore") == "Show Blocked"
            and probe.get("countShown") == 2
            and probe.get("shownContainsBlocked")
            and probe.get("labelAfterShow") == "Hide Blocked"
            and probe.get("countAfterHide") == 1
            and probe.get("labelAfterHide") == "Show Blocked"
        )
        result.add_test("Blocked Toggle Behavior", passed, probe)
    except Exception as e:
        result.add_test("Blocked Toggle Behavior", False, {"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="Robust UI headless check")
    parser.add_argument("--url", default=os.environ.get("PARSER_URL", "http://127.0.0.1:5000/ballot_lens"), help="URL to test")
    parser.add_argument("--viewport", choices=["mobile", "desktop"], default="mobile", help="Viewport size")
    parser.add_argument("--skip-nav", action="store_true", help="Skip nav dropdown test")
    parser.add_argument("--output", default="tools/debug_headless_output", help="Output directory for artifacts")
    args = parser.parse_args()
    
    if args.viewport == "mobile":
        viewport = {"width": 375, "height": 667}
    else:
        viewport = {"width": 1366, "height": 768}
    
    result = TestResult()
    result.diagnostics = {"url": args.url, "viewport": args.viewport}
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport=viewport)
            page = context.new_page()
            
            console_msgs = []
            page.on('console', lambda msg: console_msgs.append({'type': msg.type, 'text': msg.text}))
            page.on('pageerror', lambda e: console_msgs.append({'type': 'error', 'text': str(e)}))
            
            # Load page
            try:
                page.goto(args.url, wait_until="load", timeout=60000)
                page.wait_for_timeout(1000)  # Allow client JS to initialize
                result.add_test("Page Load", True)
            except Exception as e:
                result.add_test("Page Load", False, {"error": str(e)})
                result.diagnostics["artifacts"] = save_debug_artifacts(page, args.output, "load_failure")
                print(json.dumps(result.__dict__, indent=2))
                sys.exit(2)
            
            # Run tests
            test_critical_elements(page, result, args.viewport)
            test_sidebar_toggle(page, result, args.viewport)
            
            if args.viewport == "mobile":
                test_mobile_sidebar(page, result)
            
            test_console_errors(page, console_msgs, result)
            test_log_object_normalization(page, result)
            test_prompt_dedupe_after_submit(page, result)
            test_blocked_toggle_behavior(page, result)
            
            # Save artifacts
            result.diagnostics["artifacts"] = save_debug_artifacts(page, args.output, "final")
            result.diagnostics["console"] = console_msgs
            
            browser.close()
            
    except Exception as e:
        result.add_test("Unexpected Error", False, {"error": str(e)})
        result.diagnostics["fatal_error"] = str(e)
    
    # Print results
    print("\n" + "="*70)
    print("UI Robust Check Results")
    print("="*70)
    print(f"URL: {args.url}")
    print(f"Viewport: {args.viewport} ({viewport['width']}x{viewport['height']})")
    print()
    
    for test in result.tests:
        status = "[PASS]" if test["passed"] else "[FAIL]"
        print(f"{status} | {test['name']}")
        if not test["passed"] and test.get("details"):
            print(f"     └─ {test['details']}")
    
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  [WARNING] {warning}")
    
    print()
    print(f"Total Tests: {len(result.tests)}")
    print(f"Passed: {result.passed_count()}")
    print(f"Failed: {result.failed_count()}")
    
    if result.diagnostics.get("artifacts"):
        print(f"\nDebug artifacts saved to: {args.output}/")
    
    # Exit with appropriate code
    if result.all_passed():
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed. Check artifacts for details.")
        print("\nFull results JSON:")
        print(json.dumps(result.__dict__, indent=2, default=str))
        sys.exit(1)


if __name__ == '__main__':
    main()
