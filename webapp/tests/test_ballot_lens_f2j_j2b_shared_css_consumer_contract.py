from __future__ import annotations

from pathlib import Path


ROOT = Path(".")
MAIN = ROOT / "webapp/Smart_Elections_Parser_Webapp.py"
PUBLIC_PAGES = ROOT / "webapp/parser/routes/public_pages_blueprint.py"
QUALITY_DASHBOARD = ROOT / "webapp/templates/quality_dashboard.html"
URL_STATUS_DASHBOARD = ROOT / "webapp/templates/url_status_dashboard.html"
SHARED_DASHBOARD_CSS = ROOT / "webapp/static/css/quality_dashboard.css"



def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_j2b_both_runtime_dashboards_leave_legacy_ballot_lens_css():
    quality = _read(QUALITY_DASHBOARD)
    status = _read(URL_STATUS_DASHBOARD)

    assert "ballot_lens_modern.css" not in quality
    assert "ballot_lens_modern.css" not in status

    assert quality.count("quality_dashboard.css") == 1
    assert status.count("quality_dashboard.css") == 1


def test_j2b_shared_dashboard_css_owns_required_semantic_tokens():
    css = _read(SHARED_DASHBOARD_CSS)

    required = (
        "--surface-0: #0f1419;",
        "--surface-1: var(--brand-bg);",
        "--surface-2: #2a3642;",
        "--surface-3: #334155;",
        "--text-strong: #e2e8f0;",
        "--text-subtle: #cbd5e1;",
        "--text-muted-base: #6b7280;",
        "--border-weak: #334155;",
        "--accent-primary: var(--accent-1);",
        "--accent-success: var(--accent-success-strong);",
        "--accent-warning: var(--accent-warning-strong);",
        "--accent-danger: var(--accent-danger-strong);",
        "--radius-sm: 4px;",
        "--radius-md: 6px;",
        "--transition-fast: 150ms ease;",
    )
    for token in required:
        assert token in css


def test_j2b_shared_dashboard_css_preserves_existing_page_base_behavior():
    css = _read(SHARED_DASHBOARD_CSS)

    assert "*, *::before, *::after {" in css
    assert "box-sizing: border-box;" in css
    assert "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;" in css
    assert "background: var(--bg-primary);" in css
    assert "color: var(--text-primary);" in css
    assert "min-height: 100vh;" in css
    assert ':root[data-theme="light"] {' in css


def test_j2b_shared_css_migration_is_independent_of_retired_bundle_files():
    quality = _read(QUALITY_DASHBOARD)
    status = _read(URL_STATUS_DASHBOARD)
    css = _read(SHARED_DASHBOARD_CSS)

    assert "ballot_lens_modern.css" not in quality
    assert "ballot_lens_modern.css" not in status
    assert "Independent of legacy Ballot Lens assets." in css

def test_j2b_preserves_j2a_alias_extinction():
    main = _read(MAIN)
    blueprint = _read(PUBLIC_PAGES)

    assert "def ballot_lens_modern():" not in main
    assert '"ballot_lens_modern": ballot_lens_modern' not in main
    assert "/ballot_lens_modern" not in blueprint
    assert "ballot_lens_modern_route" not in blueprint


def test_j2b_direct_runtime_legacy_css_consumers_are_zero():
    direct_runtime_consumers = [
        path
        for path in (QUALITY_DASHBOARD, URL_STATUS_DASHBOARD)
        if "ballot_lens_modern.css" in _read(path)
    ]
    assert direct_runtime_consumers == []


def test_j2b_handoff_preserves_alias_extinction_and_shared_css_authority():
    main = _read(MAIN)
    blueprint = _read(PUBLIC_PAGES)
    assert "def ballot_lens_modern():" not in main
    assert "/ballot_lens_modern" not in blueprint
    assert "quality_dashboard.css" in _read(QUALITY_DASHBOARD)
    assert "quality_dashboard.css" in _read(URL_STATUS_DASHBOARD)

