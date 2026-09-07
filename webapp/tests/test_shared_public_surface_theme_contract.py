from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DF = ROOT / "webapp/templates/data_framework.html"
WL = ROOT / "webapp/templates/worklist.html"
THEME = ROOT / "webapp/static/css/public_surface_theme.css"
NAV = ROOT / "webapp/static/css/public_app_nav.css"

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def test_both_priority_surfaces_load_shared_theme_after_page_css():
    df = _read(DF)
    wl = _read(WL)
    assert "css/public_surface_theme.css" in df
    assert "css/public_surface_theme.css" in wl
    assert df.index("css/data_framework.css") < df.index("css/public_surface_theme.css")
    assert wl.index("css/workflow_public.css") < wl.index("css/public_surface_theme.css")

def test_theme_is_visual_only_and_does_not_own_navigation():
    theme = _read(THEME)
    compact = theme.replace(" ", "").lower()
    assert 'body[data-page="data-framework"]' in theme
    assert 'body[data-page="workflow-public"]' in theme
    assert ".ep-app-nav" not in theme
    assert "display:none" not in compact
    assert "visibility:hidden" not in compact
    assert "pointer-events:none" not in compact
    assert "grid-template" not in theme
    assert "display:grid" not in compact
    assert "display:flex" not in compact

def test_theme_preserves_status_and_accessibility_semantics():
    theme = _read(THEME)
    compact = theme.replace(" ", "")
    assert "--workflow-ok:var(--ep-success)" in theme
    assert "--workflow-warning:var(--ep-warning)" in theme
    assert "--workflow-error:var(--ep-danger)" in theme
    assert "--focus-outline-color:var(--ep-accent-strong)" in theme
    assert "@media(prefers-reduced-motion:reduce)" in compact
    assert theme.count("{") == theme.count("}")

def test_shared_nav_remains_separate_authority():
    nav = _read(NAV)
    theme = _read(THEME)
    assert ".ep-app-nav__link:focus-visible" in nav
    assert "@media (max-width: 860px)" in nav
    assert ".ep-app-nav" not in theme
