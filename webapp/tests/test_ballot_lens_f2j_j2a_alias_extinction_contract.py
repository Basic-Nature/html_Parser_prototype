from __future__ import annotations

from pathlib import Path

ROOT = Path(".")
MAIN = ROOT / "webapp/Smart_Elections_Parser_Webapp.py"
PUBLIC_PAGES = ROOT / "webapp/parser/routes/public_pages_blueprint.py"
F2_TEMPLATE = ROOT / "webapp/templates/ballot_lens_f2.html"
LEGACY_TEMPLATE = ROOT / "webapp/templates/ballot_lens.html"
LEGACY_JS = ROOT / "webapp/static/js/ballot_lens_modern.js"
LEGACY_PUBLIC_JS = ROOT / "webapp/static/js/ballot_lens_public_registry.js"
LEGACY_CSS = ROOT / "webapp/static/css/ballot_lens_modern.css"
QUALITY_DASHBOARD = ROOT / "webapp/templates/quality_dashboard.html"
URL_STATUS_DASHBOARD = ROOT / "webapp/templates/url_status_dashboard.html"

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")

def _ballot_lens_body() -> str:
    source = _read(MAIN)
    start = source.index("def ballot_lens():")
    end = source.index("def worklist():", start)
    return source[start:end]

def test_j2a_primary_ballot_lens_route_remains_f2_only():
    body = _ballot_lens_body()
    assert '"ballot_lens_f2.html"' in body
    assert '"ballot_lens.html"' not in body
    assert 'ballot_lens_ui_variant = "f2"' in body
    assert body.count("load_ballot_lens_f2_assets()") == 1
    assert 'return "Ballot Lens F2 assets unavailable", 503' in body

def test_j2a_transitional_alias_handler_and_public_route_are_extinct():
    main = _read(MAIN)
    blueprint = _read(PUBLIC_PAGES)
    assert "def ballot_lens_modern():" not in main
    assert '"ballot_lens_modern": ballot_lens_modern' not in main
    assert '@bp.route("/ballot_lens", methods=["GET", "POST"]' in blueprint
    assert 'return _call_handler("ballot_lens")' in blueprint
    assert "/ballot_lens_modern" not in blueprint
    assert "ballot_lens_modern_route" not in blueprint
    assert '_call_handler("ballot_lens_modern")' not in blueprint

def test_j2a_preserves_legacy_assets_for_later_consumer_extinction():
    for path in (LEGACY_TEMPLATE, LEGACY_JS, LEGACY_PUBLIC_JS, LEGACY_CSS):
        assert path.is_file()
    legacy = _read(LEGACY_TEMPLATE)
    f2 = _read(F2_TEMPLATE)
    assert "ballot_lens_modern.js" in legacy
    assert "ballot_lens_public_registry.js" in legacy
    assert "ballot_lens_modern.css" in legacy
    assert 'id="ballotLensF2Root"' in f2
    assert "ballot_lens_modern.js" not in f2
    assert "ballot_lens_public_registry.js" not in f2
    assert "ballot_lens_modern.css" not in f2

def test_j2a_explicitly_defers_dashboard_css_consumer_migration_to_j2b():
    assert "ballot_lens_modern.css" in _read(QUALITY_DASHBOARD)
    assert "ballot_lens_modern.css" in _read(URL_STATUS_DASHBOARD)

def test_j2a_does_not_claim_legacy_asset_delete_gate():
    assert LEGACY_TEMPLATE.is_file()
    assert LEGACY_JS.is_file()
    assert LEGACY_PUBLIC_JS.is_file()
    assert LEGACY_CSS.is_file()
