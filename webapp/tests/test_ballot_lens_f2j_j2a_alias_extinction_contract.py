from __future__ import annotations

from pathlib import Path

ROOT = Path(".")
MAIN = ROOT / "webapp/Smart_Elections_Parser_Webapp.py"
PUBLIC_PAGES = ROOT / "webapp/parser/routes/public_pages_blueprint.py"
F2_TEMPLATE = ROOT / "webapp/templates/ballot_lens_f2.html"
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

def test_j2a_primary_f2_template_remains_isolated_from_retired_bundle():
    f2 = _read(F2_TEMPLATE)
    assert 'id="ballotLensF2Root"' in f2
    assert "ballot_lens_modern.js" not in f2
    assert "ballot_lens_public_registry.js" not in f2
    assert "ballot_lens_modern.css" not in f2

def test_j2a_handoff_allows_j2b_to_remove_dashboard_legacy_css_consumers():
    quality = _read(QUALITY_DASHBOARD)
    status = _read(URL_STATUS_DASHBOARD)

    assert "ballot_lens_modern.css" not in quality
    assert "ballot_lens_modern.css" not in status
    assert "quality_dashboard.css" in quality
    assert "quality_dashboard.css" in status

def test_j2a_handoff_remains_valid_after_j2c_dependency_extinction():
    main = _read(MAIN)
    blueprint = _read(PUBLIC_PAGES)
    assert "def ballot_lens_modern():" not in main
    assert "/ballot_lens_modern" not in blueprint
    assert 'id="ballotLensF2Root"' in _read(F2_TEMPLATE)

