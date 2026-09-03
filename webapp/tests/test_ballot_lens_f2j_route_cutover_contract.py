from __future__ import annotations

from pathlib import Path


ROOT = Path(".")
MAIN = ROOT / "webapp/Smart_Elections_Parser_Webapp.py"
PUBLIC_PAGES = ROOT / "webapp/parser/routes/public_pages_blueprint.py"
F2_TEMPLATE = ROOT / "webapp/templates/ballot_lens_f2.html"
LEGACY_TEMPLATE = ROOT / "webapp/templates/ballot_lens.html"
LEGACY_MODERN_JS = ROOT / "webapp/static/js/ballot_lens_modern.js"
LEGACY_PUBLIC_JS = ROOT / "webapp/static/js/ballot_lens_public_registry.js"
LEGACY_MODERN_CSS = ROOT / "webapp/static/css/ballot_lens_modern.css"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _ballot_lens_body() -> str:
    source = _read(MAIN)
    start = source.index("def ballot_lens():")
    end = source.index("def ballot_lens_modern():", start)
    return source[start:end]


def test_j1_primary_ballot_lens_route_is_f2_only():
    body = _ballot_lens_body()

    assert "BALLOT_LENS_UI_VARIANT" not in body
    assert '{"legacy", "f2"}' not in body
    assert '"ballot_lens.html"' not in body
    assert '"ballot_lens_f2.html"' in body
    assert 'ballot_lens_ui_variant = "f2"' in body
    assert body.count("load_ballot_lens_f2_assets()") == 1
    assert 'return "Ballot Lens F2 assets unavailable", 503' in body


def test_j1_cutover_preserves_trusted_upload_and_file_list_gates():
    body = _ballot_lens_body()

    assert '_require_client_cert("ballot_lens_upload")' in body
    assert '_guarded_ingestion_allowed("ballot_lens_upload")' in body
    assert "_save_uploaded_file(" in body
    assert "get_request_principal()" in body
    assert "ballot_lens_trusted_controls = bool(principal)" in body
    assert '"input_files": []' in body
    assert '"output_files": []' in body
    assert '"uploaded_files": []' in body
    assert body.index("ballot_lens_trusted_controls") < body.index(
        "get_all_file_lists()"
    )


def test_j1_transitional_modern_alias_still_redirects_to_primary_route():
    source = _read(MAIN)
    start = source.index("def ballot_lens_modern():")
    end = source.index("def worklist():", start)
    alias = source[start:end]

    assert 'return redirect(url_for("ballot_lens"))' in alias

    blueprint = _read(PUBLIC_PAGES)
    assert '@bp.route("/ballot_lens", methods=["GET", "POST"]' in blueprint
    assert '@bp.route("/ballot_lens_modern", methods=["GET"]' in blueprint
    assert 'return _call_handler("ballot_lens_modern")' in blueprint


def test_j1_preserves_legacy_assets_only_for_later_reference_extinction():
    for path in (
        LEGACY_TEMPLATE,
        LEGACY_MODERN_JS,
        LEGACY_PUBLIC_JS,
        LEGACY_MODERN_CSS,
    ):
        assert path.is_file()

    legacy = _read(LEGACY_TEMPLATE)
    f2 = _read(F2_TEMPLATE)

    assert 'id="btnRunParser2"' in legacy
    assert "ballot_lens_modern.js" in legacy
    assert "ballot_lens_public_registry.js" in legacy
    assert "ballot_lens_modern.css" in legacy

    assert 'id="ballotLensF2Root"' in f2
    assert "ballot_lens_modern.js" not in f2
    assert "ballot_lens_public_registry.js" not in f2
    assert "ballot_lens_modern.css" not in f2


def test_j1_does_not_claim_legacy_asset_extinction_yet():
    source = "\n".join(
        _read(path)
        for path in (
            LEGACY_TEMPLATE,
            LEGACY_MODERN_JS,
            LEGACY_PUBLIC_JS,
            LEGACY_MODERN_CSS,
        )
    )
    assert "btnRunParser2" in source
