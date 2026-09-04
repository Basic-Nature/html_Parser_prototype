from __future__ import annotations

from pathlib import Path


ROOT = Path(".")
MAIN = ROOT / "webapp/Smart_Elections_Parser_Webapp.py"
PUBLIC_PAGES = ROOT / "webapp/parser/routes/public_pages_blueprint.py"
F2_TEMPLATE = ROOT / "webapp/templates/ballot_lens_f2.html"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _ballot_lens_body() -> str:
    source = _read(MAIN)
    start = source.index("def ballot_lens():")
    end = source.index("def worklist():", start)
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


def test_j2a_transitional_modern_alias_authority_is_extinct():
    source = _read(MAIN)
    blueprint = _read(PUBLIC_PAGES)

    assert "def ballot_lens_modern():" not in source
    assert '"ballot_lens_modern": ballot_lens_modern' not in source

    assert '@bp.route("/ballot_lens", methods=["GET", "POST"]' in blueprint
    assert 'return _call_handler("ballot_lens")' in blueprint
    assert "/ballot_lens_modern" not in blueprint
    assert "ballot_lens_modern_route" not in blueprint
    assert '_call_handler("ballot_lens_modern")' not in blueprint


def test_j1_f2_template_is_isolated_from_retired_bundle():
    f2 = _read(F2_TEMPLATE)
    assert 'id="ballotLensF2Root"' in f2
    assert "ballot_lens_modern.js" not in f2
    assert "ballot_lens_public_registry.js" not in f2
    assert "ballot_lens_modern.css" not in f2

def test_j1_route_contract_remains_valid_after_dependency_extinction():
    body = _ballot_lens_body()
    assert '"ballot_lens_f2.html"' in body
    assert '"ballot_lens.html"' not in body
    assert 'return "Ballot Lens F2 assets unavailable", 503' in body

