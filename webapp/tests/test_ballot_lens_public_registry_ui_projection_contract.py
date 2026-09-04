from __future__ import annotations

import ast
from pathlib import Path


MAIN = Path("webapp/Smart_Elections_Parser_Webapp.py")
BLUEPRINT = Path("webapp/parser/routes/url_library_blueprint.py")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def test_public_registry_endpoint_is_get_only_and_url_free():
    source = _read(BLUEPRINT)
    assert '"/api/public/ballot-lens/registry"' in source
    assert 'methods=["GET"]' in source
    assert "api_public_ballot_lens_registry" in source

    main = _read(MAIN)
    start = main.index("def api_public_ballot_lens_registry():")
    end = main.index('@_rate_limit("30/minute")\ndef api_urls():', start)
    body = main[start:end]
    assert "project_public_registry_sources(URL_LIST_FILE)" in body
    assert '"ballot_lens_public_registry_v1"' in body
    assert '"execution_enabled"' in body
    assert '"execution_source_id"' in body
    assert "configured_public_registry_pilot_source_id" in body
    assert '"sources"' in body
    assert '"url"' not in body
    assert "load_url_registry" not in body


def test_legacy_raw_url_library_requires_trusted_principal():
    main = _read(MAIN)
    start = main.index("def api_urls():")
    end = main.index('@_rate_limit("60/minute")\ndef api_urls_parse():', start)
    body = main[start:end]
    assert body.index("get_request_principal()") < body.index(
        "trusted_principal_required"
    ) < body.index("load_url_registry")


def test_anonymous_page_suppresses_server_file_enumeration():
    main = _read(MAIN)
    start = main.index("def ballot_lens():")
    end = main.index("def worklist():", start)
    body = main[start:end]
    assert "ballot_lens_trusted_controls = bool(principal)" in body
    assert '"input_files": []' in body
    assert '"output_files": []' in body
    assert '"uploaded_files": []' in body
    assert body.index("ballot_lens_trusted_controls") < body.index(
        "get_all_file_lists()"
    )


def test_public_projection_handler_has_no_request_supplied_url_authority():
    source = _read(MAIN)
    tree = ast.parse(source, filename=str(MAIN))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "api_public_ballot_lens_registry"
    )
    segment = ast.get_source_segment(source, fn) or ""
    assert "request.get_json" not in segment
    assert 'request.args.get("url"' not in segment
    assert "resolve_public_registry_source" not in segment

def test_public_runtime_legacy_log_sink_is_suppressed_before_store_log():
    main = _read(MAIN)
    start = main.index("def socketio_emit_func(line):")
    end = main.index("def get_prompt_queue(", start)
    body = main[start:end]
    assert "current_public_runtime" in body
    assert body.index("current_public_runtime") < body.index("store_log")

