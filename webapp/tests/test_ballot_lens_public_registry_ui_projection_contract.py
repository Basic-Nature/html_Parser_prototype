from __future__ import annotations

import ast
from pathlib import Path


MAIN = Path("webapp/Smart_Elections_Parser_Webapp.py")
BLUEPRINT = Path("webapp/parser/routes/url_library_blueprint.py")
TEMPLATE = Path("webapp/templates/ballot_lens.html")
MODERN_JS = Path("webapp/static/js/ballot_lens_modern.js")
PUBLIC_JS = Path("webapp/static/js/ballot_lens_public_registry.js")


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


def test_public_template_has_approved_source_browser_and_trusted_gate():
    template = _read(TEMPLATE)
    assert 'data-public-registry-api="/api/public/ballot-lens/registry"' in template
    assert 'id="publicRegistryCard"' in template
    assert 'id="publicRegistrySourceSelect"' in template
    assert 'id="btnRunPublicRegistry"' in template
    assert "ballot_lens_public_registry.js" in template
    assert (
        "{% if not ballot_lens_trusted_controls %}hidden "
        'aria-hidden="true"{% endif %}'
    ) in template


def test_public_client_emits_only_opaque_source_id():
    source = _read(PUBLIC_JS)
    assert "socket.emit('ballot_lens', {" in source
    assert "registry_source_id: source.registry_source_id" in source
    assert "executionSourceId" in source
    assert "selected.registry_source_id === executionSourceId" in source
    assert "source.registry_source_id === projectedExecutionSourceId" in source
    assert "source.registry_source_id !== executionSourceId" in source

    # Keep public execution authority exact: the outbound Ballot Lens payload
    # may contain only the opaque approved registry source identifier.
    emit_start = source.index("socket.emit('ballot_lens', {")
    emit_end = source.index("});", emit_start) + len("});")
    emit_payload = source[emit_start:emit_end]
    assert "registry_source_id: source.registry_source_id" in emit_payload
    assert "session_id" not in emit_payload
    assert "direct_urls" not in emit_payload
    assert "file_source" not in emit_payload
    assert "warehouse_override" not in emit_payload

    # Trusted/raw-source authorities remain prohibited everywhere in the
    # anonymous public client. Internal observation of server session ids is
    # permitted because it does not grant outbound session authority.
    for forbidden in (
        "direct_urls",
        "file_source",
        "warehouse_override",
        "source.url",
        "innerHTML",
        "newUrl",
    ):
        assert forbidden not in source


def test_public_client_rejects_projection_with_extra_fields():
    source = _read(PUBLIC_JS)
    assert "keys.length === allowed.length" in source
    for token in (
        "'contest',",
        "'format',",
        "'registry_category',",
        "'registry_source_id',",
        "'scope',",
        "'state',",
        "'year',",
    ):
        assert token in source


def test_legacy_source_initializers_are_trusted_only():
    source = _read(MODERN_JS)
    assert (
        "if (ballotLensTrustedControls) {\n"
        "    ManualUploadManager.init();\n"
        "    UrlListManager.init();\n"
        "  }"
    ) in source


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

def test_public_run_feedback_and_session_ownership_contract():
    template = _read(TEMPLATE)
    public = _read(PUBLIC_JS)
    modern = _read(MODERN_JS)
    for element_id in (
        "publicRegistryRunActivity", "publicRegistryRunState", "publicRegistryRunSession",
        "publicRegistryRunStage", "publicRegistryRunResult", "publicRegistryRunReason",
        "publicRegistryRunCounts", "publicRegistryRunHint",
    ):
        assert f'id="{element_id}"' in template
    assert "socket.on('public_registry_result'" in public
    assert "ballot_lens_public_runtime_result_v1" in public
    assert "terminal_status" in public and "terminal_reason_code" in public
    assert "renderTerminalResult(payload)" in public
    assert "' • Runnable'" in public and "' • Browse-only'" in public
    assert "available to browse • 1 source currently enabled for bounded public parsing" in public
    assert public.index("dispatchRunEvent('ballotlens:public-run-awaiting-session'") < public.index("socket.emit('ballot_lens'")
    assert "socket.on('session_id'" not in public
    assert "data.reason_code === 'public_registry_runtime_started'" in public
    assert "dispatchRunEvent('ballotlens:public-run-session', { session_id:" in public
    assert "ballotlens:public-run-finished" in public
    assert "session_id: runState.activeSessionId" in public
    assert "const publicRunSessionOwner" in modern
    assert "function isForeignOwnedSessionEvent(data)" in modern
    assert "publicRunSessionOwner.awaitingSession && !publicRunSessionOwner.activeSessionId" in modern
    assert "'ballotlens:public-run-session'" in modern
    assert modern.count("/** @type {CustomEvent} */ (event).detail") == 3
    assert "event?.detail" not in modern
    assert "detail.session_id" in modern
    assert "Deferring generic session_id while public run awaits correlated start" in modern
    assert "foreignOwnedHeartbeat" in modern
    for marker in ("socket.on('parser_output'", "socket.on('contest_options'", "socket.on('session_state'"):
        start = modern.index(marker)
        assert "isForeignOwnedSessionEvent(data)" in modern[start:start + 3500]
    assert "return oneventOrig.call(this, packet);" in modern

