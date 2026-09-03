from __future__ import annotations

import ast
from pathlib import Path

import pytest

from webapp.parser.services.public_ballot_lens_runtime import (
    PublicBallotLensRuntime,
    activate_public_runtime,
    current_public_runtime,
)


SOURCE_ID = "blsrc_v1_" + ("b" * 64)
SOURCE_URL = "https://results.example.gov/elections/2024"


def resolver(host):
    if host == "results.example.gov":
        return ("8.8.8.8",)
    return ("1.1.1.1",)


def projection():
    return {
        "registry_source_id": SOURCE_ID,
        "year": "2024",
        "contest": "President",
        "state": "Example",
        "scope": "statewide",
        "format": "HTML",
        "registry_category": "curated",
    }


def runtime():
    return PublicBallotLensRuntime(
        registry_source_id=SOURCE_ID,
        source_projection=projection(),
        approved_target_url=SOURCE_URL,
        resolver=resolver,
    )


def test_context_activation_is_run_scoped_and_resets():
    rt = runtime()
    assert current_public_runtime() is None
    with activate_public_runtime(rt):
        assert current_public_runtime() is rt
    assert current_public_runtime() is None


def test_runtime_result_never_projects_raw_url():
    rt = runtime()
    result = rt.result_payload()
    rendered = repr(result)
    assert SOURCE_URL not in rendered
    assert result["source"] == projection()
    assert result["download_available"] is False
    assert result["persistent_output"] is False


def test_runtime_sync_page_guard_installs_before_navigation_authority():
    calls = []

    class FakePage:
        def route(self, pattern, handler):
            calls.append((pattern, handler))

    rt = runtime()
    page = FakePage()
    rt.install_sync_page_guard(page, SOURCE_URL)
    assert calls
    assert calls[0][0] == "**/*"


def test_output_utils_public_finalize_is_true_memory_only(
    monkeypatch,
):
    from webapp.parser.utils import output_utils

    rt = runtime()
    monkeypatch.setattr(
        output_utils,
        "_ensure_dir",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("filesystem directory creation forbidden")
        ),
    )
    monkeypatch.setattr(
        output_utils,
        "_build_database_cross_check",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("database cross-check forbidden")
        ),
    )
    monkeypatch.setattr(
        output_utils,
        "transform_wide_to_smart_standard",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "public memory finalizer must not re-normalize "
                "already-structured election rows"
            )
        ),
    )

    with activate_public_runtime(rt):
        result = output_utils.finalize_election_output(
            headers=[
                "Precinct",
                "Jane Doe - Provisional",
                "Jane Doe - Total Votes",
            ],
            data=[
                {
                    "Precinct": "District 1",
                    "Jane Doe - Provisional": 0,
                    "Jane Doe - Total Votes": None,
                }
            ],
            context={"fill_blanks_with_na": True},
            session_id="sess_server_generated_123456",
        )

    assert result["csv_path"] == ""
    assert result["metadata_path"] == ""
    assert result["public_memory_preview"] == "true"
    assert len(rt.finalized_previews) == 1
    row = rt.finalized_previews[0]["rows"][0]
    assert row["Jane Doe - Provisional"] == 0
    assert row["Jane Doe - Total Votes"] is None


def test_mark_url_processed_uses_memory_not_global_cache(
    monkeypatch,
):
    import webapp.parser.html_election_parser as parser

    rt = runtime()

    class ForbiddenProcessedPath:
        def exists(self):
            raise AssertionError(
                "global processed-url cache must not be read"
            )

    monkeypatch.setattr(
        parser,
        "PROCESSED_URLS_FILE",
        ForbiddenProcessedPath(),
    )

    with activate_public_runtime(rt):
        parser.mark_url_processed(
            SOURCE_URL,
            status="success",
            session_id="sess_server_generated_123456",
        )

    assert rt.summary_counts() == {"success": 1}


def test_public_runtime_suppresses_parser_telemetry_wrappers(
    monkeypatch,
):
    import webapp.parser.html_election_parser as parser

    rt = runtime()
    calls = []
    monkeypatch.setattr(
        parser,
        "_emit_telemetry_event",
        lambda *args, **kwargs: calls.append(
            ("telemetry", args, kwargs)
        ),
    )
    monkeypatch.setattr(
        parser,
        "_increment_counter",
        lambda *args, **kwargs: calls.append(
            ("counter", args, kwargs)
        ),
    )

    with activate_public_runtime(rt):
        parser.emit_telemetry_event("navigation_start", {})
        parser.increment_counter("processed_total", 1)

    assert calls == []


def test_public_main_skips_dirs_processed_cache_database_and_parallel(
    monkeypatch,
):
    import webapp.parser.html_election_parser as parser

    rt = runtime()
    calls = []

    monkeypatch.setattr(
        parser,
        "ensure_input_directory",
        lambda: (_ for _ in ()).throw(
            AssertionError("input directory write forbidden")
        ),
    )
    monkeypatch.setattr(
        parser,
        "ensure_output_directory",
        lambda: (_ for _ in ()).throw(
            AssertionError("output directory write forbidden")
        ),
    )
    monkeypatch.setattr(
        parser,
        "load_processed_urls",
        lambda: (_ for _ in ()).throw(
            AssertionError("global processed cache read forbidden")
        ),
    )
    monkeypatch.setattr(parser, "ENABLE_PARALLEL", True)
    monkeypatch.setattr(
        parser,
        "orchestrate_url",
        lambda *args, **kwargs: calls.append(
            (args, kwargs)
        ),
    )

    with activate_public_runtime(rt):
        parser.main(
            urls=[SOURCE_URL],
            session_id="sess_server_generated_123456",
            output_bypass=True,
            manual_source="input",
            skip_url_prompt=True,
        )

    assert len(calls) == 1


def test_web_pipeline_public_branch_uses_exact_one_server_target(
    monkeypatch,
):
    import webapp.parser.web_pipeline as pipeline

    rt = runtime()
    called = []

    monkeypatch.setattr(
        pipeline,
        "main",
        lambda **kwargs: called.append(kwargs),
    )

    class Flag:
        def is_set(self):
            return False

    with activate_public_runtime(rt):
        result = pipeline.process_urls_for_web(
            prompt_queue=None,
            session_id="sess_server_generated_123456",
            cancel_flag=Flag(),
            emit_func=None,
            output_bypass=True,
            manual_source="input",
            disable_internal_heartbeat=True,
            urls=[SOURCE_URL],
            principal=None,
            principal_source=None,
            dev_isolation_bypass=False,
        )

    assert len(called) == 1
    kwargs = called[0]
    assert kwargs["urls"] == [SOURCE_URL]
    assert kwargs["skip_url_prompt"] is True
    assert kwargs["skip_database_check"] is True
    assert kwargs["force_reparse"] is True
    assert kwargs["output_bypass"] is True
    assert result["contract"] == "ballot_lens_public_runtime_result_v1"


def test_web_pipeline_public_branch_rejects_wrong_or_multiple_urls():
    import webapp.parser.web_pipeline as pipeline

    rt = runtime()

    class Flag:
        def is_set(self):
            return False

    with activate_public_runtime(rt):
        with pytest.raises(RuntimeError):
            pipeline.process_urls_for_web(
                prompt_queue=None,
                session_id="sess_server_generated_123456",
                cancel_flag=Flag(),
                output_bypass=True,
                manual_source="input",
                urls=[
                    SOURCE_URL,
                    "https://results.example.gov/other",
                ],
                principal=None,
                principal_source=None,
                dev_isolation_bypass=False,
            )


def test_browser_source_installs_public_route_before_safe_goto():
    path = Path("webapp/parser/utils/browser_utils.py")
    source = path.read_text(encoding="utf-8-sig")
    helper_at = source.index(
        "_install_public_runtime_sync_guard("
    )
    goto_at = source.index(
        "safe_goto(page, target_url",
        helper_at,
    )
    assert helper_at < goto_at
    assert 'context_kwargs["service_workers"] = "block"' in source
    assert 'context_kwargs["accept_downloads"] = False' in source


def test_output_source_public_branch_precedes_directory_creation():
    path = Path("webapp/parser/utils/output_utils.py")
    source = path.read_text(encoding="utf-8-sig")
    function_at = source.index("def finalize_election_output(")
    public_at = source.index(
        "_finalize_public_memory_output(",
        function_at,
    )
    dir_at = source.index(
        "_ensure_dir(out_dir)",
        function_at,
    )
    assert public_at < dir_at


def test_html_parser_public_guards_are_present_in_core_side_effect_boundaries():
    path = Path("webapp/parser/html_election_parser.py")
    source = path.read_text(encoding="utf-8-sig")

    required = [
        "public_runtime.record_processed_status(",
        "if ENABLE_PARALLEL and public_runtime is None:",
        "skip_database_check = True",
        "force_reparse = True",
        "public_download_fallback_disabled",
        "approved_public_registry",
        "Public runtime cannot enter batch mode.",
        "Public runtime cannot use URL-list fallback.",
        "public_runtime.finalized_previews",
    ]
    for token in required:
        assert token in source


def test_runtime_foundation_contains_no_parser_or_socket_dispatch_import():
    path = Path(
        "webapp/parser/services/public_ballot_lens_runtime.py"
    )
    tree = ast.parse(
        path.read_text(encoding="utf-8-sig"),
        filename=str(path),
    )
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert not any(
        value.endswith("html_election_parser")
        or value.endswith("web_pipeline")
        or value.endswith("socket_ballot_lens_orchestration")
        for value in imports
    )

def test_public_challenge_guard_preserves_legacy_first_call_site_contract():
    path = Path("webapp/parser/html_election_parser.py")
    source = path.read_text(encoding="utf-8-sig")

    condition = (
        'if nav_meta.get("cloudflare_detected") '
        "and ENABLE_SELENIUM_FALLBACK:"
    )
    public_guard = "if public_runtime is not None:"
    public_reason = "public_challenge_assist_disabled"
    detection = (
        'detection_count = _register_cloudflare_detection('
        'session_id, target_url, "playwright")'
    )
    observation = "_observe_legacy_challenge_noncanonical("
    threshold = "if detection_count < 2:"

    condition_index = source.index(condition)
    public_guard_index = source.index(
        public_guard,
        condition_index,
    )
    public_reason_index = source.index(
        public_reason,
        public_guard_index,
    )
    detection_index = source.index(
        detection,
        public_reason_index,
    )
    observation_index = source.index(
        observation,
        detection_index,
    )
    threshold_index = source.index(
        threshold,
        observation_index,
    )

    assert (
        condition_index
        < public_guard_index
        < public_reason_index
        < detection_index
        < observation_index
        < threshold_index
    )

def test_public_runtime_structured_checkpoint_and_action_payloads_are_url_safe():
    events = []
    rt = PublicBallotLensRuntime(registry_source_id=SOURCE_ID, source_projection=projection(), approved_target_url=SOURCE_URL, resolver=resolver, safe_emit=events.append)
    checkpoint = rt.record_checkpoint(checkpoint_id="source.resolve", state="complete", reason_code="approved_public_registry_source_resolved", summary="Approved registry source authority confirmed.", evidence_count=1)
    assert checkpoint["checkpoint_id"] == "source.resolve"
    assert checkpoint["sequence"] == 1
    assert events[-1]["reason_code"] == "public_registry_checkpoint_updated"
    action = rt.record_action_required(prompt_id="public-challenge-assist", checkpoint_id="source.acquire", action_type="challenge", summary="Browser challenge requires interaction unavailable in public mode.")
    assert action["action_type"] == "challenge"
    assert events[-1]["reason_code"] == "public_registry_action_required"
    assert SOURCE_URL not in repr(events)


def test_public_runtime_checkpoint_authority_fails_closed():
    rt = runtime()
    with pytest.raises(Exception):
        rt.record_checkpoint(checkpoint_id="unknown.checkpoint", state="complete")
    with pytest.raises(Exception):
        rt.record_checkpoint(checkpoint_id="source.resolve", state="mystery")
    with pytest.raises(Exception):
        rt.record_action_required(prompt_id="", checkpoint_id="source.acquire", action_type="challenge", summary="Malformed.")
    with pytest.raises(Exception):
        rt.record_action_required(prompt_id="public-challenge-assist", checkpoint_id="source.acquire", action_type="challenge", summary=SOURCE_URL)


def test_public_runtime_result_checkpoint_evidence_does_not_invent_vote_methods():
    events = []
    rt = PublicBallotLensRuntime(registry_source_id=SOURCE_ID, source_projection=projection(), approved_target_url=SOURCE_URL, resolver=resolver, safe_emit=events.append)
    rt.record_result_checkpoints(headers=["Precinct", "Candidate - Total Votes"], contest="President")
    checkpoints = [e["checkpoint"] for e in events if e.get("reason_code") == "public_registry_checkpoint_updated"]
    assert checkpoints[0]["checkpoint_id"] == "contest.select" and checkpoints[0]["state"] == "complete"
    assert checkpoints[1]["checkpoint_id"] == "vote_methods.detect" and checkpoints[1]["state"] == "warning" and checkpoints[1]["evidence_count"] == 0
    events.clear()
    rt.record_result_checkpoints(headers=["Precinct", "Candidate - Election Day", "Candidate - Total Votes"], contest="President")
    method_checkpoint = [e["checkpoint"] for e in events if e.get("checkpoint", {}).get("checkpoint_id") == "vote_methods.detect"][0]
    assert method_checkpoint["state"] == "complete" and method_checkpoint["evidence_count"] == 1


def test_public_memory_capture_authors_normalize_validate_preview_checkpoints():
    events = []
    rt = PublicBallotLensRuntime(registry_source_id=SOURCE_ID, source_projection=projection(), approved_target_url=SOURCE_URL, resolver=resolver, safe_emit=events.append)
    rt.capture_finalized_output(headers=["Precinct", "Candidate - Total Votes"], rows=[{"Precinct": None, "Candidate - Total Votes": 0}])
    ids = [e["checkpoint"]["checkpoint_id"] for e in events if e.get("reason_code") == "public_registry_checkpoint_updated"]
    assert ids == ["normalize.rows", "validate.results", "preview.publish"]
    assert SOURCE_URL not in repr(events)

