from __future__ import annotations

import ast
from dataclasses import replace
import math
from pathlib import Path

import pytest

from webapp.parser.services.public_ballot_lens_execution import (
    DEFAULT_PUBLIC_EXECUTION_CONTEXT,
    PUBLIC_SOURCE_PROJECTION_KEYS,
    PublicBallotLensExecutionError,
    PublicRunMemoryState,
    assert_public_execution_context,
    build_public_memory_preview,
    serialize_public_memory_preview,
)
from webapp.parser.services.public_ballot_lens_policy import (
    DEFAULT_PUBLIC_RUN_POLICY,
)


SOURCE_ID = "blsrc_v1_" + ("a" * 64)


def source_projection():
    return {
        "registry_source_id": SOURCE_ID,
        "year": "2024",
        "contest": "President",
        "state": "Example",
        "scope": "statewide",
        "format": "HTML",
        "registry_category": "curated",
    }


def test_execution_context_exactly_matches_bl_p2c_no_write_contract():
    context = DEFAULT_PUBLIC_EXECUTION_CONTEXT
    assert context.server_resolved_registry_source_only is True
    assert context.one_source_per_run is True
    assert context.principal is None
    assert context.fabricated_principal is False
    assert context.memory_preview_only is True
    assert context.persistent_output_write is False
    assert context.processed_urls_global_write is False
    assert context.output_cache_write is False
    assert context.download_manifest_write is False
    assert context.pipeline_report_write is False
    assert context.data_framework_audit_export_write is False
    assert context.database_cross_check is False
    assert context.learning_write is False
    assert context.ml_training_telemetry_write is False
    assert context.diagnostic_artifact_write is False
    assert context.manual_captcha_assist is False
    assert context.selenium_fallback is False


@pytest.mark.parametrize(
    "field_name",
    [
        "persistent_output_write",
        "processed_urls_global_write",
        "output_cache_write",
        "download_manifest_write",
        "pipeline_report_write",
        "data_framework_audit_export_write",
        "database_cross_check",
        "learning_write",
        "ml_training_telemetry_write",
        "diagnostic_artifact_write",
        "manual_captcha_assist",
        "selenium_fallback",
        "fabricated_principal",
    ],
)
def test_execution_context_fails_closed_if_forbidden_capability_is_enabled(
    field_name,
):
    drifted = replace(
        DEFAULT_PUBLIC_EXECUTION_CONTEXT,
        **{field_name: True},
    )
    with pytest.raises(PublicBallotLensExecutionError):
        assert_public_execution_context(drifted)


def test_public_source_projection_is_exact_and_url_free():
    projection = source_projection()
    assert frozenset(projection) == PUBLIC_SOURCE_PROJECTION_KEYS

    preview = build_public_memory_preview(
        registry_source_id=SOURCE_ID,
        source_projection=projection,
        headers=["Precinct", "Candidate - Total Votes"],
        rows=[
            {
                "Precinct": "District 1",
                "Candidate - Total Votes": 10,
            }
        ],
    )
    assert "url" not in preview["source"]
    assert "path" not in preview["source"]
    assert preview["source"]["registry_category"] == "curated"

    projection_with_url = dict(projection)
    projection_with_url["url"] = "https://example.gov/results"
    with pytest.raises(PublicBallotLensExecutionError):
        build_public_memory_preview(
            registry_source_id=SOURCE_ID,
            source_projection=projection_with_url,
            headers=["Precinct"],
            rows=[{"Precinct": "District 1"}],
        )


def test_public_preview_preserves_semantic_null_and_numeric_zero():
    headers = [
        "Precinct",
        "Jane Doe (DEM) - Election Day",
        "Jane Doe (DEM) - Provisional",
        "Jane Doe (DEM) - Total Votes",
    ]
    rows = [
        {
            "Precinct": "District 5",
            "Jane Doe (DEM) - Election Day": 200,
            "Jane Doe (DEM) - Provisional": 0,
            "Jane Doe (DEM) - Total Votes": None,
        }
    ]
    preview = build_public_memory_preview(
        registry_source_id=SOURCE_ID,
        source_projection=source_projection(),
        headers=headers,
        rows=rows,
    )
    row = preview["rows"][0]
    assert row["Jane Doe (DEM) - Provisional"] == 0
    assert row["Jane Doe (DEM) - Total Votes"] is None
    assert preview["headers"] == headers


def test_public_preview_requires_headers_to_cover_every_row_field():
    with pytest.raises(PublicBallotLensExecutionError):
        build_public_memory_preview(
            registry_source_id=SOURCE_ID,
            source_projection=source_projection(),
            headers=["Precinct"],
            rows=[
                {
                    "Precinct": "District 1",
                    "Unexpected": 5,
                }
            ],
        )


def test_public_preview_rejects_non_json_and_nonfinite_values():
    with pytest.raises(PublicBallotLensExecutionError):
        build_public_memory_preview(
            registry_source_id=SOURCE_ID,
            source_projection=source_projection(),
            headers=["Precinct", "Value"],
            rows=[
                {
                    "Precinct": "District 1",
                    "Value": object(),
                }
            ],
        )

    with pytest.raises(PublicBallotLensExecutionError):
        build_public_memory_preview(
            registry_source_id=SOURCE_ID,
            source_projection=source_projection(),
            headers=["Precinct", "Value"],
            rows=[
                {
                    "Precinct": "District 1",
                    "Value": math.nan,
                }
            ],
        )


def test_public_preview_has_hard_serialized_result_byte_limit():
    tiny_policy = replace(
        DEFAULT_PUBLIC_RUN_POLICY,
        public_output_max_bytes=128,
    )
    with pytest.raises(PublicBallotLensExecutionError):
        build_public_memory_preview(
            registry_source_id=SOURCE_ID,
            source_projection=source_projection(),
            headers=["Precinct", "Value"],
            rows=[
                {
                    "Precinct": "District 1",
                    "Value": "x" * 300,
                }
            ],
            policy=tiny_policy,
        )


def test_public_run_memory_state_is_one_source_and_progress_is_bounded():
    policy = replace(
        DEFAULT_PUBLIC_RUN_POLICY,
        socket_event_max_bytes=1024,
        cumulative_public_log_max_bytes=2048,
    )
    state = PublicRunMemoryState(
        registry_source_id=SOURCE_ID,
        policy=policy,
    )
    event = state.record_progress(
        processed=1,
        total_entries=2,
        status_counts={"processed": 1},
    )
    assert event["processed"] == 1
    assert "url" not in event
    assert "principal" not in event

    preview = state.build_preview(
        source_projection=source_projection(),
        headers=["Precinct"],
        rows=[{"Precinct": "District 1"}],
    )
    assert preview["registry_source_id"] == SOURCE_ID
    assert preview["progress"][0]["processed"] == 1


def test_public_progress_rejects_inconsistent_counts():
    state = PublicRunMemoryState(
        registry_source_id=SOURCE_ID,
    )
    with pytest.raises(PublicBallotLensExecutionError):
        state.record_progress(
            processed=3,
            total_entries=2,
        )


def test_public_preview_serialization_is_memory_bytes_only():
    preview = build_public_memory_preview(
        registry_source_id=SOURCE_ID,
        source_projection=source_projection(),
        headers=["Precinct"],
        rows=[{"Precinct": "District 1"}],
    )
    encoded = serialize_public_memory_preview(preview)
    assert isinstance(encoded, bytes)
    assert b"District 1" in encoded


def test_public_execution_module_has_no_io_network_database_or_process_apis():
    module_path = Path(
        "webapp/parser/services/public_ballot_lens_execution.py"
    )
    tree = ast.parse(
        module_path.read_text(encoding="utf-8-sig"),
        filename=str(module_path),
    )

    forbidden_import_roots = {
        "os",
        "pathlib",
        "subprocess",
        "shutil",
        "tempfile",
        "urllib",
        "requests",
        "httpx",
        "socket",
        "sqlalchemy",
        "psycopg",
        "asyncpg",
        "sqlite3",
    }
    forbidden_calls = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] not in forbidden_import_roots
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] not in forbidden_import_roots
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_calls
