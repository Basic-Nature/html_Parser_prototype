from __future__ import annotations

import json

from webapp.parser.services.public_ballot_lens_runtime import (
    PUBLIC_TERMINAL_REASON_CODES,
    PublicBallotLensRuntime,
)


SOURCE_ID = "blsrc_v1_" + ("c" * 64)
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


def test_reason_allowlist_is_exact_and_public_only():
    assert PUBLIC_TERMINAL_REASON_CODES == frozenset(
        {
            "public_download_fallback_disabled",
            "public_memory_preview_missing",
            "public_challenge_assist_disabled",
        }
    )


def test_known_terminal_reason_is_projected_without_metadata_leak():
    rt = runtime()
    rt.record_processed_status(
        status="error",
        metadata={
            "reason_code": "public_download_fallback_disabled",
            "url": SOURCE_URL,
            "message": "private diagnostic message",
            "exception": "private exception text",
            "nested": {"secret": "do-not-project"},
        },
    )

    result = rt.result_payload()

    assert result["status_counts"] == {"error": 1}
    assert result["terminal_status"] == "error"
    assert (
        result["terminal_reason_code"]
        == "public_download_fallback_disabled"
    )

    rendered = json.dumps(result, sort_keys=True)
    assert SOURCE_URL not in rendered
    assert "private diagnostic message" not in rendered
    assert "private exception text" not in rendered
    assert "do-not-project" not in rendered
    assert '"metadata"' not in rendered


def test_unknown_or_unsafe_reason_is_not_projected():
    for unsafe in (
        "arbitrary_internal_reason",
        "https://secret.example/path",
        "public_download_fallback_disabled\\nsecret",
        "../public_download_fallback_disabled",
        "",
    ):
        rt = runtime()
        rt.record_processed_status(
            status="error",
            metadata={
                "reason_code": unsafe,
                "url": SOURCE_URL,
                "message": "must-not-leak",
            },
        )
        result = rt.result_payload()
        assert result["terminal_status"] == "error"
        assert result["terminal_reason_code"] is None
        rendered = json.dumps(result, sort_keys=True)
        assert SOURCE_URL not in rendered
        assert "must-not-leak" not in rendered


def test_nonterminal_status_does_not_create_terminal_reason():
    rt = runtime()
    rt.record_processed_status(
        status="processing",
        metadata={
            "reason_code": "public_download_fallback_disabled",
        },
    )
    result = rt.result_payload()
    assert result["status_counts"] == {"processing": 1}
    assert result["terminal_status"] is None
    assert result["terminal_reason_code"] is None


def test_terminal_without_metadata_projects_status_but_no_reason():
    rt = runtime()
    rt.record_processed_status(status="success")
    result = rt.result_payload()
    assert result["status_counts"] == {"success": 1}
    assert result["terminal_status"] == "success"
    assert result["terminal_reason_code"] is None


def test_later_terminal_status_replaces_prior_terminal_reason():
    rt = runtime()
    rt.record_processed_status(
        status="error",
        metadata={
            "reason_code": "public_memory_preview_missing",
        },
    )
    rt.record_processed_status(status="success")

    result = rt.result_payload()
    assert result["status_counts"] == {"success": 1}
    assert result["terminal_status"] == "success"
    assert result["terminal_reason_code"] is None
