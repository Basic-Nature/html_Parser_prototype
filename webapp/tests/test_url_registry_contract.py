from __future__ import annotations

import ast
from pathlib import Path

from webapp.parser.utils.url_registry import (
    find_url_registry_entries,
    is_parser_eligible_url,
    load_url_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "webapp" / "parser" / "urls.txt"
APP = REPO_ROOT / "webapp" / "Smart_Elections_Parser_Webapp.py"
SOCKET = REPO_ROOT / "webapp" / "parser" / "socket_ballot_lens_orchestration.py"


def test_registry_is_strict_schema_and_preserves_semantic_duplicates():
    entries, diagnostics = load_url_registry(REGISTRY)

    assert diagnostics["malformed_row_count"] == 0
    assert diagnostics["row_count"] == 213
    assert diagnostics["quarantine_row_count"] == 4

    rhode_island = find_url_registry_entries(
        REGISTRY,
        "https://elections.ri.gov/elections/previous-election-results",
    )
    assert {entry["contest"] for entry in rhode_island} >= {"President", "Senate"}

    tennessee = find_url_registry_entries(
        REGISTRY,
        "https://sos.tn.gov/elections/results",
    )
    assert len(tennessee) >= 2


def test_parser_eligibility_uses_registry_review_status():
    approved, approved_reason = is_parser_eligible_url(
        REGISTRY,
        "https://apps.azsos.gov/election/2024/ge/canvass/20241105_GeneralCanvass_Signed.pdf",
    )
    assert approved is True
    assert approved_reason == "approved_registry"

    quarantined, quarantine_reason = is_parser_eligible_url(
        REGISTRY,
        "https://www.dropbox.com/scl/fo/ygo4l7p5ff0drxa9l56gr/APiOSv8Qjn37mtGVT_vBiWc?rlkey=ge2hy8v6j3eccfusbw5nb2r4i&e=1&dl=0",
    )
    assert quarantined is False
    assert quarantine_reason == "registry_quarantined"

    unknown, unknown_reason = is_parser_eligible_url(
        REGISTRY,
        "https://example.invalid/not-reviewed",
    )
    assert unknown is False
    assert unknown_reason == "url_not_in_approved_registry"


def test_api_urls_no_longer_appends_registry_file():
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)

    api_urls = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "api_urls"
    )
    body = ast.get_source_segment(source, api_urls) or ""

    assert "registry_write_mode" in body
    assert "review_required" in body
    assert 'open(urls_file, "a"' not in body
    assert "f.write(url" not in body


def test_socket_direct_urls_have_reviewed_registry_gate():
    source = SOCKET.read_text(encoding="utf-8")

    assert "is_parser_eligible_url" in source
    assert "Blocked direct URL by registry gate" in source
    assert "if not dev_isolation_bypass:" in source


def test_api_urls_parse_retains_rate_limit_contract():
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)

    api_urls_parse = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "api_urls_parse"
    )

    decorators = {
        ast.get_source_segment(source, decorator) or ""
        for decorator in api_urls_parse.decorator_list
    }

    assert '_rate_limit("60/minute")' in decorators
