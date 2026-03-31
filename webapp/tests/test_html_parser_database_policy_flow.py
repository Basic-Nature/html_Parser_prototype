"""Integration-style tests for main() URL policy orchestration in html_election_parser."""

from __future__ import annotations

from webapp.parser import html_election_parser as hep
from webapp.parser.utils import database_comparison as dc


def _patch_common_runtime(monkeypatch):
    """Patch heavy runtime side effects so main() can run as a focused unit."""
    monkeypatch.setattr(hep, "ensure_input_directory", lambda: None)
    monkeypatch.setattr(hep, "ensure_output_directory", lambda: None)
    monkeypatch.setattr(hep, "load_processed_urls", lambda: {})
    monkeypatch.setattr(hep, "ENABLE_PARALLEL", False)


def test_main_force_reparse_processes_all_selected_urls(monkeypatch):
    _patch_common_runtime(monkeypatch)

    selected_urls = [
        "https://example.com/a",
        "https://example.com/b",
    ]
    orchestrated = []

    def _orchestrate(url, *_args, **kwargs):
        orchestrated.append({"url": url, "force_reparse": bool(kwargs.get("force_reparse"))})

    # If policy helper is called in force-reparse mode, this test should fail.
    monkeypatch.setattr(
        dc,
        "evaluate_url_processing_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("policy helper should not run")),
    )
    monkeypatch.setattr(hep, "orchestrate_url", _orchestrate)

    hep.main(
        urls=selected_urls,
        session_id="test_force_reparse",
        skip_url_prompt=True,
        force_reparse=True,
    )

    assert [entry["url"] for entry in orchestrated] == selected_urls
    assert all(entry["force_reparse"] is True for entry in orchestrated)


def test_main_policy_skips_existing_and_processes_remaining(monkeypatch):
    _patch_common_runtime(monkeypatch)

    selected_urls = [
        "https://example.com/skipped",
        "https://example.com/processed",
    ]
    orchestrated = []
    marked = []

    def _policy(url, **_kwargs):
        if url.endswith("/skipped"):
            return {
                "should_skip": True,
                "decision": "skipped_data_exists",
                "data_source": "verified_datasets",
                "metadata": {"state": "NY", "contest": "Mayor"},
            }
        return {
            "should_skip": False,
            "decision": "process",
            "data_source": None,
            "metadata": None,
        }

    monkeypatch.setattr(dc, "evaluate_url_processing_policy", _policy)
    monkeypatch.setattr(hep, "orchestrate_url", lambda url, *_args, **_kwargs: orchestrated.append(url))
    monkeypatch.setattr(hep, "mark_url_processed", lambda url, **kwargs: marked.append({"url": url, **kwargs}))
    monkeypatch.setattr(hep, "infer_state_county_from_url", lambda _url: ("NY", "Westchester"))

    hep.main(
        urls=selected_urls,
        session_id="test_policy_skip",
        skip_url_prompt=True,
    )

    assert orchestrated == ["https://example.com/processed"]
    assert len(marked) == 1
    assert marked[0]["url"] == "https://example.com/skipped"
    assert marked[0]["status"] == "skipped_data_exists"
    assert marked[0]["data_source"] == "verified_datasets"
    assert marked[0]["retrieved_from_database"] is True
    assert marked[0]["state"] == "NY"
    assert marked[0]["contest"] == "Mayor"
