from __future__ import annotations

from webapp.parser.handlers.shared.parity_hooks import (
    attach_parity_note_to_metadata,
    attach_router_parity_note,
    extract_router_parity_note,
    safe_parity_note,
)


def test_safe_parity_note_filters_invalid():
    assert safe_parity_note(None) is None
    assert safe_parity_note("") is None
    assert safe_parity_note("   ") is None
    assert safe_parity_note("ok") == "ok"
    assert safe_parity_note("a" * 160) == "a" * 160
    assert safe_parity_note("a" * 161) is None
    assert safe_parity_note("bad\nvalue") is None


def test_attach_and_extract_router_parity_note():
    context = {}
    attach_router_parity_note(context, "router_cli_web_parity")
    assert extract_router_parity_note(context) == "router_cli_web_parity"


def test_attach_parity_note_to_metadata():
    metadata = {"source": "unit_test"}
    updated = attach_parity_note_to_metadata(metadata, "router_cli_web_parity")
    assert updated["router_parity_note"] == "router_cli_web_parity"
    assert updated["source"] == "unit_test"
