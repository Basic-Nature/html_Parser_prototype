"""Current VocabLoader API contract salvaged from the obsolete legacy suite.

The recovered root suite targeted an older service API that included election
mode, audit logging, rate limiting, fuzzy alias resolution, trust scoring, and
snapshot mutation controls. Those obsolete expectations are intentionally not
restored here. This contract preserves only the API exposed by the current
VocabLoader implementation.
"""

from __future__ import annotations

import inspect

from webapp.parser.Context_Integration.vocab.loader import VocabLoader


CURRENT_PUBLIC_METHODS = {
    "clear_cache",
    "get_file_hash",
    "get_load_count",
    "load_canonical",
    "load_mapping",
    "reload",
}


def test_vocab_loader_current_constructor_contract() -> None:
    parameters = inspect.signature(VocabLoader).parameters

    assert "base_dir" in parameters


def test_vocab_loader_current_public_method_contract() -> None:
    for method_name in CURRENT_PUBLIC_METHODS:
        assert callable(getattr(VocabLoader, method_name, None)), method_name