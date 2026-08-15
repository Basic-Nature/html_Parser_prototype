from __future__ import annotations

from selectolax.parser import HTMLParser

from webapp.parser.utils import html_scanner


class _NoPersistenceCoordinator:
    def __init__(self):
        self.organized = []

    def organize_and_enrich(self, value, *args, **kwargs):
        self.organized.append(value)
        return {}


def _segment_cache_for(html: str) -> dict:
    segment_htmls = [
        node.html
        for node in HTMLParser(html).root.traverse()
        if hasattr(node, "html")
    ]
    assert segment_htmls

    return {
        html_scanner.segment_hash(segment_html): {
            "ml_confidence": 0.99,
            "ml_label": "contest",
        }
        for segment_html in segment_htmls
    }


def test_segment_fast_path_without_page_hash_requires_page_rebuild():
    html = "<html><body><div>President</div></body></html>"
    page_hash = "page_hash_intentionally_absent"
    page_url = "https://example.gov/results"
    context_cache = _segment_cache_for(html)
    coordinator = _NoPersistenceCoordinator()

    assert page_hash not in context_cache

    cache_hit = html_scanner._fast_path_cache_hit(
        html,
        page_hash,
        page_url,
        context_cache,
        coordinator,
    )

    assert cache_hit is False
    assert page_hash not in context_cache
    assert coordinator.organized == []


def test_page_level_cache_still_short_circuits():
    html = "<html><body><div>President</div></body></html>"
    page_hash = "page_hash_present"
    page_url = "https://example.gov/results"
    cached_context = {
        "page_hash": page_hash,
        "contests": [],
        "tagged_segments_with_attrs": [],
    }
    context_cache = {
        page_hash: cached_context,
    }
    coordinator = _NoPersistenceCoordinator()

    cache_hit = html_scanner._fast_path_cache_hit(
        html,
        page_hash,
        page_url,
        context_cache,
        coordinator,
    )

    assert cache_hit is True
    assert coordinator.organized == [cached_context]