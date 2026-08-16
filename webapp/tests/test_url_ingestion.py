from __future__ import annotations

from webapp.parser.utils.url_ingestion import url_already_listed


def test_url_already_listed_true(tmp_path):
    urls_path = tmp_path / "urls.txt"
    urls_path.write_text("https://example.com\n", encoding="utf-8")
    assert url_already_listed(str(urls_path), "https://example.com") is True


def test_url_already_listed_false(tmp_path):
    urls_path = tmp_path / "urls.txt"
    urls_path.write_text("# comment\n", encoding="utf-8")
    assert url_already_listed(str(urls_path), "https://example.com") is False
