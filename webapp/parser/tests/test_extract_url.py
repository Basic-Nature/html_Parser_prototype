import tempfile
from pathlib import Path

import importlib

import pytest

from webapp.parser.utils import misc_utils
from webapp.parser.utils.misc_utils import extract_url_and_label


@pytest.mark.parametrize(
    "line,expected",
    [
        ("https://example.com", ("https://example.com", None)),
        ("County page - https://example.org/results", ("https://example.org/results", "County page")),
        ("https://example.org/results (latest)", ("https://example.org/results", "(latest)")),
        ("Label|https://x.y/abc", ("https://x.y/abc", "Label")),
        ("no url here", (None, None)),
        ("# commented line", (None, None)),
        ("https://example.com,", ("https://example.com", None)),
    ],
)
def test_extract_url_and_label_cases(line, expected, monkeypatch):
    monkeypatch.setattr(
        misc_utils,
        "safe_validate_external_url",
        lambda *args, **kwargs: (True, "unit_test"),
    )
    url, label = extract_url_and_label(line)
    assert (url, label) == expected


def test_load_urls_integration(monkeypatch, tmp_path):
    # lazy import to avoid import-time side-effects
    mod = importlib.import_module('webapp.parser.html_election_parser')
    test_lines = [
        "# a comment\n",
        "County A - https://a.example/results\n",
        "https://b.example/data.csv (csv)\n",
        "not a url\n",
    ]
    f = tmp_path / 'urls.txt'
    f.write_text(''.join(test_lines), encoding='utf-8')

    monkeypatch.setattr(mod, 'URL_LIST_FILE', Path(f))
    monkeypatch.setattr(
        misc_utils,
        "safe_validate_external_url",
        lambda *args, **kwargs: (True, "integration_test"),
    )
    out = mod.load_urls()
    assert out == [
        "https://a.example/results",
        "https://b.example/data.csv",
        "not a url",
    ]
