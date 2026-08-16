"""Offline URL-parser contracts recovered from the legacy print-based smoke script."""

from __future__ import annotations

import json
from urllib.parse import urlsplit

import pytest

from webapp.parser.url_parser import parse_url_components, parse_url_simple


ELECTION_URLS = (
    "https://results.enr.clarityelections.com/GA/105940/web.264614/#/summary",
    "https://electionresults.sos.state.co.us/results/2024/general",
    "https://www.sos.alabama.gov/alabama-votes/voter/election-data",
    "https://elections.virginia.gov/resultsreports/registeredvoter-history/",
    "https://www.ncsbe.gov/results-data?election=11/03/2020&county=Wake",
    "https://vote.wa.gov/elections/results/2024/general-election/county/King/precinct",
    "https://results.voteworks.com/results/2024/primary/jefferson-county",
    "https://www.co.jefferson.wa.us/elections/results/2024/november",
)


@pytest.mark.parametrize("url", ELECTION_URLS)
def test_parse_url_components_offline_contract(url: str) -> None:
    parsed = parse_url_components(url)
    expected = urlsplit(url)

    # Current API returns a UrlComponents object rather than the legacy dict.
    assert parsed.original_url == url
    assert parsed.protocol == expected.scheme
    assert parsed.domain == expected.netloc
    assert parsed.path == expected.path
    assert isinstance(parsed.path_segments, list)
    assert parsed.path_depth == len(parsed.path_segments)
    assert isinstance(parsed.query_params, dict)


def test_parse_url_simple_is_nonempty_json_serializable_mapping() -> None:
    url = (
        "https://results.enr.clarityelections.com/"
        "GA/Jefferson/105940/web.264614/#/summary"
    )

    parsed = parse_url_simple(url)

    assert isinstance(parsed, dict)
    assert parsed

    encoded = json.dumps(parsed, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded == parsed