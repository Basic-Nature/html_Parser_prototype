# Authority-first contracts for recovered Google Sheets-backed data routes.
#
# The recovered root diagnostic predates the current authorization boundary
# and expected anonymous requests to reach data availability handling (503).
# Current behavior is intentionally stronger: authorization is evaluated first,
# so anonymous access must be rejected with 403 before Google configuration is
# considered.

from __future__ import annotations

import pytest

from webapp.Smart_Elections_Parser_Webapp import app


PROTECTED_ENDPOINTS = ['/api/election_data/worklist/overview?limit=200', '/api/election_data/db_lite/finalized?limit=200', '/api/election_data/db_lite/down_ballot?limit=200']


@pytest.mark.parametrize("endpoint", PROTECTED_ENDPOINTS)
def test_anonymous_google_backed_data_route_is_rejected_before_data_access(endpoint):
    app.config["TESTING"] = True

    with app.test_client() as client:
        response = client.get(endpoint)

    payload = response.get_json(silent=True)

    assert response.status_code == 403, (
        f"{endpoint} returned {response.status_code} instead of the "
        f"authority-first 403 contract; body={response.get_data(as_text=True)!r}"
    )
    assert isinstance(payload, dict)
    assert str(payload.get("error", "")).lower() == "unauthorized"
