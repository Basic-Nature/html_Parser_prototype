# Offline contracts for current Google Sheets credential helpers.
#
# The service-account environment vocabulary below is generated from the
# current production helper's own GOOGLE_* string constants. No real
# credentials are used and no Google request is made.

from __future__ import annotations

import json

import pytest

from webapp.parser.data_standardization import google_sheets_client as gsc


BUILDER_ENV_KEYS = ('GOOGLE_SHEETS_SA_TYPE', 'GOOGLE_SHEETS_SA_PROJECT_ID', 'GOOGLE_SHEETS_SA_PRIVATE_KEY_ID', 'GOOGLE_SHEETS_SA_PRIVATE_KEY', 'GOOGLE_SHEETS_SA_CLIENT_EMAIL', 'GOOGLE_SHEETS_SA_CLIENT_ID', 'GOOGLE_SHEETS_SA_AUTH_URI', 'GOOGLE_SHEETS_SA_TOKEN_URI', 'GOOGLE_SHEETS_SA_AUTH_PROVIDER_CERT_URL', 'GOOGLE_SHEETS_SA_CLIENT_CERT_URL', 'GOOGLE_SHEETS_SA_UNIVERSE_DOMAIN')

ALL_GOOGLE_ENV_KEYS = tuple(dict.fromkeys(BUILDER_ENV_KEYS + (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_SHEETS_ELECTION_DB_LITE_CREDENTIALS",
    "GOOGLE_SHEETS_DB_LITE_ID",
    "GOOGLE_SHEETS_WORKLIST_ID",
)))


@pytest.fixture(autouse=True)
def _clear_google_environment(monkeypatch):
    for key in ALL_GOOGLE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _synthetic_env_value(key: str) -> str:
    upper = key.upper()

    if "PRIVATE_KEY_ID" in upper:
        return "offline-private-key-id"
    if "PRIVATE_KEY" in upper:
        return (
            "-----BEGIN PRIVATE KEY-----\\n"
            "offline-test-key\\n"
            "-----END PRIVATE KEY-----\\n"
        )
    if "CLIENT_EMAIL" in upper:
        return "offline-test@electionpulse-test.iam.gserviceaccount.com"
    if "CLIENT_ID" in upper:
        return "1234567890"
    if "PROJECT_ID" in upper:
        return "electionpulse-test-project"
    if "AUTH_PROVIDER_X509_CERT_URL" in upper:
        return "https://www.googleapis.com/oauth2/v1/certs"
    if "CLIENT_X509_CERT_URL" in upper:
        return (
            "https://www.googleapis.com/robot/v1/metadata/x509/"
            "offline-test%40electionpulse-test.iam.gserviceaccount.com"
        )
    if "AUTH_URI" in upper:
        return "https://accounts.google.com/o/oauth2/auth"
    if "TOKEN_URI" in upper:
        return "https://oauth2.googleapis.com/token"
    if "UNIVERSE_DOMAIN" in upper:
        return "googleapis.com"
    if upper.endswith("_TYPE"):
        return "service_account"

    return "offline-test-value"


def test_load_credentials_from_file_round_trips_json(tmp_path):
    expected = {
        "type": "service_account",
        "project_id": "electionpulse-test-project",
        "client_email": "offline-test@example.invalid",
        "private_key": "offline-only",
    }

    path = tmp_path / "service-account.json"
    path.write_text(json.dumps(expected), encoding="utf-8")

    assert gsc._load_credentials_from_file(str(path)) == expected


def test_builder_returns_none_when_service_account_env_is_absent():
    assert gsc._build_service_account_json_from_env() is None


def test_builder_accepts_complete_current_environment(monkeypatch):
    assert BUILDER_ENV_KEYS

    for key in BUILDER_ENV_KEYS:
        monkeypatch.setenv(key, _synthetic_env_value(key))

    built = gsc._build_service_account_json_from_env()

    assert built is not None
    assert isinstance(built, dict)
    assert built
    assert all(value is not None for value in built.values())

    # The helper normalizes escaped private-key newlines when a private-key
    # field is part of the current vocabulary.
    private_key = built.get("private_key")
    if private_key is not None:
        assert "\\n" not in private_key
        assert "\n" in private_key


def test_client_rejects_missing_credentials():
    with pytest.raises(ValueError):
        gsc.GoogleSheetsElectionClient()
