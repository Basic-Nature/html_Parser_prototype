"""W24REALPRECINCT1B4 reviewed-source profile selector, shadow candidate ONLY.

This module does not approve or add sources to the registry. Only a separately
reviewed and existing server-owned registry source identity is eligible. The
selected source byte digest, URL, exact v1 identity and contest must all match.
No model inference, web reads, database operations or profile writes occur here.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from typing import Any


class ReviewedSourceProfileDenied(ValueError):
    """Fail-closed source identity or schema drift."""


_MARYLAND_SOURCE_URL = (
    'https://elections.maryland.gov/elections/archive/2024/'
    'election_data/PG24_AllPrecincts.csv'
)
_MARYLAND_APPROVED_ROW = (
    '2024', 'President', 'Maryland', 'statewide', 'CSV',
    'Official 2024 General Election statewide precinct results',
    _MARYLAND_SOURCE_URL,
)
_MARYLAND_EXPECTED_ID = (
    'blsrc_v1_fd0d8be9659f0c22e4c03cca55b52742'
    'aa809ae8650729e63bc186235cb64837'
)
_MARYLAND_SOURCE_SHA256 = (
    'fa25069f8504691c03528c70c2a76240b06514ab54b491fd624229867018aacc'
)
_MARYLAND_PROFILE = 'md2024_general_president.json'


def proposed_maryland_registry_row() -> str:
    """Review-only exact row; does NOT mutate the registry."""
    return '\t'.join(_MARYLAND_APPROVED_ROW)


def proposed_maryland_registry_id() -> str:
    raw = '\x1f'.join(('blsrc_v1', 'curated', *_MARYLAND_APPROVED_ROW))
    return 'blsrc_v1_' + hashlib.sha256(raw.encode('utf-8')).hexdigest()


def select_reviewed_profile(runtime: Any, payload: bytes, digest: str) -> dict[str, Any] | None:
    """Match exact *server-owned* public-source and acquired bytes, else strict fallback.

    A URL matching an approved profile never falls back on identity/digest drift.
    An unrelated source returns None, preserving existing strict-CSV behavior.
    """
    selected_url = getattr(runtime, 'approved_target_url', None)
    if selected_url != _MARYLAND_SOURCE_URL:
        return None
    if proposed_maryland_registry_id() != _MARYLAND_EXPECTED_ID:
        raise ReviewedSourceProfileDenied('PROFILE_REGISTRY_ROW_ID_DRIFT')
    identity = getattr(runtime, 'registry_source_id', None)
    projection = getattr(runtime, 'source_projection', None)
    if (identity != _MARYLAND_EXPECTED_ID
        or not isinstance(projection, dict)
        or projection.get('registry_source_id') != identity
        or projection.get('registry_category') != 'curated'
        or projection.get('format') != 'CSV'
        or any(projection.get(k) != v for k, v in (
            ('state', 'Maryland'), ('year', '2024'), ('contest', 'President')))):
        raise ReviewedSourceProfileDenied('PROFILE_SERVER_AUTHORITY_MISMATCH')
    if (not isinstance(payload, bytes)
        or not isinstance(digest, str)
        or not re.fullmatch(r'[0-9a-f]{64}', digest)
        or digest != _MARYLAND_SOURCE_SHA256
        or hashlib.sha256(payload).hexdigest() != digest):
        raise ReviewedSourceProfileDenied('PROFILE_SOURCE_BYTES_OR_DIGEST_DRIFT')
    path = Path(__file__).with_name('reviewed_source_profiles') / _MARYLAND_PROFILE
    raw = path.read_bytes()
    if len(raw) > 8192:
        raise ReviewedSourceProfileDenied('PROFILE_FILE_TOO_LARGE')
    try:
        profile = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReviewedSourceProfileDenied('PROFILE_JSON_INVALID') from exc
    if (not isinstance(profile, dict)
        or profile.get('source_sha256') != digest
        or profile.get('source_url') != selected_url
        or profile.get('scope') != {'state': 'Maryland', 'year': '2024', 'contest': 'President'}):
        raise ReviewedSourceProfileDenied('PROFILE_MANIFEST_DRIFT')
    return profile
