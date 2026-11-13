"""Bounded, memory-only, one-time HTTP CSV capabilities for approved public runs.

Explicitly off unless BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED=1. Not durable
across workers/restarts, intentionally Redis-free, and never accepts client rows.
"""
from __future__ import annotations

import csv
import io
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Callable

SOURCE_ID = re.compile(r"\Ablsrc_v1_[0-9a-f]{64}\Z")
TOKEN = re.compile(r"\A[A-Za-z0-9_-]{43}\Z")
MAX_ROWS = 10000
MAX_HEADERS = 1500
MAX_BYTES = 2 * 1024 * 1024
TTL_SECONDS = 120
MAX_PENDING = 64
CSV_EVENT = 'public_registry_csv_ready'
CSV_ENDPOINT = '/api/public/csv-download'


def feature_enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return values.get('BALLOT_LENS_PUBLIC_CSV_DOWNLOAD_ENABLED', '').strip().lower() in {'1','true','yes','on'}


def _safe_cell(value: object) -> str:
    if value is None:
        return ''  # Missing is never converted to the numeric value zero.
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('CSV contains an unsupported cell value')
    if isinstance(value, float) and (value != value or value in (float('inf'), float('-inf'))):
        raise ValueError('CSV contains an invalid number')
    text = str(value)
    if len(text) > 2048 or '\x00' in text:
        raise ValueError('CSV cell exceeds the accepted bound')
    if isinstance(value, str) and text.lstrip(' \r\n\t').startswith(('=', '+', '-', '@')):
        return "'" + text   # Spreadsheet formula injection is not allowed.
    return text


def encode_finalized_preview(preview: Mapping[str, Any], *, source_id: str) -> bytes:
    """Never infer omitted columns, zero votes or unprovided source revisions."""
    if not SOURCE_ID.fullmatch(source_id):
        raise ValueError('Invalid source identity')
    if not isinstance(preview, Mapping) or preview.get('registry_source_id') != source_id:
        raise ValueError('Mismatched public preview source')
    if preview.get('contract') != 'ballot_lens_public_memory_preview_v1':
        raise ValueError('Untrusted preview contract')
    if preview.get('output_mode') != 'MEMORY_PREVIEW_ONLY' or preview.get('persistent_output') is not False:
        raise ValueError('Output must be a memory-only finalized preview')
    headers = preview.get('headers')
    rows = preview.get('rows')
    if not isinstance(headers, list) or not (1 <= len(headers) <= MAX_HEADERS):
        raise ValueError('Invalid CSV column count')
    if any(not isinstance(h, str) or not h or len(h) > 200 or '\x00' in h for h in headers):
        raise ValueError('Invalid CSV headers')
    if len(set(headers)) != len(headers) or headers[0] != 'Precinct' or 'Grand Total' not in headers:
        raise ValueError('CSV requires unique headers, leading Precinct and trailing Grand Total')
    if not isinstance(rows, list) or not (1 <= len(rows) <= MAX_ROWS):
        raise ValueError('No bounded finalized precinct rows')
    if preview.get('row_count') != len(rows):
        raise ValueError('Preview row count mismatch')
    out = io.StringIO(newline='')
    writer = csv.writer(out, lineterminator='\r\n')
    writer.writerow(headers)
    approved = set(headers)
    for row in rows:
        if not isinstance(row, Mapping) or set(row) - approved:
            raise ValueError('Row contains out-of-contract columns')
        if not isinstance(row.get('Precinct'), str) or not row['Precinct'].strip():
            raise ValueError('Missing precinct identity')
        writer.writerow([_safe_cell(row.get(name)) for name in headers])
        if out.tell() > MAX_BYTES:
            raise ValueError('CSV exceeds allowed byte size')
    result = out.getvalue().encode('utf-8-sig')
    if len(result) > MAX_BYTES:
        raise ValueError('CSV exceeds allowed byte size')
    return result


class OneUseCsvBroker:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, bytes, str]] = OrderedDict()
        self._lock = threading.RLock()

    def _expire(self, now: float) -> None:
        for key, (deadline, _, _) in list(self._entries.items()):
            if now >= deadline:
                del self._entries[key]

    def publish(self, *, source_id: str, content: bytes) -> dict[str, object]:
        if not SOURCE_ID.fullmatch(source_id) or not isinstance(content, bytes) or not (0 < len(content) <= MAX_BYTES):
            raise ValueError('Invalid CSV publish request')
        with self._lock:
            now = self._clock()
            self._expire(now)
            if len(self._entries) >= MAX_PENDING:
                raise RuntimeError('Public CSV capacity exhausted')  # Never evict a live token.
            token = secrets.token_urlsafe(32)
            self._entries[token] = (now + TTL_SECONDS, content, source_id)
            return {
                'contract': 'ballot_lens_public_csv_offer_v1',
                'registry_source_id': source_id,
                'token': token,
                'endpoint': CSV_ENDPOINT,
                'expires_in_seconds': TTL_SECONDS,
            }

    def take(self, token: str) -> bytes | None:
        if not isinstance(token, str) or not TOKEN.fullmatch(token):
            return None
        with self._lock:
            self._expire(self._clock())
            item = self._entries.pop(token, None)
            return item[1] if item is not None else None


BROKER = OneUseCsvBroker()


def publish_from_public_result(result: Mapping[str, Any]) -> dict[str, object] | None:
    """Accept only a server-created, completed result. Never accept browser rows."""
    if not feature_enabled():
        return None
    if not isinstance(result, Mapping) or result.get('contract') != 'ballot_lens_public_runtime_result_v1':
        return None
    source_id = result.get('registry_source_id')
    if not isinstance(source_id, str) or not SOURCE_ID.fullmatch(source_id):
        return None
    if result.get('persistent_output') is not False or result.get('download_available') is not False:
        return None
    if result.get('terminal_status') not in ('completed', 'success'):
        return None
    outputs = result.get('outputs')
    if not isinstance(outputs, list) or len(outputs) != 1:
        return None  # No ambiguous multiple-output selection.
    try:
        content = encode_finalized_preview(outputs[0], source_id=source_id)
        return BROKER.publish(source_id=source_id, content=content)
    except (TypeError, ValueError, RuntimeError):
        return None


# Request rate guard is process-local, pseudonymous, and strictly bounded.
# This does not alter the public-parser quota or its attribution machinery.
_REQUEST_LOCK = threading.RLock()
_REQUEST_HMAC_SECRET = secrets.token_bytes(32)
_REQUEST_TIMES: OrderedDict[str, list[float]] = OrderedDict()
_REQUEST_WINDOW_SECONDS = 60
_REQUEST_MAX_PER_WINDOW = 20
_REQUEST_MAX_KEYS = 2048


def allow_public_csv_request(peer_address: str, *, now: float | None = None) -> bool:
    """Bound anonymous HTTP ticket attempts without storing raw IP addresses."""
    import hashlib
    import hmac

    if not isinstance(peer_address, str) or not (1 <= len(peer_address) <= 128):
        return False
    moment = time.monotonic() if now is None else float(now)
    key = hmac.new(_REQUEST_HMAC_SECRET, peer_address.encode('utf-8'), hashlib.sha256).hexdigest()
    with _REQUEST_LOCK:
        for old_key, times in list(_REQUEST_TIMES.items()):
            valid = [t for t in times if 0 <= moment - t < _REQUEST_WINDOW_SECONDS]
            if valid:
                _REQUEST_TIMES[old_key] = valid
            else:
                _REQUEST_TIMES.pop(old_key, None)
        times = _REQUEST_TIMES.get(key)
        if times is None:
            if len(_REQUEST_TIMES) >= _REQUEST_MAX_KEYS:
                return False
            times = []
            _REQUEST_TIMES[key] = times
        if len(times) >= _REQUEST_MAX_PER_WINDOW:
            return False
        times.append(moment)
        return True
