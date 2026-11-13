"""B12 candidate: bounded compact-run CSV handoff; deliberately dormant by default.

Only a server-finalized Smart Elections matrix may be staged. The server's
trusted socket peer must acknowledge the last compact page before a one-use
HTTP bearer offer is minted. The HTTP capability is a bearer credential, not a
cross-channel proof of the browser's Socket.IO identity. No Redis or DB.
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .public_csv_delivery import (
    BROKER, MAX_BYTES, SOURCE_ID, encode_finalized_preview, feature_enabled,
)

_SNAPSHOT = re.compile(r"\A[A-Za-z0-9_-]{12,64}\Z")
_DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")
_MAX_STAGED = 4
_MAX_STAGED_BYTES = 8 * 1024 * 1024
_TTL_SECONDS = 600


@dataclass(frozen=True)
class _Pending:
    deadline: float
    session_id: str
    source_id: str
    source_sha256: str
    content: bytes


class CompactCsvHandoff:
    """In-process only; per-source, per-session, per-snapshot and one use."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._staged: OrderedDict[str, _Pending] = OrderedDict()

    def _purge(self, now: float) -> None:
        for key, entry in list(self._staged.items()):
            if entry.deadline <= now:
                del self._staged[key]

    @staticmethod
    def _identity(snapshot_id: str, session_id: str, source_id: str) -> None:
        if (not isinstance(snapshot_id, str) or not _SNAPSHOT.fullmatch(snapshot_id)
            or not isinstance(session_id, str) or not session_id.startswith('sess_')
            or len(session_id) > 200 or not isinstance(source_id, str)
            or not SOURCE_ID.fullmatch(source_id)):
            raise ValueError('Untrusted compact CSV handoff identity')

    def stage(self, *, snapshot_id: str, session_id: str, source_id: str,
              source_sha256: str, headers: list[str], rows: list[dict[str, Any]]) -> bool:
        if not feature_enabled():
            return False
        self._identity(snapshot_id, session_id, source_id)
        if not isinstance(source_sha256, str) or not _DIGEST.fullmatch(source_sha256):
            raise ValueError('Untrusted compact CSV source digest')
        # No URL or rows from an HTTP request ever enter this function.
        preview: Mapping[str, Any] = {
            'contract': 'ballot_lens_public_memory_preview_v1',
            'registry_source_id': source_id,
            'output_mode': 'MEMORY_PREVIEW_ONLY',
            'persistent_output': False,
            'headers': headers, 'rows': rows, 'row_count': len(rows),
        }
        content = encode_finalized_preview(preview, source_id=source_id)
        if not (0 < len(content) <= MAX_BYTES):
            raise ValueError('Compact CSV export exceeds byte budget')
        with self._lock:
            now = self._clock()
            self._purge(now)
            if snapshot_id in self._staged or len(self._staged) >= _MAX_STAGED:
                raise RuntimeError('Compact CSV snapshot already staged or capacity reached')
            if sum(len(e.content) for e in self._staged.values()) + len(content) > _MAX_STAGED_BYTES:
                raise RuntimeError('Compact CSV staged byte budget reached')
            self._staged[snapshot_id] = _Pending(
                now + _TTL_SECONDS, session_id, source_id, source_sha256, content,
            )
        return True

    def finish(self, *, snapshot_id: str, session_id: str, source_id: str) -> dict[str, object] | None:
        self._identity(snapshot_id, session_id, source_id)
        with self._lock:
            self._purge(self._clock())
            item = self._staged.get(snapshot_id)
            if item is None:
                return None
            if item.session_id != session_id or item.source_id != source_id:
                return None
            del self._staged[snapshot_id]  # One shot, including broker capacity failures.
        if not feature_enabled():
            return None
        try:
            return BROKER.publish(source_id=source_id, content=item.content)
        except (ValueError, RuntimeError):
            return None

    def cancel(self, *, snapshot_id: str, session_id: str, source_id: str) -> bool:
        self._identity(snapshot_id, session_id, source_id)
        with self._lock:
            self._purge(self._clock())
            item = self._staged.get(snapshot_id)
            if item is None or item.session_id != session_id or item.source_id != source_id:
                return False
            del self._staged[snapshot_id]
            return True

    def pending_count(self) -> int:
        with self._lock:
            self._purge(self._clock())
            return len(self._staged)


STAGED_COMPACT_CSV = CompactCsvHandoff()
