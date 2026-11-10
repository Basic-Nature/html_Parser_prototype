"""P6C5B1 read-only server-side candidate (NOT INSTALLED / NOT ROUTED).

A broker wrapping the P6C5A2 compact pager. Socket ownership, registry
resolution, connection origin, and session metadata are trusted ONLY when
passed by a live server adapter; browser payloads never supply them.

The binder in P6C5B2 must resolve `request.sid` through the existing server
session manager and check anonymous public-registry metadata, allowed Origin,
and run correlation before calling these methods. HMAC cursors are not auth.

This prototype is process-local. It cannot be promoted until single-worker,
Socket.IO runtime and RSS checks are performed. No Redis / DB / file writes.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import json
import threading
import time
from typing import Any

from webapp.parser.services.public_contest_compact_pager import (
    CompactContestPager, CompactDeliveryError, packed,
    PAGE_BUDGET, EVENT_HARD_LIMIT,
)

MAX_REQUEST_BYTES = 2048
MAX_WIRE_PER_SESSION = 8 * 1024 * 1024
MAX_EVENTS_PER_60S = 120
ENVELOPE_MARGIN = 1024

class PublicPageDenied(ValueError):
    pass

@dataclass(frozen=True)
class TrustedSocketIdentity:
    """Construct ONLY by the server's socket-session/Origin adapter."""
    server_session_id: str
    socket_sid: str
    registry_source_id: str
    public_registry_session: bool
    origin_authorized: bool
    principal_present: bool = False

@dataclass
class _Active:
    snapshot_id: str
    registry_source_id: str
    events: deque[float]
    bytes_sent: int = 0
    output_pages: int = 0


def _exact_dict(value: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PublicPageDenied('PAGE_REQUEST_SHAPE_DENIED')
    try:
        if len(packed(value)) > MAX_REQUEST_BYTES:
            raise PublicPageDenied('PAGE_REQUEST_TOO_LARGE')
    except (ValueError, TypeError, OverflowError) as exc:
        raise PublicPageDenied('PAGE_REQUEST_NOT_SERIALIZABLE') from exc
    return value

class PublicContestPageBroker:
    def __init__(self, secret: bytes, **pager_options: Any):
        self._pager = CompactContestPager(secret, **pager_options)
        self._lock = threading.RLock()
        self._active: dict[str, _Active] = {}

    @staticmethod
    def _owner(peer: TrustedSocketIdentity) -> str:
        if (not isinstance(peer, TrustedSocketIdentity)
            or not isinstance(peer.server_session_id, str)
            or not peer.server_session_id.startswith('sess_')
            or len(peer.server_session_id) > 200
            or not isinstance(peer.socket_sid, str)
            or not peer.socket_sid.strip()
            or len(peer.socket_sid) > 200
            or not isinstance(peer.registry_source_id, str)
            or not peer.registry_source_id.startswith('blsrc_v1_')
            or len(peer.registry_source_id) != 73
            or not peer.public_registry_session
            or not peer.origin_authorized
            or peer.principal_present):
            raise PublicPageDenied('PAGE_SERVER_SOCKET_AUTHORITY_DENIED')
        return peer.server_session_id

    def _cleanup(self, now: int) -> None:
        self._pager.purge(now=now)
        for sid, entry in list(self._active.items()):
            if entry.snapshot_id not in self._pager.snapshots:
                del self._active[sid]

    def _entry(self, peer: TrustedSocketIdentity, now: int) -> tuple[str, _Active]:
        sid = self._owner(peer)
        self._cleanup(now)
        ent = self._active.get(sid)
        if ent is None or ent.registry_source_id != peer.registry_source_id:
            raise PublicPageDenied('PAGE_RUN_NOT_OWNED_BY_SOCKET')
        return sid, ent

    @staticmethod
    def _charge(entry: _Active, frame: dict[str, Any], now: int) -> None:
        encoded = packed(frame)
        if len(encoded) > EVENT_HARD_LIMIT - ENVELOPE_MARGIN:
            raise PublicPageDenied('PAGE_EVENT_ENVELOPE_TOO_LARGE')
        cutoff = now - 60
        while entry.events and entry.events[0] <= cutoff:
            entry.events.popleft()
        if len(entry.events) >= MAX_EVENTS_PER_60S:
            raise PublicPageDenied('PAGE_RATE_LIMIT_REACHED')
        if entry.bytes_sent + len(encoded) > MAX_WIRE_PER_SESSION:
            raise PublicPageDenied('PAGE_TOTAL_BYTES_EXCEEDED')
        entry.events.append(now)
        entry.bytes_sent += len(encoded)

    def begin(self, peer: TrustedSocketIdentity, *, scope: dict[str,str],
              source_sha256: str, headers: list[str], rows: list[dict[str,Any]],
              quarantine: dict[str,Any] | None=None, now: int | None=None) -> dict[str, Any]:
        now = int(time.time()) if now is None else int(now)
        with self._lock:
            sid=self._owner(peer)
            self._cleanup(now)
            if sid in self._active:
                raise PublicPageDenied('PAGE_ACTIVE_RUN_MUST_CANCEL_BEFORE_REPLACE')
            if scope.get('registry_source_id') != peer.registry_source_id:
                raise PublicPageDenied('PAGE_SCOPE_SOURCE_AUTHORITY_MISMATCH')
            if quarantine is None:
                quarantine={'observations':0,'votes':0,'reason':None}
            _exact_dict(quarantine, {'observations','votes','reason'})
            n=quarantine['observations']; votes=quarantine['votes']; reason=quarantine['reason']
            if type(n) is not int or n<0 or type(votes) is not int or votes<0:
                raise PublicPageDenied('PAGE_QUARANTINE_COUNTS_INVALID')
            if reason is not None and (not isinstance(reason,str) or len(reason)>100):
                raise PublicPageDenied('PAGE_QUARANTINE_REASON_INVALID')
            if n and not reason:
                raise PublicPageDenied('PAGE_QUARANTINE_REASON_REQUIRED')
            try:
                manifest, first=self._pager.register(
                    server_session_id=sid,scope=scope,source_sha256=source_sha256,
                    headers=headers,rows=rows,now=now)
            except CompactDeliveryError as exc:
                raise PublicPageDenied('PAGE_REGISTRATION_REJECTED') from exc
            event={'kind':'public_compact_manifest','manifest':manifest,
                   'first_cursor':first,'quarantine':quarantine,
                   'download_available':False,'persistent_output':False}
            entry=_Active(manifest['snapshot_id'],peer.registry_source_id,deque())
            try:
                self._charge(entry,event,now)
            except Exception:
                self._pager.cancel(manifest['snapshot_id'],server_session_id=sid)
                raise
            self._active[sid]=entry
            return event

    def next_page(self, peer: TrustedSocketIdentity, payload: dict[str,Any],
                  *,now: int | None=None) -> dict[str,Any]:
        now=int(time.time()) if now is None else int(now)
        with self._lock:
            request=_exact_dict(payload,{'cursor'})
            if not isinstance(request['cursor'],str) or len(request['cursor'])>1200:
                raise PublicPageDenied('PAGE_CURSOR_INVALID')
            sid,entry=self._entry(peer,now)
            try:
                frame=self._pager.fetch(request['cursor'],server_session_id=sid,now=now)
            except CompactDeliveryError as exc:
                raise PublicPageDenied(str(exc)) from exc
            if frame['i']!=entry.snapshot_id:
                raise PublicPageDenied('PAGE_SNAPSHOT_DIFFERENT_RUN')
            event={'kind':'public_compact_page','frame':frame}
            self._charge(entry,event,now)
            entry.output_pages+=1
            return event

    def finish(self, peer: TrustedSocketIdentity, payload: dict[str,Any],
               *,now: int | None=None) -> dict[str,Any]:
        now=int(time.time()) if now is None else int(now)
        with self._lock:
            request=_exact_dict(payload,{'last_cursor'})
            if not isinstance(request['last_cursor'],str) or len(request['last_cursor'])>1200:
                raise PublicPageDenied('PAGE_CURSOR_INVALID')
            sid,entry=self._entry(peer,now)
            try:
                success=self._pager.acknowledge_complete(
                    request['last_cursor'],server_session_id=sid,now=now)
            except CompactDeliveryError as exc:
                raise PublicPageDenied(str(exc)) from exc
            del self._active[sid]
            return {'kind':'public_compact_complete','released':success,
                    'snapshot_id':entry.snapshot_id}

    def cancel(self, peer: TrustedSocketIdentity, payload: dict[str,Any],
               *,now: int | None=None) -> dict[str,Any]:
        now=int(time.time()) if now is None else int(now)
        with self._lock:
            request=_exact_dict(payload,{'snapshot_id'})
            sid,entry=self._entry(peer,now)
            if request['snapshot_id'] != entry.snapshot_id:
                raise PublicPageDenied('PAGE_CANCEL_SNAPSHOT_NOT_OWNED')
            try:
                released=self._pager.cancel(entry.snapshot_id,server_session_id=sid)
            except CompactDeliveryError as exc:
                raise PublicPageDenied(str(exc)) from exc
            del self._active[sid]
            return {'kind':'public_compact_cancelled','released':released}

    def audit_counts(self)->dict[str,int]:
        with self._lock:
            self._cleanup(int(time.time()))
            return {'active_snapshots':len(self._active),
                    'active_serialized_snapshot_limit_bytes':self._pager.snapshot_limit*self._pager.max_live_snapshots}
