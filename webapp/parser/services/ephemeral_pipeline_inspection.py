"""Bounded process-local store for noncanonical parser inspection evidence.

C2G 2.0 intentionally does not create a global singleton, HTTP route, parser
call-site integration, persistence, database access, or Azure configuration.

The store may be instantiated only with a topology attestation proving the
current process-local safety boundary: one App Service instance and one
Gunicorn worker. A future integration phase must re-establish that deployment
assumption instead of silently treating this store as distributed authority.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Mapping
import math
import time

from webapp.parser.services.pipeline_inspection import (
    INSPECTION_AUTHORITY,
    INSPECTION_CONTRACT,
)

STORE_AUTHORITY = "noncanonical_process_local_ephemeral_evidence"


@dataclass(frozen=True)
class ProcessLocalTopologyAttestation:
    """Deployment-topology evidence required to instantiate the store."""

    app_service_instance_capacity: int
    gunicorn_workers: int
    evidence_ref: str

    @property
    def process_local_safe(self) -> bool:
        return (
            self.app_service_instance_capacity == 1
            and self.gunicorn_workers == 1
        )


@dataclass(frozen=True)
class InspectionStoreRecord:
    """Internal immutable ownership/expiry envelope."""

    session_id: str
    principal: str
    payload: dict[str, Any]
    stored_monotonic: float
    expires_monotonic: float


def _positive_finite_seconds(value: float, *, name: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return seconds


def _validate_identifier(value: str, *, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{name} is required")
    if len(normalized) > 512:
        raise ValueError(f"{name} is too long")
    return normalized


def _validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("inspection payload must be a mapping")

    copy = deepcopy(dict(payload))

    if copy.get("contract") != INSPECTION_CONTRACT:
        raise ValueError(
            f"inspection payload contract must be {INSPECTION_CONTRACT!r}"
        )

    authority = copy.get("authority")
    if not isinstance(authority, Mapping):
        raise ValueError("inspection payload authority is required")

    if authority.get("inspection") != INSPECTION_AUTHORITY:
        raise ValueError("inspection payload authority mismatch")

    if authority.get("canonical") is not False:
        raise ValueError("canonical payloads are forbidden in ephemeral store")

    if copy.get("rows_included") is not False or "rows" in copy:
        raise ValueError("inspection payload must not contain election rows")

    if copy.get("headers_included") is not False or "headers" in copy:
        raise ValueError("inspection payload must not contain election headers")

    provenance = copy.get("source_provenance")
    if isinstance(provenance, Mapping):
        if provenance.get("source_uri_included") is not False:
            raise ValueError("source URI exposure is forbidden")
        if provenance.get("source_metadata_included") is not False:
            raise ValueError("source metadata exposure is forbidden")
        if "source_uri" in provenance or "metadata" in provenance:
            raise ValueError(
                "source URI/metadata fields are forbidden in store payload"
            )

    return copy


class ProcessLocalInspectionStore:
    """Thread-safe bounded TTL store scoped by session and principal."""

    def __init__(
        self,
        *,
        topology: ProcessLocalTopologyAttestation,
        ttl_seconds: float = 300.0,
        max_entries: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(topology, ProcessLocalTopologyAttestation):
            raise TypeError("topology attestation is required")
        if not topology.process_local_safe:
            raise RuntimeError(
                "process-local inspection store requires exactly one "
                "App Service instance and one Gunicorn worker"
            )
        if not str(topology.evidence_ref or "").strip():
            raise ValueError("topology evidence_ref is required")

        self._topology = topology
        self._ttl_seconds = _positive_finite_seconds(
            ttl_seconds,
            name="ttl_seconds",
        )

        if isinstance(max_entries, bool):
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = int(max_entries)
        if self._max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")

        if not callable(clock):
            raise TypeError("clock must be callable")

        self._clock = clock
        self._lock = RLock()
        self._records: OrderedDict[str, InspectionStoreRecord] = OrderedDict()

    @property
    def topology(self) -> ProcessLocalTopologyAttestation:
        return self._topology

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise RuntimeError("inspection store clock returned non-finite time")
        return now

    def _purge_expired_locked(self, now: float) -> int:
        expired = [
            session_id
            for session_id, record in self._records.items()
            if record.expires_monotonic <= now
        ]
        for session_id in expired:
            self._records.pop(session_id, None)
        return len(expired)

    def _enforce_bound_locked(self) -> None:
        while len(self._records) > self._max_entries:
            self._records.popitem(last=False)

    def put(
        self,
        *,
        session_id: str,
        principal: str,
        payload: Mapping[str, Any],
    ) -> None:
        session_key = _validate_identifier(session_id, name="session_id")
        principal_key = _validate_identifier(principal, name="principal")
        safe_payload = _validate_payload(payload)
        now = self._now()

        record = InspectionStoreRecord(
            session_id=session_key,
            principal=principal_key,
            payload=safe_payload,
            stored_monotonic=now,
            expires_monotonic=now + self._ttl_seconds,
        )

        with self._lock:
            self._purge_expired_locked(now)
            self._records.pop(session_key, None)
            self._records[session_key] = record
            self._enforce_bound_locked()

    def get(
        self,
        *,
        session_id: str,
        principal: str,
    ) -> dict[str, Any] | None:
        session_key = _validate_identifier(session_id, name="session_id")
        principal_key = _validate_identifier(principal, name="principal")
        now = self._now()

        with self._lock:
            self._purge_expired_locked(now)
            record = self._records.get(session_key)
            if record is None:
                return None

            if record.principal != principal_key:
                return None

            self._records.move_to_end(session_key)
            return deepcopy(record.payload)

    def delete(
        self,
        *,
        session_id: str,
        principal: str,
    ) -> bool:
        session_key = _validate_identifier(session_id, name="session_id")
        principal_key = _validate_identifier(principal, name="principal")

        with self._lock:
            record = self._records.get(session_key)
            if record is None or record.principal != principal_key:
                return False
            self._records.pop(session_key, None)
            return True

    def purge_expired(self) -> int:
        now = self._now()
        with self._lock:
            return self._purge_expired_locked(now)

    def size(self) -> int:
        now = self._now()
        with self._lock:
            self._purge_expired_locked(now)
            return len(self._records)