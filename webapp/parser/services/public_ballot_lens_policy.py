"""Fail-closed policy foundation for anonymous Ballot Lens registry runs.

This module is deliberately inert until a later runtime-wiring milestone.
It contains no Flask/Socket.IO registration and performs no parser, network,
filesystem, database, workflow, canonical, or Azure operation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import re
import threading
import time
from typing import Mapping

from webapp.parser.auth.capability_policy import (
    Capability,
    assert_ballot_lens_public_registry_parse,
)

PUBLIC_REGISTRY_PARSE_ENV = "BALLOT_LENS_PUBLIC_REGISTRY_PARSE_ENABLED"
PUBLIC_REGISTRY_PILOT_SOURCE_ENV = "BALLOT_LENS_PUBLIC_REGISTRY_PILOT_SOURCE_ID"
PUBLIC_REGISTRY_RATE_HMAC_SECRET_ENV = "BALLOT_LENS_PUBLIC_RATE_HMAC_SECRET"
PUBLIC_REGISTRY_SOURCE_ID_PATTERN = re.compile(r"\Ablsrc_v1_[0-9a-f]{64}\Z")
PSEUDONYMOUS_CLIENT_KEY_PATTERN = re.compile(r"\Aclient:[0-9a-f]{64}\Z")


class PublicBallotLensPolicyError(RuntimeError):
    pass


class PublicRunAdmissionError(PublicBallotLensPolicyError):
    pass


@dataclass(frozen=True)
class PublicBallotLensRunPolicy:
    contract: str = "ballot_lens_public_run_policy_v1"
    start_payload_max_bytes: int = 4096
    session_rate_max_runs: int = 2
    session_rate_window_seconds: int = 600
    client_rate_max_runs: int = 6
    client_rate_window_seconds: int = 3600
    global_concurrent_runs: int = 1
    hard_wall_clock_seconds: int = 180
    navigation_timeout_ms: int = 60000
    navigation_max_attempts: int = 2
    top_level_redirect_max: int = 3
    browser_network_request_max: int = 250
    remote_download_max_bytes: int = 25 * 1024 * 1024
    pdf_max_pages: int = 50
    ocr_wall_clock_seconds: int = 120
    public_output_max_bytes: int = 16 * 1024 * 1024
    socket_event_max_bytes: int = 64 * 1024
    cumulative_public_log_max_bytes: int = 128 * 1024
    public_result_ttl_seconds: int = 1800
    public_session_ttl_seconds: int = 1800
    public_output_mode: str = "MEMORY_PREVIEW_ONLY"
    public_file_download: bool = False
    caller_supplied_session_id: bool = False
    caller_supplied_url: bool = False
    one_source_per_run: bool = True
    global_history_as_public_authority: bool = False
    shared_processed_urls_as_public_authority: bool = False
    shared_output_cache_as_public_authority: bool = False
    shared_download_manifest_as_public_authority: bool = False
    learning_or_training_write: bool = False
    ml_telemetry_write: bool = False
    canonical_write: bool = False
    workflow_write: bool = False
    production_database_write: bool = False
    diagnostic_screenshot_write: bool = False
    ocr_debug_write: bool = False
    manual_captcha_assist: bool = False
    selenium_fallback: bool = False


DEFAULT_PUBLIC_RUN_POLICY = PublicBallotLensRunPolicy()
PUBLIC_START_ALLOWED_KEYS = frozenset({"registry_source_id"})


def public_registry_parse_feature_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    source = os.environ if environ is None else environ
    value = str(source.get(PUBLIC_REGISTRY_PARSE_ENV, "false") or "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configured_public_registry_pilot_source_id(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    source = os.environ if environ is None else environ
    value = str(
        source.get(PUBLIC_REGISTRY_PILOT_SOURCE_ENV, "") or ""
    ).strip()
    if not PUBLIC_REGISTRY_SOURCE_ID_PATTERN.fullmatch(value):
        return None
    return value


def public_registry_rate_hmac_secret(
    environ: Mapping[str, str] | None = None,
) -> bytes:
    source = os.environ if environ is None else environ
    value = str(
        source.get(PUBLIC_REGISTRY_RATE_HMAC_SECRET_ENV, "") or ""
    )
    encoded = value.encode("utf-8")
    if len(encoded) < 32:
        raise PublicRunAdmissionError(
            "Public admission HMAC secret must contain at least 32 bytes."
        )
    return encoded


def _serialized_payload_size(payload: Mapping[str, object]) -> int:
    try:
        encoded = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except Exception as exc:
        raise PublicBallotLensPolicyError(
            "Public Ballot Lens payload is not safely serializable."
        ) from exc
    return len(encoded)


def validate_public_start_payload(
    payload: Mapping[str, object] | None,
    *,
    policy: PublicBallotLensRunPolicy = DEFAULT_PUBLIC_RUN_POLICY,
) -> str:
    if not isinstance(payload, Mapping):
        raise PublicBallotLensPolicyError(
            "Public Ballot Lens start payload must be a mapping."
        )

    keys = frozenset(str(key) for key in payload.keys())
    if keys != PUBLIC_START_ALLOWED_KEYS:
        raise PublicBallotLensPolicyError(
            "Public Ballot Lens start payload accepts only registry_source_id."
        )

    if _serialized_payload_size(payload) > int(policy.start_payload_max_bytes):
        raise PublicBallotLensPolicyError(
            "Public Ballot Lens start payload exceeds the byte limit."
        )

    registry_source_id = payload.get("registry_source_id")
    if not isinstance(registry_source_id, str):
        raise PublicBallotLensPolicyError(
            "registry_source_id must be a string."
        )
    registry_source_id = registry_source_id.strip()
    if not PUBLIC_REGISTRY_SOURCE_ID_PATTERN.fullmatch(registry_source_id):
        raise PublicBallotLensPolicyError(
            "registry_source_id has an invalid public source ID format."
        )
    return registry_source_id


def authorize_public_registry_parse(
    payload: Mapping[str, object] | None,
    *,
    registry_source_resolved: bool,
    environ: Mapping[str, str] | None = None,
    policy: PublicBallotLensRunPolicy = DEFAULT_PUBLIC_RUN_POLICY,
) -> tuple[Capability, str]:
    registry_source_id = validate_public_start_payload(
        payload,
        policy=policy,
    )
    capability = assert_ballot_lens_public_registry_parse(
        feature_enabled=public_registry_parse_feature_enabled(environ),
        payload_validated=True,
        registry_source_resolved=registry_source_resolved is True,
    )
    pilot_source_id = configured_public_registry_pilot_source_id(
        environ
    )
    if pilot_source_id is None:
        raise PublicBallotLensPolicyError(
            "Public registry pilot source is not configured."
        )
    if registry_source_id != pilot_source_id:
        raise PublicBallotLensPolicyError(
            "Public registry source is outside bounded pilot authority."
        )
    return capability, registry_source_id


def derive_pseudonymous_client_rate_key(
    trusted_client_address: str,
    *,
    secret: bytes,
) -> str:
    address = str(trusted_client_address or "").strip()
    if not address:
        raise PublicRunAdmissionError(
            "Trusted client address is required for public admission."
        )
    if not isinstance(secret, (bytes, bytearray)) or len(secret) < 32:
        raise PublicRunAdmissionError(
            "Public admission HMAC secret must contain at least 32 bytes."
        )
    digest = hmac.new(
        bytes(secret),
        address.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"client:{digest}"


@dataclass
class PublicRunAdmissionLease:
    _controller: "PublicRunAdmissionController"
    client_key: str
    server_session_id: str
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._controller._release_active_slot()
        self._released = True

    def __enter__(self) -> "PublicRunAdmissionLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class PublicRunAdmissionController:
    backend_scope = "process_local"

    def __init__(
        self,
        policy: PublicBallotLensRunPolicy = DEFAULT_PUBLIC_RUN_POLICY,
    ) -> None:
        self._policy = policy
        self._lock = threading.RLock()
        self._active = 0
        self._client_events: dict[str, deque[float]] = {}
        self._session_events: dict[str, deque[float]] = {}

    @staticmethod
    def _prune(
        bucket: deque[float],
        *,
        now: float,
        window_seconds: int,
    ) -> None:
        cutoff = now - max(1, int(window_seconds))
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

    def _validate_keys(
        self,
        client_key: str,
        server_session_id: str,
    ) -> tuple[str, str]:
        client = str(client_key or "").strip()
        session_id = str(server_session_id or "").strip()
        if not PSEUDONYMOUS_CLIENT_KEY_PATTERN.fullmatch(client):
            raise PublicRunAdmissionError(
                "Public admission requires a pseudonymous client key."
            )
        if not session_id.startswith("sess_") or len(session_id) < 16:
            raise PublicRunAdmissionError(
                "Public admission requires a server-generated session ID."
            )
        return client, session_id

    def acquire(
        self,
        *,
        client_key: str,
        server_session_id: str,
        now: float | None = None,
    ) -> PublicRunAdmissionLease:
        client, session_id = self._validate_keys(
            client_key,
            server_session_id,
        )
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._active >= int(self._policy.global_concurrent_runs):
                raise PublicRunAdmissionError(
                    "Global public parser concurrency limit reached."
                )
            client_bucket = self._client_events.setdefault(client, deque())
            session_bucket = self._session_events.setdefault(session_id, deque())
            self._prune(
                client_bucket,
                now=current,
                window_seconds=self._policy.client_rate_window_seconds,
            )
            self._prune(
                session_bucket,
                now=current,
                window_seconds=self._policy.session_rate_window_seconds,
            )
            if len(client_bucket) >= int(self._policy.client_rate_max_runs):
                raise PublicRunAdmissionError(
                    "Public parser client rate limit reached."
                )
            if len(session_bucket) >= int(self._policy.session_rate_max_runs):
                raise PublicRunAdmissionError(
                    "Public parser session rate limit reached."
                )
            client_bucket.append(current)
            session_bucket.append(current)
            self._active += 1
        return PublicRunAdmissionLease(self, client, session_id)

    def _release_active_slot(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1

    def active_count(self) -> int:
        with self._lock:
            return int(self._active)
