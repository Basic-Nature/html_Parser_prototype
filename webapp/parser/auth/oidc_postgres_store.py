"""PostgreSQL-backed session storage for existing Keycloak OIDC session API.

Reuse the existing PostgreSQL Flexible Server with a separately provisioned,
least-privileged session role. Explicit opt-in; never use Redis or a client
cookie for authentication data. Connection errors fail closed.

The SID is a 256-bit random secret from oidc_session_store.issue_session.
Only SHA-256(PREFIX + SID), never SID itself, is persisted in PostgreSQL.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from typing import Any

from .oidc_session_store import MAX_SESSION_SECONDS, PREFIX

KEY_RE = re.compile(r"^electionpulse:oidc:v2:[0-9a-f]{64}$")
TABLE = "public.electionpulse_oidc_sessions"
MAX_RECORD_BYTES = 2048


class PostgresSessionStore:
    """Process-independent, transactional NX/TTL API used by existing OIDC logic."""

    def __init__(self, dsn: str, *, connection_factory: Callable[[str], Any] | None = None):
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("oidc_postgres_dsn_missing")
        # DSN is a secret. Never include it in exceptions, repr, or logs.
        self._dsn = dsn
        self._connection_factory = connection_factory

    @classmethod
    def from_runtime_env(cls) -> "PostgresSessionStore":
        dsn = os.environ.get("ELECTIONPULSE_OIDC_PG_DSN", "")
        if not dsn:
            raise RuntimeError("oidc_postgres_dsn_not_configured")
        try:
            from psycopg2.extensions import parse_dsn
            args = parse_dsn(dsn)
        except Exception as exc:
            raise RuntimeError("oidc_postgres_dsn_invalid") from None
        if (args.get("sslmode") != "verify-full" or not args.get("host") or
            not args.get("dbname") or not args.get("user") or not args.get("password")):
            raise RuntimeError("oidc_postgres_tls_or_identity_invalid")
        return cls(dsn)

    def _connect(self):
        if self._connection_factory is not None:
            return self._connection_factory(self._dsn)
        from psycopg2 import connect
        return connect(self._dsn, connect_timeout=8)

    @staticmethod
    def _key_hash(key: object) -> str:
        if not isinstance(key, str) or KEY_RE.fullmatch(key) is None or not key.startswith(PREFIX):
            raise ValueError("oidc_session_key_invalid")
        return hashlib.sha256(key.encode("ascii")).hexdigest()

    def set(self, key: str, value: str, *, ex: int, nx: bool) -> bool:
        key_hash = self._key_hash(key)
        if (not isinstance(value, str) or not (0 < len(value.encode("utf-8")) <= MAX_RECORD_BYTES)
            or type(ex) is not int or not (1 <= ex <= MAX_SESSION_SECONDS) or nx is not True):
            raise ValueError("oidc_session_write_invalid")
        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(f"DELETE FROM {TABLE} WHERE expires_at <= NOW()")
                cur.execute(
                    f"INSERT INTO {TABLE} (key_sha256, value_json, expires_at) "
                    "VALUES (%s, %s, NOW() + (%s * INTERVAL '1 second')) "
                    "ON CONFLICT (key_sha256) DO NOTHING",
                    (key_hash, value, ex),
                )
                inserted = cur.rowcount == 1
                conn.commit()
                return inserted
            finally:
                cur.close()
        except Exception:
            conn.rollback()
            raise RuntimeError("oidc_session_store_unavailable") from None
        finally:
            conn.close()

    def get(self, key: str) -> str | None:
        key_hash = self._key_hash(key)
        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    f"SELECT value_json FROM {TABLE} "
                    "WHERE key_sha256 = %s AND expires_at > NOW()",
                    (key_hash,),
                )
                row = cur.fetchone()
                return row[0] if row else None
            finally:
                cur.close()
        except Exception:
            raise RuntimeError("oidc_session_store_unavailable") from None
        finally:
            conn.close()

    def delete(self, key: str) -> int:
        key_hash = self._key_hash(key)
        conn = self._connect()
        try:
            cur = conn.cursor()
            try:
                cur.execute(f"DELETE FROM {TABLE} WHERE key_sha256 = %s", (key_hash,))
                count = int(cur.rowcount)
                conn.commit()
                return count
            finally:
                cur.close()
        except Exception:
            conn.rollback()
            raise RuntimeError("oidc_session_store_unavailable") from None
        finally:
            conn.close()
