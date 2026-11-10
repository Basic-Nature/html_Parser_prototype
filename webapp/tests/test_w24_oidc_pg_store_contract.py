"""Session-store contract tests; no live DB or Redis required."""
from __future__ import annotations
import json
import pytest
from webapp.parser.auth.oidc_postgres_store import PostgresSessionStore
from webapp.parser.auth.oidc_session_store import PREFIX, issue_session, read_session, revoke_session

class Cursor:
    def __init__(self, state):self.state=state;self.row=None;self.rowcount=0
    def execute(self, sql, params=None):
        assert 'electionpulse_oidc_sessions' in sql
        if sql.startswith('INSERT'):
            k,v,ex=params
            self.rowcount=int(k not in self.state);self.state.setdefault(k,v)
        elif sql.startswith('SELECT'):
            self.row=(self.state[params[0]],) if params[0] in self.state else None
        elif sql.startswith('DELETE') and 'expires_at' not in sql:
            self.rowcount=int(params[0] in self.state);self.state.pop(params[0],None)
        else:self.rowcount=0
    def fetchone(self):return self.row
    def close(self):pass

class Connection:
    def __init__(self,state):self.state=state
    def cursor(self):return Cursor(self.state)
    def commit(self):pass
    def rollback(self):pass
    def close(self):pass


def test_pg_backend_supports_original_opaque_session_lifecycle():
    state={}
    store=PostgresSessionStore('fake-dsn',connection_factory=lambda d:Connection(state))
    sid=issue_session(store,principal='oidc:keycloak:'+'b'*64,project_access=True,token_expires_at=1500,now=1200)
    assert len(sid)==64 and all(sid not in record for record in state.keys())
    assert read_session(store,sid,now=1201)['project_access'] is True
    assert revoke_session(store,sid) is True
    assert read_session(store,sid,now=1202) is None


def test_invalid_session_key_or_ttl_fail_closed():
    store=PostgresSessionStore('fake-dsn',connection_factory=lambda d:Connection({}))
    with pytest.raises(ValueError):store.get('not-a-verified-sid')
    with pytest.raises(ValueError):store.set(PREFIX+'a'*64,'{}',ex=3600,nx=True)
