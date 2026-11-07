"""W24AUTHOIDC1E — real offline RSA signatures; no live Keycloak claims.

Tests verify the *independent access-token verifier*, not an Authlib browser
code exchange or a real provider's JWKS. No RSA fixture keys are shipped.
"""
from __future__ import annotations

import time
import unittest

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from webapp.parser.auth.electionpulse_client_roles import (
    CLIENT_ID, ISSUER, ROLE_MAP, roles_from_verified_access_token_claims,
)
from webapp.parser.auth.keycloak_tokens import verify_keycloak_access_token
from webapp.parser.auth.oidc_session_store import (
    PREFIX, issue_session, read_session, revoke_session,
)


class FakeStore:
    def __init__(self):
        self.data = {}

    def set(self, key, value, *, ex, nx):
        if nx and key in self.data:
            return False
        self.data[key] = value
        return True

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        return int(self.data.pop(key, None) is not None)


class CryptographicRoleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.public_key = cls.private_key.public_key()

    def claims(self, **overrides):
        now = int(time.time())
        d = {"iss": ISSUER, "sub": "opaque-test-subject", "aud": [CLIENT_ID],
             "azp": CLIENT_ID, "iat": now, "nbf": now, "exp": now + 300,
             "resource_access": {CLIENT_ID: {"roles": ["electionpulse-workflow-contributor"]}}}
        d.update(overrides)
        return d

    def signed(self, claims, *, algorithm="RS256"):
        key = self.private_key if algorithm == "RS256" else "synthetic-test-only-hmac-key-over-32-bytes"
        return jwt.encode(claims, key, algorithm=algorithm, headers={"kid": "offline-rsa"})

    def valid(self, claims=None):
        return verify_keycloak_access_token(self.signed(claims or self.claims()),
                                            _test_signing_key=self.public_key)

    def test_valid_rsa_client_contributor(self):
        verified = self.valid()
        dec = roles_from_verified_access_token_claims(verified)
        self.assertTrue(dec.authenticated and dec.project_access)
        self.assertFalse(dec.certificate_trusted or dec.reviewer_operations_authorized)

    def test_every_noncontributor_role_is_inert(self):
        for role in list(ROLE_MAP)[1:]:
            with self.subTest(role=role):
                c = self.claims(resource_access={CLIENT_ID:{"roles":[role]}})
                self.assertFalse(roles_from_verified_access_token_claims(self.valid(c)).project_access)

    def test_unrelated_doors_roles_do_not_grant_project_access(self):
        c = self.claims(resource_access={"doors":{"roles":["electionpulse-workflow-contributor"]}})
        self.assertFalse(roles_from_verified_access_token_claims(self.valid(c)).project_access)

    def test_invalid_signature_rejected(self):
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        forged = jwt.encode(self.claims(), other, algorithm="RS256")
        with self.assertRaises(ValueError):
            verify_keycloak_access_token(forged, _test_signing_key=self.public_key)

    def test_modified_payload_rejected(self):
        valid = self.signed(self.claims())
        first, middle, last = valid.split(".")
        corrupted = first + "." + middle[:-1] + ("A" if middle[-1] != "A" else "B") + "." + last
        with self.assertRaises(ValueError):
            verify_keycloak_access_token(corrupted, _test_signing_key=self.public_key)

    def test_hs256_algorithm_confusion_rejected(self):
        with self.assertRaises(ValueError):
            verify_keycloak_access_token(self.signed(self.claims(), algorithm="HS256"),
                                         _test_signing_key=self.public_key)

    def test_wrong_issuer_rejected(self):
        with self.assertRaises(ValueError):
            self.valid(self.claims(iss="https://evil.example/realms/smartelections"))

    def test_wrong_audience_rejected(self):
        with self.assertRaises(ValueError):
            self.valid(self.claims(aud=["doors"]))

    def test_wrong_authorized_party_rejected(self):
        with self.assertRaises(ValueError):
            self.valid(self.claims(azp="doors"))

    def test_expired_token_rejected(self):
        with self.assertRaises(ValueError):
            self.valid(self.claims(exp=int(time.time()) - 90))

    def test_not_before_future_rejected(self):
        with self.assertRaises(ValueError):
            self.valid(self.claims(nbf=int(time.time()) + 90))

    def test_future_issued_at_rejected(self):
        with self.assertRaises(ValueError):
            self.valid(self.claims(iat=int(time.time()) + 90))

    def test_missing_required_claim_rejected(self):
        for claim in ["iss", "sub", "aud", "exp", "iat"]:
            with self.subTest(claim=claim):
                c = self.claims()
                del c[claim]
                with self.assertRaises(ValueError):
                    self.valid(c)

    def test_malformed_subject_rejected(self):
        with self.assertRaises(ValueError):
            self.valid(self.claims(sub="\n"))

    def test_absent_roles_means_authenticated_only(self):
        dec = roles_from_verified_access_token_claims(
            self.valid(self.claims(resource_access={})))
        self.assertTrue(dec.authenticated)
        self.assertFalse(dec.project_access)

    def test_revocable_shared_session_with_short_ttl(self):
        store = FakeStore()
        now = 1800000000
        sid = issue_session(store, principal="oidc:keycloak:" + "a" * 64,
                            project_access=True, token_expires_at=now+600, now=now)
        self.assertEqual(len(sid), 64)
        self.assertTrue(read_session(store, sid, now=now)["project_access"])
        self.assertIsNone(read_session(store, sid, now=now+301))
        self.assertTrue(revoke_session(store, sid))
        self.assertIsNone(read_session(store, sid, now=now+1))

    def test_invalid_sid_and_malformed_stored_data_fail_closed(self):
        store = FakeStore()
        self.assertIsNone(read_session(store, "bad!"))
        self.assertFalse(revoke_session(store, "bad!"))
        store.data[PREFIX + "a"*64] = '{"project_access":true}'
        self.assertIsNone(read_session(store, "a"*64))

    def test_invalid_issued_session_denied(self):
        store = FakeStore()
        with self.assertRaises(ValueError):
            issue_session(store,principal="untrusted",project_access=True,
                          token_expires_at=1800000100,now=1800000000)
        with self.assertRaises(ValueError):
            issue_session(store,principal="oidc:keycloak:"+"a"*64,project_access=True,
                          token_expires_at=1800000000,now=1800000000)



class RedisFreeDevStoreTests(unittest.TestCase):
    """Offline tests for the bounded, in-process-only development backend."""

    def test_no_external_backend_required(self):
        from webapp.parser.auth.oidc_session_store import MemorySessionStore
        store = MemorySessionStore()
        self.assertIsNone(store.get(PREFIX + "x" * 64))

    def test_ttl_is_enforced_by_monotonic_clock(self):
        from webapp.parser.auth.oidc_session_store import MemorySessionStore
        ticks = [10.0]
        store = MemorySessionStore(clock=lambda: ticks[0])
        now = int(time.time())
        sid = issue_session(store, principal="oidc:keycloak:" + "a"*64,
                            project_access=True, token_expires_at=now+300, now=now)
        self.assertIsNotNone(read_session(store, sid, now=now))
        ticks[0] += 301
        self.assertIsNone(read_session(store, sid, now=now+1))

    def test_revocation_blocks_reuse(self):
        from webapp.parser.auth.oidc_session_store import MemorySessionStore
        store = MemorySessionStore()
        now = int(time.time())
        sid = issue_session(store, principal="oidc:keycloak:" + "b"*64,
                            project_access=True, token_expires_at=now+90, now=now)
        self.assertTrue(revoke_session(store, sid))
        self.assertFalse(revoke_session(store, sid))
        self.assertIsNone(read_session(store, sid, now=now))

    def test_capacity_is_fail_closed_not_eviction(self):
        from webapp.parser.auth.oidc_session_store import MemorySessionStore
        store = MemorySessionStore(max_sessions=1)
        now = int(time.time())
        sid = issue_session(store, principal="oidc:keycloak:" + "a"*64,
                            project_access=True, token_expires_at=now+50, now=now)
        with self.assertRaises(RuntimeError):
            issue_session(store, principal="oidc:keycloak:" + "b"*64,
                          project_access=False, token_expires_at=now+50, now=now)
        self.assertIsNotNone(read_session(store, sid, now=now))

    def test_independent_process_stores_do_not_share_sessions(self):
        from webapp.parser.auth.oidc_session_store import MemorySessionStore
        first, second = MemorySessionStore(), MemorySessionStore()
        now = int(time.time())
        sid = issue_session(first, principal="oidc:keycloak:" + "c"*64,
                            project_access=True, token_expires_at=now+50, now=now)
        self.assertIsNone(read_session(second, sid, now=now))

    def test_duplicate_and_invalid_write_rejected(self):
        from webapp.parser.auth.oidc_session_store import MemorySessionStore
        store = MemorySessionStore()
        key = PREFIX + "a"*64
        self.assertTrue(store.set(key, "test", ex=1, nx=True))
        self.assertFalse(store.set(key, "different", ex=1, nx=True))
        for bad in [0, 301, True]:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                store.set(key, "test", ex=bad, nx=True)

    def test_expired_space_can_be_reused(self):
        from webapp.parser.auth.oidc_session_store import MemorySessionStore
        clock = [10.0]
        store = MemorySessionStore(max_sessions=1, clock=lambda: clock[0])
        self.assertTrue(store.set(PREFIX + "a"*64, "one", ex=1, nx=True))
        clock[0] = 11.1
        self.assertTrue(store.set(PREFIX + "b"*64, "two", ex=1, nx=True))
        self.assertIsNone(store.get(PREFIX + "a"*64))

if __name__ == '__main__':
    unittest.main(verbosity=2)
