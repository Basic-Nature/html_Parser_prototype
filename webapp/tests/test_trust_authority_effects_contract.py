# Contracts for Tranche 1F-A live trust-authority extraction.

from __future__ import annotations

from pathlib import Path

from webapp.parser.trust_authority import (
    admin_boost_amount,
    should_apply_admin_boost_policy,
    should_quarantine_for_tier,
    should_reject_for_tier,
)
from webapp.parser.utils.privilege_tiers import PrivilegeTier


ROOT = Path(__file__).resolve().parents[2]
PRIVILEGE_PATH = ROOT / "webapp" / "parser" / "utils" / "privilege_tiers.py"
URL_TRUST_PATH = ROOT / "webapp" / "parser" / "utils" / "url_trust_scorer.py"
AUTHORIZATION_PATH = ROOT / "webapp" / "parser" / "auth" / "authorization.py"
TRUST_AUTHORITY_PATH = ROOT / "webapp" / "parser" / "trust_authority.py"


def test_live_admin_boost_amount_matrix():
    assert admin_boost_amount(PrivilegeTier.STANDARD_USER) == 0
    assert admin_boost_amount(PrivilegeTier.ADMIN_REVIEWER) == 5
    assert admin_boost_amount(PrivilegeTier.ADMIN_FULL_TRUST) == 10
    assert admin_boost_amount(PrivilegeTier.ROOT_ADMIN) == 10
    assert admin_boost_amount(None) == 0


def test_live_admin_boost_policy_requires_admin_and_trusted_domain():
    trusted = {"verified_domain": True}
    untrusted = {"verified_domain": False, "gov_domain": False}

    assert not should_apply_admin_boost_policy(
        trusted,
        PrivilegeTier.STANDARD_USER,
        "example.gov",
        domain_allowlisted=True,
    )

    assert should_apply_admin_boost_policy(
        trusted,
        PrivilegeTier.ADMIN_REVIEWER,
        "example.gov",
        domain_allowlisted=False,
    )

    assert should_apply_admin_boost_policy(
        untrusted,
        PrivilegeTier.ADMIN_FULL_TRUST,
        "elections.example",
        domain_allowlisted=True,
    )

    assert not should_apply_admin_boost_policy(
        trusted,
        PrivilegeTier.ROOT_ADMIN,
        "malicious.zip",
        domain_allowlisted=True,
    )


def test_live_quarantine_policy_matrix():
    low = 30
    medium = 50

    assert should_quarantine_for_tier(
        30,
        PrivilegeTier.STANDARD_USER,
        low_threshold=low,
        medium_threshold=medium,
    )
    assert should_quarantine_for_tier(
        49,
        PrivilegeTier.ADMIN_REVIEWER,
        low_threshold=low,
        medium_threshold=medium,
    )
    assert not should_quarantine_for_tier(
        35,
        PrivilegeTier.ADMIN_FULL_TRUST,
        low_threshold=low,
        medium_threshold=medium,
    )
    assert should_quarantine_for_tier(
        40,
        PrivilegeTier.ADMIN_FULL_TRUST,
        low_threshold=low,
        medium_threshold=medium,
    )
    assert not should_quarantine_for_tier(
        49,
        PrivilegeTier.ROOT_ADMIN,
        low_threshold=low,
        medium_threshold=medium,
    )


def test_live_rejection_policy_matrix():
    low = 30

    assert should_reject_for_tier(
        29,
        PrivilegeTier.STANDARD_USER,
        low_threshold=low,
    )
    assert should_reject_for_tier(
        29,
        PrivilegeTier.ADMIN_REVIEWER,
        low_threshold=low,
    )
    assert not should_reject_for_tier(
        25,
        PrivilegeTier.ADMIN_FULL_TRUST,
        low_threshold=low,
    )
    assert should_reject_for_tier(
        19,
        PrivilegeTier.ADMIN_FULL_TRUST,
        low_threshold=low,
    )
    assert not should_reject_for_tier(
        0,
        PrivilegeTier.ROOT_ADMIN,
        low_threshold=low,
    )


def test_url_trust_public_wrappers_delegate_to_live_policy():
    source = URL_TRUST_PATH.read_text(encoding="utf-8")

    assert "admin_boost_amount(privilege_tier)" in source
    assert "should_quarantine_for_tier(" in source
    assert "should_reject_for_tier(" in source

    assert (
        "10 if privilege_tier in "
        "(PrivilegeTier.ADMIN_FULL_TRUST, PrivilegeTier.ROOT_ADMIN) else 5"
        not in source
    )


def test_legacy_privilege_surfaces_remain_compatibility_only():
    source = PRIVILEGE_PATH.read_text(encoding="utf-8")

    assert "def get_tier_trust_thresholds" in source
    assert "def boost_amount" in source
    assert "def should_apply_admin_boost" in source
    assert "should_apply_admin_boost_policy" in source


def test_authorization_module_remains_free_of_trust_policy():
    source = AUTHORIZATION_PATH.read_text(encoding="utf-8")

    assert "trust_authority" not in source
    assert "should_quarantine" not in source
    assert "should_reject" not in source
    assert "admin_boost" not in source


def test_trust_authority_does_not_absorb_identity_or_route_policy():
    source = TRUST_AUTHORITY_PATH.read_text(encoding="utf-8")

    forbidden = (
        "get_request_principal",
        "get_principal_tier",
        "resolve_session_id",
        "flask",
        "socketio",
        "jsonify",
        "session_manager",
    )

    for token in forbidden:
        assert token not in source
